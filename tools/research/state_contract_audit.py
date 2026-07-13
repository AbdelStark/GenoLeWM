# SPDX-License-Identifier: Apache-2.0
"""Audit raw and normalized Carbon state semantics on a pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from geno_lewm.encoder import CarbonStateEncoder
from geno_lewm.encoder._identity import encoder_runtime_hash, encoder_weights_hash
from geno_lewm.encoder._normalization import l2_normalize_state
from geno_lewm.encoder.windowing import CARBON_TOKEN_BP, SUPPORTED_WINDOW_BP
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.state_contract_audit"
DEFAULT_CARBON_REVISION: Final = "5d31d59b3c845b288a13aedb1358934196852eec"
DEFAULT_CARBON_WEIGHTS_HASH: Final = (
    "sha256:e257506988203fdb8bb46976ee81c97e24f29073754bbff70137c7704dbadaa8"
)
DEFAULT_CARBON_RUNTIME_HASH: Final = (
    "sha256:add3c1a663a35fb92fbd3fd935b067da1aed8aeb143ea01f7d92c2cd3ed2aa5e"
)


def build_state_contract_report(
    *,
    raw_states: tuple[tuple[float, ...], ...],
    normalized_states: tuple[tuple[float, ...], ...],
    sequence_hashes: tuple[str, ...],
    commit_sha: str,
    carbon_model_dir: Path,
    carbon_revision: str,
    carbon_weights_hash: str,
    expected_carbon_weights_hash: str,
    carbon_runtime_hash: str,
    expected_carbon_runtime_hash: str,
    encoder_parameter_count: int,
    encoder_trainable_parameter_count: int,
    expected_d_state: int,
    resolved_pool_type: str,
    resolved_pool_radius: int,
    resolved_center_token: int | None,
    expected_center_token: int,
    execution_device: str,
    window_bp: int,
    state_layer: int,
    pool_radius: int,
    dtype: str,
    unit_tolerance: float = 1.0e-5,
    parity_tolerance: float = 1.0e-6,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build a machine-readable normalization contract report."""
    if not raw_states or len(raw_states) != len(normalized_states):
        raise InputError("raw and normalized state batches must be non-empty and aligned")
    if len(sequence_hashes) != len(raw_states):
        raise InputError("sequence hashes must align with audited states")
    if not commit_sha:
        raise InputError("commit_sha must be non-empty")
    if unit_tolerance <= 0.0 or parity_tolerance <= 0.0:
        raise InputError("audit tolerances must be positive")
    if expected_d_state <= 0:
        raise InputError("expected_d_state must be positive")

    rows: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    weights_identity_verified = carbon_weights_hash == expected_carbon_weights_hash
    runtime_identity_verified = carbon_runtime_hash == expected_carbon_runtime_hash
    parameters_frozen = encoder_parameter_count > 0 and encoder_trainable_parameter_count == 0
    pooling_identity_verified = (
        resolved_pool_type == "centered_mean"
        and resolved_pool_radius == pool_radius
        and resolved_center_token == expected_center_token
    )
    if not weights_identity_verified:
        blockers.append(
            {
                "code": "encoder_weights_identity_mismatch",
                "message": "mounted Carbon weights do not match the pinned identity",
            }
        )
    if not runtime_identity_verified:
        blockers.append(
            {
                "code": "encoder_runtime_identity_mismatch",
                "message": "mounted Carbon runtime does not match the pinned identity",
            }
        )
    if not parameters_frozen:
        blockers.append(
            {
                "code": "encoder_parameters_not_frozen",
                "message": "Carbon encoder parameters are absent or remain trainable",
                "parameter_count": encoder_parameter_count,
                "trainable_parameter_count": encoder_trainable_parameter_count,
            }
        )
    if not pooling_identity_verified:
        blockers.append(
            {
                "code": "pooling_coordinate_mismatch",
                "message": "live Carbon pooling identity does not match the pinned DNA-token layout",
                "observed": {
                    "pool_type": resolved_pool_type,
                    "pool_radius": resolved_pool_radius,
                    "center_token": resolved_center_token,
                },
                "expected": {
                    "pool_type": "centered_mean",
                    "pool_radius": pool_radius,
                    "center_token": expected_center_token,
                },
            }
        )
    for index, (raw, normalized, sequence_hash) in enumerate(
        zip(raw_states, normalized_states, sequence_hashes, strict=True)
    ):
        if not raw or len(raw) != len(normalized):
            raise InputError(
                "raw and normalized state dimensions must match",
                details={"index": index, "raw_dim": len(raw), "normalized_dim": len(normalized)},
            )
        expected = l2_normalize_state(raw, item_index=index)
        raw_norm = math.hypot(*raw)
        normalized_norm = math.hypot(*normalized)
        max_abs_diff = max(
            abs(left - right) for left, right in zip(expected, normalized, strict=True)
        )
        dimension_ok = len(raw) == expected_d_state
        normalization_ok = (
            abs(normalized_norm - 1.0) <= unit_tolerance and max_abs_diff <= parity_tolerance
        )
        row_ok = dimension_ok and normalization_ok
        if not dimension_ok:
            blockers.append(
                {
                    "code": "state_dimension_mismatch",
                    "message": "live Carbon state width does not match the pinned checkpoint contract",
                    "index": index,
                    "observed": len(raw),
                    "expected": expected_d_state,
                }
            )
        if abs(raw_norm - 1.0) <= unit_tolerance:
            blockers.append(
                {
                    "code": "raw_state_already_unit_norm",
                    "message": "pinned Carbon raw pooling did not expose a scale-changing audit case",
                    "index": index,
                    "raw_norm": raw_norm,
                }
            )
        if not normalization_ok:
            blockers.append(
                {
                    "code": "normalized_state_contract_mismatch",
                    "message": "live normalized state failed unit-norm or raw-view parity",
                    "index": index,
                    "normalized_norm": normalized_norm,
                    "max_abs_diff": max_abs_diff,
                }
            )
        rows.append(
            {
                "index": index,
                "sequence_sha256": sequence_hash,
                "d_state": len(raw),
                "raw_norm": raw_norm,
                "normalized_norm": normalized_norm,
                "max_abs_diff_vs_normalized_raw": max_abs_diff,
                "ok": row_ok,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "ok": not blockers,
        "commit_sha": commit_sha,
        "encoder": {
            "model_dir": carbon_model_dir.name,
            "revision": carbon_revision,
            "weights_hash": carbon_weights_hash,
            "expected_weights_hash": expected_carbon_weights_hash,
            "weights_identity_verified": weights_identity_verified,
            "runtime_hash": carbon_runtime_hash,
            "expected_runtime_hash": expected_carbon_runtime_hash,
            "runtime_identity_verified": runtime_identity_verified,
            "parameter_count": encoder_parameter_count,
            "trainable_parameter_count": encoder_trainable_parameter_count,
            "parameters_frozen": parameters_frozen,
            "expected_d_state": expected_d_state,
            "window_bp": window_bp,
            "state_layer": state_layer,
            "pool_type": "centered_mean",
            "pool_radius": pool_radius,
            "resolved_pool_type": resolved_pool_type,
            "resolved_pool_radius": resolved_pool_radius,
            "resolved_center_token": resolved_center_token,
            "expected_center_token": expected_center_token,
            "pooling_identity_verified": pooling_identity_verified,
            "dtype": dtype,
            "raw_cache_representation": "post_pool_pre_normalization",
            "normalized_state_contract": "l2_normalized_v2",
        },
        "runtime": _runtime_identity(execution_device=execution_device),
        "tolerances": {
            "unit_norm": unit_tolerance,
            "raw_view_parity": parity_tolerance,
        },
        "rows": rows,
        "blockers": blockers,
        "claim_boundary": (
            "This audit verifies encoder state normalization semantics only. It does not validate "
            "checkpoint quality, variant-effect prediction, rollout fidelity, planning, or clinical use."
        ),
    }


def run_state_contract_audit(
    *,
    carbon_model_dir: Path,
    carbon_revision: str,
    expected_carbon_weights_hash: str,
    expected_carbon_runtime_hash: str,
    commit_sha: str,
    output_json: Path,
    window_bp: int = 12_288,
    state_layer: int = 20,
    pool_radius: int = 8,
    expected_d_state: int = 1024,
    dtype: str = "bf16",
    device: str | None = None,
    trust_remote_code: bool = True,
) -> dict[str, object]:
    """Run the pinned Carbon encoder in raw and corrected normalized modes."""
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise InputError(
            "state-contract audit requires an explicitly offline Hugging Face runtime",
            details={
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
            remediation="set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1",
        )
    validate_audit_window_bp(window_bp)
    observed_carbon_weights_hash = verify_encoder_weights(
        carbon_model_dir,
        expected_hash=expected_carbon_weights_hash,
    )
    observed_carbon_runtime_hash = verify_encoder_runtime(
        carbon_model_dir,
        expected_hash=expected_carbon_runtime_hash,
    )
    sequence = ("ACGTGC" * ((window_bp + 5) // 6))[:window_bp]
    edit_locus = window_bp // 2
    encoder = CarbonStateEncoder(
        str(carbon_model_dir),
        carbon_revision,
        dtype=dtype,
        state_layer=state_layer,
        pool_type="centered_mean",
        pool_radius=pool_radius,
        normalize=False,
        encoder_hash=observed_carbon_runtime_hash,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    if encoder.parameter_count <= 0:
        raise InputError("loaded Carbon encoder must expose at least one parameter")
    if encoder.trainable_parameter_count != 0:
        raise InputError(
            "loaded Carbon encoder must have zero trainable parameters",
            details={"trainable_parameter_count": encoder.trainable_parameter_count},
        )
    raw_states = encoder.encode_batch([sequence], [edit_locus])
    encoder.normalize = True
    normalized_states = encoder.encode_batch([sequence], [edit_locus])
    resolved_pool_type, resolved_pool_radius, resolved_center_token = encoder.pooling_identity(
        sequence,
        edit_locus,
    )
    expected_center_token = 1 + (edit_locus // CARBON_TOKEN_BP)
    report = build_state_contract_report(
        raw_states=raw_states,
        normalized_states=normalized_states,
        sequence_hashes=(hashlib.sha256(sequence.encode("ascii")).hexdigest(),),
        commit_sha=commit_sha,
        carbon_model_dir=carbon_model_dir,
        carbon_revision=carbon_revision,
        carbon_weights_hash=observed_carbon_weights_hash,
        expected_carbon_weights_hash=expected_carbon_weights_hash,
        carbon_runtime_hash=observed_carbon_runtime_hash,
        expected_carbon_runtime_hash=expected_carbon_runtime_hash,
        encoder_parameter_count=encoder.parameter_count,
        encoder_trainable_parameter_count=encoder.trainable_parameter_count,
        expected_d_state=expected_d_state,
        resolved_pool_type=resolved_pool_type,
        resolved_pool_radius=resolved_pool_radius,
        resolved_center_token=resolved_center_token,
        expected_center_token=expected_center_token,
        execution_device=encoder.device,
        window_bp=window_bp,
        state_layer=state_layer,
        pool_radius=pool_radius,
        dtype=dtype,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_audit_window_bp(window_bp: int) -> None:
    """Require one encoder-supported window width for audit/train parity."""
    if (
        isinstance(window_bp, bool)
        or not isinstance(window_bp, int)
        or window_bp not in SUPPORTED_WINDOW_BP
    ):
        raise InputError(
            "state-contract audit window_bp is unsupported",
            details={"window_bp": window_bp, "supported": list(SUPPORTED_WINDOW_BP)},
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_state_contract_audit(
            carbon_model_dir=args.carbon_model_dir,
            carbon_revision=args.carbon_revision,
            expected_carbon_weights_hash=args.expected_carbon_weights_hash,
            expected_carbon_runtime_hash=args.expected_carbon_runtime_hash,
            commit_sha=args.commit_sha,
            output_json=args.output_json,
            window_bp=args.window_bp,
            state_layer=args.state_layer,
            pool_radius=args.pool_radius,
            expected_d_state=args.expected_d_state,
            dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] is True else 1


def _runtime_identity(*, execution_device: str) -> dict[str, object]:
    identity: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": execution_device,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }
    try:
        torch = importlib.import_module("torch")
        identity["torch"] = torch.__version__
        identity["cuda_available"] = bool(torch.cuda.is_available())
        identity["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        mps_backend = getattr(torch.backends, "mps", None)
        identity["mps_available"] = bool(mps_backend is not None and mps_backend.is_available())
    except ImportError:
        identity["torch"] = None
        identity["cuda_available"] = False
        identity["cuda_device_name"] = None
        identity["mps_available"] = False
    return identity


def verify_encoder_weights(carbon_model_dir: Path, *, expected_hash: str) -> str:
    """Return the mounted encoder hash after matching the pinned Hub identity."""
    observed_hash = encoder_weights_hash(carbon_model_dir)
    if observed_hash != expected_hash:
        raise InputError(
            "mounted Carbon weights do not match the pinned audit identity",
            details={
                "carbon_model_dir": str(carbon_model_dir),
                "expected": expected_hash,
                "observed": observed_hash,
            },
            remediation="mount the pinned Carbon-500M revision or update both revision and hash",
        )
    return observed_hash


def verify_encoder_runtime(carbon_model_dir: Path, *, expected_hash: str) -> str:
    """Return the mounted runtime hash after matching the pinned Hub revision."""
    observed_hash = encoder_runtime_hash(carbon_model_dir)
    if observed_hash != expected_hash:
        raise InputError(
            "mounted Carbon runtime does not match the pinned audit identity",
            details={
                "carbon_model_dir": str(carbon_model_dir),
                "expected": expected_hash,
                "observed": observed_hash,
            },
            remediation="mount the pinned Carbon-500M revision or update the runtime hash",
        )
    return observed_hash


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon-model-dir", type=Path, required=True)
    parser.add_argument("--carbon-revision", default=DEFAULT_CARBON_REVISION)
    parser.add_argument(
        "--expected-carbon-weights-hash",
        default=DEFAULT_CARBON_WEIGHTS_HASH,
    )
    parser.add_argument(
        "--expected-carbon-runtime-hash",
        default=DEFAULT_CARBON_RUNTIME_HASH,
    )
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--window-bp", type=int, default=12_288)
    parser.add_argument("--state-layer", type=int, default=20)
    parser.add_argument("--pool-radius", type=int, default=8)
    parser.add_argument("--expected-d-state", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
