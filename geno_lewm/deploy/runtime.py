# SPDX-License-Identifier: Apache-2.0
"""On-device runtime contract and fail-closed network guard."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import platform
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY, SCORE_JSONL_SCHEMA_VERSION
from geno_lewm._inference import torch_inference_context
from geno_lewm.action import EditSpec, RelEdit
from geno_lewm.encoder.windowing import canonicalize_dna
from geno_lewm.errors import (
    BackendUnsupportedError,
    InputError,
    ManifestHashMismatchError,
    ModelNotFoundError,
    NetworkCallProhibitedError,
    RuntimeSetupError,
)
from geno_lewm.provenance import (
    RECEIPT_SCHEMA_VERSION,
    DtypeConfig,
    Manifest,
    PoolingConfig,
    Receipt,
    ReceiptOutput,
    ReceiptProvenance,
    ReceiptRuntime,
    compute_input_commitment,
    compute_output_commitment,
    load_manifest,
    sha256_file,
    write_receipt,
)
from geno_lewm.surprise import (
    CalibrationTable,
    SurpriseResult,
    read_calibration_table,
    score_variant as score_surprise_variant,
    score_vcf as score_surprise_vcf,
)
from geno_lewm.surprise.score import _iter_vcf_scores

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_COREML",
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_ONNX",
    "BACKEND_PRIORITY",
    "BackendProbe",
    "GenoLeWMRuntime",
    "fail_closed_network_guard",
    "load_scorer_modules",
    "probe_backends",
    "select_backend",
]


BACKEND_AUTO = "auto"
BACKEND_COREML = "coreml"
BACKEND_CUDA = "cuda"
BACKEND_ONNX = "onnx"
BACKEND_CPU = "cpu"
BACKEND_PRIORITY: tuple[str, ...] = (
    BACKEND_COREML,
    BACKEND_CUDA,
    BACKEND_ONNX,
    BACKEND_CPU,
)
_SUPPORTED_BACKENDS = (BACKEND_AUTO, *BACKEND_PRIORITY)
_APPLE_SILICON_MACHINES = frozenset({"arm64", "aarch64"})
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """Capability probe result for one runtime backend."""

    backend: str
    available: bool
    reason: str

    def __post_init__(self) -> None:
        if self.backend not in BACKEND_PRIORITY:
            raise BackendUnsupportedError(
                "unsupported runtime backend",
                details={"backend": self.backend, "supported": list(BACKEND_PRIORITY)},
            )
        if not isinstance(self.available, bool):
            raise InputError(
                "backend availability must be bool",
                details={"backend": self.backend, "type": type(self.available).__name__},
            )
        if not self.reason:
            raise InputError("backend probe reason must be non-empty")


@dataclass(frozen=True, slots=True)
class _RuntimeScorerComponents:
    encoder: object
    action_encoder: object
    predictor: object
    calibration: CalibrationTable


class GenoLeWMRuntime:
    """Top-level runtime facade for on-device inference workflows."""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = BACKEND_AUTO,
        *,
        encoder: object | None = None,
        action_encoder: object | None = None,
        predictor: object | None = None,
        calibration: CalibrationTable | None = None,
    ) -> None:
        root = Path(model_dir).expanduser()
        if not root.exists() or not root.is_dir():
            raise ModelNotFoundError(
                "model_dir must be an existing directory",
                details={"model_dir": str(root)},
            )
        self.model_dir = root
        self.manifest = _load_runtime_manifest(root)
        if self.manifest is not None:
            _verify_manifest_artifacts(root, self.manifest)
        self.probes = probe_backends(root)
        self.backend = select_backend(backend, probes=self.probes)
        self._scorer = _resolve_scorer_components(
            root,
            self.manifest,
            encoder=encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            calibration=calibration,
        )

    def score_variant(
        self,
        variant: EditSpec,
        window: str | None = None,
        *,
        receipt_path: str | Path | None = None,
    ) -> Any:
        """Score a single variant through local scorer components when available."""
        if not isinstance(variant, EditSpec):
            raise InputError(
                "variant must be an EditSpec",
                details={"type": type(variant).__name__},
            )
        normalized_window = None
        if window is not None:
            normalized_window = canonicalize_dna(window)
        scorer = self._scorer
        with fail_closed_network_guard(), torch_inference_context():
            if scorer is not None:
                if normalized_window is None:
                    raise InputError(
                        "score_variant requires a reference window",
                        remediation="pass window=... or use score_vcf with a local FASTA",
                    )
                result = score_surprise_variant(
                    variant,
                    scorer.encoder,
                    scorer.action_encoder,
                    scorer.predictor,
                    scorer.calibration,
                    reference_window=normalized_window,
                )
                if receipt_path is not None:
                    _write_score_variant_receipt(
                        backend=self.backend,
                        model_dir=self.model_dir,
                        manifest=self.manifest,
                        variant=variant,
                        reference_window=normalized_window,
                        result=result,
                        receipt_path=receipt_path,
                    )
                return result
            _raise_backend_not_ready("score_variant", self.backend, self.model_dir)

    def score_vcf(
        self,
        vcf_path: str | Path,
        fasta_path: str | Path,
        output_path: str | Path,
        batch_size: int = 64,
        progress: bool = True,
        *,
        receipt_path: str | Path | None = None,
    ) -> None:
        """Score a VCF through local scorer components when available.

        When ``receipt_path`` is provided, the runtime writes JSONL with
        one canonical v1 receipt per scored alternate. The v1 schema
        commits a single output, so this is intentionally not a batch
        aggregate receipt.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise InputError(
                "batch_size must be a positive integer",
                details={"batch_size": batch_size, "type": type(batch_size).__name__},
            )
        if not isinstance(progress, bool):
            raise InputError(
                "progress must be bool",
                details={"type": type(progress).__name__},
            )
        # Normalize path-like values now so type errors surface at the API boundary.
        Path(vcf_path)
        Path(fasta_path)
        normalized_output = Path(output_path)
        normalized_receipt = None if receipt_path is None else Path(receipt_path)
        if normalized_receipt is not None and normalized_receipt == normalized_output:
            raise InputError("--receipt must differ from --output for VCF scoring")
        scorer = self._scorer
        with fail_closed_network_guard(), torch_inference_context():
            if scorer is not None:
                if normalized_receipt is None:
                    score_surprise_vcf(
                        vcf_path,
                        scorer.encoder,
                        scorer.action_encoder,
                        scorer.predictor,
                        scorer.calibration,
                        normalized_output,
                        reference_fasta=fasta_path,
                        batch_size=batch_size,
                        show_progress=progress,
                    )
                else:
                    _write_vcf_scores_and_receipts(
                        backend=self.backend,
                        model_dir=self.model_dir,
                        manifest=self.manifest,
                        scorer=scorer,
                        vcf_path=vcf_path,
                        fasta_path=fasta_path,
                        output_path=normalized_output,
                        receipt_path=normalized_receipt,
                        batch_size=batch_size,
                    )
                return
            _raise_backend_not_ready("score_vcf", self.backend, self.model_dir)

    def encode_window(self, window: str, edit_locus: int | None = None) -> Any:
        """Encode a DNA window once the encoder backend is installed."""
        canonicalize_dna(window)
        if edit_locus is not None and (
            not isinstance(edit_locus, int) or isinstance(edit_locus, bool) or edit_locus < 0
        ):
            raise InputError(
                "edit_locus must be a non-negative integer or None",
                details={"edit_locus": edit_locus, "type": type(edit_locus).__name__},
            )
        with fail_closed_network_guard():
            _raise_backend_not_ready("encode_window", self.backend, self.model_dir)

    def predict(self, state: Any, edits: Sequence[RelEdit]) -> Any:
        """Run the predictor once a predictor backend is installed."""
        if state is None:
            raise InputError("state must be non-None")
        if not isinstance(edits, Sequence):
            raise InputError(
                "edits must be a sequence of RelEdit values",
                details={"type": type(edits).__name__},
            )
        for idx, edit in enumerate(edits):
            if not isinstance(edit, RelEdit):
                raise InputError(
                    "edits must contain RelEdit values",
                    details={"index": idx, "type": type(edit).__name__},
                )
        with fail_closed_network_guard():
            _raise_backend_not_ready("predict", self.backend, self.model_dir)


def probe_backends(model_dir: str | Path | None = None) -> tuple[BackendProbe, ...]:
    """Probe runtime backends in RFC-0010 auto-selection order."""
    root = None if model_dir is None else Path(model_dir).expanduser()
    return (
        _probe_coreml(root),
        _probe_cuda(root),
        _probe_onnx(root),
        BackendProbe(BACKEND_CPU, True, "portable CPU fallback is always available"),
    )


def select_backend(
    backend: str = BACKEND_AUTO,
    *,
    probes: Sequence[BackendProbe] | None = None,
) -> str:
    """Select a backend from probe results, or raise if the requested one is unavailable."""
    normalized = _normalize_backend(backend)
    observed = tuple(probe_backends() if probes is None else probes)
    by_backend = {probe.backend: probe for probe in observed}

    if normalized == BACKEND_AUTO:
        for name in BACKEND_PRIORITY:
            probe = by_backend.get(name)
            if probe is not None and probe.available:
                return name
        raise BackendUnsupportedError(
            "no runtime backend is available",
            details={"probes": [_probe_details(probe) for probe in observed]},
        )

    probe = by_backend.get(normalized)
    if probe is None:
        raise BackendUnsupportedError(
            "requested runtime backend was not probed",
            details={"backend": normalized, "probed": sorted(by_backend)},
        )
    if not probe.available:
        raise BackendUnsupportedError(
            "requested runtime backend is unavailable",
            details={"backend": normalized, "reason": probe.reason},
        )
    return normalized


@contextlib.contextmanager
def fail_closed_network_guard() -> Iterator[None]:
    """Block common network entry points inside an inference path."""

    def _blocked(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise NetworkCallProhibitedError(
            "runtime network call attempted after setup",
            remediation="perform downloads only through explicit setup/update commands",
        )

    with contextlib.ExitStack() as stack:
        for target in (
            "socket.create_connection",
            "socket.socket.connect",
            "urllib.request.urlopen",
            "http.client.HTTPConnection.connect",
            "http.client.HTTPSConnection.connect",
        ):
            stack.enter_context(patch(target, _blocked))
        yield


def _probe_coreml(model_dir: Path | None) -> BackendProbe:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin" or machine not in _APPLE_SILICON_MACHINES:
        return BackendProbe(
            BACKEND_COREML,
            False,
            f"Core ML requires Apple Silicon macOS; observed {system}/{machine}",
        )
    if importlib.util.find_spec("coremltools") is None:
        return BackendProbe(BACKEND_COREML, False, "coremltools is not installed")
    artifact_reason = _artifact_reason(model_dir, ("model.mlpackage", "model.mlmodelc"))
    if artifact_reason is not None:
        return BackendProbe(BACKEND_COREML, False, artifact_reason)
    return BackendProbe(BACKEND_COREML, True, "Core ML runtime and model artifact available")


def _probe_cuda(model_dir: Path | None) -> BackendProbe:
    available, reason = _torch_cuda_available()
    if not available:
        return BackendProbe(BACKEND_CUDA, False, reason)
    artifact_reason = _artifact_reason(model_dir, ("predictor.safetensors",))
    if artifact_reason is not None:
        return BackendProbe(BACKEND_CUDA, False, artifact_reason)
    return BackendProbe(BACKEND_CUDA, True, "torch CUDA backend and model artifact available")


def _probe_onnx(model_dir: Path | None) -> BackendProbe:
    if importlib.util.find_spec("onnxruntime") is None:
        return BackendProbe(BACKEND_ONNX, False, "onnxruntime is not installed")
    artifact_reason = _artifact_reason(model_dir, ("predictor.onnx",))
    if artifact_reason is not None:
        return BackendProbe(BACKEND_ONNX, False, artifact_reason)
    return BackendProbe(BACKEND_ONNX, True, "ONNX Runtime and model artifact available")


def _torch_cuda_available() -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None:
        return False, "torch is not installed"
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on optional torch installation
        return False, f"torch import failed: {exc}"
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return False, "torch CUDA is not available"
    return True, "torch CUDA is available"


def _artifact_reason(model_dir: Path | None, names: tuple[str, ...]) -> str | None:
    if model_dir is None:
        return None
    if any((model_dir / name).exists() for name in names):
        return None
    return "missing backend artifact(s): " + ", ".join(names)


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str) or not backend:
        raise BackendUnsupportedError(
            "runtime backend must be a non-empty string",
            details={"backend": backend, "type": type(backend).__name__},
        )
    normalized = backend.lower()
    if normalized not in _SUPPORTED_BACKENDS:
        raise BackendUnsupportedError(
            "unsupported runtime backend",
            details={"backend": backend, "supported": list(_SUPPORTED_BACKENDS)},
        )
    return normalized


def _probe_details(probe: BackendProbe) -> dict[str, str | bool]:
    return {
        "backend": probe.backend,
        "available": probe.available,
        "reason": probe.reason,
    }


def _load_runtime_manifest(model_dir: Path) -> Manifest | None:
    path = model_dir / _MANIFEST_NAME
    if not path.exists():
        return None
    if not path.is_file():
        raise ModelNotFoundError(
            "manifest.json must be a file",
            details={"path": str(path)},
        )
    return load_manifest(path)


def _resolve_scorer_components(
    model_dir: Path,
    manifest: Manifest | None,
    *,
    encoder: object | None,
    action_encoder: object | None,
    predictor: object | None,
    calibration: CalibrationTable | None,
) -> _RuntimeScorerComponents | None:
    supplied_model_parts = (encoder is not None, action_encoder is not None, predictor is not None)
    if any(supplied_model_parts) and not all(supplied_model_parts):
        raise InputError(
            "encoder, action_encoder, and predictor must be supplied together",
            details={
                "encoder": encoder is not None,
                "action_encoder": action_encoder is not None,
                "predictor": predictor is not None,
            },
        )
    if calibration is not None and not isinstance(calibration, CalibrationTable):
        raise InputError(
            "calibration must be a CalibrationTable",
            details={"type": type(calibration).__name__},
        )
    if not any(supplied_model_parts):
        if calibration is not None:
            raise InputError(
                "calibration cannot be supplied without encoder, action_encoder, and predictor",
            )
        if manifest is not None:
            return _try_load_manifest_scorer_components(model_dir, manifest)
        return None

    resolved_calibration = calibration
    if resolved_calibration is None:
        resolved_calibration = _load_runtime_calibration(model_dir, manifest)

    return _RuntimeScorerComponents(
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        calibration=resolved_calibration,
    )


def _build_loaded_scorer_modules(
    model_dir: Path,
    manifest: Manifest,
) -> tuple[object, object, object]:
    """Build encoder/action_encoder/predictor and load their safetensors weights."""
    cfg = _load_commitment_config_source(model_dir, manifest)
    encoder = _build_runtime_encoder(manifest, cfg)
    action_encoder = _build_runtime_action_encoder(cfg)
    predictor = _build_runtime_predictor(cfg)
    _load_module_state(
        action_encoder,
        _artifact_path(model_dir, manifest.action_encoder.file),
        artifact="action_encoder",
    )
    _load_module_state(
        predictor,
        _artifact_path(model_dir, manifest.predictor.file),
        artifact="predictor",
    )
    return encoder, action_encoder, predictor


def _try_load_manifest_scorer_components(
    model_dir: Path,
    manifest: Manifest,
) -> _RuntimeScorerComponents | None:
    if not _native_scorer_runtime_available():
        return None
    try:
        encoder, action_encoder, predictor = _build_loaded_scorer_modules(model_dir, manifest)
        return _RuntimeScorerComponents(
            encoder=encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            calibration=_load_runtime_calibration(model_dir, manifest),
        )
    except RuntimeSetupError:
        raise
    except Exception as exc:
        raise RuntimeSetupError(
            "could not load manifest-backed runtime scorer components",
            details={"model_dir": str(model_dir), "error": str(exc)},
            remediation=(
                "verify that the checkpoint was exported for this geno-lewm version "
                "and install geno-lewm[train]"
            ),
        ) from exc


def load_scorer_modules(model_dir: Path | str) -> tuple[object, object, object]:
    """Load (encoder, action_encoder, predictor) from a model dir without calibration.

    Used by calibration-table generation, which must run the model over a
    background set *before* ``calibration.parquet`` exists and therefore cannot
    use :class:`GenoLeWMRuntime` (whose scorer requires a calibration artifact).
    Requires ``geno-lewm[train]`` (torch + transformers + safetensors) and a
    ``manifest.json`` plus the exported ``predictor``/``action_encoder``
    safetensors artifacts.
    """
    model_dir = Path(model_dir)
    manifest = _load_runtime_manifest(model_dir)
    if manifest is None:
        raise ModelNotFoundError(
            "model_dir must contain manifest.json",
            details={"model_dir": str(model_dir)},
        )
    if not _native_scorer_runtime_available():
        raise RuntimeSetupError(
            "scoring requires torch, transformers, and safetensors",
            remediation="install geno-lewm[train]",
        )
    try:
        return _build_loaded_scorer_modules(model_dir, manifest)
    except RuntimeSetupError:
        raise
    except Exception as exc:
        raise RuntimeSetupError(
            "could not load scorer modules from model_dir",
            details={"model_dir": str(model_dir), "error": str(exc)},
            remediation=(
                "verify that the checkpoint was exported for this geno-lewm version "
                "and install geno-lewm[train]"
            ),
        ) from exc


def _native_scorer_runtime_available() -> bool:
    return all(
        _optional_module_available(name) for name in ("torch", "safetensors.torch", "transformers")
    )


def _optional_module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _build_runtime_encoder(manifest: Manifest, cfg: Any) -> object:
    from geno_lewm.encoder import CarbonStateEncoder

    return CarbonStateEncoder(
        cfg.encoder.model_id,
        manifest.encoder.revision,
        dtype=cfg.encoder.dtype,
        state_layer=cfg.encoder.state_layer,
        pool_type=cfg.encoder.pool_type,
        pool_radius=cfg.encoder.pool_radius,
        normalize=cfg.encoder.normalize,
        encoder_hash=manifest.encoder.hash,
        local_files_only=True,
        trust_remote_code=getattr(cfg.encoder, "trust_remote_code", False),
    )


def _build_runtime_action_encoder(cfg: Any) -> object:
    from geno_lewm.action import ActionEncoder

    return ActionEncoder(
        d_action=cfg.action.d_action,
        max_window_bp=getattr(cfg.encoder, "window_bp", 12_288),
    )


def _build_runtime_predictor(cfg: Any) -> object:
    # Must build the SAME predictor as training (geno_lewm.training.real) or an
    # exported checkpoint will not load. build_predictor is the shared source of
    # truth for that construction.
    from geno_lewm.predictor import build_predictor

    return build_predictor(cfg)


def _load_module_state(module: object, path: Path, *, artifact: str) -> None:
    try:
        safetensors_torch = importlib.import_module("safetensors.torch")
    except ImportError as exc:  # pragma: no cover - gated by availability check.
        raise RuntimeSetupError(
            "safetensors is required to load runtime components",
            remediation="install geno-lewm[train]",
        ) from exc
    load_file = getattr(safetensors_torch, "load_file", None)
    if not callable(load_file):
        raise RuntimeSetupError("safetensors.torch.load_file is unavailable")
    state_dict = load_file(str(path))
    load_state_dict = getattr(module, "load_state_dict", None)
    if not callable(load_state_dict):
        raise RuntimeSetupError(
            "runtime component does not support load_state_dict",
            details={"artifact": artifact, "type": type(module).__name__},
        )
    load_state_dict(state_dict, strict=True)
    eval_method = getattr(module, "eval", None)
    if callable(eval_method):
        eval_method()


def _resolve_commitment_configs(
    model_dir: Path,
    manifest: Manifest | None,
) -> tuple[PoolingConfig, DtypeConfig]:
    cfg = _load_commitment_config_source(model_dir, manifest)
    resolved_pooling = PoolingConfig(
        state_layer=cfg.encoder.state_layer,
        pool_type=cfg.encoder.pool_type,
        pool_radius=cfg.encoder.pool_radius,
        normalize=cfg.encoder.normalize,
    )

    predictor_dtype = cfg.predictor.dtype
    if manifest is not None and manifest.predictor.dtype is not None:
        predictor_dtype = manifest.predictor.dtype
    resolved_dtype = DtypeConfig(
        encoder_dtype=cfg.encoder.dtype,
        predictor_dtype=predictor_dtype,
    )

    return resolved_pooling, resolved_dtype


def _load_commitment_config_source(model_dir: Path, manifest: Manifest | None) -> Any:
    from geno_lewm.config import load_config, load_default

    if manifest is None:
        return load_default("score")
    return load_config(_artifact_path(model_dir, manifest.training.config_file))


def _require_receipt_manifest(manifest: Manifest | None) -> Manifest:
    if manifest is None:
        raise RuntimeSetupError(
            "receipt writing requires manifest.json",
            remediation="provide a model directory with a verified manifest.json",
        )
    return manifest


def _write_score_variant_receipt(
    *,
    backend: str,
    model_dir: Path,
    manifest: Manifest | None,
    variant: EditSpec,
    reference_window: str,
    result: Any,
    receipt_path: str | Path,
) -> Path:
    resolved_manifest = _require_receipt_manifest(manifest)
    pooling_config, dtype_config = _resolve_commitment_configs(model_dir, resolved_manifest)
    receipt = _build_score_receipt(
        backend=backend,
        manifest=resolved_manifest,
        variant=variant,
        reference_window=reference_window,
        result=result,
        pooling_config=pooling_config,
        dtype_config=dtype_config,
        scope="single_variant",
    )
    return write_receipt(receipt, receipt_path)


def _write_vcf_scores_and_receipts(
    *,
    backend: str,
    model_dir: Path,
    manifest: Manifest | None,
    scorer: _RuntimeScorerComponents,
    vcf_path: str | Path,
    fasta_path: str | Path,
    output_path: Path,
    receipt_path: Path,
    batch_size: int,
) -> None:
    resolved_manifest = _require_receipt_manifest(manifest)
    pooling_config, dtype_config = _resolve_commitment_configs(model_dir, resolved_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("w", encoding="utf-8") as score_handle,
        receipt_path.open("w", encoding="utf-8") as receipt_handle,
    ):
        for row_index, record in enumerate(
            _iter_vcf_scores(
                vcf_path,
                scorer.encoder,
                scorer.action_encoder,
                scorer.predictor,
                scorer.calibration,
                reference_fasta=fasta_path,
                batch_size=batch_size,
            ),
            start=1,
        ):
            variant = record.variant
            score_handle.write(
                json.dumps(
                    {
                        "schema_version": SCORE_JSONL_SCHEMA_VERSION,
                        "generated_by": SCORE_JSONL_GENERATED_BY,
                        "chrom": variant.chrom,
                        "pos": variant.pos,
                        "ref": variant.ref,
                        "alt": variant.alt,
                        **record.result.to_dict(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            receipt = _build_score_receipt(
                backend=backend,
                manifest=resolved_manifest,
                variant=variant,
                reference_window=record.reference_window,
                result=record.result,
                pooling_config=pooling_config,
                dtype_config=dtype_config,
                scope="vcf_row",
                details={
                    "receipt_stream": "jsonl_per_scored_alternate_v1",
                    "row_index": row_index,
                },
            )
            receipt_handle.write(receipt.to_canonical_json().decode("utf-8") + "\n")


def _build_score_receipt(
    *,
    backend: str,
    manifest: Manifest,
    variant: EditSpec,
    reference_window: str,
    result: Any,
    pooling_config: PoolingConfig,
    dtype_config: DtypeConfig,
    scope: str,
    details: dict[str, Any] | None = None,
) -> Receipt:
    output = _receipt_output(result)
    provenance_details = _receipt_provenance_details(
        pooling_config=pooling_config,
        dtype_config=dtype_config,
        scope=scope,
    )
    if details is not None:
        provenance_details.update(details)
    return Receipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        model_id=manifest.model_id(),
        input_commitment=compute_input_commitment(
            reference_window,
            variant,
            pooling_config,
            dtype_config,
        ),
        output=output,
        output_commitment=compute_output_commitment(output),
        calibration_hash=manifest.calibration.hash,
        runtime=ReceiptRuntime(
            backend=backend,
            device=_receipt_device_name(backend),
            geno_lewm_version=_geno_lewm_version(),
            carbon_revision=manifest.encoder.revision,
        ),
        timestamp=_utc_timestamp(),
        provenance=ReceiptProvenance(
            kind="checksum_only",
            details=provenance_details,
        ),
    )


def _receipt_provenance_details(
    *,
    pooling_config: PoolingConfig,
    dtype_config: DtypeConfig,
    scope: str,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "input_commitment_schema": "reference_window_edit_pool_dtype_v1",
        "pooling_config": {
            "state_layer": pooling_config.state_layer,
            "pool_type": pooling_config.pool_type,
            "pool_radius": pooling_config.pool_radius,
            "normalize": pooling_config.normalize,
        },
        "dtype_config": {
            "encoder_dtype": dtype_config.encoder_dtype,
            "predictor_dtype": dtype_config.predictor_dtype,
        },
    }


def _receipt_output(result: Any) -> ReceiptOutput:
    if isinstance(result, SurpriseResult):
        return ReceiptOutput(
            sigma_raw=result.sigma_raw,
            sigma_calibrated=result.sigma_calibrated,
            bucket_id=result.bucket_id,
            confidence=result.confidence,
            low_confidence=result.low_confidence,
        )
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    else:
        payload = result
    if not isinstance(payload, Mapping):
        raise InputError(
            "score result must be a SurpriseResult or mapping to write a receipt",
            details={"type": type(payload).__name__},
        )
    return ReceiptOutput(
        sigma_raw=_payload_float(payload, "sigma_raw"),
        sigma_calibrated=_payload_float(payload, "sigma_calibrated"),
        bucket_id=_payload_str(payload, "bucket_id"),
        confidence=_payload_float(payload, "confidence"),
        low_confidence=_payload_bool(payload, "low_confidence"),
    )


def _payload_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(
            f"score result field {key} must be numeric",
            details={"field": key, "type": type(value).__name__},
        )
    return float(value)


def _payload_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(
            f"score result field {key} must be a non-empty string",
            details={"field": key, "type": type(value).__name__},
        )
    return value


def _payload_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise InputError(
            f"score result field {key} must be bool",
            details={"field": key, "type": type(value).__name__},
        )
    return value


def _receipt_device_name(backend: str) -> str:
    machine = platform.machine() or "unknown"
    if backend == BACKEND_CPU:
        return f"cpu/{machine}"
    if backend == BACKEND_CUDA:
        return "cuda"
    if backend == BACKEND_COREML:
        return f"coreml/{machine}"
    if backend == BACKEND_ONNX:
        return f"onnxruntime/{machine}"
    return f"{backend}/{machine}"


def _geno_lewm_version() -> str:
    from geno_lewm import __version__

    return __version__


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_runtime_calibration(model_dir: Path, manifest: Manifest | None) -> CalibrationTable:
    if manifest is None:
        raise RuntimeSetupError(
            "runtime scoring components require a calibration table",
            remediation=(
                "pass calibration=... or provide a manifest.json with a calibration artifact"
            ),
        )
    return read_calibration_table(_artifact_path(model_dir, manifest.calibration.file))


def _verify_manifest_artifacts(model_dir: Path, manifest: Manifest) -> None:
    artifacts = {
        "predictor": manifest.predictor,
        "action_encoder": manifest.action_encoder,
        "calibration": manifest.calibration,
        "eval": manifest.eval,
    }
    for name, artifact in artifacts.items():
        _verify_artifact_hash(model_dir, name, artifact.file, artifact.hash)
    _verify_artifact_hash(
        model_dir,
        "training",
        manifest.training.config_file,
        manifest.training.hash,
    )


def _verify_artifact_hash(model_dir: Path, name: str, file_name: str, expected_hash: str) -> None:
    path = _artifact_path(model_dir, file_name)
    if not path.is_file():
        raise ModelNotFoundError(
            "manifest artifact is missing",
            details={"artifact": name, "path": str(path)},
        )
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ManifestHashMismatchError(
            "manifest artifact hash mismatch",
            details={
                "artifact": name,
                "path": str(path),
                "expected": expected_hash,
                "observed": observed,
            },
        )


def _artifact_path(model_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute() or ".." in path.parts:
        raise InputError(
            "manifest artifact paths must stay inside model_dir",
            details={"file": file_name},
        )
    return model_dir / path


def _raise_backend_not_ready(operation: str, backend: str, model_dir: Path) -> NoReturn:
    raise RuntimeSetupError(
        "deploy runtime backend operation is not available yet",
        details={"operation": operation, "backend": backend, "model_dir": str(model_dir)},
        remediation="score/export backends land with the scorer and deploy backend issues",
    )
