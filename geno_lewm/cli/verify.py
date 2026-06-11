# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-verify`` CLI — checksum-mode receipt verifier (artifact-provenance contract).

The CLI takes a receipt JSON path and validates the receipt against a
local manifest. Three checks compose the v1 protocol:

1. **Receipt schema** — ``read_receipt`` already raises
   :class:`ReceiptSchemaError` on a malformed receipt.
2. **Manifest hash** — recompute ``model_id`` from the manifest and
   compare against the receipt's ``model_id``. Mismatch raises
   :class:`ManifestHashMismatchError`.
3. **Input commitment** — if the caller passes ``--input-window`` plus
   the edit / pool / dtype flags, recompute the commitment and
   compare. Mismatch raises :class:`InputCommitmentMismatchError`.
4. **Output commitment** — recompute the output commitment from the
   receipt's ``output`` block and compare against the stored
   ``output_commitment``. Mismatch raises
   :class:`OutputCommitmentMismatchError`.

5. **Bit-exact re-inference** (``--rerun``) — load the deploy runtime
   from ``--model-dir``, re-score the receipt's input (passed via
   ``--input-window`` + ``--edit-{chrom,pos,ref,alt}``), recompute the
   output commitment, and require it to match the receipt bit-for-bit.
   Mismatch raises :class:`OutputCommitmentMismatchError`. This proves
   the receipt describes output the model actually produces, not merely
   that the receipt is internally self-consistent.

Exit codes follow error taxonomy / ``public API contract``:

- 0 — verification passed.
- 2 — `InputError` (bad CLI args).
- 3 — `ConfigError` (e.g. manifest schema mismatch).
- 8 — any ``ProvenanceError`` subclass (the main failure mode).
- 1 — uncategorized.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from geno_lewm.action.spec import EditSpec
from geno_lewm.deploy import GenoLeWMRuntime
from geno_lewm.errors import (
    GenoLeWMError,
    InputCommitmentMismatchError,
    InputError,
    ManifestHashMismatchError,
    OutputCommitmentMismatchError,
    exit_code_for,
)
from geno_lewm.provenance import (
    SUPPORTED_PROVENANCE_KINDS,
    DtypeConfig,
    PoolingConfig,
    ReceiptOutput,
    compute_input_commitment,
    compute_output_commitment,
    load_manifest,
    read_receipt,
)

__all__ = ["VERIFIER_SUPPORTED_KINDS", "main", "verify"]

VERIFIER_SUPPORTED_KINDS: frozenset[str] = SUPPORTED_PROVENANCE_KINDS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geno-lewm-verify",
        description="Verify a GenoLeWM inference receipt (checksum mode).",
    )
    p.add_argument("receipt", type=Path, help="path to the receipt JSON file")
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to the manifest.json this receipt claims to come from",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="re-run inference from --model-dir and require a bit-exact output commitment",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        help="deploy model directory used for --rerun bit-exact re-inference",
    )

    # Optional input-commitment recomputation. All four fields must be
    # passed together; without them, the input-commitment step is
    # skipped (the receipt is still useful as a manifest-anchored
    # output-commitment witness).
    g = p.add_argument_group("input-commitment recomputation (optional)")
    g.add_argument("--input-window", help="reference window bases (uppercase ACGT)")
    g.add_argument("--edit-chrom")
    g.add_argument("--edit-pos", type=int)
    g.add_argument("--edit-ref")
    g.add_argument("--edit-alt")
    g.add_argument("--state-layer", type=int)
    g.add_argument("--pool-type")
    g.add_argument("--pool-radius", type=int)
    g.add_argument("--normalize", action="store_true")
    g.add_argument("--encoder-dtype")
    g.add_argument("--predictor-dtype")
    return p


def _maybe_recompute_input_commitment(args: argparse.Namespace) -> str | None:
    """Build the input commitment from CLI args when present."""
    input_fields = (
        args.input_window,
        args.edit_chrom,
        args.edit_pos,
        args.edit_ref,
        args.edit_alt,
        args.state_layer,
        args.pool_type,
        args.pool_radius,
        args.encoder_dtype,
        args.predictor_dtype,
    )
    nonempty = [f for f in input_fields if f is not None]
    if not nonempty:
        return None
    if any(f is None for f in input_fields):
        raise InputError(
            "input-commitment recomputation requires all of "
            "--input-window, --edit-{chrom,pos,ref,alt}, "
            "--state-layer, --pool-type, --pool-radius, "
            "--encoder-dtype, --predictor-dtype",
            details={
                "missing": [
                    name
                    for name, val in zip(
                        [
                            "input_window",
                            "edit_chrom",
                            "edit_pos",
                            "edit_ref",
                            "edit_alt",
                            "state_layer",
                            "pool_type",
                            "pool_radius",
                            "encoder_dtype",
                            "predictor_dtype",
                        ],
                        input_fields,
                        strict=True,
                    )
                    if val is None
                ]
            },
        )
    edit = EditSpec(
        chrom=args.edit_chrom,
        pos=args.edit_pos,
        ref=args.edit_ref,
        alt=args.edit_alt,
    )
    pool = PoolingConfig(
        state_layer=args.state_layer,
        pool_type=args.pool_type,
        pool_radius=args.pool_radius,
        normalize=args.normalize,
    )
    dtype = DtypeConfig(
        encoder_dtype=args.encoder_dtype,
        predictor_dtype=args.predictor_dtype,
    )
    return compute_input_commitment(args.input_window, edit, pool, dtype)


def verify(args: argparse.Namespace, *, stream: IO[str] | None = None) -> None:
    """Run the verification protocol; raise on any failure.

    Side effect: writes a human-readable progress line per check to
    ``stream`` (defaults to ``sys.stdout`` resolved at call time, not
    at import time — so a pytest-captured stdout works without
    snapshot churn).
    """
    if stream is None:
        stream = sys.stdout
    print(f"reading receipt:  {args.receipt}", file=stream)
    receipt = read_receipt(args.receipt)
    print(
        f"  schema_version={receipt.schema_version} provenance.kind={receipt.provenance.kind}",
        file=stream,
    )

    print(f"reading manifest: {args.manifest}", file=stream)
    manifest = load_manifest(args.manifest)
    recomputed_model_id = manifest.model_id()
    if recomputed_model_id != receipt.model_id:
        raise ManifestHashMismatchError(
            "manifest hash does not match receipt.model_id",
            details={
                "receipt_model_id": receipt.model_id,
                "manifest_model_id": recomputed_model_id,
            },
            remediation="check that you are using the manifest the receipt was produced against",
        )
    print(f"  model_id ok ({recomputed_model_id[:23]}…)", file=stream)

    maybe_input = _maybe_recompute_input_commitment(args)
    if maybe_input is not None:
        if maybe_input != receipt.input_commitment:
            raise InputCommitmentMismatchError(
                "recomputed input commitment does not match the receipt",
                details={
                    "receipt": receipt.input_commitment,
                    "recomputed": maybe_input,
                },
            )
        print(f"  input_commitment ok ({receipt.input_commitment[:23]}…)", file=stream)
    else:
        print("  input_commitment: skipped (no input flags supplied)", file=stream)

    recomputed_output = compute_output_commitment(receipt.output)
    if recomputed_output != receipt.output_commitment:
        raise OutputCommitmentMismatchError(
            "recomputed output commitment does not match the receipt",
            details={
                "receipt": receipt.output_commitment,
                "recomputed": recomputed_output,
            },
        )
    print(f"  output_commitment ok ({recomputed_output[:23]}…)", file=stream)

    if args.rerun:
        rerun_commitment = _rerun_output_commitment(args)
        if rerun_commitment != receipt.output_commitment:
            raise OutputCommitmentMismatchError(
                "re-run inference output commitment does not match the receipt",
                details={
                    "receipt": receipt.output_commitment,
                    "rerun": rerun_commitment,
                },
                remediation=(
                    "ensure --model-dir is the exact model that produced the receipt; "
                    "a mismatch means the receipt does not describe this model's output"
                ),
            )
        print(f"  rerun output_commitment ok ({rerun_commitment[:23]}…)", file=stream)

    print("ok", file=stream)


def _rerun_output_commitment(args: argparse.Namespace) -> str:
    """Re-run inference from ``--model-dir`` and return the fresh output commitment.

    Requires the receipt's input to be supplied (``--model-dir`` plus
    ``--input-window`` and ``--edit-{chrom,pos,ref,alt}``). Loads the deploy
    runtime, scores the variant, and recomputes the output commitment so the
    caller can assert it matches the receipt bit-for-bit.
    """
    if args.model_dir is None:
        raise InputError(
            "--rerun requires --model-dir",
            remediation="pass --model-dir pointing at the deploy model directory",
        )
    required = {
        "input_window": args.input_window,
        "edit_chrom": args.edit_chrom,
        "edit_pos": args.edit_pos,
        "edit_ref": args.edit_ref,
        "edit_alt": args.edit_alt,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise InputError(
            "--rerun requires --input-window and --edit-{chrom,pos,ref,alt}",
            details={"missing": missing},
        )
    edit = EditSpec(
        chrom=args.edit_chrom,
        pos=args.edit_pos,
        ref=args.edit_ref,
        alt=args.edit_alt,
    )
    runtime = GenoLeWMRuntime(args.model_dir)
    result = runtime.score_variant(edit, window=args.input_window)
    output = ReceiptOutput(
        sigma_raw=result.sigma_raw,
        sigma_calibrated=result.sigma_calibrated,
        bucket_id=result.bucket_id,
        confidence=result.confidence,
        low_confidence=result.low_confidence,
    )
    return compute_output_commitment(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        verify(args, stream=sys.stdout)
    except GenoLeWMError as exc:
        # Print a one-line digest of code + message to stderr; full
        # details go to the structured logger when wired in (#23).
        print(f"verify failed: [{exc.code}] {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"  details: {exc.details}", file=sys.stderr)
        return exit_code_for(exc)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # uncategorized — bug in dispatch
        print(f"verify failed: uncategorized: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
