# SPDX-License-Identifier: Apache-2.0
"""Verify the mounted Carbon runtime identity without publishing artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from geno_lewm.encoder._identity import encoder_runtime_hash
from geno_lewm.errors import GenoLeWMError

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
_TERMINAL_MARKER = "GENO_LEWM_V03_CARBON_RUNTIME_HASH_PROBE_OK"


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """Immutable inputs bound by one runtime-hash probe."""

    carbon_dir: Path
    source_commit: str
    container_image: str
    carbon_repository: str
    carbon_revision: str
    expected_runtime_hash: str
    flavor: str
    namespace: str
    purpose: str
    timeout: str
    run_attempt: int


def verify_runtime_hash(request: ProbeRequest) -> str:
    """Return the observed hash only when it matches the explicit expectation."""
    observed = encoder_runtime_hash(request.carbon_dir)
    if observed != request.expected_runtime_hash:
        raise RuntimeError(
            "mounted Carbon runtime hash differs from the explicit expectation: "
            f"expected {request.expected_runtime_hash}, observed {observed}"
        )
    return observed


def terminal_marker(request: ProbeRequest, *, observed_runtime_hash: str) -> str:
    """Render the single terminal success line for a verified probe."""
    return " ".join(
        (
            _TERMINAL_MARKER,
            f"source_commit={request.source_commit}",
            f"carbon_repository={request.carbon_repository}",
            f"carbon_revision={request.carbon_revision}",
            f"carbon_mount_path={request.carbon_dir}",
            f"encoder_runtime_hash={observed_runtime_hash}",
            f"container_image={request.container_image}",
            f"flavor={request.flavor}",
            f"namespace={request.namespace}",
            f"purpose={request.purpose}",
            f"timeout={request.timeout}",
            f"run_attempt={request.run_attempt}",
        )
    )


def _exact_commit(value: str) -> str:
    if _COMMIT.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a full lowercase 40-character Git SHA")
    return value


def _exact_sha256(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a full lowercase sha256:<64-hex> digest")
    return value


def _positive_integer(value: str) -> int:
    if _POSITIVE_INTEGER.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a positive canonical integer")
    return int(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True, type=_exact_commit)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--carbon-repository", required=True)
    parser.add_argument("--carbon-revision", required=True, type=_exact_commit)
    parser.add_argument("--expected-runtime-hash", required=True, type=_exact_sha256)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--timeout", required=True)
    parser.add_argument("--run-attempt", required=True, type=_positive_integer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = ProbeRequest(
        carbon_dir=args.carbon_dir,
        source_commit=args.source_commit,
        container_image=args.container_image,
        carbon_repository=args.carbon_repository,
        carbon_revision=args.carbon_revision,
        expected_runtime_hash=args.expected_runtime_hash,
        flavor=args.flavor,
        namespace=args.namespace,
        purpose=args.purpose,
        timeout=args.timeout,
        run_attempt=args.run_attempt,
    )
    try:
        observed = verify_runtime_hash(request)
    except (GenoLeWMError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(terminal_marker(request, observed_runtime_hash=observed))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the job command.
    raise SystemExit(main())
