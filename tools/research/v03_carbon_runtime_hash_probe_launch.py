# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub==1.8.0",
# ]
# ///
"""Launch the exact-volume v0.3 Carbon runtime-hash CPU probe."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

HUGGINGFACE_HUB_VERSION: Final = "1.8.0"
NAMESPACE: Final = "abdelstark"
FLAVOR: Final = "cpu-basic"
TIMEOUT: Final = "30m"
PURPOSE: Final = "geno-lewm-v03-carbon-runtime-hash-probe"
CONTAINER_IMAGE: Final = (
    "ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
)
CARBON_REPOSITORY: Final = "HuggingFaceBio/Carbon-500M"
CARBON_REVISION: Final = "5d31d59b3c845b288a13aedb1358934196852eec"
CARBON_MOUNT_PATH: Final = "/carbon"
_ORIGIN: Final = "https://github.com/AbdelStark/GenoLeWM.git"
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_COMMIT: Final = re.compile(r"[0-9a-f]{40}\Z")
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_POSITIVE_INTEGER: Final = re.compile(r"[1-9][0-9]*\Z")
_INNER_COMMAND: Final = (
    "set -euo pipefail; "
    "test -d /carbon; "
    "test ! -L /carbon; "
    'test "$(cd /carbon && pwd -P)" = /carbon; '
    "test ! -L /workspace; "
    "if [ ! -e /workspace ]; then mkdir /workspace; fi; "
    "test -d /workspace; "
    'test "$(cd /workspace && pwd -P)" = /workspace; '
    "test -w /workspace; "
    "test ! -e /workspace/GenoLeWM; "
    "git clone --filter=blob:none --no-checkout "
    "https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; "
    "cd /workspace/GenoLeWM; "
    'git checkout --detach "$COMMIT_SHA"; '
    'test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; '
    'test "$(git remote get-url origin)" = "https://github.com/AbdelStark/GenoLeWM.git"; '
    'test -z "$(git status --porcelain=v1 --untracked-files=all)"; '
    "uv sync --frozen --no-dev; "
    "exec uv run --no-sync python -m tools.research.v03_carbon_runtime_hash_probe "
    '--carbon-dir "$CARBON_MOUNT_PATH" '
    '--source-commit "$COMMIT_SHA" '
    '--container-image "$CONTAINER_IMAGE" '
    '--carbon-repository "$CARBON_REPOSITORY" '
    '--carbon-revision "$CARBON_REVISION" '
    '--expected-runtime-hash "$EXPECTED_RUNTIME_HASH" '
    '--flavor "$JOB_FLAVOR" '
    '--namespace "$JOB_NAMESPACE" '
    '--purpose "$JOB_PURPOSE" '
    '--timeout "$JOB_TIMEOUT" '
    '--run-attempt "$RUN_ATTEMPT"'
)


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    """Exact public model-volume request."""

    type: str
    source: str
    mount_path: str
    revision: str
    read_only: bool
    path: str | None = None


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Secret-free public contract for one CPU probe launch."""

    image: str
    command: tuple[str, ...]
    environment: dict[str, str]
    secret_names: tuple[str, ...]
    flavor: str
    labels: dict[str, str]
    timeout: str
    namespace: str
    volume: VolumeSpec

    def public_payload(self) -> dict[str, object]:
        """Return all public launch fields and no credential metadata."""
        return {
            "image": self.image,
            "command": list(self.command),
            "environment": dict(self.environment),
            "flavor": self.flavor,
            "labels": dict(self.labels),
            "timeout": self.timeout,
            "namespace": self.namespace,
            "volumes": [asdict(self.volume)],
        }


def build_launch_spec(
    *,
    source_commit: str,
    expected_runtime_hash: str,
    run_attempt: int,
) -> LaunchSpec:
    """Build the one allowed exact-source, exact-model CPU probe contract."""
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase 40-character Git SHA")
    if _SHA256.fullmatch(expected_runtime_hash) is None:
        raise ValueError("expected_runtime_hash must be a full lowercase sha256:<64-hex> digest")
    if type(run_attempt) is not int or run_attempt <= 0:
        raise ValueError("run_attempt must be a positive integer")
    return LaunchSpec(
        image=CONTAINER_IMAGE,
        command=("bash", "-lc", _INNER_COMMAND),
        environment={
            "COMMIT_SHA": source_commit,
            "CONTAINER_IMAGE": CONTAINER_IMAGE,
            "CARBON_REPOSITORY": CARBON_REPOSITORY,
            "CARBON_REVISION": CARBON_REVISION,
            "CARBON_MOUNT_PATH": CARBON_MOUNT_PATH,
            "EXPECTED_RUNTIME_HASH": expected_runtime_hash,
            "JOB_FLAVOR": FLAVOR,
            "JOB_NAMESPACE": NAMESPACE,
            "JOB_PURPOSE": PURPOSE,
            "JOB_TIMEOUT": TIMEOUT,
            "RUN_ATTEMPT": str(run_attempt),
        },
        secret_names=(),
        flavor=FLAVOR,
        labels={"purpose": PURPOSE},
        timeout=TIMEOUT,
        namespace=NAMESPACE,
        volume=VolumeSpec(
            type="model",
            source=CARBON_REPOSITORY,
            mount_path=CARBON_MOUNT_PATH,
            revision=CARBON_REVISION,
            read_only=True,
        ),
    )


def submit_exact_revision_job(
    spec: LaunchSpec,
    *,
    token: str,
    api: Any | None = None,
    volume_class: Any | None = None,
) -> Any:
    """Submit and accept only a JobInfo that round-trips every visible field."""
    if not token:
        raise RuntimeError("HF_TOKEN is required to submit the CPU probe")
    if api is None or volume_class is None:
        api, volume_class = _load_hub_runtime(token)
    volume = volume_class(
        type=spec.volume.type,
        source=spec.volume.source,
        mount_path=spec.volume.mount_path,
        revision=spec.volume.revision,
        read_only=spec.volume.read_only,
        path=spec.volume.path,
    )
    try:
        job = api.run_job(
            image=spec.image,
            command=list(spec.command),
            env=dict(spec.environment),
            secrets={},
            flavor=spec.flavor,
            labels=dict(spec.labels),
            timeout=spec.timeout,
            namespace=spec.namespace,
            volumes=[volume],
        )
    except Exception as exc:
        raise RuntimeError("HF exact-revision CPU probe submission failed") from exc
    try:
        require_exact_job_contract(job, expected=spec)
    except RuntimeError as exc:
        job_id = getattr(job, "id", None)
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"{exc}; returned JobInfo has no cancellable job id") from exc
        try:
            api.cancel_job(job_id=job_id, namespace=spec.namespace)
        except Exception as cancel_exc:
            raise RuntimeError(f"{exc}; cancellation also failed") from cancel_exc
        raise
    return job


def require_exact_job_contract(job: Any, *, expected: LaunchSpec) -> None:
    """Reject any JobInfo-visible drift from the submitted probe contract."""
    volumes = getattr(job, "volumes", None)
    if not isinstance(volumes, list) or len(volumes) != 1:
        raise RuntimeError("HF JobInfo did not expose exactly one model volume")
    volume = volumes[0]
    observed_volume = {
        "type": getattr(volume, "type", None),
        "source": getattr(volume, "source", None),
        "mount_path": getattr(volume, "mount_path", None),
        "revision": getattr(volume, "revision", None),
        "read_only": getattr(volume, "read_only", None),
        "path": getattr(volume, "path", None),
    }
    if observed_volume != asdict(expected.volume):
        raise RuntimeError(
            "HF JobInfo model volume differs from the exact Carbon revision contract: "
            f"expected={asdict(expected.volume)!r}, observed={observed_volume!r}"
        )

    owner = getattr(job, "owner", None)
    observed = {
        "image": getattr(job, "docker_image", None),
        "space_id": getattr(job, "space_id", None),
        "command": getattr(job, "command", None),
        "arguments": getattr(job, "arguments", None),
        "environment": getattr(job, "environment", None),
        "flavor": getattr(job, "flavor", None),
        "labels": getattr(job, "labels", None),
        "secret_names": _job_secret_names(getattr(job, "secrets", None)),
        "namespace": getattr(owner, "name", None),
    }
    wanted = {
        "image": expected.image,
        "space_id": None,
        "command": list(expected.command),
        "arguments": [],
        "environment": dict(expected.environment),
        "flavor": expected.flavor,
        "labels": dict(expected.labels),
        "secret_names": list(expected.secret_names),
        "namespace": expected.namespace,
    }
    if observed != wanted:
        raise RuntimeError(
            "HF JobInfo differs from the exact observable CPU probe contract: "
            f"expected={wanted!r}, observed={observed!r}"
        )


def _job_secret_names(value: object) -> list[str] | None:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return sorted(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if len(set(value)) != len(value):
            return None
        return sorted(value)
    return None


def _load_hub_runtime(token: str) -> tuple[Any, Any]:
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:  # pragma: no cover - installed by PEP 723.
        raise RuntimeError("launcher requires huggingface-hub==1.8.0") from exc
    if getattr(huggingface_hub, "__version__", None) != HUGGINGFACE_HUB_VERSION:
        raise RuntimeError("launcher must run with huggingface-hub==1.8.0; use `uv run --script`")
    return huggingface_hub.HfApi(token=token), huggingface_hub.Volume


def verify_clean_canonical_checkout(source_commit: str) -> None:
    """Require the launcher itself to come from the exact canonical checkout."""
    if _git_output("rev-parse", "HEAD") != source_commit:
        raise RuntimeError("launcher source commit differs from the local checkout")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("launcher source checkout must be clean")
    if _git_output("remote", "get-url", "origin") != _ORIGIN:
        raise RuntimeError("launcher source origin is not canonical")


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(_REPOSITORY_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("launcher Git state could not be verified") from exc
    return completed.stdout.strip()


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
    parser.add_argument("--source-commit", required=True, type=_exact_commit)
    parser.add_argument("--expected-runtime-hash", required=True, type=_exact_sha256)
    parser.add_argument("--run-attempt", required=True, type=_positive_integer)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the credential-free exact launch contract without contacting Hugging Face",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = build_launch_spec(
            source_commit=args.source_commit,
            expected_runtime_hash=args.expected_runtime_hash,
            run_attempt=args.run_attempt,
        )
        if args.dry_run:
            print(json.dumps(spec.public_payload(), indent=2, sort_keys=True))
            return 0
        verify_clean_canonical_checkout(args.source_commit)
        job = submit_exact_revision_job(spec, token=os.environ.get("HF_TOKEN", ""))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "GENO_LEWM_V03_CARBON_RUNTIME_HASH_PROBE_JOB_ACCEPTED "
        f"{job.id} {args.source_commit} {CARBON_REVISION} {args.expected_runtime_hash}"
    )
    print(getattr(job, "url", ""))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the PEP 723 launcher.
    raise SystemExit(main())
