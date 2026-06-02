# SPDX-License-Identifier: Apache-2.0
"""Bind publication evidence assets before uploading them to a GitHub release."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.publication_assets"
DEFAULT_OUTPUT: Final = "publication_evidence_assets.json"
DEFAULT_ENV_OUTPUT: Final = "publication-evidence-target.env"
REPLAY_ASSETS: Final = (
    ("terminal_transcript", "terminal-demo-transcript.md"),
    ("terminal_demo_manifest", "terminal_demo_manifest.json"),
    ("scores_jsonl", "scores.jsonl"),
    ("receipts_jsonl", "receipts.jsonl"),
    ("runtime_preflight", "runtime_preflight_report.json"),
    ("batch_receipt_report", "batch_receipt_report.json"),
)


@dataclass(frozen=True, slots=True)
class GitHubReleaseTarget:
    """Public GitHub release target for publication evidence assets."""

    repo: str
    tag: str
    url: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationAsset:
    """One publication evidence asset intended for the GitHub release."""

    label: str
    path: str
    destination: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationAssetReport:
    """Machine-readable identity report for final publication evidence uploads."""

    schema_version: str
    generated_by: str
    generated_at: str
    target: GitHubReleaseTarget
    assets: tuple[PublicationAsset, ...]
    upload_command: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "target": self.target.to_dict(),
            "assets": [asset.to_dict() for asset in self.assets],
            "upload_command": list(self.upload_command),
        }


def build_publication_asset_report(
    *,
    demo_url: str,
    hub_release_plan_path: Path,
    release_candidate_path: Path,
    publish_report_path: Path,
    clean_machine_report_path: Path,
    publication_report_path: Path,
    replay_dir: Path,
    asset_manifest_path: Path | None = None,
    generated_at: str | None = None,
) -> PublicationAssetReport:
    """Build an identity manifest for evidence assets uploaded after publication."""
    target = _github_release_target(demo_url)
    manifest_path = asset_manifest_path or Path(DEFAULT_OUTPUT)
    assets = (
        _asset("hub_release_plan", hub_release_plan_path),
        _asset("release_candidate", release_candidate_path),
        _asset("hub_publish", publish_report_path),
        _asset("clean_machine_demo", clean_machine_report_path),
        *(_asset(label, replay_dir / filename) for label, filename in REPLAY_ASSETS),
        _asset("publication_evidence", publication_report_path),
    )
    _reject_duplicate_destinations(assets)
    upload_command = (
        "gh",
        "release",
        "upload",
        target.tag,
        *(asset.path for asset in assets),
        _public_path(manifest_path),
        "--repo",
        target.repo,
        "--clobber",
    )
    return PublicationAssetReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        target=target,
        assets=assets,
        upload_command=upload_command,
    )


def write_publication_asset_report(
    *,
    demo_url: str,
    hub_release_plan_path: Path,
    release_candidate_path: Path,
    publish_report_path: Path,
    clean_machine_report_path: Path,
    publication_report_path: Path,
    replay_dir: Path,
    output: Path,
    env_output: Path | None = None,
    generated_at: str | None = None,
) -> PublicationAssetReport:
    """Write ``publication_evidence_assets.json`` and optional shell target env."""
    report = build_publication_asset_report(
        demo_url=demo_url,
        hub_release_plan_path=hub_release_plan_path,
        release_candidate_path=release_candidate_path,
        publish_report_path=publish_report_path,
        clean_machine_report_path=clean_machine_report_path,
        publication_report_path=publication_report_path,
        replay_dir=replay_dir,
        asset_manifest_path=output,
        generated_at=generated_at,
    )
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    if env_output is not None:
        env_output.write_text(_target_env(report.target), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        write_publication_asset_report(
            demo_url=args.demo_url,
            hub_release_plan_path=args.hub_release_plan,
            release_candidate_path=args.release_candidate,
            publish_report_path=args.publish_report,
            clean_machine_report_path=args.clean_machine_demo_report,
            publication_report_path=args.publication_report,
            replay_dir=args.replay_dir,
            output=args.output,
            env_output=args.env_output,
        )
    except GenoLeWMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    print(f"wrote {args.output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-url", required=True)
    parser.add_argument("--hub-release-plan", type=Path, required=True)
    parser.add_argument("--release-candidate", type=Path, required=True)
    parser.add_argument("--publish-report", type=Path, required=True)
    parser.add_argument("--clean-machine-demo-report", type=Path, required=True)
    parser.add_argument("--publication-report", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--env-output", type=Path, default=Path(DEFAULT_ENV_OUTPUT))
    return parser


def _github_release_target(demo_url: str) -> GitHubReleaseTarget:
    parsed = urlparse(demo_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise InputError(
            "demo_url must be a GitHub release tag URL",
            details={"demo_url": demo_url},
        )
    if len(parts) != 5 or parts[2:4] != ("releases", "tag"):
        raise InputError(
            "demo_url must point to a GitHub release tag",
            details={"demo_url": demo_url},
        )
    owner, repo, _, _, tag = parts
    return GitHubReleaseTarget(repo=f"{owner}/{repo}", tag=unquote(tag), url=demo_url)


def _asset(label: str, path: Path) -> PublicationAsset:
    if not path.is_file():
        raise InputError(
            "publication evidence asset is missing",
            details={"label": label, "path": str(path)},
        )
    return PublicationAsset(
        label=label,
        path=_public_path(path),
        destination=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _public_path(path: Path) -> str:
    if path.is_absolute():
        return path.name
    if ".." in path.parts or not path.parts:
        return path.name
    return path.as_posix()


def _reject_duplicate_destinations(assets: tuple[PublicationAsset, ...]) -> None:
    seen: dict[str, str] = {}
    for asset in assets:
        previous = seen.get(asset.destination)
        if previous is not None:
            raise InputError(
                "publication evidence asset destinations must be unique",
                details={
                    "destination": asset.destination,
                    "first": previous,
                    "duplicate": asset.label,
                },
            )
        seen[asset.destination] = asset.label


def _target_env(target: GitHubReleaseTarget) -> str:
    return f"DEMO_REPO={shlex.quote(target.repo)}\nDEMO_TAG={shlex.quote(target.tag)}\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
