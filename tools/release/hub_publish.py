# SPDX-License-Identifier: Apache-2.0
"""Publish a verified model/dataset/demo release candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse

from geno_lewm.errors import GenoLeWMError, InputError, ResourceError, exit_code_for
from tools.release.hub_release import HubReleasePlan, UploadFile, build_hub_release_plan
from tools.release.release_candidate import (
    DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS,
    ReleaseCandidateReport,
    write_release_candidate_report,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.hub_publish"
DEFAULT_PLAN_OUTPUT: Final = "hub_release_plan.json"
DEFAULT_CANDIDATE_OUTPUT: Final = "release_candidate_report.json"
DEFAULT_PUBLISH_OUTPUT: Final = "hub_publish_report.json"

CommandRunner = Callable[[Sequence[str]], None]


@dataclass(frozen=True, slots=True)
class PublishCommand:
    """One credentialed publication command executed by the helper."""

    name: str
    argv: tuple[str, ...]
    public_argv: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "argv": list(self.public_argv or self.argv)}


@dataclass(frozen=True, slots=True)
class HubPublishReport:
    """Machine-readable record of a credentialed publication attempt."""

    schema_version: str
    generated_by: str
    generated_at: str
    plan: HubReleasePlan
    commands: tuple[PublishCommand, ...]
    final_candidate_ready: bool
    final_candidate_report: ReleaseCandidateReport

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "plan": self.plan.to_dict(),
            "commands": [command.to_dict() for command in self.commands],
            "final_candidate_ready": self.final_candidate_ready,
            "final_candidate_report": self.final_candidate_report.to_dict(),
        }


def publish_hub_release(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    repo_id: str,
    dataset_url: str,
    demo_url: str,
    commit_sha: str,
    paper_path: Path | None = None,
    paper_url: str | None = None,
    plan_output: Path | None = None,
    candidate_output: Path | None = None,
    publish_output: Path | None = None,
    public_link_timeout_seconds: float = DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS,
    command_runner: CommandRunner | None = None,
    environ: Mapping[str, str] | None = None,
    generated_at: str | None = None,
) -> HubPublishReport:
    """Publish a verified release candidate, then re-check public readiness."""
    generated_at = generated_at or _utc_now()
    plan = build_hub_release_plan(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        repo_id=repo_id,
        dataset_url=dataset_url,
        demo_url=demo_url,
        commit_sha=commit_sha,
        paper_path=paper_path,
        paper_url=paper_url,
        generated_at=generated_at,
    )
    if plan_output is not None:
        _write_json(plan_output, plan.to_dict())

    commands = _publish_commands(
        plan=plan,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        paper_path=paper_path,
        demo_files=plan.demo_files,
    )
    _validate_publish_environment(commands, os.environ if environ is None else environ)
    runner = command_runner or _run_command
    for command in commands:
        runner(command.argv)

    candidate_path = candidate_output or Path(DEFAULT_CANDIDATE_OUTPUT)
    final_candidate = write_release_candidate_report(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        paper_path=paper_path,
        repo_id=repo_id,
        dataset_url=dataset_url,
        demo_url=demo_url,
        paper_url=paper_url,
        commit_sha=commit_sha,
        output=candidate_path,
        generated_at=generated_at,
        public_link_timeout_seconds=public_link_timeout_seconds,
    )
    report = HubPublishReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at,
        plan=plan,
        commands=commands,
        final_candidate_ready=final_candidate.ready,
        final_candidate_report=final_candidate,
    )
    if publish_output is not None:
        _write_json(publish_output, report.to_dict())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = publish_hub_release(
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            demo_dir=args.demo_dir,
            paper_path=args.paper_path,
            repo_id=args.repo_id,
            dataset_url=args.dataset_url,
            demo_url=args.demo_url,
            paper_url=args.paper_url,
            commit_sha=args.commit_sha,
            plan_output=args.plan_output,
            candidate_output=args.candidate_output,
            publish_output=args.publish_output,
            public_link_timeout_seconds=args.public_link_timeout,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.publish_output}\n")
    return 0 if report.final_candidate_ready else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a verified Hugging Face/GitHub model release candidate, "
            "then regenerate release_candidate_report.json."
        ),
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--paper-path", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--demo-url", required=True)
    parser.add_argument("--paper-url")
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--plan-output", type=Path, default=Path(DEFAULT_PLAN_OUTPUT))
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path(DEFAULT_CANDIDATE_OUTPUT),
    )
    parser.add_argument("--publish-output", type=Path, default=Path(DEFAULT_PUBLISH_OUTPUT))
    parser.add_argument(
        "--public-link-timeout",
        type=float,
        default=DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS,
        help="Timeout in seconds for each final public URL/artifact listing check.",
    )
    return parser


def _publish_commands(
    *,
    plan: HubReleasePlan,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    paper_path: Path | None,
    demo_files: tuple[UploadFile, ...],
) -> tuple[PublishCommand, ...]:
    dataset_repo_id = _dataset_repo_id_from_url(plan.dataset_url)
    if dataset_repo_id is None:
        raise InputError(
            "dataset_url must be a Hugging Face dataset URL for credentialed publishing",
            details={"dataset_url": plan.dataset_url},
        )
    demo_target = _github_release_from_url(plan.demo_url)
    if demo_target is None:
        raise InputError(
            "demo_url must be a GitHub release tag URL for credentialed publishing",
            details={"demo_url": plan.demo_url},
        )
    demo_repo, demo_tag = demo_target
    paper_command = _paper_publish_command(
        plan=plan,
        paper_path=paper_path,
    )
    return (
        *(
            _hf_upload_command(
                name="model",
                repo_id=plan.repo_id,
                repo_type="model",
                file=file,
                source_root=model_dir,
                public_source=_public_command_source("model", file),
                commit_message=f"Release {plan.release_id} ({plan.model_id})",
            )
            for file in plan.files
        ),
        *(
            _hf_upload_command(
                name="dataset",
                repo_id=dataset_repo_id,
                repo_type="dataset",
                file=file,
                source_root=dataset_dir,
                public_source=_public_command_source("dataset", file),
                commit_message=f"Release dataset for {plan.release_id} ({plan.model_id})",
            )
            for file in plan.dataset_files
        ),
        PublishCommand(
            name="demo",
            argv=(
                "gh",
                "release",
                "upload",
                demo_tag,
                *(_resolve_plan_source(demo_dir, file.source) for file in demo_files),
                "--repo",
                demo_repo,
                "--clobber",
            ),
            public_argv=(
                "gh",
                "release",
                "upload",
                demo_tag,
                *(_public_command_source("demo", file) for file in demo_files),
                "--repo",
                demo_repo,
                "--clobber",
            ),
        ),
        *(() if paper_command is None else (paper_command,)),
    )


def _hf_upload_command(
    *,
    name: str,
    repo_id: str,
    repo_type: str,
    file: UploadFile,
    source_root: Path,
    public_source: str,
    commit_message: str,
) -> PublishCommand:
    return PublishCommand(
        name=name,
        argv=(
            "huggingface-cli",
            "upload",
            repo_id,
            _resolve_plan_source(source_root, file.source),
            file.destination,
            "--repo-type",
            repo_type,
            "--commit-message",
            commit_message,
        ),
        public_argv=(
            "huggingface-cli",
            "upload",
            repo_id,
            public_source,
            file.destination,
            "--repo-type",
            repo_type,
            "--commit-message",
            commit_message,
        ),
    )


def _resolve_plan_source(root: Path, source: str) -> str:
    path = Path(source)
    if path.is_absolute():
        return str(path)
    return str(root / path)


def _public_command_source(prefix: str, file: UploadFile) -> str:
    path = Path(file.source)
    if path.is_absolute():
        return f"{prefix}/{path.name}"
    return f"{prefix}/{path.as_posix()}"


def _paper_publish_command(
    *,
    plan: HubReleasePlan,
    paper_path: Path | None,
) -> PublishCommand | None:
    if plan.paper_file is None:
        return None
    if paper_path is None:
        raise InputError("paper_path is required when the Hub plan includes a paper file")
    target = _github_release_download_from_url(plan.paper_url or "")
    if target is None:
        raise InputError(
            "paper_url must be a GitHub release download URL for credentialed paper publication",
            details={"paper_url": plan.paper_url},
        )
    paper_repo, paper_tag, asset_name = target
    if asset_name != plan.paper_file.destination:
        raise InputError(
            "paper_url asset name must match the verified paper file destination",
            details={
                "paper_url": plan.paper_url,
                "asset": asset_name,
                "destination": plan.paper_file.destination,
            },
        )
    return PublishCommand(
        name="paper",
        argv=(
            "gh",
            "release",
            "upload",
            paper_tag,
            str(paper_path),
            "--repo",
            paper_repo,
            "--clobber",
        ),
        public_argv=(
            "gh",
            "release",
            "upload",
            paper_tag,
            plan.paper_file.source,
            "--repo",
            paper_repo,
            "--clobber",
        ),
    )


def _validate_publish_environment(
    commands: tuple[PublishCommand, ...],
    environ: Mapping[str, str],
) -> None:
    needs_hf = any(command.argv[0] == "huggingface-cli" for command in commands)
    if needs_hf and not environ.get("HF_TOKEN"):
        raise InputError("HF_TOKEN is required for Hugging Face publishing")
    needs_gh = any(command.argv[0] == "gh" for command in commands)
    if needs_gh and not (environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN")):
        raise InputError("GH_TOKEN or GITHUB_TOKEN is required for GitHub release publishing")


def _run_command(argv: Sequence[str]) -> None:
    try:
        subprocess.run(list(argv), check=True)
    except FileNotFoundError as exc:
        raise ResourceError(
            "release publication command was not found",
            details={"command": argv[0]},
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ResourceError(
            "release publication command failed",
            details={"command": list(argv), "returncode": exc.returncode},
        ) from exc


def _dataset_repo_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "huggingface.co" or len(parts) < 3 or parts[0] != "datasets":
        return None
    return f"{parts[1]}/{parts[2]}"


def _github_release_from_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2:4] != ("releases", "tag"):
        return None
    owner, repo, _releases, _tag, tag = parts[:5]
    return f"{owner}/{repo}", tag


def _github_release_download_from_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2] != "releases":
        return None
    if parts[3] != "download":
        return None
    owner, repo, _releases, _download, tag, *asset_parts = parts
    if not asset_parts:
        return None
    return f"{owner}/{repo}", tag, unquote("/".join(asset_parts))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
