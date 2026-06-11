"""Tests for the desktop signing preflight report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.release.desktop_signing_preflight import (
    LINUX_APPIMAGE_ENV,
    MACOS_CODE_SIGNING_ENV,
    WINDOWS_AZURE_ENV,
    WINDOWS_PFX_ENV,
    build_desktop_signing_preflight_report,
    main,
    write_desktop_signing_preflight_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP = REPO_ROOT / "desktop"


def test_desktop_signing_preflight_dry_run_warns_without_blocking() -> None:
    report = build_desktop_signing_preflight_report(
        desktop_dir=DESKTOP,
        env={},
        generated_at="2026-06-11T12:00:00Z",
    )

    assert report["ok"] is True
    assert report["generated_by"] == "tools.release.desktop_signing_preflight"
    assert report["desktop"]["bundle_targets"] == "all"
    assert "does not sign, notarize" in str(report["claim_boundary"])
    assert {issue["severity"] for issue in report["issues"]} == {"warning"}

    codes = _codes(report)
    assert "desktop.signing.macos.code_signing_missing" in codes
    assert "desktop.signing.macos.notarization_missing" in codes
    assert "desktop.signing.linux.appimage_missing" in codes
    assert "desktop.signing.windows.credentials_missing" in codes


def test_desktop_signing_preflight_require_secrets_blocks_missing_credentials() -> None:
    report = build_desktop_signing_preflight_report(
        desktop_dir=DESKTOP,
        env={},
        require_secrets=True,
    )

    assert report["ok"] is False
    assert {issue["severity"] for issue in report["issues"]} == {"error"}

    codes = _codes(report)
    assert "desktop.signing.macos.code_signing_missing" in codes
    assert "desktop.signing.macos.notarization_missing" in codes
    assert "desktop.signing.linux.appimage_missing" in codes
    assert "desktop.signing.linux.force_sign" in codes
    assert "desktop.signing.windows.credentials_missing" in codes


def test_desktop_signing_preflight_accepts_complete_synthetic_env() -> None:
    report = build_desktop_signing_preflight_report(
        desktop_dir=DESKTOP,
        env=_complete_env(),
        require_secrets=True,
    )

    assert report["ok"] is True
    assert report["issues"] == []
    assert report["platforms"]["macos"]["complete"] is True
    assert report["platforms"]["linux"]["complete"] is True
    assert report["platforms"]["windows"]["complete"] is True


def test_desktop_signing_preflight_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "desktop_signing_preflight_report.json"
    report = write_desktop_signing_preflight_report(
        desktop_dir=DESKTOP,
        output=output,
        env=_complete_env(),
        require_secrets=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == report
    assert payload["ok"] is True


def test_desktop_signing_preflight_main_returns_two_for_missing_release_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in _all_env_names():
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "report.json"

    rc = main(["--desktop-dir", str(DESKTOP), "--output", str(output), "--require-secrets"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "wrote" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is False


def _complete_env() -> dict[str, str]:
    env = dict.fromkeys(_all_env_names(), "value")
    env["SIGN"] = "1"
    env["APPIMAGETOOL_FORCE_SIGN"] = "1"
    return env


def _all_env_names() -> tuple[str, ...]:
    return (
        *MACOS_CODE_SIGNING_ENV,
        "APPLE_ID",
        "APPLE_PASSWORD",
        "APPLE_TEAM_ID",
        *LINUX_APPIMAGE_ENV,
        *WINDOWS_PFX_ENV,
        *WINDOWS_AZURE_ENV,
    )


def _codes(report: dict[str, object]) -> set[str]:
    issues = report["issues"]
    assert isinstance(issues, list)
    return {str(issue["code"]) for issue in issues if isinstance(issue, dict)}
