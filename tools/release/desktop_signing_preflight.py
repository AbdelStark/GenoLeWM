# SPDX-License-Identifier: Apache-2.0
"""Preflight the desktop release-signing contract without signing artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.desktop_signing_preflight"
DEFAULT_OUTPUT: Final = "desktop_signing_preflight_report.json"
CLAIM_BOUNDARY: Final = (
    "This report checks desktop bundle configuration and signing/notarization "
    "credential presence only. It does not sign, notarize, staple, publish, "
    "verify, or establish deployment readiness for any desktop binary."
)

Severity = Literal["error", "warning"]
Platform = Literal["macos", "linux", "windows"]

MACOS_CODE_SIGNING_ENV: Final = (
    "APPLE_CERTIFICATE",
    "APPLE_CERTIFICATE_PASSWORD",
    "KEYCHAIN_PASSWORD",
)
MACOS_NOTARIZATION_GROUPS: Final = (
    ("APPLE_API_ISSUER", "APPLE_API_KEY", "APPLE_API_KEY_PATH"),
    ("APPLE_ID", "APPLE_PASSWORD", "APPLE_TEAM_ID"),
)
LINUX_APPIMAGE_ENV: Final = (
    "SIGN",
    "SIGN_KEY",
    "APPIMAGETOOL_SIGN_PASSPHRASE",
    "APPIMAGETOOL_FORCE_SIGN",
)
WINDOWS_PFX_ENV: Final = ("WINDOWS_CERTIFICATE", "WINDOWS_CERTIFICATE_PASSWORD")
WINDOWS_AZURE_ENV: Final = ("AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID")


@dataclass(frozen=True, slots=True)
class DesktopSigningIssue:
    """One desktop signing preflight issue."""

    severity: Severity
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


def build_desktop_signing_preflight_report(
    *,
    desktop_dir: Path,
    env: Mapping[str, str] | None = None,
    require_secrets: bool = False,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build a machine-readable desktop signing readiness report."""
    env = env or os.environ
    issues: list[DesktopSigningIssue] = []
    config_path = desktop_dir / "src-tauri" / "tauri.conf.json"
    config = _load_config(config_path, issues)
    desktop = _desktop_report(desktop_dir, config_path, config, issues)
    platforms = {
        "macos": _macos_report(env, require_secrets, issues),
        "linux": _linux_report(env, require_secrets, issues),
        "windows": _windows_report(env, require_secrets, issues),
    }
    ok = not any(issue.severity == "error" for issue in issues)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "ok": ok,
        "requirements": {
            "secrets_required": require_secrets,
            "all_platforms": ["macos", "linux", "windows"],
        },
        "desktop": desktop,
        "platforms": platforms,
        "issues": [issue.to_dict() for issue in issues],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_desktop_signing_preflight_report(
    *,
    desktop_dir: Path,
    output: Path,
    env: Mapping[str, str] | None = None,
    require_secrets: bool = False,
) -> dict[str, object]:
    """Build and write the desktop signing preflight report."""
    report = build_desktop_signing_preflight_report(
        desktop_dir=desktop_dir,
        env=env,
        require_secrets=require_secrets,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    report = write_desktop_signing_preflight_report(
        desktop_dir=args.desktop_dir,
        output=args.output,
        require_secrets=args.require_secrets,
    )
    sys.stdout.write(f"wrote {args.output}\n")
    return 0 if report["ok"] else 2


def _load_config(path: Path, issues: list[DesktopSigningIssue]) -> dict[str, object]:
    if not path.is_file():
        _issue(issues, "error", "desktop.config.missing", path, "Tauri config is missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _issue(
            issues,
            "error",
            "desktop.config.invalid_json",
            path,
            f"Tauri config is invalid JSON at line {exc.lineno}, column {exc.colno}",
        )
        return {}
    if not isinstance(payload, dict):
        _issue(issues, "error", "desktop.config.not_object", path, "Tauri config must be an object")
        return {}
    return payload


def _desktop_report(
    desktop_dir: Path,
    config_path: Path,
    config: Mapping[str, object],
    issues: list[DesktopSigningIssue],
) -> dict[str, object]:
    product_name = _text(config.get("productName"))
    version = _text(config.get("version"))
    identifier = _text(config.get("identifier"))
    bundle = _dict(config.get("bundle"))
    targets = bundle.get("targets")
    icons = _text_list(bundle.get("icon"))

    if product_name != "GenoLeWM":
        _issue(issues, "error", "desktop.product_name", config_path, "productName must be GenoLeWM")
    if identifier != "org.genolewm.desktop":
        _issue(
            issues,
            "error",
            "desktop.identifier",
            config_path,
            "identifier must be org.genolewm.desktop",
        )
    if not version:
        _issue(issues, "error", "desktop.version", config_path, "desktop version is required")
    if bundle.get("active") is not True:
        _issue(
            issues, "error", "desktop.bundle.inactive", config_path, "bundle.active must be true"
        )
    if targets not in ("all", ["all"]):
        _issue(
            issues,
            "warning",
            "desktop.bundle.targets",
            config_path,
            "bundle.targets should cover all desktop platforms before signed releases",
        )

    icon_reports = []
    for icon in icons:
        path = desktop_dir / "src-tauri" / icon
        exists = path.is_file()
        if not exists:
            _issue(issues, "error", "desktop.bundle.icon_missing", path, "bundle icon is missing")
        icon_reports.append({"path": f"desktop/src-tauri/{icon}", "exists": exists})
    if not icon_reports:
        _issue(issues, "error", "desktop.bundle.icons", config_path, "bundle icons are required")

    return {
        "path": "desktop",
        "tauri_config": "desktop/src-tauri/tauri.conf.json",
        "product_name": product_name,
        "version": version,
        "identifier": identifier,
        "bundle_active": bundle.get("active") is True,
        "bundle_targets": targets,
        "icons": icon_reports,
    }


def _macos_report(
    env: Mapping[str, str],
    require_secrets: bool,
    issues: list[DesktopSigningIssue],
) -> dict[str, object]:
    signing = _env_group_report(
        platform="macos",
        group_name="code_signing",
        names=MACOS_CODE_SIGNING_ENV,
        env=env,
        require_secrets=require_secrets,
        issues=issues,
    )
    notarization_groups = [
        _env_group_status("app_store_connect_api", MACOS_NOTARIZATION_GROUPS[0], env),
        _env_group_status("apple_id", MACOS_NOTARIZATION_GROUPS[1], env),
    ]
    if require_secrets and not any(group["complete"] for group in notarization_groups):
        _issue(
            issues,
            "error",
            "desktop.signing.macos.notarization_missing",
            "env",
            "macOS notarization requires either App Store Connect API or Apple ID credentials",
        )
    elif not any(group["complete"] for group in notarization_groups):
        _issue(
            issues,
            "warning",
            "desktop.signing.macos.notarization_missing",
            "env",
            "macOS notarization credentials are not configured",
        )
    return {
        "code_signing": signing,
        "notarization": notarization_groups,
        "complete": signing["complete"] and any(group["complete"] for group in notarization_groups),
    }


def _linux_report(
    env: Mapping[str, str],
    require_secrets: bool,
    issues: list[DesktopSigningIssue],
) -> dict[str, object]:
    appimage = _env_group_report(
        platform="linux",
        group_name="appimage",
        names=LINUX_APPIMAGE_ENV,
        env=env,
        require_secrets=require_secrets,
        issues=issues,
    )
    force_sign = env.get("APPIMAGETOOL_FORCE_SIGN") == "1"
    if env.get("SIGN") and env.get("SIGN") != "1":
        _issue(
            issues,
            "error" if require_secrets else "warning",
            "desktop.signing.linux.sign_flag",
            "env",
            "SIGN must be set to 1 for signed AppImage builds",
        )
    if require_secrets and not force_sign:
        _issue(
            issues,
            "error",
            "desktop.signing.linux.force_sign",
            "env",
            "APPIMAGETOOL_FORCE_SIGN must be 1 so AppImage signing failures fail CI",
        )
    return {
        "appimage": appimage,
        "force_sign": force_sign,
        "complete": appimage["complete"] and force_sign,
    }


def _windows_report(
    env: Mapping[str, str],
    require_secrets: bool,
    issues: list[DesktopSigningIssue],
) -> dict[str, object]:
    pfx = _env_group_status("pfx_certificate", WINDOWS_PFX_ENV, env)
    azure = _env_group_status("azure_trusted_signing", WINDOWS_AZURE_ENV, env)
    if require_secrets and not pfx["complete"] and not azure["complete"]:
        _issue(
            issues,
            "error",
            "desktop.signing.windows.credentials_missing",
            "env",
            "Windows signing requires either PFX certificate or Azure signing credentials",
        )
    elif not pfx["complete"] and not azure["complete"]:
        _issue(
            issues,
            "warning",
            "desktop.signing.windows.credentials_missing",
            "env",
            "Windows signing credentials are not configured",
        )
    return {"credential_sets": [pfx, azure], "complete": pfx["complete"] or azure["complete"]}


def _env_group_report(
    *,
    platform: Platform,
    group_name: str,
    names: Sequence[str],
    env: Mapping[str, str],
    require_secrets: bool,
    issues: list[DesktopSigningIssue],
) -> dict[str, object]:
    group = _env_group_status(group_name, names, env)
    if group["complete"]:
        return group
    severity: Severity = "error" if require_secrets else "warning"
    missing = ", ".join(cast(list[str], group["missing"]))
    _issue(
        issues,
        severity,
        f"desktop.signing.{platform}.{group_name}_missing",
        "env",
        f"{platform} {group_name} credentials are incomplete: missing {missing}",
    )
    return group


def _env_group_status(
    group_name: str,
    names: Sequence[str],
    env: Mapping[str, str],
) -> dict[str, object]:
    present = [name for name in names if _env_present(env, name)]
    missing = [name for name in names if name not in present]
    return {
        "name": group_name,
        "required_env": list(names),
        "present": present,
        "missing": missing,
        "complete": not missing,
    }


def _env_present(env: Mapping[str, str], name: str) -> bool:
    return bool(env.get(name, "").strip())


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _issue(
    issues: list[DesktopSigningIssue],
    severity: Severity,
    code: str,
    path: Path | str,
    message: str,
) -> None:
    issues.append(
        DesktopSigningIssue(
            severity=severity,
            code=code,
            path=_display_path(path),
            message=message,
        )
    )


def _display_path(path: Path | str) -> str:
    if isinstance(path, str):
        return path
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools.release.desktop_signing_preflight",
        description="Check desktop signing prerequisites without signing release artifacts.",
    )
    parser.add_argument("--desktop-dir", type=Path, default=Path("desktop"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--require-secrets",
        action="store_true",
        help="Treat missing signing/notarization credentials as release-blocking errors.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
