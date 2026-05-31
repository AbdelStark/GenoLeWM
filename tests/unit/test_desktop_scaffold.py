"""Static checks for the RFC-0019 desktop scaffold."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP = REPO_ROOT / "desktop"


def test_desktop_scaffold_has_tauri_layout() -> None:
    expected = [
        DESKTOP / "package.json",
        DESKTOP / "pnpm-lock.yaml",
        DESKTOP / "src" / "main.ts",
        DESKTOP / "src-tauri" / "Cargo.toml",
        DESKTOP / "src-tauri" / "src" / "main.rs",
        DESKTOP / "src-tauri" / "src" / "lib.rs",
        DESKTOP / "src-tauri" / "src" / "runtime.rs",
        DESKTOP / "src-tauri" / "tauri.conf.json",
    ]
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in expected if not path.is_file()]
    assert not missing


def test_desktop_network_policy_is_default_deny() -> None:
    capability = json.loads(
        (DESKTOP / "src-tauri" / "capabilities" / "default.json").read_text(encoding="utf-8")
    )
    permissions = capability["permissions"]
    http_permission = next(item for item in permissions if isinstance(item, dict))

    assert http_permission["identifier"] == "http:default"
    assert http_permission["allow"] == [
        {"url": "https://huggingface.co/**"},
        {"url": "https://*.huggingface.co/**"},
        {"url": "https://ftp.1000genomes.ebi.ac.uk/**"},
    ]
    assert "deny" not in http_permission


def test_desktop_tauri_config_csp_mirrors_allowed_hosts() -> None:
    config = json.loads((DESKTOP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    csp = config["app"]["security"]["csp"]

    assert config["productName"] == "GenoLeWM"
    assert config["identifier"] == "org.genolewm.desktop"
    assert "default-src 'self'" in csp
    assert "https://huggingface.co" in csp
    assert "https://*.huggingface.co" in csp
    assert "https://ftp.1000genomes.ebi.ac.uk" in csp
    assert "http://*" not in csp


def test_desktop_rust_host_registers_pyo3_runtime_probe() -> None:
    cargo_toml = (DESKTOP / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    lib_rs = (DESKTOP / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
    runtime_rs = (DESKTOP / "src-tauri" / "src" / "runtime.rs").read_text(encoding="utf-8")

    assert 'pyo3 = { version = "0.28.3", features = ["auto-initialize"] }' in cargo_toml
    assert "tauri_plugin_http::init()" in lib_rs
    assert "runtime_probe" in lib_rs
    assert 'py.import("geno_lewm.deploy")' in runtime_rs
    assert 'module.getattr("GenoLeWMRuntime")' in runtime_rs
