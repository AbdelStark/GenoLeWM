"""Static checks for the RFC-0019 desktop scaffold."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP = REPO_ROOT / "desktop"


class SafetyBannerParser(HTMLParser):
    """Extract the desktop shell's safety banner without adding parser deps."""

    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, str] | None = None
        self.child_tags: list[str] = []
        self.text_parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = {name: value or "" for name, value in attrs}
        if self._depth:
            self.child_tags.append(tag)
            self._depth += 1
            return

        if tag == "section" and "safety-banner" in attrs_by_name.get("class", "").split():
            self.attrs = attrs_by_name
            self._depth = 1

    def handle_data(self, data: str) -> None:
        if self._depth:
            self.text_parts.append(data)

    def handle_endtag(self, _tag: str) -> None:
        if self._depth:
            self._depth -= 1

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.text_parts).split())


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


def test_desktop_safety_banner_is_persistent_and_non_dismissible() -> None:
    parser = SafetyBannerParser()
    parser.feed((DESKTOP / "index.html").read_text(encoding="utf-8"))

    assert parser.attrs is not None
    assert parser.attrs["data-persistent-safety-banner"] == "true"
    assert parser.attrs["role"] == "status"
    assert parser.attrs["aria-live"] == "polite"
    assert parser.attrs["aria-label"] == "Permanent safety notice"
    assert "hidden" not in parser.attrs
    assert "aria-hidden" not in parser.attrs
    assert "button" not in parser.child_tags
    assert "research tool, not a clinical diagnostic" in parser.text
    assert "qualified genetic counselor" in parser.text


def test_desktop_safety_banner_stays_visible_while_scrolling() -> None:
    styles = (DESKTOP / "src" / "styles.css").read_text(encoding="utf-8")
    start = styles.index(".safety-banner {")
    end = styles.index("}", start)
    banner_rules = {
        name.strip(): value.strip()
        for declaration in styles[start:end].split("{", maxsplit=1)[1].split(";")
        if ":" in declaration
        for name, value in [declaration.split(":", maxsplit=1)]
    }

    assert banner_rules["position"] == "sticky"
    assert banner_rules["top"] == "0"
    assert banner_rules["z-index"] == "10"


def test_desktop_rust_host_registers_pyo3_runtime_probe() -> None:
    cargo_toml = (DESKTOP / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    lib_rs = (DESKTOP / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
    runtime_rs = (DESKTOP / "src-tauri" / "src" / "runtime.rs").read_text(encoding="utf-8")

    assert 'pyo3 = { version = "0.28.3", features = ["auto-initialize"] }' in cargo_toml
    assert "tauri_plugin_http::init()" in lib_rs
    assert "runtime_probe" in lib_rs
    assert 'py.import("geno_lewm.deploy")' in runtime_rs
    assert 'module.getattr("GenoLeWMRuntime")' in runtime_rs
