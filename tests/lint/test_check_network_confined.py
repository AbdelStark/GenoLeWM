"""Unit tests for ``tools.lint.check_network_confined``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.lint import check_network_confined as linter


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "src",
    [
        "import urllib\n",
        "import urllib.request\n",
        "import httpx\n",
        "import requests\n",
        "import aiohttp\n",
        "import socket\n",
        "import ssl\n",
        "from urllib.request import urlopen\n",
        "from httpx import AsyncClient\n",
        "from requests.adapters import HTTPAdapter\n",
    ],
)
def test_forbidden_imports_flagged(tmp_path: Path, src: str) -> None:
    f = _write(tmp_path, "bad.py", src)
    v = linter.check_file(f)
    assert len(v) == 1
    assert "network_confined" not in v[0].message  # message itself doesn't repeat the check name
    assert "runtime contract" in v[0].message


def test_unrelated_imports_pass(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "ok.py",
        "import json\nimport hashlib\nfrom pathlib import Path\n",
    )
    assert linter.check_file(f) == []


def test_relative_imports_not_flagged(tmp_path: Path) -> None:
    # ``from . import x`` carries a non-zero level; never refers to
    # the forbidden top-level set.
    f = _write(tmp_path, "rel.py", "from . import sibling\n")
    assert linter.check_file(f) == []


def test_allowlisted_path_is_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Synthesise a deploy/runtime.py-like layout and confirm allowlist
    # behaviour by pointing PACKAGE_DIR at the synthetic root.
    pkg = tmp_path / "pkg"
    deploy = pkg / "deploy"
    deploy.mkdir(parents=True)
    f = deploy / "runtime.py"
    f.write_text("import urllib\nimport requests\n", encoding="utf-8")

    monkeypatch.setattr(linter, "PACKAGE_DIR", pkg)
    assert linter.check_file(f) == []


def test_real_package_passes() -> None:
    # The shipped package today imports no network-capable modules.
    assert linter.main([str(linter.PACKAGE_DIR)]) == 0


def test_main_returns_one_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, "bad.py", "import httpx\n")
    assert linter.main([str(tmp_path)]) == 1
