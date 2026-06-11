# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixture-backed scoring tutorial notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    ROOT / "examples" / "01_score_single_variant.ipynb",
    ROOT / "examples" / "03_score_vcf.ipynb",
)


def _notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(path: Path) -> list[dict[str, Any]]:
    return [cell for cell in _notebook(path)["cells"] if cell["cell_type"] == "code"]


def _joined_outputs(cells: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for cell in cells:
        for output in cell["outputs"]:
            text = output.get("text", "")
            if isinstance(text, list):
                chunks.extend(str(part) for part in text)
            else:
                chunks.append(str(text))
    return "".join(chunks)


def test_scoring_notebooks_are_committed_executed() -> None:
    for notebook in NOTEBOOKS:
        code_cells = _code_cells(notebook)
        output_text = _joined_outputs(code_cells)

        assert code_cells
        assert all(cell["execution_count"] is not None for cell in code_cells)
        assert "exit code: 0" in output_text
        assert "verifier final line: ok" in output_text


def test_scoring_notebook_cells_execute(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(ROOT)

    for notebook in NOTEBOOKS:
        namespace: dict[str, Any] = {"__name__": "__notebook__"}
        for index, cell in enumerate(_code_cells(notebook), start=1):
            source = "".join(cell["source"])
            exec(compile(source, f"{notebook}:cell-{index}", "exec"), namespace)

    captured = capsys.readouterr()
    assert '"receipt_written": true' in captured.out
    assert '"score_rows": 1' in captured.out
    assert "first receipt validation exit code: 0" in captured.out
