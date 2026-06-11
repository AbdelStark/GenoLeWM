"""Tests for the BRCA2 saturation tutorial fixture notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from examples.brca2_saturation_fixture import (
    BRCA2_EXON_FIXTURE,
    enumerate_fixture_saturation,
    render_text_heatmap,
    summarize_rows,
)

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "02_score_brca2_saturation.ipynb"


def _code_cells() -> list[dict[str, Any]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_fixture_saturation_enumerates_three_snvs_per_position() -> None:
    rows = enumerate_fixture_saturation()
    summary = summarize_rows(rows)

    assert len(rows) == 3 * len(BRCA2_EXON_FIXTURE)
    assert summary["positions"] == len(BRCA2_EXON_FIXTURE)
    assert summary["snvs"] == len(rows)
    assert summary["fixture_spearman"] < -0.95
    assert "ref " in render_text_heatmap(rows)


def test_brca2_saturation_notebook_is_committed_executed() -> None:
    code_cells = _code_cells()

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert any(
        output.get("output_type") == "stream"
        and "fixture rows=72 positions=24" in "".join(output.get("text", []))
        for cell in code_cells
        for output in cell["outputs"]
    )


def test_brca2_saturation_notebook_cells_execute(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, Any] = {"__name__": "__notebook__"}

    for index, cell in enumerate(_code_cells(), start=1):
        source = "".join(cell["source"])
        exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)

    captured = capsys.readouterr()
    assert "fixture rows=72 positions=24" in captured.out
    assert "heatmap cells=72" in captured.out
    assert "Findlay comparison: not run in this fixture notebook" in captured.out
