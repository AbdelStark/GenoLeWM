# SPDX-License-Identifier: Apache-2.0
"""Tests for the Carbon-windows export tool and corpus revision pinning."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from geno_lewm.data.corpus import CarbonCorpusConfig, CarbonRecord, load_hf_carbon_records
from tools.data import carbon_windows


def _records() -> list[CarbonRecord]:
    return [
        CarbonRecord(record_id="r1", source="mrna", sequence="ACGT" * 200),
        CarbonRecord(record_id="r2", source="gtdb", sequence="ACGT" * 200),
    ]


def test_export_carbon_windows_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, CarbonCorpusConfig] = {}

    def fake_load(config: CarbonCorpusConfig) -> object:
        captured["config"] = config
        return iter(_records())

    monkeypatch.setattr(carbon_windows, "load_hf_carbon_records", fake_load)
    out = tmp_path / "carbon" / "source-mix-windows.jsonl"

    summary = carbon_windows.export_carbon_windows(
        output=out, revision="cb4c13a", window_bp=128, subset_fraction=1.0
    )

    assert out.is_file()
    assert captured["config"].revision == "cb4c13a"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert {"record_id", "source", "start_bp", "end_bp", "sequence"} <= set(row)
        assert len(row["sequence"]) == 128
    assert summary["windows"] == len(rows)
    assert summary["revision"] == "cb4c13a"
    assert summary["records"] == 2
    assert set(summary["sources"]) <= {"mrna", "gtdb"}


def test_export_carbon_windows_respects_max_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(carbon_windows, "load_hf_carbon_records", lambda config: iter(_records()))
    out = tmp_path / "w.jsonl"
    summary = carbon_windows.export_carbon_windows(
        output=out, window_bp=128, subset_fraction=1.0, max_windows=1
    )
    assert summary["windows"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_carbon_windows_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(carbon_windows, "load_hf_carbon_records", lambda config: iter(_records()))
    out = tmp_path / "w.jsonl"
    rc = carbon_windows.main(
        [
            "--output",
            str(out),
            "--revision",
            "rev1",
            "--window-bp",
            "128",
            "--subset-fraction",
            "1.0",
        ]
    )
    assert rc == 0
    assert out.is_file()
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["revision"] == "rev1"


def test_load_hf_carbon_records_passes_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake = types.ModuleType("datasets")

    def load_dataset(*args: object, **kwargs: object) -> list[object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return []

    fake.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    list(load_hf_carbon_records(CarbonCorpusConfig(revision="cb4c13a", subset_fraction=1.0)))

    assert captured["kwargs"]["revision"] == "cb4c13a"
