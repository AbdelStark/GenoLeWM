# SPDX-License-Identifier: Apache-2.0
"""Tests for the upstream VCF downloader (tools.data.download)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.data import download
from tools.data.download import DownloadError, download_file, download_manifest


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(
        download.urllib_request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse(data),
    )


def test_download_file_writes_and_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"##fileformat=VCFv4.2\n" * 500
    _patch_urlopen(monkeypatch, payload)
    out = tmp_path / "clinvar" / "clinvar.vcf.gz"

    report = download_file(
        "https://ftp.example.test/clinvar.vcf.gz",
        out,
        acknowledge_source_terms=True,
    )

    assert out.read_bytes() == payload
    assert report["size_bytes"] == len(payload)
    assert report["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert report["source_terms_acknowledged"] is True
    assert report["source_terms"] == "NCBI ClinVar source terms"
    assert not out.with_name(out.name + ".part").exists()


def test_download_file_requires_source_terms_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_urlopen(monkeypatch, b"payload")

    with pytest.raises(DownloadError, match="source terms must be acknowledged"):
        download_file("https://gnomad.broadinstitute.org/downloads", tmp_path / "gnomad.vcf.gz")

    assert not (tmp_path / "gnomad.vcf.gz").exists()


def test_download_file_verifies_expected_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"data"
    _patch_urlopen(monkeypatch, payload)
    good = "sha256:" + hashlib.sha256(payload).hexdigest()
    out = tmp_path / "ok.vcf.gz"
    assert (
        download_file(
            "https://x.test/a",
            out,
            expected_sha256=good,
            acknowledge_source_terms=True,
        )["sha256"]
        == good
    )

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        download_file(
            "https://x.test/b",
            tmp_path / "bad.vcf.gz",
            expected_sha256="sha256:" + "0" * 64,
            acknowledge_source_terms=True,
        )
    assert not (tmp_path / "bad.vcf.gz").exists()


def test_download_file_rejects_unsupported_scheme(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="unsupported URL scheme"):
        download_file("file:///etc/passwd", tmp_path / "x")


def test_download_file_refuses_existing_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_urlopen(monkeypatch, b"x")
    out = tmp_path / "exists.vcf.gz"
    out.write_bytes(b"old")
    with pytest.raises(DownloadError, match="already exists"):
        download_file("https://x.test/a", out, acknowledge_source_terms=True)
    assert (
        download_file(
            "https://x.test/a",
            out,
            overwrite=True,
            acknowledge_source_terms=True,
        )["size_bytes"]
        == 1
    )


def test_download_manifest_fetches_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, b"payload")
    entries = [
        {
            "url": "https://x.test/a",
            "output": str(tmp_path / "a"),
            "license_terms": "test source terms",
        },
        {
            "url": "https://x.test/b",
            "output": str(tmp_path / "b"),
            "license_terms": "test source terms",
        },
    ]
    report = download_manifest(entries, acknowledge_source_terms=True)
    assert report["count"] == 2
    assert {item["source_terms"] for item in report["files"]} == {"test source terms"}
    assert (tmp_path / "a").is_file() and (tmp_path / "b").is_file()


def test_download_manifest_accepts_per_entry_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_urlopen(monkeypatch, b"payload")
    entries = [
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
            "output": str(tmp_path / "clinvar.vcf.gz"),
            "acknowledge_source_terms": True,
        }
    ]

    report = download_manifest(entries)

    assert report["files"][0]["source_terms"] == "NCBI ClinVar source terms"


def test_download_cli_requires_url_or_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        download.main([])


def test_download_cli_single(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_urlopen(monkeypatch, b"payload")
    out = tmp_path / "x.vcf.gz"
    rc = download.main(
        [
            "--url",
            "https://x.test/x.vcf.gz",
            "--output",
            str(out),
            "--acknowledge-source-terms",
            "--license-terms",
            "test source terms",
        ]
    )
    assert rc == 0
    assert out.is_file()
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["size_bytes"] == 7
    assert payload["source_terms"] == "test source terms"
