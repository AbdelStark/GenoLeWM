# SPDX-License-Identifier: Apache-2.0
"""Download pinned upstream genomic source files (ClinVar / gnomAD VCFs).

Operator-initiated acquisition tool. Lives under ``tools/`` so it may use the
network, unlike the fail-closed ``geno_lewm`` runtime (RFC-0010 §3.7). Fetches
a URL (https/ftp) to a local path — typically under
``configs/first_experiment/inputs/`` — verifying an expected SHA-256 when one is
given and always recording the realized digest + size. Run as::

    python -m tools.data.download \
        --url https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz \
        --output configs/first_experiment/inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz \
        --sha256 sha256:<expected>

A ``--manifest`` JSON ([{url, output, sha256?}, ...]) fetches multiple files.
Each fetch requires an explicit ``--acknowledge-source-terms`` flag or a
manifest entry with ``"acknowledge_source_terms": true`` so operators cannot
stage ClinVar/gnomAD inputs without recording that upstream source terms were
reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import urlparse

GENERATED_BY = "tools.data.download"
_CHUNK_BYTES = 1 << 20
_ALLOWED_SCHEMES = frozenset({"https", "ftp"})
_DEFAULT_SOURCE_TERMS = "upstream source terms"


class DownloadError(RuntimeError):
    """A download failed, was unsafe, or did not match the expected digest."""


def download_file(
    url: str,
    output: Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = False,
    acknowledge_source_terms: bool = False,
    license_terms: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Fetch ``url`` to ``output``, returning its realized sha256 + size."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise DownloadError(f"unsupported URL scheme {scheme!r}; allowed: https, ftp")
    source_terms = _require_source_terms_acknowledged(
        url,
        acknowledged=acknowledge_source_terms,
        license_terms=license_terms,
    )
    output = Path(output)
    if output.exists() and not overwrite:
        raise DownloadError(f"output already exists (pass overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    size = 0
    partial = output.with_name(output.name + ".part")
    try:
        request = urllib_request.Request(url, headers={"User-Agent": "geno-lewm-download/1"})
        with (
            urllib_request.urlopen(request, timeout=timeout_seconds) as response,
            partial.open("wb") as handle,
        ):
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except (urllib_error.HTTPError, urllib_error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"failed to download {url}: {exc}") from exc

    realized = f"sha256:{digest.hexdigest()}"
    if expected_sha256 is not None and realized != expected_sha256:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"sha256 mismatch for {url}: expected {expected_sha256}, got {realized}"
        )
    partial.replace(output)
    return {
        "generated_by": GENERATED_BY,
        "url": url,
        "output": str(output),
        "sha256": realized,
        "size_bytes": size,
        "source_terms": source_terms,
        "source_terms_acknowledged": True,
    }


def download_manifest(
    entries: list[dict[str, Any]],
    *,
    overwrite: bool = False,
    acknowledge_source_terms: bool = False,
) -> dict[str, Any]:
    """Fetch every ``{url, output, sha256?}`` entry; return a combined report."""
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        url = entry.get("url")
        output = entry.get("output")
        if not isinstance(url, str) or not isinstance(output, str):
            raise DownloadError(f"manifest entry {index} must include string url + output")
        entry_ack = _manifest_entry_acknowledgement(entry, acknowledge_source_terms, index)
        license_terms = entry.get("license_terms")
        if license_terms is not None and not isinstance(license_terms, str):
            raise DownloadError(f"manifest entry {index} license_terms must be a string")
        results.append(
            download_file(
                url,
                Path(output),
                expected_sha256=entry.get("sha256"),
                overwrite=overwrite,
                acknowledge_source_terms=entry_ack,
                license_terms=license_terms,
            )
        )
    return {"generated_by": GENERATED_BY, "count": len(results), "files": results}


def _manifest_entry_acknowledgement(
    entry: dict[str, Any],
    global_acknowledgement: bool,
    index: int,
) -> bool:
    if global_acknowledgement:
        return True
    raw = entry.get("acknowledge_source_terms", entry.get("license_acknowledged", False))
    if not isinstance(raw, bool):
        raise DownloadError(f"manifest entry {index} acknowledge_source_terms must be a boolean")
    return raw


def _require_source_terms_acknowledged(
    url: str,
    *,
    acknowledged: bool,
    license_terms: str | None,
) -> str:
    terms = _source_terms_label(url, license_terms)
    if not acknowledged:
        raise DownloadError(
            "source terms must be acknowledged before downloading "
            f"{url} ({terms}); pass --acknowledge-source-terms or set "
            "acknowledge_source_terms=true in the manifest"
        )
    return terms


def _source_terms_label(url: str, license_terms: str | None) -> str:
    if license_terms is not None:
        stripped = license_terms.strip()
        if not stripped:
            raise DownloadError("license_terms must not be empty")
        return stripped
    parsed = urlparse(url)
    normalized = f"{parsed.netloc}{parsed.path}".lower()
    if "gnomad" in normalized or "broadinstitute.org" in normalized:
        return "gnomAD data-use terms"
    if "clinvar" in normalized or "ncbi.nlm.nih.gov" in normalized:
        return "NCBI ClinVar source terms"
    return _DEFAULT_SOURCE_TERMS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Single source URL (https/ftp).")
    parser.add_argument("--output", type=Path, help="Destination path for --url.")
    parser.add_argument("--sha256", default=None, help="Expected sha256:<hex> for --url.")
    parser.add_argument("--manifest", type=Path, help="JSON list of {url, output, sha256?}.")
    parser.add_argument(
        "--acknowledge-source-terms",
        action="store_true",
        help="Record that upstream source terms for each URL were reviewed before download.",
    )
    parser.add_argument("--license-terms", default=None, help="Source-terms label for --url.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.manifest is not None:
            entries = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                raise DownloadError("manifest must be a JSON list")
            report = download_manifest(
                entries,
                overwrite=args.overwrite,
                acknowledge_source_terms=args.acknowledge_source_terms,
            )
        elif args.url is not None and args.output is not None:
            report = download_file(
                args.url,
                args.output,
                expected_sha256=args.sha256,
                overwrite=args.overwrite,
                acknowledge_source_terms=args.acknowledge_source_terms,
                license_terms=args.license_terms,
            )
        else:
            parser.error("provide either --manifest or both --url and --output")
    except DownloadError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
