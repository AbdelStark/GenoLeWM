# SPDX-License-Identifier: Apache-2.0
"""Prepare and verify the membership lineage boundary from an installed wheel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _prepare(root: Path) -> None:
    from geno_lewm.data.membership_store import build_membership_store
    from geno_lewm.provenance import sha256_file
    from tests.unit.test_data_membership_store import (
        BUILDER_CONTAINER_IMAGE,
        BUILDER_GIT_COMMIT,
        _write_source_bundle,
    )
    from tests.unit.test_data_v03_snapshot_lineage import (
        SOURCE_LOCK,
        _write_evidence_bundle,
    )
    from tools.data.v03_snapshot_lineage import assemble_snapshot_lineage

    root.mkdir(parents=True, exist_ok=False)
    evidence_root = root / "lineage-evidence"
    evidence_root.mkdir()
    spec_path, _spec = _write_evidence_bundle(evidence_root)
    lineage_path = root / "nonfixture-lineage.json"
    assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=lineage_path,
    )

    fixture_lineage, sources = _write_source_bundle(root / "membership-sources")
    os.environ["GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE"] = BUILDER_CONTAINER_IMAGE
    build_membership_store(
        artifact_id="geno-lewm-wheel-membership-smoke",
        snapshot_lineage_path=fixture_lineage,
        expected_snapshot_lineage_sha256=sha256_file(fixture_lineage),
        builder_git_commit=BUILDER_GIT_COMMIT,
        container_image=BUILDER_CONTAINER_IMAGE,
        sources=sources,
        output_dir=root / "membership-store",
    )


def _verify(root: Path) -> None:
    if importlib.util.find_spec("tools") is not None:
        raise AssertionError("isolated wheel smoke unexpectedly resolved the source tools package")

    from geno_lewm.data import _membership_store_lineage as membership_lineage
    from geno_lewm.data._snapshot_lineage import capture_verified_snapshot_lineage
    from geno_lewm.data.membership_store import MembershipStore

    lineage_path = root / "nonfixture-lineage.json"
    captured = capture_verified_snapshot_lineage(lineage_path)
    if captured.lineage["assembly_inputs"] == {"fixture": True}:
        raise AssertionError("isolated wheel smoke did not capture a non-fixture lineage")
    if captured.payload != lineage_path.read_bytes():
        raise AssertionError("isolated wheel capture did not preserve the exact lineage bytes")
    if capture_verified_snapshot_lineage.__module__ != "geno_lewm.data._snapshot_lineage":
        raise AssertionError("snapshot-lineage verifier is not provided by the installed package")

    calls: list[Path] = []
    original_capture = membership_lineage.capture_verified_snapshot_lineage

    def _traced_capture(path: Path) -> object:
        calls.append(path)
        return original_capture(path)

    membership_lineage.capture_verified_snapshot_lineage = _traced_capture
    with MembershipStore.open(root / "membership-store", verify=True) as store:
        if store.manifest.row_count != 28:
            raise AssertionError("isolated wheel opened an unexpected membership store")
        if not store.contains_variant("GRCh38:21:121:A:G", roles=("evaluation",)):
            raise AssertionError(
                "isolated wheel membership lookup did not reach the packaged store"
            )
    if len(calls) != 1:
        raise AssertionError(
            "membership verification did not call the packaged lineage verifier once"
        )

    print(
        json.dumps(
            {
                "ok": True,
                "lineage_sha256": captured.payload_sha256,
                "membership_lineage_verifier_calls": len(calls),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args.root)
    else:
        _verify(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
