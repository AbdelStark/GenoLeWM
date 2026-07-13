#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Exact-revision v0.3 membership-split publication for Hugging Face Jobs.
#
# This job publishes deterministic variant memberships and placed-window
# nonintersection evidence, not phased haplotypes. Success is not a released
# v0.3 snapshot and does not establish dataset representativeness, model
# quality, benchmark performance, or clinical validity.
#
# Submit only after the split implementation, schema, and this runner are
# merged at the same immutable, clean commit:
#
#   SHA="<40-character GenoLeWM commit>"
#   RUN_ATTEMPT=1
#   IMAGE="ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
#   hf jobs run \
#     --flavor cpu-upgrade \
#     --secrets HF_TOKEN \
#     --env COMMIT_SHA="$SHA" \
#     --env RUN_ATTEMPT="$RUN_ATTEMPT" \
#     --env CONTAINER_IMAGE="$IMAGE" \
#     --timeout 4h \
#     --detach \
#     -- "$IMAGE" \
#     bash -lc 'set -euo pipefail; git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; cd /workspace/GenoLeWM; git checkout --detach "$COMMIT_SHA"; test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; test -z "$(git status --porcelain=v1 --untracked-files=all)"; uv sync --frozen --extra evidence; exec uv run --no-sync bash tools/jobs/v03_publish_membership_splits.sh'

set -euo pipefail

WORK="/tmp/geno-lewm-v03-membership-splits"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"

DATA_REPO="abdelstark/geno-lewm-data"
UPLOAD_REPO="abdelstark/geno-lewm-data"
INPUT_REVISION="96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2"
MEMBERSHIP_CANDIDATE_ROOT="candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1"
MEMBERSHIP_PATH="candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/success"
MEMBERSHIP_ARTIFACT_ID="geno-lewm-data-v0.3.0-membership-r1"
MEMBERSHIP_CONTENT_IDENTITY="sha256:7fa661eefacf70258b8392aff88a6faea2749c812680d4a2bfc41376d061ff7a"
MEMBERSHIP_PHYSICAL_IDENTITY="sha256:d7ea2c4b8413768c9128c70a299a11f4adf35140102778a71cf56e69fb4db536"
MEMBERSHIP_ROWSET_SHA256="sha256:d268f2e2b67cce56c5d5099ec1ddcbd810fbb5973e6c96a929fd2c99fbd25f68"

DATASET_MANIFEST_PATH="dataset_manifest.json"
DATASET_MANIFEST_SHA256="sha256:c3aa8f22b79e76fa5b6e3a43e02675cfc02d56dc7dc9fa36128c81874537016c"
DATASET_MANIFEST_SIZE_BYTES="5051"
DATASET_SNAPSHOT_ID="geno-lewm-data-v0.1.0-r1"
PLACED_WINDOWS_PATH="placed/gnomad-common-windows.jsonl"
PLACED_WINDOWS_SHA256="sha256:ec76046771a163fbc22f326df26e2a332767eaa045dd919718c1cf86c4fbe0ac"
PLACED_WINDOWS_SIZE_BYTES="4186560"
PLACED_WINDOWS_RECORD_COUNT="976"

ARTIFACT_ID="geno-lewm-v03-membership-splits-r1"
RUN_NAME="${RUN_NAME:-geno-lewm-v03-membership-splits-${COMMIT_SHA:0:12}-r${RUN_ATTEMPT}}"
PUBLISH_NAMESPACE="${MEMBERSHIP_CANDIDATE_ROOT}/splits/${RUN_NAME}/success"

log() { echo "=== $* ==="; }
fatal() { echo "FATAL: $*" >&2; exit 2; }

log "validate immutable source, container, attempt, and clean checkout before writes"
[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "COMMIT_SHA must be a full lowercase 40-character Git SHA"
[[ "$CONTAINER_IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] \
  || fatal "CONTAINER_IMAGE must be digest-pinned"
[[ "$RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]] \
  || fatal "RUN_ATTEMPT must be a positive canonical integer"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fatal "RUN_NAME is not a safe immutable namespace component"

OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"
[ "$OBSERVED_COMMIT_SHA" = "$COMMIT_SHA" ] \
  || fatal "commit drift: expected $COMMIT_SHA, observed $OBSERVED_COMMIT_SHA"
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
[ "$(pwd -P)" = "$(cd "$REPOSITORY_ROOT" && pwd -P)" ] \
  || fatal "membership split job must run from the repository root"
case "$REPOSITORY_ROOT/" in
  "$WORK/"*) fatal "fixed membership split workspace must not contain the checkout" ;;
esac
case "$WORK/" in
  "$REPOSITORY_ROOT/"*) fatal "fixed membership split workspace must remain outside the checkout" ;;
esac
git diff --quiet -- . || fatal "tracked worktree differs from $COMMIT_SHA"
git diff --cached --quiet -- . || fatal "index differs from $COMMIT_SHA"
[ -z "$(git status --porcelain=v1 --untracked-files=all)" ] \
  || fatal "membership split checkout contains untracked or modified inputs"
[ "$(git remote get-url origin)" = "https://github.com/AbdelStark/GenoLeWM.git" ] \
  || fatal "origin is not the canonical GenoLeWM repository"

for tracked_path in \
  tools/jobs/v03_publish_membership_splits.sh \
  tools/data/v03_membership_splits.py \
  tools/data/v03_gnomad_lock.py \
  tools/data/v03_membership_store.py \
  configs/data_v03/membership-split-evidence.schema.json
do
  git cat-file -e "$COMMIT_SHA:$tracked_path" \
    || fatal "required split-publication input is not tracked at $COMMIT_SHA: $tracked_path"
done

command -v hf >/dev/null 2>&1 \
  || fatal "the locked evidence environment does not provide the Hugging Face CLI"
python - <<'PY'
import huggingface_hub
import jsonschema
import pyarrow
PY

export HF_TOKEN
export GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE="$CONTAINER_IMAGE"

rm -rf "$WORK"
INPUT_ROOT="$WORK/input"
PUBLIC_DIR="$WORK/public"
MEMBERSHIP_BUNDLE="$INPUT_ROOT/$MEMBERSHIP_PATH"
MEMBERSHIP_VERIFY_REPORT="$WORK/membership-verify-report.json"
DATASET_MANIFEST="$INPUT_ROOT/$DATASET_MANIFEST_PATH"
PLACED_WINDOWS="$INPUT_ROOT/$PLACED_WINDOWS_PATH"
EXPORT_REPORT="$WORK/membership-split-export-report.json"
mkdir -p "$INPUT_ROOT"

log "force-download the exact published membership candidate"
hf download "$DATA_REPO" \
  --repo-type dataset \
  --revision "$INPUT_REVISION" \
  --include "$MEMBERSHIP_PATH/**" \
  --force-download \
  --local-dir "$INPUT_ROOT"

log "independently verify membership namespace inventory and checksums"
python - "$MEMBERSHIP_BUNDLE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "SHA256SUMS",
    "contract/membership-build-receipt.schema.json",
    "contract/membership-build-spec.schema.json",
    "contract/membership-build.json",
    "contract/membership-store.schema.json",
    "evidence/download-plan.json",
    "evidence/job-summary.json",
    "evidence/membership-build-report.json",
    "evidence/membership-verify-report.json",
    "evidence/source-download-identities.json",
    "store/build-receipt.json",
    "store/lookup.sqlite",
    "store/manifest.json",
    "store/memberships.parquet",
    "store/snapshot-lineage.json",
}
if not root.is_dir():
    raise SystemExit("FATAL: exact membership namespace is absent")
if any(path.is_symlink() for path in root.rglob("*")):
    raise SystemExit("FATAL: membership namespace contains a symbolic link")
observed = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
if observed != expected:
    raise SystemExit(
        f"FATAL: exact membership inventory drifted: "
        f"missing={sorted(expected - observed)!r} unexpected={sorted(observed - expected)!r}"
    )
PY
(
  cd "$MEMBERSHIP_BUNDLE"
  sha256sum -c SHA256SUMS
)

log "independently full-scan the exact membership store and pin audited invariants"
python -m tools.data.v03_membership_store verify \
  --store-dir "$MEMBERSHIP_BUNDLE/store" \
  > "$MEMBERSHIP_VERIFY_REPORT"
cmp "$MEMBERSHIP_BUNDLE/evidence/membership-verify-report.json" "$MEMBERSHIP_VERIFY_REPORT" \
  || fatal "fresh membership verification differs from the published verification report"
python - \
  "$MEMBERSHIP_VERIFY_REPORT" \
  "$MEMBERSHIP_ARTIFACT_ID" \
  "$MEMBERSHIP_CONTENT_IDENTITY" \
  "$MEMBERSHIP_PHYSICAL_IDENTITY" \
  "$MEMBERSHIP_ROWSET_SHA256" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "ok": True,
    "artifact_id": sys.argv[2],
    "lineage_evidence_profile": "official",
    "content_identity": sys.argv[3],
    "physical_identity": sys.argv[4],
    "rowset_sha256": sys.argv[5],
    "row_count": 2_335_042,
    "variant_count": 2_259_268,
    "role_counts": {
        "train": 2_251_087,
        "validation": 53_002,
        "evaluation": 30_953,
    },
    "source_kind_role_counts": {
        "gnomad": {"train": 705_827, "validation": 18_345, "evaluation": 10_300},
        "clinvar": {"train": 1_545_260, "validation": 34_657, "evaluation": 20_653},
    },
    "clinvar_class_role_counts": {
        "B": {"train": 189_595, "validation": 4_720, "evaluation": 2_851},
        "LB": {"train": 1_061_095, "validation": 24_886, "evaluation": 14_395},
        "LP": {"train": 137_046, "validation": 2_297, "evaluation": 1_549},
        "P": {"train": 157_524, "validation": 2_754, "evaluation": 1_858},
    },
}
for field, value in expected.items():
    if report.get(field) != value:
        raise SystemExit(f"FATAL: membership {field} differs from the pinned audited invariant")
if len(report.get("source_role_counts", {})) != 23:
    raise SystemExit("FATAL: membership store is not bound to exactly 23 source roles")
PY

log "force-download the exact dataset manifest and placed-window artifact"
hf download "$DATA_REPO" \
  "$DATASET_MANIFEST_PATH" \
  "$PLACED_WINDOWS_PATH" \
  --repo-type dataset \
  --revision "$INPUT_REVISION" \
  --force-download \
  --local-dir "$INPUT_ROOT"

log "verify the exact dataset manifest and placed-window bytes"
python - \
  "$DATASET_MANIFEST" \
  "$DATASET_MANIFEST_SHA256" \
  "$DATASET_MANIFEST_SIZE_BYTES" \
  "$DATASET_SNAPSHOT_ID" \
  "$PLACED_WINDOWS" \
  "$PLACED_WINDOWS_PATH" \
  "$PLACED_WINDOWS_SHA256" \
  "$PLACED_WINDOWS_SIZE_BYTES" \
  "$PLACED_WINDOWS_RECORD_COUNT" <<'PY'
import hashlib
import json
import stat
import sys
from pathlib import Path

(
    manifest_path_text,
    expected_manifest_sha256,
    expected_manifest_size,
    expected_snapshot_id,
    windows_path_text,
    windows_artifact_path,
    expected_windows_sha256,
    expected_windows_size,
    expected_windows_records,
) = sys.argv[1:]
manifest_path = Path(manifest_path_text)
windows_path = Path(windows_path_text)


def identity(path: Path) -> tuple[str, int]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise SystemExit(f"FATAL: expected a regular non-symlink input: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


manifest_identity = identity(manifest_path)
if manifest_identity != (expected_manifest_sha256, int(expected_manifest_size)):
    raise SystemExit("FATAL: dataset manifest identity drifted")
windows_identity = identity(windows_path)
if windows_identity != (expected_windows_sha256, int(expected_windows_size)):
    raise SystemExit("FATAL: placed-window artifact identity drifted")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("snapshot_id") != expected_snapshot_id:
    raise SystemExit("FATAL: dataset manifest snapshot_id drifted")
matches = [
    item
    for item in manifest.get("files", [])
    if isinstance(item, dict) and item.get("path") == windows_artifact_path
]
if len(matches) != 1:
    raise SystemExit("FATAL: dataset manifest does not bind the placed windows exactly once")
binding = matches[0]
expected_binding = {
    "sha256": expected_windows_sha256,
    "size_bytes": int(expected_windows_size),
    "records": int(expected_windows_records),
    "split": "train_placed_gnomad_common",
}
observed_binding = {field: binding.get(field) for field in expected_binding}
if observed_binding != expected_binding:
    raise SystemExit("FATAL: dataset manifest placed-window binding drifted")

record_count = 0
with windows_path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            raise SystemExit("FATAL: placed-window artifact contains a blank line")
        json.loads(line)
        record_count += 1
if record_count != int(expected_windows_records):
    raise SystemExit("FATAL: placed-window record count drifted")
PY

log "export deterministic ClinVar streams and placed-window nonintersection evidence"
test ! -e "$PUBLIC_DIR" || fatal "immutable split output already exists: $PUBLIC_DIR"
python -m tools.data.v03_membership_splits \
  --store-dir "$MEMBERSHIP_BUNDLE/store" \
  --placed-windows-jsonl "$PLACED_WINDOWS" \
  --dataset-manifest-json "$DATASET_MANIFEST" \
  --output-dir "$PUBLIC_DIR" \
  --artifact-id "$ARTIFACT_ID" \
  --membership-repository "$DATA_REPO" \
  --membership-revision "$INPUT_REVISION" \
  --membership-artifact-path "$MEMBERSHIP_PATH" \
  --training-windows-repository "$DATA_REPO" \
  --training-windows-revision "$INPUT_REVISION" \
  --training-windows-artifact-path "$PLACED_WINDOWS_PATH" \
  --expected-store-content-identity "$MEMBERSHIP_CONTENT_IDENTITY" \
  --expected-store-physical-identity "$MEMBERSHIP_PHYSICAL_IDENTITY" \
  --expected-store-rowset-sha256 "$MEMBERSHIP_ROWSET_SHA256" \
  --expected-dataset-manifest-sha256 "$DATASET_MANIFEST_SHA256" \
  --expected-dataset-snapshot-id "$DATASET_SNAPSHOT_ID" \
  --expected-placed-windows-sha256 "$PLACED_WINDOWS_SHA256" \
  --expected-placed-windows-size-bytes "$PLACED_WINDOWS_SIZE_BYTES" \
  --expected-placed-windows-record-count "$PLACED_WINDOWS_RECORD_COUNT" \
  --producer-git-commit "$COMMIT_SHA" \
  --container-image "$CONTAINER_IMAGE" \
  --sample-seed 20260713 \
  --sample-size 128 \
  --report-schema-path configs/data_v03/membership-split-evidence.schema.json \
  > "$EXPORT_REPORT"
cmp "$EXPORT_REPORT" "$PUBLIC_DIR/evidence/membership-split-evidence.json" \
  || fatal "export stdout differs from the published evidence report"
cmp \
  configs/data_v03/membership-split-evidence.schema.json \
  "$PUBLIC_DIR/contract/membership-split-evidence.schema.json" \
  || fatal "bundled split schema differs from the exact checked-out contract"

verify_split_bundle() {
  local bundle="$1"
  local verification_label="$2"
  python - \
    "$bundle" \
    "$MEMBERSHIP_BUNDLE/store" \
    "$PLACED_WINDOWS" \
    "$ARTIFACT_ID" \
    "$COMMIT_SHA" \
    "$CONTAINER_IMAGE" \
    "$DATA_REPO" \
    "$INPUT_REVISION" \
    "$MEMBERSHIP_PATH" \
    "$MEMBERSHIP_ARTIFACT_ID" \
    "$MEMBERSHIP_CONTENT_IDENTITY" \
    "$MEMBERSHIP_PHYSICAL_IDENTITY" \
    "$MEMBERSHIP_ROWSET_SHA256" \
    "$DATASET_MANIFEST_SHA256" \
    "$DATASET_MANIFEST_SIZE_BYTES" \
    "$DATASET_SNAPSHOT_ID" \
    "$PLACED_WINDOWS_PATH" \
    "$PLACED_WINDOWS_SHA256" \
    "$PLACED_WINDOWS_SIZE_BYTES" \
    "$PLACED_WINDOWS_RECORD_COUNT" \
    "$verification_label" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator

from geno_lewm.data.builder import WindowContext
from geno_lewm.data.membership_store import MembershipStore, MembershipStoreHoldoutPolicy

(
    bundle_text,
    store_text,
    windows_text,
    artifact_id,
    producer_commit,
    container_image,
    repository,
    revision,
    membership_path,
    membership_artifact_id,
    membership_content_identity,
    membership_physical_identity,
    membership_rowset_sha256,
    manifest_sha256,
    manifest_size,
    snapshot_id,
    windows_artifact_path,
    windows_sha256,
    windows_size,
    windows_records,
    verification_label,
) = sys.argv[1:]
bundle = Path(bundle_text)
store_path = Path(store_text)
placed_windows = Path(windows_text)
expected_files = {
    "SHA256SUMS",
    "contract/membership-split-evidence.schema.json",
    "evidence/membership-split-evidence.json",
    "splits/evaluation/clinvar-chr21.labels.jsonl",
    "splits/evaluation/clinvar-chr21.vcf",
    "splits/validation/clinvar-chr20.labels.jsonl",
    "splits/validation/clinvar-chr20.vcf",
}


def fail(message: str) -> None:
    raise SystemExit(f"FATAL: {verification_label}: {message}")


def file_identity(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


if not bundle.is_dir():
    fail("bundle directory is absent")
if any(path.is_symlink() for path in bundle.rglob("*")):
    fail("bundle contains a symbolic link")
observed_files = {
    path.relative_to(bundle).as_posix()
    for path in bundle.rglob("*")
    if path.is_file()
}
if observed_files != expected_files:
    fail(
        "exact output inventory drifted: "
        f"missing={sorted(expected_files - observed_files)!r} "
        f"unexpected={sorted(observed_files - expected_files)!r}"
    )

checksum_lines = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
checksum_paths: list[str] = []
for line in checksum_lines:
    if len(line) < 67 or line[64:66] != "  ":
        fail("SHA256SUMS contains a malformed entry")
    digest, relative = line[:64], line[66:]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        fail("SHA256SUMS contains a noncanonical digest")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        fail("SHA256SUMS contains an unsafe path")
    checksum_paths.append(relative)
    observed = hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
    if observed != digest:
        fail(f"checksum mismatch for {relative}")
if checksum_paths != sorted(expected_files - {"SHA256SUMS"}):
    fail("SHA256SUMS does not close the exact sorted output inventory")

schema = json.loads(
    (bundle / "contract/membership-split-evidence.schema.json").read_text(encoding="utf-8")
)
report = json.loads(
    (bundle / "evidence/membership-split-evidence.json").read_text(encoding="utf-8")
)
Draft202012Validator.check_schema(schema)
errors = sorted(
    Draft202012Validator(schema).iter_errors(report),
    key=lambda error: tuple(str(item) for item in error.absolute_path),
)
if errors:
    fail(f"report does not satisfy its bundled schema: {errors[0].message}")

expected_producer = {
    "generated_by": "tools.data.v03_membership_splits",
    "git_commit": producer_commit,
    "container_image": container_image,
    "invocation_verified": True,
}
if report.get("producer") != expected_producer:
    fail("producer binding drifted")
membership = report.get("membership_store")
expected_membership = {
    "repository": repository,
    "revision": revision,
    "artifact_path": membership_path,
    "artifact_id": membership_artifact_id,
    "content_identity": membership_content_identity,
    "physical_identity": membership_physical_identity,
    "rowset_sha256": membership_rowset_sha256,
    "lineage": {
        "lineage_id": "sha256:1cdff2f256f3dc63bf74fc8092c2644050030c92a377bde234abddd44542d986",
        "sha256": "sha256:dcc7031bb1b409e55112c1f6576a878b9566b954d32ea75056a04b9ba1e95bea",
        "candidate_snapshot_id": "geno-lewm-data-v0.3.0-r1",
        "evidence_profile": "official",
    },
    "chromosome_roles": {
        "train": [*map(str, range(1, 20)), "22"],
        "validation": ["20"],
        "evaluation": ["21"],
    },
}
if membership != expected_membership:
    fail("membership-store provenance drifted")

training = report.get("training_windows")
expected_training = {
    "source": {
        "repository": repository,
        "revision": revision,
        "artifact_path": windows_artifact_path,
    },
    "sha256": windows_sha256,
    "size_bytes": int(windows_size),
    "record_count": int(windows_records),
    "assembly": "GRCh38",
    "role": "train",
    "split": "train_placed_gnomad_common",
    "chromosomes": ["22"],
    "dataset_manifest": {
        "path": "dataset_manifest.json",
        "sha256": manifest_sha256,
        "size_bytes": int(manifest_size),
        "snapshot_id": snapshot_id,
    },
    "record_fields": [
        "record_id",
        "source",
        "variant_source",
        "chrom",
        "start_bp",
        "end_bp",
        "sequence",
        "variant_count",
    ],
}
if training != expected_training:
    fail("training-window provenance drifted")

expected_streams = {
    "validation": {
        "chromosome": "20",
        "record_count": 34_657,
        "class_counts": {"B": 4_720, "LB": 24_886, "LP": 2_297, "P": 2_754},
        "binary_counts": {"negative": 29_606, "positive": 5_051},
    },
    "evaluation": {
        "chromosome": "21",
        "record_count": 20_653,
        "class_counts": {"B": 2_851, "LB": 14_395, "LP": 1_549, "P": 1_858},
        "binary_counts": {"negative": 17_246, "positive": 3_407},
    },
}
for role, expected in expected_streams.items():
    stream = report["streams"][role]
    if stream.get("role") != role:
        fail(f"{role} stream role drifted")
    for field in ("chromosome", "record_count", "class_counts", "binary_counts"):
        if stream.get(field) != expected[field]:
            fail(f"{role} stream {field} drifted")
    stem = f"splits/{role}/clinvar-chr{expected['chromosome']}"
    labels_relative = f"{stem}.labels.jsonl"
    vcf_relative = f"{stem}.vcf"
    labels_path = bundle / labels_relative
    vcf_path = bundle / vcf_relative
    if stream.get("labels_jsonl") != file_identity(labels_path, labels_relative):
        fail(f"{role} JSONL identity drifted")
    if stream.get("vcf") != file_identity(vcf_path, vcf_relative):
        fail(f"{role} VCF identity drifted")

    labels: list[dict[str, object]] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                fail(f"{role} JSONL contains a blank line at {line_number}")
            row = json.loads(line)
            if set(row) != {"chrom", "pos", "ref", "alt", "clinical_significance"}:
                fail(f"{role} JSONL row shape drifted")
            labels.append(row)
    label_keys = [
        (str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"]))
        for row in labels
    ]
    if len(labels) != expected["record_count"] or len(set(label_keys)) != len(label_keys):
        fail(f"{role} JSONL cardinality or uniqueness drifted")
    if any(key[0] != expected["chromosome"] for key in label_keys):
        fail(f"{role} JSONL chromosome drifted")
    class_counts = {label: 0 for label in ("B", "LB", "LP", "P")}
    for row in labels:
        label = str(row["clinical_significance"])
        if label not in class_counts:
            fail(f"{role} JSONL contains an unknown class")
        class_counts[label] += 1
    if class_counts != expected["class_counts"]:
        fail(f"{role} JSONL class counts drifted")
    with MembershipStore.open(store_path, verify=False) as store:
        expected_labels = [
            {
                "chrom": labeled.membership.variant.chrom,
                "pos": labeled.membership.variant.pos,
                "ref": labeled.membership.variant.ref,
                "alt": labeled.membership.variant.alt,
                "clinical_significance": labeled.clinical_significance,
            }
            for labeled in store.iter_labeled_clinvar(role)
        ]
    if labels != expected_labels:
        fail(f"{role} JSONL rows or classes differ from the pinned membership store")

    vcf_keys: list[tuple[str, int, str, str]] = []
    vcf_classes: list[str] = []
    with vcf_path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines or lines[0] != "##fileformat=VCFv4.3":
        fail(f"{role} VCF header drifted")
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 8 or fields[6] != "PASS":
            fail(f"{role} VCF row shape drifted")
        info = dict(item.split("=", 1) for item in fields[7].split(";"))
        if set(info) != {"CLNSIG", "ROLE", "LABEL"} or info["ROLE"] != role:
            fail(f"{role} VCF INFO drifted")
        expected_label = "1" if info["CLNSIG"] in {"LP", "P"} else "0"
        if info["LABEL"] != expected_label:
            fail(f"{role} VCF binary label drifted")
        vcf_keys.append((fields[0], int(fields[1]), fields[3], fields[4]))
        vcf_classes.append(info["CLNSIG"])
    if vcf_keys != label_keys:
        fail(f"{role} JSONL/VCF key order or parity drifted")
    if vcf_classes != [str(row["clinical_significance"]) for row in labels]:
        fail(f"{role} JSONL/VCF class parity drifted")

    keyset = hashlib.sha256()
    for chrom, pos, ref, alt in label_keys:
        key = f"GRCh38:{chrom}:{pos}:{ref}:{alt}".encode()
        keyset.update(len(key).to_bytes(8, "big"))
        keyset.update(key)
    if stream.get("keyset_sha256") != "sha256:" + keyset.hexdigest():
        fail(f"{role} keyset digest drifted")

audits = report.get("audits")
if audits.get("exhaustive") != {
    "windows_scanned": 976,
    "policy_exclusions": 0,
    "indexed_overlaps": 0,
    "status": "passed",
}:
    fail("exhaustive window audit drifted")
sample = audits.get("deterministic_sample")
expected_sample_fields = {
    "algorithm": "sha256-priority-v1",
    "seed": 20_260_713,
    "requested_size": 128,
    "observed_size": 128,
    "policy_exclusions": 0,
    "indexed_overlaps": 0,
    "status": "passed",
}
for field, value in expected_sample_fields.items():
    if sample.get(field) != value:
        fail(f"deterministic sample {field} drifted")

window_rows = [json.loads(line) for line in placed_windows.read_text(encoding="utf-8").splitlines()]
if len(window_rows) != 976:
    fail("independent window universe drifted")
sample_candidates: list[tuple[str, dict[str, object]]] = []
with MembershipStore.open(store_path, verify=False) as store:
    policy = MembershipStoreHoldoutPolicy(store)
    for row in window_rows:
        identity = {
            "record_id": row["record_id"],
            "chrom": row["chrom"],
            "start_bp": row["start_bp"],
            "end_bp": row["end_bp"],
            "window_sha256": "sha256:"
            + hashlib.sha256(str(row["sequence"]).encode()).hexdigest(),
        }
        priority = hashlib.sha256(b"20260713\x00" + canonical_json(identity)).hexdigest()
        sample_candidates.append((priority, identity))
        context = WindowContext(
            record_id=str(row["record_id"]),
            source=str(row["source"]),
            sequence=str(row["sequence"]),
            start_bp=int(row["start_bp"]),
            chrom=str(row["chrom"]),
        )
        if policy.excludes_window(context):
            fail("independent holdout policy excluded a placed training window")
        if store.overlaps_interval(
            str(row["chrom"]),
            start_bp=int(row["start_bp"]),
            end_bp=int(row["end_bp"]),
            roles=("validation", "evaluation"),
        ):
            fail("independent index scan found a held-out overlap")
sample_payload = [
    {"priority_sha256": "sha256:" + priority, **identity}
    for priority, identity in sorted(sample_candidates)[:128]
]
expected_sample_digest = "sha256:" + hashlib.sha256(canonical_json(sample_payload)).hexdigest()
if sample.get("sample_digest") != expected_sample_digest:
    fail("deterministic sample digest drifted")

claim = report.get("claim_boundary")
expected_claim_flags = {
    "variant_membership": True,
    "phased_haplotype_membership": False,
    "released_v03_snapshot": False,
    "publication_eligible": True,
}
for field, value in expected_claim_flags.items():
    if claim.get(field) != value:
        fail(f"claim-boundary field {field} drifted")
limitations = " ".join(claim.get("limitations", [])).lower()
for phrase in (
    "variant memberships",
    "phased haplotypes",
    "released v0.3 snapshot",
    "dataset representativeness",
    "model quality",
    "benchmark performance",
    "clinical validity",
):
    if phrase not in limitations:
        fail(f"claim-boundary limitation is missing: {phrase}")
if report.get("artifact_id") != artifact_id or report.get("ok") is not True:
    fail("top-level artifact identity or status drifted")
PY
}

log "independently validate the complete split evidence bundle"
verify_split_bundle "$PUBLIC_DIR" "local split evidence"
(
  cd "$PUBLIC_DIR"
  sha256sum -c SHA256SUMS
)

log "publish the complete verified split bundle in one conflict-safe Hub commit"
PUBLISH_REPORT="$(
  python -m tools.data.v03_gnomad_lock publish \
    --repo-id "$UPLOAD_REPO" \
    --repo-type dataset \
    --namespace "$PUBLISH_NAMESPACE" \
    --publish-dir "$PUBLIC_DIR" \
    --commit-message "publish verified v0.3 membership splits from $COMMIT_SHA"
)"
HUB_REVISION="${PUBLISH_REPORT#uploaded commit: }"
[[ "$HUB_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "Hub publication did not return an immutable 40-character revision"

log "force-download the exact resulting Hub commit"
test ! -e "$WORK/verified-remote" \
  || fatal "remote verification directory already exists"
hf download "$UPLOAD_REPO" \
  --repo-type dataset \
  --revision "$HUB_REVISION" \
  --include "$PUBLISH_NAMESPACE/**" \
  --force-download \
  --local-dir "$WORK/verified-remote"
REMOTE_BUNDLE="$WORK/verified-remote/$PUBLISH_NAMESPACE"
cmp "$PUBLIC_DIR/SHA256SUMS" "$REMOTE_BUNDLE/SHA256SUMS" \
  || fatal "remote checksum manifest differs from the locally verified manifest"
cmp \
  "$PUBLIC_DIR/evidence/membership-split-evidence.json" \
  "$REMOTE_BUNDLE/evidence/membership-split-evidence.json" \
  || fatal "remote evidence report differs from the locally verified report"

log "reverify the exact published split bundle"
verify_split_bundle "$REMOTE_BUNDLE" "remote split evidence"
(
  cd "$REMOTE_BUNDLE"
  sha256sum -c SHA256SUMS
)

printf '%s\n' "$PUBLISH_REPORT"
echo "GENO_LEWM_V03_SPLITS_OK $HUB_REVISION $PUBLISH_NAMESPACE"
