# Finite window-cache builds

`geno-lewm-cache-windows` can build corrected cache-schema-3 shards for one
explicit, immutable request artifact. The mode is intended for bounded proof
runs and resumable production slices. It does not infer a corpus iterator or
silently expand the requested scope.

## Request artifact

The input is UTF-8, newline-terminated JSONL. Blank lines, duplicate JSON keys,
duplicate `request_id` values, unknown fields, invalid DNA, inconsistent
coordinates, and out-of-window edit loci fail before cache writes.

```json
{"chrom":"22","edit_locus":6,"end_bp":12,"request_id":"chr22-window-0001-edit-6","start_bp":0,"window":"ACGTACGTACGT"}
```

Each object has exactly these fields:

| Field | Contract |
| --- | --- |
| `request_id` | Unique non-empty text identifier. |
| `chrom` | Non-empty contig identity recorded in the shard row. |
| `start_bp`, `end_bp` | Zero-based half-open coordinates; their span equals `len(window)`. |
| `window` | Non-empty DNA over `A`, `C`, `G`, `T`, and `N`. |
| `edit_locus` | Zero-based offset in `window`, or `null` for untargeted global pooling. |

Deduplication occurs only after `CarbonStateEncoder.pooling_identity` resolves
the complete `WindowCacheKey`. Two requests for the same window remain distinct
when their edit loci map to different `center_token` values.

## Canonical v0.3 training trace

`tools.data.v03_training_trace` derives cache requests from the same
`PreparedTrainingStream` consumed by the real trainer. The canonical config is
`configs/data_v03/train-carbon-500m-snv-l2-epoch-r1.yaml`: corrected
`l2_normalized_v2`, tokenizer-resolved `centered_mean` pooling, the pinned
Carbon revision, eight samples per batch, and 938 steps. Construction fails if
the prepared epoch no longer contains exactly 938 complete batches.

The producer checkout must be clean, at the exact supplied commit, use the
canonical Git origin, and resolve through an unauthenticated exact-commit
lookup on canonical GitHub. The local dataset passes the complete v0.3
snapshot verifier before and after authoring. Its 52-file namespace is also
rebound without credentials to the exact Hub commit through sizes and Git/LFS
content identities at the hard-coded `https://huggingface.co` endpoint.

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
DATASET_REVISION="712d612d85ea6341b8ce17bd3460ff5c2207b802"
DATASET_PATH="candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/snapshots/geno-lewm-v03-dataset-snapshot-959079248000-r3/success"

GENO_LEWM_TRAINING_TRACE_DECLARED_CONTAINER_IMAGE="$CONTAINER_IMAGE" \
uv run --extra evidence python -m tools.data.v03_training_trace \
  --dataset-dir "$SNAPSHOT_DIR" \
  --training-config configs/data_v03/train-carbon-500m-snv-l2-epoch-r1.yaml \
  --output-dir "$TRACE_DIR" \
  --producer-git-commit "$SOURCE_SHA" \
  --container-image "$CONTAINER_IMAGE" \
  --dataset-repository abdelstark/geno-lewm-data \
  --dataset-revision "$DATASET_REVISION" \
  --dataset-artifact-path "$DATASET_PATH"
```

The closed output bundles the exact request JSONL, training config, report
schema, report, and `SHA256SUMS`. Re-run the command with `--verify-existing`
and the same arguments to reopen the published dataset, re-author the trace in
a separate temporary directory, and require byte identity. This proves one
deterministic epoch's request schedule; it does not prove that Carbon states
were encoded, that a cache was completed, or that throughput was measured. The
container digest is explicitly a launcher declaration, not self-attestation;
publication evidence must pair it with the external digest-pinned Hugging Face
Job receipt.

For no-follow publication, every `TRACE_DIR` parent must be a physical
directory rather than a symlink. On macOS use `/private/tmp/...`, not the
symlinked `/var/...`; Hugging Face Jobs should use `/work/...`.

## Build command

Use a closed encoder-runtime identity, committed config, local pinned Carbon
runtime, and an explicit UTC nanosecond timestamp. Cache production always
constructs Carbon with `normalize=False`; normalization remains a consumer-side
view.

For the corrected Carbon runtime, the identity file is independent of any
predictor, action encoder, training run, calibration, or evaluation release:

```json
{"model_id":"/carbon","revision":"5d31d59b3c845b288a13aedb1358934196852eec","runtime_hash":"sha256:add3c1a663a35fb92fbd3fd935b067da1aed8aeb143ea01f7d92c2cd3ed2aa5e","schema_version":"1.0.0","state_contract_version":"l2_normalized_v2"}
```

The object is closed: unknown fields are rejected. `runtime_hash` is always
required; `weights_hash` is optional for corrected L2 runtimes and required for
`legacy_raw_v1`. `revision` must be an exact lowercase 40-character hexadecimal
commit SHA. Short SHAs, tags, uppercase hashes, and floating refs such as `main`
are rejected; the runtime and optional weight digests bind the actual bytes.

```bash
geno-lewm-cache-windows \
  --cache-dir /work/cache \
  --requests-jsonl /inputs/cache-requests.jsonl \
  --evidence-dir /work/evidence \
  --encoder-runtime-identity /inputs/model/encoder-runtime-identity.json \
  --carbon-model-dir /carbon \
  --config configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml \
  --created-at-ns 1783965600000000000 \
  --batch-size 8 \
  --rows-per-shard 1024 \
  --device cuda \
  --hardware "NVIDIA H200 141GB; CUDA 12.8; single GPU" \
  --run-id cache-proof-example
```

The CLI captures the request, config, and runtime-identity files once, then parses,
validates, stages, and builds from those same immutable bytes. It verifies the
local Carbon runtime hash and, when declared, its weight hash. The model id,
exact revision, and state-contract version must match the resolved encoder
configuration. In particular, `l2_normalized_v2` requires the full corrected
runtime identity, not merely matching weight bytes. Newly encoded paths use a namespace derived from the
complete immutable plan identity: request bytes, resolved logical rows,
encoder/runtime identity, fixed timestamp, batch and shard sizes,
hardware/device, resolved config, and staged input identities. Distinct plans
therefore coexist even when they use the same request bytes. Logical keys
already present in the shared schema-3 index are fully inspected and reported
as reused rows rather than resume-owned rows.

## Resume and evidence

The evidence directory contains:

- `cache_build_requests.jsonl`: the exact input bytes;
- `cache_build_plan.json`: the exact plan rederived from immutable requests,
  pooling identities, batch size, hardware/device identity, resolved-config
  identity, shard size, and fixed creation timestamp;
- `cache_build_state.json`: evidence-owned shard byte identities and wall time
  measured strictly inside `encoder.encode_batch`, atomically updated after each
  verified shard, plus the sealed completion invocation used to replay the
  report;
- `resolved_config.json`: canonical resolved configuration after CLI overrides;
- `encoder_runtime_identity.json`: canonical closed Carbon runtime identity;
- `cache_build_report.json`: counts, request-scoped logical index mappings,
  immutable shard identities, narrowly labeled timing, event contract, and
  explicit claim boundary;
- `inputs/encoder_config.yaml` and
  `inputs/encoder_runtime_identity_source.json`: exact CLI contract inputs
  copied into the bundle;
- `SHA256SUMS`: checksum closure over the exact fixed evidence inventory above.

Evidence traversal, capture, no-clobber installation, atomic replacement,
inventory, and checksum hashing use no-follow directory descriptors. Every held
parent and final filename is rebound after I/O, so parent swaps, same-directory
replacement, and symlink races fail closed without overwriting an outside
target. The plan is validated or installed before caller-provided artifacts are staged.
Unexpected files, directories, symlinks, logs, or report copies in the evidence
tree are rejected rather than dynamically added to `SHA256SUMS`; unsafe or
case-aliased input artifact names are rejected. `--log-dir` and `--json-report`
must remain outside `--evidence-dir` after parent traversal, symlink resolution,
and portable case folding. The builder performs no writes after installing and
verifying `SHA256SUMS`; after an external JSON report write, the CLI immediately
re-verifies the closed bundle before returning success.

On resume, every existing planned shard is opened without following symlinks,
hashed and fully decoded through the same held file descriptor, compared
row-for-row with the plan, and re-indexed without encoding. The builder then
resolves every requested logical key, inspects referenced shared shards one at
a time, and encodes only true misses. Evidence-owned verified rows are reported
as `resumed_rows`; equivalent logical winners from other plans are reported as
`reused_rows`. It retains row/provenance metadata, not all decoded embedding
vectors. Any missing shard named as complete, digest
drift, schema drift, metadata drift, fixed-time drift, changed batch/hardware/
device/resolved config, or noncanonical recovered partition fails before
`encode_batch` processes missing work. Only the precise serialized
logical-key-reservation race can be recovered after encoding, and only after an
equivalent winner plus any evidence-owned planned path/state are reverified;
all other cache corruption remains fatal and cannot seal an `ok: true` report.
The completed report is not trusted merely because its checksum was recomputed:
replay reconstructs the complete deterministic payload from the immutable plan,
durable state, and freshly resolved cache. Invocation elapsed time and run id
are accepted only from the narrowly typed completion record in state.

The report deliberately excludes the byte identity of the shared mutable
`embeddings/index.sqlite`. It instead binds each requested key to an immutable
shard, row offset, and shard digest after validating the current strict index.
Unrelated legitimate cache growth therefore does not invalidate a completed
finite-build report.

Progress uses the registered JSONL events `data.cache.build.start`,
`data.cache.build.progress`, `data.shard.write`, and `data.cache.build.end`.

## Claim boundary

A successful report proves byte-level completion of the exact finite request
artifact under the recorded encoder/cache identities. It does **not** prove a
10% Carbon corpus build, completion within 24 hours, model quality, biological
validity, or clinical validity. Those require separate hardware and scientific
evidence.

Training cache requests must mirror the consumer's pooling identity. The
corrected trainer supplies each edit's `rel_pos`, so its source-state lookups are
`centered_mean` with an edit-conditioned `center_token`; an `edit_locus: null`
global-mean artifact will not satisfy those lookups.
