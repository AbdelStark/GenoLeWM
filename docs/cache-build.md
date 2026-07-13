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

## Build command

Use an exact model manifest, committed config, local pinned Carbon runtime, and
an explicit UTC nanosecond timestamp. Cache production always constructs Carbon
with `normalize=False`; normalization remains a consumer-side view.

```bash
geno-lewm-cache-windows \
  --cache-dir /work/cache \
  --requests-jsonl /inputs/cache-requests.jsonl \
  --evidence-dir /work/evidence \
  --model-manifest /inputs/model/manifest.json \
  --carbon-model-dir /carbon \
  --config configs/reproducibility/train-carbon-500m-snv-baseline-500.yaml \
  --created-at-ns 1783965600000000000 \
  --batch-size 8 \
  --rows-per-shard 1024 \
  --device cuda \
  --run-id cache-proof-example
```

The request SHA-256 is part of the cache namespace, so independently planned
finite slices cannot reuse the same shard paths accidentally.

## Resume and evidence

The evidence directory contains:

- `cache_build_requests.jsonl`: the exact input bytes;
- `cache_build_plan.json`: deterministic keys, aliases, shard paths, and fixed
  creation timestamp committed before the first forward pass;
- `cache_build_state.json`: per-shard byte identities and measured encoder time,
  atomically updated after each verified shard;
- `cache_build_report.json`: counts, cache/index/shard identities, throughput,
  event contract, and explicit claim boundary;
- `inputs/encoder_config.yaml` and `inputs/model_manifest.json`: exact CLI
  contract inputs copied into the bundle;
- `SHA256SUMS`: checksum closure over every regular evidence file above.

On resume, every existing planned shard is opened without following symlinks,
hashed and fully decoded through the same held file descriptor, compared
row-for-row with the plan, and re-indexed without encoding. Any missing shard
named as complete, digest drift, schema drift, metadata drift, or fixed-time
drift fails before `encode_batch` processes missing work.

Progress uses the registered JSONL events `data.cache.build.start`,
`data.cache.build.progress`, `data.shard.write`, and `data.cache.build.end`.

## Claim boundary

A successful report proves byte-level completion of the exact finite request
artifact under the recorded encoder/cache identities. It does **not** prove a
10% Carbon corpus build, completion within 24 hours, model quality, biological
validity, or clinical validity. Those require separate hardware and scientific
evidence.
