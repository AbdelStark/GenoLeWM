# Quickstart

A 5-minute tour of the modules that ship today, plus the public v0.1
release artifacts and v0.2.1 benchmark/planning evidence.

## Install

GenoLeWM requires **Python 3.10+**. Install the alpha package from PyPI
after the `v0.2.1` tag workflow publishes, or install from source. The
implemented surface keeps heavyweight ML dependencies behind optional
extras; the base install is small and includes the CLI scaffolding
dependencies.

=== "PyPI"

    ```bash
    python -m pip install geno-lewm
    ```

=== "uv source install"

    ```bash
    git clone https://github.com/AbdelStark/GenoLeWM.git
    cd GenoLeWM
    uv venv && source .venv/bin/activate
    uv pip install -e "."
    ```

=== "development"

    ```bash
    git clone https://github.com/AbdelStark/GenoLeWM.git
    cd GenoLeWM
    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev,docs]"
    pytest
	```

## Public v0.1 demo evidence

The first public paper/demo release is available as
`geno-lewm-v0.1.0-r1`:

- Model: <https://huggingface.co/abdelstark/geno-lewm>
- Dataset: <https://huggingface.co/datasets/abdelstark/geno-lewm-data>
- Demo assets:
  <https://github.com/AbdelStark/GenoLeWM/releases/tag/geno-lewm-v0.1.0-r1>
- Terminal transcript:
  <https://github.com/AbdelStark/GenoLeWM/releases/download/geno-lewm-v0.1.0-r1/terminal-demo-transcript.md>

The released transcript records the artifact-backed command:

```console
$ geno-lewm-score --quiet --no-banner --model-dir model --backend auto --vcf demo/demo.vcf --fasta demo/chr21.fa.gz --output demo/scores.jsonl --receipt demo/receipts.jsonl --batch-size 64 --no-progress
{"output_path": "/tmp/geno-publish/demo/scores.jsonl", "receipt_path": "/tmp/geno-publish/demo/receipts.jsonl"}
```

It scored 32 VCF records and recorded matching score/receipt JSONL
hashes. This is release replay evidence, not a clinical or broad
model-quality claim.

## Public v0.2.1 benchmark and planning evidence

The v0.2.1 serious-completion bundle records broader benchmark rows and
a released-artifact planning demo:

- Artifact tree:
  <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>
- Benchmark readiness:
  <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/suite/model/v0.2_benchmark_readiness_report.json>
- Planning demo:
  <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1/planning-demo>
- Serious-completion paper:
  <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md>

Those artifacts support a negative-results/systems framing: GenoLeWM
does not broadly beat Carbon, K20 rollout speed remains below the
RFC-0004 target, and the planning demo does not prove useful planning
behavior.

## 1. Edit specs

`EditSpec` is the canonical 1-based VCF-style edit; `RelEdit` is its
window-relative form consumed by the predictor.

```python
from geno_lewm import EditSpec

snv = EditSpec(chrom="chr17", pos=43_091_983, ref="A", alt="T")
print(snv.edit_type)   # EditType.SNV (derived from ref/alt lengths)

# Re-anchor inside an encoder window:
window_start, window_end = 43_091_900, 43_092_100  # 0-based inclusive
rel = snv.relative_to(window_start, window_end)
print(rel.rel_pos)     # 82 (offset inside the window)
```

Validation is strict and typed: bad input raises subclasses of
`GenoLeWMError`. See the [error model](spec/04-error-model.md).

```python
from geno_lewm import InvalidEditError, EditSpec

try:
    EditSpec(chrom="1", pos=1, ref="A", alt="A")  # ref == alt
except InvalidEditError as exc:
    print(exc.code, exc.message)
    # INPUT.INVALID_EDIT  ref and alt must differ
```

## 2. Applying edits

Pure-Python `apply_edit` / `apply_edits` are pre-encoder helpers used
by the trainer and eval pipeline to build the post-edit reference
string for `s_{t+1}`.

```python
from geno_lewm import RelEdit, EditType, apply_edit, apply_edits

window = "ACGTACGTACGT"
edit = RelEdit(rel_pos=2, edit_type=EditType.SNV, ref_bases="G", alt_bases="C")
apply_edit(window, edit)  # 'ACCTACGTACGT'

# Multi-edit: order-invariant after the internal sort (right-to-left).
edits = [
    RelEdit(rel_pos=0, edit_type=EditType.SNV, ref_bases="A", alt_bases="T"),
    RelEdit(rel_pos=4, edit_type=EditType.SNV, ref_bases="A", alt_bases="C"),
]
apply_edits(window, edits)  # 'TCGTCCGTACGT'
```

## 3. Structured logging with privacy redaction

The logger emits **JSONL** records with a per-event allowlist; payload
keys that fail the four redaction rules (allowlist / type / DNA
pattern / personal-data deny list) are dropped, and in strict mode
(default) the bypass raises.

```python
import os, tempfile
from geno_lewm import get_logger

with tempfile.TemporaryDirectory() as d:
    os.environ["GENO_LEWM_LOG_DIR"] = d
    log = get_logger("inference", run_id="run-quickstart")
    log.info("inference.batch.end", n=10, batch_id="b-1", throughput_per_s=87.2)
    # `sample_id` would be denied even with an allowlist hit:
    log.info("inference.batch.end", n=10, sample_id="alice")  # strict-mode error
```

Pretty-format on a TTY, JSONL on a pipe. Environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GENO_LEWM_LOG_DIR` | `~/.geno-lewm/logs` | sink directory |
| `GENO_LEWM_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |
| `GENO_LEWM_LOG_FORMAT` | autodetect | `pretty` / `json` |
| `GENO_LEWM_REDACTION_STRICT` | `1` | `0` to soft-drop instead of raise |

## 4. Checksum provenance

The provenance layer ships the manifest schema, content-addressed
hashing, and the input / output commitment helpers. Receipts are
deterministic and byte-stable on disk (canonical JSON).

```python
from geno_lewm import (
    EditSpec, PoolingConfig, DtypeConfig,
    compute_input_commitment,
)

edit = EditSpec(chrom="1", pos=10, ref="A", alt="T")
pool = PoolingConfig(state_layer=12, pool_type="centered_mean", pool_radius=64, normalize=True)
dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")

window = "ACGT" * 64  # toy 256-bp window
commit = compute_input_commitment(window, edit, pool, dtype)
print(commit)  # 'sha256:<64-hex>'
```

## 5. The verify CLI

`geno-lewm-verify` validates an inference receipt against the
producing manifest. Three checks (receipt schema, manifest hash,
output commitment) are always run; the input commitment is
recomputed when the CLI is supplied with the original edit + pool +
dtype flags.

```console
$ geno-lewm-verify path/to/receipt.json --manifest path/to/manifest.json
reading receipt:  path/to/receipt.json
  schema_version=1.0.0 provenance.kind=checksum_only
reading manifest: path/to/manifest.json
  model_id ok (sha256:0123456789abcdef0…)
  input_commitment: skipped (no input flags supplied)
  output_commitment ok (sha256:fedcba9876543210…)
ok
```

Exit codes follow the [error model](spec/04-error-model.md): `0` =
verified, `8` = receipt/provenance mismatch, etc.

## What's next

- **[Spec corpus](spec/index.md)** — the canonical contract.
- **[API reference](reference/index.md)** — generated from docstrings.
- **[Roadmap](roadmap/IMPLEMENTATION.md)** — what's landing next.
