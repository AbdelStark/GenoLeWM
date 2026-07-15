# Quickstart

> **Published-checkpoint notice:** v0.1 and v0.2.1 use
> `legacy_raw_v1`. Their raw Carbon targets were compared with
> unit-normalized predictions. Training sources were globally pooled while
> targets were edit-centered, but every historical centered pool was one hidden
> token too far left because the leading `<dna>` token was omitted from the
> coordinate conversion. Cache v1 omitted the pooling center. The pinned
> Carbon tokenizer also made an unpinned, network-capable
> `Qwen/Qwen3-4B-Base` lookup, so the historical runtime was not self-contained. The
> v0.2.1 Phase 2 KL supplied no gradient
> to trainable parameters. Published residual, VEP, and planning scores are
> invalid as evidence for the intended `l2_normalized_v2` method. Use those
> packages only for historical artifact replay.

The source checkout now uses a local pure-DNA tokenizer and resolves the edit
center from Carbon's validated token layout. These are runtime-contract fixes.
They do not make a published legacy checkpoint scientifically valid and do not
constitute model-quality evidence.

## Install

```bash
python -m pip install geno-lewm
```

Optional extras:

```bash
python -m pip install "geno-lewm[eval]"
python -m pip install "geno-lewm[train]"
```

The wheel-installed calibration and evaluation aggregation commands run
without the repository's source-only `tools` package. Training release-run
assembly via `geno-lewm-train --package-release-run` is wheel-contained too.

Source checkout:

```bash
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,docs]"
```

## Edit Specs

```python
from geno_lewm import EditSpec

edit = EditSpec(chrom="chr17", pos=43_091_983, ref="A", alt="T")
rel = edit.relative_to(window_start_bp=43_091_900, window_end_bp=43_092_100)
print(edit.edit_type, rel.rel_pos)
```

Bad inputs raise typed `GenoLeWMError` subclasses with stable error
codes.

## Apply Edits

```python
from geno_lewm import EditType, RelEdit, apply_edits

window = "ACGTACGTACGT"
edited = apply_edits(
    window,
    [
        RelEdit(rel_pos=0, edit_type=EditType.SNV, ref_bases="A", alt_bases="T"),
        RelEdit(rel_pos=4, edit_type=EditType.SNV, ref_bases="A", alt_bases="C"),
    ],
)
print(edited)
```

## Verify A Receipt

```bash
geno-lewm-verify examples/data/verify_receipt/receipt.json \
  --manifest examples/data/verify_receipt/manifest.json
```

## Fixture Training Smoke

```bash
geno-lewm-train --fixture-smoke --run-dir /tmp/geno-lewm-smoke --steps 50
```

This is a CI/development contract. It is not model-quality evidence.

## Score A Local VCF

```bash
geno-lewm-score \
  --model-dir /path/to/model \
  --backend auto \
  --vcf variants.vcf \
  --fasta reference.fa.gz \
  --output scores.jsonl \
  --receipt receipts.jsonl \
  --batch-size 64 \
  --no-progress
```

The model directory must contain a verified GenoLeWM model package. Scientific
scoring additionally requires a freshly trained `l2_normalized_v2` lineage;
no such corrected public checkpoint is currently published. Running a
published legacy package reproduces historical implementation output only.

## Plan Edit Sequences

```bash
geno-lewm-plan \
  --model-dir /path/to/model \
  --window-fasta window.fa \
  --target-fasta target.fa \
  --output plan.json \
  --horizon 5 \
  --iterations 5 \
  --samples 1024 \
  --elite 64
```

Manifest-backed planning uses local model artifacts. Planning with a published
`legacy_raw_v1` checkpoint only replays the historical path: its mixed-scale
L2 objective is invalid. Sequence-proxy mode is a development smoke path, not
learned-model evidence.

## Public Results

The public v0.2.1 bundle contains historical benchmark, rollout, planning,
and paper artifacts:

- <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>
- <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md>

The recorded values cannot establish either superiority or inferiority to
Carbon for the intended normalized method. The L2/VEP/planning values are
invalid; cosine values are historical and confounded. Do not treat the
package as scientific scoring, clinical software, or a model-quality result.
