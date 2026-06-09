# Quickstart

## Install

```bash
python -m pip install geno-lewm
```

Optional extras:

```bash
python -m pip install "geno-lewm[eval]"
python -m pip install "geno-lewm[train]"
```

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

The model directory must contain a verified GenoLeWM model package.

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

Manifest-backed planning uses local model artifacts. Sequence-proxy mode
is a development smoke path, not learned-model evidence.

## Public Results

The public v0.2.1 bundle contains benchmark, rollout, planning, and
paper evidence:

- <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>
- <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md>

Current evidence is mixed or negative versus Carbon. Do not treat the
package as clinical software or as a broad model-quality result.
