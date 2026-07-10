# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Space for GenoLeWM artifact and checkpoint inspection."""

from __future__ import annotations

import json
import os
import shutil
import traceback
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download

RUN_REPO = "abdelstark/geno-lewm-runs"
MODEL_REPO = "abdelstark/geno-lewm"
DATA_REPO = "abdelstark/geno-lewm-data"
RUN_PREFIX = "geno-lewm-v021-strong-4f36eef-10k-r1"
SPACE_CACHE = Path(os.getenv("HF_HOME", "/tmp/huggingface")) / "geno_lewm_space"

CHECKPOINTS: dict[str, dict[str, Any]] = {
    "v0.2.1 serious-completion checkpoint": {
        "repo": RUN_REPO,
        "prefix": f"{RUN_PREFIX}/suite/model",
        "target": "v021-suite-model",
        "release": "geno-lewm-v0.2.1-r1",
        "files": (
            "manifest.json",
            "action_encoder.safetensors",
            "predictor.safetensors",
            "calibration.parquet",
            "eval_report.md",
            "training_config.effective.yaml",
        ),
    },
    "v0.1 public release checkpoint": {
        "repo": MODEL_REPO,
        "prefix": "",
        "target": "v010-model",
        "release": "geno-lewm-v0.1.0-r1",
        "files": (
            "manifest.json",
            "action_encoder.safetensors",
            "predictor.safetensors",
            "calibration.parquet",
            "eval_report.md",
            "training_config.effective.yaml",
        ),
    },
}

STATIC_METRIC_ROWS = [
    ["ClinVar coding", "AUROC", "0.734375", "Carbon delta -0.1875", "16"],
    ["ClinVar coding", "Average precision", "0.852976", "Carbon delta -0.098947", "16"],
    ["ClinVar coding", "Balanced accuracy", "0.75", "Carbon delta +0.0625", "16"],
    ["ClinVar non-coding", "AUROC", "0.5625", "Carbon delta -0.3125", "16"],
    ["ClinVar non-coding", "Average precision", "0.605456", "Carbon delta -0.308967", "16"],
    ["ClinVar non-coding", "Balanced accuracy", "0.4375", "Carbon delta -0.25", "16"],
    ["BRCA2 saturation", "Spearman rho", "0.149194", "Carbon delta -0.327713", "32"],
    ["TraitGym Mendelian", "Spearman rho", "-0.027965", "Carbon delta +0.055929", "32"],
    [
        "Phased-haplotype rollout",
        "Cosine mean",
        "0.288861",
        "Source-state delta -0.708970",
        "8",
    ],
    [
        "Synthetic edit-chain rollout",
        "Cosine mean",
        "0.301608",
        "Source-state delta -0.689631",
        "8",
    ],
]

CSS = """
:root {
  --ink: #17201c;
  --muted: #58645f;
  --line: #d7ded8;
  --paper: #f7f8f3;
  --panel: #ffffff;
  --accent: #1f7a4d;
  --warn: #a15c21;
}
.gradio-container {
  background: var(--paper);
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.hero {
  border-top: 8px solid var(--ink);
  padding: 28px 0 18px;
}
.kicker {
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.hero h1 {
  color: var(--ink);
  font-size: clamp(2.1rem, 4vw, 4.8rem);
  line-height: 0.92;
  margin: 8px 0 14px;
  letter-spacing: 0;
}
.hero p {
  color: var(--muted);
  font-size: 1.04rem;
  max-width: 980px;
}
.callout {
  border-left: 5px solid var(--warn);
  background: #fff8ef;
  padding: 12px 16px;
  color: #3c2a18;
}
.metric-note {
  color: var(--muted);
  font-size: 0.92rem;
}
.field-guide {
  border: 1px solid var(--line);
  background: #fbfcf7;
  padding: 14px 16px;
}
.field-guide h3 {
  margin-top: 0;
}
.field-guide li {
  margin: 0.35rem 0;
}
.link-row a {
  color: var(--accent);
  font-weight: 700;
  text-decoration: none;
}
button.primary {
  border-radius: 4px !important;
}
"""


def _space_header() -> str:
    return """
<section class="hero">
  <div class="kicker">Genomic edit world model, released artifacts only</div>
  <h1>GenoLeWM</h1>
  <p>
    GenoLeWM treats a genomic edit as an action in a latent state space:
    Carbon encodes the reference state, an action encoder embeds the edit,
    and a predictor estimates the post-edit state. This console shows the
    actual published evidence and exposes the trained checkpoint path.
  </p>
</section>
<div class="callout">
  No clinical utility claim. Every published checkpoint uses
  <code>legacy_raw_v1</code>: raw Carbon targets were combined with
  unit-normalized predictions. Training also mixed global source pooling with
  edit-centered targets; every historical centered pool was one hidden token
  left of the intended DNA token because the leading <code>&lt;dna&gt;</code>
  token was omitted. The pinned Carbon tokenizer also made an unpinned,
  network-capable <code>Qwen/Qwen3-4B-Base</code> lookup. These defects invalidate L2 residual, VEP/calibration, and
  planning-objective interpretations. Cosine values remain historical and
  confounded. Legacy scientific scoring is disabled; use this Space only for
  artifact and checkpoint inspection.
</div>
"""


@lru_cache(maxsize=16)
def _download_text(repo_id: str, filename: str) -> str:
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    return Path(path).read_text(encoding="utf-8")


@lru_cache(maxsize=16)
def _download_json(repo_id: str, filename: str) -> dict[str, Any]:
    return json.loads(_download_text(repo_id, filename))


def _artifact_markdown() -> str:
    try:
        manifest = _download_json(MODEL_REPO, "manifest.json")
        model_sha = manifest.get("predictor", {}).get("hash", "unavailable")
        release = manifest.get("release_id", "unavailable")
    except Exception:
        model_sha = "unavailable"
        release = "unavailable"
    return f"""
### Public Artifacts

| Artifact | Link |
| --- | --- |
| Python package | [geno-lewm 0.2.1](https://pypi.org/project/geno-lewm/0.2.1/) |
| Source release | [GitHub v0.2.1](https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1) |
| Model repo | [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) |
| Dataset repo | [{DATA_REPO}](https://huggingface.co/datasets/{DATA_REPO}) |
| v0.2.1 run tree | [{RUN_PREFIX}](https://huggingface.co/{RUN_REPO}/tree/main/{RUN_PREFIX}) |
| Generated paper | [paper.serious-completion.md](https://huggingface.co/{RUN_REPO}/resolve/main/{RUN_PREFIX}/paper/paper.serious-completion.md) |

Current public model release in `{MODEL_REPO}`: `{release}`.
Predictor artifact hash: `{model_sha}`.
"""


def _architecture_markdown() -> str:
    return """
### Model Flow

```text
reference window --Carbon encoder--> s_t
edit spec --------action encoder----> a_t
(s_t, a_t) -------predictor---------> s_hat_{t+1}
edited window ----Carbon encoder----> s_{t+1}
loss = distance(s_hat_{t+1}, s_{t+1}) + collapse regularization
```

The trained GenoLeWM components are the action encoder and predictor.
Carbon-500M is frozen and used as the state encoder. Planning amortizes
Carbon work by searching in latent space with the predictor. That is the
intended architecture, not a capability established by the published legacy
checkpoints.
"""


def _checkpoint_intro_markdown() -> str:
    return """
### Inspect Published Checkpoints

This panel downloads the manifest-listed action encoder, predictor,
calibration table, and config. It can verify that the trainable module weights
load. It does not run scientific scoring.

Both published checkpoint choices use `legacy_raw_v1`. Their artifacts remain
available for provenance, compatibility, and implementation audit, but their
residual scores do not evaluate the intended `l2_normalized_v2` method. The
v0.2.1 Phase 2 KL also supplied no gradient to the trainable modules. Current
source repairs for local pure-DNA tokenization and token-layout-aware centers
are code contracts only, not corrected model-quality evidence.

"""


def _results_guide_markdown() -> str:
    return """
<div class="field-guide">
<h3>How to read the historical table</h3>
<ul>
  <li><strong>Value</strong> is a reproducible output from the public v0.2.1
  <code>legacy_raw_v1</code> run, not a corrected-method result.</li>
  <li><strong>Recorded delta</strong> preserves the released arithmetic for
  audit. It does not establish superiority or inferiority. Residual/VEP values
  are invalid; rollout cosine values are confounded by invalid training.</li>
  <li><strong>N</strong> is the proof-scale row count for the reported slice;
  it is not a broad population sample.</li>
  <li>The 2026-07-10 validity correction supersedes the run's earlier positive
  and negative model interpretation.</li>
</ul>
</div>
"""


def _planning_guide_markdown() -> str:
    return """
<div class="field-guide">
<h3>What the planning artifact proves</h3>
<ul>
  <li>It proves the released manifest-backed legacy path executed against the
  published artifacts.</li>
  <li>Its L2 objective compared incompatible state scales, so
  <code>best_distance</code> is invalid as a planning objective value.</li>
  <li>It does not establish edit-selection or genomic-design capability.</li>
  <li>Use the JSON panel to inspect the command output and manifest summary,
  then compare any planning claim against the generated paper and run tree.</li>
</ul>
</div>
"""


def _metric_rows_from_payload(payload: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for metric in payload.get("metrics", []):
        if not isinstance(metric, Mapping):
            continue
        split = str(metric.get("split", "unknown"))
        name = str(metric.get("name", "unknown"))
        value = _format_number(metric.get("value"))
        n_value = str(metric.get("n", ""))
        baseline = metric.get("baseline")
        delta = metric.get("delta_vs_baseline")
        comparison = "no baseline"
        if baseline is not None and delta is not None:
            comparison = f"{baseline} delta {_format_signed(delta)}"
        rows.append([split, name, value, comparison, n_value])
    return rows


def _format_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{float(value):.6g}"


def _format_signed(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{float(value):+.6g}"


def load_results() -> tuple[list[list[str]], str]:
    try:
        payload = _download_json(
            RUN_REPO,
            f"{RUN_PREFIX}/suite/model/eval_metrics.v02.json",
        )
        rows = _metric_rows_from_payload(payload)
        summary = (
            f"Generated by `{payload.get('generated_by')}` on `{payload.get('hardware')}`.\n\n"
            "These are historical `legacy_raw_v1` implementation outputs. "
            "The released L2/VEP values are invalid, and cosine values are "
            "confounded; the recorded deltas support neither superiority nor "
            "inferiority claims."
        )
        return rows or STATIC_METRIC_ROWS, summary
    except Exception as exc:
        return STATIC_METRIC_ROWS, (
            "Live metric fetch failed; showing pinned historical outputs. The "
            "2026-07-10 validity correction still applies.\n\n"
            f"`{_exception_line(exc)}`"
        )


def load_planning_summary() -> tuple[dict[str, Any], str]:
    try:
        summary = _download_json(
            RUN_REPO,
            f"{RUN_PREFIX}/planning-demo/plan.stdout.json",
        )
        manifest = _download_json(
            RUN_REPO,
            f"{RUN_PREFIX}/planning-demo/planning_demo_manifest.json",
        )
        text = (
            "The planning artifact records execution of the manifest-backed "
            "`legacy_raw_v1` path. Its mixed-scale L2 objective is invalid, so "
            "the output is not planning-capability evidence."
        )
        return {"stdout": summary, "manifest_summary": manifest.get("summary", manifest)}, text
    except Exception as exc:
        return {}, f"Planning artifact fetch failed: `{_exception_line(exc)}`"


def _profile_paths(profile_name: str) -> tuple[dict[str, Any], list[str]]:
    profile = CHECKPOINTS[profile_name]
    prefix = profile["prefix"]
    files = list(profile["files"])
    paths = [f"{prefix}/{name}" if prefix else name for name in files]
    return profile, paths


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def _materialize_model(profile_name: str) -> Path:
    profile, allow_patterns = _profile_paths(profile_name)
    snapshot = Path(
        snapshot_download(
            repo_id=profile["repo"],
            repo_type="model",
            allow_patterns=allow_patterns,
        )
    )
    target = SPACE_CACHE / "models" / profile["target"]
    target.mkdir(parents=True, exist_ok=True)
    for name, source_name in zip(profile["files"], allow_patterns, strict=True):
        _link_or_copy(snapshot / source_name, target / name)
    return target


def inspect_checkpoint(profile_name: str, load_weights: bool) -> tuple[str, dict[str, Any]]:
    try:
        model_dir = _materialize_model(profile_name)
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        payload: dict[str, Any] = {
            "local_model_dir": str(model_dir),
            "release_id": manifest.get("release_id"),
            "model_version": manifest.get("model_version"),
            "encoder": manifest.get("encoder"),
            "predictor": manifest.get("predictor"),
            "action_encoder": manifest.get("action_encoder"),
            "calibration": manifest.get("calibration"),
        }
        status = "Downloaded manifest-required checkpoint files."
        if load_weights:
            from geno_lewm.deploy.runtime import load_action_predictor_modules

            load_action_predictor_modules(model_dir)
            status = "Downloaded and loaded action encoder + predictor weights."
        return status, payload
    except Exception as exc:
        return f"Checkpoint inspection failed: `{_exception_line(exc)}`", {
            "trace": _short_trace(exc)
        }


def _exception_line(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _short_trace(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=3))


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="GenoLeWM") as demo:
        gr.HTML(_space_header())
        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.Markdown(_architecture_markdown())
                gr.Markdown(_artifact_markdown())
                gr.Markdown(
                    """
### Claim Boundary

The supported claim is systems evidence: the project published and can replay
the legacy genomic-edit world-model pipeline with content-addressed artifacts.
The published metrics do not evaluate the intended normalized method and do
not establish model quality, clinical utility, privacy assurance, or
deployment readiness.
"""
                )
            with gr.Tab("Results"):
                gr.Markdown(
                    "### Historical v0.2.1 Outputs\n"
                    "<span class='metric-note'>Loaded from the public run tree when "
                    "available. These legacy outputs are shown for audit, not scientific "
                    "interpretation.</span>"
                )
                gr.Markdown(_results_guide_markdown())
                load_button = gr.Button("Load public metrics", variant="primary")
                table = gr.Dataframe(
                    headers=["Split", "Metric", "Value", "Recorded delta", "N"],
                    datatype=["str", "str", "str", "str", "str"],
                    value=STATIC_METRIC_ROWS,
                )
                findings = gr.Markdown()
                load_button.click(load_results, outputs=[table, findings])
            with gr.Tab("Planning Demo"):
                gr.Markdown(
                    "### Historical Planning Artifact\n"
                    "This panel reads the published legacy planning artifacts. The "
                    "mixed-scale objective is invalid; the panel is for execution and "
                    "provenance inspection only."
                )
                gr.Markdown(_planning_guide_markdown())
                planning_button = gr.Button("Load planning artifact", variant="primary")
                planning_json = gr.JSON(label="Planning artifact summary")
                planning_text = gr.Markdown()
                planning_button.click(load_planning_summary, outputs=[planning_json, planning_text])
            with gr.Tab("Checkpoint"):
                gr.Markdown(_checkpoint_intro_markdown())
                profile = gr.Dropdown(
                    choices=list(CHECKPOINTS),
                    value="v0.2.1 serious-completion checkpoint",
                    label="Checkpoint",
                )
                with gr.Row():
                    load_weights = gr.Checkbox(
                        value=True,
                        label="Load action encoder + predictor weights",
                    )
                    inspect_button = gr.Button("Inspect checkpoint", variant="primary")
                inspect_status = gr.Markdown()
                inspect_json = gr.JSON(label="Checkpoint metadata")
                inspect_button.click(
                    inspect_checkpoint,
                    inputs=[profile, load_weights],
                    outputs=[inspect_status, inspect_json],
                )
                gr.Markdown(
                    "### Legacy scientific scoring disabled\n"
                    "Published checkpoints mix raw source/target states with "
                    "unit-normalized predictions. Running their residual scorer would "
                    "produce invalid scientific output, so this Space exposes no score "
                    "action. A scoring UI can return only after a fresh "
                    "`l2_normalized_v2` checkpoint and calibration lineage are published."
                )
            with gr.Tab("Reproduce"):
                gr.Markdown(
                    f"""
### Local Commands

```bash
python -m pip install "geno-lewm[train,eval]==0.2.1"
```

Download artifacts:

```python
from huggingface_hub import snapshot_download

snapshot_download("{MODEL_REPO}")
snapshot_download("{RUN_REPO}", allow_patterns="{RUN_PREFIX}/suite/model/*")
```

Verify a historical receipt against its manifest locally:

```bash
geno-lewm-verify /path/to/receipt.json --manifest /path/to/model/manifest.json
```

This verifies artifact identity, not scientific validity. Every historical
output should route back to a JSON/JSONL/Markdown artifact in the model repo or
the `{RUN_PREFIX}` run tree. New scientific claims require a fresh
`l2_normalized_v2` lineage.
"""
                )
    return demo


if __name__ == "__main__":
    build_app().launch()
