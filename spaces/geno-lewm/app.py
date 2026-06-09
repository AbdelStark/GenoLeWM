# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Space for the public GenoLeWM research demo."""

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
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

RUN_REPO = "abdelstark/geno-lewm-runs"
MODEL_REPO = "abdelstark/geno-lewm"
DATA_REPO = "abdelstark/geno-lewm-data"
RUN_PREFIX = "geno-lewm-v021-strong-4f36eef-10k-r1"
CARBON_REPO = "HuggingFaceBio/Carbon-500M"
CARBON_REVISION = "5d31d59b3c845b288a13aedb1358934196852eec"
SPACE_CACHE = Path(os.getenv("HF_HOME", "/tmp/huggingface")) / "geno_lewm_space"
DEFAULT_VARIANT = "chrSynthetic:3073:A:T"
DEFAULT_WINDOW_START_BP = 0

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
  No clinical utility claim. Current benchmark evidence is mixed or negative
  versus Carbon on most rows. Use this Space as a research demo and artifact
  browser, not as a diagnostic or deployment system.
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
Carbon work by searching in latent space with the predictor.
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
        negative = payload.get("negative_findings", [])
        findings = "\n".join(f"- {item}" for item in negative if isinstance(item, str))
        summary = (
            f"Generated by `{payload.get('generated_by')}` on `{payload.get('hardware')}`.\n\n"
            f"{findings}"
        )
        return rows or STATIC_METRIC_ROWS, summary
    except Exception as exc:
        return STATIC_METRIC_ROWS, (
            "Live metric fetch failed; showing the pinned README summary.\n\n"
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
            "The planning demo ran the released manifest-backed model path. "
            "It is execution evidence, not useful-planning evidence."
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


def _sha256_text(text: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_with_public_carbon(model_dir: Path) -> tuple[Path, str]:
    """Create an operational runtime copy that resolves Carbon from the Hub cache."""
    snapshot_download(repo_id=CARBON_REPO, revision=CARBON_REVISION, repo_type="model")
    canonical = Path("/carbon")
    if canonical.exists():
        return model_dir, "canonical /carbon path exists"

    try:
        canonical.symlink_to(Path(snapshot_download(repo_id=CARBON_REPO, revision=CARBON_REVISION)))
        return model_dir, "created canonical /carbon symlink"
    except OSError:
        pass

    target = SPACE_CACHE / "models" / f"{model_dir.name}-carbon-hub-runtime"
    target.mkdir(parents=True, exist_ok=True)
    for path in model_dir.iterdir():
        if path.name not in {"manifest.json", "training_config.effective.yaml"}:
            _link_or_copy(path, target / path.name)

    config = yaml.safe_load((model_dir / "training_config.effective.yaml").read_text())
    config["encoder"]["model_id"] = CARBON_REPO
    config["encoder"]["revision"] = CARBON_REVISION
    config_text = yaml.safe_dump(config, sort_keys=False)
    (target / "training_config.effective.yaml").write_text(config_text, encoding="utf-8")

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["encoder"]["id"] = CARBON_REPO
    manifest["training"]["hash"] = _sha256_text(config_text)
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return target, (
        "remapped encoder path to HuggingFaceBio/Carbon-500M; artifact weight hashes "
        "are unchanged, but the runtime manifest id differs from the released manifest"
    )


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


def score_single_variant(
    profile_name: str,
    variant: str,
    window: str,
    window_start_bp: int | float | str,
    backend: str,
    resolve_carbon_from_hub: bool,
) -> tuple[str, dict[str, Any]]:
    try:
        from geno_lewm.action import EditSpec
        from geno_lewm.deploy import GenoLeWMRuntime

        chrom, pos, ref, alt, normalized_window, start_bp, preflight = _prepare_scoring_inputs(
            variant,
            window,
            window_start_bp,
        )
        edit = EditSpec(chrom=chrom, pos=pos, ref=ref, alt=alt)
        model_dir = _materialize_model(profile_name)
        runtime_note = "using manifest paths as published"
        runtime_dir = model_dir
        if resolve_carbon_from_hub:
            runtime_dir, runtime_note = _runtime_with_public_carbon(model_dir)
        receipt_path = SPACE_CACHE / "last_single_variant_receipt.json"
        runtime = GenoLeWMRuntime(runtime_dir, backend=backend)
        result = runtime.score_variant(
            edit,
            window=normalized_window,
            window_start_bp=start_bp,
            receipt_path=receipt_path,
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        payload["receipt_path"] = str(receipt_path)
        payload["runtime_note"] = runtime_note
        payload["input_preflight"] = preflight
        return "Scored with the trained checkpoint runtime.", payload
    except Exception as exc:
        message = (
            "Scoring did not complete. If this is an input error, fix the variant/window "
            "pair and retry. Otherwise the Space runtime may still be downloading "
            "Carbon-500M or loading the optional ML stack.\n\n"
            f"`{_exception_line(exc)}`"
        )
        return message, {"trace": _short_trace(exc)}


def _prepare_scoring_inputs(
    variant: str,
    window: str,
    window_start_bp: int | float | str,
) -> tuple[str, int, str, str, str, int, dict[str, Any]]:
    chrom, pos, ref, alt = _parse_variant_text(variant)
    start_bp = _coerce_window_start_bp(window_start_bp)
    normalized_window = _normalize_window_text(window)
    rel_pos = pos - 1 - start_bp
    if rel_pos < 0 or rel_pos + len(ref) > len(normalized_window):
        raise ValueError(
            "variant is outside the supplied reference window "
            f"(relative offset {rel_pos}, window length {len(normalized_window)})"
        )
    observed_ref = normalized_window[rel_pos : rel_pos + len(ref)]
    if observed_ref != ref:
        raise ValueError(
            "reference base mismatch before scoring: "
            f"variant REF={ref!r}, observed window bases={observed_ref!r}, "
            f"relative offset={rel_pos}. Use a matching FASTA window or correct the REF allele."
        )
    return (
        chrom,
        pos,
        ref,
        alt,
        normalized_window,
        start_bp,
        {
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "window_start_bp": start_bp,
            "relative_offset": rel_pos,
            "observed_ref": observed_ref,
            "window_bp": len(normalized_window),
        },
    )


def _coerce_window_start_bp(raw: int | float | str) -> int:
    if isinstance(raw, bool):
        raise ValueError("window_start_bp must be an integer, not bool")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError("window_start_bp must be an integer")
        value = int(raw)
    elif isinstance(raw, str):
        value = int(raw.strip())
    else:
        raise ValueError(f"window_start_bp must be an integer, got {type(raw).__name__}")
    if value < 0:
        raise ValueError("window_start_bp must be non-negative")
    return value


def _normalize_window_text(raw: str) -> str:
    return "".join(str(raw).split()).upper()


def _parse_variant_text(raw: str) -> tuple[str, int, str, str]:
    parts = raw.strip().split(":")
    if len(parts) != 4:
        raise ValueError("variant must have the form CHROM:POS:REF:ALT")
    chrom, pos_text, ref, alt = parts
    return chrom, int(pos_text), ref.upper(), alt.upper()


def _exception_line(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _short_trace(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=3))


def _example_window() -> str:
    return "ACGT" * 3072


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

The strongest supported claim is systems evidence: the project can train,
package, evaluate, benchmark, publish, and replay a genomic-edit world-model
pipeline with content-addressed evidence. The published metrics do not
establish broad model superiority, clinical utility, privacy assurance, or
deployment readiness.
"""
                )
            with gr.Tab("Results"):
                gr.Markdown(
                    "### Measured v0.2.1 Metrics\n"
                    "<span class='metric-note'>Loaded from the public run tree when "
                    "available; the table falls back to pinned values if the Hub fetch "
                    "fails.</span>"
                )
                load_button = gr.Button("Load public metrics", variant="primary")
                table = gr.Dataframe(
                    headers=["Split", "Metric", "Value", "Baseline comparison", "N"],
                    datatype=["str", "str", "str", "str", "str"],
                    value=STATIC_METRIC_ROWS,
                )
                findings = gr.Markdown()
                load_button.click(load_results, outputs=[table, findings])
            with gr.Tab("Planning Demo"):
                gr.Markdown(
                    "### Released-Artifact Planning\n"
                    "This panel reads the published planning-demo artifacts. It is "
                    "execution evidence for the manifest-backed path, not proof of useful "
                    "planning behavior."
                )
                planning_button = gr.Button("Load planning artifact", variant="primary")
                planning_json = gr.JSON(label="Planning artifact summary")
                planning_text = gr.Markdown()
                planning_button.click(load_planning_summary, outputs=[planning_json, planning_text])
            with gr.Tab("Checkpoint"):
                gr.Markdown(
                    "### Trained Checkpoint Path\n"
                    "Download the released model package, verify the trained action "
                    "encoder/predictor can load, then attempt single-variant scoring. "
                    "Full scoring needs Carbon-500M as the frozen state encoder."
                )
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
                gr.Markdown("### Single-Variant Scoring")
                gr.Markdown(
                    "The prefilled example is synthetic and sequence-consistent. "
                    "For real variants, paste the reference window from FASTA; the "
                    "REF allele must match the supplied window at the relative locus."
                )
                with gr.Row():
                    variant = gr.Textbox(
                        label="Variant",
                        value=DEFAULT_VARIANT,
                        placeholder="CHROM:POS:REF:ALT",
                    )
                    window_start = gr.Number(
                        label="Window start bp",
                        value=DEFAULT_WINDOW_START_BP,
                        precision=0,
                    )
                    backend = gr.Dropdown(
                        label="Backend",
                        choices=["auto", "cpu", "cuda"],
                        value="auto",
                    )
                window = gr.Textbox(
                    label="Reference window",
                    value=_example_window(),
                    lines=4,
                    max_lines=8,
                )
                carbon = gr.Checkbox(
                    value=True,
                    label="Resolve Carbon-500M from Hugging Face Hub before scoring",
                )
                score_button = gr.Button("Score variant", variant="primary")
                score_status = gr.Markdown()
                score_json = gr.JSON(label="Score result or runtime issue")
                score_button.click(
                    score_single_variant,
                    inputs=[profile, variant, window, window_start, backend, carbon],
                    outputs=[score_status, score_json],
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

Run a local score once the model directory and Carbon-500M are available:

```bash
geno-lewm-score \\
  --model-dir /path/to/model \\
  --backend auto \\
  --variant {DEFAULT_VARIANT} \\
  --window ACGT... \\
  --window-start-bp {DEFAULT_WINDOW_START_BP} \\
  --receipt receipt.json
```

Every published claim should route back to a JSON/JSONL/Markdown artifact in
the model repo or the `{RUN_PREFIX}` run tree.
"""
                )
    return demo


if __name__ == "__main__":
    build_app().launch()
