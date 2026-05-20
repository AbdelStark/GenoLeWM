# RFC-0017: Configuration system

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0001, RFC-0012, RFC-0013, RFC-0018
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

GenoLeWM uses a Hydra-based composable YAML configuration system with a
typed schema (Pydantic v2 or dataclass-based) and a single canonical
override syntax. This RFC specifies the configuration namespaces, the
override precedence, the schema-versioning policy, the secret-handling
rules, and the discovery mechanism (`--print-config`).

## 2. Motivation

A scientific ML project needs configs that are:

- **Composable.** Sweeping over `encoder=carbon-500m` vs
  `encoder=carbon-3b` should not require duplicating the whole training
  config.
- **Reproducible.** The resolved config is part of the run artifact and
  is hashed into the manifest.
- **Discoverable.** A new contributor finds every knob without grepping
  argparse calls.
- **Typed.** A typo (`predictor.heads_n: 8` vs `predictor.n_heads: 8`)
  fails at load time, not at step 10,000.

Hydra + dataclass / Pydantic schema is the established pattern; we
adopt it with a small set of project conventions.

## 3. Specification

### 3.1 Layout

```
geno_lewm/config/
├── defaults/
│   ├── train.yaml           # top-level training config
│   ├── score.yaml           # top-level scoring config
│   ├── eval.yaml            # top-level eval config
│   ├── plan.yaml            # top-level planning config
│   ├── encoder/
│   │   ├── carbon-500m.yaml
│   │   ├── carbon-3b.yaml
│   │   └── carbon-8b.yaml
│   ├── predictor/
│   │   ├── default-bf16.yaml
│   │   ├── small-bf16.yaml
│   │   └── tiny-int8.yaml
│   ├── data/
│   │   ├── default.yaml
│   │   ├── small.yaml
│   │   └── debug.yaml
│   ├── optimizer/
│   │   ├── adamw.yaml
│   │   └── sgd-momentum.yaml
│   └── observability/
│       ├── local.yaml
│       ├── wandb.yaml
│       └── full.yaml
└── schema.py                # Pydantic / dataclass schema
```

Top-level configs reference subconfigs via Hydra's `defaults:` block.

### 3.2 Schema

The schema is declared with `pydantic.BaseModel` (or `dataclasses`,
TBD by implementation; the field set is fixed regardless):

```python
class GenoLeWMConfig(BaseModel):
    run_id: str
    seed: int
    phase: Literal["phase1", "phase2"]
    encoder: EncoderConfig
    predictor: PredictorConfig
    action: ActionEncoderConfig
    optimizer: OptimizerConfig
    data: DataConfig
    eval: EvalConfig
    observability: ObservabilityConfig
    runtime: RuntimeConfig
    deterministic: bool = False
    schema_version: str = "1.0.0"
```

Sub-schemas mirror the RFC subsystems. Every field has a default that
matches the RFC's documented default; every field has a docstring.

### 3.3 Namespaces

Top-level keys, all required, no top-level free-form fields:

| Key | Owner |
|-----|-------|
| `run_id` | trainer; auto-generated if absent |
| `seed` | trainer |
| `phase` | trainer |
| `encoder` | RFC-0002 |
| `predictor` | RFC-0004 |
| `action` | RFC-0003 |
| `optimizer` | RFC-0005 |
| `data` | RFC-0006 |
| `eval` | RFC-0007 |
| `observability` | RFC-0013 |
| `runtime` | RFC-0010 |
| `deterministic` | RFC-0005 |
| `schema_version` | RFC-0017 |

Unknown top-level keys are an error: `ConfigError.UnknownTopLevelKey`.

### 3.4 Override syntax

Hydra-style:

```
geno-lewm-train encoder=carbon-3b predictor=small-bf16 optimizer.lr=1e-4
```

Multi-run sweeps via `-m`:

```
geno-lewm-train -m encoder=carbon-500m,carbon-3b optimizer.lr=3e-4,1e-4
```

### 3.5 Resolution

1. Load the top-level config (e.g., `train.yaml`).
2. Merge in the `defaults:` listed subconfigs.
3. Apply CLI overrides.
4. Validate via Pydantic; fail on type mismatch / unknown field.
5. Persist resolved config to `{run_id}/config.resolved.yaml` and to
   `manifest.json` `training.config_file` field.

### 3.6 Secrets

No secrets in YAML. Secrets enter via env vars (`HF_TOKEN`,
`WANDB_API_KEY`, etc.) referenced by name in the YAML:

```yaml
observability:
  wandb:
    api_key_env: WANDB_API_KEY
```

The loader reads from the named env var at runtime. The resolved-config
artifact records the env-var **name** but not its value.

### 3.7 Schema versioning

`schema_version` is a top-level field; loaders accept any same-MAJOR
version and warn on lower-MINOR; raise on MAJOR mismatch. The schema
itself is versioned per the policy in
[`docs/spec/09-release-and-versioning.md`](../docs/spec/09-release-and-versioning.md).

### 3.8 Discovery

```
geno-lewm-train --print-config                  # prints resolved YAML
geno-lewm-train --print-config-tree             # prints with sources
geno-lewm-train --explain encoder.pool_radius   # prints docstring
```

`--explain` reads the field docstrings from the schema and renders them
in CLI-friendly form.

### 3.9 No alternative config formats

We do not support TOML, JSON, or INI as primary config formats. YAML is
the single source. JSON-Schema export of the Pydantic schema is
documented as a downstream convenience.

## 4. Rationale and alternatives

### 4.1 Why Hydra over OmegaConf-bare or argparse?

Hydra adds the composition (`defaults:`), the sweeping (`-m`), and the
output-directory conventions. OmegaConf alone leaves us to invent these
ourselves. argparse alone doesn't compose at all.

### 4.2 Why Pydantic for the schema rather than just OmegaConf-structured?

Pydantic's error messages are markedly better than OmegaConf's on type
mismatch, and Pydantic-models double as runtime API objects (Receipt is
already specced as a dataclass; aligning the config schema with the
runtime schema is a win).

### 4.3 Why YAML over TOML?

TOML's lack of merge-friendly multi-document support is the deciding
factor. The Hydra defaults pattern is YAML-native.

### 4.4 Why a single top-level `phase` field?

The `phase1 / phase2` choice gates whether `L_reg` is active, whether
LoRA is enabled, and a few other settings. Centralizing it prevents the
"oh I forgot to switch one of the three flags" bug.

### 4.5 Why prohibit unknown top-level keys?

Hydra otherwise silently swallows typos in top-level keys (e.g.,
`encoderr: carbon-3b`). A strict schema catches this at load.

## 5. Unresolved questions

- Whether to use Pydantic v2 vs dataclasses + omegaconf-structured. The
  field set is identical; implementation choice deferred to the first
  config PR.
- Whether to support env-var interpolation in YAML values (e.g.,
  `cache_dir: ${env:GENO_LEWM_CACHE}`). Hydra supports it; we may
  enable selectively.
- Whether to publish a JSON Schema for the config from CI for editor
  autocomplete.

## 6. Future work

- VS Code / nvim schema integration.
- A `geno-lewm config diff a.yaml b.yaml` helper for explaining sweep
  deltas.
- A web-based config explorer for the desktop app's settings UI.

## 7. Changelog

- 2026-05-20 — Initial draft.
