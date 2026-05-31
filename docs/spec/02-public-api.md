# 02 — Public API

- Status: Authoritative for v0.1
- Companion RFC: [RFC-0014](../rfcs/0014-public-api-and-stability.md)
- Versioning policy: [`09-release-and-versioning.md`](09-release-and-versioning.md)

The "public API" is the contract on which downstream code may depend.
Everything outside the public API is internal, may change without notice,
and must not be imported by external code.

## Stability classes

| Class | Contract | Marker |
|-------|----------|--------|
| **Stable** | semver applies; breaking changes require a MAJOR bump, a deprecation period, and an entry in CHANGELOG | re-exported from `geno_lewm.__init__` or `geno_lewm.<module>.__init__` |
| **Experimental** | may change without notice across MINORs; documented but not in `__init__` re-exports | `@experimental` decorator and entry in [`docs/api/experimental.md`](../api/experimental.md) |
| **Internal** | no stability guarantee; not documented as public | underscore prefix or `_internal/` submodule |

Modules under `geno_lewm/internal/` are not public regardless of name.

## Stable Python surface (v0.1)

### Top-level

```python
geno_lewm.__version__: str
geno_lewm.GenoLeWMRuntime         # see deploy/runtime
geno_lewm.EditSpec                # see action/spec
geno_lewm.EditType                # see action/spec
geno_lewm.SurpriseResult          # see surprise/score
geno_lewm.PlanningResult          # see planning/cem
geno_lewm.errors                  # the entire submodule (RFC-0012)
```

### `geno_lewm.encoder`

```python
class CarbonStateEncoder:
    def __init__(self, model_id: str, revision: str, *,
                 dtype: str = "bf16",
                 state_layer: int = -1,
                 pool_type: str = "centered_mean",
                 pool_radius: int = 256,
                 normalize: bool = True,
                 lora_config: LoRAConfig | None = None) -> None: ...

    def encode(self, window: str, edit_locus: int | None = None) -> Tensor: ...
    def encode_batch(self, windows: list[str],
                     edit_loci: list[int | None]) -> Tensor: ...
    @property
    def encoder_hash(self) -> bytes: ...
    @property
    def d_state(self) -> int: ...
```

Defined by [RFC-0002 §3.8](../rfcs/0002-state-encoder-carbon-integration.md#38-encoder-api).

The pure-Python windowing helpers from [RFC-0002 §3.2](../rfcs/0002-state-encoder-carbon-integration.md#32-window-format)
are importable without the optional ML runtime:

```python
@dataclass(frozen=True, slots=True)
class ExtractedWindow:
    sequence: str
    start_bp: int
    end_bp: int
    window_bp: int
    edit_locus: int | None = None
    relative_edit_locus: int | None = None
    pad_right_bp: int = 0

    @property
    def untargeted(self) -> bool: ...
    @property
    def sha256(self) -> bytes: ...
    def as_tokenizer_input(self) -> str: ...

def canonicalize_dna(sequence: str) -> str: ...
def window_sha256(sequence: str) -> bytes: ...
def extract_window(source_sequence: str, *,
                   edit_locus: int | None = None,
                   window_bp: int = 12_288) -> ExtractedWindow: ...
def pad_for_carbon_tokenizer(sequence: str, *,
                             token_bp: int = 6) -> str: ...
def wrap_dna_for_tokenizer(sequence: str) -> str: ...

def global_mean(hidden_states: Sequence[Sequence[float]]) -> tuple[float, ...]: ...
def centered_mean(hidden_states: Sequence[Sequence[float]], *,
                  center_token: int,
                  pool_radius: int = 256) -> tuple[float, ...]: ...

@dataclass(frozen=True, slots=True)
class PoolingResult:
    vector: tuple[float, ...]
    pool_type: Literal["centered_mean", "global_mean"]
    pool_radius: int
    untargeted: bool
    center_token: int | None
    token_count: int

    @property
    def d_state(self) -> int: ...
    def as_cache_fields(self) -> Mapping[str, object]: ...

def pool_hidden_states(hidden_states: Sequence[Sequence[float]], *,
                       edit_locus: int | None = None,
                       pool_type: Literal["centered_mean", "global_mean"] = "centered_mean",
                       pool_radius: int = 256,
                       token_bp: int = 6) -> PoolingResult: ...

@dataclass(frozen=True, slots=True)
class WindowCacheKey:
    window_hash: bytes
    encoder_hash: bytes
    state_layer: int
    pool_type: str
    pool_radius: int
    dtype: str

@dataclass(frozen=True, slots=True)
class WindowCacheRecord:
    chrom: str
    start_bp: int
    end_bp: int
    window_hash: bytes
    encoder_hash: bytes
    state_layer: int
    pool_type: str
    pool_radius: int
    dtype: str
    embedding: tuple[float, ...]
    untargeted: bool
    created_at: int = 0
    schema_version: str = "1.0.0"

    @property
    def key(self) -> WindowCacheKey: ...
    def with_created_at(self) -> "WindowCacheRecord": ...

@dataclass(frozen=True, slots=True)
class CacheReindexReport:
    indexed_shards: int
    indexed_rows: int
    index_path: Path

@dataclass(frozen=True, slots=True)
class CacheRepairReport:
    checked_shards: int
    quarantined: tuple[Path, ...]
    reindex: CacheReindexReport

def default_cache_dir() -> Path: ...
def shard_path_for(cache_dir: Path | str, *,
                   encoder_id: str,
                   state_layer: int,
                   pool_type: str,
                   pool_radius: int,
                   contig: str,
                   stride_block: int) -> Path: ...
def write_shard(cache_dir: Path | str, *,
                encoder_id: str,
                contig: str,
                stride_block: int,
                records: Sequence[WindowCacheRecord]) -> Path: ...
def read_embedding(cache_dir: Path | str,
                   key: WindowCacheKey) -> tuple[float, ...] | None: ...
def reindex_cache(cache_dir: Path | str) -> CacheReindexReport: ...
def repair_cache(cache_dir: Path | str) -> CacheRepairReport: ...
```

### `geno_lewm.action`

```python
@dataclass(frozen=True, slots=True)
class EditSpec:
    chrom: str
    pos: int
    ref: str
    alt: str
    edit_type: EditType
    def relative_to(self, window_start_bp: int, window_end_bp: int) -> "RelEdit": ...

@dataclass(frozen=True, slots=True)
class RelEdit:
    rel_pos: int
    edit_type: EditType
    ref_bases: str
    alt_bases: str

class EditType(IntEnum):
    SNV = 0
    INS = 1
    DEL = 2
    MNV = 3
    INDEL = 4
    SV = 5

class ActionEncoder(nn.Module):
    def __init__(self, *,
                 d_action: int = 512,
                 d_pos: int = 128,
                 d_type: int = 64,
                 d_seq: int = 256,
                 max_window_bp: int = 12_288,
                 carbon_tokenizer: PreTrainedTokenizer | None = None) -> None: ...

    def forward(self, edits: list[RelEdit]) -> Tensor: ...
    @property
    def d_action(self) -> int: ...

def apply_edit(window: str, edit: RelEdit) -> str: ...
def apply_edits(window: str, edits: list[RelEdit]) -> str: ...
```

Defined by [RFC-0003 §3](../rfcs/0003-action-representation-genomic-edits.md#3-specification).

### `geno_lewm.predictor`

```python
class Predictor(nn.Module):
    def forward(self,
                state: Tensor,
                actions: Tensor,
                action_mask: Tensor) -> Tensor: ...
    def predict_single(self, s_t: Tensor, edit: RelEdit) -> Tensor: ...
    def predict_haplotype(self, s_t: Tensor, edits: list[RelEdit]) -> Tensor: ...
    def predict_trajectory(self, s_t: Tensor,
                           edits: list[RelEdit]) -> list[Tensor]: ...

class ARPredictor(nn.Module):
    def rollout(self, state: Tensor,
                action_sequence: list[Tensor]) -> list[Tensor]: ...
```

Defined by [RFC-0004 §3](../rfcs/0004-predictor-architecture.md#3-specification).

### `geno_lewm.surprise`

```python
@dataclass
class SurpriseResult:
    sigma_raw: float
    sigma_calibrated: float
    bucket_id: str
    confidence: float
    low_confidence: bool

def score_variant(variant: EditSpec,
                  encoder: CarbonStateEncoder,
                  action_encoder: ActionEncoder,
                  predictor: Predictor,
                  calibration: CalibrationTable,
                  aggregation: str = "mean") -> SurpriseResult: ...

def score_vcf(vcf_path: Path,
              encoder: CarbonStateEncoder,
              action_encoder: ActionEncoder,
              predictor: Predictor,
              calibration: CalibrationTable,
              output_path: Path,
              show_progress: bool = True) -> None: ...
```

Defined by [RFC-0009 §3.10](../rfcs/0009-surprise-based-pathogenicity-scoring.md#310-scorer-api).

### `geno_lewm.planning`

```python
@dataclass
class PlanningConfig:
    horizon: int = 5
    n_iterations: int = 5
    n_samples: int = 1024
    n_elite: int = 64
    distance: str = "l2"
    cost: str = "count"
    cost_weight: float = 0.0
    stopping_eps: float = 0.05
    patience: int = 2
    seed: int | None = None

@dataclass
class PlanningResult:
    best_edits: list[RelEdit]
    best_distance: float
    best_predicted_state: Tensor
    n_predictor_calls: int
    iterations: list[CEMIterationLog]
    elapsed_seconds: float

def plan(initial_state: Tensor,
         target_state: Tensor,
         predictor: Predictor,
         action_encoder: ActionEncoder,
         sampler: ActionSampler | None = None,
         config: PlanningConfig | None = None) -> PlanningResult: ...
```

Defined by [RFC-0008 §3.8](../rfcs/0008-latent-planning.md#38-planning-api).

### `geno_lewm.deploy`

```python
class GenoLeWMRuntime:
    def __init__(self, model_dir: Path, backend: str = "auto") -> None: ...
    def score_variant(self, variant: EditSpec,
                      window: str | None = None) -> SurpriseResult: ...
    def score_vcf(self, vcf_path: Path,
                  fasta_path: Path,
                  output_path: Path,
                  batch_size: int = 64,
                  progress: bool = True) -> None: ...
    def encode_window(self, window: str,
                      edit_locus: int | None = None) -> Tensor: ...
    def predict(self, state: Tensor, edits: list[RelEdit]) -> Tensor: ...
```

Defined by [RFC-0010 §3.4](../rfcs/0010-on-device-personal-genome-deployment.md#34-runtime-contract).

### `geno_lewm.attestation`

```python
@dataclass
class Receipt:
    schema_version: str
    model_id: str
    input_commitment: str
    output: dict[str, object]
    output_commitment: str
    calibration_hash: str
    runtime: RuntimeMetadata
    timestamp: datetime
    attestation: Attestation

def write_receipt(receipt: Receipt, path: Path) -> None: ...
def read_receipt(path: Path) -> Receipt: ...
def verify_receipt(receipt: Receipt,
                   model_dir: Path | None = None,
                   rerun: bool = False) -> VerificationResult: ...
```

Defined by [RFC-0011 §3.3, §3.4](../rfcs/0011-verifiable-inference-attestation.md).

## Stable CLI surface (v0.1)

| Command | Purpose | RFC |
|---------|---------|-----|
| `geno-lewm-train` | train predictor end-to-end | RFC-0005, RFC-0018 |
| `geno-lewm-score` | score a single variant or a VCF | RFC-0009, RFC-0010 |
| `geno-lewm-rollout` | run multi-edit haplotype rollout | RFC-0004 |
| `geno-lewm-plan` | run CEM planning to a target state | RFC-0008 |
| `geno-lewm-eval` | run a single benchmark | RFC-0007 |
| `geno-lewm-eval-all` | run the release evaluation suite | RFC-0007 |
| `geno-lewm-export` | export to ONNX / Core ML / GGUF | RFC-0010 |
| `geno-lewm-verify` | verify a receipt | RFC-0011 |
| `geno-lewm-cache-windows` | pre-compute the reference window cache | RFC-0006 |
| `geno-lewm-prepare-gnomad` | build the gnomAD Parquet shard | RFC-0006 |
| `geno-lewm-prepare-clinvar` | build the ClinVar Parquet shard | RFC-0006 |
| `geno-lewm-update` | check for model updates | RFC-0010 |

All commands accept `--config FILE` (Hydra-compatible), `--seed INT`,
`--log-level {debug,info,warn,error}`, and `--receipt PATH | --no-receipt`
where receipts are applicable.

Defined by [RFC-0018](../rfcs/0018-cli-design.md).

## Runtime backends

```python
backend ∈ {"auto", "coreml", "cuda", "onnx", "cpu"}
```

`auto` selects the best available in the order documented in
[RFC-0010 §3.4](../rfcs/0010-on-device-personal-genome-deployment.md#34-runtime-contract).

## Type-stub contract

All public APIs ship inline type annotations under `py.typed`. Mypy in
`strict` mode passes against the public surface; `tests/typecheck/` pins
the contract with `reveal_type` assertions.

## Backwards compatibility

- Adding a new keyword argument with a default that preserves prior
  behavior is a MINOR change.
- Adding a new optional return field on a dataclass is a MINOR change.
- Renaming any public symbol is a MAJOR change.
- Changing a dtype, return shape, or numerical contract is a MAJOR change.
- Tightening validation (e.g., narrowing an accepted enum value) is a
  MAJOR change.
- Changing default values that affect numerical outputs is a MAJOR change.

Deprecations carry at least one MINOR release of `DeprecationWarning`
before removal in the subsequent MAJOR.

## Experimental surface

The following are explicitly experimental in v0.1 and may change in any
MINOR release:

- `geno_lewm.planning.mcts.*` (Phase 2 surface)
- `geno_lewm.deploy.tee_attestation.*` (Phase 3 surface)
- `geno_lewm.attestation.stark.*` (Phase 4 surface)
- `geno_lewm.encoder.lora.*` (Phase 2)
- `geno_lewm.surprise.bayesian.*` (Phase 2)
- `geno_lewm.surprise.directional.*` (Phase 2)

Each lives behind an `@experimental` decorator that emits a
`FutureWarning` on first import per process.

## Out-of-scope public API

The following are explicitly **not** public, regardless of how
convenient that might be:

- Any module-private helper named `_*` or under any `_internal/` submodule.
- The contents of `geno_lewm.config.defaults.*` (Hydra defaults are
  internal to the CLI; user configs override them).
- The Hydra YAML schema is internal except as documented in
  [RFC-0017](../rfcs/0017-configuration-system.md).
- Test fixtures under `tests/fixtures/`.

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-API-1 | Whether to expose `Receipt` as a Pydantic v2 model for downstream JSON Schema generation | core | v0.2 |
| OQ-API-2 | Whether `EditSpec.relative_to` should return `Either[RelEdit, OutOfWindow]` rather than raising | core | v0.2 |
| OQ-API-3 | Whether to provide a `geno_lewm.bench` namespace for downstream benchmark harnesses | core | v0.3 |
