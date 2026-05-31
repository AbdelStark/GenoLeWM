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

### `geno_lewm.data`

```python
@dataclass(frozen=True, slots=True)
class CarbonSourceMix:
    source: str
    fraction: float

@dataclass(frozen=True, slots=True)
class CarbonCorpusConfig:
    dataset_id: str = "HuggingFaceBio/carbon-pretraining-corpus"
    dataset_config: str | None = None
    split: str = "train"
    streaming: bool = True
    subset_fraction: float = 0.10
    subset_seed: int = 0
    sequence_field: str = "sequence"
    source_field: str = "source"
    source_id_field: str = "id"
    window_bp: int = 12_288
    margin_bp: int = 256
    stride_bp: int = 8_192

@dataclass(frozen=True, slots=True)
class CarbonRecord:
    record_id: str
    source: str
    sequence: str
    @property
    def length_bp(self) -> int: ...

@dataclass(frozen=True, slots=True)
class CarbonWindow:
    record_id: str
    source: str
    start_bp: int
    end_bp: int
    sequence: str
    @property
    def window_bp(self) -> int: ...
    @property
    def window_id(self) -> str: ...

def sample_source(rng: random.Random,
                  *,
                  mix: Sequence[CarbonSourceMix] = CARBON_SUBMIX) -> str: ...
def draw_source_counts(n: int,
                       *,
                       rng: random.Random,
                       mix: Sequence[CarbonSourceMix] = CARBON_SUBMIX) -> dict[str, int]: ...
def stable_subset_includes(record_id: str, *,
                           fraction: float,
                           seed: int = 0) -> bool: ...
def iter_window_starts(sequence_length: int,
                       *,
                       window_bp: int = 12_288,
                       margin_bp: int = 256,
                       stride_bp: int = 8_192,
                       rng: random.Random | None = None) -> Iterator[int]: ...
def iter_record_windows(record: CarbonRecord,
                        *,
                        window_bp: int = 12_288,
                        margin_bp: int = 256,
                        stride_bp: int = 8_192,
                        rng: random.Random | None = None) -> Iterator[CarbonWindow]: ...
def iter_carbon_records(rows: Iterable[Mapping[str, Any]],
                        *,
                        sequence_field: str = "sequence",
                        source_field: str = "source",
                        source_id_field: str = "id",
                        subset_fraction: float = 1.0,
                        subset_seed: int = 0) -> Iterator[CarbonRecord]: ...
def load_hf_carbon_records(config: CarbonCorpusConfig | None = None) -> Iterator[CarbonRecord]: ...
def normalize_source_label(value: object) -> str: ...
```

Defined by [RFC-0006 §3.1–§3.2](../rfcs/0006-data-pipeline.md#31-reference-corpus).

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

@dataclass(frozen=True, slots=True)
class PredictionLossResult:
    loss: Tensor
    pred_loss: Tensor
    kl_reg: Tensor
    phase: Literal["phase1", "phase2"]

def prediction_loss(prediction: Tensor,
                    target: Tensor,
                    *,
                    alpha: float = 1.0,
                    beta: float = 0.1,
                    mask: Tensor | None = None,
                    eps: float = 1e-8) -> Tensor: ...

def lejepa_kl_regularizer(states: Tensor,
                          *,
                          eps: float = 1e-6) -> Tensor: ...

def predictor_loss(prediction: Tensor,
                   target: Tensor,
                   *,
                   phase: Literal["phase1", "phase2"] = "phase1",
                   alpha: float = 1.0,
                   beta: float = 0.1,
                   gamma: float = 0.5,
                   mask: Tensor | None = None,
                   regularizer_states: Tensor | None = None,
                   eps: float = 1e-6) -> PredictionLossResult: ...
```

Defined by [RFC-0004 §3](../rfcs/0004-predictor-architecture.md#3-specification).

### `geno_lewm.training`

```python
@dataclass(frozen=True, slots=True)
class CollapseMetrics:
    pred_cos_mean: float
    pred_l2_mean: float
    target_var_per_dim: float
    pred_var_per_dim: float
    pred_target_corr: float
    pairwise_pred_dist_mean: float
    kl_reg: float

@dataclass(frozen=True, slots=True)
class CollapseThresholds:
    pred_var_to_target_var: float = 0.5
    pairwise_to_initial: float = 0.5
    kl_reg_max: float = 10.0

@dataclass(frozen=True, slots=True)
class CollapseAlert:
    criterion: str
    value: float
    threshold: float

@dataclass(frozen=True, slots=True)
class CollapseCheck:
    metrics: CollapseMetrics
    alerts: tuple[CollapseAlert, ...]
    @property
    def tripped(self) -> bool: ...

@dataclass(slots=True)
class CollapseMonitor:
    log_every_steps: int = 500
    thresholds: CollapseThresholds = CollapseThresholds()
    initial_pairwise_pred_dist_mean: float | None = None
    def should_log(self, step: int) -> bool: ...
    def observe(self, prediction: object, target: object, *,
                kl_reg: float, step: int,
                logger: GenoLeWMLogger | None = None,
                force: bool = False) -> CollapseCheck | None: ...

def compute_collapse_metrics(prediction: object,
                             target: object,
                             *,
                             kl_reg: float) -> CollapseMetrics: ...
def detect_collapse(metrics: CollapseMetrics,
                    *,
                    thresholds: CollapseThresholds | None = None,
                    initial_pairwise_pred_dist_mean: float | None = None
                    ) -> tuple[CollapseAlert, ...]: ...
def record_collapse_metrics(metrics: CollapseMetrics,
                            *,
                            alerts: Iterable[CollapseAlert] = (),
                            logger: GenoLeWMLogger | None = None,
                            step: int | None = None) -> None: ...

@dataclass(frozen=True, slots=True)
class EditTypeWeight:
    edit_type: EditType
    weight: float

@dataclass(frozen=True, slots=True)
class RolloutStepWeight:
    steps: int
    weight: float

DEFAULT_EDIT_TYPE_WEIGHTS: tuple[EditTypeWeight, ...]
DEFAULT_ROLLOUT_STEP_MIX: tuple[RolloutStepWeight, ...]

def sample_edit_type(rng: random.Random,
                     *,
                     weights: Sequence[EditTypeWeight] = DEFAULT_EDIT_TYPE_WEIGHTS) -> EditType: ...
def draw_edit_type_counts(n: int,
                          *,
                          rng: random.Random,
                          weights: Sequence[EditTypeWeight] = DEFAULT_EDIT_TYPE_WEIGHTS
                          ) -> dict[EditType, int]: ...
def sample_rollout_steps(rng: random.Random,
                         *,
                         mix: Sequence[RolloutStepWeight] = DEFAULT_ROLLOUT_STEP_MIX) -> int: ...
def draw_rollout_step_counts(n: int,
                             *,
                             rng: random.Random,
                             mix: Sequence[RolloutStepWeight] = DEFAULT_ROLLOUT_STEP_MIX
                             ) -> dict[int, int]: ...
```

Defined by [RFC-0005 §3.6](../rfcs/0005-training-objective.md#36-collapse-monitoring)
and [§3.7](../rfcs/0005-training-objective.md#37-batching-and-gradient-accumulation).

### `geno_lewm.surprise`

```python
CALIBRATION_SCHEMA_VERSION: str
DEFAULT_CDF_POINTS: int
DEFAULT_REFERENCE_PER_BUCKET: int
LOW_CONFIDENCE_BUCKET_SIZE: int

REGION_CLASSES: tuple[str, ...]
GC_BINS: tuple[str, ...]
REPEAT_CLASSES: tuple[str, ...]
UNKNOWN_BUCKET_ID: str
DEFAULT_MIN_BUCKET_SIZE: int

@dataclass(frozen=True)
class CalibrationExample:
    bucket_id: str
    sigma_raw: float

@dataclass(frozen=True)
class CalibrationWarning:
    bucket_id: str
    resolved_bucket_id: str
    n_calibration: int
    min_bucket_size: int
    low_confidence: bool

@dataclass(frozen=True)
class CalibrationBucket:
    bucket_id: str
    n_calibration: int
    cdf: tuple[float, ...]
    sigma_grid: tuple[float, ...]
    back_off_to: str | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    @property
    def confidence(self) -> float: ...

    @property
    def low_confidence(self) -> bool: ...

@dataclass(frozen=True)
class CalibrationTable:
    buckets: tuple[CalibrationBucket, ...]
    warnings: tuple[CalibrationWarning, ...] = ()
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def get(self, bucket_id: str) -> CalibrationBucket | None: ...
    def require(self, bucket_id: str) -> CalibrationBucket: ...
    def resolve(self,
                label_or_bucket: str,
                *,
                min_bucket_size: int = 1000) -> CalibrationBucket: ...

def build_calibration_table(examples: Iterable[CalibrationExample],
                            *,
                            seed: int = 0,
                            per_bucket_sample: int = 10000,
                            grid_size: int = 1001,
                            min_bucket_size: int = 1000,
                            low_confidence_size: int = 100,
                            warn_sparse: bool = True) -> CalibrationTable: ...

def write_calibration_table(table: CalibrationTable, path: str | Path) -> Path: ...
def read_calibration_table(path: str | Path) -> CalibrationTable: ...

@dataclass(frozen=True)
class ContextLabel:
    region_class: str
    gc_bin: str
    repeat_class: str

    @property
    def bucket_id(self) -> str: ...

def classify_context(*,
                     region: str | Sequence[str] | None,
                     gc_window: str,
                     repeat: str | Sequence[str] | None = None,
                     low_gc_cutoff: float = 1 / 3,
                     high_gc_cutoff: float = 2 / 3) -> ContextLabel: ...

def classify_region(annotation: str | Sequence[str] | None) -> str: ...
def classify_repeat(annotation: str | Sequence[str] | None) -> str: ...
def classify_gc_bin(sequence: str,
                    *,
                    low_cutoff: float = 1 / 3,
                    high_cutoff: float = 2 / 3) -> str: ...
def gc_fraction(sequence: str) -> float: ...
def make_bucket_id(region_class: str, gc_bin: str, repeat_class: str) -> str: ...
def backoff_chain(label_or_bucket: ContextLabel | str) -> tuple[str, ...]: ...
def select_backoff_bucket(label_or_bucket: ContextLabel | str,
                          bucket_sizes: Mapping[str, int],
                          *,
                          min_count: int = 1000) -> str: ...

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

Context labels and calibration bucket back-off are defined by
[RFC-0009 §3.3](../rfcs/0009-surprise-based-pathogenicity-scoring.md#33-context-stratification).
Calibration table building and the on-disk `calibration.parquet` schema
are defined by
[RFC-0009 §3.4](../rfcs/0009-surprise-based-pathogenicity-scoring.md#34-calibration-distribution).
The model-dependent scorer is defined by
[RFC-0009 §3.10](../rfcs/0009-surprise-based-pathogenicity-scoring.md#310-scorer-api).

### `geno_lewm.planning`

```python
DEFAULT_ACTION_TYPE_WEIGHTS: tuple[EditTypeWeight, ...]
DEFAULT_TYPE_COSTS: Mapping[EditType, float]

class ActionSampler:
    def __init__(self,
                 window: str,
                 *,
                 seed: int | None = None,
                 rng: random.Random | None = None,
                 edge_margin: int = 64,
                 type_weights: Sequence[EditTypeWeight] = DEFAULT_ACTION_TYPE_WEIGHTS,
                 length_dist: Mapping[int, float] | Sequence[float] | None = None,
                 position_bin_bp: int = 8,
                 position_weights: Mapping[int, float] | Sequence[float] | None = None,
                 max_attempts: int = 256) -> None: ...

    def sample_edit(self, edit_type: EditType | int | None = None) -> RelEdit: ...
    def sample_sequence(self, horizon: int) -> tuple[RelEdit, ...]: ...
    def sample_sequences(self, n: int, horizon: int) -> tuple[tuple[RelEdit, ...], ...]: ...

def count_cost(edits: Sequence[RelEdit]) -> float: ...
def edit_bp_cost(edit: RelEdit) -> float: ...
def bp_cost(edits: Sequence[RelEdit]) -> float: ...
def weighted_type_cost(edits: Sequence[RelEdit],
                       weights: Mapping[EditType, float] = DEFAULT_TYPE_COSTS) -> float: ...
def custom_cost(edits: Sequence[RelEdit],
                cost_fn: Callable[[Sequence[RelEdit]], float]) -> float: ...

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

The cost functions and sampler are implemented. `PlanningConfig`,
`PlanningResult`, and `plan` define the upcoming CEM surface.

Defined by [RFC-0008 §3.3](../rfcs/0008-latent-planning.md#33-edit-search-space),
[§3.5](../rfcs/0008-latent-planning.md#35-cost-functions), and
[§3.8](../rfcs/0008-latent-planning.md#38-planning-api).

### `geno_lewm.deploy`

```python
BACKEND_PRIORITY: tuple[str, ...]

@dataclass(frozen=True)
class BackendProbe:
    backend: str
    available: bool
    reason: str

def probe_backends(model_dir: str | Path | None = None) -> tuple[BackendProbe, ...]: ...
def select_backend(backend: str = "auto",
                   *,
                   probes: Sequence[BackendProbe] | None = None) -> str: ...

@contextmanager
def fail_closed_network_guard() -> Iterator[None]: ...

class GenoLeWMRuntime:
    model_dir: Path
    backend: str
    probes: tuple[BackendProbe, ...]

    def __init__(self, model_dir: str | Path, backend: str = "auto") -> None: ...
    def score_variant(self, variant: EditSpec,
                      window: str | None = None) -> Any: ...
    def score_vcf(self, vcf_path: str | Path,
                  fasta_path: str | Path,
                  output_path: str | Path,
                  batch_size: int = 64,
                  progress: bool = True) -> None: ...
    def encode_window(self, window: str,
                      edit_locus: int | None = None) -> Any: ...
    def predict(self, state: Any, edits: Sequence[RelEdit]) -> Any: ...
```

Backend probing and the fail-closed network guard are implemented.
Model-dependent scoring / encoding / prediction methods currently fail
fast with `RuntimeSetupError` until the scorer and deploy backends land.
Defined by [RFC-0010 §3.4](../rfcs/0010-on-device-personal-genome-deployment.md#34-runtime-contract)
and [§3.7](../rfcs/0010-on-device-personal-genome-deployment.md#37-privacy-contract).

### `geno_lewm.deploy.import_`

```python
@dataclass(frozen=True)
class VcfConversionSummary:
    output_path: Path
    records_written: int
    ref_calls_skipped: int
    no_calls_skipped: int

def convert_23andme_to_vcf(input_path: str | Path,
                           output_path: str | Path,
                           reference_alleles: Mapping[tuple[str, int], str],
                           *,
                           sample_id: str = "sample") -> VcfConversionSummary: ...
def convert_ancestry_to_vcf(...): ...
def convert_myheritage_to_vcf(...): ...
def convert_sequencing_json_to_vcf(input_path: str | Path,
                                   output_path: str | Path,
                                   *,
                                   sample_id: str = "sample") -> VcfConversionSummary: ...
```

Array raw-data formats do not carry VCF `REF` alleles, so their
converters require a local `reference_alleles` map keyed by
`(chrom, pos)`. Missing or non-ACGT alleles raise `VcfParseError`; the
converters do not perform network calls. Defined by
[RFC-0010 §3.9](../rfcs/0010-on-device-personal-genome-deployment.md#39-compatibility-with-personal-genome-data-formats).

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
