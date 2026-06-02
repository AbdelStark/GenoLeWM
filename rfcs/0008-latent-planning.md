# RFC-0008: Latent planning

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-02
- **Depends on:** RFC-0002, RFC-0003, RFC-0004
- **Supersedes:** —
- **Implementation status:** Partial; cost functions and the factored
  `ActionSampler` are implemented. CEM solver and CLI integration are
  not yet implemented.

---

## 1. Summary

This RFC specifies the planning primitive: model-predictive control
(MPC) in Carbon's latent space, over discrete edit actions, using the
GenoLeWM predictor as the dynamics model. The default solver is the
Cross-Entropy Method (CEM); a Monte-Carlo Tree Search (MCTS) variant is
specified for Phase 2. The planner is what turns GenoLeWM from a
scoring tool into a decision-making tool — given a target latent state,
it returns an ordered edit list that the predictor believes will reach
(or approach) that state.

## 2. Motivation

A world model that cannot plan is just a scoring function. The whole
reason to train an action-conditioned predictor is that, once you have
one, you can search over action sequences to achieve goals.

Three concrete uses motivate the planning primitive:

1. **Reverse engineering pathogenic variants.** Given a pathogenic
   variant, find the minimal set of further edits that would restore
   the reference-like latent neighborhood. Useful for designing
   compensatory mutations and for understanding which downstream
   features the pathogenic edit perturbed.
2. **CRISPR guide selection.** Given a desired latent direction (e.g.,
   "move the locus toward the latent cluster of high-expression
   variants"), enumerate candidate guide RNA targets and rank them by
   predicted latent displacement.
3. **Latent counterfactual exploration.** Given any genomic locus and
   a target state, ask "what is the smallest edit sequence (by edit
   count, by base-pair cost, by edit-type cost) that gets there?"

All three are search problems over a discrete edit space, scored by a
distance function in latent space. The predictor provides the
dynamics; everything else is search.

Planning never calls Carbon during the search loop. This is the
efficiency thesis of the world-model framing: pay for Carbon once at
the start, then run thousands of CEM rollouts at predictor cost only.

## 3. Specification

### 3.1 Problem statement

Given:

- An initial state `s_0 ∈ ℝ^{d_state}` (a Carbon-encoded reference
  window).
- A target specification: either a target state `s_target ∈ ℝ^{d_state}`,
  a target region (set of states), or a target functional defined by
  another scorer.
- An edit search space `A` (a sampler that produces candidate
  `RelEdit`s, see §3.3).
- A horizon `K ∈ ℕ` (maximum number of edits).
- A cost function `c: list[RelEdit] → ℝ_{≥0}`.

Find:

```
a*_{1:K} = argmin_{a_{1:K} ∈ A^K, k ≤ K}  d(g^k(s_0, a_{1:k}), s_target) + λ · c(a_{1:k})
```

where `g^k` denotes the predictor's k-step autoregressive rollout
(RFC-0004 §3.3), `d` is a distance function in latent space, and `λ`
balances goal-achievement against edit cost.

### 3.2 Distance functions

The default distance is L2 in normalized latent space:

```
d(ŝ, s_target) = ||ŝ − s_target||₂
```

Alternative distances are supported via configuration:

| Name | Formula | Use case |
|------|---------|----------|
| `l2` (default) | `‖ŝ − s_target‖₂` | general-purpose |
| `cos` | `1 − cos(ŝ, s_target)` | direction-only matching |
| `region` | `min_{s ∈ S} ‖ŝ − s‖₂` | match any state in a set |
| `projection` | `‖P(ŝ) − P(s_target)‖₂` | match along a subspace |

For region targets, `S` is a finite set provided as input (e.g., the
cached embeddings of all "benign" ClinVar variants near the locus,
which together define the benign latent neighborhood).

### 3.3 Edit search space

The search space `A` is a sampler — not an enumeration. For typical
windows there are millions of candidate edits, and exhaustive search is
infeasible. CEM operates over a sampler with a learnable proposal
distribution.

Default sampler decomposition:

```
A = A_type × A_pos × A_bases
```

where the three factors are sampled independently:

- **`A_type`**: categorical over `{SNV, INS, DEL, MNV, INDEL}` with
  initial probabilities from RFC-0006 §3.3 (the training mix).
- **`A_pos`**: uniform over window positions, respecting the 64 bp
  edge margin. CEM updates this to a categorical over discretized
  positions (bins of 8 bp) so the proposal can become peaky.
- **`A_bases`**: conditional on type. For SNV, uniform over the three
  non-reference bases. For INS/DEL/MNV/INDEL, length drawn from a
  truncated geometric distribution and bases uniform.

The user may pass a custom sampler that restricts the search space
(e.g., "only SNVs in this 200 bp window", "only edits at known CRISPR
PAM sites").

### 3.4 Cross-Entropy Method (CEM) solver

The default solver:

```
Input: s_0, s_target, K, sampler A, n_iterations, n_samples, n_elite

For i in 1..n_iterations:
    Sample n_samples candidate edit sequences from A
    For each candidate a_{1:K}:
        Compute ŝ_K = g^K(s_0, a_{1:K})    [predictor rollout]
        Compute score = d(ŝ_K, s_target) + λ · c(a_{1:K})
    Select the n_elite candidates with lowest score
    Re-fit A to the elite (per-factor MLE: counts for categorical,
                          empirical for continuous)
    Optionally apply Gaussian smoothing to the re-fitted A to maintain
    exploration

Return the best candidate seen
```

**Defaults:**

- `n_iterations = 5`
- `n_samples = 1024` (per iteration)
- `n_elite = 64` (top 6.25%)
- `K = 5` (horizon)
- `λ = 0.0` (no edit cost by default; pure target-distance optimization)
- Sampler smoothing: 0.1 weight on a uniform prior, mixed in after each
  re-fit, to prevent the sampler from collapsing to a single edit and
  losing exploration.

With these defaults, a single planning call performs
`5 × 1024 = 5,120` K-step predictor rollouts, totaling
`5 × 1024 × 5 = 25,600` predictor calls. On an H100, this is < 1
second; on a laptop, < 30 seconds.

### 3.5 Cost functions

`c(a_{1:k})` supports three common formulations:

| Name | Formula | Use case |
|------|---------|----------|
| `count` (default `λ=0`) | `k` | minimize edit count |
| `bp` | `Σ_k |edit_k|` | minimize total bp change |
| `weighted_type` | `Σ_k w(type(edit_k))` | bias against SVs / indels |
| `custom` | user-provided | application-specific |

For CRISPR guide selection, a typical custom cost weights edits by the
inverse of an off-target score (so the planner is steered toward guides
with cleaner specificity).

### 3.6 Monte-Carlo Tree Search (Phase 2)

For problems where the search space has strong local structure
(e.g., compensatory mutations near a specific locus), MCTS may be more
sample-efficient than CEM. The Phase 2 MCTS specification:

- Nodes are partial edit sequences `a_{1:k}`.
- Node values are predictor rollout distances `d(g^k(s_0, a_{1:k}),
  s_target)`.
- Selection uses UCB1 with the standard exploration constant
  `c_uct = √2`.
- Expansion samples a new edit from `A` and adds it as a child.
- Simulation runs predictor rollout to depth `K`.
- Backpropagation updates value estimates along the path.

MCTS is gated behind a `planner=mcts` configuration option; CEM is the
default.

### 3.7 Stopping criteria

The planner stops when any of the following is true:

- `n_iterations` complete.
- Best distance `d(ŝ, s_target) < ε` (default `ε = 0.05` in normalized
  L2; this is "close enough").
- Best distance stops improving for `patience = 2` consecutive
  iterations.

### 3.8 Planning API

```python
# geno_lewm.planning

class ActionSampler:
    def sample_edit(self, edit_type: EditType | int | None = None) -> RelEdit:
        ...

    def sample_sequence(self, horizon: int) -> tuple[RelEdit, ...]:
        ...

    def sample_sequences(self, n: int, horizon: int) -> tuple[tuple[RelEdit, ...], ...]:
        ...

def count_cost(edits: Sequence[RelEdit]) -> float:
    ...

def bp_cost(edits: Sequence[RelEdit]) -> float:
    ...

def weighted_type_cost(edits: Sequence[RelEdit], weights: Mapping[EditType, float]) -> float:
    ...

def custom_cost(edits: Sequence[RelEdit], cost_fn: Callable[[Sequence[RelEdit]], float]) -> float:
    ...

# geno_lewm.planning.cem

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

def plan(
    initial_state: Tensor,
    target_state: Tensor,
    predictor: Predictor,
    action_encoder: ActionEncoder,
    sampler: ActionSampler | None = None,
    config: PlanningConfig | None = None,
) -> PlanningResult:
    ...
```

The function is pure (deterministic given a seed) and side-effect-free
on the predictor.

### 3.9 CLI

```
geno-lewm plan \
    --window-fasta region.fa \
    --target-fasta target_region.fa \
    --horizon 5 \
    --iterations 5 \
    --samples 1024 \
    --output plan.json
```

The CLI accepts target states either as a FASTA (which gets encoded to
a target state) or as a pre-computed `.npy` latent vector (for advanced
users who have constructed targets in latent space directly).

## 4. Rationale and alternatives

### 4.1 Why CEM over gradient-based search?

Edits are discrete (position is integer, type is categorical, bases are
categorical). Gradient-based optimization over the predictor's input
space would require continuous relaxation, which introduces a separate
optimization difficulty: the relaxed optimum may not correspond to any
realizable edit. CEM optimizes directly in the discrete space and
returns valid edits by construction.

We considered:

- **Beam search.** Works for discrete spaces but scales poorly with
  `K` (branching factor × `K` exponentially). CEM's sampled-distribution
  approach is more compute-efficient for moderate `K`.
- **REINFORCE / policy gradient.** Requires training a policy. CEM
  is amortization-free: it does not require any per-task training.
- **Exhaustive enumeration.** Feasible for very small windows and
  K=1 (a few thousand SNVs in a 1 kbp window). Supported as an explicit
  `planner=exhaustive` option for K=1.

### 4.2 Why MCTS as a Phase 2 alternative?

CEM's weakness is that the sampler is a product of marginals; it cannot
easily capture correlations between edits (e.g., "edit at position 100
and edit at position 200 only work well together"). MCTS naturally
handles such structure through its tree. We expect CEM to dominate on
problems with weak inter-edit structure and MCTS to dominate on
problems with strong structure; offering both lets the user choose.

### 4.3 Why no model-based RL?

A model-based RL formulation would train a policy network on top of the
world model. This is appealing for re-use across many planning queries.
We deferred it because:

- The training-set distribution of "planning queries" is undefined; we
  do not have a reward function over goals.
- Per-query CEM is fast enough on the H100 (< 1 second) that policy
  amortization is not on the critical path.
- A policy is harder to verify (RFC-0011) than a per-query CEM run
  whose entire trace is reproducible.

If, in v2, a clear set of canonical planning queries emerges (e.g.,
"design a CRISPR guide for gene X"), a policy could be trained for
those.

### 4.4 Why is the predictor's quality the bottleneck for planning?

Planning quality is bounded by predictor quality: if the predictor is
inaccurate, the planner finds edits that minimize predictor distance
but not true distance. The rollout-fidelity benchmarks (RFC-0007 §3.2)
are therefore the right indicator of planning quality. We do not run a
separate "planning success" eval in v1; it would conflate two things.

In v2, a planning-specific eval may be added: "given a held-out
single-edit variant, can the planner recover that edit from a target
state?" This is a clean test of the joint predictor + planner stack.

## 5. Unresolved questions

- The right value of `λ` (edit-cost weight) for typical applications.
  v1 defaults to 0; users override per task.
- Whether to expose a streaming API that yields intermediate
  candidates as they are evaluated, for interactive UIs.
- Whether to support multi-objective planning (Pareto frontier over
  target distance and edit cost). Probably v2.
- Whether to provide a probabilistic guarantee on planning quality
  (e.g., "with probability ≥ 0.95, the returned plan is within 10% of
  the CEM-optimal"). Hard to make non-vacuous; deferred.

## 6. Future work

- A library of canned planning targets (e.g., "the latent neighborhood
  of typical benign variants for gene X", "the cluster of high-
  expression promoter variants") that users can compose into queries.
- Differentiable planning via a smoothed sampler relaxation, for
  end-to-end gradient flow from a downstream loss back through the
  planner. Research-grade.
- Distributed planning over GPU clusters for very large `K` or `n_samples`.
- Integration with the surprise scorer (RFC-0009): plan toward
  low-surprise regions to find variants the model is confident are
  benign.

## 7. Changelog

- 2026-05-20 — Initial draft.
