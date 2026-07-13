# GenoLeWM Corrected Paper Outline

**Status:** Authoritative outline for the post-release validity correction issued 2026-07-10.

**Evidence source:** `paper/EVIDENCE_DOSSIER.md`, with live published artifacts taking precedence over code defaults and historical prose.

## Claim Boundary

The released `v0.2.1-r1` experiment does not validly establish positive or negative model quality.

- Encoder source and target states were raw despite `normalize: true`; predictor outputs were unit norm.
- Released phased source/target/prediction norms were `33.982/33.595/1.000`.
- Released synthetic source/target/prediction norms were `29.253/29.089/1.000`.
- L2 loss, surprise/VEP, rollout L2, and planning-distance interpretations are invalid for the intended normalized method.
- Historical cosine is scale-invariant but confounded by invalid training and train/rollout distribution shift.
- Training sources used global-mean pooling, targets used edit-centered pooling, rollout sources changed to centered pooling, and historical candidates could use different centers.
- Every historical centered pool omitted the leading `<dna>` control token and
  was shifted one hidden token left; some early-window loci centered on that
  control token.
- The pinned Carbon `tokenizer.py` performed an unpinned, network-capable
  `Qwen/Qwen3-4B-Base` load, so its local hash did not establish a self-contained
  runtime identity.
- The run was labeled Phase 2, but Carbon was frozen, no LoRA existed, and target-only KL supplied no gradient.
- No latent-residual, frozen-encoder, capability, inferiority, superiority, or useful-planning conclusion survives.
- Corrected evidence requires a new checkpoint, manifests, reports, and receipt graph under a new identity.

## Title

**GenoLeWM: An Action-Conditioned Latent World Model for Genomic Edits. Post-Release Validity Correction for `v0.2.1-r1`**

## One-Sentence Contribution

We preserve the included bytes of a semantically invalid release as an auditable
historical artifact, identify the normalization, token-layout, transitive-runtime,
and gradient-path failures precisely, withdraw the original interpretation, and
specify the fail-closed experiment required before scientific claims resume.

## Released Computation

| Field | Released value | Intended or previously reported |
| --- | --- | --- |
| Placed-window length | 4,096 bp | 12,288 bp code default |
| Carbon layer | 20 | final layer (`-1`) |
| Pool radius | 8 tokens (+/-48 bp) | 256 tokens (+/-1,536 bp) |
| Training pooling | global source / centered target | one shared edit center |
| Rollout candidates | candidate-specific centers | one shared comparison center |
| Center token | `edit_locus // 6` (one token left) | validated DNA-content start + `edit_locus // 6` |
| Tokenizer runtime | unpinned transitive Qwen load | committed local pure-DNA implementation |
| Encoder states | raw | unit norm |
| Predictor output | unit norm | unit norm |
| State/action dimensions | 1024 / 64 | 1024 / 512 |
| Predictor | 6 cross + 2 self, hidden 768, FFN 768 | 4 cross + 2 self, hidden 1024, FFN 2048 |
| Run | seed 271828, 10,000 steps, 80,000 samples | seed 104729 / 20,000 steps was earlier lineage |
| Adaptation | `phase2` label, no LoRA | trainable LoRA |
| KL | frozen target only; no gradient | gradient-bearing regularizer |

## Paper Structure

1. **Introduction**
   - State the interventional latent-prediction question.
   - Lead with the validity correction, not a negative result.
   - Separate content-addressed identity from semantic correctness.

2. **Related Work**
   - DNA language/representation models.
   - AlphaGenome as functional-track prediction, not a representation-only model.
   - VEP resources and non-clinical scope.
   - JEPA objectives and gradient-bearing regularization.
   - Latent world models with continuous or discrete controls; GenoLeWM differs through structured genomic edit semantics.

3. **Method**
   - Describe intended and released state contracts separately.
   - Distinguish the historical network-capable hybrid tokenizer from the
     corrected, identity-bound local pure-DNA implementation.
   - Derive the intended center after the leading `<dna>` control token and
     disclose the historical one-hidden-token offset.
   - Record the 4,096 bp released placed-window length and 12,288 bp code default.
   - Show raw encoder states versus unit predictor outputs.
   - Explain the constant target-only KL and absent LoRA path.

4. **Systems and Reproducibility**
   - Describe checksum identities and replay graph.
   - State that receipts did not verify tensor normalization, token-layout offsets, transitive tokenizer closure, pooling-coordinate compatibility, gradient flow, or distribution compatibility.
   - Require semantic invariant gates for the corrected lineage.

5. **Experiments as Historical Measurements**
   - Preserve released bytes and signed deltas without method-quality interpretation.
   - Mark VEP and L2 rows invalid for the intended method.
   - Mark cosine rows historical but confounded.
   - Report planning as manifest-backed code-path execution with an invalid mixed-scale objective.
   - Report cold scoring timing without attributing unprofiled component costs.
   - Report AR timing from the published command: H200/CUDA bf16, batch 8, `d_state=1024`, `d_action=64`, benchmark-specific hidden 1024 / 6+1 blocks / FFN 4096.

6. **Discussion**
   - Established: normalization violation, control-token offset,
     pooling-coordinate mismatch, unpinned transitive tokenizer dependency,
     cache-key ambiguity, and no-gradient KL.
   - Unresolved: optimization, action use, distribution shift, encoder geometry, and learnability.
   - Withdraw every causal mechanism proposed by the original manuscript.

7. **Limitations**
   - Semantic validity failure.
   - Historical runtime identity excluded an unpinned transitive Qwen tokenizer.
   - Small slices and narrow coverage.
   - 4,096 bp released windows versus 12,288 bp default.
   - No matched warm-cache benchmark or component profile.
   - AR micro-benchmark is not exact-checkpoint or end-to-end timing.
   - The BRCA2 assay is Sahu et al. (2025), bound through MaveDB
     `urn:mavedb:00001242-a-1`; the historical 32-row score remains invalid
     because of the mixed-norm residual, not because its label source is unknown.

8. **Historical Released Measurements**
   - Keep values tied to original identities.
   - Do not relabel or overwrite the invalidated run.

9. **Reproducibility**
   - Reproduction of old bytes is forensic reproduction of the flawed computation.
   - Corrected reproduction requires a new identity and invariant reports.

10. **Future Work**
    - Run correction control D0 before any mechanistic extension.

11. **Conclusion**
    - Scientific question remains open.

## Figure Contracts

1. Released computation graph with raw source/target and unit prediction;
   caption discloses the one-token centering offset; no overlapping annotations.
2. Historical VEP rows labeled invalid as a method comparison.
3. Historical rollout cosine labeled confounded; mechanism withdrawn.
4. Cold-process timing only; warm and latent regimes shown without numeric estimates.
5. AR speedup against targets with the published H200 synthetic-tensor command scope.

## Table Contracts

1. Historical VEP rows, with BRCA2 continuous scores and TraitGym binary labels distinguished.
2. Historical rollout rows plus measured state norms.
3. Cold scoring and AR implementation timing with precise scope.
4. Invalidated artifact identity, excluded transitive tokenizer dependency, and
   new-identity requirement.

## Corrected Experiment D0

1. Apply one explicit normalization view to live and cached source/target states.
2. Tokenize only from a committed local pure-DNA implementation and validate
   Carbon's control-token/six-mer layout.
3. Pool every compared state at the token-layout-derived shared edit center and
   commit that center in cache identity.
4. Assert state norms at training, scoring, rollout, and planning boundaries.
5. Run a true frozen-encoder Phase 1 control first.
6. If Phase 2 is enabled, require actual LoRA parameters and a nonzero KL gradient into them.
7. Match training and rollout distributions or pre-register the shift.
8. Include source-state and shuffled-action controls.
9. Use powered slices and repeated seeds.
10. Publish under a new content-addressed identity.

## Forbidden Claims

- No clinical, diagnostic, privacy, deployment, or runtime-assurance claim.
- No superiority or inferiority claim from `v0.2.1-r1`.
- No latent-residual-trap or frozen-encoder causal diagnosis from the released run.
- No useful planning claim.
- No claim that cold-process latency identifies component cost.
- No claim that the AR micro-benchmark measures the exact checkpoint or full pipeline.
- No claim that the historical Carbon runtime was self-contained or fully local.
