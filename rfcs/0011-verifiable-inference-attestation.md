# RFC-0011: Verifiable inference and attestation

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0002, RFC-0004, RFC-0009, RFC-0010
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

Inference on personal-genome data is the highest-trust setting consumer
AI has. A pathogenicity score that influences a person's decisions
about their own genome must be verifiable: the user, or anyone they
share results with, must be able to confirm that the score was produced
by the specific published model weights running on the specific input,
without having to re-run the inference themselves. This RFC specifies
the attestation primitives GenoLeWM exposes (content-addressed model
identifiers, input commitments, output receipts, and the STARK proof
target for the predictor forward pass), the JSON receipt format, and
the verification protocol. Full STARK proving of the inference path is
the North Star; the lighter-weight ingredients are usable today.

## 2. Motivation

Three asymmetries make verifiable inference a category-defining feature
for personal-health AI, not a nice-to-have.

1. **The inference is private but the result is public.** A user runs
   GenoLeWM on their own variants locally. They may then share a
   surprise score with a genetic counselor, a researcher, or a family
   member. Without attestation, the recipient cannot verify the score
   was produced by the published model; the user could have produced
   any number, perhaps unintentionally (a buggy local build) or
   intentionally (a tampered model).

2. **Model identity is the trust anchor.** Once we accept that the
   inference itself is private, model identity becomes the entire
   trust story. "This score came from `geno-lewm-v0.1.0-carbon-500m-r1`
   running on this exact input" is a verifiable claim that lets the
   recipient reason about the score's properties (training distribution,
   eval performance, known limitations).

3. **STARK proofs are the right cryptographic primitive for ML.**
   Among the established verifiable-computation schemes, STARKs are
   post-quantum, do not require trusted setup, and are well-matched to
   the arithmetic-circuit shape of Transformer forward passes. They are
   the natural cryptographic substrate. (We make this argument fully in
   §4.3 and acknowledge that practical STARK proving of Transformer
   inference is still research-grade.)

The RFC is structured to deliver the **lightweight ingredients** in v1
(content-addressed model identifiers, input commitments, output
receipts) and to specify the **full STARK proving** track as the
Phase 4 North Star.

## 3. Specification

### 3.1 Content-addressed model identifiers

Every GenoLeWM artifact is identified by a triple of content hashes:

```
GenoLeWM-MID = (encoder_hash, predictor_hash, action_encoder_hash)
            = SHA-256(carbon weights) || SHA-256(predictor weights)
              || SHA-256(action_encoder weights)
```

These hashes are computed over the canonical serialized weights
(`safetensors` format, in canonical key order) and recorded in the
checkpoint's `manifest.json`. The manifest itself is also hashed.

A GenoLeWM model's full identifier is the SHA-256 of its manifest:

```
model_id = SHA-256(canonical_json(manifest))
```

`model_id` is what published results cite. It is reproducible byte-for-
byte from the public weights and the manifest schema specified in §3.7.

### 3.2 Input commitment

For every inference call, the inputs are committed via:

```
input_commitment = SHA-256(
    canonical_serialize(
        reference_window || edit_spec || pooling_config || dtype_config
    )
)
```

The commitment binds the inference output to:

- The exact reference window bases (12,288 bp string).
- The edit specification (chromosome, position, ref, alt).
- The state-encoder configuration (state layer, pool type, pool radius,
  normalization).
- The numerical precision used (bf16 / fp16 / int8 / int4).

Two runs with identical inputs produce identical commitments. Two runs
with different inputs (even in nominally-equivalent ways, e.g., a
different state layer) produce different commitments.

### 3.3 Output receipt

Every scoring call emits a receipt:

```json
{
  "schema_version": "1.0.0",
  "model_id": "sha256:...",
  "input_commitment": "sha256:...",
  "output": {
    "sigma_raw": 0.347,
    "sigma_calibrated": 0.92,
    "bucket_id": "coding_missense|mid|none",
    "confidence": 1.0,
    "low_confidence": false
  },
  "output_commitment": "sha256:...",
  "calibration_hash": "sha256:...",
  "runtime": {
    "backend": "coreml",
    "device": "Apple M3 Max",
    "geno_lewm_version": "0.1.0",
    "carbon_revision": "main@<sha>"
  },
  "timestamp": "2026-MM-DDTHH:MM:SSZ",
  "attestation": {
    "kind": "checksum_only",
    "details": null
  }
}
```

The `attestation.kind` field grows over time:

- `checksum_only` (v1): no cryptographic proof; the receipt is a
  reproducibility token. Anyone with the same weights and inputs can
  re-run and compare.
- `tee` (v1.1, optional): the receipt is signed by a Trusted Execution
  Environment (e.g., Apple Secure Enclave, Intel SGX). The recipient
  trusts the TEE vendor.
- `stark` (Phase 4): the receipt includes a STARK proof of the
  predictor forward pass. The recipient verifies the proof without
  trusting either us or any hardware vendor.

The receipt format is forward-compatible: older verifiers ignore newer
attestation kinds gracefully, with a clear "unverified attestation
kind" message.

### 3.4 Verification protocol (v1, checksum-only)

A receipt verifier:

1. Parses the receipt JSON and checks schema validity.
2. Fetches the model with the receipt's `model_id` from the Hugging
   Face Hub (or a local mirror).
3. Re-computes the model's manifest hash; rejects if it does not match
   `model_id`.
4. If the input is provided alongside the receipt, recomputes
   `input_commitment` and checks it matches the receipt.
5. Optionally re-runs inference and confirms the output bit-matches.

Verification time:

- Without re-running inference: < 1 second (manifest hash + commitment
  check).
- With re-running inference (full bit-match): inference time on the
  verifier's hardware.

Bit-exact reproducibility across hardware is hard for floating-point
inference. The runtime documents which hardware/backend combinations
produce bit-exact results vs which produce nominally-equivalent results.

### 3.5 STARK target: predictor forward pass

The Phase 4 STARK proving target is **the predictor + action encoder
forward pass**, *not* Carbon. Justification:

- Carbon is large (500M params at minimum). Proving 500M-param
  Transformer inference is beyond current STARK feasibility.
- The predictor + action encoder is ~25M params. This is in the
  challenging-but-tractable range for current STARK research.
- Carbon's input (the reference window) and output (the state vector)
  are committed via §3.2. A verifier who trusts the published Carbon
  weights and verifies the commitment can take Carbon's output as a
  trusted intermediate.

So the trust composition is:

```
Trust(Carbon weights, by public commitment + community audit)
  ⊕ Verify(predictor forward pass, by STARK proof)
  = Verify(GenoLeWM score)
```

This is a pragmatic decomposition: it concentrates the cryptographic
work on the GenoLeWM-specific component and treats Carbon as a public
input.

### 3.6 STARK circuit specification (Phase 4)

The STARK circuit proves:

```
Given:
    public:  predictor_weights_hash, action_encoder_weights_hash,
             input_state_commitment, input_action_commitment,
             output_commitment
    private: predictor_weights, action_encoder_weights,
             input_state, input_actions, output

The prover knows:
    weights matching the public hashes,
    inputs matching the public commitments,
    and the output is the deterministic result of running the
    predictor + action encoder forward pass on those inputs with
    those weights, in the specified numerical precision.
```

The circuit's main components are:

- Hash equality checks (Poseidon / Rescue over the STARK field for
  efficiency; Merkle commitment to the weight tensors).
- Quantized integer arithmetic for the predictor (matrix multiplies,
  attention softmax, GELU, LayerNorm).
- Fixed-precision arithmetic constraints (int8 quantization specified
  exactly).
- Lookup arguments for non-linear ops (softmax, GELU).

Recent advances in STARK arithmetization for fixed-point ML inference
(Circle STARKs over Mersenne primes, lookup-heavy circuits, jolt-style
zkVMs targeting RISC-V) make this category feasible in 2026.

The reference circuit will be a sequence of constraints expressible in
the Cairo intermediate representation, leveraging the Starknet
ecosystem's STARK proving infrastructure. This is a deliberate
alignment with the wider verifiable-compute ecosystem, not a
GenoLeWM-internal choice.

### 3.7 Manifest schema

`manifest.json` is a canonical JSON document with the following schema:

```json
{
  "schema_version": "1.0.0",
  "model_name": "geno-lewm",
  "model_version": "0.1.0",
  "release_id": "geno-lewm-v0.1.0-carbon-500m-r1",
  "encoder": {
    "id": "HuggingFaceBio/Carbon-500M",
    "revision": "main@<sha>",
    "hash": "sha256:..."
  },
  "predictor": {
    "architecture_id": "cross-attention-4x2-d1024",
    "file": "predictor.safetensors",
    "hash": "sha256:...",
    "dtype": "bf16"
  },
  "action_encoder": {
    "file": "action_encoder.safetensors",
    "hash": "sha256:...",
    "dtype": "bf16"
  },
  "calibration": {
    "file": "calibration.parquet",
    "hash": "sha256:...",
    "version": "1.0.0"
  },
  "training": {
    "config_file": "train_config.yaml",
    "hash": "sha256:...",
    "data_snapshot": {
      "corpus_id": "HuggingFaceBio/carbon-pretraining-corpus",
      "corpus_revision": "main@<sha>",
      "gnomad_release": "v4.1",
      "clinvar_release": "2026-04-15"
    }
  },
  "eval": {
    "report_file": "eval_report.md",
    "hash": "sha256:..."
  }
}
```

Canonical-JSON serialization: keys sorted lexicographically, no
whitespace, UTF-8.

### 3.8 Threat model

The attestation system defends against:

| Threat | Defense |
|--------|---------|
| User unintentionally runs a tampered build | `model_id` mismatch detected at verification |
| User intentionally fabricates a score | Output commitment + (Phase 4) STARK proof |
| Backend bug produces wrong output | Bit-exact re-run by verifier (when supported) |
| Calibration table substitution | Calibration hash in manifest + receipt |
| Replay of old receipts | Receipts include timestamps; consumers check freshness if relevant |

The system does **not** defend against:

| Threat | Why |
|--------|-----|
| Backdoored Carbon weights | Carbon is a third-party dependency; we cite the published hash |
| Bugs in the model itself | This is a quality problem, not a verification problem |
| Misinterpretation of correctly-computed scores | This is a UX / documentation problem |

### 3.9 Receipt storage and sharing

The runtime stores receipts at `{output_path}.receipt.json` adjacent to
the scoring output. Sharing a score therefore means sharing the
receipt; the recipient runs `geno-lewm verify receipt.json` to confirm
authenticity.

For batch VCF scoring, a single receipt is produced per VCF that
commits to the input VCF hash and the output Parquet hash, with the
manifest hash. Per-variant receipts are not produced by default (the
volume would be prohibitive); they are available via a `--per-variant-
receipts` flag.

## 4. Rationale and alternatives

### 4.1 Why verifiable inference for personal-health AI?

The personal-health setting is the worst case for opaque AI. The user
has the most reason to want certainty, the least technical capacity to
re-verify, and the most consequential downstream decisions. Verifiable
inference is the structural answer to this asymmetry, in the same way
that public-key cryptography is the structural answer to authenticating
remote parties.

### 4.2 Why STARKs rather than SNARKs?

- **No trusted setup.** SNARKs (Groth16, PLONK) require a trusted-setup
  ceremony. For a personal-health-AI artifact, every additional trust
  anchor is a liability.
- **Post-quantum security.** STARKs rest on hash-function assumptions,
  not pairings or discrete logarithms. The half-life of any
  cryptographic assumption in personal-health contexts matters.
- **Arithmetic-circuit shape.** STARK constraint systems handle the
  large multiplicative complexity of Transformer inference more
  gracefully than the linear-circuit shape SNARKs prefer.

For a fuller argument, see the StarkWare "Integrity Thesis" memo and
the wider literature on STARK arithmetization for ML inference.

### 4.3 Why is Carbon not in the STARK circuit?

Practical feasibility. Proving 500M-parameter Transformer inference is
beyond the current state of the art. Proving 25M-parameter Transformer
inference is on the research frontier, but tractable. The pragmatic
decomposition (trust Carbon by public commitment, verify predictor by
proof) is the right v1 / Phase 4 compromise. If practical STARK proving
of foundation-model-class inference becomes feasible (an active
research area in 2026), Carbon can be moved into the circuit later.

### 4.4 Why optional TEE attestation as an intermediate step?

TEE attestation (Apple Secure Enclave, Intel SGX, AMD SEV-SNP) is
production-mature today. It is a weaker primitive than STARK proving
because it requires trusting the TEE vendor, but it is available
*now* and gives a meaningful trust improvement over checksum-only
attestation. Shipping it as a v1.1 intermediate, while the STARK track
matures, gives users a real-world option immediately.

### 4.5 Why a separate receipt format rather than embedding in the
output?

Separation lets receipts be stored, shared, and verified independently
of the scoring output. A user might publish a result in a paper and
attach the receipt as a supplementary file; the receipt is the
proof-of-provenance, the output is the science. Coupling them would
force the consumer to handle the entire payload to verify any part.

### 4.6 Why canonical JSON for the manifest?

Canonical JSON has a unique byte representation per logical document,
which is essential for hash stability. The trade-off vs richer formats
(CBOR, Protobuf) is human-readability; for a manifest that humans need
to audit, JSON wins.

## 5. Unresolved questions

- The exact Cairo / STARK toolchain to target. Likely Stwo or a
  successor, but the ecosystem is moving fast.
- Quantization-precision policy for STARK proving: int8 is the natural
  precision target, but the v1 model is bf16. We may need a separate
  "STARK-friendly" quantized variant alongside the deployment variant.
- Whether to provide a hosted verification service for users without
  the technical capacity to verify locally. Goes against the freedom-tech
  framing but reaches more people; reluctantly deferred.
- How to handle revoked model versions: if a model is found to have a
  serious bug, how do receipts for that model get flagged? Probably
  via a published revocation list, but the mechanism needs design.

## 6. Future work

- Recursive STARK proofs: proving multiple variant scorings under a
  single aggregated proof, for batch VCFs.
- Differential privacy guarantees on the receipt: prove that the score
  was computed, without revealing the variant. Genuinely useful for
  shared family-genetics scenarios.
- Cross-model attestation: prove "this score matches one of the
  GenoLeWM models on this approved list", giving flexibility while
  preserving authenticity.
- Public bulletin board for verifiable inference: third parties
  publish receipts to a public log; the log itself is cryptographically
  auditable. Aligned with the broader Bitcoin / cypherpunk transparency
  tradition.

## 7. Changelog

- 2026-05-20 — Initial draft.
