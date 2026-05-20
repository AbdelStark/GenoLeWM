# GenoLeWM RFCs

This directory holds the design RFCs for GenoLeWM. RFCs are the source
of truth for individual design decisions. The
[top-level SPECIFICATION](../SPECIFICATION.md) and
[`docs/spec/`](../docs/spec/) corpus synthesize them into a single
canonical view.

A mirrored path `docs/rfcs/` symlinks to this directory so that both
naming conventions resolve correctly.

---

## Process

1. **Drafting.** Copy `0000-template.md` to a new file named
   `NNNN-short-title.md`, where `NNNN` is the next unused number.
   Set status to `Draft`. Open a PR.
2. **Discussion.** PR review on GitHub. Reviewers leave inline comments.
3. **Acceptance.** When the author and at least one core reviewer agree,
   status moves to `Accepted` and the PR is merged.
4. **Implementation.** Acceptance does not require code. Implementation
   may happen later. The RFC's `Implementation status` field is updated
   when code lands.
5. **Amendment.** Non-trivial changes to an accepted RFC require a new
   PR that bumps the RFC's `Updated` date and documents the diff in a
   trailing changelog section.
6. **Supersession.** If a new RFC replaces an old one, the new RFC's
   `Supersedes` field names the old one, and the old one's status moves
   to `Superseded` with a back-link.

---

## Index

| #    | Title | Status | Subsystem |
|------|-------|--------|-----------|
| 0001 | [Project scope and goals](0001-project-scope-and-goals.md) | Draft | scope |
| 0002 | [State encoder — Carbon integration](0002-state-encoder-carbon-integration.md) | Draft | encoder |
| 0003 | [Action representation — genomic edits](0003-action-representation-genomic-edits.md) | Draft | action |
| 0004 | [Predictor architecture](0004-predictor-architecture.md) | Draft | predictor |
| 0005 | [Training objective](0005-training-objective.md) | Draft | training |
| 0006 | [Data pipeline](0006-data-pipeline.md) | Draft | data |
| 0007 | [Evaluation suite](0007-evaluation-suite.md) | Draft | eval |
| 0008 | [Latent planning](0008-latent-planning.md) | Draft | planning |
| 0009 | [Surprise-based pathogenicity scoring](0009-surprise-based-pathogenicity-scoring.md) | Draft | surprise |
| 0010 | [On-device personal-genome deployment](0010-on-device-personal-genome-deployment.md) | Draft | deploy |
| 0011 | [Verifiable inference and attestation](0011-verifiable-inference-attestation.md) | Draft | attestation |
| 0012 | [Error taxonomy and exception hierarchy](0012-error-taxonomy.md) | Draft | cross-cutting |
| 0013 | [Observability — logging, metrics, redaction](0013-observability.md) | Draft | cross-cutting |
| 0014 | [Public Python API surface and stability policy](0014-public-api-and-stability.md) | Draft | cross-cutting |
| 0015 | [Testing strategy and CI gates](0015-testing-strategy.md) | Draft | cross-cutting |
| 0016 | [Performance budget and profiling discipline](0016-performance-budget.md) | Draft | cross-cutting |
| 0017 | [Configuration system](0017-configuration-system.md) | Draft | cross-cutting |
| 0018 | [CLI design](0018-cli-design.md) | Draft | cli |
| 0019 | [Reference desktop app skeleton](0019-reference-desktop-app.md) | Draft | desktop |

---

## Reading order

Each RFC only assumes the ones before it. Specifically:

- **0001** sets the scope. Read first.
- **0002, 0003, 0004** are the three component RFCs (state encoder,
  action encoder, predictor). They define the architectural primitives.
- **0005, 0006, 0007** define how those primitives are trained, fed,
  and measured.
- **0008, 0009** define the two main downstream uses (planning and
  surprise scoring).
- **0010, 0011** define how the trained system is deployed and made
  verifiable. The freedom-tech RFCs.
- **0012–0017** define cross-cutting engineering concerns (errors,
  observability, API stability, testing, performance, configuration).
  They cut across the subsystem RFCs and lock the load-bearing decisions
  about how the engineering layer behaves.
- **0018, 0019** define the user-facing surfaces (CLI and reference
  desktop app).

For a 30-minute path: 0001 → 0002 → 0004 → 0005 → 0007 → 0010 → 0011.

---

## Conventions

- RFC numbers are zero-padded to four digits.
- File names are kebab-case after the number.
- Each RFC follows the template in `0000-template.md`.
- Mathematical notation: lowercase italic for scalars (`s`, `a`),
  lowercase plain for vectors (`s_t`, `a_emb`), uppercase for sets /
  matrices.
- `→` denotes function mapping; `≜` denotes definition.

## Open-question registry

Each RFC's open questions are tagged with an `OQ-<area>-<n>` identifier;
the same identifiers appear in the spec corpus under "Open questions"
sections. The aggregate live list is in
[`docs/roadmap/IMPLEMENTATION.md`](../docs/roadmap/IMPLEMENTATION.md).
