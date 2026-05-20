# GenoLeWM RFCs

This directory holds the design RFCs for GenoLeWM. RFCs are the source
of truth for individual design decisions. The
[top-level SPECIFICATION](../SPECIFICATION.md) synthesizes them into a
single canonical view.

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

| #    | Title                                                | Status   |
|------|------------------------------------------------------|----------|
| 0001 | [Project scope and goals](0001-project-scope-and-goals.md)              | Draft    |
| 0002 | [State encoder: Carbon integration](0002-state-encoder-carbon-integration.md) | Draft    |
| 0003 | [Action representation: genomic edits](0003-action-representation-genomic-edits.md) | Draft    |
| 0004 | [Predictor architecture](0004-predictor-architecture.md)                | Draft    |
| 0005 | [Training objective](0005-training-objective.md)                        | Draft    |
| 0006 | [Data pipeline](0006-data-pipeline.md)                                  | Draft    |
| 0007 | [Evaluation suite](0007-evaluation-suite.md)                            | Draft    |
| 0008 | [Latent planning](0008-latent-planning.md)                              | Draft    |
| 0009 | [Surprise-based pathogenicity scoring](0009-surprise-based-pathogenicity-scoring.md) | Draft    |
| 0010 | [On-device personal-genome deployment](0010-on-device-personal-genome-deployment.md) | Draft    |
| 0011 | [Verifiable inference and attestation](0011-verifiable-inference-attestation.md) | Draft    |

---

## Reading order

If you read them in numerical order, each RFC only assumes the ones
before it. Specifically:

- **0001** sets the scope. Read first.
- **0002, 0003, 0004** are the three component RFCs (state encoder,
  action encoder, predictor). They define the architectural primitives.
- **0005, 0006, 0007** define how those primitives are trained, fed,
  and measured.
- **0008, 0009** define the two main downstream uses (planning and
  surprise scoring).
- **0010, 0011** define how the trained system is deployed and made
  verifiable. These are the freedom-tech RFCs.

---

## Conventions

- RFC numbers are zero-padded to four digits.
- File names are kebab-case after the number.
- Each RFC follows the template in `0000-template.md`.
- Mathematical notation uses lowercase italic for scalars (`s`, `a`),
  bold lowercase for vectors (typically rendered as plain in markdown:
  `s_t`, `a_emb`), and uppercase for sets / matrices.
- `→` denotes function mapping; `≜` denotes definition.
