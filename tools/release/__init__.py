# SPDX-License-Identifier: Apache-2.0
"""Maintainer-side release tooling.

Two mechanical helpers used by the release checklist in
``docs/spec/09-release-and-versioning.md#release-process``:

* :mod:`tools.release.bump` updates the single source of truth for
  the package version (``__version__`` in :mod:`geno_lewm`) and
  cross-checks that ``pyproject.toml`` consumes it dynamically.
* :mod:`tools.release.changelog` rewrites ``CHANGELOG.md`` by lifting
  the ``[Unreleased]`` section to a versioned heading (or by emitting a
  fresh section synthesised from ``git log``) following the
  Keep-a-Changelog 1.1.0 grammar.
* :mod:`tools.release.check_sdist_assets` verifies that source
  distributions include the benchmark harness, first-experiment configs,
  release tooling, docs, RFCs, examples, and agent context needed by the
  documented release gates.
* :mod:`tools.release.eval_report` renders ``eval_report.md`` from
  measured metrics JSON, keeping paper/demo results generated and
  claim-bounded; ``geno-lewm-eval-all --require-v02-vep-metrics`` can
  fail the aggregate when required #197 VEP metric rows are missing before
  the all-up readiness report is generated.
* :mod:`tools.release.efficiency_report` validates measured single-variant
  latency, batched throughput, peak memory, hardware/runtime notes, and
  input identities as ``efficiency_report.json``.
* :mod:`tools.release.v02_benchmark_readiness` reconciles measured eval,
  efficiency, and AR rollout speed artifacts into
  ``v0.2_benchmark_readiness_report.json``; release mode records measured
  values/deltas, checks CI-bearing VEP metrics plus package-relative,
  non-fixture input provenance, and keeps missing or failed #197
  benchmark rows explicit.
* :mod:`tools.release.dataset_snapshot` stages explicit local Carbon,
  gnomAD, and ClinVar inputs, writes ``dataset_snapshot_report.json``,
  then builds the first-experiment dataset package.
* :mod:`tools.release.dataset_package` renders the dataset card,
  normalized metadata, manifest, split-integrity report, and checksum
  file from release metadata plus shard files.
* :mod:`tools.release.dataset_integrity` recomputes dataset file
  identities, split record counts, Parquet variant-key counts, and
  train/eval comparable-key leakage checks from ``dataset_manifest.json``.
* :mod:`tools.release.training_run` renders ``training_run_manifest.json``,
  ``training_run_card.md``, and ``training_run_SHA256SUMS`` for a
  completed training run, including the Carbon training preflight report
  when release metadata names it.
* :mod:`tools.release.model_package` renders normalized
  ``model_package.json``, ``model_card.md``, and ``SHA256SUMS`` from a
  checkpoint manifest plus model-release metadata, while cross-checking
  eval and efficiency evidence identities.
* :mod:`tools.release.batch_receipt_report` verifies terminal-demo score
  and receipt JSONL streams as one generated batch artifact.
* :mod:`tools.release.runtime_preflight` verifies the local runtime,
  native dependency, input-file, and fail-closed network envelope before
  publishing a terminal-demo transcript.
* :mod:`tools.release.paper_draft` generates a first-experiment paper
  draft from the release artifact set.
* :mod:`tools.release.hub_release` dry-runs the Hugging Face Hub upload
  plan after the package verifier passes, including model, dataset, and
  terminal-demo upload file inventories with checksum manifests and
  portable demo paths.
* :mod:`tools.release.hub_publish` runs the credentialed publication
  commands for a verified Hub plan, then regenerates the final
  release-candidate report from public links and fetched public artifact
  bytes.
* :mod:`tools.release.clean_machine_demo` downloads published model,
  dataset, and GitHub release demo artifacts from a ready
  release-candidate report, verifies their hashes, re-runs the package
  verifier on the downloaded artifacts, reruns the terminal demo, and
  records the clean-machine replay evidence. Optional Hub/GitHub tokens
  are scoped to fetch requests and are not serialized.
* :mod:`tools.release.publication_report` writes the final
  ``publication_evidence_report.json`` binder after credentialed Hub
  publication and clean-machine replay agree on the release candidate,
  public artifact downloads, and replay outputs, with live issue
  references on any final publication failures.
* :mod:`tools.release.publication_assets` writes
  ``publication_evidence_assets.json`` for the final evidence files that
  the protected publish workflow attaches to the GitHub release.
* :mod:`tools.release.release_candidate` writes a single publication
  decision report that binds the package verifier, Hub dry-run plan,
  public URL reachability and artifact hash/size checks, commit SHA, and
  key artifact identities, and emits a high-level readiness checklist
  with live issue references for the first paper/demo release.
* :mod:`tools.release.paper_package` validates the first paper/demo
  release artifact set before publication.

These helpers run as ``python -m tools.release.*``. Parquet split
integrity requires ``pyarrow``; the runtime preflight records optional
runtime dependency availability without importing the model package
verifier into those dependencies.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
