# SPDX-License-Identifier: Apache-2.0
"""GenoLeWM command-line entry points.

The CLI package has a mixed alpha surface:

* implemented release/demo paths such as verify, update, data prep,
  score, train preflight/smoke/launch, evaluation, baseline scoring,
  rollout-fidelity metric aggregation, CEM planning, export, and cache
  repair/reindex;
* explicit entry-point scaffolds for remaining target-specific export
  surfaces, which remain tracked by their subsystem issues.
"""

__all__: list[str] = []
