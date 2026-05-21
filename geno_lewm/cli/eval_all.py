# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-eval-all`` — release-grade eval report (RFC-0018 §3.3).

Phase 1 stub. Lands with issue #56 alongside the per-benchmark eval
CLI; the writer that emits the Markdown release report attaches once
the rollout-fidelity harness (#57) is in place.
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-eval-all",
    help_text=(
        "Run the full release-grade eval suite and emit a Markdown report "
        "(RFC-0018 §3.3). Phase 1 stub; lands with issue #56."
    ),
    command="eval-all",
    issue="#56",
)

cli_main = make_cli_main(app)
