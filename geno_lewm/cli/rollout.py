# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-rollout`` — multi-edit haplotype rollout (RFC-0018 §3.3).

Phase 1 stub. The real rollout CLI follows once the AR predictor
rollout API lands with issue #42.
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-rollout",
    help_text=(
        "Multi-edit haplotype rollout (RFC-0018 §3.3). Phase 1 stub; "
        "the rollout CLI follows from issue #42."
    ),
    command="rollout",
    issue="#42",
)

cli_main = make_cli_main(app)
