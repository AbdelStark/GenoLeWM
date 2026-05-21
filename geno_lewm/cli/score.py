# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-score`` — score a single variant or a VCF (RFC-0018 §3.3).

Phase 1 stub. The real scoring CLI lands with issue #65 (geno-lewm-score
scaffold) on top of the surprise-score implementation (#62).
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-score",
    help_text=(
        "Score a single variant or a VCF (RFC-0018 §3.3). Phase 1 stub; "
        "the scoring CLI lands with issue #65."
    ),
    command="score",
    issue="#65",
    default_config_name="score",
)

cli_main = make_cli_main(app)
