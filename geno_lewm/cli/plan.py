# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-plan`` — CEM-based latent planning (RFC-0018 §3.3).

Phase 1 stub. The real planning CLI lands with issue #61, on top of
the CEM solver (#59) and the cost-function library (#60).
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-plan",
    help_text=(
        "Latent planning via the cross-entropy method (RFC-0018 §3.3). "
        "Phase 1 stub; the planner lands with issue #61."
    ),
    command="plan",
    issue="#61",
    default_config_name="plan",
)

cli_main = make_cli_main(app)
