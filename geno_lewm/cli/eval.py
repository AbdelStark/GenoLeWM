# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-eval`` — run a single benchmark (RFC-0018 §3.3).

Phase 1 stub. The real eval CLI lands with issue #56, on top of the
ClinVar VEP harness (#53) and the Carbon zero-shot baseline (#55).
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-eval",
    help_text=(
        "Run a single evaluation benchmark (RFC-0018 §3.3). Phase 1 stub; "
        "the eval CLI lands with issue #56."
    ),
    command="eval",
    issue="#56",
)

cli_main = make_cli_main(app)
