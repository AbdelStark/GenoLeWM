# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-train`` — train the predictor (RFC-0018 §3.3).

Phase 1 stub. The real trainer lands with issue #44 (training scaffold)
and resolves config from ``geno_lewm/config/defaults/train.yaml`` plus
the shared override flags documented by :mod:`geno_lewm.cli._dispatch`.
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-train",
    help_text=(
        "Train the predictor end-to-end (RFC-0018 §3.3). Phase 1 stub; "
        "the trainer scaffold lands with issue #44."
    ),
    command="train",
    issue="#44",
)

cli_main = make_cli_main(app)
