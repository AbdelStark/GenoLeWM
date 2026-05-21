# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-prepare-gnomad`` — gnomAD shard builder (RFC-0018 §3.3).

Phase 1 stub. Lands with issue #49.
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-prepare-gnomad",
    help_text=(
        "Build a gnomAD shard from a public release (RFC-0018 §3.3). "
        "Network downloads are gated on a user-visible URL preview. "
        "Phase 1 stub; lands with issue #49."
    ),
    command="prepare-gnomad",
    issue="#49",
)

cli_main = make_cli_main(app)
