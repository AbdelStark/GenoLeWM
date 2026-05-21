# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-update`` — explicit, user-initiated model updates (RFC-0018 §3.3).

Phase 1 stub. Lands with issue #73 on top of the deploy runtime contract
(#66). The update path is the only place in the runtime allowed to make
network calls (RFC-0010 §3.7) and the only command the
``check_network_confined`` lint allowlists.
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-update",
    help_text=(
        "Explicit, user-initiated model update (RFC-0018 §3.3). Phase 1 stub; lands with issue #73."
    ),
    command="update",
    issue="#73",
)

cli_main = make_cli_main(app)
