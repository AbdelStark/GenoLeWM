# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-cache-windows`` — window-cache management (RFC-0018 §3.3).

Phase 1 stub. Lands with issue #36 on top of the Parquet shard cache
(#35).
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import build_stub_app, make_cli_main

app = build_stub_app(
    name="geno-lewm-cache-windows",
    help_text=(
        "Build / repair / reindex the window cache over the Parquet shards "
        "(RFC-0018 §3.3). Phase 1 stub; lands with issue #36."
    ),
    command="cache-windows",
    issue="#36",
)

cli_main = make_cli_main(app)
