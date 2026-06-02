# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-export`` — ONNX / Core ML / GGUF export (RFC-0018 §3.3).

Phase 1 stub. Lands with issue #71 on top of the per-target export
modules (#67 ONNX, #68 Core ML, #69 GGUF) and the int8/int4
quantization (#70).
"""

from __future__ import annotations

from geno_lewm.cli._stub_main import (
    build_stub_app as _build_stub_app,
    make_cli_main as _make_cli_main,
)

__all__ = ["app", "cli_main"]

app = _build_stub_app(
    name="geno-lewm-export",
    help_text=(
        "Export a trained checkpoint to a deployable format (RFC-0018 §3.3). "
        "Phase 1 stub; lands with issue #71."
    ),
    command="export",
    issue="#71",
)

cli_main = _make_cli_main(app)
