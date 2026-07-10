# SPDX-License-Identifier: Apache-2.0
"""Executable training-phase capability checks."""

from __future__ import annotations

from typing import Final

from geno_lewm.config import GenoLeWMConfig
from geno_lewm.errors import RuntimeSetupError

PHASE2_ADAPTER_UNAVAILABLE_CODE: Final = "training_config.phase2_adapter_unavailable"
PHASE2_ADAPTER_UNAVAILABLE_MESSAGE: Final = (
    "phase2 requires a graph-preserving trainable encoder-adapter path; "
    "the current Carbon trainer freezes and detaches encoder states"
)


def require_executable_training_phase(config: GenoLeWMConfig, *, boundary: str) -> None:
    """Reject Phase 2 until encoder adapters participate in optimization."""
    if config.phase != "phase2":
        return
    raise RuntimeSetupError(
        PHASE2_ADAPTER_UNAVAILABLE_MESSAGE,
        details={"phase": config.phase, "boundary": boundary},
        remediation=(
            "use phase1 for frozen-Carbon training, or implement graph-preserving LoRA "
            "with an optimizer-owned adapter group and a normalization-compatible "
            "regularizer view"
        ),
    )
