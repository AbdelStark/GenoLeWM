# SPDX-License-Identifier: Apache-2.0
"""Compatibility rules for encoder state semantics across checkpoint lineages."""

from __future__ import annotations

from geno_lewm.errors import InputError

LEGACY_RAW_STATE_CONTRACT = "legacy_raw_v1"
L2_NORMALIZED_STATE_CONTRACT = "l2_normalized_v2"


def encoder_uses_normalized_states(encoder_config: object) -> bool:
    """Resolve the effective normalization view for an encoder config."""
    normalize = getattr(encoder_config, "normalize", None)
    if not isinstance(normalize, bool):
        raise InputError("encoder.normalize must be boolean")
    state_contract = getattr(
        encoder_config,
        "state_contract_version",
        LEGACY_RAW_STATE_CONTRACT,
    )
    if state_contract == LEGACY_RAW_STATE_CONTRACT:
        return False
    if state_contract == L2_NORMALIZED_STATE_CONTRACT:
        if not normalize:
            raise InputError(
                "l2_normalized_v2 requires encoder.normalize=true",
                remediation="use legacy_raw_v1 for raw states or enable encoder normalization",
            )
        return True
    raise InputError(
        "unsupported encoder state_contract_version",
        details={"state_contract_version": state_contract},
    )
