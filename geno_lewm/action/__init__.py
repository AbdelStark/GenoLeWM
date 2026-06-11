# SPDX-License-Identifier: Apache-2.0
"""Action representation for GenoLeWM.

Public surface defined by edit contract. The package ships the canonical
edit types, pure-Python apply functions, synthetic samplers, and the
optional PyTorch action encoder.
"""

from geno_lewm.action.apply import apply_edit, apply_edits
from geno_lewm.action.encoder import ActionEncoder
from geno_lewm.action.spec import V1_MAX_LEN, EditSpec, EditType, RelEdit
from geno_lewm.action.synthetic import DEFAULT_EDGE_MARGIN, indel, mnv, uniform_snv

__all__ = [
    "DEFAULT_EDGE_MARGIN",
    "V1_MAX_LEN",
    "ActionEncoder",
    "EditSpec",
    "EditType",
    "RelEdit",
    "apply_edit",
    "apply_edits",
    "indel",
    "mnv",
    "uniform_snv",
]
