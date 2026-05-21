# SPDX-License-Identifier: Apache-2.0
"""Action representation for GenoLeWM.

Public surface defined by RFC-0003. The package ships the canonical
edit types and the pure-Python apply functions; the action encoder
(#39) and synthetic samplers (#40) land in follow-up issues.
"""

from geno_lewm.action.apply import apply_edit, apply_edits
from geno_lewm.action.spec import V1_MAX_LEN, EditSpec, EditType, RelEdit
from geno_lewm.action.synthetic import DEFAULT_EDGE_MARGIN, indel, mnv, uniform_snv

__all__ = [
    "DEFAULT_EDGE_MARGIN",
    "V1_MAX_LEN",
    "EditSpec",
    "EditType",
    "RelEdit",
    "apply_edit",
    "apply_edits",
    "indel",
    "mnv",
    "uniform_snv",
]
