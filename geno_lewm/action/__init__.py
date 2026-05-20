"""Action representation for GenoLeWM.

Public surface defined by RFC-0003. The package ships the canonical
edit types and the pure-Python apply functions; the action encoder
(#39) and synthetic samplers (#40) land in follow-up issues.
"""

from geno_lewm.action.apply import apply_edit, apply_edits
from geno_lewm.action.spec import EditSpec, EditType, RelEdit, V1_MAX_LEN

__all__ = [
    "EditSpec",
    "EditType",
    "RelEdit",
    "V1_MAX_LEN",
    "apply_edit",
    "apply_edits",
]
