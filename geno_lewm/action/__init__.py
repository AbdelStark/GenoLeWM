"""Action representation for GenoLeWM.

Public surface defined by RFC-0003. Today the package ships only the
canonical edit types — encoders and samplers land in #38, #39, #40.
"""

from geno_lewm.action.spec import EditSpec, EditType, RelEdit, V1_MAX_LEN

__all__ = ["EditSpec", "EditType", "RelEdit", "V1_MAX_LEN"]
