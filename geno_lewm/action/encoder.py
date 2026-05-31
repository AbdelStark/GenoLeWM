# SPDX-License-Identifier: Apache-2.0
"""PyTorch action encoder for window-relative genomic edits.

The module keeps PyTorch optional so importing :mod:`geno_lewm.action`
does not pull the training stack into the base package. Instantiate
``ActionEncoder`` only in a ``geno-lewm[train]`` environment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

from geno_lewm.action.spec import V1_MAX_LEN, EditType, RelEdit
from geno_lewm.errors import InputError, RuntimeSetupError, UnsupportedEditError

__all__ = ["ActionEncoder"]

try:  # pragma: no cover - exercised by optional-runtime tests with torch installed.
    import torch  # type: ignore[import-not-found]
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - covered through the lightweight fallback class.
    torch = None
    Tensor = Any
    nn = None


if nn is None:

    class ActionEncoder:
        """Placeholder that reports the missing optional training runtime."""

        def __init__(
            self,
            *,
            d_action: int = 512,
            d_pos: int = 128,
            d_type: int = 64,
            d_seq: int = 256,
            max_window_bp: int = 12_288,
            carbon_tokenizer: Any | None = None,
        ) -> None:
            del d_action, d_pos, d_type, d_seq, max_window_bp, carbon_tokenizer
            raise RuntimeSetupError(
                "ActionEncoder requires PyTorch",
                remediation="install geno-lewm[train] or install torch",
            )

else:  # pragma: no cover - optional torch runtime is validated outside base CI.
    _BASE_TO_BITS: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
    _KMER = 6
    _SEQ_TOKENS = 4
    _TOKEN_EMBED_DIM = 128
    _OOV_TOKEN_ID = 4**_KMER
    _VOCAB_SIZE = _OOV_TOKEN_ID + 1

    class SeqMicroEncoder(nn.Module):  # type: ignore[misc]
        """Shared 6-mer micro-encoder for reference and alternate bases."""

        def __init__(self, *, d_seq: int) -> None:
            super().__init__()
            _require_positive("d_seq", d_seq)
            if d_seq % 4 != 0:
                raise InputError("d_seq must be divisible by 4 so attention heads divide evenly")
            token_dim = min(_TOKEN_EMBED_DIM, d_seq)
            self.token_embedding = nn.Embedding(_VOCAB_SIZE, token_dim, padding_idx=_OOV_TOKEN_ID)
            self.token_projection = (
                nn.Identity() if token_dim == d_seq else nn.Linear(token_dim, d_seq)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_seq,
                nhead=4,
                dim_feedforward=d_seq,
                activation="gelu",
                batch_first=True,
                norm_first=False,
                dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, token_ids: Tensor) -> Tensor:
            hidden = self.encoder(self.token_projection(self.token_embedding(token_ids)))
            return hidden.mean(dim=1)

    class ActionEncoder(nn.Module):  # type: ignore[no-redef,misc]
        """Encode :class:`RelEdit` objects into learned action embeddings."""

        def __init__(
            self,
            *,
            d_action: int = 512,
            d_pos: int = 128,
            d_type: int = 64,
            d_seq: int = 256,
            max_window_bp: int = 12_288,
            carbon_tokenizer: Any | None = None,
        ) -> None:
            super().__init__()
            _require_positive("d_action", d_action)
            _require_positive("d_pos", d_pos)
            _require_positive("d_type", d_type)
            _require_positive("max_window_bp", max_window_bp)
            if d_pos % 2 != 0:
                raise InputError("d_pos must be even for sinusoidal position embeddings")
            self._d_action = d_action
            self.d_pos = d_pos
            self.max_window_bp = max_window_bp
            self.carbon_tokenizer = carbon_tokenizer
            self.type_embedding = nn.Embedding(len(EditType), d_type)
            self.seq_encoder = SeqMicroEncoder(d_seq=d_seq)
            projection_in = d_pos + d_type + (2 * d_seq)
            self.projection = nn.Sequential(
                nn.Linear(projection_in, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
                nn.Linear(1024, d_action),
            )
            self.padding_embedding = nn.Parameter(torch.zeros(d_action))

        @property
        def d_action(self) -> int:
            return self._d_action

        def forward(self, edits: Sequence[RelEdit] | Sequence[Sequence[RelEdit]]) -> Tensor:
            batches = _normalize_batches(edits)
            batch_size = len(batches)
            max_len = max((len(batch) for batch in batches), default=0)
            output = self.padding_embedding.expand(batch_size, max_len, -1).clone()
            flat: list[RelEdit] = [edit for batch in batches for edit in batch]
            if not flat:
                return output

            device = self.padding_embedding.device
            pos = torch.tensor([edit.rel_pos for edit in flat], device=device, dtype=torch.long)
            type_ids = torch.tensor(
                [int(edit.edit_type) for edit in flat], device=device, dtype=torch.long
            )
            ref_ids = torch.tensor(
                [_tokenize_short_dna(edit.ref_bases) for edit in flat],
                device=device,
                dtype=torch.long,
            )
            alt_ids = torch.tensor(
                [_tokenize_short_dna(edit.alt_bases) for edit in flat],
                device=device,
                dtype=torch.long,
            )
            encoded = self.projection(
                torch.cat(
                    (
                        _sinusoidal_positions(pos, d_pos=self.d_pos, max_bp=self.max_window_bp),
                        self.type_embedding(type_ids),
                        self.seq_encoder(ref_ids),
                        self.seq_encoder(alt_ids),
                    ),
                    dim=-1,
                )
            )
            cursor = 0
            for batch_idx, batch in enumerate(batches):
                width = len(batch)
                if width:
                    output[batch_idx, :width] = encoded[cursor : cursor + width]
                    cursor += width
            return output

    def _require_positive(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InputError(
                f"{name} must be a positive integer",
                details={"field": name, "value": value, "type": type(value).__name__},
            )

    def _normalize_batches(
        edits: Sequence[RelEdit] | Sequence[Sequence[RelEdit]],
    ) -> list[list[RelEdit]]:
        if not edits:
            return []
        first = edits[0]
        if isinstance(first, RelEdit):
            return [list(cast(Sequence[RelEdit], edits))]
        return [list(batch) for batch in cast(Sequence[Sequence[RelEdit]], edits)]

    def _sinusoidal_positions(pos: Tensor, *, d_pos: int, max_bp: int) -> Tensor:
        if torch.any(pos < 0) or torch.any(pos >= max_bp):
            raise InputError(
                "rel_pos must fall inside max_window_bp",
                details={"max_window_bp": max_bp},
            )
        half = d_pos // 2
        scale = torch.exp(
            torch.arange(half, device=pos.device, dtype=torch.float32)
            * (-math.log(float(max_bp)) / max(half - 1, 1))
        )
        angles = pos.to(dtype=torch.float32).unsqueeze(1) * scale.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)

    def _tokenize_short_dna(bases: str) -> list[int]:
        if len(bases) > V1_MAX_LEN:
            raise UnsupportedEditError(
                "edit length exceeds V1_MAX_LEN; structural variants use the v2 adapter",
                details={"length": len(bases), "v1_max_len": V1_MAX_LEN},
            )
        token_ids: list[int] = []
        for offset in range(0, _SEQ_TOKENS * _KMER, _KMER):
            chunk = bases[offset : offset + _KMER]
            if not chunk:
                token_ids.append(_OOV_TOKEN_ID)
                continue
            token_ids.append(_kmer_id(chunk.ljust(_KMER, "A")))
        return token_ids

    def _kmer_id(kmer: str) -> int:
        value = 0
        for base in kmer:
            value = (value << 2) | _BASE_TO_BITS[base]
        return value
