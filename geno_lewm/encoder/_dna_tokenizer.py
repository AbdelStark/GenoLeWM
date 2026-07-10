# SPDX-License-Identifier: Apache-2.0
"""Offline pure-DNA tokenizer for the pinned Carbon runtime package.

Carbon's upstream ``HybridDNATokenizer`` delegates non-DNA text to Qwen, but
the state encoder only ever supplies one ``<dna>...</dna>`` region. Loading the
remote tokenizer implementation for that restricted path is both unnecessary
and unsafe for reproducibility: the pinned implementation performs an
unpinned, network-capable Qwen tokenizer lookup. This module implements the
documented DNA branch directly from files in the mounted Carbon package.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from geno_lewm.encoder.windowing import (
    CARBON_DNA_CLOSE_TAG,
    CARBON_DNA_OPEN_TAG,
    CARBON_TOKEN_BP,
    canonicalize_dna,
)
from geno_lewm.errors import InputError, RuntimeSetupError

_DNA_BASE_ORDER = "ATCG"
_PINNED_DNA_SPECIAL_TOKENS = (CARBON_DNA_OPEN_TAG, CARBON_DNA_CLOSE_TAG, "<oov>")


class CarbonDNATokenizer:
    """Tokenize the pure-DNA subset consumed by :class:`CarbonStateEncoder`."""

    def __init__(
        self,
        *,
        k: int,
        dna_start_id: int,
        dna_vocab_size: int,
        pad_token_id: int,
    ) -> None:
        if k != CARBON_TOKEN_BP:
            raise RuntimeSetupError(
                "Carbon DNA tokenizer k does not match the encoder contract",
                details={"observed": k, "expected": CARBON_TOKEN_BP},
            )
        minimum_vocab = len(_PINNED_DNA_SPECIAL_TOKENS) + (len(_DNA_BASE_ORDER) ** k)
        if dna_vocab_size < minimum_vocab:
            raise RuntimeSetupError(
                "Carbon DNA vocabulary is too small for its declared k-mer contract",
                details={"observed": dna_vocab_size, "minimum": minimum_vocab, "k": k},
            )
        self.k = k
        self.dna_start_id = dna_start_id
        self.dna_vocab_size = dna_vocab_size
        self.dna_begin_token_id = dna_start_id
        self.dna_end_token_id = dna_start_id + 1
        self.oov_token_id = dna_start_id + 2
        self.pad_token_id = pad_token_id
        self._kmer_start_id = dna_start_id + len(_PINNED_DNA_SPECIAL_TOKENS)

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> CarbonDNATokenizer:
        """Load the tokenizer contract from one local Carbon package."""
        root = Path(model_dir)
        dna = _read_json_object(root / "dna_config.json", label="Carbon DNA config")
        tokenizer = _read_json_object(
            root / "tokenizer_config.json",
            label="Carbon tokenizer config",
        )
        special_tokens = dna.get("dna_special_tokens")
        if not isinstance(special_tokens, list) or tuple(special_tokens) != (
            _PINNED_DNA_SPECIAL_TOKENS
        ):
            raise RuntimeSetupError(
                "Carbon DNA special-token order does not match the pinned encoder contract",
                details={
                    "observed": special_tokens,
                    "expected": list(_PINNED_DNA_SPECIAL_TOKENS),
                },
            )
        if dna.get("auto_dna_tags") is not False:
            raise RuntimeSetupError(
                "Carbon DNA runtime must disable implicit DNA tags",
                details={"auto_dna_tags": dna.get("auto_dna_tags")},
            )
        return cls(
            k=_required_int(dna, "k", label="Carbon DNA config"),
            dna_start_id=_required_int(dna, "dna_start_id", label="Carbon DNA config"),
            dna_vocab_size=_required_int(dna, "dna_vocab_size", label="Carbon DNA config"),
            pad_token_id=_resolve_pad_token_id(tokenizer),
        )

    def __call__(
        self,
        texts: Sequence[str],
        *,
        return_tensors: str,
        padding: bool,
        add_special_tokens: bool = False,
    ) -> Mapping[str, object]:
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
            raise InputError("Carbon DNA tokenizer input must be a sequence of strings")
        if not texts:
            raise InputError("Carbon DNA tokenizer input must be non-empty")
        if return_tensors != "pt":
            raise InputError(
                "Carbon DNA tokenizer only supports PyTorch tensors",
                details={"return_tensors": return_tensors},
            )
        if not padding:
            raise InputError("Carbon DNA tokenizer batch encoding requires right padding")
        if add_special_tokens:
            raise InputError("Carbon DNA tokenizer does not permit implicit special tokens")

        encoded = [self._encode_one(text) for text in texts]
        width = max(len(row) for row in encoded)
        input_ids = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        attention_mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - Carbon model loading also requires torch.
            raise RuntimeSetupError(
                "Carbon DNA tokenization requires PyTorch",
                remediation="install geno-lewm[train]",
            ) from exc
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def _encode_one(self, text: object) -> list[int]:
        if not isinstance(text, str):
            raise InputError(
                "Carbon DNA tokenizer items must be strings",
                details={"type": type(text).__name__},
            )
        if not text.startswith(CARBON_DNA_OPEN_TAG) or not text.endswith(CARBON_DNA_CLOSE_TAG):
            raise InputError(
                "Carbon state encoding requires one explicit DNA region",
                remediation="wrap the canonical sequence in <dna>...</dna>",
            )
        content = text[len(CARBON_DNA_OPEN_TAG) : -len(CARBON_DNA_CLOSE_TAG)]
        if CARBON_DNA_OPEN_TAG in content or CARBON_DNA_CLOSE_TAG in content:
            raise InputError("Carbon state encoding requires exactly one DNA region")
        sequence = canonicalize_dna(content)
        if not sequence or len(sequence) % self.k != 0:
            raise InputError(
                "Carbon DNA region must be non-empty and padded to its k-mer width",
                details={"sequence_bp": len(sequence), "k": self.k},
            )
        ids = [self.dna_begin_token_id]
        ids.extend(
            self._kmer_id(sequence[offset : offset + self.k])
            for offset in range(0, len(sequence), self.k)
        )
        ids.append(self.dna_end_token_id)
        return ids

    def _kmer_id(self, kmer: str) -> int:
        if any(base not in _DNA_BASE_ORDER for base in kmer):
            return self.oov_token_id
        index = 0
        for base in kmer:
            index = (index * len(_DNA_BASE_ORDER)) + _DNA_BASE_ORDER.index(base)
        return self._kmer_start_id + index


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSetupError(
            f"{label} is missing or invalid",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeSetupError(f"{label} must be a JSON object", details={"path": str(path)})
    return dict(payload)


def _required_int(payload: Mapping[str, object], key: str, *, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeSetupError(
            f"{label} field must be a non-negative integer",
            details={"field": key, "value": value},
        )
    return value


def _resolve_pad_token_id(tokenizer_config: Mapping[str, object]) -> int:
    pad_token = tokenizer_config.get("pad_token")
    if isinstance(pad_token, Mapping):
        pad_token = pad_token.get("content")
    if not isinstance(pad_token, str) or not pad_token:
        raise RuntimeSetupError("Carbon tokenizer config must declare a pad_token")
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, Mapping):
        raise RuntimeSetupError("Carbon tokenizer config must declare added_tokens_decoder")
    matches: list[int] = []
    for raw_id, raw_token in decoder.items():
        if not isinstance(raw_token, Mapping) or raw_token.get("content") != pad_token:
            continue
        try:
            token_id = int(cast(str | int, raw_id))
        except (TypeError, ValueError) as exc:
            raise RuntimeSetupError("Carbon tokenizer decoder IDs must be integers") from exc
        matches.append(token_id)
    if len(matches) != 1:
        raise RuntimeSetupError(
            "Carbon tokenizer pad token must resolve to exactly one ID",
            details={"pad_token": pad_token, "matches": matches},
        )
    return matches[0]
