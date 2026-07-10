"""Unit tests for the lazy Carbon state encoder wrapper."""

from __future__ import annotations

import contextlib
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import geno_lewm.encoder.carbon as carbon_mod
from geno_lewm.encoder import CarbonStateEncoder
from geno_lewm.encoder._dna_tokenizer import CarbonDNATokenizer
from geno_lewm.encoder._identity import encoder_weights_hash
from geno_lewm.errors import InputError, RuntimeSetupError


def test_carbon_state_encoder_reports_missing_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeSetupError, match="Transformers"):
        CarbonStateEncoder("HuggingFaceBio/Carbon-500M", "main")


def test_carbon_state_encoder_encodes_with_injected_components() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=tokenizer,
        encoder_hash="sha256:" + ("a" * 64),
        pool_radius=0,
    )

    state = encoder.encode("acgtac", edit_locus=0)

    assert state == (1.0, 0.0)
    assert tokenizer.calls == [["<dna>ACGTAC</dna>"]]
    assert model.eval_called is True
    assert encoder.parameter_count == 0
    assert encoder.trainable_parameter_count == 0
    assert encoder.d_state == 2
    assert encoder.encoder_hash == bytes.fromhex("a" * 64)


def test_carbon_state_encoder_batch_uses_per_item_loci() -> None:
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    states = encoder.encode_batch(["ACGTAC", "CCCCCC"], [0, None])

    assert states == ((1.0, 0.0), (0.0, 1.0))


def test_carbon_state_encoder_can_return_raw_pooled_states() -> None:
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
        normalize=False,
    )

    states = encoder.encode_batch(["ACGTAC", "CCCCCC"], [0, None])

    assert states == ((3.0, 0.0), (0.0, 4.0))


def test_carbon_state_encoder_rejects_zero_norm_state() -> None:
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=ZeroStateModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    with pytest.raises(InputError, match="finite non-zero norm"):
        encoder.encode("ACGTAC", edit_locus=0)


def test_carbon_state_encoder_calls_model_in_inference_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_active = False

    @contextlib.contextmanager
    def fake_inference_context() -> Iterator[None]:
        nonlocal context_active
        context_active = True
        try:
            yield
        finally:
            context_active = False

    class AssertingModel(FakeModel):
        def __call__(
            self,
            *,
            input_ids: list[list[int]],
            attention_mask: list[list[int]],
            output_hidden_states: bool,
        ) -> object:
            assert context_active is True
            return super().__call__(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )

    monkeypatch.setattr(carbon_mod, "torch_inference_context", fake_inference_context)
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=AssertingModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    assert encoder.encode("ACGTAC", edit_locus=0) == (1.0, 0.0)


def test_carbon_state_encoder_validates_component_pairing() -> None:
    with pytest.raises(InputError, match="supplied together"):
        CarbonStateEncoder("HuggingFaceBio/Carbon-500M", "main", model=FakeModel())


def test_local_encoder_identity_rejects_wrong_weights(tmp_path: Path) -> None:
    from geno_lewm.encoder.carbon import _verify_local_encoder_weights

    model_dir = tmp_path / "carbon"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"observed")

    with pytest.raises(RuntimeSetupError, match="does not match the committed encoder hash"):
        _verify_local_encoder_weights(model_dir.as_posix(), expected_hash=bytes.fromhex("0" * 64))


def test_carbon_state_encoder_freezes_all_model_parameters() -> None:
    model = ParameterizedFakeModel()

    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    assert encoder.parameter_count == 7
    assert encoder.trainable_parameter_count == 0
    assert all(parameter.requires_grad is False for parameter in model.parameters())


class FakeTokenizer:
    k = 6
    dna_start_id = 151_669
    dna_vocab_size = 4_107
    dna_begin_token_id = 151_669
    dna_end_token_id = 151_670
    oov_token_id = 151_671
    pad_token_id = 151_643

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[list[int]]]:
        assert return_tensors == "pt"
        assert padding is True
        assert add_special_tokens is False
        self.calls.append(texts)
        return {
            "input_ids": [[151_669, 151_672 + idx, 151_670] for idx, _text in enumerate(texts)],
            "attention_mask": [[1, 1, 1] for _text in texts],
        }


class FakeModel:
    config = SimpleNamespace(hidden_size=2)

    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(
        self,
        *,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        output_hidden_states: bool,
    ) -> object:
        assert output_hidden_states is True
        assert all(mask == [1, 1, 1] for mask in attention_mask)
        rows_by_item = []
        for idx, _row in enumerate(input_ids):
            if idx == 0:
                rows_by_item.append(((1.0, 0.0), (3.0, 0.0), (5.0, 0.0)))
            else:
                rows_by_item.append(((0.0, 2.0), (0.0, 4.0), (0.0, 6.0)))
        return SimpleNamespace(hidden_states=(rows_by_item,))


class ZeroStateModel(FakeModel):
    def __call__(
        self,
        *,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        output_hidden_states: bool,
    ) -> object:
        assert output_hidden_states is True
        rows_by_item = [((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)) for _row in input_ids]
        return SimpleNamespace(hidden_states=(rows_by_item,))


class FakeParameter:
    def __init__(self, size: int) -> None:
        self.size = size
        self.requires_grad = True

    def numel(self) -> int:
        return self.size

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self


class ParameterizedFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self._parameters = (FakeParameter(3), FakeParameter(4))

    def parameters(self) -> tuple[FakeParameter, ...]:
        return self._parameters


class _DeviceFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.moved_to: str | None = None

    def to(self, device: str) -> _DeviceFakeModel:
        self.moved_to = device
        return self


def test_resolve_device_explicit_and_default() -> None:
    from geno_lewm.encoder.carbon import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:0") == "cuda:0"
    # None / "auto" resolve to cuda when a GPU is present, else cpu.
    assert _resolve_device(None) in {"cpu", "cuda"}
    assert _resolve_device("auto") in {"cpu", "cuda"}


def test_carbon_state_encoder_moves_model_to_cuda() -> None:
    model = _DeviceFakeModel()
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=FakeTokenizer(),
        pool_radius=0,
        device="cuda",
    )
    assert encoder.device == "cuda"
    assert model.moved_to == "cuda"
    # Encoding still works; tokenizer outputs without .to() pass through.
    assert encoder.encode("ACGTAC", edit_locus=0) == (1.0, 0.0)


def test_carbon_state_encoder_cpu_does_not_move_model() -> None:
    model = _DeviceFakeModel()
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=FakeTokenizer(),
        pool_radius=0,
        device="cpu",
    )
    assert encoder.device == "cpu"
    assert model.moved_to is None


def test_carbon_state_encoder_rejects_nonzero_radius_for_global_pooling() -> None:
    with pytest.raises(InputError, match="global_mean pooling requires pool_radius=0"):
        CarbonStateEncoder(
            "HuggingFaceBio/Carbon-500M",
            "main@deadbeef",
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            pool_type="global_mean",
            pool_radius=1,
        )


def test_carbon_state_encoder_rejects_hidden_state_token_count_mismatch() -> None:
    class ShortHiddenStateModel(FakeModel):
        def __call__(
            self,
            *,
            input_ids: list[list[int]],
            attention_mask: list[list[int]],
            output_hidden_states: bool,
        ) -> object:
            del input_ids, attention_mask
            assert output_hidden_states is True
            return SimpleNamespace(hidden_states=((((1.0, 0.0), (2.0, 0.0)),),))

    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=ShortHiddenStateModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    with pytest.raises(InputError, match="hidden-state length"):
        encoder.encode("ACGTAC", edit_locus=0)


def test_carbon_pooling_identity_accounts_for_dna_open_token() -> None:
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    assert encoder.pooling_identity("ACGTAC", 0) == ("centered_mean", 0, 1)


def test_carbon_pooling_identity_canonicalizes_global_centers() -> None:
    centered = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )
    global_encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        pool_type="global_mean",
        pool_radius=0,
    )

    assert centered.pooling_identity("ACGTAC", None) == ("global_mean", 0, None)
    assert global_encoder.pooling_identity("ACGTAC", 0) == ("global_mean", 0, None)


@pytest.mark.parametrize("edit_locus", [True, 1.5, "1"])
def test_dna_token_layout_rejects_noninteger_edit_locus(edit_locus: object) -> None:
    layout = carbon_mod._DNATokenLayout(3, 3, 1, 1, 6)

    with pytest.raises(InputError, match="edit_locus must be an integer"):
        layout.center_token(edit_locus, sequence_bp=6)  # type: ignore[arg-type]


@pytest.mark.parametrize("edit_locus", [-1, 6])
def test_dna_token_layout_rejects_out_of_window_edit_locus(edit_locus: int) -> None:
    layout = carbon_mod._DNATokenLayout(3, 3, 1, 1, 6)

    with pytest.raises(InputError, match="outside the encoded DNA window"):
        layout.center_token(edit_locus, sequence_bp=6)


def test_dna_token_layout_rejects_locus_outside_tokenized_content() -> None:
    layout = carbon_mod._DNATokenLayout(2, 2, 1, 0, 6)

    with pytest.raises(InputError, match="outside the tokenized DNA content"):
        layout.center_token(0, sequence_bp=6)


class AmbiguousTokenizer(FakeTokenizer):
    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[list[int]]]:
        payload = super().__call__(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )
        payload["input_ids"][0] = [151_669, 151_669, 151_670]
        return payload


def test_carbon_state_encoder_rejects_ambiguous_dna_control_layout() -> None:
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=AmbiguousTokenizer(),
        pool_radius=0,
    )

    with pytest.raises(InputError, match="exactly one DNA control-token pair"):
        encoder.encode("ACGTAC", edit_locus=0)


def test_packaged_carbon_dna_tokenizer_matches_pinned_token_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_pinned_tokenizer_contract(tmp_path)
    tokenizer = CarbonDNATokenizer.from_model_dir(tmp_path)
    real_import = importlib.import_module

    class FakeTensor:
        def __init__(self, values: list[list[int]]) -> None:
            self._values = values

        def tolist(self) -> list[list[int]]:
            return self._values

    fake_torch = SimpleNamespace(tensor=FakeTensor)

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "torch":
            return fake_torch
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    tokenized = tokenizer(
        [
            "<dna>AAAAAA</dna>",
            "<dna>ATCGAT</dna>",
            "<dna>NNNNNN</dna>",
            "<dna>ATCGATATCGAT</dna>",
        ],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    assert cast(Any, tokenized["input_ids"]).tolist() == [
        [151_669, 151_672, 151_670, 151_643],
        [151_669, 152_105, 151_670, 151_643],
        [151_669, 151_671, 151_670, 151_643],
        [151_669, 152_105, 152_105, 151_670],
    ]
    assert cast(Any, tokenized["attention_mask"]).tolist() == [
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("dna_config.json", "{", "missing or invalid"),
        ("dna_config.json", "[]", "must be a JSON object"),
        (
            "dna_config.json",
            json.dumps(
                {
                    "k": 6,
                    "dna_start_id": 151_669,
                    "dna_vocab_size": 4_107,
                    "dna_special_tokens": ["</dna>", "<dna>", "<oov>"],
                    "auto_dna_tags": False,
                }
            ),
            "special-token order",
        ),
        (
            "dna_config.json",
            json.dumps(
                {
                    "k": 6,
                    "dna_start_id": 151_669,
                    "dna_vocab_size": 4_107,
                    "dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
                    "auto_dna_tags": True,
                }
            ),
            "disable implicit DNA tags",
        ),
        (
            "dna_config.json",
            json.dumps(
                {
                    "k": 5,
                    "dna_start_id": 151_669,
                    "dna_vocab_size": 4_107,
                    "dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
                    "auto_dna_tags": False,
                }
            ),
            "k does not match",
        ),
        (
            "dna_config.json",
            json.dumps(
                {
                    "k": 6,
                    "dna_start_id": True,
                    "dna_vocab_size": 4_107,
                    "dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
                    "auto_dna_tags": False,
                }
            ),
            "non-negative integer",
        ),
        (
            "dna_config.json",
            json.dumps(
                {
                    "k": 6,
                    "dna_start_id": 151_669,
                    "dna_vocab_size": 4_098,
                    "dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
                    "auto_dna_tags": False,
                }
            ),
            "vocabulary is too small",
        ),
        ("tokenizer_config.json", "{}", "declare a pad_token"),
        (
            "tokenizer_config.json",
            json.dumps({"pad_token": "<pad>"}),
            "declare added_tokens_decoder",
        ),
        (
            "tokenizer_config.json",
            json.dumps(
                {
                    "pad_token": "<pad>",
                    "added_tokens_decoder": {"not-an-id": {"content": "<pad>"}},
                }
            ),
            "decoder IDs must be integers",
        ),
        (
            "tokenizer_config.json",
            json.dumps(
                {
                    "pad_token": {"content": "<pad>"},
                    "added_tokens_decoder": {
                        "1": {"content": "<pad>"},
                        "2": {"content": "<pad>"},
                    },
                }
            ),
            "exactly one ID",
        ),
    ],
)
def test_packaged_carbon_dna_tokenizer_rejects_invalid_runtime_contract(
    tmp_path: Path,
    filename: str,
    payload: str,
    message: str,
) -> None:
    _write_pinned_tokenizer_contract(tmp_path)
    (tmp_path / filename).write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeSetupError, match=message):
        CarbonDNATokenizer.from_model_dir(tmp_path)


@pytest.mark.parametrize(
    ("texts", "kwargs", "message"),
    [
        ("<dna>AAAAAA</dna>", {}, "sequence of strings"),
        ([], {}, "non-empty"),
        (["<dna>AAAAAA</dna>"], {"return_tensors": "np"}, "PyTorch tensors"),
        (["<dna>AAAAAA</dna>"], {"padding": False}, "right padding"),
        (["<dna>AAAAAA</dna>"], {"add_special_tokens": True}, "implicit special"),
        ([1], {}, "items must be strings"),
        (["AAAAAA"], {}, "explicit DNA region"),
        (["<dna><dna>AAAAAA</dna></dna>"], {}, "exactly one DNA region"),
        (["<dna></dna>"], {}, "non-empty and padded"),
        (["<dna>AAAAA</dna>"], {}, "non-empty and padded"),
    ],
)
def test_packaged_carbon_dna_tokenizer_rejects_invalid_batch_input(
    tmp_path: Path,
    texts: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _write_pinned_tokenizer_contract(tmp_path)
    tokenizer = CarbonDNATokenizer.from_model_dir(tmp_path)
    call_kwargs: dict[str, object] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }
    call_kwargs.update(kwargs)

    with pytest.raises(InputError, match=message):
        tokenizer(texts, **call_kwargs)  # type: ignore[arg-type]


def test_transformers_loader_uses_one_local_runtime_without_auto_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_pinned_tokenizer_contract(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            calls.append((model_id, kwargs))
            return object()

    fake_transformers = SimpleNamespace(AutoModel=FakeAutoModel)
    fake_torch = SimpleNamespace(bfloat16="bf16")
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "transformers":
            return fake_transformers
        if name == "torch":
            return fake_torch
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    tokenizer, _model = carbon_mod._load_transformers_components(
        model_id=str(tmp_path),
        revision="pinned-revision",
        dtype="bf16",
        local_files_only=True,
        trust_remote_code=False,
    )

    assert isinstance(tokenizer, CarbonDNATokenizer)
    assert calls == [
        (
            str(tmp_path.resolve()),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": "bf16",
            },
        )
    ]


def test_transformers_loader_requires_auto_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "transformers":
            return SimpleNamespace()
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeSetupError, match="expose AutoModel"):
        carbon_mod._load_transformers_components(
            model_id=str(tmp_path),
            revision="pinned-revision",
            dtype="bf16",
            local_files_only=True,
            trust_remote_code=False,
        )


def test_resolve_runtime_directory_fails_closed_on_hub_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_snapshot_download(name: str) -> object:
        assert name == "huggingface_hub"
        return SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", missing_snapshot_download)
    with pytest.raises(RuntimeSetupError, match="expose snapshot_download"):
        carbon_mod._resolve_runtime_directory(
            model_id="HuggingFaceBio/Carbon-500M",
            revision="pinned",
            local_files_only=True,
        )

    def failing_snapshot_download(**kwargs: object) -> str:
        assert kwargs["revision"] == "pinned"
        raise OSError("cache miss")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(snapshot_download=failing_snapshot_download),
    )
    with pytest.raises(RuntimeSetupError, match="snapshot could not be resolved"):
        carbon_mod._resolve_runtime_directory(
            model_id="HuggingFaceBio/Carbon-500M",
            revision="pinned",
            local_files_only=True,
        )

    snapshot_file = tmp_path / "not-a-directory"
    snapshot_file.write_text("not a runtime", encoding="utf-8")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot_file)),
    )
    with pytest.raises(RuntimeSetupError, match="not a directory"):
        carbon_mod._resolve_runtime_directory(
            model_id="HuggingFaceBio/Carbon-500M",
            revision="pinned",
            local_files_only=True,
        )


def test_freeze_module_parameters_rejects_invalid_parameter_interfaces() -> None:
    class EnumerationFailure:
        def parameters(self) -> object:
            raise OSError("cannot enumerate")

    with pytest.raises(RuntimeSetupError, match="failed to enumerate"):
        carbon_mod._freeze_module_parameters(EnumerationFailure())

    class InvalidCount:
        requires_grad = False

        def numel(self) -> bool:
            return True

    with pytest.raises(RuntimeSetupError, match=r"numel\(\)"):
        carbon_mod._freeze_module_parameters(SimpleNamespace(parameters=lambda: (InvalidCount(),)))

    class StubbornParameter:
        @property
        def requires_grad(self) -> bool:
            return True

        @requires_grad.setter
        def requires_grad(self, value: bool) -> None:
            del value

        def numel(self) -> int:
            return 2

    with pytest.raises(RuntimeSetupError, match="must be frozen"):
        carbon_mod._freeze_module_parameters(
            SimpleNamespace(parameters=lambda: (StubbornParameter(),))
        )


def test_device_helpers_move_torch_like_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert carbon_mod._resolve_device("auto") == "cuda"

    class TensorLike:
        def __init__(self) -> None:
            self.device: str | None = None

        def to(self, device: str) -> TensorLike:
            self.device = device
            return self

    tensor = TensorLike()
    tokenized = carbon_mod._move_inputs_to_device({"input_ids": tensor, "meta": "raw"}, "cuda")

    assert tokenized == {"input_ids": tensor, "meta": "raw"}
    assert tensor.device == "cuda"


@pytest.mark.parametrize(
    "value",
    [b"short", "not-hex", "sha256:aa", 42],
)
def test_encoder_hash_rejects_malformed_values(value: object) -> None:
    with pytest.raises(InputError, match="encoder_hash"):
        carbon_mod._coerce_encoder_hash(value)  # type: ignore[arg-type]


def test_local_encoder_identity_accepts_exact_weight_hash_and_requires_directory(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "carbon"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"exact weights")
    expected = bytes.fromhex(encoder_weights_hash(model_dir).removeprefix("sha256:"))

    carbon_mod._verify_local_encoder_weights(str(model_dir), expected_hash=expected)
    with pytest.raises(RuntimeSetupError, match="requires a local model directory"):
        carbon_mod._verify_local_encoder_weights(
            str(tmp_path / "missing"),
            expected_hash=expected,
        )


def _write_pinned_tokenizer_contract(path: Path) -> None:
    (path / "dna_config.json").write_text(
        json.dumps(
            {
                "k": 6,
                "dna_start_id": 151_669,
                "dna_vocab_size": 4_107,
                "dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
                "auto_dna_tags": False,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<|endoftext|>",
                "added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                },
            }
        ),
        encoding="utf-8",
    )
