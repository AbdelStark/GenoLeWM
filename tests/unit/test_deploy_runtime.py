"""Unit tests for the RFC-0010 deploy runtime contract."""

from __future__ import annotations

import contextlib
import importlib
import json
import platform
import socket
import urllib.request
from collections.abc import Iterator
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import NoReturn

import pytest

import geno_lewm.deploy.runtime as runtime_mod
from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY
from geno_lewm.action import EditSpec
from geno_lewm.deploy import (
    BACKEND_COREML,
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_ONNX,
    BACKEND_PRIORITY,
    BackendProbe,
    GenoLeWMRuntime,
    fail_closed_network_guard,
    probe_backends,
    select_backend,
)
from geno_lewm.errors import (
    BackendUnsupportedError,
    InputError,
    ManifestHashMismatchError,
    ModelNotFoundError,
    NetworkCallProhibitedError,
    RuntimeSetupError,
)
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    DtypeConfig,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    PoolingConfig,
    compute_input_commitment,
    read_receipt,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from geno_lewm.surprise import (
    CalibrationBucket,
    CalibrationTable,
    SurpriseResult,
    write_calibration_table,
)


class FakeEncoder:
    def encode(self, window: str, *, edit_locus: int | None = None) -> tuple[float, ...]:
        del edit_locus
        denom = float(len(window))
        return tuple(window.count(base) / denom for base in "ACGT")


class FakeActionEncoder:
    def __call__(self, edits: object) -> tuple[float, ...]:
        del edits
        return (1.0,)


class EchoPredictor:
    def __call__(self, state: object, action: object) -> object:
        del action
        return state


class LoadableActionEncoder(FakeActionEncoder):
    def __init__(self) -> None:
        self.loaded = False
        self.eval_called = False

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        assert strict is True
        assert state == {"ok": True}
        self.loaded = True

    def eval(self) -> None:
        self.eval_called = True


class LoadableEchoPredictor(EchoPredictor):
    def __init__(self) -> None:
        self.loaded = False
        self.eval_called = False

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        assert strict is True
        assert state == {"ok": True}
        self.loaded = True

    def eval(self) -> None:
        self.eval_called = True


def test_runtime_auto_falls_back_to_cpu_when_optional_backends_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_mod, "_torch_cuda_available", lambda: (False, "no cuda"))

    def fake_find_spec(name: str) -> ModuleSpec | None:
        if name in {"coremltools", "onnxruntime"}:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    runtime = GenoLeWMRuntime(tmp_path)

    assert runtime.backend == BACKEND_CPU
    assert tuple(probe.backend for probe in runtime.probes) == BACKEND_PRIORITY


def test_runtime_rejects_missing_model_dir(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        GenoLeWMRuntime(tmp_path / "missing")


def test_select_backend_uses_priority_order_and_rejects_unavailable_backend() -> None:
    probes = (
        BackendProbe(BACKEND_COREML, False, "no coreml"),
        BackendProbe(BACKEND_CUDA, True, "cuda ok"),
        BackendProbe(BACKEND_ONNX, True, "onnx ok"),
        BackendProbe(BACKEND_CPU, True, "cpu ok"),
    )

    assert select_backend("auto", probes=probes) == BACKEND_CUDA
    assert select_backend("onnx", probes=probes) == BACKEND_ONNX

    with pytest.raises(BackendUnsupportedError):
        select_backend("coreml", probes=probes)
    with pytest.raises(BackendUnsupportedError):
        select_backend("bad-backend", probes=probes)


def test_backend_probe_and_selection_validate_contract_inputs() -> None:
    with pytest.raises(BackendUnsupportedError):
        BackendProbe("webgpu", False, "not supported")
    with pytest.raises(InputError):
        BackendProbe(BACKEND_CPU, "yes", "truthy")  # type: ignore[arg-type]
    with pytest.raises(InputError):
        BackendProbe(BACKEND_CPU, True, "")

    unavailable = (
        BackendProbe(BACKEND_COREML, False, "no coreml"),
        BackendProbe(BACKEND_CUDA, False, "no cuda"),
        BackendProbe(BACKEND_ONNX, False, "no onnx"),
        BackendProbe(BACKEND_CPU, False, "no cpu"),
    )
    with pytest.raises(BackendUnsupportedError):
        select_backend("auto", probes=unavailable)
    with pytest.raises(BackendUnsupportedError):
        select_backend("cuda", probes=(BackendProbe(BACKEND_CPU, True, "cpu ok"),))
    with pytest.raises(BackendUnsupportedError):
        select_backend("", probes=unavailable)


def test_coreml_probe_documents_platform_and_artifact_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    linux_probe = probe_backends(tmp_path)[0]
    assert linux_probe.backend == BACKEND_COREML
    assert linux_probe.available is False
    assert "Apple Silicon" in linux_probe.reason

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name == "coremltools" else None,
    )
    missing_artifact = probe_backends(tmp_path)[0]
    assert missing_artifact.available is False
    assert "missing backend artifact" in missing_artifact.reason

    (tmp_path / "model.mlpackage").write_text("placeholder", encoding="utf-8")
    available = probe_backends(tmp_path)[0]
    assert available.available is True


def test_cuda_and_onnx_probes_are_testable_without_optional_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_mod, "_torch_cuda_available", lambda: (True, "cuda ok"))
    cuda_missing_artifact = probe_backends(tmp_path)[1]
    assert cuda_missing_artifact.backend == BACKEND_CUDA
    assert cuda_missing_artifact.available is False
    assert "predictor.safetensors" in cuda_missing_artifact.reason

    (tmp_path / "predictor.safetensors").write_text("placeholder", encoding="utf-8")
    cuda_available = probe_backends(tmp_path)[1]
    assert cuda_available.available is True

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name == "onnxruntime" else None,
    )
    onnx_missing_artifact = probe_backends(tmp_path)[2]
    assert onnx_missing_artifact.backend == BACKEND_ONNX
    assert onnx_missing_artifact.available is False

    (tmp_path / "predictor.onnx").write_text("placeholder", encoding="utf-8")
    onnx_available = probe_backends(tmp_path)[2]
    assert onnx_available.available is True


def test_torch_cuda_probe_handles_mocked_torch_states(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        def __init__(self, available: bool) -> None:
            self._available = available

        def is_available(self) -> bool:
            return self._available

    class FakeTorchWithoutCuda:
        cuda = None

    class FakeTorchWithoutGpu:
        cuda = FakeCuda(False)

    class FakeTorchWithGpu:
        cuda = FakeCuda(True)

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name == "torch" else None,
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorchWithoutCuda)
    available, reason = runtime_mod._torch_cuda_available()
    assert available is False
    assert reason == "torch CUDA is not available"

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorchWithoutGpu)
    available, reason = runtime_mod._torch_cuda_available()
    assert available is False
    assert reason == "torch CUDA is not available"

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorchWithGpu)
    available, reason = runtime_mod._torch_cuda_available()
    assert available is True
    assert reason == "torch CUDA is available"


def test_fail_closed_network_guard_blocks_common_entrypoints() -> None:
    with fail_closed_network_guard():
        with pytest.raises(NetworkCallProhibitedError):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        with pytest.raises(NetworkCallProhibitedError):
            urllib.request.urlopen("https://example.com", timeout=0.01)

        sock = socket.socket()
        try:
            with pytest.raises(NetworkCallProhibitedError):
                sock.connect(("127.0.0.1", 9))
        finally:
            sock.close()


def test_runtime_methods_enter_fail_closed_network_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = GenoLeWMRuntime(tmp_path)
    variant = EditSpec(chrom="1", pos=1, ref="A", alt="C")

    def attempt_network(_operation: str, _backend: str, _model_dir: Path) -> NoReturn:
        urllib.request.urlopen("https://example.com", timeout=0.01)
        raise AssertionError("network guard did not block urlopen")

    monkeypatch.setattr(runtime_mod, "_raise_backend_not_ready", attempt_network)

    with pytest.raises(NetworkCallProhibitedError):
        runtime.score_variant(variant)


def test_runtime_methods_validate_inputs_and_fail_until_backends_land(tmp_path: Path) -> None:
    runtime = GenoLeWMRuntime(tmp_path)
    variant = EditSpec(chrom="1", pos=1, ref="A", alt="C")

    with pytest.raises(InputError):
        runtime.score_variant(object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeSetupError):
        runtime.score_variant(variant, window="ACGT")
    with pytest.raises(RuntimeSetupError):
        runtime.score_variant(variant)
    with pytest.raises(InputError):
        runtime.score_vcf("input.vcf", "ref.fa", "scores.parquet", batch_size=0)
    with pytest.raises(InputError):
        runtime.score_vcf("input.vcf", "ref.fa", "scores.parquet", progress="yes")  # type: ignore[arg-type]
    with pytest.raises(RuntimeSetupError):
        runtime.score_vcf("input.vcf", "ref.fa", "scores.parquet")
    with pytest.raises(InputError):
        runtime.encode_window("ACGT", edit_locus=-1)
    with pytest.raises(RuntimeSetupError):
        runtime.encode_window("ACGT")
    with pytest.raises(InputError):
        runtime.predict(state=None, edits=())
    with pytest.raises(InputError):
        runtime.predict(state=object(), edits=object())  # type: ignore[arg-type]
    with pytest.raises(InputError):
        runtime.predict(state=object(), edits=["bad"])  # type: ignore[list-item]
    with pytest.raises(RuntimeSetupError):
        runtime.predict(state=object(), edits=())

    with pytest.raises(BackendUnsupportedError):
        GenoLeWMRuntime(tmp_path, backend="cuda")


def test_runtime_scores_with_manifest_verified_local_components(tmp_path: Path) -> None:
    _write_runtime_model_dir(tmp_path)
    runtime = GenoLeWMRuntime(
        tmp_path,
        encoder=FakeEncoder(),
        action_encoder=FakeActionEncoder(),
        predictor=EchoPredictor(),
    )

    result = runtime.score_variant(EditSpec(chrom="1", pos=1, ref="A", alt="T"), window="ACGT")

    assert result.bucket_id == "*"
    assert result.sigma_raw > 0.0


def test_runtime_score_variant_enters_inference_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_model_dir(tmp_path)
    context_active = False

    @contextlib.contextmanager
    def fake_inference_context() -> Iterator[None]:
        nonlocal context_active
        context_active = True
        try:
            yield
        finally:
            context_active = False

    def fake_score_variant(*_args: object, **_kwargs: object) -> SurpriseResult:
        assert context_active is True
        return SurpriseResult(
            sigma_raw=0.25,
            sigma_calibrated=0.5,
            bucket_id="*",
            confidence=1.0,
            low_confidence=False,
        )

    monkeypatch.setattr(runtime_mod, "torch_inference_context", fake_inference_context)
    monkeypatch.setattr(runtime_mod, "score_surprise_variant", fake_score_variant)
    runtime = GenoLeWMRuntime(
        tmp_path,
        encoder=FakeEncoder(),
        action_encoder=FakeActionEncoder(),
        predictor=EchoPredictor(),
    )

    result = runtime.score_variant(EditSpec(chrom="1", pos=1, ref="A", alt="T"), window="ACGT")

    assert result.sigma_raw == 0.25


def test_runtime_auto_loads_manifest_components_when_optional_runtime_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_model_dir(tmp_path)
    action_encoder = LoadableActionEncoder()
    predictor = LoadableEchoPredictor()
    loaded_paths: list[tuple[str, Path]] = []

    monkeypatch.setattr(runtime_mod, "_native_scorer_runtime_available", lambda: True)
    monkeypatch.setattr(
        runtime_mod,
        "_build_runtime_encoder",
        lambda _manifest, _cfg: FakeEncoder(),
    )
    monkeypatch.setattr(
        runtime_mod,
        "_build_runtime_action_encoder",
        lambda _cfg: action_encoder,
    )
    monkeypatch.setattr(
        runtime_mod,
        "_build_runtime_predictor",
        lambda _cfg: predictor,
    )

    def fake_load_module_state(module: object, path: Path, *, artifact: str) -> None:
        loaded_paths.append((artifact, path))
        module.load_state_dict({"ok": True}, strict=True)  # type: ignore[attr-defined]
        module.eval()  # type: ignore[attr-defined]

    monkeypatch.setattr(runtime_mod, "_load_module_state", fake_load_module_state)

    runtime = GenoLeWMRuntime(tmp_path)
    result = runtime.score_variant(EditSpec(chrom="1", pos=1, ref="A", alt="T"), window="ACGT")

    assert result.bucket_id == "*"
    assert action_encoder.loaded is True
    assert action_encoder.eval_called is True
    assert predictor.loaded is True
    assert predictor.eval_called is True
    assert loaded_paths == [
        ("action_encoder", tmp_path / "action_encoder.safetensors"),
        ("predictor", tmp_path / "predictor.safetensors"),
    ]


def test_load_action_predictor_modules_does_not_build_carbon_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_model_dir(tmp_path)
    action_encoder = LoadableActionEncoder()
    predictor = LoadableEchoPredictor()
    loaded_paths: list[tuple[str, Path]] = []

    monkeypatch.setattr(runtime_mod, "_native_action_predictor_runtime_available", lambda: True)
    monkeypatch.setattr(
        runtime_mod,
        "_build_runtime_encoder",
        lambda _manifest, _cfg: pytest.fail("Carbon encoder should not be built"),
    )
    monkeypatch.setattr(
        runtime_mod,
        "_build_runtime_action_encoder",
        lambda _cfg: action_encoder,
    )
    monkeypatch.setattr(
        runtime_mod,
        "_build_runtime_predictor",
        lambda _cfg: predictor,
    )

    def fake_load_module_state(module: object, path: Path, *, artifact: str) -> None:
        loaded_paths.append((artifact, path))
        module.load_state_dict({"ok": True}, strict=True)  # type: ignore[attr-defined]
        module.eval()  # type: ignore[attr-defined]

    monkeypatch.setattr(runtime_mod, "_load_module_state", fake_load_module_state)

    loaded_action_encoder, loaded_predictor = runtime_mod.load_action_predictor_modules(tmp_path)

    assert loaded_action_encoder is action_encoder
    assert loaded_predictor is predictor
    assert action_encoder.loaded is True
    assert action_encoder.eval_called is True
    assert predictor.loaded is True
    assert predictor.eval_called is True
    assert loaded_paths == [
        ("action_encoder", tmp_path / "action_encoder.safetensors"),
        ("predictor", tmp_path / "predictor.safetensors"),
    ]


def test_runtime_keeps_manifest_scoring_unloaded_without_optional_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_model_dir(tmp_path)
    monkeypatch.setattr(runtime_mod, "_native_scorer_runtime_available", lambda: False)

    runtime = GenoLeWMRuntime(tmp_path)

    with pytest.raises(RuntimeSetupError):
        runtime.score_variant(EditSpec(chrom="1", pos=1, ref="A", alt="T"), window="ACGT")


def test_runtime_writes_single_variant_receipt_with_manifest_commitment(tmp_path: Path) -> None:
    manifest = _write_runtime_model_dir(tmp_path)
    runtime = GenoLeWMRuntime(
        tmp_path,
        encoder=FakeEncoder(),
        action_encoder=FakeActionEncoder(),
        predictor=EchoPredictor(),
    )
    variant = EditSpec(chrom="1", pos=1, ref="A", alt="T")
    receipt_path = tmp_path / "score.receipt.json"

    result = runtime.score_variant(variant, window="acgt", receipt_path=receipt_path)

    receipt = read_receipt(receipt_path)
    assert receipt.model_id == manifest.model_id()
    assert receipt.calibration_hash == manifest.calibration.hash
    assert receipt.output.sigma_raw == result.sigma_raw
    assert receipt.output.sigma_calibrated == result.sigma_calibrated
    assert receipt.runtime.backend == runtime.backend
    assert receipt.runtime.carbon_revision == manifest.encoder.revision
    assert receipt.provenance.kind == "checksum_only"
    assert receipt.provenance.details is not None
    assert receipt.provenance.details["scope"] == "single_variant"
    assert receipt.input_commitment == compute_input_commitment(
        "ACGT",
        variant,
        PoolingConfig(state_layer=20, pool_type="centered_mean", pool_radius=8, normalize=True),
        DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16"),
    )


def test_runtime_requires_manifest_for_receipt_writing(tmp_path: Path) -> None:
    runtime = GenoLeWMRuntime(
        tmp_path,
        encoder=FakeEncoder(),
        action_encoder=FakeActionEncoder(),
        predictor=EchoPredictor(),
        calibration=_calibration(),
    )

    with pytest.raises(RuntimeSetupError, match=r"manifest\.json"):
        runtime.score_variant(
            EditSpec(chrom="1", pos=1, ref="A", alt="T"),
            window="ACGT",
            receipt_path=tmp_path / "score.receipt.json",
        )


def test_runtime_requires_manifest_for_vcf_receipt_writing(tmp_path: Path) -> None:
    runtime = GenoLeWMRuntime(
        tmp_path,
        encoder=FakeEncoder(),
        action_encoder=FakeActionEncoder(),
        predictor=EchoPredictor(),
        calibration=_calibration(),
    )

    with pytest.raises(RuntimeSetupError, match=r"manifest\.json"):
        runtime.score_vcf(
            tmp_path / "input.vcf",
            tmp_path / "ref.fa",
            tmp_path / "scores.jsonl",
            receipt_path=tmp_path / "scores.receipts.jsonl",
        )


def test_runtime_scores_vcf_with_manifest_verified_components_and_fasta(tmp_path: Path) -> None:
    manifest = _write_runtime_model_dir(tmp_path)
    runtime = GenoLeWMRuntime(
        tmp_path,
        encoder=FakeEncoder(),
        action_encoder=FakeActionEncoder(),
        predictor=EchoPredictor(),
    )
    sequence = "ACGT" * 3072
    fasta_path = tmp_path / "ref.fa"
    fasta_path.write_text(f">1\n{sequence}\n", encoding="utf-8")
    vcf_path = tmp_path / "input.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "scores.jsonl"
    receipt_path = tmp_path / "scores.receipts.jsonl"

    runtime.score_vcf(vcf_path, fasta_path, output_path, receipt_path=receipt_path)

    assert output_path.is_file()
    score_row = json.loads(output_path.read_text(encoding="utf-8"))
    assert score_row["generated_by"] == SCORE_JSONL_GENERATED_BY
    assert score_row["bucket_id"] == "*"
    receipt_lines = receipt_path.read_text(encoding="utf-8").splitlines()
    assert len(receipt_lines) == 1
    assert '"chrom"' not in receipt_lines[0]
    assert '"alt"' not in receipt_lines[0]

    one_receipt = tmp_path / "one.receipt.json"
    one_receipt.write_text(receipt_lines[0], encoding="utf-8")
    receipt = read_receipt(one_receipt)
    variant = EditSpec(chrom="1", pos=1, ref="A", alt="T")
    assert receipt.model_id == manifest.model_id()
    assert receipt.calibration_hash == manifest.calibration.hash
    assert receipt.provenance.details is not None
    assert receipt.provenance.details["scope"] == "vcf_row"
    assert receipt.provenance.details["receipt_stream"] == "jsonl_per_scored_alternate_v1"
    assert receipt.provenance.details["row_index"] == 1
    assert receipt.input_commitment == compute_input_commitment(
        sequence,
        variant,
        PoolingConfig(state_layer=20, pool_type="centered_mean", pool_radius=8, normalize=True),
        DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16"),
    )


def test_runtime_rejects_manifest_artifact_hash_mismatch(tmp_path: Path) -> None:
    _write_runtime_model_dir(tmp_path)
    (tmp_path / "predictor.safetensors").write_bytes(b"tampered")

    with pytest.raises(ManifestHashMismatchError):
        GenoLeWMRuntime(tmp_path)


def test_runtime_rejects_partial_component_injection(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="supplied together"):
        GenoLeWMRuntime(tmp_path, encoder=FakeEncoder())


def _write_runtime_model_dir(root: Path) -> Manifest:
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    calibration_path = write_calibration_table(_calibration(), root / "calibration.parquet")
    (root / "train_config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (root / "eval_report.md").write_text("# eval\n", encoding="utf-8")

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0-test",
        release_id="geno-lewm-test",
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="local-test",
            hash=sha256_bytes(b"encoder"),
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=sha256_file(root / "predictor.safetensors"),
            dtype="bf16",
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=sha256_file(root / "action_encoder.safetensors"),
            dtype="bf16",
        ),
        calibration=ManifestArtifact(
            file=calibration_path.name,
            hash=sha256_file(calibration_path),
            version="1.0.0",
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=sha256_file(root / "train_config.yaml"),
            data_snapshot={"fixture": "runtime"},
        ),
        eval=ManifestArtifact(file="eval_report.md", hash=sha256_file(root / "eval_report.md")),
    )
    write_manifest(manifest, root / "manifest.json")
    return manifest


def _calibration() -> CalibrationTable:
    return CalibrationTable(
        buckets=(
            CalibrationBucket(
                bucket_id="coding_missense|mid|none",
                n_calibration=1_000,
                cdf=(0.0, 0.5, 1.0),
                sigma_grid=(0.0, 0.5, 1.0),
            ),
            CalibrationBucket(
                bucket_id="*",
                n_calibration=1_000,
                cdf=(0.0, 0.5, 1.0),
                sigma_grid=(0.0, 0.5, 1.0),
            ),
        )
    )
