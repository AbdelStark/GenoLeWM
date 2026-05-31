"""Unit tests for the RFC-0010 deploy runtime contract."""

from __future__ import annotations

import importlib
import platform
import socket
import urllib.request
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import NoReturn

import pytest

import geno_lewm.deploy.runtime as runtime_mod
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
    ModelNotFoundError,
    NetworkCallProhibitedError,
    RuntimeSetupError,
)


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
