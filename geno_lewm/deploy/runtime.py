# SPDX-License-Identifier: Apache-2.0
"""On-device runtime contract and fail-closed network guard."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import platform
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

from geno_lewm.action import EditSpec, RelEdit
from geno_lewm.encoder.windowing import canonicalize_dna
from geno_lewm.errors import (
    BackendUnsupportedError,
    InputError,
    ModelNotFoundError,
    NetworkCallProhibitedError,
    RuntimeSetupError,
)

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_COREML",
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_ONNX",
    "BACKEND_PRIORITY",
    "BackendProbe",
    "GenoLeWMRuntime",
    "fail_closed_network_guard",
    "probe_backends",
    "select_backend",
]


BACKEND_AUTO = "auto"
BACKEND_COREML = "coreml"
BACKEND_CUDA = "cuda"
BACKEND_ONNX = "onnx"
BACKEND_CPU = "cpu"
BACKEND_PRIORITY: tuple[str, ...] = (
    BACKEND_COREML,
    BACKEND_CUDA,
    BACKEND_ONNX,
    BACKEND_CPU,
)
_SUPPORTED_BACKENDS = (BACKEND_AUTO, *BACKEND_PRIORITY)
_APPLE_SILICON_MACHINES = frozenset({"arm64", "aarch64"})


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """Capability probe result for one runtime backend."""

    backend: str
    available: bool
    reason: str

    def __post_init__(self) -> None:
        if self.backend not in BACKEND_PRIORITY:
            raise BackendUnsupportedError(
                "unsupported runtime backend",
                details={"backend": self.backend, "supported": list(BACKEND_PRIORITY)},
            )
        if not isinstance(self.available, bool):
            raise InputError(
                "backend availability must be bool",
                details={"backend": self.backend, "type": type(self.available).__name__},
            )
        if not self.reason:
            raise InputError("backend probe reason must be non-empty")


class GenoLeWMRuntime:
    """Top-level runtime facade for on-device inference workflows."""

    def __init__(self, model_dir: str | Path, backend: str = BACKEND_AUTO) -> None:
        root = Path(model_dir).expanduser()
        if not root.exists() or not root.is_dir():
            raise ModelNotFoundError(
                "model_dir must be an existing directory",
                details={"model_dir": str(root)},
            )
        self.model_dir = root
        self.probes = probe_backends(root)
        self.backend = select_backend(backend, probes=self.probes)

    def score_variant(self, variant: EditSpec, window: str | None = None) -> Any:
        """Score a single variant once the scorer backend is installed."""
        if not isinstance(variant, EditSpec):
            raise InputError(
                "variant must be an EditSpec",
                details={"type": type(variant).__name__},
            )
        if window is not None:
            canonicalize_dna(window)
        with fail_closed_network_guard():
            _raise_backend_not_ready("score_variant", self.backend, self.model_dir)

    def score_vcf(
        self,
        vcf_path: str | Path,
        fasta_path: str | Path,
        output_path: str | Path,
        batch_size: int = 64,
        progress: bool = True,
    ) -> None:
        """Score a VCF once the scorer backend is installed."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise InputError(
                "batch_size must be a positive integer",
                details={"batch_size": batch_size, "type": type(batch_size).__name__},
            )
        if not isinstance(progress, bool):
            raise InputError(
                "progress must be bool",
                details={"type": type(progress).__name__},
            )
        # Normalize path-like values now so type errors surface at the API boundary.
        Path(vcf_path)
        Path(fasta_path)
        Path(output_path)
        with fail_closed_network_guard():
            _raise_backend_not_ready("score_vcf", self.backend, self.model_dir)

    def encode_window(self, window: str, edit_locus: int | None = None) -> Any:
        """Encode a DNA window once the encoder backend is installed."""
        canonicalize_dna(window)
        if edit_locus is not None and (
            not isinstance(edit_locus, int) or isinstance(edit_locus, bool) or edit_locus < 0
        ):
            raise InputError(
                "edit_locus must be a non-negative integer or None",
                details={"edit_locus": edit_locus, "type": type(edit_locus).__name__},
            )
        with fail_closed_network_guard():
            _raise_backend_not_ready("encode_window", self.backend, self.model_dir)

    def predict(self, state: Any, edits: Sequence[RelEdit]) -> Any:
        """Run the predictor once a predictor backend is installed."""
        if state is None:
            raise InputError("state must be non-None")
        if not isinstance(edits, Sequence):
            raise InputError(
                "edits must be a sequence of RelEdit values",
                details={"type": type(edits).__name__},
            )
        for idx, edit in enumerate(edits):
            if not isinstance(edit, RelEdit):
                raise InputError(
                    "edits must contain RelEdit values",
                    details={"index": idx, "type": type(edit).__name__},
                )
        with fail_closed_network_guard():
            _raise_backend_not_ready("predict", self.backend, self.model_dir)


def probe_backends(model_dir: str | Path | None = None) -> tuple[BackendProbe, ...]:
    """Probe runtime backends in RFC-0010 auto-selection order."""
    root = None if model_dir is None else Path(model_dir).expanduser()
    return (
        _probe_coreml(root),
        _probe_cuda(root),
        _probe_onnx(root),
        BackendProbe(BACKEND_CPU, True, "portable CPU fallback is always available"),
    )


def select_backend(
    backend: str = BACKEND_AUTO,
    *,
    probes: Sequence[BackendProbe] | None = None,
) -> str:
    """Select a backend from probe results, or raise if the requested one is unavailable."""
    normalized = _normalize_backend(backend)
    observed = tuple(probe_backends() if probes is None else probes)
    by_backend = {probe.backend: probe for probe in observed}

    if normalized == BACKEND_AUTO:
        for name in BACKEND_PRIORITY:
            probe = by_backend.get(name)
            if probe is not None and probe.available:
                return name
        raise BackendUnsupportedError(
            "no runtime backend is available",
            details={"probes": [_probe_details(probe) for probe in observed]},
        )

    probe = by_backend.get(normalized)
    if probe is None:
        raise BackendUnsupportedError(
            "requested runtime backend was not probed",
            details={"backend": normalized, "probed": sorted(by_backend)},
        )
    if not probe.available:
        raise BackendUnsupportedError(
            "requested runtime backend is unavailable",
            details={"backend": normalized, "reason": probe.reason},
        )
    return normalized


@contextlib.contextmanager
def fail_closed_network_guard() -> Iterator[None]:
    """Block common network entry points inside an inference path."""

    def _blocked(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise NetworkCallProhibitedError(
            "runtime network call attempted after setup",
            remediation="perform downloads only through explicit setup/update commands",
        )

    with contextlib.ExitStack() as stack:
        for target in (
            "socket.create_connection",
            "socket.socket.connect",
            "urllib.request.urlopen",
            "http.client.HTTPConnection.connect",
            "http.client.HTTPSConnection.connect",
        ):
            stack.enter_context(patch(target, _blocked))
        yield


def _probe_coreml(model_dir: Path | None) -> BackendProbe:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin" or machine not in _APPLE_SILICON_MACHINES:
        return BackendProbe(
            BACKEND_COREML,
            False,
            f"Core ML requires Apple Silicon macOS; observed {system}/{machine}",
        )
    if importlib.util.find_spec("coremltools") is None:
        return BackendProbe(BACKEND_COREML, False, "coremltools is not installed")
    artifact_reason = _artifact_reason(model_dir, ("model.mlpackage", "model.mlmodelc"))
    if artifact_reason is not None:
        return BackendProbe(BACKEND_COREML, False, artifact_reason)
    return BackendProbe(BACKEND_COREML, True, "Core ML runtime and model artifact available")


def _probe_cuda(model_dir: Path | None) -> BackendProbe:
    available, reason = _torch_cuda_available()
    if not available:
        return BackendProbe(BACKEND_CUDA, False, reason)
    artifact_reason = _artifact_reason(model_dir, ("predictor.safetensors",))
    if artifact_reason is not None:
        return BackendProbe(BACKEND_CUDA, False, artifact_reason)
    return BackendProbe(BACKEND_CUDA, True, "torch CUDA backend and model artifact available")


def _probe_onnx(model_dir: Path | None) -> BackendProbe:
    if importlib.util.find_spec("onnxruntime") is None:
        return BackendProbe(BACKEND_ONNX, False, "onnxruntime is not installed")
    artifact_reason = _artifact_reason(model_dir, ("predictor.onnx",))
    if artifact_reason is not None:
        return BackendProbe(BACKEND_ONNX, False, artifact_reason)
    return BackendProbe(BACKEND_ONNX, True, "ONNX Runtime and model artifact available")


def _torch_cuda_available() -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None:
        return False, "torch is not installed"
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on optional torch installation
        return False, f"torch import failed: {exc}"
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return False, "torch CUDA is not available"
    return True, "torch CUDA is available"


def _artifact_reason(model_dir: Path | None, names: tuple[str, ...]) -> str | None:
    if model_dir is None:
        return None
    if any((model_dir / name).exists() for name in names):
        return None
    return "missing backend artifact(s): " + ", ".join(names)


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str) or not backend:
        raise BackendUnsupportedError(
            "runtime backend must be a non-empty string",
            details={"backend": backend, "type": type(backend).__name__},
        )
    normalized = backend.lower()
    if normalized not in _SUPPORTED_BACKENDS:
        raise BackendUnsupportedError(
            "unsupported runtime backend",
            details={"backend": backend, "supported": list(_SUPPORTED_BACKENDS)},
        )
    return normalized


def _probe_details(probe: BackendProbe) -> dict[str, str | bool]:
    return {
        "backend": probe.backend,
        "available": probe.available,
        "reason": probe.reason,
    }


def _raise_backend_not_ready(operation: str, backend: str, model_dir: Path) -> NoReturn:
    raise RuntimeSetupError(
        "deploy runtime backend operation is not available yet",
        details={"operation": operation, "backend": backend, "model_dir": str(model_dir)},
        remediation="score/export backends land with the scorer and deploy backend issues",
    )
