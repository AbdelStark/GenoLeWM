"""Behavioral tests for production Carbon checkpoint continuation."""

from __future__ import annotations

import random
import threading
from pathlib import Path

import pytest

import geno_lewm._atomic as atomic_module
from geno_lewm._atomic import atomic_text_writer, exclusive_writer_lock
from geno_lewm.errors import InputError
from geno_lewm.training import resume as resume_module
from geno_lewm.training.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_state,
    load_resume_checkpoint,
    restore_rng_state,
    write_resume_checkpoint,
)


def test_rng_state_round_trips_every_training_domain() -> None:
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    random.seed(17)
    numpy.random.seed(18)
    torch.manual_seed(19)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20)

    state = capture_rng_state()
    expected = (
        random.random(),
        float(numpy.random.random()),
        torch.rand(3),
    )
    random.random()
    numpy.random.random()
    torch.rand(7)

    restore_rng_state(state)
    observed = (
        random.random(),
        float(numpy.random.random()),
        torch.rand(3),
    )

    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)
    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_production_checkpoint_round_trips_as_one_closed_atomic_payload(tmp_path) -> None:
    path = tmp_path / "predictor_checkpoint.pt"
    payload = _write_fixture_checkpoint(path)

    loaded = load_resume_checkpoint(path)

    assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert loaded["payload_digest"] == payload["payload_digest"]
    assert loaded["progress"]["steps_completed"] == 3
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_checkpoint_loader_rejects_raw_tensor_tampering(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    _write_fixture_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["states"]["predictor"]["weight"][0] = 99.0
    torch.save(payload, path)

    with pytest.raises(InputError, match="payload digest"):
        load_resume_checkpoint(path)


def test_checkpoint_loader_rejects_invalid_closed_progress_even_with_fresh_digest(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    _write_fixture_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["progress"]["collapse_alert_count"] = -1
    payload["payload_digest"] = resume_module._payload_digest(payload)
    torch.save(payload, path)

    with pytest.raises(InputError, match="collapse-alert count"):
        load_resume_checkpoint(path)


@pytest.mark.parametrize("field", ["trainer_state", "rng_state"])
def test_checkpoint_loader_rejects_missing_closed_top_level_field(
    tmp_path: Path,
    field: str,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    _write_fixture_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload[field]
    torch.save(payload, path)

    with pytest.raises(InputError, match="fields do not match the closed contract"):
        load_resume_checkpoint(path)


def test_interrupted_atomic_checkpoint_write_preserves_previous_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    original = _write_fixture_checkpoint(path)
    original_bytes = path.read_bytes()

    def fail_after_partial_write(_payload, stream) -> None:
        stream.write(b"partial replacement")
        stream.flush()
        raise RuntimeError("injected checkpoint write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected checkpoint write failure"):
        _write_fixture_checkpoint(path)

    assert path.read_bytes() == original_bytes
    assert load_resume_checkpoint(path)["payload_digest"] == original["payload_digest"]
    assert not path.with_name(f".{path.name}.tmp").exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    assert not path.with_name(f".{path.name}.lock").exists()


@pytest.mark.skipif(
    not atomic_module._supports_anchored_directory_operations(),
    reason="directory durability commit-point test requires anchored operations",
)
def test_atomic_writer_keeps_durable_replacement_if_backup_cleanup_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    path.write_text("old\n", encoding="utf-8")
    original_fsync = atomic_module.os.fsync
    directory_fsyncs = 0

    def fail_second_directory_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if atomic_module.stat.S_ISDIR(atomic_module.os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("injected backup-cleanup fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(atomic_module.os, "fsync", fail_second_directory_fsync)

    with pytest.raises(OSError, match="backup-cleanup fsync failure"):
        with atomic_text_writer(path) as stream:
            stream.write("new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(".geno-lewm-backup-*")) == []
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    assert not path.with_name(f".{path.name}.lock").exists()


@pytest.mark.skipif(
    not atomic_module._supports_anchored_directory_operations(),
    reason="ownership-race rejection requires anchored directory operations",
)
def test_checkpoint_cleanup_never_unlinks_a_replacement_temp_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    replacement = tmp_path / "replacement-owned-elsewhere"
    replacement.write_bytes(b"replacement bytes")
    original_rename = atomic_module.os.rename
    swapped = False

    def fail_after_partial_write(_payload, stream) -> None:
        stream.write(b"partial checkpoint")
        stream.flush()
        raise RuntimeError("injected checkpoint write failure")

    def replace_temp_during_cleanup(source, destination, *args, **kwargs):
        nonlocal swapped
        source_name = str(source)
        destination_name = str(destination)
        if (
            not swapped
            and source_name.endswith(".tmp")
            and destination_name.startswith(".geno-lewm-cleanup-")
        ):
            directory = kwargs["src_dir_fd"]
            atomic_module.os.unlink(source, dir_fd=directory)
            original_rename(
                replacement.name,
                source,
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            swapped = True
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    monkeypatch.setattr(atomic_module.os, "rename", replace_temp_during_cleanup)

    with pytest.raises(RuntimeError, match="injected checkpoint write failure"):
        _write_fixture_checkpoint(path)

    preserved = list(tmp_path.glob(f".{path.name}.*.tmp"))
    assert swapped
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"replacement bytes"
    assert not replacement.exists()
    assert not path.exists()


def test_checkpoint_rejects_concurrent_writer_for_same_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original_save = torch.save

    def blocking_save(payload, stream) -> None:
        if threading.current_thread().name == "checkpoint-owner":
            entered.set()
            assert release.wait(timeout=5)
        original_save(payload, stream)

    def first_writer() -> None:
        try:
            _write_fixture_checkpoint(path)
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    monkeypatch.setattr(torch, "save", blocking_save)
    owner = threading.Thread(target=first_writer, name="checkpoint-owner")
    owner.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(InputError, match="already active"):
            _write_fixture_checkpoint(path)
    finally:
        release.set()
        owner.join(timeout=5)

    assert not owner.is_alive()
    assert errors == []
    assert load_resume_checkpoint(path)["progress"]["steps_completed"] == 3


def test_writer_lock_is_reentrant_in_the_same_thread_context(tmp_path: Path) -> None:
    target = tmp_path / "production-carbon-run"

    with exclusive_writer_lock(target):
        with exclusive_writer_lock(target):
            assert target.with_name(f".{target.name}.lock").is_file()

    assert not target.with_name(f".{target.name}.lock").exists()


def test_checkpoint_writer_never_follows_or_removes_precreated_candidate_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "predictor_checkpoint.pt"
    victim = tmp_path / "victim.txt"
    victim.write_text("owned elsewhere\n", encoding="utf-8")
    tokens = iter(("precreated", "writer-owned", "cleanup"))
    monkeypatch.setattr(atomic_module.secrets, "token_hex", lambda _size: next(tokens))
    precreated_temporary = path.with_name(
        f".{path.name}.{atomic_module.os.getpid()}.precreated.tmp"
    )
    try:
        precreated_temporary.symlink_to(victim)
    except OSError as exc:  # pragma: no cover - platform capability boundary.
        pytest.skip(f"symlinks unavailable: {exc}")

    _write_fixture_checkpoint(path)

    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert precreated_temporary.is_symlink()
    assert load_resume_checkpoint(path)["progress"]["steps_completed"] == 3


def test_checkpoint_writer_rejects_precreated_lock_symlink_without_removing_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictor_checkpoint.pt"
    victim = tmp_path / "victim.txt"
    victim.write_text("owned elsewhere\n", encoding="utf-8")
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        lock_path.symlink_to(victim)
    except OSError as exc:  # pragma: no cover - platform capability boundary.
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(InputError, match="already active"):
        _write_fixture_checkpoint(path)

    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert lock_path.is_symlink()
    assert not path.exists()


def test_checkpoint_writer_rejects_symlinked_parent_without_writing_target(
    tmp_path: Path,
) -> None:
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    linked_parent = tmp_path / "run"
    try:
        linked_parent.symlink_to(attacker, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform capability boundary.
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(InputError, match=r"parent.*symlink"):
        _write_fixture_checkpoint(linked_parent / "predictor_checkpoint.pt")

    assert not (attacker / "predictor_checkpoint.pt").exists()
    assert list(attacker.iterdir()) == []


def test_checkpoint_writer_fails_closed_when_anchored_operations_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "missing-parent" / "predictor_checkpoint.pt"
    monkeypatch.setattr(
        atomic_module,
        "_supports_anchored_directory_operations",
        lambda: False,
    )

    with pytest.raises(InputError, match="requires anchored directory operations"):
        _write_fixture_checkpoint(path)

    assert not path.parent.exists()
    assert not path.exists()


@pytest.mark.skipif(
    not atomic_module._supports_anchored_directory_operations(),
    reason="parent-swap rejection requires anchored directory operations",
)
def test_checkpoint_writer_rejects_parent_swap_and_cleans_owned_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "run"
    moved_parent = tmp_path / "run-moved"
    attacker = tmp_path / "attacker"
    parent.mkdir()
    attacker.mkdir()
    path = parent / "predictor_checkpoint.pt"
    original = _write_fixture_checkpoint(path)
    original_bytes = path.read_bytes()
    original_rename = atomic_module.os.rename
    swapped = False

    def swap_parent_during_install(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and str(source).endswith(".tmp") and destination == path.name:
            original_rename(parent, moved_parent)
            try:
                parent.symlink_to(attacker, target_is_directory=True)
            except OSError as exc:  # pragma: no cover - platform capability boundary.
                pytest.skip(f"directory symlinks unavailable: {exc}")
            swapped = True
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(atomic_module.os, "rename", swap_parent_during_install)

    with pytest.raises(InputError, match="parent directory changed"):
        _write_fixture_checkpoint(path)

    restored = moved_parent / path.name
    assert swapped
    assert not (attacker / path.name).exists()
    assert restored.read_bytes() == original_bytes
    assert load_resume_checkpoint(restored)["payload_digest"] == original["payload_digest"]
    assert list(moved_parent.iterdir()) == [restored]


def _write_fixture_checkpoint(path: Path) -> dict[str, object]:
    torch = pytest.importorskip("torch")
    return write_resume_checkpoint(
        path,
        source={
            "commit_sha": "a" * 40,
            "tree_sha": "b" * 40,
            "package_version": "0.2.1",
        },
        training_contract={"target_steps": 8, "batch_size": 2, "config": {"seed": 7}},
        identities={"dataset": "sha256:" + ("c" * 64), "encoder": "sha256:" + ("d" * 64)},
        progress={
            "steps_completed": 3,
            "samples_consumed": 6,
            "consumed_window_ids": [f"w{index}" for index in range(6)],
            "collapse_alert_count": 1,
        },
        states={
            "predictor": {"weight": torch.tensor([1.0])},
            "action_encoder": {"weight": torch.tensor([2.0])},
            "optimizer": {"state": {}, "param_groups": []},
        },
        trainer_state={"schema_version": "fixture", "total_steps": 8},
        rng_state=capture_rng_state(),
        metric_history=[{"step": 1, "lr_multiplier": 0.5}],
    )
