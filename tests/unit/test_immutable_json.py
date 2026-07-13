# SPDX-License-Identifier: Apache-2.0
"""Adversarial contracts for immutable JSON publication."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tools.data._immutable_json import (
    ImmutableJsonError,
    supports_secure_immutable_json_publication,
    write_immutable_json,
)

requires_anchored_dir_fd = pytest.mark.skipif(
    not supports_secure_immutable_json_publication(),
    reason="secure immutable publication requires anchored dir_fd operations",
)


@requires_anchored_dir_fd
def test_writer_fails_closed_when_the_parent_path_is_swapped_after_temp_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "publish"
    moved_parent = tmp_path / "publish-moved"
    attacker = tmp_path / "attacker"
    parent.mkdir()
    attacker.mkdir()
    output = parent / "result.json"
    real_link = os.link
    swapped = False

    def swap_parent_then_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            parent.rename(moved_parent)
            try:
                parent.symlink_to(attacker, target_is_directory=True)
            except OSError as exc:
                pytest.skip(f"directory symlinks unavailable: {exc}")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr("tools.data._immutable_json.os.link", swap_parent_then_link)

    with pytest.raises(ImmutableJsonError, match="parent directory changed during publication"):
        write_immutable_json(output, {"trusted": True})

    assert swapped
    assert not (attacker / output.name).exists()
    assert not (moved_parent / output.name).exists()
    assert not list(moved_parent.glob(f".{output.name}.*.tmp"))


@requires_anchored_dir_fd
def test_anchored_writer_cleans_every_link_when_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr("tools.data._immutable_json.os.fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="injected directory fsync failure"):
        write_immutable_json(output, {"trusted": True})

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_writer_fails_closed_before_writes_when_anchored_operations_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "missing-parent" / "result.json"

    monkeypatch.setattr(
        "tools.data._immutable_json._supports_anchored_directory_operations",
        lambda: False,
    )

    with pytest.raises(
        ImmutableJsonError,
        match="requires anchored dir_fd operations; this platform is unsupported",
    ):
        write_immutable_json(output, {"trusted": True})

    assert not output.exists()
    assert not output.parent.exists()
    assert not list(tmp_path.rglob(f".{output.name}.*.tmp"))
