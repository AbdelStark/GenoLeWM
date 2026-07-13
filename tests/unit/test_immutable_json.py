# SPDX-License-Identifier: Apache-2.0
"""Adversarial contracts for immutable JSON publication."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.data._immutable_json import ImmutableJsonError, write_immutable_json


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
