# SPDX-License-Identifier: Apache-2.0
"""Durably publish deterministic JSON without replacing an existing winner."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path


class ImmutableJsonError(ValueError):
    """Raised when an immutable JSON destination cannot be safely reused."""


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create ``path`` or accept an identical existing regular file."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        while True:
            try:
                os.link(temporary, path)
            except FileExistsError:
                try:
                    observed = _read_regular_file_without_following_symlinks(path)
                except FileNotFoundError:
                    continue
                if observed != encoded:
                    raise ImmutableJsonError(
                        f"refusing to replace different bytes at immutable output: {path}"
                    ) from None
                return
            _fsync_directory(path.parent)
            return
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_directory(path.parent)


def _read_regular_file_without_following_symlinks(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ImmutableJsonError(f"immutable output is a symlink or non-regular file: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
        ):
            raise ImmutableJsonError(
                f"immutable output is a replaced, symlink, or non-regular file: {path}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
