# SPDX-License-Identifier: Apache-2.0
"""Durably publish deterministic JSON without replacing an existing winner."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

_ANCHOR_CAPABLE_AT_IMPORT = (
    all(function in os.supports_dir_fd for function in (os.open, os.link, os.stat, os.unlink))
    and os.link in os.supports_follow_symlinks
    and os.stat in os.supports_follow_symlinks
)


class ImmutableJsonError(ValueError):
    """Raised when an immutable JSON destination cannot be safely reused."""


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create ``path`` or accept an identical existing regular file.

    On platforms with ``dir_fd`` support, every temporary-file, link, read, and
    cleanup operation is anchored to one opened parent directory.  The textual
    parent path is checked again before success so a concurrent rename/symlink
    swap fails closed instead of publishing attacker-controlled bytes.
    """
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _supports_anchored_directory_operations():
        _write_immutable_json_anchored(path, encoded)
        return
    raise ImmutableJsonError(
        "secure immutable JSON publication requires anchored dir_fd operations; "
        "this platform is unsupported"
    )


def _supports_anchored_directory_operations() -> bool:
    return getattr(os, "O_DIRECTORY", None) is not None and _ANCHOR_CAPABLE_AT_IMPORT


def _write_immutable_json_anchored(path: Path, encoded: bytes) -> None:
    directory, directory_metadata = _open_stable_parent_directory(path.parent)
    temporary_name = ""
    temporary_descriptor = -1
    temporary_metadata: os.stat_result | None = None
    created_destination = False
    publication_accepted = False
    try:
        temporary_name, temporary_descriptor = _create_temporary_at(directory, path.name)
        temporary_metadata = os.fstat(temporary_descriptor)
        _write_all(temporary_descriptor, encoded)
        os.fsync(temporary_descriptor)

        for _attempt in range(128):
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    observed = _read_regular_file_at(directory, path.name, path)
                except FileNotFoundError:
                    continue
                if observed != encoded:
                    raise ImmutableJsonError(
                        f"refusing to replace different bytes at immutable output: {path}"
                    ) from None
                break
            else:
                created_destination = True
                destination_metadata = os.stat(
                    path.name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
                if not _same_file_identity(temporary_metadata, destination_metadata):
                    raise ImmutableJsonError(
                        f"immutable output is not linked to the trusted temporary file: {path}"
                    )
                if _read_regular_file_at(directory, path.name, path) != encoded:
                    raise ImmutableJsonError(
                        f"immutable output bytes changed during publication: {path}"
                    )
                break
        else:
            raise ImmutableJsonError(
                f"immutable output changed too frequently during publication: {path}"
            )

        os.fsync(directory)
        try:
            _assert_parent_path_matches_open_directory(
                path.parent,
                directory_metadata,
                directory,
            )
        except ImmutableJsonError:
            if created_destination:
                _unlink_if_same_file_at(
                    directory,
                    path.name,
                    temporary_metadata,
                )
                created_destination = False
                os.fsync(directory)
            raise
        publication_accepted = True
    finally:
        try:
            if created_destination and not publication_accepted and temporary_metadata is not None:
                try:
                    _unlink_if_same_file_at(directory, path.name, temporary_metadata)
                finally:
                    os.fsync(directory)
        finally:
            try:
                if temporary_descriptor >= 0:
                    os.close(temporary_descriptor)
            finally:
                removed_temporary = False
                try:
                    if temporary_name:
                        try:
                            os.unlink(temporary_name, dir_fd=directory)
                        except FileNotFoundError:
                            pass
                        else:
                            removed_temporary = True
                finally:
                    try:
                        if removed_temporary:
                            os.fsync(directory)
                    finally:
                        os.close(directory)


def _open_stable_parent_directory(path: Path) -> tuple[int, os.stat_result]:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ImmutableJsonError(f"immutable output parent does not exist: {path}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ImmutableJsonError(f"immutable output parent is a symlink or non-directory: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ImmutableJsonError(
            f"immutable output parent cannot be opened without following symlinks: {path}"
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(before, opened):
        os.close(descriptor)
        raise ImmutableJsonError(
            f"immutable output parent directory changed during publication: {path}"
        )
    return descriptor, opened


def _create_temporary_at(directory: int, destination_name: str) -> tuple[str, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = f".{destination_name}.{secrets.token_hex(12)}.tmp"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=directory)
        except FileExistsError:
            continue
    raise ImmutableJsonError("could not allocate a unique immutable-output temporary file")


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short write while publishing immutable JSON")
        remaining = remaining[written:]


def _read_regular_file_at(directory: int, name: str, display_path: Path) -> bytes:
    metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ImmutableJsonError(
            f"immutable output is a symlink or non-regular file: {display_path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(metadata, opened):
            raise ImmutableJsonError(
                f"immutable output is a replaced, symlink, or non-regular file: {display_path}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _assert_parent_path_matches_open_directory(
    path: Path,
    expected: os.stat_result,
    descriptor: int,
) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise ImmutableJsonError(
            f"immutable output parent directory changed during publication: {path}"
        ) from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or not _same_file_identity(expected, observed)
        or not _same_file_identity(expected, opened)
    ):
        raise ImmutableJsonError(
            f"immutable output parent directory changed during publication: {path}"
        )


def _unlink_if_same_file_at(
    directory: int,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _same_file_identity(expected, observed):
        os.unlink(name, dir_fd=directory)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory_flag,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
