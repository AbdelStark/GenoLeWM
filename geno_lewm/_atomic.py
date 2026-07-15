# SPDX-License-Identifier: Apache-2.0
"""Secure same-directory atomic file replacement primitives."""

from __future__ import annotations

import errno
import os
import secrets
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import BinaryIO, TextIO, cast

from geno_lewm.errors import InputError, RuntimeSetupError

_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_CLOEXEC"):
    _CREATE_FLAGS |= os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    _CREATE_FLAGS |= os.O_NOFOLLOW

_ANCHOR_CAPABLE_AT_IMPORT = (
    getattr(os, "O_DIRECTORY", None) is not None
    and getattr(os, "O_NOFOLLOW", None) is not None
    and all(
        function in os.supports_dir_fd
        for function in (os.open, os.link, os.rename, os.stat, os.unlink)
    )
    and os.link in os.supports_follow_symlinks
    and os.stat in os.supports_follow_symlinks
)

_LockKey = tuple[int, int, str, int]
_HELD_WRITER_LOCKS: ContextVar[frozenset[_LockKey]] = ContextVar(
    "geno_lewm_held_writer_locks",
    default=frozenset(),
)


@contextmanager
def atomic_binary_writer(path: Path) -> Iterator[BinaryIO]:
    """Yield an exclusive temporary stream and atomically install it on success."""
    with _atomic_writer(path, binary=True) as stream:
        yield cast(BinaryIO, stream)


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Yield an exclusive UTF-8 temporary stream and atomically install it on success."""
    with _atomic_writer(path, binary=False) as stream:
        yield cast(TextIO, stream)


def supports_secure_atomic_publication() -> bool:
    """Return whether anchored writer operations are available."""
    return _supports_anchored_directory_operations()


@contextmanager
def exclusive_writer_lock(path: Path) -> Iterator[None]:
    """Reject another writer, while allowing same-context nested acquisition."""
    _require_anchored_directory_operations()
    path = _prepare_path(path)
    directory, directory_identity = _open_stable_parent_directory(path.parent)
    try:
        with _exclusive_writer_lock_at(
            directory,
            directory_identity,
            parent_path=path.parent,
            target_name=path.name,
            display_path=path,
        ):
            yield
    finally:
        os.close(directory)


@contextmanager
def _atomic_writer(path: Path, *, binary: bool) -> Iterator[BinaryIO | TextIO]:
    _require_anchored_directory_operations()
    path = _prepare_path(path)
    directory, directory_identity = _open_stable_parent_directory(path.parent)
    try:
        with (
            _exclusive_writer_lock_at(
                directory,
                directory_identity,
                parent_path=path.parent,
                target_name=path.name,
                display_path=path,
            ),
            _atomic_writer_at(
                directory,
                directory_identity,
                path=path,
                binary=binary,
            ) as stream,
        ):
            yield stream
    finally:
        os.close(directory)


@contextmanager
def _exclusive_writer_lock_at(
    directory: int,
    directory_identity: os.stat_result,
    *,
    parent_path: Path,
    target_name: str,
    display_path: Path,
) -> Iterator[None]:
    key = (
        directory_identity.st_dev,
        directory_identity.st_ino,
        target_name,
        threading.get_ident(),
    )
    held = _HELD_WRITER_LOCKS.get()
    if key in held:
        _assert_parent_path_matches_open_directory(parent_path, directory_identity, directory)
        yield
        _assert_parent_path_matches_open_directory(parent_path, directory_identity, directory)
        return

    lock_name = f".{target_name}.lock"
    try:
        descriptor = os.open(lock_name, _CREATE_FLAGS, 0o600, dir_fd=directory)
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ELOOP}:
            raise InputError(
                "another writer is already active for this target",
                details={"path": str(display_path), "lock": lock_name},
            ) from exc
        raise
    identity = os.fstat(descriptor)
    token = _HELD_WRITER_LOCKS.set(held | {key})
    try:
        _write_all(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        _assert_parent_path_matches_open_directory(parent_path, directory_identity, directory)
        yield
        _assert_parent_path_matches_open_directory(parent_path, directory_identity, directory)
    finally:
        try:
            os.close(descriptor)
        finally:
            try:
                _unlink_if_owned_at(directory, lock_name, identity)
                os.fsync(directory)
            finally:
                _HELD_WRITER_LOCKS.reset(token)


@contextmanager
def _atomic_writer_at(
    directory: int,
    directory_identity: os.stat_result,
    *,
    path: Path,
    binary: bool,
) -> Iterator[BinaryIO | TextIO]:
    temporary_name, descriptor, identity = _open_unique_temporary_at(directory, path.name)
    stream: BinaryIO | TextIO | None = None
    installed = False
    publication_committed = False
    backup: tuple[str, os.stat_result] | None = None
    try:
        if binary:
            stream = os.fdopen(descriptor, "wb")
        else:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        _assert_parent_path_matches_open_directory(path.parent, directory_identity, directory)
        backup = _backup_destination_at(directory, path.name)
        os.rename(
            temporary_name,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        installed = True
        installed_identity = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if not _same_file_identity(identity, installed_identity):
            raise InputError("atomic-write destination is not the trusted temporary file")
        os.fsync(directory)
        _assert_parent_path_matches_open_directory(path.parent, directory_identity, directory)
        publication_committed = True
        if backup is not None:
            _unlink_if_owned_at(directory, backup[0], backup[1], require_regular=False)
            backup = None
            os.fsync(directory)
    except BaseException:
        if installed and not publication_committed:
            _restore_or_remove_installed_destination(
                directory,
                destination_name=path.name,
                installed_identity=identity,
                backup=backup,
            )
            backup = None
            os.fsync(directory)
        raise
    finally:
        if stream is None:
            os.close(descriptor)
        elif not stream.closed:
            stream.close()
        if not installed:
            _unlink_if_owned_at(directory, temporary_name, identity)
        if backup is not None:
            _unlink_if_owned_at(directory, backup[0], backup[1], require_regular=False)
        os.fsync(directory)


def _prepare_path(path: Path) -> Path:
    # ``resolve()`` would follow the very parent symlink this boundary rejects.
    absolute = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    absolute.parent.mkdir(parents=True, exist_ok=True)
    try:
        observed = absolute.parent.lstat()
    except FileNotFoundError as exc:  # pragma: no cover - mkdir above owns this boundary.
        raise InputError("atomic-write parent directory is missing") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise InputError("atomic-write parent is a symlink or non-directory")
    return absolute


def _supports_anchored_directory_operations() -> bool:
    return _ANCHOR_CAPABLE_AT_IMPORT


def _require_anchored_directory_operations() -> None:
    if not _supports_anchored_directory_operations():
        raise InputError(
            "secure atomic publication requires anchored directory operations",
            remediation="run production checkpoint/report publication on a supported POSIX filesystem",
        )


def _open_stable_parent_directory(path: Path) -> tuple[int, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise InputError("atomic-write parent is a symlink or non-directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputError("atomic-write parent cannot be opened without following symlinks") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(before, opened):
        os.close(descriptor)
        raise InputError("atomic-write parent directory changed during publication")
    return descriptor, opened


def _assert_parent_path_matches_open_directory(
    path: Path,
    expected: os.stat_result,
    descriptor: int,
) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise InputError("atomic-write parent directory changed during publication") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or not _same_file_identity(expected, observed)
        or not _same_file_identity(expected, opened)
    ):
        raise InputError("atomic-write parent directory changed during publication")


def _open_unique_temporary_at(
    directory: int,
    destination_name: str,
) -> tuple[str, int, os.stat_result]:
    for _attempt in range(128):
        name = f".{destination_name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(name, _CREATE_FLAGS, 0o600, dir_fd=directory)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ELOOP}:
                continue
            raise
        return name, descriptor, os.fstat(descriptor)
    raise InputError("could not allocate a unique atomic-write temporary file")


def _backup_destination_at(
    directory: int,
    destination_name: str,
) -> tuple[str, os.stat_result] | None:
    try:
        os.stat(destination_name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    for _attempt in range(128):
        backup_name = f".geno-lewm-backup-{os.getpid()}-{secrets.token_hex(16)}"
        try:
            os.link(
                destination_name,
                backup_name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise InputError("atomic-write destination could not be backed up safely") from exc
        return backup_name, os.stat(backup_name, dir_fd=directory, follow_symlinks=False)
    raise InputError("could not allocate an atomic-write destination backup")


def _restore_or_remove_installed_destination(
    directory: int,
    *,
    destination_name: str,
    installed_identity: os.stat_result,
    backup: tuple[str, os.stat_result] | None,
) -> None:
    try:
        observed = os.stat(destination_name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        observed = None
    if observed is None or not _same_file_identity(installed_identity, observed):
        return
    if backup is None:
        _unlink_if_owned_at(directory, destination_name, installed_identity)
        return
    os.rename(
        backup[0],
        destination_name,
        src_dir_fd=directory,
        dst_dir_fd=directory,
    )


def _unlink_if_owned_at(
    directory: int,
    name: str,
    identity: os.stat_result,
    *,
    require_regular: bool = True,
) -> bool:
    try:
        observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not _same_file_identity(identity, observed) or (
        require_regular and not stat.S_ISREG(observed.st_mode)
    ):
        return False

    quarantine = f".geno-lewm-cleanup-{os.getpid()}-{secrets.token_hex(16)}"
    try:
        os.rename(name, quarantine, src_dir_fd=directory, dst_dir_fd=directory)
    except FileNotFoundError:
        return False
    moved = os.stat(quarantine, dir_fd=directory, follow_symlinks=False)
    if _same_file_identity(identity, moved):
        os.unlink(quarantine, dir_fd=directory)
        return True

    try:
        os.link(
            quarantine,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise InputError(
            "atomic writer cleanup ownership changed while another writer acquired the target",
            details={"preserved_name": quarantine},
        ) from exc
    os.unlink(quarantine, dir_fd=directory)
    return False


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise RuntimeSetupError("short write while publishing atomic output")
        remaining = remaining[written:]
