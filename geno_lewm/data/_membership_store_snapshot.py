# SPDX-License-Identifier: Apache-2.0
"""Single-read private snapshots for membership artifact verification and lookup."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from geno_lewm.data._membership_store_contract import _ARTIFACT_FILE_NAMES
from geno_lewm.errors import InputError


@dataclass(frozen=True, slots=True)
class _CapturedFile:
    sha256: str
    size_bytes: int


class _CapturedMembershipStore:
    """Private immutable copies produced from one descriptor per published file."""

    __slots__ = ("files", "root")

    def __init__(self, root: Path, files: dict[str, _CapturedFile]) -> None:
        self.root = root
        self.files = files

    def close(self) -> None:
        for path in self.root.glob("*"):
            with suppress(OSError):
                path.chmod(0o600)
        shutil.rmtree(self.root, ignore_errors=True)

    def retain_only(self, names: set[str]) -> None:
        """Release verified files that are not needed for subsequent runtime lookup."""
        for name in set(self.files) - names:
            path = self.root / name
            with suppress(OSError):
                path.chmod(0o600)
            path.unlink()
            del self.files[name]

    def __enter__(self) -> _CapturedMembershipStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _capture_membership_store(store_dir: Path) -> _CapturedMembershipStore:
    """Capture each published artifact exactly once through a checked descriptor."""
    source_root = Path(store_dir)
    capture_root = Path(tempfile.mkdtemp(prefix="geno-lewm-membership-snapshot-"))
    if platform.system() == "Windows":
        return _capture_membership_store_by_path(source_root, capture_root)
    root_fd: int | None = None
    try:
        root_fd = os.open(
            source_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise InputError("membership store root must be a real directory")
        observed = set(os.listdir(root_fd))  # noqa: PTH208 - descriptor-bound listing
        if observed != _ARTIFACT_FILE_NAMES:
            raise InputError(
                "membership store files do not match the exact layout",
                details={
                    "missing": sorted(_ARTIFACT_FILE_NAMES - observed),
                    "unexpected": sorted(observed - _ARTIFACT_FILE_NAMES),
                },
            )
        captured: dict[str, _CapturedFile] = {}
        for name in sorted(_ARTIFACT_FILE_NAMES):
            captured[name] = _capture_file(root_fd, name, capture_root / name)
        if set(os.listdir(root_fd)) != _ARTIFACT_FILE_NAMES:  # noqa: PTH208
            raise InputError("membership store layout changed during capture")
        return _CapturedMembershipStore(capture_root, captured)
    except InputError:
        shutil.rmtree(capture_root, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(capture_root, ignore_errors=True)
        raise InputError("membership store layout cannot be captured") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _capture_file(root_fd: int, name: str, destination: Path) -> _CapturedFile:
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        entry_mode = os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_mode
        if stat.S_ISLNK(entry_mode):
            raise InputError("membership store must not contain symlinks", details={"path": name})
        source_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise InputError(
                "membership store must contain only exact top-level files",
                details={"path": name},
            )
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        return _copy_descriptor(source_fd, destination_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)


def _capture_membership_store_by_path(
    source_root: Path, capture_root: Path
) -> _CapturedMembershipStore:
    """Windows fallback where directory descriptors are unavailable."""
    try:
        invalid_root = source_root.is_symlink() or not source_root.is_dir()
        paths = tuple(source_root.iterdir()) if not invalid_root else ()
        if invalid_root:
            raise InputError("membership store root must be a real directory")
        observed = {path.name for path in paths}
        if observed != _ARTIFACT_FILE_NAMES:
            raise InputError("membership store files do not match the exact layout")
        captured: dict[str, _CapturedFile] = {}
        for name in sorted(_ARTIFACT_FILE_NAMES):
            captured[name] = _capture_path_file(source_root / name, capture_root / name)
        if {path.name for path in source_root.iterdir()} != _ARTIFACT_FILE_NAMES:
            raise InputError("membership store layout changed during capture")
        return _CapturedMembershipStore(capture_root, captured)
    except InputError:
        shutil.rmtree(capture_root, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(capture_root, ignore_errors=True)
        raise InputError("membership store layout cannot be captured") from exc


def _capture_path_file(source: Path, destination: Path) -> _CapturedFile:
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        if source.is_symlink():
            raise InputError(
                "membership store must not contain symlinks", details={"path": source.name}
            )
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise InputError("membership store must contain only exact top-level files")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o400,
        )
        return _copy_descriptor(source_fd, destination_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)


def _copy_descriptor(source_fd: int, destination_fd: int) -> _CapturedFile:
    digest = hashlib.sha256()
    size_bytes = 0
    while chunk := os.read(source_fd, 1024 * 1024):
        digest.update(chunk)
        size_bytes += len(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written < 1:
                raise InputError("membership artifact capture produced a short write")
            view = view[written:]
    os.fsync(destination_fd)
    return _CapturedFile("sha256:" + digest.hexdigest(), size_bytes)
