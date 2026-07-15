# SPDX-License-Identifier: Apache-2.0
"""Platform-backed no-clobber atomic directory publication."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from pathlib import Path
from typing import Any

from geno_lewm.errors import InputError

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x4


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` and fail if any destination already exists."""
    system = platform.system()
    if system == "Darwin":
        libc: Any = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL)
        _require_rename_success(result, destination)
        return
    if system == "Linux":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise InputError("atomic no-clobber directory publication is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        _require_rename_success(result, destination)
        return
    if system == "Windows":
        try:
            source.rename(destination)
        except FileExistsError as exc:
            _raise_destination_exists(destination, cause=exc)
        return
    raise InputError(
        "atomic no-clobber directory publication is unavailable",
        details={"platform": system},
    )


def _require_rename_success(result: int, destination: Path) -> None:
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        _raise_destination_exists(destination)
    raise InputError(
        "atomic membership store publication failed",
        details={"path": str(destination), "errno": error, "error": os.strerror(error)},
    )


def _raise_destination_exists(destination: Path, *, cause: BaseException | None = None) -> None:
    if cause is None:
        raise InputError(
            "membership store output appeared before publication",
            details={"path": str(destination)},
            remediation="choose a new immutable artifact directory",
        )
    raise InputError(
        "membership store output appeared before publication",
        details={"path": str(destination)},
        remediation="choose a new immutable artifact directory",
    ) from cause
