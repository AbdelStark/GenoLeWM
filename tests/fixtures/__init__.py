# SPDX-License-Identifier: Apache-2.0
"""Committed test fixtures (testing contract).

The Python module exposes :func:`load_json` for tests that prefer to
read a fixture by name rather than via the ``fixtures_dir`` pytest
fixture. The fixture files themselves live next to this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent


def load_json(name: str) -> Any:
    """Load ``<FIXTURES_DIR>/<name>`` as JSON.

    ``name`` may include sub-directories. The function raises
    ``FileNotFoundError`` if the fixture is missing.
    """
    path = FIXTURES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))
