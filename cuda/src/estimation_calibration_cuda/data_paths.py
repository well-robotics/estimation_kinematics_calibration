"""Optional paths for real-data tests and profiling."""

from __future__ import annotations

import os
from pathlib import Path


def optional_env_path(name: str) -> Path | None:
    """Return an expanded environment path, or ``None`` when unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()


def leg_bical_data_root() -> Path | None:
    return optional_env_path("LEG_BICAL_DATA_ROOT")


def leg_bical_golden_path() -> Path | None:
    return optional_env_path("LEG_BICAL_GOLDEN_PATH")
