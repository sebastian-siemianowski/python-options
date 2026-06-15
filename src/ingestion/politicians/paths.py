"""Filesystem paths for politician disclosure data."""

from __future__ import annotations

import os
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POLITICIANS_DATA_DIR = SRC_DIR / "data" / "politicians"


def get_politicians_data_dir(data_root: str | Path | None = None) -> Path:
    """Return the politician data root, honoring env override for tests/ops."""
    if data_root is not None:
        return Path(data_root)
    configured = os.getenv("POLITICIANS_DATA_DIR")
    if configured:
        return Path(configured)
    return DEFAULT_POLITICIANS_DATA_DIR


def ensure_politicians_data_dirs(data_root: str | Path | None = None) -> Path:
    """Create the common politician data directories and return the root."""
    root = get_politicians_data_dir(data_root)
    for subdir in ("raw", "manifests"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root
