"""
Story 3.8: Tuning Cache Versioning and Migration.

Provides:
  - Version detection for tune cache files
  - Migration chain v1 -> v2 (adds garch_params, ou_params)
  - Backup before migration
  - Automatic migration on load

Usage:
    from calibration.cache_version import load_and_migrate, CURRENT_CACHE_VERSION
    params = load_and_migrate("src/data/tune/SPY.json")
"""
import json
import os
import shutil
from typing import Optional

CURRENT_CACHE_VERSION = 2

# Default params injected during v1 -> v2 migration
GARCH_DEFAULTS = {
    "omega": 1e-6,
    "alpha": 0.05,
    "beta": 0.90,
    "persistence": 0.95,
    "long_run_vol": 0.01,
    "converged": False,
}

OU_DEFAULTS = {
    "kappa": 0.05,
    "theta": 0.0,
    "sigma_ou": 0.01,
    "half_life_days": 14.0,
}


def detect_version(data: dict) -> int:
    """Detect cache version. Unversioned files are v1."""
    return data.get("cache_version", 1)


def migrate_v1_to_v2(data: dict) -> dict:
    """
    v1 -> v2: Add garch_params and ou_params with defaults.
    Also stamps cache_version = 2.
    """
    if "garch_params" not in data:
        data["garch_params"] = dict(GARCH_DEFAULTS)
    if "ou_params" not in data:
        data["ou_params"] = dict(OU_DEFAULTS)
    data["cache_version"] = 2
    return data


# Migration chain: (from_version, to_version, function)
MIGRATIONS = [
    (1, 2, migrate_v1_to_v2),
]


def run_migrations(data: dict) -> dict:
    """Apply all needed migrations in sequence."""
    current = detect_version(data)
    for from_v, to_v, fn in MIGRATIONS:
        if current == from_v:
            data = fn(data)
            current = to_v
    return data


def load_and_migrate(filepath: str, backup: bool = True) -> Optional[dict]:
    """
    Load a tune cache file, migrate if needed, save back.
    
    Creates .bak backup before modifying.
    Returns migrated data, or None if file doesn't exist.
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    old_version = detect_version(data)
    data = run_migrations(data)
    new_version = detect_version(data)
    
    if new_version > old_version:
        # Backup then save
        if backup:
            bak_path = filepath + ".bak"
            shutil.copy2(filepath, bak_path)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    return data


def stamp_version(data: dict) -> dict:
    """Stamp current cache version on new tune results."""
    data["cache_version"] = CURRENT_CACHE_VERSION
    return data
