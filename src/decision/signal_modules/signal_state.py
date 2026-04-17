"""Signal state persistence: load / save previous signal labels.

Extracted from signals.py - Story 8.5.
Contains load_signal_state() and save_signal_state() for two-day
confirmation logic.
"""

import json
import os
from typing import Dict

import sys as _sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

# ─── Story 3.3: Two-Day Confirmation State Persistence ──────────────────
SIGNAL_STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "data", "signal_state")
SIGNAL_STATE_DEFAULT_P = 0.5  # default p_up when no previous state exists


def load_signal_state(symbol: str, state_dir: str = "") -> Dict:
    """Load previous signal state for confirmation logic.

    Returns dict mapping horizon -> {"p_up": float, "label": str, "timestamp": str}.
    If no state file exists, returns empty dict (first run -> default to HOLD).
    """
    _dir = state_dir or SIGNAL_STATE_DIR
    path = os.path.join(_dir, f"{symbol}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_signal_state(symbol: str, state: Dict, state_dir: str = "") -> str:
    """Save signal state for next run's confirmation logic.

    Args:
        symbol: Asset symbol.
        state: Dict mapping horizon (str) -> {"p_up": float, "label": str, "timestamp": str}.
        state_dir: Override state directory (for testing).
    Returns:
        Path to the saved state file.
    """
    _dir = state_dir or SIGNAL_STATE_DIR
    os.makedirs(_dir, exist_ok=True)
    path = os.path.join(_dir, f"{symbol}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return path
