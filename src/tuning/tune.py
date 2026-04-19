#!/usr/bin/env python3
"""
tune.py -- Backward-compatible shim.

This file re-exports all public symbols from the tune_modules subpackage.
The actual implementation has been decomposed into:

    tune_modules/config.py              Feature flags, conditional imports
    tune_modules/utilities.py           Utility functions, constants
    tune_modules/calibration_pipeline.py  EMOS/DIG calibration
    tune_modules/process_noise.py       Process noise tuning (tune_asset_q)
    tune_modules/volatility_fitting.py  GARCH/GJR-GARCH MLE, OU, cache I/O
    tune_modules/kalman_wrappers.py     Kalman filter compatibility stubs
    tune_modules/pit_diagnostics.py     PIT calibration diagnostics
    tune_modules/model_fitting.py       Core model fitting (fit_all_models_for_regime)
    tune_modules/regime_bma.py          Bayesian Model Averaging engine
    tune_modules/asset_tuning.py        Per-asset BMA tuning pipeline
    tune_modules/cli.py                 CLI argument parsing and main()

See Architecture.md for the full decomposition story map.
"""
from __future__ import annotations

import os
import sys

# Ensure import paths are set up (same as original tune.py)
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Re-export everything from tune_modules for backward compatibility
from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403
from tuning.tune_modules.calibration_pipeline import *  # noqa: F401,F403
from tuning.tune_modules.process_noise import *  # noqa: F401,F403
from tuning.tune_modules.volatility_fitting import *  # noqa: F401,F403
from tuning.tune_modules.kalman_wrappers import *  # noqa: F401,F403
from tuning.tune_modules.pit_diagnostics import *  # noqa: F401,F403
from tuning.tune_modules.model_fitting import *  # noqa: F401,F403
from tuning.tune_modules.regime_bma import *  # noqa: F401,F403
from tuning.tune_modules.asset_tuning import *  # noqa: F401,F403
from tuning.tune_modules.cli import *  # noqa: F401,F403

# Direct ingestion imports (some callers import these transitively via tune)
from ingestion.data_utils import fetch_px, _download_prices, get_default_asset_universe  # noqa: F401
from ingestion.adaptive_quality import adaptive_data_quality  # noqa: F401


if __name__ == "__main__":
    main()
