#!/usr/bin/env python3
"""
signals.py -- backward-compatible shim.

All logic lives in decision.signal_modules.*; this file re-exports every
public symbol so that ``from decision.signals import X`` keeps working.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Re-export all public symbols from signal_modules ────────────────────
from decision.signal_modules import *  # noqa: F401,F403,E402

# ── Private names (not covered by wildcard) ─────────────────────────────
from decision.signal_modules.config import (  # noqa: E402
    _to_float, _download_prices, _resolve_display_name, _fetch_px_symbol,
    _fetch_with_fallback, _as_series, _ensure_float_series, _align_fx_asof,
    _resolve_symbol_candidates,
)
from decision.signal_modules.asset_processing import _enrich_signal_with_epic8  # noqa: E402
from decision.signal_modules.regime_classification import (  # noqa: E402
    _SIG_H_ANNUAL_CAP, _CI_LOG_FLOOR, _CI_LOG_CAP,
    _DISPLAY_PRICE_CACHE, _compute_sig_h_cap, _smooth_display_price,
    _logistic, _CUSUM_STATE, _get_cusum_state,
)
from decision.signal_modules.momentum_features import _compute_simple_exhaustion  # noqa: E402
from decision.signal_modules.data_fetching import (  # noqa: E402
    _garch11_mle, _fit_student_nu_mle,
)
from decision.signal_modules.kalman_diagnostics import (  # noqa: E402
    _test_innovation_whiteness, _compute_kalman_log_likelihood,
    _compute_kalman_log_likelihood_heteroskedastic, _estimate_regime_drift_priors,
)
from decision.signal_modules.parameter_loading import (  # noqa: E402
    _safe_get_nested, _load_tuned_kalman_params, _select_regime_params,
)
from decision.signal_modules.kalman_filtering import (  # noqa: E402
    _compute_kalman_gain_from_filtered, _apply_gain_monitoring_reset,
    _kalman_filter_drift,
)
from decision.signal_modules.walk_forward import _WFFeatureCache  # noqa: E402
from decision.signal_modules.monte_carlo import _simulate_forward_paths  # noqa: E402
from decision.signal_modules.probability_mapping import (  # noqa: E402
    _load_signals_calibration, _apply_single_p_map, _apply_p_up_calibration,
    _apply_emos_correction, _apply_magnitude_bias_correction,
    _get_calibrated_label_thresholds,
)
from decision.signal_modules.cli import _process_assets_with_retries  # noqa: E402

# ── Module-level constants (canonical definitions) ──────────────────────
PAIR = "PLNJPY=X"
DEFAULT_HORIZONS = [1, 3, 7, 21, 63, 126, 252]
NOTIONAL_PLN = 1_000_000  # for profit column
# Transaction-cost/slippage hurdle (overridable via EDGE_FLOOR env var)
try:
    _edge_env = os.getenv("EDGE_FLOOR", "0.10")
    EDGE_FLOOR = float(_edge_env)
except Exception:
    EDGE_FLOOR = 0.10
EDGE_FLOOR = float(np.clip(EDGE_FLOOR, 0.0, 1.5))  # noqa: F405
# Story 5.1: adaptive edge floor scaling factor
EDGE_FLOOR_Z = 0.65
DEFAULT_CACHE_PATH = os.path.join("src", "data", "currencies", "fx_plnjpy.json")
# Momentum-augmented drift (Story 1.1)
MOM_DRIFT_SCALE = 0.10
MOM_CROSSOVER_HORIZON = 63
MOM_MIN_OBSERVATIONS = 126
# Dual-frequency drift propagation (Story 1.4)
MOMENTUM_HALF_LIFE_DAYS = 42
MOM_SLOW_FRAC = 0.30
SLOW_Q_RATIO = 0.0               # Slow component noise ratio (0 = deterministic decay)
# Story 3.1: Coherent Multi-Horizon MC
COHERENT_MC_ENABLED = True
# Story 3.2: Quantile-Based Confidence Intervals
QUANTILE_CI_MIN_SAMPLES = 100
# Story 3.4: Asset-Class-Aware Per-Step Return Cap
RETURN_CAP_BY_CLASS: Dict[str, float] = {  # noqa: F405
    "equity": 0.30,    # 30% daily max (NYSE circuit breakers)
    "currency": 0.15,  # 15% daily max (CHF flash crash 2015)
    "metal": 0.20,     # 20% daily max
    "crypto": 1.00,    # 100% daily max (genuine crypto volatility)
    "etf": 0.25,       # 25% daily max (ETFs have NAV arbitrage)
}
RETURN_CAP_DEFAULT = 0.30  # conservative default if class unknown

if __name__ == "__main__":
    main()  # noqa: F405
