"""regime.py -- Single source of truth for regime classification.

Story 4.2: Shared Regime Module (DRY).
All regime constants, functions, and CUSUM state live here.
Both tune.py and signals.py import from this module.
"""
import math
import warnings
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

class MarketRegime(IntEnum):
    """Market regime definitions for conditional parameter estimation."""
    LOW_VOL_TREND = 0
    HIGH_VOL_TREND = 1
    LOW_VOL_RANGE = 2
    HIGH_VOL_RANGE = 3
    CRISIS_JUMP = 4


# Plain integer constants (backward-compatible with signals.py)
REGIME_LOW_VOL_TREND = 0
REGIME_HIGH_VOL_TREND = 1
REGIME_LOW_VOL_RANGE = 2
REGIME_HIGH_VOL_RANGE = 3
REGIME_CRISIS_JUMP = 4

REGIME_NAMES = {
    REGIME_LOW_VOL_TREND: "LOW_VOL_TREND",
    REGIME_HIGH_VOL_TREND: "HIGH_VOL_TREND",
    REGIME_LOW_VOL_RANGE: "LOW_VOL_RANGE",
    REGIME_HIGH_VOL_RANGE: "HIGH_VOL_RANGE",
    REGIME_CRISIS_JUMP: "CRISIS_JUMP",
}

# Alias: REGIME_LABELS (tune.py used this name)
REGIME_LABELS = {
    MarketRegime.LOW_VOL_TREND: "LOW_VOL_TREND",
    MarketRegime.HIGH_VOL_TREND: "HIGH_VOL_TREND",
    MarketRegime.LOW_VOL_RANGE: "LOW_VOL_RANGE",
    MarketRegime.HIGH_VOL_RANGE: "HIGH_VOL_RANGE",
    MarketRegime.CRISIS_JUMP: "CRISIS_JUMP",
}

# Minimum samples per regime for stable estimation (used by tune.py)
MIN_REGIME_SAMPLES = 60


# =============================================================================
# Story 4.3: ADAPTIVE REGIME THRESHOLDS
# =============================================================================

# Drift threshold as fraction of median daily volatility
DRIFT_THRESHOLD_SIGMA = 0.05   # 5% of median daily vol

# Default hard thresholds (fallback when vol history unavailable)
DEFAULT_DRIFT_THRESHOLD = 0.0005
DEFAULT_VOL_HIGH_BOUNDARY = 1.3
DEFAULT_VOL_LOW_BOUNDARY = 0.85


class AdaptiveThresholds:
    """Computed regime thresholds adaptive to an asset's own distribution."""
    __slots__ = ("drift_threshold", "vol_high_boundary", "vol_low_boundary",
                 "median_daily_vol", "source")

    def __init__(self, drift_threshold: float, vol_high_boundary: float,
                 vol_low_boundary: float, median_daily_vol: float,
                 source: str = "adaptive"):
        self.drift_threshold = drift_threshold
        self.vol_high_boundary = vol_high_boundary
        self.vol_low_boundary = vol_low_boundary
        self.median_daily_vol = median_daily_vol
        self.source = source

    def to_dict(self) -> Dict:
        return {
            "drift_threshold": self.drift_threshold,
            "vol_high_boundary": self.vol_high_boundary,
            "vol_low_boundary": self.vol_low_boundary,
            "median_daily_vol": self.median_daily_vol,
            "source": self.source,
        }

    @staticmethod
    def default() -> "AdaptiveThresholds":
        return AdaptiveThresholds(
            drift_threshold=DEFAULT_DRIFT_THRESHOLD,
            vol_high_boundary=DEFAULT_VOL_HIGH_BOUNDARY,
            vol_low_boundary=DEFAULT_VOL_LOW_BOUNDARY,
            median_daily_vol=0.01,
            source="default",
        )


def compute_adaptive_thresholds(vol_history: np.ndarray,
                                min_samples: int = 21) -> AdaptiveThresholds:
    """Compute asset-specific regime thresholds from volatility history.

    Drift threshold scales linearly with median daily vol:
        drift_threshold = DRIFT_THRESHOLD_SIGMA * median_daily_vol

    Vol boundaries scale from the asset's own percentile distribution:
        vol_high = P75(vol) / median(vol)
        vol_low  = P25(vol) / median(vol)

    Args:
        vol_history: Array of EWMA daily volatility values
        min_samples: Minimum samples required for adaptive computation

    Returns:
        AdaptiveThresholds with drift_threshold, vol_high_boundary, vol_low_boundary
    """
    vol_history = np.asarray(vol_history, dtype=float)
    valid = vol_history[np.isfinite(vol_history) & (vol_history > 1e-12)]

    if len(valid) < min_samples:
        return AdaptiveThresholds.default()

    median_vol = float(np.median(valid))
    if median_vol < 1e-12:
        return AdaptiveThresholds.default()

    drift_threshold = DRIFT_THRESHOLD_SIGMA * median_vol

    p75 = float(np.percentile(valid, 75))
    p25 = float(np.percentile(valid, 25))
    vol_high_boundary = p75 / median_vol
    vol_low_boundary = p25 / median_vol

    # Clamp to sane ranges
    vol_high_boundary = max(1.05, min(3.0, vol_high_boundary))
    vol_low_boundary = max(0.3, min(0.98, vol_low_boundary))

    return AdaptiveThresholds(
        drift_threshold=drift_threshold,
        vol_high_boundary=vol_high_boundary,
        vol_low_boundary=vol_low_boundary,
        median_daily_vol=median_vol,
        source="adaptive",
    )


# =============================================================================
# CUSUM STATE (Story 1.12) + Auto-Tuning (Story 4.4)
# =============================================================================

CUSUM_THRESHOLD = 3.0       # Sigma-unit trigger threshold (default)
CUSUM_COOLDOWN = 5          # Bars of accelerated smoothing (default)
CUSUM_ALPHA_ACCEL = 0.85    # Accelerated alpha during CUSUM trigger
CUSUM_ALPHA_NORMAL = 0.40   # Normal smoothing alpha

# Story 4.4: Auto-tuning constants
CUSUM_TARGET_ARL = 252      # Target average run length (1 false alarm/year)
CUSUM_DRIFT_ALLOWANCE = 0.5 # CUSUM drift allowance parameter
CUSUM_MIN_THRESHOLD = 1.5   # Minimum CUSUM threshold (fast assets)
CUSUM_MAX_THRESHOLD = 6.0   # Maximum CUSUM threshold (slow assets)
CUSUM_MIN_COOLDOWN = 3      # Minimum cooldown bars
CUSUM_MAX_COOLDOWN = 15     # Maximum cooldown bars
CUSUM_ACF_CUTOFF = 0.1      # Autocorrelation threshold for decorrelation time


class CUSUMParams:
    """Auto-tuned CUSUM parameters for an asset."""
    __slots__ = ("threshold", "cooldown", "alpha_accel", "alpha_normal",
                 "sigma_returns", "decorrelation_bars", "source")

    def __init__(self, threshold: float, cooldown: int,
                 alpha_accel: float = CUSUM_ALPHA_ACCEL,
                 alpha_normal: float = CUSUM_ALPHA_NORMAL,
                 sigma_returns: float = 0.0,
                 decorrelation_bars: int = 5,
                 source: str = "auto"):
        self.threshold = threshold
        self.cooldown = cooldown
        self.alpha_accel = alpha_accel
        self.alpha_normal = alpha_normal
        self.sigma_returns = sigma_returns
        self.decorrelation_bars = decorrelation_bars
        self.source = source

    def to_dict(self) -> Dict:
        return {
            "threshold": self.threshold,
            "cooldown": self.cooldown,
            "alpha_accel": self.alpha_accel,
            "alpha_normal": self.alpha_normal,
            "sigma_returns": self.sigma_returns,
            "decorrelation_bars": self.decorrelation_bars,
            "source": self.source,
        }

    @staticmethod
    def default() -> "CUSUMParams":
        return CUSUMParams(
            threshold=CUSUM_THRESHOLD,
            cooldown=CUSUM_COOLDOWN,
            alpha_accel=CUSUM_ALPHA_ACCEL,
            alpha_normal=CUSUM_ALPHA_NORMAL,
            source="default",
        )


def decorrelation_time(returns: np.ndarray, max_lag: int = 30,
                       cutoff: float = CUSUM_ACF_CUTOFF) -> int:
    """Compute decorrelation time: first lag where |ACF| < cutoff.

    Args:
        returns: Array of returns
        max_lag: Maximum lag to check
        cutoff: ACF threshold (default 0.1)

    Returns:
        Number of bars until autocorrelation drops below cutoff
    """
    returns = np.asarray(returns, dtype=float)
    valid = returns[np.isfinite(returns)]
    n = len(valid)
    if n < 10:
        return 5  # default

    mean_r = np.mean(valid)
    var_r = np.var(valid)
    if var_r < 1e-20:
        return 5

    for lag in range(1, min(max_lag + 1, n // 3)):
        cov = np.mean((valid[lag:] - mean_r) * (valid[:-lag] - mean_r))
        acf = cov / var_r
        if abs(acf) < cutoff:
            return lag

    return max_lag


def compute_arl_threshold(returns: np.ndarray,
                          target_arl: int = CUSUM_TARGET_ARL,
                          drift_allowance: float = CUSUM_DRIFT_ALLOWANCE) -> float:
    """Compute CUSUM threshold for target Average Run Length.

    For a CUSUM with drift allowance k on standardized observations,
    the ARL for in-control is approximately:
        ARL_0 ~ exp(2*h*(h/sigma + 1.166)) for two-sided CUSUM

    We use a simplified Siegmund approximation inverted:
        h ~ sigma * sqrt(2 * ln(ARL_0)) * correction_factor

    Args:
        returns: Array of historical returns
        target_arl: Desired average run length (default 252)
        drift_allowance: CUSUM drift parameter k

    Returns:
        CUSUM threshold value
    """
    returns = np.asarray(returns, dtype=float)
    valid = returns[np.isfinite(returns)]
    if len(valid) < 30:
        return CUSUM_THRESHOLD  # default

    sigma_r = float(np.std(valid))
    if sigma_r < 1e-12:
        return CUSUM_THRESHOLD

    # Siegmund approximation: h ~ sqrt(2 * ln(ARL)) * correction
    # For vol_relative z-scores (not raw returns), we compute a scaled threshold.
    # The CUSUM accumulates z = vol_relative - 1.0, with drift allowance 0.5.
    # We want ARL_0 ~ target_arl under null (no change).
    # Approximate: h ~ sqrt(2 * ln(target_arl)) * sigma_z_factor
    log_arl = math.log(max(target_arl, 10))
    h_raw = math.sqrt(2.0 * log_arl)

    # Scale by volatility of vol-relative changes
    # Higher vol assets have noisier vol_relative -> need higher threshold
    # Lower vol assets have smoother vol_relative -> can use lower threshold
    vol_of_returns = sigma_r
    # Median daily vol for equities ~0.01, crypto ~0.04, currencies ~0.003
    # Normalize: vol_factor = sigma / 0.01 (equity baseline)
    vol_factor = vol_of_returns / 0.01
    # Dampen: use sqrt to avoid extreme scaling
    vol_factor = math.sqrt(max(0.1, min(10.0, vol_factor)))

    threshold = h_raw * vol_factor * 0.9  # 0.9 calibration factor

    return max(CUSUM_MIN_THRESHOLD, min(CUSUM_MAX_THRESHOLD, threshold))


def compute_cusum_params(returns: np.ndarray,
                         target_arl: int = CUSUM_TARGET_ARL) -> CUSUMParams:
    """Auto-tune CUSUM parameters from return distribution.

    Args:
        returns: Array of historical returns
        target_arl: Target average run length

    Returns:
        CUSUMParams with auto-tuned threshold and cooldown
    """
    returns = np.asarray(returns, dtype=float)
    valid = returns[np.isfinite(returns)]
    if len(valid) < 30:
        return CUSUMParams.default()

    threshold = compute_arl_threshold(valid, target_arl)
    decor = decorrelation_time(valid)
    cooldown = max(CUSUM_MIN_COOLDOWN, min(CUSUM_MAX_COOLDOWN, decor))
    sigma_r = float(np.std(valid))

    return CUSUMParams(
        threshold=threshold,
        cooldown=cooldown,
        alpha_accel=CUSUM_ALPHA_ACCEL,
        alpha_normal=CUSUM_ALPHA_NORMAL,
        sigma_returns=sigma_r,
        decorrelation_bars=decor,
        source="auto",
    )


# Module-level state keyed by asset symbol for persistence across calls
_CUSUM_STATE: Dict[str, Dict] = {}


def _get_cusum_state(asset: str) -> Dict:
    """Get or create CUSUM state for an asset."""
    if asset not in _CUSUM_STATE:
        _CUSUM_STATE[asset] = {
            "cusum_pos": 0.0,
            "cusum_neg": 0.0,
            "cooldown_remaining": 0,
            "smoothed_vol_relative": 1.0,
        }
    return _CUSUM_STATE[asset]


# =============================================================================
# DETERMINISTIC REGIME ASSIGNMENT (tune.py: per-bar array)
# =============================================================================

def assign_regime_labels(returns: np.ndarray, vol: np.ndarray,
                         lookback: int = 21,
                         adaptive: Optional["AdaptiveThresholds"] = None,
                         cusum_params: Optional["CUSUMParams"] = None) -> np.ndarray:
    """Assign market regime labels to each observation.

    Story 1.12: CUSUM-accelerated regime transition detection.
    Story 4.3: Asset-adaptive thresholds when `adaptive` is provided.
    Story 4.4: Auto-tuned CUSUM when `cusum_params` is provided.

    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        lookback: Rolling window for feature computation
        adaptive: Optional AdaptiveThresholds for asset-specific scaling
        cusum_params: Optional CUSUMParams for auto-tuned CUSUM sensitivity

    Returns:
        Array of regime labels (0-4) for each observation
    """
    n = len(returns)
    regime_labels = np.zeros(n, dtype=int)

    # Story 4.3: Use adaptive thresholds when provided
    if adaptive is not None:
        drift_threshold = adaptive.drift_threshold
        vol_high = adaptive.vol_high_boundary
        vol_low = adaptive.vol_low_boundary
    else:
        drift_threshold = DEFAULT_DRIFT_THRESHOLD
        vol_high = DEFAULT_VOL_HIGH_BOUNDARY
        vol_low = DEFAULT_VOL_LOW_BOUNDARY

    # Story 4.4: Use auto-tuned CUSUM params when provided
    if cusum_params is not None:
        cusum_threshold = cusum_params.threshold
        cusum_cooldown = cusum_params.cooldown
        alpha_accel = cusum_params.alpha_accel
        alpha_normal = cusum_params.alpha_normal
    else:
        cusum_threshold = 3.0
        cusum_cooldown = 5
        alpha_accel = 0.85
        alpha_normal = 0.40

    cusum_pos = 0.0
    cusum_neg = 0.0
    cooldown_remaining = 0

    vol_series = pd.Series(vol)
    vol_median_expanding = vol_series.expanding(min_periods=lookback).median().values
    vol_median_expanding[:lookback] = (
        np.nanmedian(vol[:lookback]) if lookback <= n else np.nanmedian(vol)
    )

    smoothed_vol_relative = 1.0

    for t in range(n):
        vol_now = vol[t]
        ret_now = returns[t]
        vol_median = (
            vol_median_expanding[t] if vol_median_expanding[t] > 1e-12 else vol_now
        )
        vol_relative = vol_now / vol_median if vol_median > 1e-12 else 1.0

        z_vol = vol_relative - 1.0
        cusum_pos = max(0.0, cusum_pos + z_vol - 0.5)
        cusum_neg = max(0.0, cusum_neg - z_vol - 0.5)

        if (cusum_pos > cusum_threshold or cusum_neg > cusum_threshold) and cooldown_remaining == 0:
            cooldown_remaining = cusum_cooldown
            cusum_pos = 0.0
            cusum_neg = 0.0

        if cooldown_remaining > 0:
            alpha = alpha_accel
            cooldown_remaining -= 1
        else:
            alpha = alpha_normal

        smoothed_vol_relative = alpha * vol_relative + (1 - alpha) * smoothed_vol_relative

        start_idx = max(0, t - lookback + 1)
        drift_abs = abs(np.mean(returns[start_idx : t + 1]))
        tail_indicator = abs(ret_now) / vol_now if vol_now > 1e-12 else 0.0

        if vol_relative > 2.0 or tail_indicator > 4.0:
            regime_labels[t] = MarketRegime.CRISIS_JUMP
        elif smoothed_vol_relative > vol_high:
            if drift_abs > drift_threshold:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE
        elif smoothed_vol_relative < vol_low:
            if drift_abs > drift_threshold:
                regime_labels[t] = MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.LOW_VOL_RANGE
        else:
            if drift_abs > drift_threshold * 1.5:
                regime_labels[t] = (
                    MarketRegime.HIGH_VOL_TREND
                    if smoothed_vol_relative > 1.0
                    else MarketRegime.LOW_VOL_TREND
                )
            else:
                regime_labels[t] = (
                    MarketRegime.HIGH_VOL_RANGE
                    if smoothed_vol_relative > 1.0
                    else MarketRegime.LOW_VOL_RANGE
                )

    return regime_labels


# =============================================================================
# DETERMINISTIC REGIME ASSIGNMENT (signals.py: current-bar only)
# =============================================================================

def assign_current_regime(feats: Dict[str, pd.Series], lookback: int = 21,
                          asset: str = "__default__",
                          adaptive: Optional["AdaptiveThresholds"] = None,
                          cusum_params: Optional["CUSUMParams"] = None) -> int:
    """Assign current regime using SAME logic as assign_regime_labels().

    Story 4.3: Accepts adaptive thresholds for asset-specific scaling.
    Story 4.4: Accepts cusum_params for auto-tuned CUSUM sensitivity.

    Args:
        feats: Feature dictionary with 'ret' and 'vol' series
        lookback: Rolling window for feature computation
        adaptive: Optional AdaptiveThresholds for asset-specific scaling

    Returns:
        Integer regime index (0-4)
    """
    ret_series = feats.get("ret", pd.Series(dtype=float))
    vol_series = feats.get("vol", pd.Series(dtype=float))

    if not isinstance(ret_series, pd.Series) or ret_series.empty:
        return REGIME_LOW_VOL_RANGE
    if not isinstance(vol_series, pd.Series) or vol_series.empty:
        return REGIME_LOW_VOL_RANGE

    vol_now = float(vol_series.iloc[-1]) if len(vol_series) > 0 else 0.0
    ret_now = float(ret_series.iloc[-1]) if len(ret_series) > 0 else 0.0

    if len(ret_series) >= lookback:
        drift_abs = abs(float(ret_series.tail(lookback).mean()))
    else:
        drift_abs = abs(float(ret_series.mean()))

    if len(vol_series) >= lookback:
        vol_median = float(
            vol_series.expanding(min_periods=min(lookback, len(vol_series))).median().iloc[-1]
        )
    else:
        vol_median = float(vol_series.median())

    vol_relative = vol_now / vol_median if vol_median > 1e-12 else 1.0
    tail_indicator = abs(ret_now) / vol_now if vol_now > 1e-12 else 0.0

    # Story 4.3: Use adaptive thresholds when provided
    if adaptive is not None:
        drift_threshold = adaptive.drift_threshold
        vol_high = adaptive.vol_high_boundary
        vol_low = adaptive.vol_low_boundary
    else:
        drift_threshold = DEFAULT_DRIFT_THRESHOLD
        vol_high = DEFAULT_VOL_HIGH_BOUNDARY
        vol_low = DEFAULT_VOL_LOW_BOUNDARY

    # Story 4.4: Use auto-tuned CUSUM params when provided
    if cusum_params is not None:
        _cs_thresh = cusum_params.threshold
        _cs_cool = cusum_params.cooldown
        _cs_alpha_accel = cusum_params.alpha_accel
        _cs_alpha_normal = cusum_params.alpha_normal
    else:
        _cs_thresh = CUSUM_THRESHOLD
        _cs_cool = CUSUM_COOLDOWN
        _cs_alpha_accel = CUSUM_ALPHA_ACCEL
        _cs_alpha_normal = CUSUM_ALPHA_NORMAL

    cs = _get_cusum_state(asset)
    z_vol = vol_relative - 1.0
    cs["cusum_pos"] = max(0.0, cs["cusum_pos"] + z_vol - 0.5)
    cs["cusum_neg"] = max(0.0, cs["cusum_neg"] - z_vol - 0.5)

    if (cs["cusum_pos"] > _cs_thresh or cs["cusum_neg"] > _cs_thresh) and cs["cooldown_remaining"] == 0:
        cs["cooldown_remaining"] = _cs_cool
        cs["cusum_pos"] = 0.0
        cs["cusum_neg"] = 0.0

    if cs["cooldown_remaining"] > 0:
        alpha = _cs_alpha_accel
        cs["cooldown_remaining"] -= 1
    else:
        alpha = _cs_alpha_normal

    cs["smoothed_vol_relative"] = alpha * vol_relative + (1 - alpha) * cs["smoothed_vol_relative"]
    svr = cs["smoothed_vol_relative"]

    if vol_relative > 2.0 or tail_indicator > 4.0:
        return REGIME_CRISIS_JUMP
    if svr > vol_high:
        return REGIME_HIGH_VOL_TREND if drift_abs > drift_threshold else REGIME_HIGH_VOL_RANGE
    if svr < vol_low:
        return REGIME_LOW_VOL_TREND if drift_abs > drift_threshold else REGIME_LOW_VOL_RANGE
    if drift_abs > drift_threshold * 1.5:
        return REGIME_HIGH_VOL_TREND if svr > 1.0 else REGIME_LOW_VOL_TREND
    return REGIME_HIGH_VOL_RANGE if svr > 1.0 else REGIME_LOW_VOL_RANGE


# =============================================================================
# LABEL/INDEX MAPPING
# =============================================================================

def map_regime_label_to_index(regime_label: str,
                              regime_meta: Optional[Dict] = None) -> int:
    """Map a regime label (string) to a regime index (0-4).

    Args:
        regime_label: String regime label
        regime_meta: Optional metadata with vol_regime, trend_z, etc.

    Returns:
        Integer regime index (0-4)
    """
    label_lower = regime_label.lower() if regime_label else ""

    vol_regime = None
    trend_z = None
    if regime_meta is not None:
        vol_regime = regime_meta.get("vol_regime")
        trend_z = regime_meta.get("trend_z")

    if "crisis" in label_lower:
        return REGIME_CRISIS_JUMP
    if "high-vol" in label_lower or "high_vol" in label_lower:
        if vol_regime is not None and vol_regime > 2.0:
            return REGIME_CRISIS_JUMP
        elif trend_z is not None and abs(trend_z) > 0.5:
            return REGIME_HIGH_VOL_TREND
        else:
            return REGIME_HIGH_VOL_RANGE
    if "trending" in label_lower or "trend" in label_lower:
        if vol_regime is not None and vol_regime > 1.3:
            return REGIME_HIGH_VOL_TREND
        return REGIME_LOW_VOL_TREND
    if "calm" in label_lower or "low" in label_lower:
        if trend_z is not None and abs(trend_z) > 0.5:
            return REGIME_LOW_VOL_TREND
        return REGIME_LOW_VOL_RANGE
    if "normal" in label_lower:
        return REGIME_LOW_VOL_RANGE
    if label_lower in ("calm", "0"):
        return REGIME_LOW_VOL_RANGE
    elif label_lower in ("trending", "1"):
        return REGIME_LOW_VOL_TREND
    elif label_lower in ("crisis", "2"):
        return REGIME_CRISIS_JUMP
    return REGIME_LOW_VOL_RANGE


# =============================================================================
# FEATURE EXTRACTION (for softmax-based regime probs)
# =============================================================================

def extract_regime_features(feats: Dict[str, pd.Series]) -> Dict[str, float]:
    """Extract features for regime likelihood computation.

    Features:
    - vol_level: EWMA volatility (normalized)
    - drift_strength: |mu| (absolute drift)
    - drift_persistence: phi from Kalman
    - return_autocorr: autocorrelation of returns
    - tail_indicator: |return| / EWMA_sigma

    Args:
        feats: Feature dictionary from compute_features()

    Returns:
        Dictionary of regime features
    """
    vol_series = feats.get("vol", pd.Series(dtype=float))
    if isinstance(vol_series, pd.Series) and not vol_series.empty:
        vol_now = float(vol_series.iloc[-1])
        vol_median = float(vol_series.median())
        vol_level = vol_now / vol_median if vol_median > 1e-12 else 1.0
    else:
        vol_level = 1.0
        vol_now = 0.0

    mu_series = feats.get("mu_post", feats.get("mu_kf", feats.get("mu", pd.Series(dtype=float))))
    if isinstance(mu_series, pd.Series) and not mu_series.empty:
        drift_strength = abs(float(mu_series.iloc[-1]))
    else:
        drift_strength = 0.0

    km = feats.get("kalman_metadata", {}) or {}
    phi = km.get("phi_used") or km.get("kalman_phi")
    if phi is None or not np.isfinite(phi):
        phi = feats.get("phi_used")
    drift_persistence = float(phi) if phi is not None and np.isfinite(phi) else 0.95

    ret_series = feats.get("ret", pd.Series(dtype=float))
    if isinstance(ret_series, pd.Series) and len(ret_series) >= 21:
        try:
            return_autocorr = float(ret_series.autocorr(lag=1))
            if not np.isfinite(return_autocorr):
                return_autocorr = 0.0
        except Exception:
            return_autocorr = 0.0
    else:
        return_autocorr = 0.0

    if isinstance(ret_series, pd.Series) and not ret_series.empty and vol_now > 1e-12:
        recent_ret = abs(float(ret_series.iloc[-1]))
        tail_indicator = recent_ret / vol_now
    else:
        tail_indicator = 0.0

    return {
        "vol_level": float(np.clip(vol_level, 0.1, 10.0)),
        "drift_strength": float(np.clip(drift_strength, 0.0, 0.01)),
        "drift_persistence": float(np.clip(drift_persistence, -1.0, 1.0)),
        "return_autocorr": float(np.clip(return_autocorr, -1.0, 1.0)),
        "tail_indicator": float(np.clip(tail_indicator, 0.0, 10.0)),
    }


# =============================================================================
# SOFTMAX-BASED REGIME PROBABILITIES (Gaussian scoring)
# =============================================================================

def compute_regime_log_likelihoods(features: Dict[str, float]) -> np.ndarray:
    """Compute log-likelihood scores for each regime given features.

    Uses Gaussian scoring based on regime characteristics.

    Args:
        features: Dictionary from extract_regime_features()

    Returns:
        Array of log-likelihoods for regimes 0-4
    """
    vol = features["vol_level"]
    drift = features["drift_strength"]
    persist = features["drift_persistence"]
    autocorr = features["return_autocorr"]
    tail = features["tail_indicator"]

    log_L = np.zeros(5)

    log_L[0] = (
        -0.5 * ((vol - 0.7) / 0.3) ** 2
        - 0.5 * ((drift - 0.002) / 0.001) ** 2
        - 0.5 * ((persist - 0.98) / 0.02) ** 2
    )
    log_L[1] = (
        -0.5 * ((vol - 1.5) / 0.4) ** 2
        - 0.5 * ((drift - 0.003) / 0.002) ** 2
        - 0.5 * ((persist - 0.95) / 0.03) ** 2
    )
    log_L[2] = (
        -0.5 * ((vol - 0.6) / 0.25) ** 2
        - 0.5 * ((drift - 0.0003) / 0.0005) ** 2
        - 0.5 * ((persist - 0.85) / 0.1) ** 2
    )
    log_L[3] = (
        -0.5 * ((vol - 1.3) / 0.35) ** 2
        - 0.5 * ((drift - 0.0005) / 0.001) ** 2
        - 0.5 * ((persist - 0.80) / 0.15) ** 2
        - 0.5 * ((autocorr - (-0.1)) / 0.2) ** 2
    )
    log_L[4] = (
        -0.5 * ((vol - 2.5) / 0.5) ** 2
        - 0.5 * ((tail - 3.0) / 1.0) ** 2
    )

    return log_L


def compute_regime_probabilities(
    features: Dict[str, float],
    smoothing_alpha: float = 0.3,
    prev_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute regime probabilities via softmax of log-likelihoods.

    Args:
        features: Dictionary from extract_regime_features()
        smoothing_alpha: EMA smoothing factor (0=full smooth, 1=no smooth)
        prev_probs: Previous regime probabilities for smoothing

    Returns:
        Array of probabilities for regimes 0-4, summing to 1
    """
    log_L = compute_regime_log_likelihoods(features)
    log_L_shifted = log_L - np.max(log_L)
    exp_L = np.exp(log_L_shifted)
    probs = exp_L / np.sum(exp_L)

    if prev_probs is not None:
        prev_probs = np.asarray(prev_probs)
        if prev_probs.shape == probs.shape:
            probs = smoothing_alpha * probs + (1.0 - smoothing_alpha) * prev_probs
            probs = probs / np.sum(probs)

    return probs


# =============================================================================
# Story 4.1: LOGISTIC-BOUNDARY REGIME PROBABILITIES
# =============================================================================

REGIME_TRANSITION_WIDTH_VOL = 0.15
REGIME_TRANSITION_WIDTH_DRIFT = 0.0002
REGIME_VOL_HIGH_BOUNDARY = 1.3
REGIME_VOL_LOW_BOUNDARY = 0.85
REGIME_CRISIS_VOL_THRESHOLD = 2.0
REGIME_CRISIS_TAIL_THRESHOLD = 4.0
REGIME_CRISIS_TRANSITION_WIDTH = 0.3


def _logistic(x: float, center: float, width: float) -> float:
    """Standard logistic sigmoid: 1 / (1 + exp(-(x-center)/width))."""
    z = (x - center) / max(width, 1e-12)
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def compute_regime_probabilities_v2(
    vol_relative: float,
    drift_abs: float,
    tail_indicator: float = 0.0,
    drift_threshold: float = 0.0005,
) -> np.ndarray:
    """Probabilistic regime assignment via logistic boundaries.

    Args:
        vol_relative: Current volatility / median volatility
        drift_abs: Absolute drift magnitude |mu|
        tail_indicator: |return| / sigma
        drift_threshold: Drift threshold for trend detection

    Returns:
        Array of probabilities for regimes 0-4, summing to 1.
    """
    p_crisis_vol = _logistic(vol_relative, REGIME_CRISIS_VOL_THRESHOLD,
                             REGIME_CRISIS_TRANSITION_WIDTH)
    p_crisis_tail = _logistic(tail_indicator, REGIME_CRISIS_TAIL_THRESHOLD,
                              REGIME_CRISIS_TRANSITION_WIDTH)
    p_crisis = max(p_crisis_vol, p_crisis_tail)

    p_high_vol = _logistic(vol_relative, REGIME_VOL_HIGH_BOUNDARY,
                           REGIME_TRANSITION_WIDTH_VOL)
    p_low_vol = 1.0 - _logistic(vol_relative, REGIME_VOL_LOW_BOUNDARY,
                                 REGIME_TRANSITION_WIDTH_VOL)
    p_mid_vol = max(0.0, 1.0 - p_high_vol - p_low_vol)

    vol_sum = p_high_vol + p_low_vol + p_mid_vol
    if vol_sum > 1e-12:
        p_high_vol /= vol_sum
        p_low_vol /= vol_sum
        p_mid_vol /= vol_sum

    p_trend = _logistic(drift_abs, drift_threshold, REGIME_TRANSITION_WIDTH_DRIFT)
    p_non_crisis = 1.0 - p_crisis

    probs = np.array([
        p_low_vol * p_trend * p_non_crisis,
        p_high_vol * p_trend * p_non_crisis,
        p_low_vol * (1.0 - p_trend) * p_non_crisis,
        p_high_vol * (1.0 - p_trend) * p_non_crisis,
        p_crisis,
    ], dtype=float)

    mid_trend = p_mid_vol * p_trend * p_non_crisis
    mid_range = p_mid_vol * (1.0 - p_trend) * p_non_crisis
    probs[0] += mid_trend * 0.5
    probs[1] += mid_trend * 0.5
    probs[2] += mid_range * 0.5
    probs[3] += mid_range * 0.5

    total = probs.sum()
    if total > 1e-12:
        probs /= total
    else:
        probs = np.full(5, 0.2)

    return probs


def compute_soft_bma_weights(
    regime_probs: np.ndarray,
    regime_data: Dict,
    global_model_posterior: Dict[str, float],
) -> Dict[str, float]:
    """Mix BMA weights across regimes using soft probabilities.

    effective_weights[m] = sum_r p_r * regime_model_posterior[r][m]

    Args:
        regime_probs: 5-element probability vector
        regime_data: tuned_params['regime'] dict
        global_model_posterior: global fallback posteriors

    Returns:
        Dict mapping model_name -> effective mixed weight
    """
    if regime_data is None:
        regime_data = {}

    all_models: set = set(global_model_posterior.keys())
    for r_key in regime_data:
        r_block = regime_data[r_key]
        if isinstance(r_block, dict) and 'model_posterior' in r_block:
            all_models.update(r_block['model_posterior'].keys())

    if not all_models:
        return {}

    effective: Dict[str, float] = {m: 0.0 for m in all_models}
    for r_idx in range(5):
        p_r = float(regime_probs[r_idx]) if r_idx < len(regime_probs) else 0.0
        if p_r < 1e-10:
            continue

        r_key = str(r_idx)
        r_block = regime_data.get(r_key) or regime_data.get(r_idx)
        if r_block is not None and isinstance(r_block, dict):
            r_posterior = r_block.get('model_posterior', {})
            r_meta = r_block.get('regime_meta', {})
            is_fallback = r_meta.get('fallback', False) or r_meta.get('borrowed_from_global', False)
        else:
            r_posterior = {}
            is_fallback = True

        if not r_posterior or is_fallback:
            r_posterior = global_model_posterior

        for m in all_models:
            effective[m] += p_r * r_posterior.get(m, 0.0)

    total = sum(effective.values())
    if total > 1e-12:
        effective = {m: w / total for m, w in effective.items()}
    return effective


# =============================================================================
# Story 4.5: REGIME TRANSITION SMOOTHING (EMA)
# =============================================================================
REGIME_EMA_ALPHA = 0.3  # 30% new, 70% previous


def smooth_regime_probabilities(
    current_probs: np.ndarray,
    prev_probs: Optional[np.ndarray] = None,
    alpha: float = REGIME_EMA_ALPHA,
) -> np.ndarray:
    """EMA smooth regime probabilities to reduce single-day flips.

    Args:
        current_probs: Array of shape (5,) from compute_regime_probabilities_v2.
        prev_probs: Previous smoothed probs (None on first run).
        alpha: EMA weight on current observation (0 < alpha <= 1).

    Returns:
        Smoothed probabilities summing to 1.0.
    """
    current_probs = np.asarray(current_probs, dtype=float)
    if current_probs.shape != (5,):
        return current_probs

    if prev_probs is None:
        return current_probs

    prev_probs = np.asarray(prev_probs, dtype=float)
    if prev_probs.shape != (5,):
        return current_probs

    smoothed = alpha * current_probs + (1.0 - alpha) * prev_probs

    total = smoothed.sum()
    if total > 1e-12:
        smoothed /= total
    else:
        smoothed = current_probs

    return smoothed
