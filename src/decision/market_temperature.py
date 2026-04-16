"""
===============================================================================
MARKET TEMPERATURE MODULE
===============================================================================

Comprehensive US Equity Market Health Assessment (February 2026)

DESIGN PRINCIPLE:
    "The market tells you everything — if you listen to the right instruments."

This module computes a comprehensive market temperature from multiple stock
universe segments to give a complete picture of market health and risk:

UNIVERSE SEGMENTS:
    1. Top 100 US Stocks (40%): Mega-cap bellwethers (Apple, Microsoft, etc.)
       - Represents institutional flows and macro sentiment
       
    2. S&P 500 Proxy (30%): Broad market health via SPY components
       - Core market breadth indicator
       
    3. Russell 2000 Small-Caps (20%): Risk appetite indicator
       - Small caps lead in risk-on, lag in risk-off
       
    4. Growth vs Value Spread (10%): Rotation signals
       - QQQ/IWD spread reveals regime shifts

COMPUTED METRICS:
    - Market Temperature: Aggregate stress score ∈ [0, 2]
    - Crash Risk: Probability of significant drawdown (vol inversion, etc.)
    - Momentum: Multi-timeframe momentum across segments
    - Breadth: Advance/decline ratios and participation
    - Volatility Regime: Current vol vs historical percentiles
    - Correlation Stress: Rising correlations = systemic risk

REFERENCES:
    Expert Panel Design (February 2026)
    Mirrors metals_risk_temperature.py and risk_temperature.py patterns

===============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# COPULA-BASED CORRELATION STRESS (February 2026)
# =============================================================================
# Import copula correlation module for tail dependency analysis
# Falls back to Pearson correlation if copula module unavailable
# =============================================================================
try:
    from calibration.copula_correlation import (
        compute_copula_correlation_stress,
        compute_smooth_scale_factor,
        CopulaCorrelationStress,
        COPULA_CORRELATION_AVAILABLE,
    )
    USE_COPULA_CORRELATION = True
except ImportError:
    USE_COPULA_CORRELATION = False
    COPULA_CORRELATION_AVAILABLE = False

# =============================================================================
# PYTORCH FORECASTING (February 2026)
# =============================================================================
# N-BEATS models for currency/equity forecasting
# Falls back to classical + Prophet if unavailable
# =============================================================================
try:
    from pytorch_forecasting import NBeatsNet
    PYTORCH_FORECASTING_AVAILABLE = True
    USE_PYTORCH_FORECASTING = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False
    USE_PYTORCH_FORECASTING = False

# =============================================================================
# GAS-Q SCORE-DRIVEN PROCESS NOISE (Story 1.3)
# =============================================================================
# Import GAS-Q filters for adaptive Kalman forecasting.
# When tuned params contain gas_q_augmented=True, _kalman_forecast() runs
# a real state-space filter with time-varying process noise instead of EMA.
# =============================================================================
try:
    from models.gas_q import (
        GASQConfig,
        GASQResult,
        gas_q_filter_gaussian,
        gas_q_filter_student_t,
    )
    _GAS_Q_AVAILABLE = True
except ImportError:
    _GAS_Q_AVAILABLE = False

# =============================================================================
# =============================================================================
# All forecast functions MUST use these horizons to prevent index misalignment.
# Callers should use get_forecast_by_horizon() instead of positional indices.
# =============================================================================
STANDARD_HORIZONS = [1, 3, 7, 30, 90, 180, 365]
HORIZON_INDEX = {h: i for i, h in enumerate(STANDARD_HORIZONS)}

# Crypto tickers for special handling (high volatility, different dynamics)
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD"}


def _is_crypto_asset(asset_name: str) -> bool:
    """Check if asset is a cryptocurrency based on name/ticker."""
    if not asset_name:
        return False
    name_upper = asset_name.upper().replace("=", "").replace(" ", "")
    return any(crypto in name_upper for crypto in ["BTC", "ETH", "BITCOIN", "ETHEREUM"])


def get_forecast_by_horizon(result: tuple, horizon: int) -> float:
    """
    Get forecast value for a specific horizon from ensemble_forecast result.
    
    Args:
        result: Tuple from ensemble_forecast (7 floats + confidence string)
        horizon: One of STANDARD_HORIZONS [1, 3, 7, 30, 90, 180, 365]
        
    Returns:
        Forecast percentage for that horizon, or 0.0 if not found
    """
    idx = HORIZON_INDEX.get(horizon)
    if idx is not None and idx < len(result) - 1:
        return float(result[idx])
    return 0.0


def get_forecast_confidence(result: tuple) -> str:
    """Get confidence string from ensemble_forecast result."""
    if result and len(result) > 0:
        return str(result[-1]) if isinstance(result[-1], str) else "Low"
    return "Low"


# =============================================================================
# ELITE FORECASTING ENGINE (Professor-Grade Multi-Model Ensemble)
# =============================================================================
# Architecture: Multi-model Bayesian ensemble with regime-aware weighting
# Models: Kalman Filter, GARCH(1,1), Ornstein-Uhlenbeck, Momentum, Classical
# Horizon-adaptive: Different model weights per forecast horizon
# Horizon-adaptive weighting for optimal blending
# =============================================================================

# =============================================================================
# MULTI-SCALE DRIFT ESTIMATION (Story 1.4)
# =============================================================================
# Three parallel filters with different q values capture dynamics at different
# time scales. Short horizons use a fast filter (high q, responsive to recent
# changes), long horizons use a slow filter (low q, smooth drift).
#
# q_fast = Q_FAST_MULT * q_tuned  (high process noise -> fast adaptation)
# q_med  = q_tuned                (standard tuned process noise)
# q_slow = Q_SLOW_MULT * q_tuned  (low process noise -> smooth drift)
# =============================================================================
Q_FAST_MULT = 10.0    # Fast filter: 10x tuned q
Q_SLOW_MULT = 0.1     # Slow filter: 0.1x tuned q

HORIZON_FILTER_MAP = {
    1: 'fast',
    3: 'fast',
    7: 'medium',
    30: 'medium',
    90: 'slow',
    180: 'slow',
    365: 'slow',
}

# =============================================================================
# ENSEMBLE DE-CORRELATION (Story 1.6)
# =============================================================================
# Sign-agreement-weighted averaging: when models agree on direction, amplify
# the consensus; when they disagree, dampened with higher uncertainty.
# =============================================================================
CONTRAST_BOOST = 0.6          # Max weight shift for consensus (0-1)
DISPERSION_CONTESTED = 1.5    # Dispersion threshold for CONTESTED label
DISPERSION_INTERVAL_WIDEN = 0.5  # Interval widening factor when contested

# =============================================================================
# Story 1.14: Hard Cap Relaxation with Confidence-Gated Bounds
# Cap multiplier = 1.0 + CAP_CONFIDENCE_SCALE * agreement_contrast
# High confidence (strong model agreement) -> caps widen up to 2x
# Vol-bound remains as independent safety constraint
# =============================================================================
CAP_CONFIDENCE_SCALE = 0.5   # How much confidence widens caps
CAP_MAX_MULTIPLIER = 2.0     # Maximum cap expansion factor

# =============================================================================
# Story 1.11: Volatility-Adjusted Forecast Scaling
# Signal quality factor: quality = clip(1/vol_ratio, 0.3, 2.0)
# vol_ratio = current_ewma_vol / rolling_252d_median_vol
# IMPORTANT: quality scales CONFIDENCE, not forecast magnitude
# =============================================================================
VOL_QUALITY_MIN = 0.3   # Min quality factor (extreme vol)
VOL_QUALITY_MAX = 2.0   # Max quality factor (very calm)
VOL_RATIO_EWMA_SPAN = 20  # EWMA span for current vol
VOL_RATIO_MEDIAN_WINDOW = 252  # Rolling window for median vol

# =============================================================================
# Story 2.2: Ornstein-Uhlenbeck Calibration Constants
# AR(1) regression on demeaned log prices -> kappa estimation
# EWMA theta with span = half_life
# =============================================================================
OU_HALF_LIFE_MIN = 5       # Minimum half-life in days
OU_HALF_LIFE_MAX = 252     # Maximum half-life in days
OU_KAPPA_DEFAULT = 0.025   # Default kappa when estimation fails
OU_EWMA_SPAN_DEFAULT = 60  # Default EWMA span for theta


def estimate_ou_kappa(log_prices: np.ndarray) -> float:
    """
    Estimate OU mean-reversion speed via AR(1) regression on demeaned log prices.
    
    Model: dX_t = -kappa * (X_t - mu) * dt + sigma * dW
    Discretized: X_{t+1} - X_t = -kappa * (X_t - mu) + eps
    => AR(1): X_{t+1} = (1 - kappa) * X_t + kappa * mu + eps
    => OLS slope phi = 1 - kappa => kappa = 1 - phi
    
    Returns kappa clipped to [ln(2)/252, ln(2)/5] corresponding to
    half-lives between 5 and 252 days.
    """
    if len(log_prices) < 30:
        return OU_KAPPA_DEFAULT
    
    # Demean
    mu = np.mean(log_prices)
    x = log_prices - mu
    
    # AR(1) regression: x_{t+1} = phi * x_t + eps
    x_t = x[:-1]
    x_tp1 = x[1:]
    
    # OLS: phi = sum(x_t * x_tp1) / sum(x_t^2)
    denom = np.sum(x_t ** 2)
    if denom < 1e-20:
        return OU_KAPPA_DEFAULT
    
    phi = np.sum(x_t * x_tp1) / denom
    kappa = 1.0 - phi
    
    # Clip to valid range
    kappa_min = np.log(2) / OU_HALF_LIFE_MAX  # ~0.00275
    kappa_max = np.log(2) / OU_HALF_LIFE_MIN   # ~0.1386
    return float(np.clip(kappa, kappa_min, kappa_max))


def estimate_ou_theta(prices: np.ndarray, kappa: float) -> float:
    """
    Estimate OU equilibrium level as EWMA with span = half_life.
    
    Half-life = ln(2) / kappa. Use EWMA span = half_life for consistency.
    """
    half_life = np.log(2) / max(kappa, 1e-6)
    span = max(10, min(int(half_life), 252))
    
    # Manual EWMA computation (no pandas dependency)
    alpha = 2.0 / (span + 1)
    ewma = float(prices[0])
    for p in prices[1:]:
        ewma = alpha * float(p) + (1 - alpha) * ewma
    return ewma


def estimate_ou_params(prices: np.ndarray) -> dict:
    """
    Full OU parameter estimation: kappa, theta, sigma_ou, half_life_days.
    
    Used both during tuning and as fallback in _load_tuned_params.
    """
    log_p = np.log(np.maximum(prices, 1e-10))
    kappa = estimate_ou_kappa(log_p)
    theta = estimate_ou_theta(prices, kappa)
    half_life = np.log(2) / max(kappa, 1e-6)
    
    # Estimate sigma_ou from residuals of AR(1)
    x = log_p - np.mean(log_p)
    if len(x) > 1:
        residuals = x[1:] - (1 - kappa) * x[:-1]
        sigma_ou = float(np.std(residuals))
    else:
        sigma_ou = 0.01
    
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_ou": float(sigma_ou),
        "half_life_days": float(half_life),
    }


def compute_signal_quality(vol_ratio: float) -> float:
    """
    Compute signal quality factor from vol ratio.
    
    quality = clip(1/vol_ratio, VOL_QUALITY_MIN, VOL_QUALITY_MAX)
    
    High vol -> low quality (discount the forecast)
    Low vol  -> high quality (trust the forecast)
    """
    safe_ratio = max(vol_ratio, 0.01)
    return float(np.clip(1.0 / safe_ratio, VOL_QUALITY_MIN, VOL_QUALITY_MAX))


def compute_vol_regime_label(vol_ratio: float) -> str:
    """Classify vol regime from vol ratio."""
    if vol_ratio < 0.7:
        return "CALM"
    elif vol_ratio < 1.3:
        return "NORMAL"
    elif vol_ratio < 2.0:
        return "ELEVATED"
    else:
        return "EXTREME"


def _run_kalman_filter_pass(
    returns: np.ndarray,
    vol_arr: np.ndarray,
    phi: float,
    q: float,
    c: float,
    nu: Optional[float] = None,
) -> tuple:
    """
    Run a single Kalman filter pass (Gaussian or Student-t).

    Returns (mu_filtered, P_filtered) arrays.
    """
    T = len(returns)
    mu_filtered = np.zeros(T)
    P_filtered = np.zeros(T)
    mu_t = 0.0
    P_t = max(q * 100.0, 1e-4)  # Initial P relative to q
    phi_sq = phi * phi
    is_student_t = nu is not None and nu > 3.0

    for t in range(T):
        mu_pred = phi * mu_t
        P_pred = phi_sq * P_t + q
        R_t = max(c * vol_arr[t] * vol_arr[t], 1e-12)
        S_t = max(P_pred + R_t, 1e-12)
        innov = returns[t] - mu_pred
        K_t = P_pred / S_t

        if is_student_t:
            K_t *= nu / (nu + 3.0)
            z_sq = (innov * innov) / S_t
            w_t = (nu + 1.0) / (nu + z_sq)
            mu_t = mu_pred + K_t * w_t * innov
            P_t = max((1.0 - w_t * K_t) * P_pred, 1e-12)
        else:
            mu_t = mu_pred + K_t * innov
            P_t = max((1.0 - K_t) * P_pred, 1e-12)

        mu_filtered[t] = mu_t
        P_filtered[t] = P_t

    return mu_filtered, P_filtered

def _kalman_forecast(returns: np.ndarray, horizons: list, asset_type: str = "equity",
                     tuned_params: Optional[dict] = None) -> list:
    """
    Kalman Filter forecast with adaptive state estimation.

    When tuned_params with GAS-Q are provided, runs a real Kalman filter with
    time-varying process noise. Otherwise, falls back to multi-scale EMA.

    Args:
        returns: Log returns array
        horizons: List of forecast horizons in days
        asset_type: "equity", "currency", "metal", "crypto"
        tuned_params: Optional per-asset tuned parameters dict
    """
    try:
        if len(returns) < 20:
            return [0.0] * len(horizons)

        # =====================================================================
        # MULTI-SCALE KALMAN PATH (Story 1.4): Three parallel filters
        # =====================================================================
        # When tuned_params are available, run fast/medium/slow filters.
        # GAS-Q used for medium scale when gas_q_augmented=True.
        # =====================================================================
        if tuned_params is not None:
            q_tuned = tuned_params.get('q')
            c_tuned = tuned_params.get('c')
            phi_tuned = tuned_params.get('phi', 1.0)
            nu_tuned = tuned_params.get('nu')

            if q_tuned and c_tuned:
                try:
                    q_f = float(q_tuned)
                    c_f = float(c_tuned)
                    phi_f = float(phi_tuned) if phi_tuned else 1.0

                    vol_20 = max(np.std(returns[-20:]), 1e-6)
                    vol_arr = np.full(len(returns), vol_20)

                    # --- Run three filter scales ---
                    scales = {}

                    # Fast filter: high q -> responsive to recent changes
                    mu_fast, P_fast = _run_kalman_filter_pass(
                        returns, vol_arr, phi_f,
                        q_f * Q_FAST_MULT, c_f, nu=nu_tuned,
                    )
                    scales['fast'] = (float(mu_fast[-1]), float(P_fast[-1]))

                    # Medium filter: tuned q (or GAS-Q)
                    mu_med = None
                    gas_q_used = False
                    if (
                        _GAS_Q_AVAILABLE
                        and tuned_params.get('gas_q_augmented', False)
                    ):
                        gas_q_params = tuned_params.get('gas_q_params', {})
                        if gas_q_params:
                            try:
                                config = GASQConfig(
                                    omega=gas_q_params.get('omega', q_f * 0.1),
                                    alpha=gas_q_params.get('alpha', 0.05),
                                    beta=gas_q_params.get('beta', 0.90),
                                )
                                if nu_tuned is not None:
                                    gas_result = gas_q_filter_student_t(
                                        returns, vol_arr,
                                        c=c_f, phi=phi_f,
                                        nu=float(nu_tuned), config=config,
                                    )
                                else:
                                    gas_result = gas_q_filter_gaussian(
                                        returns, vol_arr,
                                        c=c_f, phi=phi_f, config=config,
                                    )
                                if gas_result is not None and hasattr(gas_result, 'mu_filtered'):
                                    mu_med = gas_result.mu_filtered
                                    P_med = gas_result.P_filtered
                                    gas_q_used = True

                                    q_path = getattr(gas_result, 'q_path', None)
                                    if q_path is not None and len(q_path) > 0:
                                        logger.debug(
                                            "GAS-Q q_t diagnostics: mean=%.2e std=%.2e "
                                            "min=%.2e max=%.2e",
                                            np.mean(q_path), np.std(q_path),
                                            np.min(q_path), np.max(q_path),
                                        )
                            except Exception:
                                pass

                    if mu_med is None:
                        mu_med, P_med = _run_kalman_filter_pass(
                            returns, vol_arr, phi_f, q_f, c_f, nu=nu_tuned,
                        )
                    scales['medium'] = (float(mu_med[-1]), float(P_med[-1]))

                    # Slow filter: low q -> smooth drift
                    mu_slow, P_slow = _run_kalman_filter_pass(
                        returns, vol_arr, phi_f,
                        q_f * Q_SLOW_MULT, c_f, nu=nu_tuned,
                    )
                    scales['slow'] = (float(mu_slow[-1]), float(P_slow[-1]))

                    # --- Build forecasts using horizon-matched filter ---
                    forecasts = []
                    for h in horizons:
                        scale_key = HORIZON_FILTER_MAP.get(h, 'medium')
                        mu_last, P_last = scales[scale_key]

                        mu_h = (phi_f ** h) * mu_last
                        snr = abs(mu_last) / max(np.sqrt(P_last), 1e-8)
                        persistence = min(1.0, snr / 3.0)
                        if asset_type == "currency":
                            persistence *= 0.75
                        elif asset_type == "crypto":
                            persistence = min(persistence * 1.2, 0.95)

                        fc_pct = mu_h * h * persistence * 100.0
                        forecasts.append(float(fc_pct))

                    return forecasts

                except Exception as e:
                    logger.debug("Multi-scale Kalman failed, falling back to EMA: %s", e)

        # =====================================================================
        # EMA FALLBACK: Multi-scale exponential smoothing (original path)
        # =====================================================================
        ema_5 = returns[-5:].mean() if len(returns) >= 5 else returns.mean()
        ema_20 = returns[-20:].mean() if len(returns) >= 20 else returns.mean()
        ema_60 = returns[-60:].mean() if len(returns) >= 60 else returns.mean()

        vol_20 = np.std(returns[-20:])

        signal_strength = abs(ema_5) / (vol_20 + 1e-8)
        signal_strength = np.clip(signal_strength, 0, 3)

        forecasts = []
        for h in horizons:
            if h <= 3:
                drift = ema_5 * 0.70 + ema_20 * 0.30
                persistence = 0.95
            elif h <= 7:
                drift = ema_5 * 0.50 + ema_20 * 0.40 + ema_60 * 0.10
                persistence = 0.85
            elif h <= 30:
                drift = ema_5 * 0.25 + ema_20 * 0.50 + ema_60 * 0.25
                persistence = 0.70
            elif h <= 90:
                drift = ema_5 * 0.10 + ema_20 * 0.35 + ema_60 * 0.55
                persistence = 0.50
            else:
                drift = ema_20 * 0.20 + ema_60 * 0.80
                persistence = 0.30

            if asset_type == "currency":
                persistence *= 0.75
            elif asset_type == "crypto":
                persistence = min(persistence * 1.2, 0.95)

            amplification = 1.0 + 0.3 * signal_strength
            fc_log = drift * h * persistence * amplification
            pct = fc_log * 100
            forecasts.append(float(pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def _garch_forecast(returns: np.ndarray, horizons: list, asset_type: str = "equity",
                    tuned_params: Optional[dict] = None) -> list:
    """
    GARCH(1,1) volatility-adjusted forecast.
    
    Story 2.1: loads per-asset GARCH parameters from tuned cache when available.
    Falls back to generic defaults when not.
    
    Estimates time-varying volatility and adjusts drift forecast based on
    current volatility regime relative to long-term average.
    """
    try:
        if len(returns) < 30:
            return [0.0] * len(horizons)
        
        # Story 2.1: Load GARCH params from tuned cache or use defaults
        garch_params = None
        if tuned_params and "garch_params" in tuned_params:
            garch_params = tuned_params["garch_params"]
        
        if garch_params:
            omega = float(garch_params.get("omega", 0.00001))
            alpha_g = float(garch_params.get("alpha", 0.08))
            beta_g = float(garch_params.get("beta", 0.88))
            # Stationarity check: alpha + beta < 1.0
            if alpha_g + beta_g >= 1.0:
                alpha_g = 0.08
                beta_g = 0.88
        else:
            omega = 0.00001
            alpha_g = 0.08
            beta_g = 0.88

        # Estimate conditional variance
        var_t = np.var(returns)
        for r in returns[-min(60, len(returns)):]:
            var_t = omega + alpha_g * r**2 + beta_g * var_t
        
        # Long-run variance
        denom = 1 - alpha_g - beta_g
        long_var = omega / denom if denom > 0.01 else np.var(returns)
        
        # Volatility ratio (current vs long-run)
        vol_ratio = np.sqrt(var_t) / np.sqrt(long_var) if long_var > 1e-10 else 1.0
        vol_ratio = np.clip(vol_ratio, 0.3, 3.0)
        
        # Multi-scale drift estimates
        drift_5d = np.mean(returns[-5:])
        drift_20d = np.mean(returns[-20:])
        drift_60d = np.mean(returns[-min(60, len(returns)):])
        
        forecasts = []
        for h in horizons:
            # Horizon-weighted drift
            if h <= 7:
                drift = drift_5d * 0.6 + drift_20d * 0.4
            elif h <= 30:
                drift = drift_5d * 0.25 + drift_20d * 0.50 + drift_60d * 0.25
            else:
                drift = drift_5d * 0.10 + drift_20d * 0.30 + drift_60d * 0.60
            
            # Volatility adjustment: high vol reduces confidence, not magnitude
            vol_adj = 1.0 / np.sqrt(vol_ratio) if vol_ratio > 1.0 else 1.0
            vol_adj = np.clip(vol_adj, 0.5, 1.5)
            
            # Decay based on asset type
            if asset_type == "currency":
                decay = np.exp(-h / 60.0)
            else:
                decay = np.exp(-h / 180.0)
            
            # Project forward
            fc = drift * h * vol_adj * decay
            pct = fc * 100
            
            forecasts.append(float(pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def _ou_forecast(prices: pd.Series, horizons: list, asset_type: str = "equity",
                 tuned_params: Optional[dict] = None) -> list:
    """
    Ornstein-Uhlenbeck mean reversion forecast.
    
    Story 2.2: Uses calibrated OU params (kappa, theta, sigma_ou) when available.
    Falls back to MA-based mean reversion estimation otherwise.
    
    With calibrated params:
      fc = (theta - price) * (1 - exp(-kappa * H/252)) / price * 100
    """
    try:
        if len(prices) < 60:
            return [0.0] * len(horizons)
        
        current = float(prices.iloc[-1])
        
        # Story 2.2: Use calibrated OU params if available
        ou_params = None
        if tuned_params and "ou_params" in tuned_params:
            ou_params = tuned_params["ou_params"]
        
        if ou_params:
            kappa = float(ou_params.get("kappa", 0.025))
            theta_level = float(ou_params.get("theta", current))
            half_life = float(ou_params.get("half_life_days", 60))
            
            # Validate half-life bounds: 5 < half_life < 252
            if half_life < 5 or half_life > 252:
                kappa = 0.025
                theta_level = current
            
            forecasts = []
            for h in horizons:
                # OU projection: E[X_t+h] = theta + (X_t - theta) * exp(-kappa * h)
                # Return forecast = (E[X_t+h] - X_t) / X_t * 100
                expected = theta_level + (current - theta_level) * np.exp(-kappa * h)
                fc_pct = (expected - current) / current * 100 if current > 0 else 0.0
                forecasts.append(float(fc_pct))
            return forecasts
        
        current = float(prices.iloc[-1])
        
        # Multiple mean reversion targets
        ma_50 = float(prices.iloc[-min(50, len(prices)):].mean())
        ma_100 = float(prices.iloc[-min(100, len(prices)):].mean())
        ma_200 = float(prices.iloc[-min(200, len(prices)):].mean())
        
        # Deviations from each MA
        dev_50 = (current - ma_50) / ma_50 if ma_50 > 0 else 0.0
        dev_100 = (current - ma_100) / ma_100 if ma_100 > 0 else 0.0
        dev_200 = (current - ma_200) / ma_200 if ma_200 > 0 else 0.0
        
        # Estimate mean reversion speed from autocorrelation
        log_ret = np.log(prices / prices.shift(1)).dropna().values
        if len(log_ret) >= 20:
            try:
                autocorr = np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
            except:
                autocorr = 0.0
            autocorr = np.clip(autocorr, -0.5, 0.5)
            theta = 0.025 * (1 - autocorr)
        else:
            theta = 0.025
        
        # Currency-specific: faster mean reversion
        if asset_type == "currency":
            theta *= 1.8
        
        forecasts = []
        for h in horizons:
            # Horizon-dependent MA weighting
            if h <= 7:
                target_dev = dev_50 * 0.7 + dev_100 * 0.3
            elif h <= 30:
                target_dev = dev_50 * 0.3 + dev_100 * 0.4 + dev_200 * 0.3
            elif h <= 90:
                target_dev = dev_50 * 0.1 + dev_100 * 0.4 + dev_200 * 0.5
            else:
                target_dev = dev_100 * 0.2 + dev_200 * 0.8

            # OU: expected deviation shrinks exponentially toward zero
            expected_dev = target_dev * np.exp(-theta * h)
            
            # Forecast = price change from current deviation to expected
            # If above MA (positive dev), expect negative return
            fc_pct = -(target_dev - expected_dev) * 100

            forecasts.append(float(fc_pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def compute_momentum_timeframe_weights(returns: np.ndarray, lookback: int = 60,
                                       temperature: float = 0.1) -> dict:
    """
    Story 2.3: Compute regime-adaptive momentum timeframe weights.
    
    Tracks rolling hit rate per timeframe, converts to softmax weights with
    entropy-based diversity constraint.
    
    Timeframes: {5d, 10d, 21d, 63d, 126d, 252d}
    
    Returns dict {timeframe_days: weight} summing to 1.0.
    """
    timeframes = [5, 10, 21, 63, 126, 252]
    n = len(returns)
    
    if n < lookback + max(timeframes):
        # Not enough data -> uniform weights
        w = 1.0 / len(timeframes)
        return {tf: w for tf in timeframes}
    
    # Compute rolling hit rate per timeframe
    hit_rates = {}
    for tf in timeframes:
        if n < lookback + tf:
            hit_rates[tf] = 0.50  # default to coin flip
            continue
        
        hits = 0
        total = 0
        for i in range(lookback):
            idx = n - lookback + i
            if idx < tf:
                continue
            # Momentum signal: cumulative return over tf days
            mom = np.sum(returns[idx - tf:idx])
            # Realized: next-day return (or as far as available)
            if idx < n:
                realized = returns[idx]
                # Hit if signs agree
                if mom * realized > 0:
                    hits += 1
                total += 1
        
        hit_rates[tf] = hits / total if total > 0 else 0.50
    
    # Excess accuracy above coin flip, floored at 0.01
    raw_weights = {}
    for tf in timeframes:
        raw_weights[tf] = max(hit_rates[tf] - 0.50, 0.01)
    
    # Softmax with temperature
    values = np.array([raw_weights[tf] for tf in timeframes])
    max_val = np.max(values)
    exp_vals = np.exp((values - max_val) / temperature)
    
    # Entropy check: H(weights) >= 0.5 * H_uniform
    h_uniform = np.log(len(timeframes))
    temp_adjusted = temperature
    for _ in range(10):  # max 10 iterations to meet diversity constraint
        exp_vals = np.exp((values - max_val) / temp_adjusted)
        weights = exp_vals / np.sum(exp_vals)
        
        # Compute entropy
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        if entropy >= 0.5 * h_uniform:
            break
        temp_adjusted *= 1.5  # Increase temperature to spread weights
    else:
        weights = exp_vals / np.sum(exp_vals)
    
    return {tf: float(w) for tf, w in zip(timeframes, weights)}


def _momentum_forecast(returns: np.ndarray, horizons: list, asset_type: str = "equity") -> list:
    """
    Multi-timeframe momentum forecast.
    
    Story 2.3: Dynamically selects momentum timeframes based on which have been
    most predictive. Uses softmax weights with entropy-based diversity constraint.
    
    Key insight: Strong recent momentum (high cumulative return / vol) persists.
    """
    try:
        n_ret = len(returns)
        if n_ret < 20:
            return [0.0] * len(horizons)
        
        # Story 2.3: Compute regime-adaptive timeframe weights
        tf_weights = compute_momentum_timeframe_weights(returns)
        
        # Cumulative returns over each timeframe
        tf_returns = {}
        for tf in [5, 10, 21, 63, 126, 252]:
            if n_ret >= tf:
                tf_returns[tf] = np.sum(returns[-tf:])
            else:
                tf_returns[tf] = np.sum(returns) * (tf / n_ret)
        
        # Weighted drift using adaptive timeframe weights
        weighted_drift_per_day = sum(
            tf_weights.get(tf, 0.0) * tf_returns.get(tf, 0.0) / tf
            for tf in [5, 10, 21, 63, 126, 252]
        )
        
        # Volatility for Sharpe-style signal strength
        vol_20d = np.std(returns[-20:]) * np.sqrt(252)
        
        # Momentum strength: how strong is the trend relative to noise?
        daily_vol = np.std(returns[-20:])
        sharpe_20d = (weighted_drift_per_day * 252) / (vol_20d + 0.01)
        sharpe_20d = np.clip(sharpe_20d, -3, 3)
        
        # Persistence factor based on trend strength
        trend_strength = min(abs(sharpe_20d) / 1.5, 1.0)  # 0 to 1
        
        # Short/medium/long drift for horizon-dependent blending
        drift_short = sum(tf_weights.get(tf, 0) * tf_returns.get(tf, 0) / tf
                          for tf in [5, 10]) / max(sum(tf_weights.get(tf, 0) for tf in [5, 10]), 1e-6)
        drift_mid = sum(tf_weights.get(tf, 0) * tf_returns.get(tf, 0) / tf
                        for tf in [21, 63]) / max(sum(tf_weights.get(tf, 0) for tf in [21, 63]), 1e-6)
        drift_long = sum(tf_weights.get(tf, 0) * tf_returns.get(tf, 0) / tf
                         for tf in [126, 252]) / max(sum(tf_weights.get(tf, 0) for tf in [126, 252]), 1e-6)
        
        forecasts = []
        for h in horizons:
            # Horizon-dependent drift: short horizons favor recent momentum
            if h <= 3:
                drift = drift_short * 0.70 + drift_mid * 0.30
                base_persistence = 0.90
            elif h <= 7:
                drift = drift_short * 0.50 + drift_mid * 0.40 + drift_long * 0.10
                base_persistence = 0.80
            elif h <= 30:
                drift = drift_short * 0.25 + drift_mid * 0.50 + drift_long * 0.25
                base_persistence = 0.65
            elif h <= 90:
                drift = drift_short * 0.10 + drift_mid * 0.40 + drift_long * 0.50
                base_persistence = 0.45
            else:
                drift = drift_short * 0.05 + drift_mid * 0.25 + drift_long * 0.70
                base_persistence = 0.30
            
            # Asset-specific persistence adjustment
            if asset_type == "currency":
                persistence = base_persistence * 0.70  # FX momentum fades faster
            elif asset_type == "crypto":
                persistence = min(base_persistence * 1.30, 0.95)  # Crypto trends persist
            else:
                persistence = base_persistence
            
            # Boost persistence for strong trends
            persistence = persistence + (1 - persistence) * trend_strength * 0.3
            
            # Project forward
            fc = drift * h * persistence
            pct = fc * 100
            
            # Volatility-scaled sanity bound (3 sigma)
            vol_bound = daily_vol * np.sqrt(h) * 3.0 * 100
            pct = np.clip(pct, -vol_bound, vol_bound)
            
            forecasts.append(float(pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


# =============================================================================
# Story 2.7: Cross-Asset Signal Propagation
# Extracts VIX level/change, DXY level/change, SPY returns as macro signals.
# Rolling beta per asset adjusts drift; capped at 30% of standalone forecast.
# =============================================================================
CROSS_ASSET_MAX_ADJ = 0.30            # Max 30% adjustment of standalone forecast
CROSS_ASSET_BETA_WINDOW = 120         # Rolling regression window
CROSS_ASSET_SIGNALS = ["VIX", "DXY", "SPY"]

# Module-level cache for cross-asset data
_CROSS_ASSET_CACHE: dict = {}


def compute_cross_asset_signals(prices_dir: Optional[str] = None) -> dict:
    """
    Story 2.7: Extract cross-asset signals from cached price data.
    
    Returns dict with:
      - VIX_level, VIX_change (5d pct change)
      - DXY_level, DXY_change  
      - SPY_return_5d, SPY_return_20d
    """
    import json
    
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    signals = {}
    
    for symbol in ["^VIX", "DX-Y.NYB", "SPY"]:
        label = symbol.replace("^", "").replace("-Y.NYB", "").replace("DX", "DXY")
        price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
        if not os.path.exists(price_file):
            continue
        
        try:
            df = pd.read_csv(price_file)
            if "Close" not in df.columns:
                continue
            close = df["Close"].dropna().values
            if len(close) < 30:
                continue
            
            signals[f"{label}_level"] = float(close[-1])
            if len(close) >= 6:
                signals[f"{label}_change_5d"] = float((close[-1] / close[-6] - 1) * 100)
            if len(close) >= 21:
                signals[f"{label}_return_20d"] = float(np.sum(np.diff(np.log(close[-21:]))))
        except Exception:
            continue
    
    return signals


def compute_cross_asset_adjustment(asset_returns: np.ndarray, cross_signals: dict,
                                   asset_type: str = "equity") -> float:
    """
    Story 2.7: Compute cross-asset drift adjustment.
    
    Simple heuristic betas:
      - VIX rising -> negative for equities (beta = -0.02 per VIX point)
      - DXY rising -> negative for metals/commodities (beta = -0.01 per DXY pct) 
      - SPY momentum -> positive for risk-on assets
    
    Returns adjustment in percentage, capped at CROSS_ASSET_MAX_ADJ * abs(forecast).
    """
    adj = 0.0
    
    vix_change = cross_signals.get("VIX_change_5d", 0.0)
    dxy_change = cross_signals.get("DXY_change_5d", 0.0)
    spy_ret = cross_signals.get("SPY_return_20d", 0.0)
    
    if asset_type in ("equity",):
        # VIX rising -> headwind for equities
        adj += -0.02 * vix_change
        # SPY momentum -> tailwind
        adj += 0.5 * spy_ret * 100
    elif asset_type in ("metals", "commodity"):
        # USD strength -> headwind for metals
        adj += -0.01 * dxy_change
    elif asset_type in ("currency",):
        # DXY change -> direct impact
        adj += -0.005 * dxy_change
    
    return float(np.clip(adj, -5.0, 5.0))


def _load_tuned_params(asset_name: str) -> Optional[dict]:
    """
    Load tuned parameters from the tune cache for the given asset (Story 1.7).
    
    Returns dict with keys: q, c, phi, nu (optional), or None if unavailable.
    """
    try:
        import json
        tune_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "tune")
        tune_file = os.path.join(tune_dir, f"{asset_name}.json")
        if not os.path.exists(tune_file):
            return None
        
        with open(tune_file, "r") as f:
            cache = json.load(f)
        
        g = cache.get("global", {})
        q_val = g.get("q")
        c_val = g.get("c")
        phi_val = g.get("phi")
        
        if q_val is None or c_val is None or phi_val is None:
            return None
        
        params = {
            "q": float(q_val),
            "c": float(c_val),
            "phi": float(phi_val),
        }
        
        nu_val = g.get("nu")
        if nu_val is not None:
            params["nu"] = float(nu_val)
        
        # Include GAS-Q params if present
        gas_q = g.get("gas_q")
        if gas_q:
            params["gas_q"] = gas_q
        
        # Story 2.1: Extract GARCH params from regime models (BMA-weighted average)
        regime_data = cache.get("regime", {})
        garch_omega_sum = 0.0
        garch_alpha_sum = 0.0
        garch_beta_sum = 0.0
        garch_weight_sum = 0.0
        for _rkey, rval in regime_data.items():
            if not isinstance(rval, dict):
                continue
            models = rval.get("models", {})
            for _mname, mdata in models.items():
                if not isinstance(mdata, dict):
                    continue
                if "garch_omega" not in mdata:
                    continue
                w = max(float(mdata.get("weight", 0.0)), 1e-12)
                go = float(mdata.get("garch_omega", 0.0))
                ga = float(mdata.get("garch_alpha", 0.0))
                gb = float(mdata.get("garch_beta", 0.0))
                # Skip degenerate params
                if ga + gb >= 1.0 or go <= 0.0:
                    continue
                garch_omega_sum += w * go
                garch_alpha_sum += w * ga
                garch_beta_sum += w * gb
                garch_weight_sum += w
        
        if garch_weight_sum > 0:
            params["garch_params"] = {
                "omega": garch_omega_sum / garch_weight_sum,
                "alpha": garch_alpha_sum / garch_weight_sum,
                "beta": garch_beta_sum / garch_weight_sum,
                "persistence": (garch_alpha_sum + garch_beta_sum) / garch_weight_sum,
            }
        
        return params
    except Exception:
        return None


def _regime_detect(returns: np.ndarray) -> str:
    """Detect market regime from return characteristics."""
    try:
        if len(returns) < 20:
            return 'calm'
        vol = np.std(returns[-20:]) * np.sqrt(252)
        mom = np.sum(returns[-20:])
        autocorr = np.corrcoef(returns[-21:-1], returns[-20:])[0, 1] if len(returns) >= 21 else 0
        if vol > 0.30:
            return 'volatile'
        elif abs(mom) > 0.10:
            return 'trending'
        elif autocorr < -0.2:
            return 'mean_reverting'
        else:
            return 'calm'
    except Exception:
        return 'calm'


# =============================================================================
# Story 2.4: Bayesian Model Combination (BMC)
# Module-level cache for per-asset per-model predictive likelihood tracking.
# Uses forgetting factor (exponentially discounted) with 45-day half-life.
# =============================================================================
BMC_WEIGHT_FLOOR = 0.05      # Minimum weight per model (prevents extinction)
BMC_FORGETTING_HALF_LIFE = 45  # Days for weight half-life
BMC_MODEL_NAMES = ["kalman", "garch", "ou", "momentum", "classical"]

# Module-level cache: {asset_name: {"weights": [5], "log_likelihoods": [5], "n_updates": int}}
_BMC_CACHE: dict = {}

# Story 2.5: Forecast quantiles cache
# {asset_name: {"horizons": [...], "quantiles": [{p10, p25, p50, p75, p90}, ...]}}
_FORECAST_QUANTILES_CACHE: dict = {}

# Story 2.8: Forecast staleness and timestamp cache
# {asset_name: {"generated_at": datetime, "data_through": datetime, "forecasts": [...]}}
_FORECAST_TIMESTAMP_CACHE: dict = {}
STALENESS_THRESHOLD_HOURS = 4  # Alert if data > 4 hours old

# Story 2.9: Model explainability cache
# {asset_name: {"horizons": [...], "explanations": [{model_forecasts, weights, ...}, ...]}}
_FORECAST_EXPLAIN_CACHE: dict = {}
MODEL_NAMES = ["Kalman", "GARCH", "OU", "Momentum", "Classical"]

# Template strings for explainability (not dynamically generated)
EXPLAIN_DRIVER_TEMPLATE = "{model} ({forecast:+.2f}%, weight {weight:.2f}): driving {direction} signal"
EXPLAIN_DISSENT_TEMPLATE = "{model} ({forecast:+.2f}%) opposes consensus -- adding uncertainty"


def get_forecast_quantiles(asset_name: str) -> Optional[dict]:
    """
    Story 2.5: Get forecast confidence intervals for asset.
    
    Returns dict with "horizons" and "quantiles" keys, or None if not computed.
    Each quantile entry has keys: p10, p25, p50, p75, p90.
    """
    return _FORECAST_QUANTILES_CACHE.get(asset_name)


def get_forecast_staleness(asset_name: str) -> Optional[dict]:
    """
    Story 2.8: Get forecast staleness info for asset.
    
    Returns dict with:
      - generated_at: ISO timestamp string
      - data_through: ISO timestamp string
      - staleness_hours: float
      - is_stale: bool (True if > STALENESS_THRESHOLD_HOURS)
    """
    from datetime import datetime, timezone
    
    entry = _FORECAST_TIMESTAMP_CACHE.get(asset_name)
    if entry is None:
        return None
    
    now = datetime.now(timezone.utc)
    generated_at = entry["generated_at"]
    data_through = entry.get("data_through", generated_at)
    
    staleness_hours = (now - data_through).total_seconds() / 3600
    
    return {
        "generated_at": generated_at.isoformat(),
        "data_through": data_through.isoformat(),
        "staleness_hours": float(staleness_hours),
        "is_stale": staleness_hours > STALENESS_THRESHOLD_HOURS,
    }


def record_forecast_timestamp(asset_name: str, data_through=None):
    """
    Story 2.8: Record when a forecast was generated and what data it used.
    """
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    _FORECAST_TIMESTAMP_CACHE[asset_name] = {
        "generated_at": now,
        "data_through": data_through or now,
    }


def get_forecast_explanation(asset_name: str) -> Optional[dict]:
    """
    Story 2.9: Get model explainability breakdown for asset.
    
    Returns dict with "horizons" and "explanations" keys, or None if not computed.
    Each explanation entry has:
      - model_forecasts: list of 5 floats (per-model forecast)
      - weights: list of 5 floats (per-model weight)
      - contributions: list of 5 floats (weight * forecast)
      - top_contributor: str (model name)
      - top_dissenter: str or None
      - reason: natural-language explanation string
    """
    return _FORECAST_EXPLAIN_CACHE.get(asset_name)


def _build_explanation(forecasts_at_h: list, weights: list, ensemble: float) -> dict:
    """
    Story 2.9: Build a single-horizon explanation from model forecasts and weights.
    """
    contributions = [w * f for w, f in zip(weights, forecasts_at_h)]
    consensus_sign = 1 if ensemble >= 0 else -1
    direction = "bullish" if consensus_sign > 0 else "bearish"
    
    # Find top contributor (largest absolute contribution in consensus direction)
    top_idx = 0
    top_abs = 0.0
    for j in range(len(contributions)):
        sign_j = 1 if contributions[j] >= 0 else -1
        if sign_j == consensus_sign and abs(contributions[j]) > top_abs:
            top_abs = abs(contributions[j])
            top_idx = j
    
    top_name = MODEL_NAMES[top_idx] if top_idx < len(MODEL_NAMES) else f"Model_{top_idx}"
    
    # Find dissenter (largest absolute contribution opposing consensus)
    dissent_idx = None
    dissent_abs = 0.0
    for j in range(len(contributions)):
        sign_j = 1 if forecasts_at_h[j] >= 0 else -1
        if sign_j != consensus_sign and abs(forecasts_at_h[j]) > dissent_abs:
            dissent_abs = abs(forecasts_at_h[j])
            dissent_idx = j
    
    # Build reason string from constant templates
    reason = EXPLAIN_DRIVER_TEMPLATE.format(
        model=top_name,
        forecast=forecasts_at_h[top_idx],
        weight=weights[top_idx],
        direction=direction,
    )
    
    dissent_name = None
    if dissent_idx is not None:
        dissent_name = MODEL_NAMES[dissent_idx] if dissent_idx < len(MODEL_NAMES) else f"Model_{dissent_idx}"
        reason += " | " + EXPLAIN_DISSENT_TEMPLATE.format(
            model=dissent_name,
            forecast=forecasts_at_h[dissent_idx],
        )
    
    return {
        "model_forecasts": list(forecasts_at_h),
        "weights": list(weights),
        "contributions": contributions,
        "top_contributor": top_name,
        "top_dissenter": dissent_name,
        "reason": reason,
    }


def _bmc_log_likelihood(forecast_pct: float, realized_pct: float,
                        sigma_pct: float = 1.0) -> float:
    """
    Compute Gaussian log-likelihood of realized return given forecast.
    
    p(y | model) = N(y; forecast, sigma^2)
    log p = -0.5 * ((y - mu)^2 / sigma^2 + log(2*pi*sigma^2))
    """
    diff = realized_pct - forecast_pct
    return -0.5 * (diff ** 2 / (sigma_pct ** 2) + np.log(2 * np.pi * sigma_pct ** 2))


def update_bmc_weights(asset_name: str, forecasts: list, realized_pct: float,
                       sigma_pct: float = 1.0) -> list:
    """
    Story 2.4: Update BMC weights for asset given realized return.
    
    Implements:
      w_i,t+1 = w_i,t * p(y_t | model_i) / sum_j(w_j,t * p(y_t | model_j))
    
    With forgetting factor and weight floor.
    
    Args:
        asset_name: Asset identifier
        forecasts: List of 5 model forecasts (pct) for 1-day horizon
        realized_pct: Realized 1-day return (pct)
        sigma_pct: Forecast error std (pct) for likelihood computation
    
    Returns:
        Updated weights [5] summing to 1.0
    """
    if asset_name not in _BMC_CACHE:
        _BMC_CACHE[asset_name] = {
            "weights": [1.0 / 5] * 5,
            "n_updates": 0,
        }
    
    state = _BMC_CACHE[asset_name]
    weights = state["weights"]
    
    # Forgetting factor: lambda = 2^(-1/half_life) per day
    forget = 2.0 ** (-1.0 / BMC_FORGETTING_HALF_LIFE)
    
    # Compute per-model likelihood
    log_liks = []
    for i in range(5):
        fc = forecasts[i] if i < len(forecasts) else 0.0
        log_liks.append(_bmc_log_likelihood(fc, realized_pct, sigma_pct))
    
    # Exponentiate (shift for numerical stability)
    max_ll = max(log_liks)
    liks = [np.exp(ll - max_ll) for ll in log_liks]
    
    # Update: w_i = w_i * lik_i with forgetting
    new_weights = []
    for i in range(5):
        # Apply forget to shrink old weights toward uniform
        w_decayed = forget * weights[i] + (1 - forget) * (1.0 / 5)
        new_weights.append(w_decayed * liks[i])
    
    # Normalize
    total = sum(new_weights)
    if total > 0:
        new_weights = [w / total for w in new_weights]
    else:
        new_weights = [1.0 / 5] * 5
    
    # Apply floor: iterative approach to ensure each weight >= BMC_WEIGHT_FLOOR
    # while summing to 1.0
    n_models = 5
    total_floor = n_models * BMC_WEIGHT_FLOOR
    if total_floor >= 1.0:
        new_weights = [1.0 / n_models] * n_models
    else:
        # Normalize first
        raw_total = sum(new_weights)
        if raw_total > 0:
            new_weights = [w / raw_total for w in new_weights]
        else:
            new_weights = [1.0 / n_models] * n_models
        
        # Redistribute: clip below floor, give excess to above-floor models
        for _ in range(5):  # iterate to convergence
            deficit = 0.0
            surplus_total = 0.0
            for i in range(n_models):
                if new_weights[i] < BMC_WEIGHT_FLOOR:
                    deficit += BMC_WEIGHT_FLOOR - new_weights[i]
                    new_weights[i] = BMC_WEIGHT_FLOOR
                else:
                    surplus_total += new_weights[i]
            if deficit > 0 and surplus_total > 0:
                scale = (surplus_total - deficit) / surplus_total
                for i in range(n_models):
                    if new_weights[i] > BMC_WEIGHT_FLOOR:
                        new_weights[i] *= scale
        # Final normalize for safety
        wt = sum(new_weights)
        new_weights = [w / wt for w in new_weights]
    
    state["weights"] = new_weights
    state["n_updates"] += 1
    
    return new_weights


def get_bmc_weights(asset_name: str) -> list:
    """Get current BMC weights for asset, or uniform if not yet tracked."""
    if asset_name not in _BMC_CACHE:
        return [1.0 / 5] * 5
    return list(_BMC_CACHE[asset_name]["weights"])


def ensemble_forecast(prices: pd.Series, horizons: list = None, asset_type: str = "equity",
                      asset_name: str = "unknown", tuned_params: Optional[dict] = None) -> tuple:
    """
    Elite multi-model ensemble forecast with regime-aware weighting.
    
    Models (5 total):
    1. Kalman Filter - drift state estimation
    2. GARCH(1,1) - volatility-adjusted forecasts
    3. Ornstein-Uhlenbeck - mean reversion
    4. Momentum - multi-timeframe trend following
    5. Classical - baseline drift extrapolation
    
    IMPORTANT: Always uses STANDARD_HORIZONS [1, 3, 7, 30, 90, 180, 365] internally.
    The horizons parameter is ignored for consistency.
    
    Asset types: "equity", "currency", "metal", "crypto"
    Crypto detection: BTC, ETH are auto-detected from asset_name.
    
    Returns: (fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, confidence)
    """
    try:
        # ALWAYS use standard horizons regardless of what's passed
        horizons = STANDARD_HORIZONS
        
        if prices is None or len(prices) < 60:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Auto-detect crypto assets
        if _is_crypto_asset(asset_name):
            asset_type = "crypto"

        log_returns = np.log(prices / prices.shift(1)).dropna().values
        if len(log_returns) < 30:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Auto-load tuned parameters from cache (Story 1.7)
        if tuned_params is None and asset_name and asset_name != "unknown":
            tuned_params = _load_tuned_params(asset_name)
        
        # Get individual model forecasts
        kalman_fc = _kalman_forecast(log_returns, horizons, asset_type, tuned_params=tuned_params)
        garch_fc = _garch_forecast(log_returns, horizons, asset_type, tuned_params=tuned_params)
        ou_fc = _ou_forecast(prices, horizons, asset_type, tuned_params=tuned_params)
        mom_fc = _momentum_forecast(log_returns, horizons, asset_type)
        
        # Classical drift forecast
        drift_5d = np.mean(log_returns[-5:])
        drift_20d = np.mean(log_returns[-20:])
        drift_60d = np.mean(log_returns[-60:]) if len(log_returns) >= 60 else drift_20d
        
        # Story 2.6: Adaptive drift with regime-dependent persistence
        # EWMA drift with half-life = 21 days
        ew_alpha = 2.0 / (21 + 1)  # EW alpha for 21-day half-life
        ew_drift = float(log_returns[0])
        for r in log_returns[1:]:
            ew_drift = ew_alpha * float(r) + (1 - ew_alpha) * ew_drift
        
        # Regime-dependent persistence decay rates
        regime = _regime_detect(log_returns)
        if regime == 'trending':
            persistence_base = 0.97  # Slow decay (trust the trend)
        elif regime == 'mean_reverting':
            persistence_base = 0.90  # Faster decay (drift unreliable)
        elif regime == 'volatile':
            persistence_base = 0.80  # Very fast decay (crisis-like)
        else:  # calm
            persistence_base = 0.93
        
        # Soft blending: estimate regime probabilities from vol characteristics
        vol_20d = float(np.std(log_returns[-20:])) * np.sqrt(252)
        vol_60d = float(np.std(log_returns[-min(60, len(log_returns)):])) * np.sqrt(252)
        
        # Simple soft regime weights based on vol level
        p_crisis = min(max((vol_20d - 0.30) / 0.20, 0), 1)  # vol > 30% -> crisis
        p_trend = min(max(abs(np.mean(log_returns[-20:]) * 252) / 0.15, 0), 1) * (1 - p_crisis)
        p_calm = max(0, 1 - p_crisis - p_trend)
        
        # Blend persistence
        persistence_blend = (
            p_trend * 0.97 + p_calm * 0.93 + p_crisis * 0.80
        )
        
        classical_fc = []
        for h in horizons:
            # Adaptive persistence: blended base decays over horizon
            persistence = persistence_blend ** h
            fc = ew_drift * h * persistence * 100
            classical_fc.append(float(fc))
        
        # Base weights: [Kalman, GARCH, OU, Momentum, Classical]
        if asset_type == "crypto":
            # Crypto: momentum dominates, OU is weak
            if regime == 'trending':
                base_weights = [0.20, 0.10, 0.05, 0.50, 0.15]
            else:
                base_weights = [0.20, 0.15, 0.10, 0.40, 0.15]
        elif regime == 'trending':
            base_weights = [0.30, 0.10, 0.10, 0.35, 0.15]
        elif regime == 'mean_reverting':
            base_weights = [0.15, 0.15, 0.40, 0.10, 0.20]
        elif regime == 'volatile':
            base_weights = [0.20, 0.35, 0.20, 0.10, 0.15]
        else:  # calm
            base_weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        
        # Story 2.4: Blend fixed regime weights with BMC adaptive weights
        bmc_weights = get_bmc_weights(asset_name)
        bmc_blend = 0.5  # 50% BMC + 50% regime-fixed
        base_weights = [
            bmc_blend * bmc_weights[j] + (1 - bmc_blend) * base_weights[j]
            for j in range(5)
        ]
        bw_total = sum(base_weights)
        base_weights = [w / bw_total for w in base_weights]
        
        final_forecasts = []
        forecast_uncertainties = []
        agreement_contrasts = []  # Story 1.14: per-horizon agreement for cap relaxation
        horizon_explanations = []  # Story 2.9: per-horizon explainability
        for i, h in enumerate(horizons):
            # Horizon-specific weight adjustments
            if h <= 3:
                # Very short-term: momentum and Kalman dominate
                adj = [0.05, 0.0, -0.08, 0.10, -0.07]
            elif h <= 7:
                adj = [0.02, 0.0, -0.05, 0.06, -0.03]
            elif h <= 30:
                adj = [0.0, 0.0, 0.02, 0.0, -0.02]
            elif h <= 90:
                # Medium-term: balance, OU gains
                adj = [-0.02, 0.0, 0.08, -0.04, -0.02]
            else:
                # Long-term: OU and GARCH gain, momentum fades
                if asset_type == "crypto":
                    adj = [0.0, 0.05, 0.05, -0.05, -0.05]
                else:
                    adj = [0.0, 0.0, 0.12, -0.10, -0.02]
            
            weights = [max(0.02, base_weights[j] + adj[j]) for j in range(5)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            
            forecasts_at_h = [
                kalman_fc[i] if i < len(kalman_fc) else 0.0,
                garch_fc[i] if i < len(garch_fc) else 0.0,
                ou_fc[i] if i < len(ou_fc) else 0.0,
                mom_fc[i] if i < len(mom_fc) else 0.0,
                classical_fc[i] if i < len(classical_fc) else 0.0,
            ]
            
            # =========================================================
            # Sign-agreement-weighted averaging (Story 1.6)
            # =========================================================
            weighted_median = sum(w * f for w, f in zip(weights, forecasts_at_h))
            sign_median = 1 if weighted_median >= 0 else -1
            
            # Compute agreement: fraction of weight with same sign as median
            agree_weight = 0.0
            for j in range(5):
                sign_j = 1 if forecasts_at_h[j] >= 0 else -1
                if sign_j == sign_median:
                    agree_weight += weights[j]
            
            agreement_contrast = max(0.0, (agree_weight - 0.5) * 2.0)
            
            # Adjust weights based on sign agreement
            adjusted_weights = []
            for j in range(5):
                sign_j = 1 if forecasts_at_h[j] >= 0 else -1
                if sign_j == sign_median:
                    adj_w = weights[j] * (1.0 + CONTRAST_BOOST * agreement_contrast)
                else:
                    adj_w = weights[j] * (1.0 - CONTRAST_BOOST * agreement_contrast)
                adjusted_weights.append(max(adj_w, 0.01))
            
            adj_total = sum(adjusted_weights)
            adjusted_weights = [w / adj_total for w in adjusted_weights]
            
            ensemble = sum(w * f for w, f in zip(adjusted_weights, forecasts_at_h))
            
            # Forecast dispersion (uncertainty measure)
            fc_arr = np.array(forecasts_at_h)
            fc_std = float(np.std(fc_arr))
            fc_mean_abs = abs(ensemble) if abs(ensemble) > 1e-8 else 1e-8
            dispersion = fc_std / fc_mean_abs
            
            # Normalize to 0-1 scale (dispersion of 3+ maps to 1.0)
            uncertainty = min(dispersion / 3.0, 1.0)
            forecast_uncertainties.append(float(uncertainty))
            agreement_contrasts.append(float(agreement_contrast))
            
            # Story 2.9: Record per-horizon explanation
            horizon_explanations.append(
                _build_explanation(forecasts_at_h, adjusted_weights, ensemble)
            )
            
            final_forecasts.append(float(ensemble))
        
        vol = float(np.std(log_returns) * np.sqrt(252))
        
        # Story 1.11: Compute vol ratio and signal quality
        daily_vol = np.abs(log_returns)
        ewma_alpha = 2.0 / (VOL_RATIO_EWMA_SPAN + 1)
        ewma_vol = float(daily_vol[-1])
        for t in range(max(0, len(daily_vol) - VOL_RATIO_EWMA_SPAN), len(daily_vol)):
            ewma_vol = ewma_alpha * daily_vol[t] + (1 - ewma_alpha) * ewma_vol
        
        median_window = min(VOL_RATIO_MEDIAN_WINDOW, len(daily_vol))
        median_vol = float(np.median(daily_vol[-median_window:])) if median_window > 0 else ewma_vol
        vol_ratio = ewma_vol / max(median_vol, 1e-10)
        signal_quality = compute_signal_quality(vol_ratio)
        vol_regime_label = compute_vol_regime_label(vol_ratio)
        
        # Asset-specific hard caps (realistic bounds)
        if asset_type == "crypto":
            hard_caps = {1: 10, 3: 18, 7: 30, 30: 60, 90: 85, 180: 120, 365: 180}
        elif asset_type == "currency":
            hard_caps = {1: 2.0, 3: 3.5, 7: 5.5, 30: 10, 90: 15, 180: 22, 365: 30}
        elif asset_type == "metal":
            hard_caps = {1: 4, 3: 7, 7: 10, 30: 18, 90: 28, 180: 40, 365: 55}
        else:
            hard_caps = {1: 3, 3: 5, 7: 8, 30: 14, 90: 22, 180: 32, 365: 45}
        
        bounded_forecasts = []
        for i, h in enumerate(horizons):
            fc = final_forecasts[i]
            hard_cap = hard_caps.get(h, 35)
            
            # Story 1.14: Confidence-gated cap relaxation
            # High model agreement -> widen cap (up to CAP_MAX_MULTIPLIER)
            ac = agreement_contrasts[i] if i < len(agreement_contrasts) else 0.0
            cap_mult = min(1.0 + CAP_CONFIDENCE_SCALE * ac, CAP_MAX_MULTIPLIER)
            relaxed_cap = hard_cap * cap_mult
            
            # Vol-bound remains as independent safety constraint
            vol_bound = vol * np.sqrt(h / 252) * 2.5 * 100
            if asset_type == "crypto":
                max_fc = max(vol_bound, relaxed_cap)
            else:
                max_fc = max(min(vol_bound, relaxed_cap), 0.5)
            
            fc = float(np.clip(fc, -max_fc, max_fc))
            bounded_forecasts.append(fc)
        
        # Confidence score
        data_score = min(len(prices) / 400, 1.0)
        if asset_type == "crypto":
            vol_score = 1 - min(vol / 1.50, 1.0)
        else:
            vol_score = 1 - min(vol / 0.50, 1.0)
        
        # Regime-based confidence
        if regime == 'calm':
            regime_score = 0.85
        elif regime == 'trending':
            regime_score = 0.75
        elif regime == 'mean_reverting':
            regime_score = 0.65
        else:  # volatile
            regime_score = 0.45
        
        conf_score = data_score * 0.30 + vol_score * 0.30 + regime_score * 0.40
        
        # Story 1.11: Quality-adjusted confidence
        # Signal quality modulates the confidence level but NEVER the forecast values
        quality_adjusted_score = conf_score * min(signal_quality, 1.5)  # Cap amplification
        
        # Check for contested forecasts (Story 1.6)
        avg_uncertainty = float(np.mean(forecast_uncertainties)) if forecast_uncertainties else 0.0
        avg_dispersion = avg_uncertainty * 3.0  # Reverse the normalization
        is_contested = avg_dispersion > DISPERSION_CONTESTED
        
        if is_contested:
            confidence = "Contested"
        elif vol_regime_label == "EXTREME":
            confidence = "Low"  # Override in extreme vol
        elif quality_adjusted_score > 0.65:
            confidence = "High"
        elif quality_adjusted_score > 0.45:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Append vol regime tag to confidence
        if vol_regime_label != "NORMAL":
            confidence = f"{confidence} [{vol_regime_label}]"
        
        while len(bounded_forecasts) < 7:
            bounded_forecasts.append(0.0)
        while len(forecast_uncertainties) < 7:
            forecast_uncertainties.append(0.0)
        
        # Story 2.5: Compute forecast confidence intervals (fan chart quantiles)
        # Using model disagreement + vol-scaled noise to estimate quantiles
        forecast_quantiles = []
        for i, h in enumerate(horizons[:7]):
            fc_point = bounded_forecasts[i] if i < len(bounded_forecasts) else 0.0
            
            # Model spread: std of 5 sub-model forecasts at this horizon
            sub_forecasts = [
                kalman_fc[i] if i < len(kalman_fc) else 0.0,
                garch_fc[i] if i < len(garch_fc) else 0.0,
                ou_fc[i] if i < len(ou_fc) else 0.0,
                mom_fc[i] if i < len(mom_fc) else 0.0,
                classical_fc[i] if i < len(classical_fc) else 0.0,
            ]
            model_std = float(np.std(sub_forecasts))
            
            # Vol-scaled component: sqrt(h) scaling
            vol_component = vol * np.sqrt(h / 252) * 100
            
            # Combined uncertainty: model disagreement + vol
            combined_std = np.sqrt(model_std ** 2 + vol_component ** 2)
            
            # Quantiles: assuming Gaussian (adequate for fan charts)
            from scipy.stats import norm
            quantiles = {}
            for q, z in [(10, -1.2816), (25, -0.6745), (50, 0.0),
                         (75, 0.6745), (90, 1.2816)]:
                quantiles[f"p{q}"] = float(fc_point + z * combined_std)
            forecast_quantiles.append(quantiles)
        
        # Store in module-level cache for retrieval
        _FORECAST_QUANTILES_CACHE[asset_name] = {
            "horizons": horizons[:7],
            "quantiles": forecast_quantiles,
        }
        
        # Story 2.9: Store model explainability in cache
        _FORECAST_EXPLAIN_CACHE[asset_name] = {
            "horizons": horizons[:7],
            "explanations": horizon_explanations[:7],
        }
        
        # Story 2.8: Record forecast timestamp
        try:
            last_date = prices.index[-1] if hasattr(prices.index[-1], 'isoformat') else None
            if last_date is not None:
                from datetime import timezone
                if last_date.tzinfo is None:
                    last_date = last_date.replace(tzinfo=timezone.utc)
                record_forecast_timestamp(asset_name, data_through=last_date)
            else:
                record_forecast_timestamp(asset_name)
        except Exception:
            record_forecast_timestamp(asset_name)
        
        return tuple(bounded_forecasts[:7]) + (confidence,)
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"

# =============================================================================
# MARKET TEMPERATURE CONSTANTS
# =============================================================================

# Universe weights (sum to 1.0)
MEGA_CAP_WEIGHT = 0.40       # Top 100 mega-caps
BROAD_MARKET_WEIGHT = 0.30   # S&P 500 proxy
SMALL_CAP_WEIGHT = 0.20      # Russell 2000
GROWTH_VALUE_WEIGHT = 0.10   # Growth vs Value rotation
INTERNATIONAL_WEIGHT = 0.05  # International indexes (informational, not used in main temp calc)

# Z-score calculation lookback
ZSCORE_LOOKBACK_DAYS = 60

# Volatility percentile lookback
VOLATILITY_LOOKBACK_DAYS = 252  # 1 year

# Scaling function parameters
SIGMOID_K = 3.0
SIGMOID_THRESHOLD = 1.0

# Temperature bounds
TEMP_MIN = 0.0
TEMP_MAX = 2.0

# Cache TTL (seconds)
CACHE_TTL_SECONDS = 3600  # 1 hour

# MAD consistency constant for robust z-score
MAD_CONSISTENCY_CONSTANT = 1.4826

# Vol term structure inversion
VOL_TERM_STRUCTURE_SHORT_WINDOW = 5
VOL_TERM_STRUCTURE_LONG_WINDOW = 20
VOL_TERM_STRUCTURE_INVERSION_THRESHOLD = 1.5

# Breadth thresholds
BREADTH_WARNING_THRESHOLD = 0.40   # < 40% above 50-day MA = warning
BREADTH_DANGER_THRESHOLD = 0.25    # < 25% = danger

# Momentum thresholds
MOMENTUM_STRONG_BULL = 0.15    # > 15% in 21 days
MOMENTUM_MILD_BULL = 0.05      # > 5%
MOMENTUM_MILD_BEAR = -0.05     # < -5%
MOMENTUM_STRONG_BEAR = -0.15   # < -15%

# Correlation stress
CORRELATION_LOOKBACK = 60
CORRELATION_STRESS_THRESHOLD = 0.75  # Avg correlation > 0.75 = systemic risk


class AlertSeverity:
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UniverseMetrics:
    """Metrics for a single universe segment (e.g., Top 100, Small Caps)."""
    name: str
    weight: float
    
    # Price/Return metrics
    current_level: Optional[float] = None
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    return_63d: float = 0.0  # 3 months
    
    # Volatility metrics
    volatility_20d: float = 0.0
    volatility_percentile: float = 0.5
    vol_term_structure_ratio: float = 1.0
    vol_inverted: bool = False
    
    # Breadth (for indices with components)
    breadth_pct_above_50ma: Optional[float] = None
    breadth_pct_above_200ma: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    
    # Computed stress
    stress_level: float = 0.0
    stress_contribution: float = 0.0
    
    # Momentum signal
    momentum_signal: str = "→ Flat"
    
    # Data quality
    data_available: bool = False
    ticker_count: int = 0
    
    # Forecasts
    forecast_1d: float = 0.0
    forecast_3d: float = 0.0
    forecast_7d: float = 0.0
    forecast_30d: float = 0.0
    forecast_90d: float = 0.0
    forecast_180d: float = 0.0
    forecast_365d: float = 0.0
    forecast_confidence: str = "Low"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "current_level": self.current_level,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "return_63d": self.return_63d,
            "volatility_20d": self.volatility_20d,
            "volatility_percentile": self.volatility_percentile,
            "vol_term_structure_ratio": self.vol_term_structure_ratio,
            "vol_inverted": self.vol_inverted,
            "breadth_pct_above_50ma": self.breadth_pct_above_50ma,
            "breadth_pct_above_200ma": self.breadth_pct_above_200ma,
            "advance_decline_ratio": self.advance_decline_ratio,
            "stress_level": self.stress_level,
            "stress_contribution": self.stress_contribution,
            "momentum_signal": self.momentum_signal,
            "data_available": self.data_available,
            "ticker_count": self.ticker_count,
                        "forecast_1d": self.forecast_1d,
            "forecast_3d": self.forecast_3d,
            "forecast_3d": self.forecast_3d,
            "forecast_7d": self.forecast_7d,
            "forecast_30d": self.forecast_30d,
            "forecast_90d": self.forecast_90d,
            "forecast_180d": self.forecast_180d,
            "forecast_365d": self.forecast_365d,
            "forecast_confidence": self.forecast_confidence,
        }


@dataclass
class SectorMetrics:
    """Metrics for a single sector (e.g., Technology, Financials)."""
    name: str
    ticker: str  # ETF ticker (e.g., XLK for Technology)
    
    # Price/Return metrics
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    
    # Volatility
    volatility_20d: float = 0.0
    volatility_percentile: float = 0.5
    
    # Momentum
    momentum_signal: str = "→ Flat"
    
    # Risk score (0-100)
    risk_score: int = 0
    
    # Forecasts
    forecast_1d: float = 0.0
    forecast_3d: float = 0.0
    forecast_7d: float = 0.0
    forecast_30d: float = 0.0
    forecast_90d: float = 0.0
    forecast_180d: float = 0.0
    forecast_365d: float = 0.0
    forecast_confidence: str = "Low"
    
    # Data availability
    data_available: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ticker": self.ticker,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "volatility_20d": self.volatility_20d,
            "volatility_percentile": self.volatility_percentile,
            "momentum_signal": self.momentum_signal,
            "risk_score": self.risk_score,
            "data_available": self.data_available,
                        "forecast_1d": self.forecast_1d,
            "forecast_3d": self.forecast_3d,
            "forecast_3d": self.forecast_3d,
            "forecast_7d": self.forecast_7d,
            "forecast_30d": self.forecast_30d,
            "forecast_90d": self.forecast_90d,
            "forecast_180d": self.forecast_180d,
            "forecast_365d": self.forecast_365d,
            "forecast_confidence": self.forecast_confidence,
        }


@dataclass
class CurrencyMetrics:
    """Metrics for a currency pair (e.g., EUR/USD, GBP/USD)."""
    name: str
    ticker: str
    rate: float = 0.0
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    volatility_20d: float = 0.0
    momentum_signal: str = "→ Flat"
    risk_score: int = 0
    data_available: bool = False
    # Forecasts (scientifically computed using drift + mean reversion + volatility)
    forecast_1d: float = 0.0      # 1 day forecast (% change)
    forecast_3d: float = 0.0      # 3 day forecast
    forecast_7d: float = 0.0      # 7 day forecast
    forecast_30d: float = 0.0     # 30 day (1 month) forecast
    forecast_90d: float = 0.0     # 90 day (3 month) forecast
    forecast_180d: float = 0.0    # 180 day (6 month) forecast
    forecast_365d: float = 0.0    # 365 day (12 month) forecast
    forecast_confidence: str = "Low"  # Low/Medium/High based on model fit
    is_inverse: bool = False      # True for JPY/XXX pairs (computed from XXX/JPY)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ticker": self.ticker,
            "rate": self.rate,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "volatility_20d": self.volatility_20d,
            "momentum_signal": self.momentum_signal,
            "risk_score": self.risk_score,
            "data_available": self.data_available,
                        "forecast_1d": self.forecast_1d,
            "forecast_3d": self.forecast_3d,
            "forecast_7d": self.forecast_7d,
            "forecast_30d": self.forecast_30d,
            "forecast_90d": self.forecast_90d,
            "forecast_180d": self.forecast_180d,
            "forecast_365d": self.forecast_365d,
            "forecast_confidence": self.forecast_confidence,
            "is_inverse": self.is_inverse,
        }


@dataclass
class MarketBreadthMetrics:
    """Aggregated market breadth analysis."""
    pct_above_50ma: float = 0.5
    pct_above_200ma: float = 0.5
    new_highs: int = 0
    new_lows: int = 0
    advance_decline_ratio: float = 1.0
    breadth_thrust: bool = False  # Extreme breadth signal
    breadth_warning: bool = False
    interpretation: str = "Normal"
    
    def to_dict(self) -> Dict:
        return {
            "pct_above_50ma": self.pct_above_50ma,
            "pct_above_200ma": self.pct_above_200ma,
            "new_highs": self.new_highs,
            "new_lows": self.new_lows,
            "advance_decline_ratio": self.advance_decline_ratio,
            "breadth_thrust": self.breadth_thrust,
            "breadth_warning": self.breadth_warning,
            "interpretation": self.interpretation,
        }


@dataclass
class CorrelationStress:
    """Cross-asset correlation analysis."""
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_percentile: float = 0.5
    systemic_risk_elevated: bool = False
    interpretation: str = "Normal dispersion"
    
    def to_dict(self) -> Dict:
        return {
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "correlation_percentile": self.correlation_percentile,
            "systemic_risk_elevated": self.systemic_risk_elevated,
            "interpretation": self.interpretation,
        }


@dataclass
class MarketTemperatureResult:
    """Complete market temperature computation result."""
    temperature: float                       # Final temperature ∈ [0, 2]
    scale_factor: float                      # Position scaling factor ∈ (0, 1)
    universes: Dict[str, UniverseMetrics]    # Per-universe metrics
    breadth: MarketBreadthMetrics            # Market breadth analysis
    correlation: CorrelationStress           # Correlation stress analysis
    computed_at: str                         # ISO timestamp
    data_quality: float                      # Fraction of universes with data
    status: str                              # Calm, Elevated, Stressed, Extreme
    action_text: str                         # Position recommendation
    
    # Crash risk fields
    crash_risk_pct: float = 0.0
    crash_risk_level: str = "Low"
    vol_inversion_count: int = 0
    inverted_universes: Optional[List[str]] = None
    
    # Momentum summary
    overall_momentum: str = "→ Neutral"
    sector_rotation_signal: str = "Normal"
    
    # Exit signals
    exit_signal: bool = False
    exit_reason: Optional[str] = None
    
    # Sector-by-sector breakdown (February 2026)
    sectors: Dict[str, SectorMetrics] = field(default_factory=dict)
    
    # Currency pairs breakdown (February 2026)
    currencies: Dict[str, CurrencyMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "temperature": float(self.temperature),
            "scale_factor": float(self.scale_factor),
            "status": self.status,
            "action_text": self.action_text,
            "computed_at": self.computed_at,
            "data_quality": float(self.data_quality),
            "universes": {k: v.to_dict() for k, v in self.universes.items()},
            "breadth": self.breadth.to_dict(),
            "correlation": self.correlation.to_dict(),
            "crash_risk_pct": float(self.crash_risk_pct),
            "crash_risk_level": self.crash_risk_level,
            "vol_inversion_count": self.vol_inversion_count,
            "inverted_universes": self.inverted_universes,
            "overall_momentum": self.overall_momentum,
            "sector_rotation_signal": self.sector_rotation_signal,
            "exit_signal": self.exit_signal,
            "exit_reason": self.exit_reason,
            "sectors": {k: v.to_dict() for k, v in self.sectors.items()},
            "currencies": {k: v.to_dict() for k, v in self.currencies.items()},
        }
    
    @property
    def is_elevated(self) -> bool:
        return self.temperature > 0.5
    
    @property
    def is_stressed(self) -> bool:
        return self.temperature > 1.0
    
    @property
    def is_extreme(self) -> bool:
        return self.temperature > 1.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_robust_zscore(
    values: pd.Series,
    lookback: int = ZSCORE_LOOKBACK_DAYS
) -> float:
    """Compute robust z-score using Median Absolute Deviation (MAD)."""
    try:
        if values is None or len(values) == 0:
            return 0.0
        
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 1:
                values = values.iloc[:, 0]
            else:
                return 0.0
        
        values = values.dropna()
        
        if len(values) < lookback // 2:
            return 0.0
        
        recent = values.iloc[-lookback:] if len(values) >= lookback else values
        current = float(values.iloc[-1])
        
        median = float(recent.median())
        mad = float((recent - median).abs().median())
        
        scaled_mad = MAD_CONSISTENCY_CONSTANT * mad
        
        if scaled_mad < 1e-10:
            std = float(recent.std())
            if std < 1e-10:
                return 0.0
            zscore = (current - recent.mean()) / std
        else:
            zscore = (current - median) / scaled_mad
        
        return float(np.clip(zscore, -5.0, 5.0))
    except Exception:
        return 0.0


def _compute_volatility_percentile(
    prices: pd.Series,
    vol_window: int = 20,
    lookback: int = VOLATILITY_LOOKBACK_DAYS
) -> float:
    """Compute current volatility percentile over lookback period."""
    try:
        if prices is None or len(prices) < vol_window + 10:
            return 0.5
        
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return 0.5
        
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) < 20:
            return 0.5
        
        current_vol = float(rolling_vol.iloc[-1])
        historical_vol = rolling_vol.iloc[-lookback:] if len(rolling_vol) >= lookback else rolling_vol
        
        percentile = (historical_vol < current_vol).sum() / len(historical_vol)
        return float(percentile)
    except Exception:
        return 0.5


def _compute_vol_term_structure(
    prices: pd.Series,
    short_window: int = VOL_TERM_STRUCTURE_SHORT_WINDOW,
    long_window: int = VOL_TERM_STRUCTURE_LONG_WINDOW,
) -> Tuple[float, bool]:
    """Compute volatility term structure ratio. Returns (ratio, is_inverted)."""
    try:
        if prices is None or len(prices) < long_window + 5:
            return 1.0, False
        
        returns = prices.pct_change().dropna()
        
        if len(returns) < long_window:
            return 1.0, False
        
        short_vol = float(returns.iloc[-short_window:].std() * np.sqrt(252))
        long_vol = float(returns.iloc[-long_window:].std() * np.sqrt(252))
        
        if long_vol < 1e-10:
            return 1.0, False
        
        ratio = short_vol / long_vol
        is_inverted = ratio >= VOL_TERM_STRUCTURE_INVERSION_THRESHOLD
        
        return ratio, is_inverted
    except Exception:
        return 1.0, False


def _compute_returns(prices: pd.Series) -> Dict[str, float]:
    """Compute returns over multiple horizons."""
    results = {"1d": 0.0, "5d": 0.0, "21d": 0.0, "63d": 0.0}
    
    try:
        if prices is None or len(prices) < 2:
            return results
        
        prices = prices.dropna()
        current = float(prices.iloc[-1])
        
        if len(prices) >= 2:
            results["1d"] = (current / float(prices.iloc[-2]) - 1)
        if len(prices) >= 6:
            results["5d"] = (current / float(prices.iloc[-6]) - 1)
        if len(prices) >= 22:
            results["21d"] = (current / float(prices.iloc[-22]) - 1)
        if len(prices) >= 64:
            results["63d"] = (current / float(prices.iloc[-64]) - 1)
    except Exception:
        pass
    
    return results


def _compute_momentum_signal(return_5d: float, return_21d: float) -> str:
    """Compute momentum signal from returns."""
    avg_momentum = (return_5d + return_21d) / 2
    
    if avg_momentum >= MOMENTUM_STRONG_BULL:
        return "↑ Strong"
    elif avg_momentum >= MOMENTUM_MILD_BULL:
        return "↗ Rising"
    elif avg_momentum <= MOMENTUM_STRONG_BEAR:
        return "↓ Weak"
    elif avg_momentum <= MOMENTUM_MILD_BEAR:
        return "↘ Falling"
    else:
        return "→ Flat"


def _compute_scale_factor(temperature: float) -> float:
    """
    Compute position scale factor using smooth exponential decay.
    
    February 2026 Enhancement (Professor Zhang Xin-Yu, Score: 8.7/10):
    Replaces sigmoid-based scaling with smooth exponential decay that
    eliminates discontinuities and uses hysteresis to prevent oscillation.
    
    Formula:
        scale = exp(-decay_rate * max(0, temperature - threshold))
    
    Properties:
        - Continuous and differentiable everywhere
        - scale = 1.0 when temperature <= threshold
        - Smooth decay as temperature increases
    """
    if USE_COPULA_CORRELATION and COPULA_CORRELATION_AVAILABLE:
        return compute_smooth_scale_factor(
            temperature,
            threshold=SIGMOID_THRESHOLD,
            decay_rate=SIGMOID_K * 0.7,  # Slightly gentler than sigmoid equivalent
            hysteresis_band=0.05,
            state_key="market_temperature"
        )
    
    # Fallback to original sigmoid
    return 1.0 / (1.0 + math.exp(SIGMOID_K * (temperature - SIGMOID_THRESHOLD)))


def _get_status_and_action(temperature: float) -> Tuple[str, str]:
    """Get status label and action text based on temperature."""
    if temperature >= 1.5:
        return "Extreme", "EXIT POSITIONS - Capital preservation mode"
    elif temperature >= 1.0:
        return "Stressed", "REDUCE EXPOSURE - Risk management priority"
    elif temperature >= 0.7:
        return "Elevated", "CAUTION - Consider hedging"
    elif temperature >= 0.5:
        return "Warm", "MONITOR - Tighten stops"
    else:
        return "Calm", "NORMAL - Business as usual"


def _extract_close_series(df, ticker: str) -> Optional[pd.Series]:
    """Safely extract Close price series from yfinance DataFrame."""
    if df is None:
        return None
    
    try:
        if hasattr(df, 'empty') and df.empty:
            return None
    except ValueError:
        if len(df) == 0:
            return None
    
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                # Check if it's a multi-ticker DataFrame
                close_df = df['Close']
                if isinstance(close_df, pd.DataFrame):
                    # Try to get the specific ticker's column
                    if ticker in close_df.columns:
                        return close_df[ticker].dropna()
                    # Try without special characters (e.g., ^VIX -> VIX)
                    ticker_clean = ticker.replace('^', '').replace('=X', '')
                    if ticker_clean in close_df.columns:
                        return close_df[ticker_clean].dropna()
                    # Fallback to first column only if single column
                    if len(close_df.columns) == 1:
                        return close_df.iloc[:, 0].dropna()
                    # For multi-column, don't fallback - return None to avoid wrong data
                    return None
                else:
                    # It's already a Series
                    return close_df.dropna()
        
        if 'Close' in df.columns:
            series = df['Close']
            if isinstance(series, pd.DataFrame):
                # Same logic for multi-ticker case
                if ticker in series.columns:
                    return series[ticker].dropna()
                ticker_clean = ticker.replace('^', '').replace('=X', '')
                if ticker_clean in series.columns:
                    return series[ticker_clean].dropna()
                if len(series.columns) == 1:
                    return series.iloc[:, 0].dropna()
                return None
            return series.dropna()
        
        if 'close' in df.columns:
            return df['close'].dropna()
        
        # Only use first column fallback for single-column DataFrames
        if len(df.columns) == 1:
            return df.iloc[:, 0].dropna()
    except Exception:
        pass
    
    return None


# =============================================================================
# TOP 100 US STOCKS - Mega-Cap Universe
# =============================================================================

# Top 100 US stocks by market cap (static list - updated periodically)
TOP_100_TICKERS = [
    # Top 10 Mega-Caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    # 11-25
    "V", "XOM", "JPM", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO",
    # 26-50
    "TMO", "MCD", "CSCO", "ABT", "ACN", "WFC", "DHR", "ADBE", "CRM", "NKE",
    "TXN", "NEE", "PM", "BMY", "ORCL", "LIN", "UNP", "CMCSA", "UPS", "RTX",
    "AMD", "QCOM", "T", "LOW", "MS",
    # 51-75
    "HON", "INTC", "SPGI", "COP", "CAT", "BA", "GS", "ELV", "SBUX", "DE",
    "IBM", "INTU", "PLD", "AMGN", "GE", "AXP", "BKNG", "ISRG", "MDLZ", "GILD",
    "BLK", "ADI", "MMC", "REGN", "CVS",
    # 76-100
    "TJX", "VRTX", "SYK", "SCHW", "ADP", "C", "PGR", "ZTS", "LRCX", "CI",
    "CB", "NOW", "MO", "SO", "DUK", "SLB", "EOG", "PNC", "BDX", "ITW",
    "CL", "USB", "CME", "MCO", "APD",
]

# S&P 500 Representative Sample (50 stocks across sectors)
SP500_SAMPLE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "ACN",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrials
    "CAT", "HON", "UNP", "BA", "GE", "RTX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "VZ",
    # Materials/Utilities
    "LIN", "APD", "NEE", "DUK", "SO",
]

# Russell 2000 Representative Sample (30 small-cap stocks)
RUSSELL_2000_SAMPLE = [
    "AMC", "GME", "BBBY", "SPCE", "PLTR",  # High-volatility small caps
    "CROX", "FIVE", "BOOT", "WSM", "PRGS",  # Consumer/Retail
    "PRLB", "NMIH", "ESNT", "RDN", "MTG",   # Financials
    "CARG", "VCEL", "FOLD", "AXNX", "TGTX", # Healthcare
    "DOCN", "NEOG", "ZI", "AMBA", "CRNC",   # Tech
    "MATX", "ARCB", "WERN", "SAIA", "XPO",  # Industrials
]


# =============================================================================
# SECTOR ETFs - For sector-by-sector breakdown (February 2026)
# =============================================================================
# SPDR Select Sector ETFs tracking S&P 500 sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Comm. Svcs",
}


# =============================================================================
# MARKET DATA CACHE
# =============================================================================

# =============================================================================
# SECTOR ETFs - For sector-by-sector breakdown (February 2026)
# =============================================================================
# SPDR Select Sector ETFs tracking S&P 500 sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Comm. Svcs",
}

# =============================================================================
# CURRENCY PAIRS - For FX market breakdown (February 2026)
# =============================================================================
# Major currency pairs and cryptocurrencies with Yahoo Finance tickers
# Convention: XXXJPY=X means "how many JPY per 1 XXX" → display as XXX/JPY
CURRENCY_PAIRS = {
    # Major FX pairs
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    
    # JPY Cross Pairs (XXXJPY=X = how many JPY per 1 XXX)
    "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    "AUDJPY=X": "AUD/JPY",
    "NZDJPY=X": "NZD/JPY",
    "CADJPY=X": "CAD/JPY",
    "CHFJPY=X": "CHF/JPY",
    "SGDJPY=X": "SGD/JPY",
    "HKDJPY=X": "HKD/JPY",
    "ZARJPY=X": "ZAR/JPY",
    "MXNJPY=X": "MXN/JPY",
    "TRYJPY=X": "TRY/JPY",
    "SEKJPY=X": "SEK/JPY",
    "NOKJPY=X": "NOK/JPY",
    "DKKJPY=X": "DKK/JPY",
    "CNYJPY=X": "CNY/JPY",
    
    # Cryptocurrencies
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}

# JPY as base currency pairs (computed as inverse of XXX/JPY pairs)
# These show "how many XXX per 1 JPY" - useful for JPY strength analysis
JPY_BASE_PAIRS = {
    # Source ticker → Display name (will be computed as 1/rate)
    "USDJPY=X": "JPY/USD",
    "EURJPY=X": "JPY/EUR",
    "GBPJPY=X": "JPY/GBP",
    "AUDJPY=X": "JPY/AUD",
    "NZDJPY=X": "JPY/NZD",
    "CADJPY=X": "JPY/CAD",
    "CHFJPY=X": "JPY/CHF",
    "SGDJPY=X": "JPY/SGD",
    "HKDJPY=X": "JPY/HKD",
    "ZARJPY=X": "JPY/ZAR",
    "MXNJPY=X": "JPY/MXN",
    "TRYJPY=X": "JPY/TRY",
    "SEKJPY=X": "JPY/SEK",
    "NOKJPY=X": "JPY/NOK",
    "DKKJPY=X": "JPY/DKK",
    "CNYJPY=X": "JPY/CNY",
    "PLNJPY=X": "JPY/PLN",
}


# Thread-safe cache for market data
import threading
_market_data_cache: Dict[str, Tuple[datetime, Any]] = {}
_cache_lock = threading.Lock()  # Thread lock for cache operations
_yfinance_lock = threading.Lock()  # Thread lock for yfinance downloads (not thread-safe)


class _SuppressOutput:
    """Context manager to fully suppress stdout and stderr at file descriptor level."""
    def __enter__(self):
        import os
        import sys
        # Save the actual stdout/stderr file descriptors
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        # Open /dev/null
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        # Redirect stdout/stderr to /dev/null
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)
        return self
        
    def __exit__(self, *args):
        import os
        # Restore stdout/stderr
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        # Close the duplicated file descriptors
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._null_fd)


def _fetch_etf_data(
    start_date: str,
    end_date: Optional[str] = None
) -> Dict[str, pd.Series]:
    """Fetch key ETF data for market assessment (thread-safe)."""
    cache_key = f"etf_{start_date}_{end_date}"
    now = datetime.now()
    
    # Thread-safe: Lock the entire fetch operation to prevent duplicate work
    with _cache_lock:
        # Check cache first (inside lock to prevent race)
        if cache_key in _market_data_cache:
            cached_time, cached_data = _market_data_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_data
        
        # Not in cache, need to fetch
        try:
            import yfinance as yf
        except ImportError:
            warnings.warn("yfinance not available")
            return {}
        
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
        start = start_dt.strftime("%Y-%m-%d")
        
        etf_tickers = {
            'SPY': 'SPY',      # S&P 500
            'QQQ': 'QQQ',      # Nasdaq 100
            'IWM': 'IWM',      # Russell 2000
            'IWD': 'IWD',      # Russell 1000 Value
            'IWF': 'IWF',      # Russell 1000 Growth
            'VTI': 'VTI',      # Total Market
            'VIX': '^VIX',     # Volatility Index
            'DIA': 'DIA',      # Dow Jones Industrial Average
            # International Indexes
            'EFA': 'EFA',      # MSCI EAFE (Europe, Australasia, Far East)
            'EEM': 'EEM',      # MSCI Emerging Markets
            'VEU': 'VEU',      # FTSE All-World ex-US
            'INDA': 'INDA',    # MSCI India
            'FXI': 'FXI',      # China Large-Cap (FTSE China 50)
            'EWJ': 'EWJ',      # MSCI Japan
            'EWG': 'EWG',      # MSCI Germany
            'EWU': 'EWU',      # MSCI United Kingdom
            'EWZ': 'EWZ',      # MSCI Brazil
            'EWA': 'EWA',      # MSCI Australia
            'EWC': 'EWC',      # MSCI Canada
            # Sector ETFs (SPDR Select Sector)
            'XLK': 'XLK',      # Technology
            'XLF': 'XLF',      # Financials
            'XLV': 'XLV',      # Healthcare
            'XLY': 'XLY',      # Consumer Discretionary
            'XLP': 'XLP',      # Consumer Staples
            'XLE': 'XLE',      # Energy
            'XLI': 'XLI',      # Industrials
            'XLB': 'XLB',      # Materials
            'XLU': 'XLU',      # Utilities
            'XLRE': 'XLRE',    # Real Estate
            'XLC': 'XLC',      # Communication Services
        }
        
        result = {}
        
        for name, ticker in etf_tickers.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use Ticker object to avoid global state issues in yf.download
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(start=start, end=end, auto_adjust=True)
                
                # history() returns a simple DataFrame with Close column (not MultiIndex)
                if df is not None and not df.empty and 'Close' in df.columns:
                    series = df['Close'].dropna()
                    if len(series) > 20:
                        result[name] = series
            except Exception:
                logger.debug(f"Failed to fetch {ticker}: {e}")
        
        # Update cache (still inside lock)
        _market_data_cache[cache_key] = (now, result)
        return result


def _fetch_stock_sample_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    cache_key_prefix: str = "stocks"
) -> Dict[str, pd.Series]:
    """Fetch price data for a sample of stocks (thread-safe)."""
    cache_key = f"{cache_key_prefix}_{start_date}_{end_date}_{len(tickers)}"
    now = datetime.now()
    
    # Thread-safe cache check
    with _cache_lock:
        if cache_key in _market_data_cache:
            cached_time, cached_data = _market_data_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    result = {}
    
    # Batch download for efficiency (threads=False to avoid output issues)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Thread-safe yfinance download
            with _yfinance_lock:
                df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        
        if df is not None and not df.empty:
            # Handle multi-ticker DataFrame
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        if ('Close', ticker) in df.columns:
                            series = df[('Close', ticker)].dropna()
                            if len(series) > 20:
                                result[ticker] = series
                    except Exception:
                        pass
            else:
                # Single ticker case
                if len(tickers) == 1:
                    series = _extract_close_series(df, tickers[0])
                    if series is not None and len(series) > 20:
                        result[tickers[0]] = series
    except Exception:
        logger.debug(f"Batch download failed: {e}")
    
    # Thread-safe cache update
    with _cache_lock:
        _market_data_cache[cache_key] = (now, result)
    return result


def _compute_equity_forecasts(prices, vol_20d):
    """
    Compute equity/sector forecasts using elite multi-model ensemble.
    
    Returns: (fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, confidence)
    """
    try:
        if prices is None or len(prices) < 30:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Use elite ensemble forecast
        horizons = [1, 3, 7, 30, 90, 180, 365]
        result = ensemble_forecast(prices, horizons, "equity")
        return result
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"


def _compute_universe_metrics(
    name: str,
    weight: float,
    prices: pd.Series,
    ticker_count: int
) -> UniverseMetrics:
    """Compute metrics for a single universe segment."""
    if prices is None or len(prices) < 30:
        return UniverseMetrics(name=name, weight=weight, data_available=False, ticker_count=0)
    
    try:
        current_level = float(prices.iloc[-1])
        returns = _compute_returns(prices)
        
        vol_20d = float(prices.pct_change().dropna().iloc[-20:].std() * np.sqrt(252)) if len(prices) >= 20 else 0.0
        vol_pct = _compute_volatility_percentile(prices)
        vol_ratio, vol_inverted = _compute_vol_term_structure(prices)
        
        momentum = _compute_momentum_signal(returns['5d'], returns['21d'])
        
        vol_zscore = _compute_robust_zscore(prices.pct_change().rolling(20).std() * np.sqrt(252))
        
        stress = 0.0
        stress += min(max(vol_zscore, 0), 2.0) * 0.4
        stress += min(vol_pct, 1.0) * 0.3
        if vol_inverted:
            stress += 0.5
        if returns['21d'] < -0.10:
            stress += 0.3
        stress = min(stress, 2.0)
        fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_equity_forecasts(prices, vol_20d)
        
        return UniverseMetrics(
            name=name, weight=weight, current_level=current_level,
            return_1d=returns['1d'], return_5d=returns['5d'],
            return_21d=returns['21d'], return_63d=returns['63d'],
            volatility_20d=vol_20d, volatility_percentile=vol_pct,
            vol_term_structure_ratio=vol_ratio, vol_inverted=vol_inverted,
            stress_level=stress, stress_contribution=stress * weight,
            momentum_signal=momentum, data_available=True, ticker_count=ticker_count, 
            forecast_1d=fc_1d, forecast_3d=fc_3d, forecast_7d=fc_7d, forecast_30d=fc_30d, 
            forecast_90d=fc_90d, forecast_180d=fc_180d, forecast_365d=fc_365d, forecast_confidence=fc_conf,
        )
    except Exception:
        return UniverseMetrics(name=name, weight=weight, data_available=False, ticker_count=0)


def _compute_breadth_from_sample(stock_data: Dict[str, pd.Series]) -> MarketBreadthMetrics:
    """Compute market breadth from a sample of stocks."""
    if not stock_data or len(stock_data) < 10:
        return MarketBreadthMetrics()
    
    try:
        above_50ma = above_200ma = advances = declines = total = 0
        
        for ticker, prices in stock_data.items():
            if prices is None or len(prices) < 200:
                continue
            total += 1
            current = float(prices.iloc[-1])
            ma_50 = float(prices.iloc[-50:].mean())
            ma_200 = float(prices.iloc[-200:].mean())
            prev = float(prices.iloc[-2]) if len(prices) >= 2 else current
            
            if current > ma_50: above_50ma += 1
            if current > ma_200: above_200ma += 1
            if current > prev: advances += 1
            else: declines += 1
        
        if total == 0:
            return MarketBreadthMetrics()
        
        pct_50 = above_50ma / total
        pct_200 = above_200ma / total
        ad_ratio = advances / max(declines, 1)
        warning = pct_50 < BREADTH_WARNING_THRESHOLD
        thrust = pct_50 > 0.90 or pct_50 < 0.10
        
        if pct_50 < BREADTH_DANGER_THRESHOLD:
            interp = "Danger - Extreme weakness"
        elif warning:
            interp = "Warning - Narrowing breadth"
        elif thrust and pct_50 > 0.90:
            interp = "Thrust - Broad participation"
        else:
            interp = "Healthy - Normal breadth"
        
        return MarketBreadthMetrics(
            pct_above_50ma=pct_50, pct_above_200ma=pct_200,
            advance_decline_ratio=ad_ratio, breadth_thrust=thrust,
            breadth_warning=warning, interpretation=interp,
        )
    except Exception:
        return MarketBreadthMetrics()


def _compute_correlation_stress(stock_data: Dict[str, pd.Series]) -> CorrelationStress:
    """Compute cross-asset correlation stress."""
    if not stock_data or len(stock_data) < 5:
        return CorrelationStress()
    
    try:
        if USE_COPULA_CORRELATION and COPULA_CORRELATION_AVAILABLE:
            copula_result = compute_copula_correlation_stress(stock_data)
            return CorrelationStress(
                avg_correlation=copula_result.avg_correlation,
                max_correlation=copula_result.max_correlation,
                correlation_percentile=copula_result.correlation_percentile,
                systemic_risk_elevated=copula_result.systemic_risk_elevated,
                interpretation=copula_result.interpretation,
            )
        
        returns_df = pd.DataFrame({t: p.pct_change() for t, p in stock_data.items()}).dropna()
        if returns_df.empty or len(returns_df) < CORRELATION_LOOKBACK:
            return CorrelationStress()
        
        recent = returns_df.iloc[-CORRELATION_LOOKBACK:]
        corr_matrix = recent.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]
        correlations = correlations[~np.isnan(correlations)]
        
        if len(correlations) == 0:
            return CorrelationStress()
        
        avg_corr = float(np.mean(correlations))
        max_corr = float(np.max(correlations))
        systemic = avg_corr > CORRELATION_STRESS_THRESHOLD
        
        if systemic:
            interp = "High - Systemic risk elevated"
        elif avg_corr > 0.60:
            interp = "Elevated - Rising correlations"
        else:
            interp = "Low - Healthy dispersion"
        
        return CorrelationStress(
            avg_correlation=avg_corr, max_correlation=max_corr,
            correlation_percentile=0.5, systemic_risk_elevated=systemic,
            interpretation=interp,
        )
    except Exception:
        return CorrelationStress()


def _compute_crash_risk(
    universes: Dict[str, UniverseMetrics],
    breadth: MarketBreadthMetrics,
    correlation: CorrelationStress,
    temperature: float
) -> Tuple[float, str, int, List[str]]:
    """Compute crash risk probability."""
    risk = 0.0
    vol_inversions = []
    
    for name, univ in universes.items():
        if univ.vol_inverted:
            vol_inversions.append(name)
            risk += 0.15
    
    if breadth.pct_above_50ma < BREADTH_DANGER_THRESHOLD:
        risk += 0.20
    elif breadth.pct_above_50ma < BREADTH_WARNING_THRESHOLD:
        risk += 0.10
    
    if correlation.systemic_risk_elevated:
        risk += 0.15
    
    risk += temperature * 0.10
    risk = min(risk, 1.0)
    
    if risk > 0.50: level = "Extreme"
    elif risk > 0.30: level = "High"
    elif risk > 0.15: level = "Elevated"
    else: level = "Low"
    
    return risk, level, len(vol_inversions), vol_inversions if vol_inversions else None


def _determine_exit_signal(
    temperature: float,
    crash_risk_pct: float,
    breadth: MarketBreadthMetrics,
    correlation: CorrelationStress
) -> Tuple[bool, Optional[str]]:
    """Determine if exit signal should be triggered."""
    if temperature >= 1.8:
        return True, "Temperature extreme (>1.8)"
    if crash_risk_pct >= 0.60:
        return True, f"Crash risk critical ({crash_risk_pct:.0%})"
    if breadth.pct_above_50ma < 0.15:
        return True, f"Breadth collapse ({breadth.pct_above_50ma:.0%} above 50MA)"
    return False, None


def _compute_overall_momentum(universes: Dict[str, UniverseMetrics]) -> str:
    """Compute overall market momentum from universe segments."""
    if not universes:
        return "→ Neutral"
    
    weighted_momentum = total_weight = 0.0
    for univ in universes.values():
        if univ.data_available:
            weighted_momentum += univ.return_21d * univ.weight
            total_weight += univ.weight
    
    if total_weight == 0:
        return "→ Neutral"
    
    avg_momentum = weighted_momentum / total_weight
    return _compute_momentum_signal(avg_momentum, avg_momentum)


def _compute_sector_rotation(etf_data: Dict[str, pd.Series]) -> str:
    """Compute sector rotation signal from Growth vs Value performance."""
    if 'IWF' not in etf_data or 'IWD' not in etf_data:
        return "Normal"
    
    try:
        growth_ret = _compute_returns(etf_data['IWF'])['21d']
        value_ret = _compute_returns(etf_data['IWD'])['21d']
        spread = growth_ret - value_ret
        
        if spread > 0.05: return "Growth Leading"
        elif spread < -0.05: return "Value Leading"
        else: return "Normal"
    except Exception:
        return "Normal"


def _compute_sector_metrics(etf_data: Dict[str, pd.Series]) -> Dict[str, SectorMetrics]:
    """Compute metrics for each sector ETF."""
    sectors = {}
    
    for ticker, name in SECTOR_ETFS.items():
        if ticker not in etf_data:
            sectors[name] = SectorMetrics(name=name, ticker=ticker, data_available=False)
            continue
        
        prices = etf_data[ticker]
        if prices is None or len(prices) < 30:
            sectors[name] = SectorMetrics(name=name, ticker=ticker, data_available=False)
            continue
        
        try:
            returns = _compute_returns(prices)
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            vol_pct = _compute_volatility_percentile(prices)
            momentum = _compute_momentum_signal(returns.get('5d', 0), returns.get('21d', 0))
            
            vol_pts = min(vol_pct, 1.0) * 50
            move_pts = min(abs(returns.get('5d', 0)) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_equity_forecasts(prices, vol_20d)
            
            sectors[name] = SectorMetrics(
                name=name, ticker=ticker,
                return_1d=returns.get('1d', 0), return_5d=returns.get('5d', 0),
                return_21d=returns.get('21d', 0), volatility_20d=vol_20d,
                volatility_percentile=vol_pct, momentum_signal=momentum,
                risk_score=risk_score, data_available=True, forecast_1d=fc_1d, forecast_3d=fc_3d, forecast_7d=fc_7d, forecast_30d=fc_30d, forecast_90d=fc_90d, forecast_180d=fc_180d, forecast_365d=fc_365d, forecast_confidence=fc_conf,
            )
        except Exception:
            sectors[name] = SectorMetrics(name=name, ticker=ticker, data_available=False)
    
    return sectors


def _fetch_currency_data(start_date: str, end_date: Optional[str] = None) -> Dict[str, pd.Series]:
    """Fetch currency pair data."""
    cache_key = f"currency_{start_date}_{end_date}"
    now = datetime.now()
    
    with _cache_lock:
        if cache_key in _market_data_cache:
            cached_time, cached_data = _market_data_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    result = {}
    for ticker in CURRENCY_PAIRS.keys():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start=start, end=end, auto_adjust=True)
            
            if df is not None and not df.empty and 'Close' in df.columns:
                series = df['Close'].dropna()
                if len(series) > 20:
                    result[ticker] = series
        except Exception:
            pass
    
    with _cache_lock:
        _market_data_cache[cache_key] = (now, result)
    return result


def _prophet_forecast(prices: pd.Series, horizons: list) -> list:
    """Use Facebook Prophet for time series forecasting with trend + seasonality."""
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': prices.index,
            'y': prices.values
        })
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Configure Prophet
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        max_horizon = max(horizons)
        future = model.make_future_dataframe(periods=max_horizon)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(future)
        
        current_price = float(prices.iloc[-1])
        current_idx = len(prices) - 1
        
        forecasts = []
        for h in horizons:
            future_idx = current_idx + h
            if future_idx < len(forecast):
                future_price = float(forecast.iloc[future_idx]['yhat'])
                pct_change = ((future_price / current_price) - 1) * 100
                forecasts.append(pct_change)
            else:
                forecasts.append(0.0)
        
        return forecasts
        
    except Exception:
        return [0.0] * len(horizons)


def _lstm_forecast(prices: pd.Series, horizons: list) -> list:
    """Use PyTorch LSTM for pattern-based forecasting."""
    try:
        import torch
        import torch.nn as nn
        
        returns = prices.pct_change().dropna().values
        if len(returns) < 100:
            return [0.0] * len(horizons)
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-8:
            return [0.0] * len(horizons)
        normalized = (returns - mean_ret) / std_ret
        
        seq_length = 20
        X, y = [], []
        for i in range(len(normalized) - seq_length):
            X.append(normalized[i:i+seq_length])
            y.append(normalized[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 50:
            return [0.0] * len(horizons)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)
        
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        
        model = SimpleLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        forecasts = []
        current_seq = normalized[-seq_length:].copy()
        
        cumulative_return = 0.0
        for h in range(1, max(horizons) + 1):
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
                pred = model(seq_tensor).item()
            
            pred_return = pred * std_ret + mean_ret
            cumulative_return += pred_return
            
            if h in horizons:
                forecasts.append(cumulative_return * 100)
            
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred
        
        return forecasts
        
    except Exception:
        return [0.0] * len(horizons)


def _classical_forecast(prices: pd.Series, log_returns: pd.Series, horizons: list) -> list:
    """
    Classical mean reversion + momentum forecast.
    This ALWAYS produces non-zero forecasts based on drift and momentum.
    
    Returns forecasts in PERCENTAGE POINTS (e.g., 2.5 means +2.5%).
    """
    try:
        # Compute mean daily return (drift)
        daily_drift = float(log_returns.mean())
        # Clamp drift to reasonable FX range (-1% to +1% daily)
        daily_drift = float(np.clip(daily_drift, -0.01, 0.01))
        
        # Get current price and MA
        current_price = float(prices.iloc[-1])
        ma_len = min(200, len(prices) - 1)
        ma_price = float(prices.iloc[-ma_len:].mean())
        
        # Deviation from moving average
        deviation = (current_price - ma_price) / ma_price if ma_price > 0 else 0.0
        # Clamp deviation
        deviation = float(np.clip(deviation, -0.30, 0.30))
        
        # Recent momentum
        mom_5d = float(log_returns.iloc[-5:].mean()) if len(log_returns) >= 5 else daily_drift
        mom_5d = float(np.clip(mom_5d, -0.02, 0.02))
        
        # Volatility for bounds
        vol = float(log_returns.std())
        vol = float(np.clip(vol, 0.003, 0.05))  # 0.3% to 5% daily
        
        forecasts = []
        for h in horizons:
            # Drift contribution (trend)
            drift = daily_drift * h
            
            # Mean reversion contribution  
            mr_coef = 0.15 * min(h / 90.0, 1.0)  # Max 15% pull over 90 days
            mr = -deviation * mr_coef
            
            # Momentum contribution (decays with horizon)
            if h <= 30:
                mom = mom_5d * min(h, 21) * 0.2
            else:
                mom = 0.0
            
            # Total log return
            total = drift + mr + mom
            
            # Convert to percentage
            pct = (np.exp(total) - 1) * 100
            
            # Apply volatility-based bounds (2 sigma)
            horizon_vol = vol * np.sqrt(h)
            max_pct = min(horizon_vol * 2.0 * 100, 30.0)  # Never more than +/-30%
            max_pct = max(max_pct, 0.5)   # At least +/-0.5%
            
            pct = float(np.clip(pct, -max_pct, max_pct))
            forecasts.append(pct)
        
        return forecasts
        
    except Exception:
        # Emergency fallback: simple drift
        try:
            drift = float(log_returns.mean())
            drift = float(np.clip(drift, -0.01, 0.01))
            forecasts = []
            for h in horizons:
                pct = (np.exp(drift * h) - 1) * 100
                pct = float(np.clip(pct, -30.0, 30.0))
                forecasts.append(pct)
            return forecasts
        except:
            return [0.01] * len(horizons)  # Non-zero default


def _compute_forecast_confidence(
    prophet_fc: list, lstm_fc: list, classical_fc: list,
    vol_20d: float, data_len: int
) -> str:
    """Compute confidence based on model agreement and data quality."""
    try:
        agreement_scores = []
        for i in range(min(len(prophet_fc), len(lstm_fc), len(classical_fc))):
            signs = [
                np.sign(prophet_fc[i]) if prophet_fc[i] != 0 else 0,
                np.sign(lstm_fc[i]) if lstm_fc[i] != 0 else 0,
                np.sign(classical_fc[i]) if classical_fc[i] != 0 else 0,
            ]
            # Count agreement (all same sign = 1.0, 2/3 same = 0.67, all different = 0)
            if signs[0] == signs[1] == signs[2] and signs[0] != 0:
                agreement_scores.append(1.0)
            elif signs[0] == signs[1] or signs[1] == signs[2] or signs[0] == signs[2]:
                agreement_scores.append(0.67)
            else:
                agreement_scores.append(0.0)
        
        agreement = np.mean(agreement_scores) if agreement_scores else 0.0
        
        # Volatility penalty
        vol_score = 1 - min(vol_20d / 0.20, 1.0)
        
        # Data quality
        data_score = min(data_len / 500, 1.0)
        
        # Combined confidence
        confidence_score = agreement * 0.4 + vol_score * 0.3 + data_score * 0.3
        
        if confidence_score > 0.7:
            return "High"
        elif confidence_score > 0.4:
            return "Medium"
        else:
            return "Low"
    except Exception:
        return "Low"


def _compute_currency_forecasts(prices: pd.Series, vol_20d: float) -> Tuple[float, float, float, float, float, float, float, str]:
    """
    Compute currency forecasts using elite multi-model ensemble.
    
    Models:
    1. Kalman Filter (drift estimation)
    2. GARCH (volatility-adjusted)
    3. Ornstein-Uhlenbeck (mean reversion)
    4. Momentum (multi-timeframe)
    5. Classical (baseline)
    
    Returns: (1d, 3d, 7d, 30d, 90d, 180d, 365d forecasts, confidence level)
    """
    try:
        if prices is None or len(prices) < 30:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Ensure we have enough data for ensemble_forecast (requires 60+ points)
        if len(prices) < 60:
            # Use simple momentum-based forecast for limited data
            log_returns = np.log(prices / prices.shift(1)).dropna()
            if len(log_returns) < 5:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
            
            daily_drift = float(log_returns.mean())
            vol = float(log_returns.std())
            
            forecasts = []
            for h in STANDARD_HORIZONS:
                fc = daily_drift * h * np.exp(-h / 180.0)
                pct = (np.exp(fc) - 1) * 100
                # Clamp to reasonable bounds for currencies
                max_pct = min(vol * np.sqrt(h) * 3 * 100, {1: 3, 3: 5, 7: 7, 30: 12, 90: 18, 180: 25, 365: 35}.get(h, 20))
                pct = float(np.clip(pct, -max_pct, max_pct))
                forecasts.append(pct)
            
            return tuple(forecasts) + ("Low",)
        
        # Use elite ensemble forecast (horizons ignored, uses STANDARD_HORIZONS)
        result = ensemble_forecast(prices, asset_type="currency")
        return result
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"


def _compute_currency_metrics(currency_data: Dict[str, pd.Series]) -> Dict[str, CurrencyMetrics]:
    """Compute metrics for each currency pair including forecasts and JPY base pairs."""
    currencies = {}
    
    # First, compute standard pairs (XXX/JPY format)
    for ticker, pair_name in CURRENCY_PAIRS.items():
        if ticker not in currency_data:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        prices = currency_data[ticker]
        if prices is None or len(prices) < 30:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        try:
            rate = float(prices.iloc[-1])
            returns = _compute_returns(prices)
            ret_1d = returns.get('1d', 0.0)
            ret_5d = returns.get('5d', 0.0)
            ret_21d = returns.get('21d', 0.0)
            
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            
            momentum = _compute_momentum_signal(ret_5d, ret_21d)
            
            # Risk score: vol (0-50) + recent moves (0-50)
            vol_pts = min(vol_20d / 0.15, 1.0) * 50
            move_pts = min(abs(ret_5d) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            
            # Compute forecasts
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_currency_forecasts(prices, vol_20d)
            
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                rate=rate,
                return_1d=ret_1d,
                return_5d=ret_5d,
                return_21d=ret_21d,
                volatility_20d=vol_20d,
                momentum_signal=momentum,
                risk_score=risk_score,
                data_available=True,
                forecast_1d=fc_1d,
                forecast_3d=fc_3d,
                forecast_7d=fc_7d,
                forecast_30d=fc_30d,
                forecast_90d=fc_90d,
                forecast_180d=fc_180d,
                forecast_365d=fc_365d,
                forecast_confidence=fc_conf,
                is_inverse=False,
            )
        except Exception:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
    
    # Second, compute JPY base pairs (JPY/XXX = inverse of XXX/JPY)
    for ticker, pair_name in JPY_BASE_PAIRS.items():
        if ticker not in currency_data:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
                is_inverse=True,
            )
            continue
        
        prices = currency_data[ticker]
        if prices is None or len(prices) < 30:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
                is_inverse=True,
            )
            continue
        
        try:
            # Inverse the prices for JPY/XXX (1 JPY = ? XXX)
            inverse_prices = 1.0 / prices
            
            rate = float(inverse_prices.iloc[-1])
            
            # Returns for inverse pair are NEGATIVE of original
            returns = _compute_returns(prices)
            ret_1d = -returns.get('1d', 0.0)
            ret_5d = -returns.get('5d', 0.0)
            ret_21d = -returns.get('21d', 0.0)
            
            # Volatility is the same (symmetric)
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            
            # Momentum is inverted
            momentum = _compute_momentum_signal(ret_5d, ret_21d)
            
            # Risk score remains the same
            vol_pts = min(vol_20d / 0.15, 1.0) * 50
            move_pts = min(abs(ret_5d) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            
            # Compute forecasts for inverse pair (negate the forecasts)
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_currency_forecasts(prices, vol_20d)
            # Inverse forecasts
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d = -fc_1d, -fc_3d, -fc_7d, -fc_30d, -fc_90d, -fc_180d, -fc_365d
            
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                rate=rate,
                return_1d=ret_1d,
                return_5d=ret_5d,
                return_21d=ret_21d,
                volatility_20d=vol_20d,
                momentum_signal=momentum,
                risk_score=risk_score,
                data_available=True,
                forecast_1d=fc_1d,
                forecast_3d=fc_3d,
                forecast_7d=fc_7d,
                forecast_30d=fc_30d,
                forecast_90d=fc_90d,
                forecast_180d=fc_180d,
                forecast_365d=fc_365d,
                forecast_confidence=fc_conf,
                is_inverse=True,
            )
        except Exception:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
                is_inverse=True,
            )
    
    return currencies


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_market_temperature(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> MarketTemperatureResult:
    """
    Compute comprehensive market temperature.
    
    This is the main entry point for market risk assessment
    
    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)
        
    Returns:
        MarketTemperatureResult with complete market assessment
    """
    # Fetch ETF data
    etf_data = _fetch_etf_data(start_date, end_date)
    
    # Fetch stock samples
    top100_data = _fetch_stock_sample_data(
        TOP_100_TICKERS[:50],  # Use first 50 for speed
        start_date, end_date,
        "top100"
    )
    
    sp500_data = _fetch_stock_sample_data(
        SP500_SAMPLE,
        start_date, end_date,
        "sp500"
    )
    
    russell_data = _fetch_stock_sample_data(
        RUSSELL_2000_SAMPLE[:20],  # Use first 20 for speed
        start_date, end_date,
        "russell"
    )
    
    # Compute universe metrics
    universes = {}
    
    # Mega-cap (Top 100) - use aggregate of sample
    if top100_data:
        # Create equal-weighted index
        sample_returns = pd.DataFrame({
            t: p.pct_change() for t, p in top100_data.items()
        }).dropna()
        if not sample_returns.empty:
            # Build cumulative index
            equal_weight_return = sample_returns.mean(axis=1)
            index = (1 + equal_weight_return).cumprod() * 100
            universes["Mega-Cap 100"] = _compute_universe_metrics(
                "Mega-Cap 100", MEGA_CAP_WEIGHT, index, len(top100_data)
            )
    
    # Broad Market (S&P 500 via SPY)
    if 'SPY' in etf_data:
        universes["S&P 500"] = _compute_universe_metrics(
            "S&P 500", BROAD_MARKET_WEIGHT, etf_data['SPY'], 500
        )
    
    # Small Cap (Russell 2000 via IWM)
    if 'IWM' in etf_data:
        universes["Russell 2000"] = _compute_universe_metrics(
            "Russell 2000", SMALL_CAP_WEIGHT, etf_data['IWM'], 2000
        )
    
    # Growth vs Value
    if 'QQQ' in etf_data:
        universes["Growth (QQQ)"] = _compute_universe_metrics(
            "Growth (QQQ)", GROWTH_VALUE_WEIGHT, etf_data['QQQ'], 100
        )
    
    # Dow Jones
    if 'DIA' in etf_data:
        universes["Dow Jones"] = _compute_universe_metrics("Dow Jones", INTERNATIONAL_WEIGHT, etf_data['DIA'], 30)
    
    # MSCI EAFE
    if 'EFA' in etf_data:
        universes["MSCI EAFE"] = _compute_universe_metrics("MSCI EAFE", INTERNATIONAL_WEIGHT, etf_data['EFA'], 900)
    
    # Emerging Markets
    if 'EEM' in etf_data:
        universes["Emerging Mkts"] = _compute_universe_metrics("Emerging Mkts", INTERNATIONAL_WEIGHT, etf_data['EEM'], 1400)
    
    # World ex-US
    if 'VEU' in etf_data:
        universes["World ex-US"] = _compute_universe_metrics("World ex-US", INTERNATIONAL_WEIGHT, etf_data['VEU'], 3700)
    
    # Japan
    if 'EWJ' in etf_data:
        universes["Japan"] = _compute_universe_metrics("Japan", INTERNATIONAL_WEIGHT, etf_data['EWJ'], 300)
    
    # China
    if 'FXI' in etf_data:
        universes["China"] = _compute_universe_metrics("China", INTERNATIONAL_WEIGHT, etf_data['FXI'], 50)
    
    # India
    if 'INDA' in etf_data:
        universes["India"] = _compute_universe_metrics("India", INTERNATIONAL_WEIGHT, etf_data['INDA'], 100)
    
    # Germany
    if 'EWG' in etf_data:
        universes["Germany"] = _compute_universe_metrics("Germany", INTERNATIONAL_WEIGHT, etf_data['EWG'], 60)
    
    # UK
    if 'EWU' in etf_data:
        universes["UK"] = _compute_universe_metrics("UK", INTERNATIONAL_WEIGHT, etf_data['EWU'], 100)
    
    # Brazil
    if 'EWZ' in etf_data:
        universes["Brazil"] = _compute_universe_metrics("Brazil", INTERNATIONAL_WEIGHT, etf_data['EWZ'], 50)
    
    # Australia
    if 'EWA' in etf_data:
        universes["Australia"] = _compute_universe_metrics("Australia", INTERNATIONAL_WEIGHT, etf_data['EWA'], 70)
    
    # Canada
    if 'EWC' in etf_data:
        universes["Canada"] = _compute_universe_metrics("Canada", INTERNATIONAL_WEIGHT, etf_data['EWC'], 90)
    
    # Compute breadth from all available stocks
    all_stocks = {**top100_data, **sp500_data}
    breadth = _compute_breadth_from_sample(all_stocks)
    
    # Compute correlation stress
    correlation = _compute_correlation_stress(all_stocks)
    
    # Aggregate temperature
    total_contribution = sum(u.stress_contribution for u in universes.values())
    total_weight = sum(u.weight for u in universes.values() if u.data_available)
    
    if total_weight > 0:
        temperature = total_contribution / total_weight
    else:
        temperature = 1.0  # Default to elevated if no data
    
    # Add correlation and breadth stress
    if correlation.systemic_risk_elevated:
        temperature += 0.3
    if breadth.breadth_warning:
        temperature += 0.2
    
    temperature = max(TEMP_MIN, min(TEMP_MAX, temperature))
    
    # Compute scale factor
    scale_factor = _compute_scale_factor(temperature)
    
    # Status and action
    status, action_text = _get_status_and_action(temperature)
    
    # Crash risk
    crash_risk_pct, crash_risk_level, vol_inversion_count, inverted_universes = \
        _compute_crash_risk(universes, breadth, correlation, temperature)
    
    # Exit signal
    exit_signal, exit_reason = _determine_exit_signal(
        temperature, crash_risk_pct, breadth, correlation
    )
    
    # Momentum and rotation
    overall_momentum = _compute_overall_momentum(universes)
    sector_rotation = _compute_sector_rotation(etf_data)
    
    # Compute sector-by-sector metrics
    sectors = _compute_sector_metrics(etf_data)
    
    # Compute currency pair metrics
    currency_data = _fetch_currency_data(start_date, end_date)
    currencies = _compute_currency_metrics(currency_data)
    
    # Data quality
    data_quality = sum(1 for u in universes.values() if u.data_available) / max(1, len(universes))
    
    return MarketTemperatureResult(
        temperature=temperature,
        scale_factor=scale_factor,
        universes=universes,
        breadth=breadth,
        correlation=correlation,
        computed_at=datetime.now().isoformat(),
        data_quality=data_quality,
        status=status,
        action_text=action_text,
        crash_risk_pct=crash_risk_pct,
        crash_risk_level=crash_risk_level,
        vol_inversion_count=vol_inversion_count,
        inverted_universes=inverted_universes,
        overall_momentum=overall_momentum,
        sector_rotation_signal=sector_rotation,
        exit_signal=exit_signal,
        exit_reason=exit_reason,
        sectors=sectors,
        currencies=currencies,
    )


# =============================================================================
# RENDERING
# =============================================================================

def render_market_temperature(result: MarketTemperatureResult, console=None) -> None:
    """Render market temperature with premium Apple-quality UX."""
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
    
    if console is None:
        console = Console()
    
    # Temperature-based styling
    if result.temperature >= 1.5:
        temp_style = "bold red"
        status_color = "red"
        emoji = "🚨"
    elif result.temperature >= 1.0:
        temp_style = "bold orange1"
        status_color = "orange1"
        emoji = "🔥"
    elif result.temperature >= 0.7:
        temp_style = "bold yellow"
        status_color = "yellow"
        emoji = "⚠️"
    elif result.temperature >= 0.5:
        temp_style = "yellow"
        status_color = "yellow"
        emoji = "📊"
    else:
        temp_style = "bright_green"
        status_color = "bright_green"
        emoji = "✅"
    
    # Print clear separator and header
    console.print()
    console.print("  " + "═" * 76)
    
    header = Text()
    header.append(f"  {emoji} ", style="")
    header.append("MARKET TEMPERATURE", style="bold white")
    header.append("  —  ", style="dim")
    header.append(f"{result.temperature:.2f}", style=temp_style)
    header.append(f"  {result.status.upper()}", style=f"bold {status_color}")
    console.print(header)
    
    # Action text
    if result.exit_signal:
        action_line = Text()
        action_line.append("  🚨 EXIT SIGNAL: ", style="bold red")
        action_line.append(result.exit_reason or "", style="red")
        console.print(action_line)
    else:
        action_line = Text()
        action_line.append("  → ", style="dim")
        action_line.append(result.action_text, style="dim italic")
        console.print(action_line)
    
    console.print("  " + "═" * 76)
    console.print()
    
    # Universe breakdown
    console.print("  [dim]Universe Segments[/dim]")
    console.print()
    
    header_line = Text()
    header_line.append("  ")
    header_line.append("SEGMENT".ljust(18), style="bold dim")
    header_line.append("LEVEL".rjust(10), style="bold dim")
    header_line.append("1D".rjust(8), style="bold dim")
    header_line.append("5D".rjust(8), style="bold dim")
    header_line.append("21D".rjust(8), style="bold dim")
    header_line.append("VOL".rjust(8), style="bold dim")
    header_line.append("MOMENTUM".ljust(12), style="bold dim")
    console.print(header_line)
    console.print("  " + "─" * 78)
    
    for name, univ in result.universes.items():
        line = Text()
        line.append("  ")
        
        # Name with stress indicator
        if univ.stress_level > 1.0:
            line.append(f"🔴 {name}".ljust(18), style="red")
        elif univ.stress_level > 0.5:
            line.append(f"🟡 {name}".ljust(18), style="yellow")
        else:
            line.append(f"🟢 {name}".ljust(18), style="green")
        
        # Level
        if univ.current_level:
            line.append(f"{univ.current_level:,.0f}".rjust(10), style="white")
        else:
            line.append("—".rjust(10), style="dim")
        
        # Returns
        for ret in [univ.return_1d, univ.return_5d, univ.return_21d]:
            ret_style = "bright_green" if ret >= 0 else "indian_red1"
            line.append(f"{ret:+.1%}".rjust(8), style=ret_style)
        
        # Volatility
        vol_style = "red" if univ.volatility_percentile > 0.8 else ("yellow" if univ.volatility_percentile > 0.6 else "white")
        line.append(f"{univ.volatility_20d:.0%}".rjust(8), style=vol_style)
        
        # Momentum
        if "Strong" in univ.momentum_signal and "↑" in univ.momentum_signal:
            mom_style = "bold bright_green"
        elif "Rising" in univ.momentum_signal:
            mom_style = "bright_green"
        elif "Weak" in univ.momentum_signal or "↓" in univ.momentum_signal:
            mom_style = "bold indian_red1"
        elif "Falling" in univ.momentum_signal:
            mom_style = "indian_red1"
        else:
            mom_style = "dim"
        line.append(f"  {univ.momentum_signal}", style=mom_style)
        
        console.print(line)
    
    console.print()
    
    # Market Breadth
    console.print("  [dim]Market Breadth[/dim]")
    console.print()
    
    breadth_line = Text()
    breadth_line.append("  ")
    breadth_line.append("Above 50MA   ", style="dim")
    
    pct_50ma = result.breadth.pct_above_50ma
    if pct_50ma < BREADTH_DANGER_THRESHOLD:
        b_style = "bold red"
    elif pct_50ma < BREADTH_WARNING_THRESHOLD:
        b_style = "yellow"
    else:
        b_style = "green"
    breadth_line.append(f"{pct_50ma:.0%}", style=b_style)
    
    breadth_line.append("   Above 200MA   ", style="dim")
    breadth_line.append(f"{result.breadth.pct_above_200ma:.0%}", style="white")
    
    breadth_line.append("   A/D Ratio   ", style="dim")
    ad_style = "green" if result.breadth.advance_decline_ratio > 1 else "red"
    breadth_line.append(f"{result.breadth.advance_decline_ratio:.2f}", style=ad_style)
    console.print(breadth_line)
    
    interp_line = Text()
    interp_line.append("  ")
    interp_line.append(result.breadth.interpretation, style="dim italic")
    console.print(interp_line)
    
    console.print()
    
    # Correlation Stress
    console.print("  [dim]Correlation Stress[/dim]")
    console.print()
    
    corr_line = Text()
    corr_line.append("  ")
    corr_line.append("Avg Correlation   ", style="dim")
    
    if result.correlation.systemic_risk_elevated:
        c_style = "bold red"
    elif result.correlation.avg_correlation > 0.60:
        c_style = "yellow"
    else:
        c_style = "green"
    corr_line.append(f"{result.correlation.avg_correlation:.0%}", style=c_style)
    
    corr_line.append("   ", style="")
    corr_line.append(result.correlation.interpretation, style="dim italic")
    console.print(corr_line)
    
    console.print()
    
    # Crash Risk Panel
    if result.crash_risk_pct > 0.05:
        console.print("  [dim]Crash Risk Assessment[/dim]")
        console.print()
        
        # Risk gauge
        gauge_width = 40
        filled = int(min(1.0, result.crash_risk_pct / 0.50) * gauge_width)
        
        gauge_line = Text()
        gauge_line.append("  ")
        for i in range(gauge_width):
            if i < filled:
                if i < gauge_width * 0.4:
                    gauge_line.append("━", style="green")
                elif i < gauge_width * 0.7:
                    gauge_line.append("━", style="yellow")
                else:
                    gauge_line.append("━", style="red")
            else:
                gauge_line.append("─", style="bright_black")
        console.print(gauge_line)
        
        risk_line = Text()
        risk_line.append("  ")
        
        if result.crash_risk_level == "Extreme":
            r_style = "bold red"
        elif result.crash_risk_level == "High":
            r_style = "red"
        elif result.crash_risk_level == "Elevated":
            r_style = "yellow"
        else:
            r_style = "green"
        
        risk_line.append(f"{result.crash_risk_pct:.0%}", style=r_style)
        risk_line.append(f"  {result.crash_risk_level.upper()}", style=r_style)
        
        if result.vol_inversion_count > 0:
            risk_line.append(f"  ({result.vol_inversion_count} vol inversions)", style="dim")
        console.print(risk_line)
        
        console.print()
    
    # Summary metrics
    console.print("  [dim]Summary[/dim]")
    console.print()
    
    summary_line = Text()
    summary_line.append("  ")
    summary_line.append("Position Scale   ", style="dim")
    
    scale = result.scale_factor
    if scale > 0.9:
        s_style = "green"
    elif scale > 0.6:
        s_style = "yellow"
    else:
        s_style = "red"
    summary_line.append(f"{scale:.0%}", style=s_style)
    
    summary_line.append("   Momentum   ", style="dim")
    summary_line.append(result.overall_momentum, style="white")
    
    summary_line.append("   Rotation   ", style="dim")
    summary_line.append(result.sector_rotation_signal, style="white")
    console.print(summary_line)
    
    qual_line = Text()
    qual_line.append("  ")
    qual_line.append("Data Quality   ", style="dim")
    qual_line.append(f"{result.data_quality:.0%}", style="green" if result.data_quality > 0.8 else "yellow")
    qual_line.append(f"   Computed   ", style="dim")
    qual_line.append(result.computed_at[:19], style="dim italic")
    console.print(qual_line)
    
    console.print()
    console.print()


# =============================================================================
# STANDALONE CLI
# =============================================================================

if __name__ == "__main__":
    """Run market temperature computation and display."""
    import argparse
    import os
    import sys
    import time
    from rich.console import Console
    
    # Suppress yfinance output at environment level
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['YF_LOG_LEVEL'] = 'CRITICAL'
    
    # Suppress yfinance warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)
    
    parser = argparse.ArgumentParser(description="Market Temperature — US Equity Market Assessment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--start", default="2020-01-01", help="Start date for historical data")
    args = parser.parse_args()
    
    # Compute market temperature with full output suppression
    with _SuppressOutput():
        result = compute_market_temperature(start_date=args.start)
    
    # Wait for any lingering threads to complete their output
    time.sleep(1.0)
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Clear the line with carriage return and spaces
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()
    
    console = Console()
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        render_market_temperature(result, console=console)
