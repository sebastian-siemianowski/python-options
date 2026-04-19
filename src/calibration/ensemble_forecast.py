"""
Epic 24: Ensemble Forecast Combination (Beyond BMA)

Provides alternative forecast combination methods to complement BMA:
1. Equal-weight ensemble (permanent BMA benchmark)
2. Trimmed ensemble (outlier-robust)
3. Online prediction pool (adaptive weights with regret bounds)

References:
- Geweke & Amisano (2011): Optimal prediction pools
- Cesa-Bianchi & Lugosi (2006): Prediction, Learning, and Games
- Timmermann (2006): Forecast Combinations (Handbook of Forecasting)
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 24.1: Equal-weight ensemble
MIN_MODELS_FOR_ENSEMBLE = 2

# Story 24.2: Trimmed ensemble
DEFAULT_TRIM_FRAC = 0.1
MIN_MODELS_AFTER_TRIM = 2

# Story 24.3: Online prediction pool
DEFAULT_ETA = 0.1  # Learning rate for exponentiated gradient
MIN_ETA = 1e-6
MAX_ETA = 10.0
WEIGHT_FLOOR = 1e-8  # Prevent numerical underflow in weights


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EqualWeightResult:
    """Result of equal-weight ensemble forecast combination."""
    forecast: float
    variance: float
    n_models: int
    model_spread: float  # max - min of individual forecasts
    individual_forecasts: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast": float(self.forecast),
            "variance": float(self.variance),
            "n_models": self.n_models,
            "model_spread": float(self.model_spread),
        }


@dataclass
class TrimmedEnsembleResult:
    """Result of trimmed ensemble forecast combination."""
    forecast: float
    variance: float
    n_models_original: int
    n_models_used: int
    n_trimmed: int
    trim_frac: float
    trimmed_indices_low: np.ndarray  # Indices of trimmed low forecasts
    trimmed_indices_high: np.ndarray  # Indices of trimmed high forecasts
    model_spread_original: float
    model_spread_trimmed: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast": float(self.forecast),
            "variance": float(self.variance),
            "n_models_original": self.n_models_original,
            "n_models_used": self.n_models_used,
            "n_trimmed": self.n_trimmed,
            "trim_frac": float(self.trim_frac),
            "model_spread_original": float(self.model_spread_original),
            "model_spread_trimmed": float(self.model_spread_trimmed),
        }


@dataclass
class OnlinePredictionPoolResult:
    """Result of online prediction pool weight update."""
    weights: np.ndarray  # Current model weights (sum to 1)
    cumulative_loss: float  # Cumulative loss of the pool
    best_expert_loss: float  # Cumulative loss of the best single expert
    regret: float  # cumulative_loss - best_expert_loss
    regret_bound: float  # Theoretical regret bound sqrt(T ln M)
    n_models: int
    n_timesteps: int
    weight_history: np.ndarray  # (T, M) weight evolution
    best_expert_index: int
    eta: float
    forecast: float  # Weighted forecast at final timestep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "cumulative_loss": float(self.cumulative_loss),
            "best_expert_loss": float(self.best_expert_loss),
            "regret": float(self.regret),
            "regret_bound": float(self.regret_bound),
            "n_models": self.n_models,
            "n_timesteps": self.n_timesteps,
            "best_expert_index": self.best_expert_index,
            "eta": float(self.eta),
            "forecast": float(self.forecast),
        }


@dataclass
class EnsembleComparisonResult:
    """Comparison between BMA and equal-weight ensemble."""
    bma_crps: float
    ew_crps: float
    delta_crps: float  # BMA - EW (positive means BMA wins)
    bma_wins: bool
    flag_investigation: bool  # True if BMA loses (possible overfitting)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bma_crps": float(self.bma_crps),
            "ew_crps": float(self.ew_crps),
            "delta_crps": float(self.delta_crps),
            "bma_wins": self.bma_wins,
            "flag_investigation": self.flag_investigation,
        }


# ---------------------------------------------------------------------------
# Story 24.1: Equal-Weight Ensemble
# ---------------------------------------------------------------------------

def equal_weight_ensemble(
    model_forecasts: np.ndarray,
) -> EqualWeightResult:
    """
    Compute equal-weight ensemble forecast (simple average).

    This is the permanent BMA benchmark. BMA must beat this on 60%+ of
    assets to justify its complexity.

    Parameters
    ----------
    model_forecasts : array-like, shape (M,) or (M, H)
        Individual model forecasts. M = number of models.
        If 2D, each row is a model, each column is a horizon.
        For single-horizon, use 1D array.

    Returns
    -------
    EqualWeightResult
        Combined forecast and diagnostics.

    Raises
    ------
    ValueError
        If fewer than MIN_MODELS_FOR_ENSEMBLE models provided.
    """
    forecasts = np.asarray(model_forecasts, dtype=np.float64)

    # Handle NaN: exclude NaN models
    if forecasts.ndim == 1:
        valid_mask = np.isfinite(forecasts)
        valid_forecasts = forecasts[valid_mask]
    elif forecasts.ndim == 2:
        # For 2D, exclude rows (models) that have any NaN
        valid_mask = np.all(np.isfinite(forecasts), axis=1)
        valid_forecasts = forecasts[valid_mask]
    else:
        raise ValueError(f"model_forecasts must be 1D or 2D, got {forecasts.ndim}D")

    n_valid = len(valid_forecasts)
    if n_valid < MIN_MODELS_FOR_ENSEMBLE:
        raise ValueError(
            f"Need >= {MIN_MODELS_FOR_ENSEMBLE} valid models, got {n_valid}"
        )

    # Equal-weight average
    if valid_forecasts.ndim == 1:
        forecast = float(np.mean(valid_forecasts))
        variance = float(np.var(valid_forecasts, ddof=1)) if n_valid > 1 else 0.0
        spread = float(np.max(valid_forecasts) - np.min(valid_forecasts))
    else:
        # Multi-horizon: average across models (axis=0)
        forecast = float(np.mean(np.mean(valid_forecasts, axis=0)))
        variance = float(np.mean(np.var(valid_forecasts, axis=0, ddof=1))) if n_valid > 1 else 0.0
        spread = float(np.max(np.max(valid_forecasts, axis=0) - np.min(valid_forecasts, axis=0)))

    return EqualWeightResult(
        forecast=forecast,
        variance=variance,
        n_models=n_valid,
        model_spread=spread,
        individual_forecasts=valid_forecasts,
    )


def compare_bma_vs_ew(
    bma_crps: float,
    ew_crps: float,
) -> EnsembleComparisonResult:
    """
    Compare BMA vs equal-weight ensemble on CRPS.

    Parameters
    ----------
    bma_crps : float
        CRPS of BMA forecast.
    ew_crps : float
        CRPS of equal-weight ensemble forecast.

    Returns
    -------
    EnsembleComparisonResult
    """
    bma_crps = float(bma_crps)
    ew_crps = float(ew_crps)

    # Lower CRPS is better, so delta > 0 means EW has lower (better) CRPS
    # We define delta as ew_crps - bma_crps: negative means BMA wins
    delta = bma_crps - ew_crps  # positive = BMA has higher CRPS = BMA loses
    bma_wins = delta < 0  # BMA wins when its CRPS is lower

    return EnsembleComparisonResult(
        bma_crps=bma_crps,
        ew_crps=ew_crps,
        delta_crps=delta,
        bma_wins=bma_wins,
        flag_investigation=not bma_wins,
    )


# ---------------------------------------------------------------------------
# Story 24.2: Trimmed Ensemble
# ---------------------------------------------------------------------------

def trimmed_ensemble(
    model_forecasts: np.ndarray,
    trim_frac: float = DEFAULT_TRIM_FRAC,
) -> TrimmedEnsembleResult:
    """
    Compute trimmed ensemble forecast.

    Drops the top and bottom `trim_frac` fraction of forecasts, then
    averages the remaining. This protects against one rogue model
    hijacking the combined forecast.

    Parameters
    ----------
    model_forecasts : array-like, shape (M,)
        Individual model forecasts. M = number of models.
    trim_frac : float
        Fraction of models to trim from each tail. Default 0.1 (10%).
        With 14 models: drops 1 highest and 1 lowest (12 remain).

    Returns
    -------
    TrimmedEnsembleResult
        Combined forecast and diagnostics.

    Raises
    ------
    ValueError
        If fewer than MIN_MODELS_AFTER_TRIM models remain after trimming.
    """
    forecasts = np.asarray(model_forecasts, dtype=np.float64).ravel()

    # Handle NaN: exclude NaN models
    valid_mask = np.isfinite(forecasts)
    valid_forecasts = forecasts[valid_mask]
    n_original = len(valid_forecasts)

    if n_original < MIN_MODELS_AFTER_TRIM:
        raise ValueError(
            f"Need >= {MIN_MODELS_AFTER_TRIM} valid models, got {n_original}"
        )

    # Clamp trim_frac
    trim_frac = float(np.clip(trim_frac, 0.0, 0.49))

    # Number to trim from each side
    n_trim_per_side = max(int(np.floor(n_original * trim_frac)), 0)

    # Ensure enough models remain
    n_remaining = n_original - 2 * n_trim_per_side
    if n_remaining < MIN_MODELS_AFTER_TRIM:
        # Reduce trim to keep minimum models
        n_trim_per_side = max((n_original - MIN_MODELS_AFTER_TRIM) // 2, 0)
        n_remaining = n_original - 2 * n_trim_per_side

    # Sort and identify trimmed indices
    sorted_indices = np.argsort(valid_forecasts)

    if n_trim_per_side > 0:
        trimmed_low = sorted_indices[:n_trim_per_side]
        trimmed_high = sorted_indices[-n_trim_per_side:]
        kept_indices = sorted_indices[n_trim_per_side:-n_trim_per_side]
    else:
        trimmed_low = np.array([], dtype=int)
        trimmed_high = np.array([], dtype=int)
        kept_indices = sorted_indices

    kept_forecasts = valid_forecasts[kept_indices]

    # Compute trimmed forecast
    forecast = float(np.mean(kept_forecasts))
    variance = float(np.var(kept_forecasts, ddof=1)) if len(kept_forecasts) > 1 else 0.0

    # Spreads
    spread_original = float(np.max(valid_forecasts) - np.min(valid_forecasts))
    spread_trimmed = float(np.max(kept_forecasts) - np.min(kept_forecasts)) if len(kept_forecasts) > 0 else 0.0

    return TrimmedEnsembleResult(
        forecast=forecast,
        variance=variance,
        n_models_original=n_original,
        n_models_used=len(kept_forecasts),
        n_trimmed=2 * n_trim_per_side,
        trim_frac=trim_frac,
        trimmed_indices_low=trimmed_low,
        trimmed_indices_high=trimmed_high,
        model_spread_original=spread_original,
        model_spread_trimmed=spread_trimmed,
    )


# ---------------------------------------------------------------------------
# Story 24.3: Online Prediction Pool
# ---------------------------------------------------------------------------

def online_prediction_pool(
    model_losses: np.ndarray,
    eta: float = DEFAULT_ETA,
    model_forecasts: Optional[np.ndarray] = None,
) -> OnlinePredictionPoolResult:
    """
    Online prediction pool using exponentiated weighted average (EWA).

    Updates model weights using multiplicative exponential updates.
    At each timestep t:
        w_{t+1,m} = w_{t,m} * exp(-eta * loss_{t,m}) / Z_t

    This is the Hedge algorithm (Freund & Schapire 1997) which
    achieves regret bound: R_T <= sqrt(T * ln(M) / (2 * eta)) + eta * T / 8

    With eta = sqrt(8 * ln(M) / T), the optimal bound is:
        R_T <= sqrt(T * ln(M) / 2)

    Parameters
    ----------
    model_losses : array-like, shape (T, M)
        Loss of each model at each timestep.
        T = number of timesteps, M = number of models.
        Losses should be non-negative (e.g., squared error, CRPS).
    eta : float
        Learning rate. Default 0.1.
        Larger eta = more aggressive weight updates.
    model_forecasts : array-like, shape (T, M), optional
        Individual model forecasts. If provided, the weighted forecast
        at the final timestep is returned.

    Returns
    -------
    OnlinePredictionPoolResult
        Final weights, regret, weight history, diagnostics.

    Raises
    ------
    ValueError
        If model_losses has fewer than MIN_MODELS_FOR_ENSEMBLE columns.
    """
    losses = np.asarray(model_losses, dtype=np.float64)

    if losses.ndim == 1:
        raise ValueError("model_losses must be 2D (T, M)")

    T, M = losses.shape

    if M < MIN_MODELS_FOR_ENSEMBLE:
        raise ValueError(
            f"Need >= {MIN_MODELS_FOR_ENSEMBLE} models, got {M}"
        )

    if T < 1:
        raise ValueError("Need >= 1 timestep")

    # Clamp eta
    eta = float(np.clip(eta, MIN_ETA, MAX_ETA))

    # Handle NaN losses: replace with column mean
    col_means = np.nanmean(losses, axis=0)
    for m in range(M):
        nan_mask = ~np.isfinite(losses[:, m])
        if np.any(nan_mask):
            losses[nan_mask, m] = col_means[m] if np.isfinite(col_means[m]) else 0.0

    # Initialize uniform weights
    weights = np.ones(M) / M
    weight_history = np.zeros((T, M))

    # Run EWA updates
    for t in range(T):
        weight_history[t] = weights

        # Multiplicative update: w_{t+1} = w_t * exp(-eta * loss_t)
        log_weights = np.log(np.maximum(weights, WEIGHT_FLOOR)) - eta * losses[t]

        # Normalize using log-sum-exp for numerical stability
        log_weights -= np.max(log_weights)  # shift for stability
        weights = np.exp(log_weights)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights /= weights_sum
        else:
            weights = np.ones(M) / M  # Reset if all underflow

        # Apply weight floor
        weights = np.maximum(weights, WEIGHT_FLOOR)
        weights /= np.sum(weights)

    # Compute cumulative losses
    # Pool's cumulative loss: weighted average at each step
    pool_cum_loss = 0.0
    for t in range(T):
        pool_cum_loss += np.dot(weight_history[t], losses[t])

    # Best single expert
    expert_cum_losses = np.sum(losses, axis=0)
    best_expert_idx = int(np.argmin(expert_cum_losses))
    best_expert_loss = float(expert_cum_losses[best_expert_idx])

    # Regret
    regret = pool_cum_loss - best_expert_loss

    # Theoretical regret bound: sqrt(T * ln(M) / 2) (optimal eta)
    regret_bound = np.sqrt(T * np.log(M) / 2.0) if M > 1 else 0.0

    # Weighted forecast at final timestep
    forecast = 0.0
    if model_forecasts is not None:
        fc = np.asarray(model_forecasts, dtype=np.float64)
        if fc.shape == losses.shape:
            forecast = float(np.dot(weights, fc[-1]))
        elif fc.ndim == 1 and len(fc) == M:
            forecast = float(np.dot(weights, fc))

    return OnlinePredictionPoolResult(
        weights=weights,
        cumulative_loss=float(pool_cum_loss),
        best_expert_loss=best_expert_loss,
        regret=float(regret),
        regret_bound=float(regret_bound),
        n_models=M,
        n_timesteps=T,
        weight_history=weight_history,
        best_expert_index=best_expert_idx,
        eta=eta,
        forecast=forecast,
    )


def online_prediction_pool_adaptive(
    model_losses: np.ndarray,
    model_forecasts: Optional[np.ndarray] = None,
) -> OnlinePredictionPoolResult:
    """
    Online prediction pool with theoretically optimal learning rate.

    Sets eta = sqrt(8 * ln(M) / T) for optimal worst-case regret.

    Parameters
    ----------
    model_losses : array-like, shape (T, M)
        Loss matrix.
    model_forecasts : array-like, shape (T, M), optional
        Model forecasts.

    Returns
    -------
    OnlinePredictionPoolResult
    """
    losses = np.asarray(model_losses, dtype=np.float64)
    if losses.ndim != 2:
        raise ValueError("model_losses must be 2D (T, M)")

    T, M = losses.shape
    if M < 2 or T < 1:
        raise ValueError(f"Need >= 2 models and >= 1 timestep, got M={M}, T={T}")

    # Optimal learning rate for Hedge algorithm
    eta_opt = np.sqrt(8.0 * np.log(M) / max(T, 1))
    eta_opt = float(np.clip(eta_opt, MIN_ETA, MAX_ETA))

    return online_prediction_pool(losses, eta=eta_opt, model_forecasts=model_forecasts)
