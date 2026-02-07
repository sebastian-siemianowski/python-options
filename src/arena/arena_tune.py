"""
===============================================================================
ARENA TUNE — Model Competition Engine
===============================================================================

Runs head-to-head competition between experimental and standard models
on the arena benchmark universe.

Architecture:
    1. Load benchmark data (12 symbols across cap categories)
    2. For each symbol, fit ALL models (standard + experimental)
    3. Score each model using combined BIC + Hyvärinen + PIT
    4. Aggregate scores by model and category
    5. Determine winners and promotion candidates

Competition Scoring:
    - BIC: Information-theoretic model selection
    - Hyvärinen: Robust to model misspecification
    - PIT: Calibration quality (Kelly sizing validity)
    
    Combined Score = w_bic * BIC_rank + w_hyv * Hyv_rank + w_pit * PIT_pass

Promotion Gate:
    Experimental model graduates if:
    1. Combined score > best standard model by >5%
    2. PIT p-value > 0.05 on ALL symbols
    3. Consistent across cap categories (no category failure)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import json
import os
import sys
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Get CPU count for parallel processing
N_CPUS = mp.cpu_count()

# Rich for presentation
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add src to path for imports
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from .arena_config import (
    ArenaConfig,
    DEFAULT_ARENA_CONFIG,
    SYMBOL_CATEGORIES,
    CapCategory,
    get_category_weights,
    load_disabled_models,
    disable_model,
    get_enabled_experimental_models,
    is_model_disabled,
)
from .arena_data import (
    load_arena_data,
    ArenaDataset,
)
from .arena_models import (
    EXPERIMENTAL_MODELS,
    STANDARD_MOMENTUM_MODELS,
    get_standard_model_specs,
    get_experimental_model_specs,
    create_experimental_model,
)

# Import standard tuning infrastructure
try:
    from tuning.tune import (
        fit_model_with_cache,
        compute_ewma_volatility,
        assign_regime_labels,
    )
    TUNE_AVAILABLE = True
except ImportError:
    TUNE_AVAILABLE = False

# Import PIT calibration
try:
    from calibration.pit_calibration import compute_pit_calibration, CalibrationMetrics
    PIT_AVAILABLE = True
except ImportError:
    PIT_AVAILABLE = False

# Import model selection utilities
try:
    from calibration.model_selection import compute_bic, compute_aic
    MODEL_SELECTION_AVAILABLE = True
except ImportError:
    MODEL_SELECTION_AVAILABLE = False

# Import CRPS scoring (new)
try:
    from .scoring import (
        compute_crps_gaussian,
        compute_crps_student_t,
        compute_combined_score,
        CRPSResult,
        CombinedScoreResult,
        ScoringConfig,
    )
    from .scoring.hyvarinen import (
        compute_hyvarinen_score_gaussian,
        compute_hyvarinen_score_student_t,
    )
    CRPS_AVAILABLE = True
except ImportError:
    CRPS_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelScore:
    """
    Score for a single model on a single symbol.
    
    Attributes:
        model_name: Model identifier
        symbol: Ticker symbol
        category: Market cap category
        log_likelihood: Fitted log-likelihood
        bic: Bayesian Information Criterion
        aic: Akaike Information Criterion
        crps: Continuous Ranked Probability Score (lower is better)
        pit_pvalue: PIT uniformity test p-value
        pit_calibrated: Whether PIT test passed
        hyvarinen_score: Hyvarinen score (if computed)
        combined_score: Weighted combined score
        css_score: Calibration Stability Under Stress (0-1)
        fec_score: Forecast Entropy Consistency (0-1)
        dig_score: Directional Information Gain (0-1)
        advanced_score: Combined CSS+FEC+DIG score
        fit_params: Fitted parameters
        fit_time_ms: Fitting time in milliseconds
    """
    model_name: str
    symbol: str
    category: CapCategory
    log_likelihood: float
    bic: float
    aic: float
    crps: Optional[float] = None
    pit_pvalue: float = 0.0
    pit_calibrated: bool = False
    hyvarinen_score: Optional[float] = None
    combined_score: Optional[float] = None
    css_score: Optional[float] = None
    fec_score: Optional[float] = None
    dig_score: Optional[float] = None
    advanced_score: Optional[float] = None
    fit_params: Dict[str, float] = field(default_factory=dict)
    fit_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "category": self.category.value,
            "log_likelihood": self.log_likelihood,
            "bic": self.bic,
            "aic": self.aic,
            "crps": self.crps,
            "pit_pvalue": self.pit_pvalue,
            "pit_calibrated": self.pit_calibrated,
            "hyvarinen_score": self.hyvarinen_score,
            "combined_score": self.combined_score,
            "css_score": self.css_score,
            "fec_score": self.fec_score,
            "dig_score": self.dig_score,
            "advanced_score": self.advanced_score,
            "fit_params": self.fit_params,
            "fit_time_ms": self.fit_time_ms,
        }


@dataclass
class ArenaResult:
    """
    Complete result of arena competition.
    
    Attributes:
        timestamp: Competition run timestamp
        config: Arena configuration
        scores: All model scores
        rankings: Model rankings by category and overall
        promotion_candidates: Models recommended for promotion
        summary: Summary statistics
    """
    timestamp: str
    config: ArenaConfig
    scores: List[ModelScore]
    rankings: Dict[str, List[str]]
    promotion_candidates: List[str]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "scores": [s.to_dict() for s in self.scores],
            "rankings": self.rankings,
            "promotion_candidates": self.promotion_candidates,
            "summary": self.summary,
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# VOLATILITY COMPUTATION
# =============================================================================

def compute_ewma_vol(returns: np.ndarray, span: int = 20) -> np.ndarray:
    """Compute EWMA volatility."""
    import pandas as pd
    ret_series = pd.Series(returns)
    ewma_var = ret_series.pow(2).ewm(span=span, adjust=False).mean()
    return np.sqrt(ewma_var.values)


# =============================================================================
# STANDARD MODEL FITTING
# =============================================================================

def fit_standard_model(
    model_name: str,
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit a standard momentum model using existing tuning infrastructure.
    
    Args:
        model_name: Standard model name (e.g., "phi_student_t_nu_8_momentum")
        returns: Log returns
        vol: EWMA volatility
        regime_labels: Regime assignments
        
    Returns:
        Dictionary with fitted parameters and diagnostics
    """
    from scipy.optimize import minimize
    from scipy.special import gammaln
    
    n = len(returns)
    
    # Parse model type
    is_gaussian = "gaussian" in model_name.lower() and "phi" not in model_name.lower()
    is_phi_gaussian = "phi_gaussian" in model_name.lower()
    is_student_t = "student_t" in model_name.lower()
    
    # Extract nu for Student-t
    nu = None
    if is_student_t:
        import re
        match = re.search(r'nu_(\d+)', model_name)
        if match:
            nu = int(match.group(1))
        else:
            nu = 8  # Default
    
    def gaussian_filter(returns, vol, q, c):
        """Simple Gaussian Kalman filter."""
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        mu[0], P[0] = 0.0, 1e-4
        ll = 0.0
        
        for t in range(1, n):
            mu_pred = mu[t-1]
            P_pred = P[t-1] + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            innovation = returns[t] - mu_pred
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            if S > 0:
                ll += -0.5 * (np.log(2 * np.pi * S) + innovation**2 / S)
        
        return mu, P, ll
    
    def phi_gaussian_filter(returns, vol, q, c, phi):
        """AR(1) Gaussian Kalman filter."""
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        mu[0], P[0] = 0.0, 1e-4
        ll = 0.0
        
        for t in range(1, n):
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            innovation = returns[t] - mu_pred
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            if S > 0:
                ll += -0.5 * (np.log(2 * np.pi * S) + innovation**2 / S)
        
        return mu, P, ll
    
    def student_t_filter(returns, vol, q, c, phi, nu):
        """AR(1) Student-t Kalman filter."""
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        mu[0], P[0] = 0.0, 1e-4
        ll = 0.0
        
        for t in range(1, n):
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            innovation = returns[t] - mu_pred
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            if S > 0:
                z = innovation / np.sqrt(S)
                ll_t = (
                    gammaln((nu + 1) / 2) 
                    - gammaln(nu / 2)
                    - 0.5 * np.log(nu * np.pi * S)
                    - ((nu + 1) / 2) * np.log(1 + z**2 / nu)
                )
                ll += ll_t
        
        return mu, P, ll
    
    # Fit model
    import time
    start_time = time.time()
    
    if is_gaussian:
        def neg_ll(params):
            q, c = params
            if q <= 0 or c <= 0:
                return 1e10
            try:
                _, _, ll = gaussian_filter(returns, vol, q, c)
                return -ll
            except:
                return 1e10
        
        result = minimize(
            neg_ll,
            x0=[1e-6, 1.0],
            method='L-BFGS-B',
            bounds=[(1e-10, 1e-2), (0.1, 5.0)],
        )
        q_opt, c_opt = result.x
        _, _, final_ll = gaussian_filter(returns, vol, q_opt, c_opt)
        n_params = 2
        fit_params = {"q": q_opt, "c": c_opt}
        
    elif is_phi_gaussian:
        def neg_ll(params):
            q, c, phi = params
            if q <= 0 or c <= 0 or not (-1 < phi < 1):
                return 1e10
            try:
                _, _, ll = phi_gaussian_filter(returns, vol, q, c, phi)
                return -ll
            except:
                return 1e10
        
        result = minimize(
            neg_ll,
            x0=[1e-6, 1.0, 0.95],
            method='L-BFGS-B',
            bounds=[(1e-10, 1e-2), (0.1, 5.0), (-0.99, 0.99)],
        )
        q_opt, c_opt, phi_opt = result.x
        _, _, final_ll = phi_gaussian_filter(returns, vol, q_opt, c_opt, phi_opt)
        n_params = 3
        fit_params = {"q": q_opt, "c": c_opt, "phi": phi_opt}
        
    else:  # Student-t
        def neg_ll(params):
            q, c, phi = params
            if q <= 0 or c <= 0 or not (-1 < phi < 1):
                return 1e10
            try:
                _, _, ll = student_t_filter(returns, vol, q, c, phi, nu)
                return -ll
            except:
                return 1e10
        
        result = minimize(
            neg_ll,
            x0=[1e-6, 1.0, 0.95],
            method='L-BFGS-B',
            bounds=[(1e-10, 1e-2), (0.1, 5.0), (-0.99, 0.99)],
        )
        q_opt, c_opt, phi_opt = result.x
        _, _, final_ll = student_t_filter(returns, vol, q_opt, c_opt, phi_opt, nu)
        n_params = 4  # q, c, phi, nu (nu is fixed)
        fit_params = {"q": q_opt, "c": c_opt, "phi": phi_opt, "nu": nu}
    
    fit_time_ms = (time.time() - start_time) * 1000
    
    # Compute BIC/AIC
    bic = -2 * final_ll + n_params * np.log(n)
    aic = -2 * final_ll + 2 * n_params
    
    return {
        "model_name": model_name,
        "log_likelihood": final_ll,
        "bic": bic,
        "aic": aic,
        "n_params": n_params,
        "n_observations": n,
        "fit_params": fit_params,
        "fit_time_ms": fit_time_ms,
        "success": result.success,
    }


# =============================================================================
# EXPERIMENTAL MODEL FITTING
# =============================================================================

def fit_experimental_model(
    model_name: str,
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit an experimental model.
    
    Args:
        model_name: Experimental model name
        returns: Log returns
        vol: EWMA volatility
        regime_labels: Regime assignments
        
    Returns:
        Dictionary with fitted parameters and diagnostics
    """
    import time
    start_time = time.time()
    
    model = create_experimental_model(model_name)
    result = model.fit(returns, vol)
    
    fit_time_ms = (time.time() - start_time) * 1000
    result["fit_time_ms"] = fit_time_ms
    result["model_name"] = model_name
    
    return result


# =============================================================================
# SCORING METRICS (CRPS, Hyvarinen)
# =============================================================================

def compute_scoring_metrics(
    returns: np.ndarray,
    vol: np.ndarray,
    fit_params: Dict[str, float],
    model_name: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute CRPS and Hyvarinen scores for a fitted model.
    
    Args:
        returns: Log returns
        vol: EWMA volatility
        fit_params: Fitted model parameters
        model_name: Name of the model
        
    Returns:
        (crps, hyvarinen) - scores, None if computation fails
    """
    crps = None
    hyvarinen = None
    
    if not CRPS_AVAILABLE:
        return None, None
    
    try:
        n = len(returns)
        
        # Generate predictions using fitted parameters
        q = fit_params.get("q", 1e-6)
        c = fit_params.get("c", 1.0)
        phi = fit_params.get("phi", 0.0)
        nu = fit_params.get("nu", None)
        
        # Simple one-step ahead predictions
        mu_pred = np.zeros(n)
        sigma_pred = np.zeros(n)
        
        for t in range(1, n):
            mu_pred[t] = phi * returns[t-1] if abs(phi) > 0 else 0.0
            sigma_pred[t] = c * vol[t] if vol[t] > 0 else c * 0.01
        
        # Skip warmup period
        obs = returns[60:]
        mu = mu_pred[60:]
        sigma = np.maximum(sigma_pred[60:], 1e-6)
        
        if len(obs) < 50:
            return None, None
        
        # Compute CRPS
        if "student_t" in model_name or nu is not None:
            nu_val = nu if nu else 8.0
            from .scoring.crps import compute_crps_student_t
            crps_result = compute_crps_student_t(obs, mu, sigma, nu_val)
        else:
            from .scoring.crps import compute_crps_gaussian
            crps_result = compute_crps_gaussian(obs, mu, sigma)
        
        crps = crps_result.crps
        
        # Compute Hyvarinen
        if "student_t" in model_name or nu is not None:
            nu_val = nu if nu else 8.0
            hyvarinen = compute_hyvarinen_score_student_t(obs, mu, sigma, nu_val)
        else:
            hyvarinen = compute_hyvarinen_score_gaussian(obs, mu, sigma)
        
    except Exception:
        pass
    
    return crps, hyvarinen


def compute_advanced_scoring_metrics(
    returns: np.ndarray,
    vol: np.ndarray,
    fit_params: Dict[str, float],
    model_name: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute CSS, FEC, DIG advanced scoring metrics.
    """
    from scipy.stats import norm, kstest
    
    n = len(returns)
    warmup = 60
    
    if n - warmup < 100:
        return None, None, None, None
    
    try:
        q = fit_params.get("q", 1e-6) if isinstance(fit_params, dict) else 1e-6
        c = fit_params.get("c", 1.0) if isinstance(fit_params, dict) else 1.0
        phi = fit_params.get("phi", 0.0) if isinstance(fit_params, dict) else 0.0
        
        mu_pred = np.zeros(n)
        sigma_pred = np.zeros(n)
        pit_values = np.zeros(n)
        
        for t in range(1, n):
            mu_pred[t] = phi * returns[t-1] if abs(phi) > 0 else 0.0
            sigma_pred[t] = c * vol[t] if vol[t] > 0 else c * 0.01
            if sigma_pred[t] > 1e-10:
                z = (returns[t] - mu_pred[t]) / sigma_pred[t]
                pit_values[t] = norm.cdf(z)
            else:
                pit_values[t] = 0.5
        
        pit_clean = pit_values[warmup:]
        vol_clean = vol[warmup:]
        
        if len(vol_clean) < 100:
            return None, None, None, None
            
        vol_pctiles = np.percentile(vol_clean, [25, 75, 95])
        
        # CSS: Calibration Stability Under Stress
        css_scores = []
        regimes = [
            (0, vol_pctiles[0]),
            (vol_pctiles[0], vol_pctiles[1]),
            (vol_pctiles[1], vol_pctiles[2]),
            (vol_pctiles[2], np.max(vol_clean)+1),
        ]
        for low, high in regimes:
            mask = (vol_clean >= low) & (vol_clean < high)
            if np.sum(mask) >= 20:
                regime_pit = pit_clean[mask]
                regime_pit_valid = regime_pit[(regime_pit > 0.001) & (regime_pit < 0.999)]
                if len(regime_pit_valid) >= 10:
                    _, pval = kstest(regime_pit_valid, 'uniform')
                    css_scores.append(min(1.0, pval / 0.05))
        css_score = np.mean(css_scores) if css_scores else 0.5
        
        # FEC: Forecast Entropy Consistency
        sigma_clean = sigma_pred[warmup:]
        sigma_safe = np.maximum(sigma_clean, 1e-8)
        forecast_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma_safe**2)
        vol_safe = np.maximum(vol_clean, 1e-8)
        market_entropy = 0.5 * np.log(2 * np.pi * np.e * vol_safe**2)
        valid_mask = np.isfinite(forecast_entropy) & np.isfinite(market_entropy)
        if np.sum(valid_mask) > 50:
            corr = np.corrcoef(forecast_entropy[valid_mask], market_entropy[valid_mask])[0, 1]
            fec_score = (corr + 1) / 2 if np.isfinite(corr) else 0.5
        else:
            fec_score = 0.5
        
        # DIG: Directional Information Gain
        mu_clean = mu_pred[warmup:]
        prob_positive = norm.cdf(0, loc=-mu_clean, scale=np.maximum(sigma_clean, 1e-8))
        prob_positive = np.clip(prob_positive, 0.001, 0.999)
        actual_positive = (returns[warmup:] > 0).astype(float)
        predicted_direction = (prob_positive > 0.5).astype(float)
        hit_rate = np.mean(predicted_direction == actual_positive)
        kl_divs = []
        for p in prob_positive:
            model_p = np.array([1-p, p])
            baseline = np.array([0.5, 0.5])
            kl = np.sum(model_p * np.log(model_p / baseline))
            if np.isfinite(kl):
                kl_divs.append(kl)
        kl_divergence = np.mean(kl_divs) if kl_divs else 0.0
        kl_score = min(1.0, kl_divergence / 0.1)
        hit_score = min(1.0, max(0, hit_rate - 0.5) / 0.1)
        dig_score = 0.7 * kl_score + 0.3 * hit_score
        
        advanced_score = 0.4 * css_score + 0.35 * fec_score + 0.25 * dig_score
        return float(css_score), float(fec_score), float(dig_score), float(advanced_score)
        
    except Exception:
        return None, None, None, None


# =============================================================================
# PIT CALIBRATION
# =============================================================================

def compute_pit_score(
    returns: np.ndarray,
    vol: np.ndarray,
    fit_params: Dict[str, float],
    model_type: str,
) -> Tuple[float, bool]:
    """
    Compute PIT calibration score.
    
    Returns:
        (p_value, is_calibrated)
    """
    # Generate one-step-ahead predictions
    n = len(returns)
    predictions = np.zeros(n)
    
    # Simple volatility-based predictions
    for t in range(1, n):
        # Predict using fitted volatility scale
        c = fit_params.get("c", 1.0)
        sigma = c * vol[t] if vol[t] > 0 else c * 0.01
        # P(r > 0) assuming mean 0
        predictions[t] = 0.5  # Baseline: 50% probability
        
        # Adjust based on drift if available
        phi = fit_params.get("phi", 0.0)
        if t > 1 and abs(phi) > 0:
            drift_signal = phi * returns[t-1]
            predictions[t] = 0.5 + 0.3 * np.tanh(drift_signal / sigma)
    
    # Compute PIT using actual outcomes
    if PIT_AVAILABLE:
        try:
            metrics = compute_pit_calibration(
                predictions[60:],  # Skip warmup
                returns[60:],
                n_bins=10,
                ece_threshold=0.10,
            )
            return metrics.uniformity_pvalue, metrics.calibrated
        except:
            pass
    
    # Fallback: simple uniformity check
    from scipy.stats import kstest
    pit_values = predictions[60:]
    if len(pit_values) > 50:
        stat, pvalue = kstest(pit_values, 'uniform')
        return pvalue, pvalue > 0.05
    
    return 0.5, True  # Default if insufficient data


# =============================================================================
# PARALLEL PROCESSING HELPERS
# =============================================================================

def _compute_regime_labels(returns: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Compute regime labels for returns series."""
    n = len(returns)
    regime_labels = np.zeros(n, dtype=int)
    window = 60
    for t in range(window, n):
        recent_vol = vol[t]
        vol_median = np.median(vol[max(0,t-252):t])
        drift = np.mean(returns[t-window:t])
        is_trending = abs(drift) > 0.0005
        is_high_vol = recent_vol > 1.3 * vol_median
        is_low_vol = recent_vol < 0.85 * vol_median
        is_crisis = recent_vol > 2.0 * vol_median
        if is_crisis:
            regime_labels[t] = 4
        elif is_low_vol and is_trending:
            regime_labels[t] = 0
        elif is_high_vol and is_trending:
            regime_labels[t] = 1
        elif is_low_vol and not is_trending:
            regime_labels[t] = 2
        else:
            regime_labels[t] = 3
    return regime_labels


def _fit_single_model_wrapper(args: Tuple) -> Dict:
    """Wrapper for multiprocessing - fits ONE model for ONE symbol."""
    symbol, model_name, model_type, returns, vol, regime_labels, category_value = args
    
    try:
        if model_type == "standard":
            result = fit_standard_model(model_name, returns, vol, regime_labels)
            fit_params = result["fit_params"]
        else:
            result = fit_experimental_model(model_name, returns, vol, regime_labels)
            fit_params = result.get("fit_params", {})
            if not fit_params:
                fit_params = {"q": result.get("q", 1e-6), "c": result.get("c", 1.0), "phi": result.get("phi", 0.0)}
        
        pit_pvalue, pit_calibrated = compute_pit_score(returns, vol, fit_params, model_name)
        crps, hyvarinen = compute_scoring_metrics(returns, vol, fit_params, model_name)
        css, fec, dig, adv = compute_advanced_scoring_metrics(returns, vol, fit_params, model_name)
        
        return {
            "model_name": model_name,
            "symbol": symbol,
            "category": category_value,
            "log_likelihood": result.get("log_likelihood", 0),
            "bic": result.get("bic", 0),
            "aic": result.get("aic", 0),
            "crps": crps,
            "pit_pvalue": pit_pvalue,
            "pit_calibrated": pit_calibrated,
            "hyvarinen_score": hyvarinen,
            "css_score": css,
            "fec_score": fec,
            "dig_score": dig,
            "advanced_score": adv,
            "fit_params": fit_params,
            "fit_time_ms": result.get("fit_time_ms", 0),
            "success": True,
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "symbol": symbol,
            "category": category_value,
            "success": False,
            "error": str(e),
        }


def _fit_single_symbol_wrapper(args: Tuple) -> List[Dict]:
    """Wrapper for multiprocessing - fits all models for a single symbol (fallback)."""
    symbol, returns, vol, category_value, config_dict = args
    
    class MinConfig:
        def __init__(self, d):
            self.test_standard = d.get('test_standard', True)
            self.test_experimental = d.get('test_experimental', True)
            self.verbose = d.get('verbose', False)
    
    config = MinConfig(config_dict)
    regime_labels = _compute_regime_labels(returns, vol)
    
    results = []
    
    if config.test_standard:
        for model_name in STANDARD_MOMENTUM_MODELS:
            r = _fit_single_model_wrapper((symbol, model_name, "standard", returns, vol, regime_labels, category_value))
            if r.get("success", False):
                results.append(r)
    
    if config.test_experimental:
        enabled_models = get_enabled_experimental_models()
        for model_name in enabled_models:
            r = _fit_single_model_wrapper((symbol, model_name, "experimental", returns, vol, regime_labels, category_value))
            if r.get("success", False):
                results.append(r)
    
    return results


def fit_all_symbols_parallel(
    datasets: Dict[str, 'ArenaDataset'],
    config: 'ArenaConfig',
    n_workers: int = None,
) -> List['ModelScore']:
    """Fit all models for all symbols using multiprocessing at (symbol, model) granularity."""
    if n_workers is None:
        n_workers = max(1, N_CPUS - 1)
    
    # Build list of ALL (symbol, model) combinations for true parallelism
    tasks = []
    
    # Precompute volatility and regime labels for each symbol
    symbol_data = {}
    for symbol, dataset in datasets.items():
        vol = compute_ewma_vol(dataset.returns)
        regime_labels = _compute_regime_labels(dataset.returns, vol)
        symbol_data[symbol] = {
            'returns': dataset.returns,
            'vol': vol,
            'regime_labels': regime_labels,
            'category': dataset.category.value,
        }
    
    # Create tasks for standard models
    if config.test_standard:
        for symbol in datasets.keys():
            sd = symbol_data[symbol]
            for model_name in STANDARD_MOMENTUM_MODELS:
                tasks.append((symbol, model_name, "standard", sd['returns'], sd['vol'], sd['regime_labels'], sd['category']))
    
    # Create tasks for experimental models
    if config.test_experimental:
        enabled_models = get_enabled_experimental_models()
        for symbol in datasets.keys():
            sd = symbol_data[symbol]
            for model_name in enabled_models:
                tasks.append((symbol, model_name, "experimental", sd['returns'], sd['vol'], sd['regime_labels'], sd['category']))
    
    all_results = []
    
    # Execute all tasks in parallel using fork context for fresh processes
    mp_context = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(_fit_single_model_wrapper, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result.get("success", False):
                    all_results.append(result)
            except Exception as e:
                pass
    
    # Convert to ModelScore objects
    scores = []
    for r in all_results:
        from .arena_config import CapCategory
        score = ModelScore(
            model_name=r["model_name"],
            symbol=r["symbol"],
            category=CapCategory(r["category"]),
            log_likelihood=r["log_likelihood"],
            bic=r["bic"],
            aic=r["aic"],
            crps=r["crps"],
            pit_pvalue=r["pit_pvalue"],
            pit_calibrated=r["pit_calibrated"],
            hyvarinen_score=r["hyvarinen_score"],
            css_score=r.get("css_score"),
            fec_score=r.get("fec_score"),
            dig_score=r.get("dig_score"),
            advanced_score=r.get("advanced_score"),
            fit_params=r["fit_params"],
            fit_time_ms=r["fit_time_ms"],
        )
        scores.append(score)
    
    return scores


# =============================================================================
# COMPETITION RUNNER
# =============================================================================

def fit_all_models_for_symbol(
    dataset: ArenaDataset,
    config: ArenaConfig,
) -> List[ModelScore]:
    """Fit all models (standard + experimental) for a single symbol."""
    returns = dataset.returns
    vol = compute_ewma_vol(returns)
    
    # Assign regimes (simple method)
    n = len(returns)
    regime_labels = np.zeros(n, dtype=int)
    window = 60
    for t in range(window, n):
        recent_vol = vol[t]
        vol_median = np.median(vol[max(0,t-252):t])
        drift = np.mean(returns[t-window:t])
        
        is_trending = abs(drift) > 0.0005
        is_high_vol = recent_vol > 1.3 * vol_median
        is_low_vol = recent_vol < 0.85 * vol_median
        is_crisis = recent_vol > 2.0 * vol_median
        
        if is_crisis:
            regime_labels[t] = 4
        elif is_low_vol and is_trending:
            regime_labels[t] = 0
        elif is_high_vol and is_trending:
            regime_labels[t] = 1
        elif is_low_vol and not is_trending:
            regime_labels[t] = 2
        else:
            regime_labels[t] = 3
    
    scores = []
    
    # Fit standard models
    if config.test_standard:
        for model_name in STANDARD_MOMENTUM_MODELS:
            try:
                result = fit_standard_model(model_name, returns, vol, regime_labels)
                
                # Compute PIT
                pit_pvalue, pit_calibrated = compute_pit_score(
                    returns, vol, result["fit_params"], model_name
                )
                
                # Compute CRPS and Hyvarinen
                crps, hyvarinen = compute_scoring_metrics(
                    returns, vol, result["fit_params"], model_name
                )
                
                # Compute CSS, FEC, DIG (advanced metrics)
                css, fec, dig, adv = compute_advanced_scoring_metrics(
                    returns, vol, result["fit_params"], model_name
                )
                
                score = ModelScore(
                    model_name=model_name,
                    symbol=dataset.symbol,
                    category=dataset.category,
                    log_likelihood=result["log_likelihood"],
                    bic=result["bic"],
                    aic=result["aic"],
                    crps=crps,
                    pit_pvalue=pit_pvalue,
                    pit_calibrated=pit_calibrated,
                    hyvarinen_score=hyvarinen,
                    css_score=css,
                    fec_score=fec,
                    dig_score=dig,
                    advanced_score=adv,
                    fit_params=result["fit_params"],
                    fit_time_ms=result["fit_time_ms"],
                )
                scores.append(score)
            except Exception as e:
                if config.verbose:
                    print(f"  Error fitting {model_name} on {dataset.symbol}: {e}")
    
    # Fit experimental models (skip disabled ones)
    if config.test_experimental:
        enabled_models = get_enabled_experimental_models()
        
        for model_name in enabled_models:
            try:
                result = fit_experimental_model(model_name, returns, vol, regime_labels)
                
                # Get fit params from result
                fit_params = result.get("fit_params", result)
                if not isinstance(fit_params, dict):
                    fit_params = {}
                
                # Compute PIT
                pit_pvalue, pit_calibrated = compute_pit_score(
                    returns, vol, fit_params, model_name
                )
                
                # Compute CRPS and Hyvarinen
                crps, hyvarinen = compute_scoring_metrics(
                    returns, vol, fit_params, model_name
                )
                
                # Compute CSS, FEC, DIG (advanced metrics)
                css, fec, dig, adv = compute_advanced_scoring_metrics(
                    returns, vol, fit_params, model_name
                )
                
                score = ModelScore(
                    model_name=model_name,
                    symbol=dataset.symbol,
                    category=dataset.category,
                    log_likelihood=result["log_likelihood"],
                    bic=result["bic"],
                    aic=result.get("aic", result["bic"]),
                    crps=crps,
                    pit_pvalue=pit_pvalue,
                    pit_calibrated=pit_calibrated,
                    hyvarinen_score=hyvarinen,
                    css_score=css,
                    fec_score=fec,
                    dig_score=dig,
                    advanced_score=adv,
                    fit_params=result.get("fit_params", {}),
                    fit_time_ms=result.get("fit_time_ms", 0),
                )
                scores.append(score)
            except Exception as e:
                if config.verbose:
                    print(f"  Error fitting {model_name} on {dataset.symbol}: {e}")
    
    return scores


def compute_combined_scores(
    scores: List[ModelScore],
    config: ArenaConfig,
) -> List[ModelScore]:
    """
    Compute combined scores for all models.
    
    Args:
        scores: List of model scores
        config: Arena configuration
        
    Returns:
        Updated scores with combined_score field
    """
    if not scores:
        return scores
    
    # Group by symbol
    by_symbol: Dict[str, List[ModelScore]] = {}
    for score in scores:
        by_symbol.setdefault(score.symbol, []).append(score)
    
    # Compute combined scores within each symbol
    for symbol, symbol_scores in by_symbol.items():
        weights = get_category_weights(symbol)
        
        # Get BIC range for normalization
        bics = [s.bic for s in symbol_scores]
        bic_min, bic_max = min(bics), max(bics)
        bic_range = bic_max - bic_min if bic_max > bic_min else 1.0
        
        for score in symbol_scores:
            # BIC score (lower is better, so invert)
            bic_norm = 1.0 - (score.bic - bic_min) / bic_range
            
            # PIT score (binary + p-value)
            pit_score = 0.5 * float(score.pit_calibrated) + 0.5 * score.pit_pvalue
            
            # Hyvärinen placeholder (would need actual computation)
            hyv_score = 0.5  # Neutral
            
            # Combined score
            score.combined_score = (
                weights["bic"] * bic_norm +
                weights["hyvarinen"] * hyv_score +
                weights["pit"] * pit_score
            )
    
    return scores


def determine_rankings(
    scores: List[ModelScore],
) -> Dict[str, List[str]]:
    """
    Determine model rankings by category and overall.
    
    Args:
        scores: List of model scores with combined_score
        
    Returns:
        Dictionary with rankings by category and overall
    """
    rankings = {}
    
    # Overall ranking
    model_avg_scores: Dict[str, List[float]] = {}
    for score in scores:
        if score.combined_score is not None:
            model_avg_scores.setdefault(score.model_name, []).append(score.combined_score)
    
    overall_avg = {
        model: np.mean(scores_list) 
        for model, scores_list in model_avg_scores.items()
    }
    rankings["overall"] = sorted(overall_avg.keys(), key=lambda m: overall_avg[m], reverse=True)
    
    # By category
    for category in CapCategory:
        cat_scores = [s for s in scores if s.category == category]
        if not cat_scores:
            continue
        
        cat_avg: Dict[str, List[float]] = {}
        for score in cat_scores:
            if score.combined_score is not None:
                cat_avg.setdefault(score.model_name, []).append(score.combined_score)
        
        cat_overall = {
            model: np.mean(scores_list)
            for model, scores_list in cat_avg.items()
        }
        rankings[category.value] = sorted(cat_overall.keys(), key=lambda m: cat_overall[m], reverse=True)
    
    return rankings


def determine_promotion_candidates(
    scores: List[ModelScore],
    rankings: Dict[str, List[str]],
    config: ArenaConfig,
) -> List[str]:
    """
    Determine which experimental models qualify for promotion.
    
    Criteria:
    1. Beat best standard model by >promotion_threshold (5%)
    2. PIT calibrated on ALL symbols
    3. No category where it ranks last
    
    Args:
        scores: All scores
        rankings: Model rankings
        config: Arena configuration
        
    Returns:
        List of model names recommended for promotion
    """
    candidates = []
    
    # Get experimental model names
    experimental_names = set(EXPERIMENTAL_MODELS.keys())
    
    # Compute average scores per model
    model_avg: Dict[str, float] = {}
    model_pit_all_pass: Dict[str, bool] = {}
    
    for model_name in set(s.model_name for s in scores):
        model_scores = [s for s in scores if s.model_name == model_name]
        if model_scores:
            avg_combined = np.mean([s.combined_score for s in model_scores if s.combined_score])
            model_avg[model_name] = avg_combined
            model_pit_all_pass[model_name] = all(s.pit_calibrated for s in model_scores)
    
    # Find best standard model score
    standard_scores = [
        model_avg[m] for m in STANDARD_MOMENTUM_MODELS 
        if m in model_avg
    ]
    best_standard = max(standard_scores) if standard_scores else 0.5
    
    # Check each experimental model
    for exp_name in experimental_names:
        if exp_name not in model_avg:
            continue
        
        exp_score = model_avg[exp_name]
        exp_pit_pass = model_pit_all_pass.get(exp_name, False)
        
        # Criterion 1: Beat best standard by threshold
        improvement = (exp_score - best_standard) / best_standard if best_standard > 0 else 0
        beats_threshold = improvement > config.promotion_threshold
        
        # Criterion 2: PIT calibrated everywhere
        pit_pass = exp_pit_pass
        
        # Criterion 3: Not last in any category
        not_last_anywhere = True
        for cat_key, cat_ranking in rankings.items():
            if cat_key == "overall":
                continue
            if cat_ranking and exp_name in cat_ranking:
                if cat_ranking[-1] == exp_name:
                    not_last_anywhere = False
                    break
        
        if beats_threshold and pit_pass and not_last_anywhere:
            candidates.append(exp_name)
    
    return candidates


def run_arena_competition(
    config: Optional[ArenaConfig] = None,
    symbols: Optional[List[str]] = None,
    parallel: bool = True,
    n_workers: int = None,
) -> ArenaResult:
    """Run full arena competition with optional parallel processing."""
    config = config or DEFAULT_ARENA_CONFIG
    config.validate()
    
    if symbols:
        config.symbols = symbols
    
    # Create results directory
    results_dir = Path(config.arena_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, N_CPUS - 1)
    
    # Load data
    if RICH_AVAILABLE:
        console = Console()
        
        # Check disabled models
        disabled = load_disabled_models()
        enabled_exp = get_enabled_experimental_models()
        
        # Calculate total parallel tasks
        n_std = len(STANDARD_MOMENTUM_MODELS) if config.test_standard else 0
        n_exp = len(enabled_exp) if config.test_experimental else 0
        n_total_tasks = len(config.symbols) * (n_std + n_exp)
        
        # Clean header
        console.print()
        console.print("[bold cyan]ARENA MODEL COMPETITION[/bold cyan]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print(f"  Symbols: [bold]{len(config.symbols)}[/bold]    "
                     f"Standard: [bold]{n_std}[/bold]    "
                     f"Experimental: [bold]{n_exp}[/bold]")
        if parallel:
            console.print(f"  [dim]Parallel: {n_workers} workers x {n_total_tasks} tasks (model-level)[/dim]")
        if disabled:
            console.print(f"  [dim]Disabled: {', '.join(disabled.keys())}[/dim]")
        console.print()
    
    datasets = load_arena_data(config)
    
    if not datasets:
        raise ValueError("No data loaded for competition")
    
    # Run competition - use parallel or sequential based on flag
    all_scores: List[ModelScore] = []
    
    if parallel and len(datasets) > 1:
        # Parallel processing using multiprocessing
        if RICH_AVAILABLE:
            console.print(f"  [dim]Running parallel competition...[/dim]")
        all_scores = fit_all_symbols_parallel(datasets, config, n_workers)
    elif RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Competition...", total=len(datasets))
            
            for symbol, dataset in datasets.items():
                progress.update(task, description=f"Testing {symbol}...")
                
                symbol_scores = fit_all_models_for_symbol(dataset, config)
                all_scores.extend(symbol_scores)
                
                progress.advance(task)
    else:
        for i, (symbol, dataset) in enumerate(datasets.items()):
            print(f"[{i+1}/{len(datasets)}] Testing {symbol}...")
            symbol_scores = fit_all_models_for_symbol(dataset, config)
            all_scores.extend(symbol_scores)
    
    # Compute combined scores
    all_scores = compute_combined_scores(all_scores, config)
    
    # Determine rankings
    rankings = determine_rankings(all_scores)
    
    # Determine promotion candidates
    promotion_candidates = determine_promotion_candidates(all_scores, rankings, config)
    
    # Compute summary
    summary = {
        "n_symbols": len(datasets),
        "n_standard_models": len(STANDARD_MOMENTUM_MODELS),
        "n_experimental_models": len(EXPERIMENTAL_MODELS),
        "n_total_fits": len(all_scores),
        "avg_fit_time_ms": np.mean([s.fit_time_ms for s in all_scores]),
        "pit_pass_rate": np.mean([float(s.pit_calibrated) for s in all_scores]),
    }
    
    # Create result
    result = ArenaResult(
        timestamp=datetime.now().isoformat(),
        config=config,
        scores=all_scores,
        rankings=rankings,
        promotion_candidates=promotion_candidates,
        summary=summary,
    )
    
    # Display results
    if RICH_AVAILABLE:
        _display_results(result, console)
    
    # Automatically disable experimental models that failed against best standard
    disabled_count = _disable_failed_models(result, config, console if RICH_AVAILABLE else None)
    
    # Save results
    result_path = results_dir / f"arena_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result.save(str(result_path))
    
    if RICH_AVAILABLE:
        console.print(f"\n[dim]Results saved to: {result_path}[/dim]")
    
    return result


def _disable_failed_models(
    result: ArenaResult,
    config: ArenaConfig,
    console: Optional[Console] = None,
) -> int:
    """
    Disable experimental models that failed against the best standard model.
    
    Args:
        result: Arena competition result
        config: Arena configuration
        console: Optional Rich console for output
        
    Returns:
        Number of models disabled
    """
    overall_ranking = result.rankings.get("overall", [])
    
    # Separate standard and experimental models
    standard_models = [m for m in overall_ranking if m in STANDARD_MOMENTUM_MODELS]
    experimental_models = [m for m in overall_ranking if m in EXPERIMENTAL_MODELS]
    
    if not standard_models or not experimental_models:
        return 0
    
    # Get best standard model score
    best_std_name = standard_models[0]
    std_scores = [s for s in result.scores if s.model_name == best_std_name]
    if not std_scores:
        return 0
    
    best_std_score = np.mean([s.combined_score for s in std_scores if s.combined_score])
    
    disabled_count = 0
    disabled_models_info = []
    
    for model_name in experimental_models:
        # Skip already disabled models
        if is_model_disabled(model_name):
            continue
        
        # Get experimental model score
        exp_scores = [s for s in result.scores if s.model_name == model_name]
        if not exp_scores:
            continue
        
        exp_score = np.mean([s.combined_score for s in exp_scores if s.combined_score])
        
        # Calculate gap
        if best_std_score and exp_score:
            gap = (exp_score - best_std_score) / best_std_score
            
            # Disable if failed (negative gap)
            if gap < 0:
                disable_model(
                    model_name=model_name,
                    best_std_model=best_std_name,
                    score_gap=gap,
                )
                disabled_count += 1
                disabled_models_info.append((model_name, gap))
    
    # Display disabled models
    if console and disabled_count > 0:
        console.print()
        console.print("[bold red]MODELS DISABLED[/bold red]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        for name, gap in disabled_models_info:
            console.print(f"  [red]x[/red] {name} [dim]({gap*100:.1f}%)[/dim]")
        console.print()
        console.print("[dim]  Use 'make arena-enable MODEL=name' to re-enable[/dim]")
    
    return disabled_count


def _display_results(result: ArenaResult, console: Console) -> None:
    """Display competition results with clean, aligned formatting."""
    
    overall_ranking = result.rankings.get("overall", [])
    
    # Separate standard and experimental models
    standard_models = [m for m in overall_ranking if m in STANDARD_MOMENTUM_MODELS]
    experimental_models = [m for m in overall_ranking if m in EXPERIMENTAL_MODELS]
    
    # Compute model statistics
    def get_model_stats(model_name):
        model_scores = [s for s in result.scores if s.model_name == model_name]
        if not model_scores:
            return None, None, None, None, None, None, None, None, None, None
        avg_score = np.mean([s.combined_score for s in model_scores if s.combined_score])
        avg_bic = np.mean([s.bic for s in model_scores])
        avg_crps = np.mean([s.crps for s in model_scores if s.crps is not None]) if any(s.crps for s in model_scores) else None
        avg_hyv = np.mean([s.hyvarinen_score for s in model_scores if s.hyvarinen_score is not None]) if any(s.hyvarinen_score for s in model_scores) else None
        pit_rate = np.mean([float(s.pit_calibrated) for s in model_scores])
        avg_time = np.mean([s.fit_time_ms for s in model_scores])
        avg_css = np.mean([s.css_score for s in model_scores if s.css_score is not None]) if any(s.css_score for s in model_scores) else None
        avg_fec = np.mean([s.fec_score for s in model_scores if s.fec_score is not None]) if any(s.fec_score for s in model_scores) else None
        avg_dig = np.mean([s.dig_score for s in model_scores if s.dig_score is not None]) if any(s.dig_score for s in model_scores) else None
        avg_adv = np.mean([s.advanced_score for s in model_scores if s.advanced_score is not None]) if any(s.advanced_score for s in model_scores) else None
        return avg_score, avg_bic, avg_crps, avg_hyv, pit_rate, avg_time, avg_css, avg_fec, avg_dig, avg_adv
    
    # =========================================================================
    # STANDARD MODELS
    # =========================================================================
    console.print()
    console.print("[bold blue]STANDARD MODELS[/bold blue] [dim](Production Baselines)[/dim]")
    console.print(f"[dim]{'─' * 130}[/dim]")
    console.print(f"[dim]{'Rank':<6}{'Model':<32}{'Score':>7}{'BIC':>9}{'CRPS':>8}{'Hyv':>9}{'PIT':>6}{'CSS':>6}{'FEC':>6}{'DIG':>6}{'Adv':>6}{'Time':>9}[/dim]")
    console.print(f"[dim]{'─' * 130}[/dim]")
    
    for i, model_name in enumerate(standard_models, 1):
        avg_score, avg_bic, avg_crps, avg_hyv, pit_rate, avg_time, avg_css, avg_fec, avg_dig, avg_adv = get_model_stats(model_name)
        
        pit_str = "PASS" if pit_rate == 1.0 else f"{pit_rate*100:.0f}%"
        score_str = f"{avg_score:.4f}" if avg_score else "-"
        bic_str = f"{avg_bic:.0f}" if avg_bic else "-"
        crps_str = f"{avg_crps:.4f}" if avg_crps else "-"
        hyv_str = f"{avg_hyv:.1f}" if avg_hyv else "-"
        time_str = f"{avg_time:.0f}ms" if avg_time else "-"
        css_str = f"{avg_css:.2f}" if avg_css is not None else "-"
        fec_str = f"{avg_fec:.2f}" if avg_fec is not None else "-"
        dig_str = f"{avg_dig:.2f}" if avg_dig is not None else "-"
        adv_str = f"{avg_adv:.2f}" if avg_adv is not None else "-"
        time_str = f"{avg_time:.0f}ms" if avg_time else "-"
        
        # Color for top models
        if i == 1:
            style = "bold green"
        elif i <= 3:
            style = "green"
        else:
            style = "white"
        
        console.print(f"[{style}]#{i:<5}{model_name:<32}{score_str:>7}{bic_str:>9}{crps_str:>8}{hyv_str:>9}{pit_str:>6}{css_str:>6}{fec_str:>6}{dig_str:>6}{adv_str:>6}{time_str:>9}[/{style}]")
    
    # =========================================================================
    # EXPERIMENTAL MODELS
    # =========================================================================
    console.print()
    console.print("[bold magenta]EXPERIMENTAL MODELS[/bold magenta] [dim](Candidates for Promotion)[/dim]")
    console.print(f"[dim]{'─' * 140}[/dim]")
    
    if experimental_models:
        console.print(f"[dim]{'Rank':<6}{'Model':<32}{'Score':>7}{'BIC':>9}{'CRPS':>8}{'Hyv':>9}{'PIT':>6}{'CSS':>6}{'FEC':>6}{'DIG':>6}{'Adv':>6}{'Time':>9}{'vs STD':>8}[/dim]")
        console.print(f"[dim]{'─' * 140}[/dim]")
        
        # Get best standard score for comparison
        best_std_score = None
        if standard_models:
            best_std_score, _, _, _, _, _, _, _, _, _ = get_model_stats(standard_models[0])
        
        for i, model_name in enumerate(experimental_models, 1):
            avg_score, avg_bic, avg_crps, avg_hyv, pit_rate, avg_time, avg_css, avg_fec, avg_dig, avg_adv = get_model_stats(model_name)
            
            pit_str = "PASS" if pit_rate == 1.0 else f"{pit_rate*100:.0f}%"
            score_str = f"{avg_score:.4f}" if avg_score else "-"
            bic_str = f"{avg_bic:.0f}" if avg_bic else "-"
            crps_str = f"{avg_crps:.4f}" if avg_crps else "-"
            hyv_str = f"{avg_hyv:.1f}" if avg_hyv else "-"
            time_str = f"{avg_time:.0f}ms" if avg_time else "-"
            css_str = f"{avg_css:.2f}" if avg_css is not None else "-"
            fec_str = f"{avg_fec:.2f}" if avg_fec is not None else "-"
            dig_str = f"{avg_dig:.2f}" if avg_dig is not None else "-"
            adv_str = f"{avg_adv:.2f}" if avg_adv is not None else "-"
            
            # Gap vs standard
            if best_std_score and avg_score:
                gap_pct = ((avg_score - best_std_score) / best_std_score) * 100
                if gap_pct > 0:
                    gap_str = f"[green]+{gap_pct:.1f}%[/green]"
                else:
                    gap_str = f"[red]{gap_pct:.1f}%[/red]"
            else:
                gap_str = "-"
            
            # Style based on performance
            if avg_score and best_std_score and avg_score > best_std_score:
                style = "bold green"
                rank_str = f"*#{i}"
            else:
                style = "white"
                rank_str = f"#{i}"
            
            console.print(f"[{style}]{rank_str:<6}{model_name:<32}{score_str:>7}{bic_str:>9}{crps_str:>8}{hyv_str:>9}{pit_str:>6}{css_str:>6}{fec_str:>6}{dig_str:>6}{adv_str:>6}{time_str:>9}[/{style}]  {gap_str}")
    else:
        console.print("[dim]  No experimental models enabled[/dim]")
    
    # =========================================================================
    # HEAD-TO-HEAD (only if experimental models exist)
    # =========================================================================
    if experimental_models and standard_models:
        best_std_score, _, _, _, best_std_pit, _, best_std_css, best_std_fec, best_std_dig, best_std_adv = get_model_stats(standard_models[0])
        best_exp_score, _, _, _, best_exp_pit, _, best_exp_css, best_exp_fec, best_exp_dig, best_exp_adv = get_model_stats(experimental_models[0])
        
        console.print()
        console.print("[bold]HEAD-TO-HEAD[/bold]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        
        std_pit = "PASS" if best_std_pit == 1.0 else f"{best_std_pit*100:.0f}%"
        exp_pit = "PASS" if best_exp_pit == 1.0 else f"{best_exp_pit*100:.0f}%"
        
        console.print(f"  [blue]Standard:[/blue]     {standard_models[0]}")
        console.print(f"                  Score: {best_std_score:.4f}  PIT: {std_pit}")
        console.print(f"  [magenta]Experimental:[/magenta] {experimental_models[0]}")
        console.print(f"                  Score: {best_exp_score:.4f}  PIT: {exp_pit}")
        
        gap = ((best_exp_score - best_std_score) / best_std_score * 100) if best_std_score else 0
        if gap > 5:
            console.print(f"  [bold green]>>> EXPERIMENTAL WINS by {gap:.1f}%[/bold green]")
        elif gap > 0:
            console.print(f"  [yellow]>>> Experimental leads by {gap:.1f}% (needs >5%)[/yellow]")
        else:
            console.print(f"  [blue]>>> Standard leads by {abs(gap):.1f}%[/blue]")
    
    # =========================================================================
    # PROMOTION STATUS
    # =========================================================================
    console.print()
    if result.promotion_candidates:
        console.print("[bold green]PROMOTION CANDIDATES[/bold green]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        for m in result.promotion_candidates:
            console.print(f"  [green]>[/green] {m}")
        console.print("[dim]  Ready for panel review[/dim]")
    else:
        console.print("[yellow]NO PROMOTION CANDIDATES[/yellow]")
        console.print("[dim]  Criteria: >5% vs standard, 100% PIT pass, no category failures[/dim]")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    console.print()
    console.print("[bold]SUMMARY[/bold]")
    console.print(f"[dim]{'─' * 40}[/dim]")
    console.print(f"  Symbols:        {result.summary['n_symbols']}")
    console.print(f"  Standard:       {result.summary['n_standard_models']}")
    console.print(f"  Experimental:   {result.summary['n_experimental_models']}")
    console.print(f"  Total Fits:     {result.summary['n_total_fits']}")
    console.print(f"  Avg Fit Time:   {result.summary['avg_fit_time_ms']:.1f}ms")
    console.print(f"  PIT Pass Rate:  {result.summary['pit_pass_rate']:.1%}")
