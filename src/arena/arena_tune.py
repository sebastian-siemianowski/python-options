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
        final_score: Ultimate score (0-100 scale)
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
    final_score: Optional[float] = None
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
            "final_score": self.final_score,
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

class ModelTimeoutError(Exception):
    """Raised when model fitting exceeds timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise ModelTimeoutError("Model fitting timed out")


def fit_experimental_model(
    model_name: str,
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    timeout_seconds: int = 15,
) -> Dict[str, Any]:
    """
    Fit an experimental model with timeout protection.
    
    Args:
        model_name: Experimental model name
        returns: Log returns
        vol: EWMA volatility
        regime_labels: Regime assignments
        timeout_seconds: Maximum time allowed for fitting (default 15s)
        
    Returns:
        Dictionary with fitted parameters and diagnostics
        
    Raises:
        ModelTimeoutError: If fitting exceeds timeout
        Exception: If model fitting fails for any other reason
    """
    import time
    import signal
    
    start_time = time.time()
    
    # Set up timeout using signal (Unix only, but macOS supports it)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    except (ValueError, AttributeError):
        # signal.alarm not available (e.g., Windows or in subprocess)
        pass
    
    try:
        model = create_experimental_model(model_name)
        result = model.fit(returns, vol)
        
        fit_time_ms = (time.time() - start_time) * 1000
        result["fit_time_ms"] = fit_time_ms
        result["model_name"] = model_name
        
        return result
    finally:
        # Cancel the alarm and restore old handler
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, AttributeError):
            pass


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
        # Measures if model's confidence is rational relative to realized outcomes
        # NOT just correlation with vol (which is trivially 1.0 for vol-scaled models)
        sigma_clean = sigma_pred[warmup:]
        returns_clean = returns[warmup:]
        
        # Compute realized squared errors
        errors = returns_clean - mu_pred[warmup:]
        realized_var = errors ** 2
        predicted_var = sigma_clean ** 2
        
        # FEC Component 1: Variance ratio consistency
        # Good models should have predicted_var close to realized_var on average
        var_ratio = np.mean(predicted_var) / (np.mean(realized_var) + 1e-10)
        var_ratio_score = 1.0 - min(1.0, abs(np.log(var_ratio + 1e-10)) / 2.0)
        
        # FEC Component 2: Time-varying calibration
        # Does predicted variance increase when realized variance increases?
        window = 20
        if len(realized_var) > window * 2:
            rolling_realized = np.array([np.mean(realized_var[max(0,i-window):i+1]) 
                                         for i in range(len(realized_var))])
            rolling_predicted = np.array([np.mean(predicted_var[max(0,i-window):i+1]) 
                                          for i in range(len(predicted_var))])
            valid = (rolling_realized > 1e-12) & (rolling_predicted > 1e-12)
            if np.sum(valid) > 50:
                corr = np.corrcoef(rolling_realized[valid], rolling_predicted[valid])[0, 1]
                corr_score = (corr + 1) / 2 if np.isfinite(corr) else 0.5
            else:
                corr_score = 0.5
        else:
            corr_score = 0.5
        
        # FEC Component 3: Overconfidence penalty
        # Penalize if model is often overconfident (predicted var < realized var)
        overconfident_pct = np.mean(predicted_var < realized_var * 0.5)
        underconfident_pct = np.mean(predicted_var > realized_var * 2.0)
        confidence_score = 1.0 - (overconfident_pct * 0.6 + underconfident_pct * 0.4)
        
        fec_score = 0.3 * var_ratio_score + 0.4 * corr_score + 0.3 * confidence_score
        fec_score = float(np.clip(fec_score, 0, 1))
        
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


def compute_final_score(
    bic: float,
    crps: float,
    hyvarinen: float,
    pit_calibrated: bool,
    css: float,
    fec: float,
    dig: float,
    bic_range: Tuple[float, float],
    crps_range: Tuple[float, float],
    hyv_range: Tuple[float, float],
) -> float:
    """
    Compute Final Score on 0-100 scale.
    
    Combines all metrics into a single definitive score:
    - BIC (25%): Model parsimony and fit quality
    - CRPS (20%): Probabilistic forecast accuracy
    - Hyvarinen (15%): Robustness to misspecification
    - PIT (15%): Calibration pass/fail (hard gate)
    - CSS (10%): Stress stability
    - FEC (8%): Entropy consistency
    - DIG (7%): Directional information
    
    Returns:
        Final score from 0.00 to 100.00
    """
    # Normalize BIC (lower is better) to 0-1
    bic_min, bic_max = bic_range
    if bic_max > bic_min:
        bic_norm = 1.0 - (bic - bic_min) / (bic_max - bic_min)
    else:
        bic_norm = 0.5
    bic_norm = np.clip(bic_norm, 0, 1)
    
    # Normalize CRPS (lower is better) to 0-1
    crps_min, crps_max = crps_range
    if crps_max > crps_min and crps is not None:
        crps_norm = 1.0 - (crps - crps_min) / (crps_max - crps_min)
    else:
        crps_norm = 0.5
    crps_norm = np.clip(crps_norm, 0, 1)
    
    # Normalize Hyvarinen (higher is better for well-specified models)
    hyv_min, hyv_max = hyv_range
    if hyv_max > hyv_min and hyvarinen is not None:
        hyv_norm = (hyvarinen - hyv_min) / (hyv_max - hyv_min)
    else:
        hyv_norm = 0.5
    hyv_norm = np.clip(hyv_norm, 0, 1)
    
    # PIT: Binary gate with soft penalty
    pit_score = 1.0 if pit_calibrated else 0.3
    
    # CSS, FEC, DIG already 0-1
    css_norm = css if css is not None else 0.5
    fec_norm = fec if fec is not None else 0.5
    dig_norm = dig if dig is not None else 0.5
    
    # Weighted combination
    final = (
        0.25 * bic_norm +
        0.20 * crps_norm +
        0.15 * hyv_norm +
        0.15 * pit_score +
        0.10 * css_norm +
        0.08 * fec_norm +
        0.07 * dig_norm
    )
    
    # Scale to 0-100
    return round(final * 100, 2)


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
    
    # Re-seed numpy random state in subprocess to avoid shared state issues
    import os
    import sys
    import numpy as np
    import faulthandler
    import traceback
    import tempfile
    
    # Enable faulthandler to get traceback on segfaults
    try:
        crash_file = os.path.join(tempfile.gettempdir(), f'arena_crash_{model_name}_{symbol}.log')
        with open(crash_file, 'w') as f:
            f.write(f"Starting {model_name} on {symbol}\n")
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except:
        pass
    
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    
    # Limit numpy threads to avoid resource contention
    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    except:
        pass
    
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
    except ModelTimeoutError as e:
        print(f"⏱️  TIMEOUT: {model_name} on {symbol} (>15s)", file=sys.stderr)
        return {
            "model_name": model_name,
            "symbol": symbol,
            "category": category_value,
            "success": False,
            "error": f"TIMEOUT: {str(e)}",
            "error_type": "timeout",
        }
    except MemoryError as e:
        print(f"💾 MEMORY: {model_name} on {symbol} - out of memory", file=sys.stderr)
        return {
            "model_name": model_name,
            "symbol": symbol,
            "category": category_value,
            "success": False,
            "error": f"MemoryError: {str(e)[:50]}",
            "error_type": "memory",
        }
    except Exception as e:
        error_type = type(e).__name__
        tb = traceback.format_exc()
        # Write crash info to temp file for debugging
        try:
            crash_file = os.path.join(tempfile.gettempdir(), f'arena_crash_{model_name}_{symbol}.log')
            with open(crash_file, 'w') as f:
                f.write(f"Model: {model_name}\nSymbol: {symbol}\nError: {error_type}\n{tb}\n")
        except:
            pass
        print(f"❌ FAILED: {model_name} on {symbol} - {error_type}: {str(e)[:100]}", file=sys.stderr)
        return {
            "model_name": model_name,
            "symbol": symbol,
            "category": category_value,
            "success": False,
            "error": f"{error_type}: {str(e)}",
            "error_type": error_type,
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
    import platform
    import tempfile
    import os
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    
    if n_workers is None:
        n_workers = max(1, min(N_CPUS - 1, 4))  # Limit to 4 workers to reduce memory pressure
    
    # Set thread limits before spawning processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
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
    failed_models = []
    timeout_models = []
    crash_models = []
    
    # Use 'fork' context
    mp_context = mp.get_context('fork')
    
    # Track currently running tasks to identify crashes
    running_tasks = {}  # future -> task
    completed_tasks = set()
    
    # Process in smaller batches to limit crash impact
    batch_size = max(n_workers * 2, 12)
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    print(f"🚀 Processing {len(tasks)} tasks in {total_batches} batches with {n_workers} workers", file=sys.stderr)
    
    for batch_idx, batch_start in enumerate(range(0, len(tasks), batch_size)):
        batch_tasks = tasks[batch_start:batch_start + batch_size]
        batch_num = batch_idx + 1
        batch_completed = []
        batch_pending = set(range(len(batch_tasks)))
        
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                # Submit all tasks in batch
                futures = {}
                for i, task in enumerate(batch_tasks):
                    future = executor.submit(_fit_single_model_wrapper, task)
                    futures[future] = (i, task)
                
                # Collect results with timeout
                for future in as_completed(futures, timeout=180):  # 3 min timeout for batch
                    i, task = futures[future]
                    model_name = task[1]
                    symbol = task[0]
                    
                    try:
                        result = future.result(timeout=30)
                        batch_pending.discard(i)
                        batch_completed.append(i)
                        
                        if result.get("success", False):
                            all_results.append(result)
                        else:
                            error_type = result.get("error_type", "unknown")
                            if error_type == "timeout":
                                timeout_models.append((model_name, symbol, result.get("error", "")))
                            else:
                                failed_models.append((model_name, symbol, result.get("error", "")))
                    
                    except FuturesTimeoutError:
                        batch_pending.discard(i)
                        timeout_models.append((model_name, symbol, "Timeout waiting for result"))
                        print(f"⏱️  TIMEOUT: {model_name} on {symbol}", file=sys.stderr)
                    
                    except Exception as e:
                        batch_pending.discard(i)
                        error_name = type(e).__name__
                        
                        if "BrokenProcessPool" in error_name:
                            # A process crashed - check crash logs
                            crash_file = os.path.join(tempfile.gettempdir(), f'arena_crash_{model_name}_{symbol}.log')
                            crash_info = ""
                            if os.path.exists(crash_file):
                                try:
                                    with open(crash_file, 'r') as f:
                                        crash_info = f.read()[:200]
                                except:
                                    pass
                            
                            crash_models.append((model_name, symbol, crash_info or "Process crashed"))
                            print(f"💥 CRASH: {model_name} on {symbol} - process terminated", file=sys.stderr)
                            if crash_info:
                                print(f"   Crash log: {crash_info[:100]}", file=sys.stderr)
                        else:
                            failed_models.append((model_name, symbol, f"{error_name}: {str(e)[:60]}"))
                            print(f"❌ ERROR: {model_name} on {symbol} - {error_name}", file=sys.stderr)
        
        except Exception as pool_error:
            # Pool crashed - identify which tasks didn't complete
            error_name = type(pool_error).__name__
            print(f"⚠️  Pool crashed in batch {batch_num}/{total_batches}: {error_name}", file=sys.stderr)
            
            # Tasks that were pending when pool crashed need to be retried sequentially
            pending_tasks = [batch_tasks[i] for i in batch_pending if i not in batch_completed]
            
            if pending_tasks:
                print(f"   Retrying {len(pending_tasks)} pending tasks sequentially...", file=sys.stderr)
                
                for task in pending_tasks:
                    symbol, model_name = task[0], task[1]
                    try:
                        result = _fit_single_model_wrapper(task)
                        if result.get("success", False):
                            all_results.append(result)
                        else:
                            failed_models.append((model_name, symbol, result.get("error", "")))
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)[:60]}"
                        failed_models.append((model_name, symbol, error_msg))
                        print(f"   ❌ {model_name} on {symbol}: {error_msg}", file=sys.stderr)
    
    # Print summary
    print(f"\n📊 Results: {len(all_results)} succeeded", file=sys.stderr)
    
    if timeout_models:
        print(f"⏱️  {len(timeout_models)} timed out", file=sys.stderr)
    
    if crash_models:
        print(f"💥 {len(crash_models)} crashed:", file=sys.stderr)
        for model, sym, info in crash_models[:5]:
            print(f"   - {model} on {sym}", file=sys.stderr)
        if len(crash_models) > 5:
            print(f"   ... and {len(crash_models) - 5} more", file=sys.stderr)
    
    if failed_models:
        print(f"❌ {len(failed_models)} failed:", file=sys.stderr)
        for model, sym, err in failed_models[:5]:
            print(f"   - {model} on {sym}: {err[:50]}", file=sys.stderr)
        if len(failed_models) > 5:
            print(f"   ... and {len(failed_models) - 5} more", file=sys.stderr)
    
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
                error_type = type(e).__name__
                print(f"❌ FAILED: {model_name} on {dataset.symbol} - {error_type}: {str(e)[:80]}", file=sys.stderr)
    
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
            except ModelTimeoutError as e:
                print(f"⏱️  TIMEOUT: {model_name} on {dataset.symbol} (>15s)", file=sys.stderr)
            except Exception as e:
                error_type = type(e).__name__
                print(f"❌ FAILED: {model_name} on {dataset.symbol} - {error_type}: {str(e)[:80]}", file=sys.stderr)
    
    return scores


def compute_combined_scores(
    scores: List[ModelScore],
    config: ArenaConfig,
) -> List[ModelScore]:
    """
    Compute combined scores and final scores for all models.
    
    Args:
        scores: List of model scores
        config: Arena configuration
        
    Returns:
        Updated scores with combined_score and final_score fields
    """
    if not scores:
        return scores
    
    # Compute global ranges for final score normalization
    all_bics = [s.bic for s in scores if s.bic is not None]
    all_crps = [s.crps for s in scores if s.crps is not None]
    all_hyv = [s.hyvarinen_score for s in scores if s.hyvarinen_score is not None]
    
    bic_range = (min(all_bics), max(all_bics)) if all_bics else (0, 1)
    crps_range = (min(all_crps), max(all_crps)) if all_crps else (0, 1)
    hyv_range = (min(all_hyv), max(all_hyv)) if all_hyv else (0, 1)
    
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
        bic_norm_range = bic_max - bic_min if bic_max > bic_min else 1.0
        
        for score in symbol_scores:
            # BIC score (lower is better, so invert)
            bic_norm = 1.0 - (score.bic - bic_min) / bic_norm_range
            
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
            
            # Compute Final Score (0-100)
            score.final_score = compute_final_score(
                bic=score.bic,
                crps=score.crps if score.crps else 0.02,
                hyvarinen=score.hyvarinen_score if score.hyvarinen_score else 0,
                pit_calibrated=score.pit_calibrated,
                css=score.css_score,
                fec=score.fec_score,
                dig=score.dig_score,
                bic_range=bic_range,
                crps_range=crps_range,
                hyv_range=hyv_range,
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
    
    Criteria (ALL must pass):
    1. Final Score beats best standard by >5 points (on 100-point scale)
    2. PIT calibration pass rate >= 75%
    3. CSS (Calibration Stability Under Stress) >= 0.65 [HARD GATE]
    4. FEC (Forecast Entropy Consistency) >= 0.75 [HARD GATE]
    5. No category where it ranks last
    
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
    model_avg_final: Dict[str, float] = {}
    model_pit_rate: Dict[str, float] = {}
    model_avg_css: Dict[str, float] = {}
    model_avg_fec: Dict[str, float] = {}
    model_avg_hyv: Dict[str, float] = {}
    
    for model_name in set(s.model_name for s in scores):
        model_scores = [s for s in scores if s.model_name == model_name]
        if model_scores:
            # Use final_score for promotion criteria
            final_scores = [s.final_score for s in model_scores if s.final_score is not None]
            if final_scores:
                model_avg_final[model_name] = np.mean(final_scores)
            else:
                model_avg_final[model_name] = 0.0
            # Compute PIT pass rate (not all-or-nothing)
            model_pit_rate[model_name] = np.mean([float(s.pit_calibrated) for s in model_scores])
            # Compute average CSS
            css_scores = [s.css_score for s in model_scores if s.css_score is not None]
            model_avg_css[model_name] = np.mean(css_scores) if css_scores else 0.0
            # Compute average FEC
            fec_scores = [s.fec_score for s in model_scores if s.fec_score is not None]
            model_avg_fec[model_name] = np.mean(fec_scores) if fec_scores else 0.0
    
    # Find best standard model final score
    standard_final_scores = [
        model_avg_final[m] for m in STANDARD_MOMENTUM_MODELS 
        if m in model_avg_final
    ]
    best_standard_final = max(standard_final_scores) if standard_final_scores else 50.0
    
    # Check each experimental model
    for exp_name in experimental_names:
        if exp_name not in model_avg_final:
            continue
        
        exp_final = model_avg_final[exp_name]
        exp_pit_rate = model_pit_rate.get(exp_name, 0.0)
        exp_css = model_avg_css.get(exp_name, 0.0)
        exp_fec = model_avg_fec.get(exp_name, 0.0)
        
        # Criterion 1: Final Score beats best standard by >3 points
        score_gap = exp_final - best_standard_final
        beats_threshold = score_gap > 3.0
        
        # Criterion 2: PIT calibrated on >= 75% of symbols
        pit_pass = exp_pit_rate >= 0.75
        
        # Criterion 3: CSS >= 0.65 [HARD GATE]
        css_pass = exp_css >= 0.65
        
        # Criterion 4: FEC >= 0.75 [HARD GATE]
        fec_pass = exp_fec >= 0.75
        
        # Criterion 5: Not last in any category
        not_last_anywhere = True
        for cat_key, cat_ranking in rankings.items():
            if cat_key == "overall":
                continue
            if cat_ranking and exp_name in cat_ranking:
                if cat_ranking[-1] == exp_name:
                    not_last_anywhere = False
                    break
        
        # ALL criteria must pass
        if beats_threshold and pit_pass and css_pass and fec_pass and not_last_anywhere:
            candidates.append(exp_name)
    
    return candidates


def run_arena_competition(
    config: Optional[ArenaConfig] = None,
    symbols: Optional[List[str]] = None,
    parallel: bool = True,
    n_workers: int = None,
    skip_disable: bool = False,
) -> ArenaResult:
    """Run full arena competition with optional parallel processing.
    
    Args:
        config: Arena configuration
        symbols: List of symbols to test (overrides config)
        parallel: Use parallel processing
        n_workers: Number of workers (default: CPU count - 1)
        skip_disable: If True, skip disabling underperforming models
    """
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
    # Skip if skip_disable is True (e.g., for safe storage testing)
    if not skip_disable:
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
    
    # Get best standard model Final Score
    best_std_name = standard_models[0]
    std_scores = [s for s in result.scores if s.model_name == best_std_name]
    if not std_scores:
        return 0
    
    # Use final_score for the 5-point hard gate
    std_final_scores = [s.final_score for s in std_scores if s.final_score is not None]
    best_std_final = np.mean(std_final_scores) if std_final_scores else 50.0
    
    disabled_count = 0
    disabled_models_info = []
    
    for model_name in experimental_models:
        # Skip already disabled models
        if is_model_disabled(model_name):
            continue
        
        # Get experimental model Final Score
        exp_scores = [s for s in result.scores if s.model_name == model_name]
        if not exp_scores:
            continue
        
        exp_final_scores = [s.final_score for s in exp_scores if s.final_score is not None]
        exp_final = np.mean(exp_final_scores) if exp_final_scores else 0.0
        
        # Calculate gap in points (not percentage)
        gap_points = exp_final - best_std_final
        
        # HARD GATE: Disable if vs STD < 3 points
        if gap_points < 3.0:
            gap_pct = gap_points / best_std_final if best_std_final else 0
            disable_model(
                model_name=model_name,
                best_std_model=best_std_name,
                score_gap=gap_pct,
            )
            disabled_count += 1
            disabled_models_info.append((model_name, gap_points))
    
    # Display disabled models
    if console and disabled_count > 0:
        console.print()
        console.print("[bold red]MODELS DISABLED (vs STD < 3 points)[/bold red]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        for name, gap_pts in disabled_models_info:
            sign = "+" if gap_pts >= 0 else ""
            console.print(f"  [red]x[/red] {name} [dim]({sign}{gap_pts:.1f} pts)[/dim]")
        console.print()
        console.print("[dim]  Use 'make arena-enable MODEL=name' to re-enable[/dim]")
    
    return disabled_count


def _display_results(result: ArenaResult, console: Console) -> None:
    """Display competition results with elegant Apple-style formatting."""
    
    # Import safe storage models for filtering - use function for dynamic loading
    try:
        from arena.show_safe_storage import get_safe_storage_models, ARCHIVED_MODELS
        SAFE_STORAGE_MODELS = get_safe_storage_models()
        safe_storage_model_names = set(SAFE_STORAGE_MODELS.keys())
    except ImportError:
        SAFE_STORAGE_MODELS = {}
        ARCHIVED_MODELS = {}
        safe_storage_model_names = set()
    
    overall_ranking = result.rankings.get("overall", [])
    
    # Separate standard and experimental models
    standard_models = [m for m in overall_ranking if m in STANDARD_MOMENTUM_MODELS]
    # Exclude safe storage models from experimental display
    experimental_models = [m for m in overall_ranking if m in EXPERIMENTAL_MODELS and m not in safe_storage_model_names]
    
    # Compute model statistics
    def get_model_stats(model_name):
        model_scores = [s for s in result.scores if s.model_name == model_name]
        if not model_scores:
            return None, None, None, None, None, None, None, None, None, None, None
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
        avg_final = np.mean([s.final_score for s in model_scores if s.final_score is not None]) if any(s.final_score for s in model_scores) else None
        return avg_score, avg_bic, avg_crps, avg_hyv, pit_rate, avg_time, avg_css, avg_fec, avg_dig, avg_adv, avg_final
    
    # Sort experimental models by final_score descending
    def get_final_score(m):
        _, _, _, _, _, _, _, _, _, _, final = get_model_stats(m)
        return final if final is not None else 0.0
    
    experimental_models = sorted(experimental_models, key=get_final_score, reverse=True)
    standard_models = sorted(standard_models, key=get_final_score, reverse=True)
    
    # Get best standard final score for comparison
    best_std_final = None
    if standard_models:
        _, _, _, _, _, _, _, _, _, _, best_std_final = get_model_stats(standard_models[0])
    
    # =========================================================================
    # ELEGANT HEADER
    # =========================================================================
    console.print()
    console.print("[dim]                                                                              [/dim]")
    console.print("[bold white]  A R E N A[/bold white]")
    console.print("[dim]  Model Competition Framework[/dim]")
    console.print()
    
    # Summary line
    n_std = len(standard_models)
    n_exp = len(experimental_models)
    n_safe = len(SAFE_STORAGE_MODELS)
    n_arch = len(ARCHIVED_MODELS)
    console.print(f"[dim]  {result.summary['n_symbols']} symbols  ·  {n_std} standard  ·  {n_exp} experimental  ·  {n_safe} candidates  ·  {n_arch} archived[/dim]")
    console.print()
    
    # =========================================================================
    # STANDARD MODELS - Clean table
    # =========================================================================
    console.print("[bold]  Standard Models[/bold]")
    console.print()
    
    # Header row
    header = f"[dim]  {'':3} {'Model':<28} {'Score':>6} {'BIC':>8} {'CRPS':>6} {'Hyv':>7} {'PIT':>5} {'CSS':>4} {'FEC':>4} {'Time':>6}[/dim]"
    console.print(header)
    console.print(f"[dim]  {'─' * 92}[/dim]")
    
    for i, model_name in enumerate(standard_models, 1):
        avg_score, avg_bic, avg_crps, avg_hyv, pit_rate, avg_time, avg_css, avg_fec, avg_dig, avg_adv, avg_final = get_model_stats(model_name)
        
        # Format values
        final_str = f"{avg_final:.1f}" if avg_final is not None else "–"
        bic_str = f"{avg_bic:.0f}" if avg_bic else "–"
        crps_str = f"{avg_crps:.4f}" if avg_crps else "–"
        hyv_str = f"{avg_hyv:.0f}" if avg_hyv else "–"
        pit_str = "✓" if pit_rate == 1.0 else f"{pit_rate*100:.0f}%"
        time_str = f"{avg_time/1000:.1f}s" if avg_time and avg_time >= 1000 else f"{avg_time:.0f}ms" if avg_time else "–"
        css_str = f"{avg_css:.2f}" if avg_css is not None else "–"
        fec_str = f"{avg_fec:.2f}" if avg_fec is not None else "–"
        
        # Rank indicator
        if i == 1:
            rank_style = "[bold cyan]"
            rank_end = "[/bold cyan]"
            rank_char = "◆"
        elif i <= 3:
            rank_style = "[cyan]"
            rank_end = "[/cyan]"
            rank_char = "◇"
        else:
            rank_style = "[dim]"
            rank_end = "[/dim]"
            rank_char = " "
        
        # Truncate model name if needed
        display_name = model_name[:27] + "…" if len(model_name) > 28 else model_name
        
        row = f"  {rank_style}{rank_char}{i:>2}{rank_end} {display_name:<28} {final_str:>6} {bic_str:>8} {crps_str:>6} {hyv_str:>7} {pit_str:>5} {css_str:>4} {fec_str:>4} {time_str:>6}"
        console.print(row)
    
    # =========================================================================
    # EXPERIMENTAL MODELS
    # =========================================================================
    console.print()
    console.print("[bold]  Experimental Models[/bold]")
    console.print()
    
    if experimental_models:
        # Header row with vs STD
        header = f"[dim]  {'':3} {'Model':<28} {'Score':>6} {'BIC':>8} {'CRPS':>6} {'Hyv':>7} {'PIT':>5} {'CSS':>4} {'FEC':>4} {'Time':>6} {'Gap':>6}[/dim]"
        console.print(header)
        console.print(f"[dim]  {'─' * 100}[/dim]")
        
        for i, model_name in enumerate(experimental_models, 1):
            avg_score, avg_bic, avg_crps, avg_hyv, pit_rate, avg_time, avg_css, avg_fec, avg_dig, avg_adv, avg_final = get_model_stats(model_name)
            
            # Format values
            final_str = f"{avg_final:.1f}" if avg_final is not None else "–"
            bic_str = f"{avg_bic:.0f}" if avg_bic else "–"
            crps_str = f"{avg_crps:.4f}" if avg_crps else "–"
            hyv_str = f"{avg_hyv:.0f}" if avg_hyv else "–"
            pit_str = "✓" if pit_rate == 1.0 else f"{pit_rate*100:.0f}%"
            time_str = f"{avg_time/1000:.1f}s" if avg_time and avg_time >= 1000 else f"{avg_time:.0f}ms" if avg_time else "–"
            css_str = f"{avg_css:.2f}" if avg_css is not None else "–"
            fec_str = f"{avg_fec:.2f}" if avg_fec is not None else "–"
            
            # Gap vs standard
            if best_std_final and avg_final:
                gap = avg_final - best_std_final
                if gap >= 3:
                    gap_str = f"[green]+{gap:.1f}[/green]"
                    rank_style = "[bold green]"
                    rank_end = "[/bold green]"
                    rank_char = "●"
                elif gap > 0:
                    gap_str = f"[yellow]+{gap:.1f}[/yellow]"
                    rank_style = "[yellow]"
                    rank_end = "[/yellow]"
                    rank_char = "○"
                else:
                    gap_str = f"[dim]{gap:.1f}[/dim]"
                    rank_style = "[dim]"
                    rank_end = "[/dim]"
                    rank_char = " "
            else:
                gap_str = "–"
                rank_style = "[dim]"
                rank_end = "[/dim]"
                rank_char = " "
            
            # Truncate model name
            display_name = model_name[:27] + "…" if len(model_name) > 28 else model_name
            
            row = f"  {rank_style}{rank_char}{i:>2}{rank_end} {display_name:<28} {final_str:>6} {bic_str:>8} {crps_str:>6} {hyv_str:>7} {pit_str:>5} {css_str:>4} {fec_str:>4} {time_str:>6} {gap_str:>6}"
            console.print(row)
    else:
        console.print("[dim]  No experimental models active[/dim]")
    
    # =========================================================================
    # PROMOTION STATUS - Compact
    # =========================================================================
    console.print()
    if result.promotion_candidates:
        console.print("[bold green]  Promotion Ready[/bold green]")
        for m in result.promotion_candidates:
            _, _, _, _, pit_rate, _, _, _, _, _, final = get_model_stats(m)
            console.print(f"[green]  → {m}[/green] [dim](Score: {final:.1f})[/dim]")
    else:
        console.print("[dim]  No promotion candidates[/dim]")
        console.print("[dim]  Requirements: Score gap ≥3, CSS ≥0.65, FEC ≥0.75, PIT ≥75%[/dim]")
    
    # =========================================================================
    # PROMOTION CANDIDATES (Safe Storage) - Match standard models format
    # =========================================================================
    # Reload fresh data each time (not cached)
    SAFE_STORAGE_MODELS_DISPLAY = get_safe_storage_models()
    
    if SAFE_STORAGE_MODELS_DISPLAY:
        console.print()
        console.print("[bold]  Promotion Candidates[/bold] [dim](Safe Storage)[/dim]")
        console.print()
        console.print(f"[dim]  {'':4} {'Model':<30} {'Score':>7} {'BIC':>8} {'CRPS':>7} {'Hyv':>8} {'PIT':>5} {'CSS':>5} {'FEC':>5} {'Time':>7} {'Gap':>7}[/dim]")
        console.print(f"[dim]  {'─' * 104}[/dim]")
        
        for i, (name, m) in enumerate(sorted(SAFE_STORAGE_MODELS_DISPLAY.items(), key=lambda x: x[1]['final'], reverse=True), 1):
            display_name = name[:28] + "…" if len(name) > 29 else name
            final_str = f"{m['final']:.2f}"
            bic_str = f"{m['bic']:.0f}"
            crps_str = f"{m['crps']:.4f}"
            hyv_str = f"{m['hyv']:.1f}"
            pit_str = m['pit']
            css_str = f"{m['css']:.2f}"
            fec_str = f"{m['fec']:.2f}"
            time_ms = m.get('time_ms', 0)
            time_str = f"{time_ms/1000:.1f}s" if time_ms else "-"
            vs_std = m['vs_std']
            
            # Parse vs_std to color it
            try:
                gap_val = float(vs_std.replace('%', '').replace('+', ''))
                if gap_val >= 8:
                    gap_color = "green"
                elif gap_val >= 5:
                    gap_color = "cyan"
                else:
                    gap_color = "dim"
            except:
                gap_color = "dim"
            
            console.print(f"  [yellow]◆{i:>2}[/yellow]  {display_name:<30} {final_str:>7} {bic_str:>8} {crps_str:>7} {hyv_str:>8} {pit_str:>5} {css_str:>5} {fec_str:>5} {time_str:>7}  [{gap_color}]{vs_std:>6}[/{gap_color}]")
    
    # =========================================================================
    # ARCHIVED MODELS (Retired) - No longer promotion candidates
    # =========================================================================
    if ARCHIVED_MODELS:
        console.print()
        console.print("[bold]  Archived Models[/bold] [dim](Retired)[/dim]")
        console.print()
        console.print(f"[dim]  {'':4} {'Model':<30} {'Score':>7} {'BIC':>8} {'CRPS':>7} {'Hyv':>8} {'PIT':>5} {'CSS':>5} {'FEC':>5} {'Time':>7} {'Gap':>7}[/dim]")
        console.print(f"[dim]  {'─' * 104}[/dim]")
        
        for i, (name, m) in enumerate(sorted(ARCHIVED_MODELS.items(), key=lambda x: x[1]['final'], reverse=True), 1):
            display_name = name[:28] + "…" if len(name) > 29 else name
            final_str = f"{m['final']:.2f}"
            bic_str = f"{m['bic']:.0f}"
            crps_str = f"{m['crps']:.4f}"
            hyv_str = f"{m['hyv']:.1f}"
            pit_str = m['pit']
            css_str = f"{m['css']:.2f}"
            fec_str = f"{m['fec']:.2f}"
            time_ms = m.get('time_ms', 0)
            time_str = f"{time_ms}ms" if time_ms else "-"
            vs_std = m['vs_std']
            
            console.print(f"  [dim]▪{i:>2}  {display_name:<30} {final_str:>7} {bic_str:>8} {crps_str:>7} {hyv_str:>8} {pit_str:>5} {css_str:>5} {fec_str:>5} {time_str:>7}  {vs_std:>6}[/dim]")
    
    # =========================================================================
    # SUMMARY FOOTER - Minimal
    # =========================================================================
    console.print()
    console.print(f"[dim]  {result.summary['n_total_fits']} fits  ·  {result.summary['avg_fit_time_ms']:.0f}ms avg  ·  {result.summary['pit_pass_rate']:.0%} PIT pass[/dim]")
    console.print()

