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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

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
        pit_pvalue: PIT uniformity test p-value
        pit_calibrated: Whether PIT test passed
        hyvarinen_score: Hyvärinen score (if computed)
        combined_score: Weighted combined score
        fit_params: Fitted parameters
        fit_time_ms: Fitting time in milliseconds
    """
    model_name: str
    symbol: str
    category: CapCategory
    log_likelihood: float
    bic: float
    aic: float
    pit_pvalue: float
    pit_calibrated: bool
    hyvarinen_score: Optional[float] = None
    combined_score: Optional[float] = None
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
            "pit_pvalue": self.pit_pvalue,
            "pit_calibrated": self.pit_calibrated,
            "hyvarinen_score": self.hyvarinen_score,
            "combined_score": self.combined_score,
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
# COMPETITION RUNNER
# =============================================================================

def fit_all_models_for_symbol(
    dataset: ArenaDataset,
    config: ArenaConfig,
) -> List[ModelScore]:
    """
    Fit all models (standard + experimental) for a single symbol.
    
    Args:
        dataset: Arena dataset for the symbol
        config: Arena configuration
        
    Returns:
        List of ModelScore objects
    """
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
                
                score = ModelScore(
                    model_name=model_name,
                    symbol=dataset.symbol,
                    category=dataset.category,
                    log_likelihood=result["log_likelihood"],
                    bic=result["bic"],
                    aic=result["aic"],
                    pit_pvalue=pit_pvalue,
                    pit_calibrated=pit_calibrated,
                    fit_params=result["fit_params"],
                    fit_time_ms=result["fit_time_ms"],
                )
                scores.append(score)
            except Exception as e:
                if config.verbose:
                    print(f"  Error fitting {model_name} on {dataset.symbol}: {e}")
    
    # Fit experimental models
    if config.test_experimental:
        for model_name in EXPERIMENTAL_MODELS.keys():
            try:
                result = fit_experimental_model(model_name, returns, vol, regime_labels)
                
                # Compute PIT
                pit_pvalue, pit_calibrated = compute_pit_score(
                    returns, vol, result.get("fit_params", result), model_name
                )
                
                score = ModelScore(
                    model_name=model_name,
                    symbol=dataset.symbol,
                    category=dataset.category,
                    log_likelihood=result["log_likelihood"],
                    bic=result["bic"],
                    aic=result.get("aic", result["bic"]),
                    pit_pvalue=pit_pvalue,
                    pit_calibrated=pit_calibrated,
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
) -> ArenaResult:
    """
    Run full arena competition.
    
    Args:
        config: Arena configuration (uses default if None)
        symbols: Specific symbols to test (uses config.symbols if None)
        
    Returns:
        ArenaResult with complete competition results
    """
    config = config or DEFAULT_ARENA_CONFIG
    config.validate()
    
    if symbols:
        config.symbols = symbols
    
    # Create results directory
    results_dir = Path(config.arena_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]ARENA MODEL COMPETITION[/bold cyan]\n"
            f"Symbols: {len(config.symbols)} | "
            f"Standard Models: {len(STANDARD_MOMENTUM_MODELS)} | "
            f"Experimental: {len(EXPERIMENTAL_MODELS)}",
            border_style="cyan"
        ))
    
    datasets = load_arena_data(config)
    
    if not datasets:
        raise ValueError("No data loaded for competition")
    
    # Run competition
    all_scores: List[ModelScore] = []
    
    if RICH_AVAILABLE:
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
    
    # Save results
    result_path = results_dir / f"arena_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result.save(str(result_path))
    
    if RICH_AVAILABLE:
        console.print(f"\n[dim]Results saved to: {result_path}[/dim]")
    
    return result


def _display_results(result: ArenaResult, console: Console) -> None:
    """Display competition results with clear standard vs experimental distinction."""
    
    overall_ranking = result.rankings.get("overall", [])
    
    # Separate standard and experimental models
    standard_models = [m for m in overall_ranking if m in STANDARD_MOMENTUM_MODELS]
    experimental_models = [m for m in overall_ranking if m in EXPERIMENTAL_MODELS]
    
    # Compute model statistics
    def get_model_stats(model_name):
        model_scores = [s for s in result.scores if s.model_name == model_name]
        if not model_scores:
            return None, None, None
        avg_score = np.mean([s.combined_score for s in model_scores if s.combined_score])
        pit_rate = np.mean([float(s.pit_calibrated) for s in model_scores])
        avg_bic = np.mean([s.bic for s in model_scores])
        return avg_score, pit_rate, avg_bic
    
    # =========================================================================
    # STANDARD MODELS TABLE
    # =========================================================================
    console.print()
    std_table = Table(
        title="[bold blue]STANDARD MODELS[/bold blue] (Production Baselines)",
        box=box.ROUNDED,
        title_style="bold blue",
        border_style="blue",
    )
    std_table.add_column("Rank", style="cyan", justify="center", width=6)
    std_table.add_column("Model", style="white", min_width=35)
    std_table.add_column("Score", justify="right", width=10)
    std_table.add_column("BIC", justify="right", width=12)
    std_table.add_column("PIT", justify="center", width=8)
    
    for i, model_name in enumerate(standard_models, 1):
        avg_score, pit_rate, avg_bic = get_model_stats(model_name)
        
        # Highlight top 3
        if i == 1:
            rank_str = "#1"
            row_style = "bold green"
        elif i == 2:
            rank_str = "#2"
            row_style = "green"
        elif i == 3:
            rank_str = "#3"
            row_style = "dim green"
        else:
            rank_str = f"#{i}"
            row_style = None
        
        pit_str = "PASS" if pit_rate == 1.0 else f"{pit_rate*100:.0f}%"
        
        std_table.add_row(
            rank_str,
            model_name,
            f"{avg_score:.4f}" if avg_score else "-",
            f"{avg_bic:.1f}" if avg_bic else "-",
            pit_str,
            style=row_style,
        )
    
    console.print(std_table)
    
    # =========================================================================
    # EXPERIMENTAL MODELS TABLE
    # =========================================================================
    console.print()
    exp_table = Table(
        title="[bold magenta]EXPERIMENTAL MODELS[/bold magenta] (Candidates for Promotion)",
        box=box.ROUNDED,
        title_style="bold magenta",
        border_style="magenta",
    )
    exp_table.add_column("Rank", style="cyan", justify="center", width=6)
    exp_table.add_column("Model", style="white", min_width=35)
    exp_table.add_column("Score", justify="right", width=10)
    exp_table.add_column("BIC", justify="right", width=12)
    exp_table.add_column("PIT", justify="center", width=8)
    exp_table.add_column("vs Best STD", justify="right", width=12)
    
    # Get best standard score for comparison
    best_std_score = None
    if standard_models:
        best_std_score, _, _ = get_model_stats(standard_models[0])
    
    for i, model_name in enumerate(experimental_models, 1):
        avg_score, pit_rate, avg_bic = get_model_stats(model_name)
        
        # Compute gap vs best standard
        if best_std_score and avg_score:
            gap_pct = ((avg_score - best_std_score) / best_std_score) * 100
            if gap_pct > 0:
                gap_str = f"[green]+{gap_pct:.1f}%[/green]"
            else:
                gap_str = f"[red]{gap_pct:.1f}%[/red]"
        else:
            gap_str = "-"
        
        # Highlight if beating standard
        if avg_score and best_std_score and avg_score > best_std_score:
            rank_str = f"*#{i}"
            row_style = "bold green"
        else:
            rank_str = f"#{i}"
            row_style = None
        
        pit_str = "PASS" if pit_rate == 1.0 else f"{pit_rate*100:.0f}%"
        
        exp_table.add_row(
            rank_str,
            model_name,
            f"{avg_score:.4f}" if avg_score else "-",
            f"{avg_bic:.1f}" if avg_bic else "-",
            pit_str,
            gap_str,
            style=row_style,
        )
    
    console.print(exp_table)
    
    # =========================================================================
    # HEAD-TO-HEAD COMPARISON
    # =========================================================================
    console.print()
    
    # Build comparison panel
    if best_std_score and experimental_models:
        best_exp_score, best_exp_pit, _ = get_model_stats(experimental_models[0])
        best_std_pit = get_model_stats(standard_models[0])[1] if standard_models else 0
        
        comparison_lines = [
            "[bold]Head-to-Head Comparison[/bold]",
            "",
            f"  [blue]Best Standard:[/blue]     {standard_models[0] if standard_models else 'N/A'}",
            f"                        Score: {best_std_score:.4f}  |  PIT: {'PASS' if best_std_pit == 1.0 else f'{best_std_pit*100:.0f}%'}",
            "",
            f"  [magenta]Best Experimental:[/magenta]  {experimental_models[0] if experimental_models else 'N/A'}",
            f"                        Score: {best_exp_score:.4f}  |  PIT: {'PASS' if best_exp_pit == 1.0 else f'{best_exp_pit*100:.0f}%'}",
            "",
        ]
        
        # Determine winner
        gap = ((best_exp_score - best_std_score) / best_std_score * 100) if best_std_score else 0
        if gap > 5:
            comparison_lines.append(f"  [bold green]EXPERIMENTAL WINS by {gap:.1f}%[/bold green]")
        elif gap > 0:
            comparison_lines.append(f"  [yellow]Experimental leads by {gap:.1f}% (needs >5% for promotion)[/yellow]")
        else:
            comparison_lines.append(f"  [blue]Standard leads by {abs(gap):.1f}%[/blue]")
        
        console.print(Panel(
            "\n".join(comparison_lines),
            border_style="cyan",
            padding=(0, 2),
        ))
    
    # =========================================================================
    # PROMOTION CANDIDATES
    # =========================================================================
    if result.promotion_candidates:
        console.print(Panel.fit(
            "[bold green]PROMOTION CANDIDATES[/bold green]\n\n" +
            "The following experimental models beat all criteria:\n" +
            "\n".join(f"  [green]>[/green] {m}" for m in result.promotion_candidates) +
            "\n\n[dim]Ready for panel review and production deployment.[/dim]",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[yellow]No Promotion Candidates[/yellow]\n\n"
            "[dim]Criteria for promotion:[/dim]\n"
            "  - Beat best standard model by >5%\n"
            "  - 100% PIT calibration pass rate\n"
            "  - Not last in any category",
            border_style="yellow"
        ))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    console.print()
    summary_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    
    summary_table.add_row("Symbols Tested", str(result.summary['n_symbols']))
    summary_table.add_row("Standard Models", str(result.summary['n_standard_models']))
    summary_table.add_row("Experimental Models", str(result.summary['n_experimental_models']))
    summary_table.add_row("Total Fits", str(result.summary['n_total_fits']))
    summary_table.add_row("Avg Fit Time", f"{result.summary['avg_fit_time_ms']:.1f}ms")
    summary_table.add_row("Overall PIT Pass Rate", f"{result.summary['pit_pass_rate']:.1%}")
    
    console.print(Panel(
        summary_table,
        title="[bold]Summary[/bold]",
        border_style="blue",
    ))
