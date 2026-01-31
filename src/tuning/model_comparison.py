#!/usr/bin/env python3
"""
model_comparison.py

Structural model comparison framework for formal falsifiability (Level-7).

Implements systematic model selection using information criteria (AIC/BIC):
- Volatility models: GARCH(1,1) vs EWMA
- Tail distributions: Student-t vs Gaussian
- Drift models: Kalman vs EWMA (single-speed vs multi-speed)
- Factor usefulness: trend, momentum, regime adjustments

This eliminates guessing and hand-tuning by letting data speak through:
- Akaike Information Criterion (AIC): 2k - 2*ln(L)
- Bayesian Information Criterion (BIC): k*ln(n) - 2*ln(L)

Where:
    k = number of parameters
    n = number of observations
    L = likelihood

Lower AIC/BIC indicates better model fit with parsimony penalty.
BIC penalizes complexity more heavily (use for large n).
AIC is less conservative (use for model selection with finite samples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


@dataclass
class ModelSpec:
    """
    Specification for a model to be compared.
    
    Attributes:
        name: Human-readable model name
        n_params: Number of free parameters
        log_likelihood: Log-likelihood on data
        n_obs: Number of observations used
        converged: Whether optimization converged
        metadata: Additional model-specific information
    """
    name: str
    n_params: int
    log_likelihood: float
    n_obs: int
    converged: bool = True
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def aic(self) -> float:
        """Akaike Information Criterion: 2k - 2*ln(L)"""
        return 2.0 * self.n_params - 2.0 * self.log_likelihood
    
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion: k*ln(n) - 2*ln(L)"""
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood
    
    @property
    def aicc(self) -> float:
        """
        Corrected AIC for small samples: AIC + 2k(k+1)/(n-k-1)
        Use when n/k < 40 (Burnham & Anderson 2002)
        """
        if self.n_obs - self.n_params - 1 <= 0:
            return float('inf')
        correction = (2.0 * self.n_params * (self.n_params + 1)) / (self.n_obs - self.n_params - 1)
        return self.aic + correction


@dataclass
class ComparisonResult:
    """
    Result of comparing multiple models.
    
    Attributes:
        models: List of ModelSpec instances being compared
        winner_aic: Best model by AIC
        winner_bic: Best model by BIC
        delta_aic: AIC differences relative to best (dict: model_name -> Δ_AIC)
        delta_bic: BIC differences relative to best (dict: model_name -> Δ_BIC)
        akaike_weights: Model probabilities from AIC (dict: model_name -> weight)
        recommendation: Recommended model choice with reasoning
    """
    models: List[ModelSpec]
    winner_aic: str
    winner_bic: str
    delta_aic: Dict[str, float]
    delta_bic: Dict[str, float]
    akaike_weights: Dict[str, float]
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "models": [
                {
                    "name": m.name,
                    "n_params": m.n_params,
                    "log_likelihood": m.log_likelihood,
                    "n_obs": m.n_obs,
                    "aic": m.aic,
                    "bic": m.bic,
                    "aicc": m.aicc,
                    "converged": m.converged,
                }
                for m in self.models
            ],
            "winner_aic": self.winner_aic,
            "winner_bic": self.winner_bic,
            "delta_aic": self.delta_aic,
            "delta_bic": self.delta_bic,
            "akaike_weights": self.akaike_weights,
            "recommendation": self.recommendation,
        }


def compare_models(models: List[ModelSpec]) -> ComparisonResult:
    """
    Compare multiple models using AIC and BIC.
    
    Computes:
        - Δ_AIC and Δ_BIC relative to best model (lower is better)
        - Akaike weights: P(model_i | data) ∝ exp(-Δ_AIC_i / 2)
        - Recommendation based on weight of evidence
    
    Interpretation (Burnham & Anderson 2002):
        - Δ_AIC < 2: substantial support (models are competitive)
        - Δ_AIC 4-7: considerably less support
        - Δ_AIC > 10: essentially no support
    
    Args:
        models: List of ModelSpec instances to compare
        
    Returns:
        ComparisonResult with winner and weight of evidence
    """
    if not models:
        raise ValueError("Need at least one model to compare")
    
    # Filter out non-converged models
    converged = [m for m in models if m.converged and np.isfinite(m.log_likelihood)]
    if not converged:
        # All failed: return neutral result
        return ComparisonResult(
            models=models,
            winner_aic="none_converged",
            winner_bic="none_converged",
            delta_aic={m.name: float('nan') for m in models},
            delta_bic={m.name: float('nan') for m in models},
            akaike_weights={m.name: float('nan') for m in models},
            recommendation="All models failed to converge. Use fallback method.",
        )
    
    # Find best models by AIC and BIC (lower is better)
    best_aic_model = min(converged, key=lambda m: m.aic)
    best_bic_model = min(converged, key=lambda m: m.bic)
    
    # Compute deltas relative to best
    delta_aic = {m.name: m.aic - best_aic_model.aic for m in converged}
    delta_bic = {m.name: m.bic - best_bic_model.bic for m in converged}
    
    # Add NaN for non-converged models
    for m in models:
        if m not in converged:
            delta_aic[m.name] = float('nan')
            delta_bic[m.name] = float('nan')
    
    # Compute Akaike weights: w_i = exp(-Δ_i/2) / Σ_j exp(-Δ_j/2)
    # These represent the probability each model is the best approximating model
    exp_terms = {name: np.exp(-delta / 2.0) for name, delta in delta_aic.items() if np.isfinite(delta)}
    total = sum(exp_terms.values())
    akaike_weights = {name: exp_term / total for name, exp_term in exp_terms.items()}
    
    # Add zero weight for non-converged
    for m in models:
        if m.name not in akaike_weights:
            akaike_weights[m.name] = 0.0
    
    # Recommendation based on weight of evidence
    winner_aic_name = best_aic_model.name
    winner_bic_name = best_bic_model.name
    winner_aic_weight = akaike_weights.get(winner_aic_name, 0.0)
    
    # Evidence strength interpretation
    if winner_aic_name == winner_bic_name:
        # Both criteria agree
        if winner_aic_weight > 0.90:
            strength = "very strong"
        elif winner_aic_weight > 0.75:
            strength = "strong"
        elif winner_aic_weight > 0.50:
            strength = "moderate"
        else:
            strength = "weak"
        
        recommendation = f"Use {winner_aic_name} ({strength} evidence). Both AIC and BIC agree. Akaike weight: {winner_aic_weight:.2%}."
    else:
        # Criteria disagree: AIC vs BIC tradeoff
        # BIC penalizes complexity more heavily, so it tends to choose simpler models
        # AIC is better for prediction, BIC for model selection with parsimony
        aic_delta = delta_aic.get(winner_bic_name, float('inf'))
        bic_delta = delta_bic.get(winner_aic_name, float('inf'))
        
        if aic_delta < 2.0:
            # BIC winner is competitive in AIC (Δ < 2)
            recommendation = f"Use {winner_bic_name} (BIC winner, substantial AIC support Δ={aic_delta:.1f}). More parsimonious model is competitive."
        elif bic_delta < np.log(best_bic_model.n_obs):
            # AIC winner has reasonable BIC penalty
            recommendation = f"Use {winner_aic_name} (AIC winner, weight={winner_aic_weight:.2%}). Better predictive fit justifies complexity."
        else:
            # Significant disagreement: default to BIC for parsimony
            recommendation = f"Criteria disagree. Use {winner_bic_name} (BIC winner) for parsimony, or {winner_aic_name} (AIC winner) for prediction. Sample size n={best_bic_model.n_obs}."
    
    return ComparisonResult(
        models=converged,
        winner_aic=winner_aic_name,
        winner_bic=winner_bic_name,
        delta_aic=delta_aic,
        delta_bic=delta_bic,
        akaike_weights=akaike_weights,
        recommendation=recommendation,
    )


def compute_ewma_volatility_with_likelihood(
    returns: pd.Series,
    span: int = 21,
) -> Tuple[pd.Series, Dict]:
    """
    Compute EWMA volatility with log-likelihood for model comparison.
    
    Model:
        σ_t² = λ·σ_{t-1}² + (1-λ)·r_{t-1}²
    
    Where λ = (span-1)/(span+1) from pandas EWMA convention.
    
    This is a restricted GARCH(1,1) with:
        - ω = 0 (no constant term)
        - α = 1-λ
        - β = λ
        - α + β = 1 (integrated process)
    
    Number of free parameters: k = 1 (span, or equivalently λ)
    
    Log-likelihood computed as Gaussian likelihood:
        ln L = Σ_t [ -0.5*ln(2π) - 0.5*ln(σ_t²) - 0.5*(r_t²/σ_t²) ]
    
    Args:
        returns: Return series (de-meaned recommended)
        span: EWMA span (default 21 days)
        
    Returns:
        Tuple of (volatility_series, metadata_dict)
    """
    ret = returns.dropna()
    if len(ret) < 50:
        raise ValueError("Insufficient data for EWMA volatility")
    
    # Compute EWMA variance
    var_ewma = ret.ewm(span=span, adjust=False).var()
    vol_ewma = np.sqrt(var_ewma)
    
    # Compute log-likelihood
    # Gaussian likelihood: ln p(r_t | σ_t) = -0.5*ln(2πσ_t²) - 0.5*(r_t²/σ_t²)
    r2 = ret.values ** 2
    sig2 = var_ewma.values
    
    # Remove initial NaNs and ensure positive variance
    valid = np.isfinite(sig2) & (sig2 > 1e-12)
    r2_valid = r2[valid]
    sig2_valid = sig2[valid]
    
    log_lik_terms = -0.5 * (np.log(2.0 * np.pi * sig2_valid) + r2_valid / sig2_valid)
    log_likelihood = float(np.sum(log_lik_terms))
    
    n_obs = int(np.sum(valid))
    
    metadata = {
        "span": span,
        "lambda": (span - 1) / (span + 1),
        "n_params": 1,  # Only span (or λ) is free
        "log_likelihood": log_likelihood,
        "n_obs": n_obs,
        "converged": True,
    }
    
    return vol_ewma, metadata


def compare_volatility_models(
    returns: pd.Series,
    garch_params: Optional[Dict] = None,
) -> ComparisonResult:
    """
    Compare GARCH(1,1) vs EWMA for volatility modeling.
    
    Models compared:
        1. GARCH(1,1): σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²  (k=3 params)
        2. EWMA (span=21): σ_t² = λ·σ_{t-1}² + (1-λ)·r_{t-1}²  (k=1 param)
        3. EWMA (span=63): Long-term EWMA (k=1 param)
    
    Args:
        returns: Return series (should be de-meaned)
        garch_params: Optional pre-computed GARCH params from _garch11_mle
                     If None, GARCH model skipped
        
    Returns:
        ComparisonResult with recommended volatility model
    """
    models = []
    
    # GARCH(1,1) if available
    if garch_params is not None and isinstance(garch_params, dict):
        if garch_params.get("converged", False):
            models.append(ModelSpec(
                name="GARCH(1,1)",
                n_params=3,
                log_likelihood=garch_params.get("log_likelihood", float('nan')),
                n_obs=garch_params.get("n_obs", len(returns)),
                converged=True,
                metadata=garch_params,
            ))
    
    # EWMA variants
    try:
        _, ewma21_meta = compute_ewma_volatility_with_likelihood(returns, span=21)
        models.append(ModelSpec(
            name="EWMA(21)",
            n_params=1,
            log_likelihood=ewma21_meta["log_likelihood"],
            n_obs=ewma21_meta["n_obs"],
            converged=True,
            metadata=ewma21_meta,
        ))
    except Exception:
        pass
    
    try:
        _, ewma63_meta = compute_ewma_volatility_with_likelihood(returns, span=63)
        models.append(ModelSpec(
            name="EWMA(63)",
            n_params=1,
            log_likelihood=ewma63_meta["log_likelihood"],
            n_obs=ewma63_meta["n_obs"],
            converged=True,
            metadata=ewma63_meta,
        ))
    except Exception:
        pass
    
    if not models:
        raise RuntimeError("No volatility models succeeded")
    
    return compare_models(models)


def compute_gaussian_tail_loglikelihood(standardized_residuals: pd.Series) -> Dict:
    """
    Compute log-likelihood of Gaussian tail model on standardized residuals.
    
    Model: z_t ~ N(0, 1)
    
    Number of parameters: k = 0 (standard normal, no free parameters)
    
    Args:
        standardized_residuals: Returns divided by conditional volatility
        
    Returns:
        Dictionary with log-likelihood and metadata
    """
    z = standardized_residuals.replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 50:
        raise ValueError("Insufficient data for tail model")
    
    z_vals = z.values.astype(float)
    
    # Gaussian log-likelihood: ln p(z) = -0.5*ln(2π) - 0.5*z²
    log_lik = float(np.sum(norm.logpdf(z_vals)))
    
    return {
        "distribution": "Gaussian",
        "n_params": 0,  # Standard normal, no free parameters
        "log_likelihood": log_lik,
        "n_obs": len(z_vals),
        "converged": True,
    }


def compare_tail_models(
    standardized_residuals: pd.Series,
    student_t_params: Optional[Dict] = None,
) -> ComparisonResult:
    """
    Compare Student-t vs Gaussian for tail modeling.
    
    Models compared:
        1. Student-t: z_t ~ t(ν)  (k=1 param: degrees of freedom ν)
        2. Gaussian: z_t ~ N(0,1)  (k=0 params)
    
    Args:
        standardized_residuals: Returns divided by conditional volatility
        student_t_params: Optional pre-computed Student-t params from _fit_student_nu_mle
        
    Returns:
        ComparisonResult with recommended tail model
    """
    models = []
    
    # Student-t if available
    if student_t_params is not None and isinstance(student_t_params, dict):
        if student_t_params.get("converged", False):
            models.append(ModelSpec(
                name="Student-t",
                n_params=1,  # nu (degrees of freedom)
                log_likelihood=student_t_params.get("ll", float('nan')),
                n_obs=student_t_params.get("n", 0),
                converged=True,
                metadata=student_t_params,
            ))
    
    # Gaussian
    try:
        gauss_meta = compute_gaussian_tail_loglikelihood(standardized_residuals)
        models.append(ModelSpec(
            name="Gaussian",
            n_params=0,
            log_likelihood=gauss_meta["log_likelihood"],
            n_obs=gauss_meta["n_obs"],
            converged=True,
            metadata=gauss_meta,
        ))
    except Exception:
        pass
    
    if not models:
        raise RuntimeError("No tail models succeeded")
    
    return compare_models(models)


def compare_drift_models(
    returns: pd.Series,
    volatility: pd.Series,
    kalman_metadata: Optional[Dict] = None,
) -> ComparisonResult:
    """
    Compare Kalman filter vs EWMA drift models.
    
    Models compared:
        1. Kalman filter: μ_t = μ_{t-1} + η_t with optimal q  (k ≈ 1-2 params)
        2. EWMA fast (span=21): Simple exponential smoothing  (k=1 param)
        3. EWMA slow (span=126): Long-term exponential smoothing  (k=1 param)
        4. Multi-speed blend: 0.5*fast + 0.5*slow  (k=2 params: two spans)
    
    Args:
        returns: Return series
        volatility: Conditional volatility series (for Kalman)
        kalman_metadata: Optional pre-computed Kalman filter metadata
        
    Returns:
        ComparisonResult with recommended drift model
    """
    models = []
    
    # Kalman filter if available
    if kalman_metadata is not None and isinstance(kalman_metadata, dict):
        log_lik = kalman_metadata.get("log_likelihood", float('nan'))
        n_obs = kalman_metadata.get("n_obs", 0)
        if np.isfinite(log_lik) and n_obs > 0:
            # Kalman has 1-2 free params depending on whether q is optimized
            n_params = 2 if kalman_metadata.get("q_optimization_attempted", False) else 1
            models.append(ModelSpec(
                name="Kalman",
                n_params=n_params,
                log_likelihood=log_lik,
                n_obs=n_obs,
                converged=True,
                metadata=kalman_metadata,
            ))
    
    # EWMA drift models
    # These are simpler: just exponential smoothing of returns
    # Log-likelihood: treat as Gaussian with mean = EWMA, variance = σ²
    try:
        ret = returns.dropna()
        vol = volatility.reindex(ret.index).dropna()
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
        if len(df) >= 50:
            df.columns = ['ret', 'vol']
            
            # EWMA fast (span=21)
            mu_fast = df['ret'].ewm(span=21, adjust=False).mean()
            # Log-likelihood: Gaussian with drift = mu_fast, scale = vol
            resid_fast = df['ret'].values - mu_fast.values
            sig = df['vol'].values
            valid = np.isfinite(resid_fast) & np.isfinite(sig) & (sig > 1e-12)
            ll_fast = float(np.sum(-0.5 * (np.log(2.0 * np.pi * sig[valid]**2) + (resid_fast[valid]**2 / sig[valid]**2))))
            
            models.append(ModelSpec(
                name="EWMA_fast(21)",
                n_params=1,
                log_likelihood=ll_fast,
                n_obs=int(np.sum(valid)),
                converged=True,
                metadata={"span": 21},
            ))
            
            # EWMA slow (span=126)
            mu_slow = df['ret'].ewm(span=126, adjust=False).mean()
            resid_slow = df['ret'].values - mu_slow.values
            valid_slow = np.isfinite(resid_slow) & (sig > 1e-12)
            ll_slow = float(np.sum(-0.5 * (np.log(2.0 * np.pi * sig[valid_slow]**2) + (resid_slow[valid_slow]**2 / sig[valid_slow]**2))))
            
            models.append(ModelSpec(
                name="EWMA_slow(126)",
                n_params=1,
                log_likelihood=ll_slow,
                n_obs=int(np.sum(valid_slow)),
                converged=True,
                metadata={"span": 126},
            ))
            
            # Multi-speed blend
            mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
            resid_blend = df['ret'].values - mu_blend.values
            valid_blend = np.isfinite(resid_blend) & (sig > 1e-12)
            ll_blend = float(np.sum(-0.5 * (np.log(2.0 * np.pi * sig[valid_blend]**2) + (resid_blend[valid_blend]**2 / sig[valid_blend]**2))))
            
            models.append(ModelSpec(
                name="EWMA_blend(21+126)",
                n_params=2,  # Two spans
                log_likelihood=ll_blend,
                n_obs=int(np.sum(valid_blend)),
                converged=True,
                metadata={"spans": [21, 126]},
            ))
    except Exception:
        pass
    
    if not models:
        raise RuntimeError("No drift models succeeded")
    
    return compare_models(models)


def run_all_comparisons(
    returns: pd.Series,
    volatility: pd.Series,
    garch_params: Optional[Dict] = None,
    student_t_params: Optional[Dict] = None,
    kalman_metadata: Optional[Dict] = None,
) -> Dict[str, ComparisonResult]:
    """
    Run all model comparisons systematically.
    
    Args:
        returns: Return series (de-meaned recommended)
        volatility: Conditional volatility series
        garch_params: GARCH(1,1) parameters from _garch11_mle
        student_t_params: Student-t parameters from _fit_student_nu_mle
        kalman_metadata: Kalman filter metadata from _kalman_filter_drift
        
    Returns:
        Dictionary with comparison results for each category:
            - 'volatility': GARCH vs EWMA comparison
            - 'tails': Student-t vs Gaussian comparison
            - 'drift': Kalman vs EWMA comparison
    """
    results = {}
    
    # 1. Volatility model comparison
    try:
        results['volatility'] = compare_volatility_models(returns, garch_params)
    except Exception as e:
        results['volatility'] = None
        results['volatility_error'] = str(e)
    
    # 2. Tail distribution comparison
    try:
        # Compute standardized residuals
        ret_clean = returns.dropna()
        vol_clean = volatility.reindex(ret_clean.index).dropna()
        df = pd.concat([ret_clean, vol_clean], axis=1, join='inner').dropna()
        df.columns = ['ret', 'vol']
        standardized = df['ret'] / df['vol']
        
        results['tails'] = compare_tail_models(standardized, student_t_params)
    except Exception as e:
        results['tails'] = None
        results['tails_error'] = str(e)
    
    # 3. Drift model comparison
    try:
        results['drift'] = compare_drift_models(returns, volatility, kalman_metadata)
    except Exception as e:
        results['drift'] = None
        results['drift_error'] = str(e)
    
    return results
