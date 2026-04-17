from __future__ import annotations
"""
Kalman diagnostics: innovation whiteness testing, log-likelihood computation,
and regime drift prior estimation.

Extracted from signals.py (Story 6.2). Contains _test_innovation_whiteness
for Ljung-Box autocorrelation tests, two variants of Kalman log-likelihood,
and _estimate_regime_drift_priors for Bayesian regime prior estimation.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# -- path setup so "from ingestion..." works when run standalone ----------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _test_innovation_whiteness(innovations: np.ndarray, innovation_vars: np.ndarray, lags: int = 20) -> Dict[str, float]:
    """
    Test innovation whiteness using Ljung-Box test for autocorrelation.
    
    Refinement 3: Model adequacy via innovation whiteness testing.
    If innovations are not white noise (autocorrelated), the model may be misspecified.
    
    Args:
        innovations: Prediction errors from Kalman filter
        innovation_vars: Innovation variances (for standardization)
        lags: Number of lags to test
        
    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    try:
        # Standardize innovations by their predicted variance
        std_innovations = innovations / np.sqrt(np.maximum(innovation_vars, 1e-12))
        std_innovations = std_innovations[np.isfinite(std_innovations)]
        
        if len(std_innovations) < max(30, lags + 10):
            return {
                "ljung_box_statistic": float("nan"),
                "ljung_box_pvalue": float("nan"),
                "lags_tested": 0,
                "model_adequate": None,
                "note": "insufficient_data"
            }
        
        n = len(std_innovations)
        lags = min(lags, n // 5)  # conservative lag limit
        
        # Compute Ljung-Box statistic manually
        # Q = n(n+2) Σ(ρ_k² / (n-k)) for k=1..m
        # Under H0 (white noise), Q ~ χ²(m)
        
        # Compute autocorrelations
        acf_vals = []
        for lag in range(1, lags + 1):
            if lag >= n:
                break
            try:
                # Sample autocorrelation at lag k
                mean_innov = float(np.mean(std_innovations))
                numerator = float(np.sum((std_innovations[lag:] - mean_innov) * (std_innovations[:-lag] - mean_innov)))
                denominator = float(np.sum((std_innovations - mean_innov) ** 2))
                rho_k = numerator / denominator if abs(denominator) > 1e-12 else 0.0
                acf_vals.append(rho_k)
            except Exception:
                break
        
        if not acf_vals:
            return {
                "ljung_box_statistic": float("nan"),
                "ljung_box_pvalue": float("nan"),
                "lags_tested": 0,
                "model_adequate": None,
                "note": "acf_computation_failed"
            }
        
        # Ljung-Box statistic
        Q = 0.0
        m = len(acf_vals)
        for k, rho_k in enumerate(acf_vals, start=1):
            Q += (rho_k ** 2) / float(n - k)
        Q *= n * (n + 2)
        
        # Compute p-value using chi-squared distribution
        from scipy.stats import chi2
        pvalue = float(1.0 - chi2.cdf(Q, df=m))
        
        # Interpretation: reject H0 (white noise) if p < 0.05
        # model_adequate = True if we fail to reject (p >= 0.05)
        model_adequate = bool(pvalue >= 0.05)
        
        return {
            "ljung_box_statistic": float(Q),
            "ljung_box_pvalue": float(pvalue),
            "lags_tested": int(m),
            "model_adequate": model_adequate,
            "note": "pass" if model_adequate else "fail_autocorrelation_detected"
        }
        
    except Exception as e:
        return {
            "ljung_box_statistic": float("nan"),
            "ljung_box_pvalue": float("nan"),
            "lags_tested": 0,
            "model_adequate": None,
            "note": f"test_failed: {str(e)}"
        }


def _compute_kalman_log_likelihood(y: np.ndarray, sigma: np.ndarray, q: float, c: float = 1.0) -> float:
    """
    Compute log-likelihood for Kalman filter with given process noise q.
    Used for q optimization via marginal likelihood maximization.
    
    Args:
        y: Observations (returns)
        sigma: Observation noise std (volatility) per time step
        q: Process noise variance to evaluate
        
    Returns:
        Total log-likelihood of observations under this q
    """
    T = len(y)
    if T < 2:
        return float('-inf')
    
    # Initialize
    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0
    
    for t in range(T):
        # Prediction
        mu_pred = mu_t
        P_pred = P_t + q
        
        # Observation variance
        R_t = float(max(c * (sigma[t] ** 2), 1e-12))
        
        # Innovation
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))
        
        # Log-likelihood contribution
        try:
            ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
        except Exception:
            pass
        
        # Update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))
    
    return float(log_likelihood)


def _compute_kalman_log_likelihood_heteroskedastic(y: np.ndarray, sigma: np.ndarray, c: float) -> float:
    """
    Compute log-likelihood for Kalman filter with heteroskedastic process noise q_t = c * σ_t².
    
    This allows drift uncertainty to scale with market stress: higher volatility => more drift uncertainty.
    
    Args:
        y: Observations (returns)
        sigma: Observation noise std (volatility) per time step
        c: Scaling factor for heteroskedastic process noise (q_t = c * σ_t²)
        
    Returns:
        Total log-likelihood of observations under this c
    """
    T = len(y)
    if T < 2:
        return float('-inf')
    
    # Initialize
    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0
    
    for t in range(T):
        # Heteroskedastic process noise: q_t = c * σ_t²
        R_t = float(max(c * (sigma[t] ** 2), 1e-12))
        q_t = float(max(c * R_t, 1e-12))
        
        # Prediction
        mu_pred = mu_t
        P_pred = P_t + q_t
        
        # Innovation
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))
        
        # Log-likelihood contribution
        try:
            ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
        except Exception:
            pass
        
        # Update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))
    
    return float(log_likelihood)


def _estimate_regime_drift_priors(ret: pd.Series, vol: pd.Series) -> Optional[Dict[str, float]]:
    """
    Estimate regime-specific drift expectations E[μ_t | Regime=k] from historical data.
    
    Uses a quick HMM fit on returns to identify regimes, then computes mean return
    per regime as a simple proxy for regime-conditional drift.
    
    Args:
        ret: Returns series
        vol: Volatility series
        
    Returns:
        Dictionary with regime-specific drift priors, or None if estimation fails
    """
    if not HMM_AVAILABLE:
        return None
    
    try:
        # Align data
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
        if len(df) < 300:
            return None
        
        df.columns = ["ret", "vol"]
        X = df.values
        
        # Fit 3-state HMM (suppress noisy convergence messages)
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42, verbose=False)
        with suppress_stdout():
            model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        # Compute mean return per state
        regime_drifts = {}
        for state_idx in range(3):
            mask = (states == state_idx)
            if np.sum(mask) > 10:
                regime_drifts[state_idx] = float(np.mean(df.loc[mask, "ret"]))
            else:
                regime_drifts[state_idx] = 0.0
        
        # Identify regime names by volatility
        means = model.means_
        vol_means = means[:, 1]
        sorted_indices = np.argsort(vol_means)
        
        regime_map = {
            sorted_indices[0]: "calm",
            sorted_indices[1]: "trending",
            sorted_indices[2]: "crisis"
        }
        
        # Get current regime (last observation)
        current_state = states[-1]
        current_regime = regime_map.get(current_state, "calm")
        current_drift_prior = regime_drifts.get(current_state, 0.0)
        
        return {
            "current_regime": current_regime,
            "current_drift_prior": float(current_drift_prior),
            "regime_drifts": regime_drifts,
            "regime_map": regime_map,
        }
        
    except Exception:
        return None

