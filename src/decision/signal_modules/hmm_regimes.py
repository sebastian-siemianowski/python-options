from __future__ import annotations
"""
HMM regime detection and parameter stability tracking.

Extracted from signals.py (Story 6.6). Contains fit_hmm_regimes for Gaussian HMM
3-state fitting and track_parameter_stability for rolling GARCH drift tracking.
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# -- path setup ---------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from ingestion.data_utils import _ensure_float_series



# -------------------------
# HMM Regime Detection (Formal Bayesian Inference)
# -------------------------

def fit_hmm_regimes(feats: Dict[str, pd.Series], n_states: int = 3, random_seed: int = 42) -> Optional[Dict]:
    """
    Fit a Hidden Markov Model with Gaussian emissions to detect market regimes.

    Each regime (state) has:
    - Its own μ (drift) dynamics captured by emission mean
    - Its own σ (volatility) dynamics captured by emission covariance
    - Persistence captured by transition matrix

    Args:
        feats: Feature dictionary from compute_features()
        n_states: Number of hidden states (default 3: calm, trending, crisis)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with HMM model, state sequence, and regime metadata, or None on failure
    """
    if not HMM_AVAILABLE:
        return None

    try:
        # Extract returns and volatility as observations
        ret = feats.get("ret", pd.Series(dtype=float))
        vol = feats.get("vol", pd.Series(dtype=float))

        if ret.empty or vol.empty:
            return None

        # Align and clean data
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
        if len(df) < 300:  # Need sufficient history for stable HMM
            return None

        df.columns = ["ret", "vol"]
        X = df.values  # Shape (T, 2): returns and volatility as features

        # Fit Gaussian HMM with full covariance (allows each state its own μ and σ)
        # Suppress stdout to hide noisy convergence messages from hmmlearn
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_seed,
            verbose=False
        )

        with suppress_stdout():
            model.fit(X)

        # Infer hidden state sequence (Viterbi for most likely path)
        states = model.predict(X)

        # Posterior probabilities for each state at each time
        posteriors = model.predict_proba(X)

        # Identify regime characteristics from emission parameters
        means = model.means_  # Shape (n_states, 2): [drift, vol] per state
        covars = model.covars_  # Shape (n_states, 2, 2)
        transmat = model.transmat_  # Shape (n_states, n_states)

        # Label states by volatility level: calm < normal < crisis
        vol_means = means[:, 1]  # volatility component
        sorted_indices = np.argsort(vol_means)

        regime_names = {
            sorted_indices[0]: "calm",
            sorted_indices[1]: "trending" if n_states == 3 else "normal",
            sorted_indices[2]: "crisis" if n_states == 3 else "volatile"
        }

        # Build regime series aligned with returns index
        regime_series = pd.Series(
            [regime_names.get(s, f"state_{s}") for s in states],
            index=df.index,
            name="regime"
        )

        # Posterior probability series (one per state)
        posterior_df = pd.DataFrame(
            posteriors,
            index=df.index,
            columns=[regime_names.get(i, f"state_{i}") for i in range(n_states)]
        )

        # Compute log-likelihood and information criteria for model diagnostics
        try:
            log_likelihood = float(model.score(X))
            n_obs = int(len(X))
            # Count free parameters: n_states-1 for initial probs, n_states*(n_states-1) for transitions,
            # n_states*n_features for means, n_states*n_features*(n_features+1)/2 for full covariance
            n_features = X.shape[1]
            n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * n_features * (n_features + 1) // 2
            aic = float(2.0 * n_params - 2.0 * log_likelihood)
            bic = float(n_params * np.log(n_obs) - 2.0 * log_likelihood)
        except Exception:
            log_likelihood = float("nan")
            n_obs = int(len(X))
            n_params = 0
            aic = float("nan")
            bic = float("nan")

        return {
            "model": model,
            "regime_series": regime_series,
            "posterior_probs": posterior_df,
            "states": states,
            "means": means,
            "covars": covars,
            "transmat": transmat,
            "regime_names": regime_names,
            "n_states": n_states,
            "log_likelihood": log_likelihood,
            "n_obs": n_obs,
            "n_params": n_params,
            "aic": aic,
            "bic": bic,
        }

    except Exception as e:
        # Silent fallback on HMM failure
        return None


def track_parameter_stability(ret: pd.Series, window_days: int = 252, step_days: int = 63) -> Dict[str, pd.DataFrame]:
    """
    Track GARCH parameter stability over time using rolling window estimation.

    Fits GARCH(1,1) on expanding windows to detect parameter drift.
    Returns time series of parameters, standard errors, and log-likelihoods.

    Args:
        ret: Returns series
        window_days: Minimum window size for initial fit
        step_days: Days between refits (trades off compute vs resolution)

    Returns:
        Dictionary with DataFrames tracking parameters over time
    """
    ret_clean = _ensure_float_series(ret).dropna()
    if len(ret_clean) < max(300, window_days):
        return {}

    # Time points to evaluate (start at window_days, step forward)
    dates = ret_clean.index
    eval_dates = []
    for i in range(window_days, len(dates), step_days):
        eval_dates.append(dates[i])

    if not eval_dates:
        return {}

    # Storage for parameter evolution
    records = []

    for eval_date in eval_dates:
        # Use expanding window up to eval_date
        window_ret = ret_clean.loc[:eval_date]

        # Try to fit GARCH
        try:
            _, params = _garch11_mle(window_ret)
            record = {
                "date": eval_date,
                "omega": params.get("omega", float("nan")),
                "alpha": params.get("alpha", float("nan")),
                "beta": params.get("beta", float("nan")),
                "se_omega": params.get("se_omega", float("nan")),
                "se_alpha": params.get("se_alpha", float("nan")),
                "se_beta": params.get("se_beta", float("nan")),
                "log_likelihood": params.get("log_likelihood", float("nan")),
                "aic": params.get("aic", float("nan")),
                "bic": params.get("bic", float("nan")),
                "n_obs": params.get("n_obs", 0),
                "converged": params.get("converged", False),
            }
            records.append(record)
        except Exception:
            # Skip windows where GARCH fails
            continue

    if not records:
        return {}

    df = pd.DataFrame(records).set_index("date")

    # Compute parameter drift statistics (rolling z-score of parameter changes)
    param_cols = ["omega", "alpha", "beta"]
    drift_stats = {}

    for col in param_cols:
        if col in df.columns:
            changes = df[col].diff()
            se_col = f"se_{col}"
            if se_col in df.columns:
                # Normalized change (z-score): change / standard error
                z_change = changes / df[se_col].replace(0, np.nan)
                drift_stats[f"{col}_drift_z"] = z_change

    drift_df = pd.DataFrame(drift_stats, index=df.index)

    return {
        "param_evolution": df,
        "param_drift": drift_df,
    }


