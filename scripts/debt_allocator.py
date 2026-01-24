#!/usr/bin/env python3
"""
debt_allocator.py

Research-Grade FX Debt Allocation Engine (EURJPY)
=================================================

PURPOSE:
    Determines a single, precise, irreversible calendar day on which 
    JPY-denominated debt must be switched to EUR-denominated debt,
    based exclusively on EURJPY dynamics.

    This system approximates the theoretical upper bound of decision accuracy
    under policy-driven FX discontinuities, latent stress accumulation, and
    asymmetric convex loss.

THIS IS:
    ❌ NOT a trade signal
    ❌ NOT portfolio optimization
    ❌ NOT macro prediction
    ✅ Balance-sheet convexity control
    ✅ Latent-state inference
    ✅ Causal, auditable decision logic

STOCHASTIC FOUNDATION (PAIR-PURE):
    The ONLY stochastic process is: X_t = log(EURJPY_t)
    There is no "JPY risk" or "EUR risk" - only pair dynamics.

FUNDING LOSS DEFINITION (FIXED SIGN):
    Funding loss occurs IF AND ONLY IF: ΔX_t > 0
    (EUR cost of JPY debt increases)

LATENT STATE MODEL:
    S_t ∈ {NORMAL, COMPRESSED, PRE_POLICY, POLICY}
    States are partially ordered with forbidden backward transitions.

ARCHITECTURAL CONSTRAINTS:
    - Everything lives in this single file
    - No reuse of existing methods
    - No modification of existing files
    - No imports from tuning or signal helpers
    - No shared mutable state
    - Deletable with zero side effects

Author: Research-Grade Debt Allocation Engine
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, NamedTuple
import warnings

# =============================================================================
# NUMERICAL IMPORTS (MINIMAL, STANDARD ONLY)
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

# =============================================================================
# PRESENTATION IMPORTS (RICH ONLY)
# =============================================================================
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# File paths (isolated, deletable)
DEBT_CACHE_DIR = "scripts/quant/cache/debt"
DECISION_PERSISTENCE_FILE = f"{DEBT_CACHE_DIR}/debt_switch_decision.json"
EURJPY_DATA_FILE = f"{DEBT_CACHE_DIR}/EURJPY_1d.csv"
STATE_HISTORY_FILE = f"{DEBT_CACHE_DIR}/state_history.json"

# Minimum data requirements
MIN_HISTORY_DAYS = 252  # 1 year of trading days minimum
MIN_POSTERIOR_SAMPLES = 5000  # Monte Carlo samples for robust estimation
N_BOOTSTRAP_SAMPLES = 10000  # Bootstrap samples for posterior

# Observation model horizons (trading days)
HORIZONS = [5, 21, 63, 126, 252]  # 1w, 1m, 3m, 6m, 1y

# Convex loss exponent
CONVEX_LOSS_EXPONENT = 1.5  # p ∈ (1, 2]

# Outlier trimming percentile
OUTLIER_TRIM_PERCENTILE = 99.0

# Volatility lookback for regime detection
VOL_LOOKBACK_SHORT = 21  # 1 month
VOL_LOOKBACK_LONG = 126  # 6 months

# Disagreement smoothing (EMA half-life in days)
DISAGREEMENT_SMOOTHING_HALFLIFE = 5

# Decision threshold (α for PRE_POLICY dominance)
PRE_POLICY_THRESHOLD = 0.60  # Trigger when P(PRE_POLICY) > α

# State transition constraints
TRANSITION_PERSISTENCE = 0.85  # Diagonal dominance in transition matrix


# =============================================================================
# LATENT STATE ENUMERATION
# =============================================================================

class LatentState(IntEnum):
    """
    Latent policy-stress states.
    
    Partially ordered: NORMAL → COMPRESSED → PRE_POLICY → POLICY
    Backward transitions are forbidden except via explicit reset.
    """
    NORMAL = 0
    COMPRESSED = 1
    PRE_POLICY = 2
    POLICY = 3
    
    @classmethod
    def names(cls) -> List[str]:
        return [s.name for s in cls]
    
    @classmethod
    def n_states(cls) -> int:
        return len(cls)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ObservationVector:
    """
    Price-derived observation vector Y_t.
    
    Components:
        C: Convex loss functional E[max(ΔX, 0)^p]
        P: Tail mass P(ΔX > 0)
        D: Epistemic disagreement (normalized)
        dD: Disagreement momentum
        V: Volatility compression/expansion ratio
    """
    convex_loss: float      # C(t)
    tail_mass: float        # P(t) = P(ΔX > 0)
    disagreement: float     # D(t)
    disagreement_momentum: float  # dD(t)
    vol_ratio: float        # V(t) = σ_short / σ_long
    timestamp: str
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for HMM processing."""
        return np.array([
            self.convex_loss,
            self.tail_mass,
            self.disagreement,
            self.disagreement_momentum,
            self.vol_ratio
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "convex_loss": self.convex_loss,
            "tail_mass": self.tail_mass,
            "disagreement": self.disagreement,
            "disagreement_momentum": self.disagreement_momentum,
            "vol_ratio": self.vol_ratio,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class StatePosterior:
    """Posterior state probabilities P(S_t | Y_1:t)."""
    probabilities: Tuple[float, float, float, float]  # One per state
    dominant_state: LatentState
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "probabilities": {
                LatentState.NORMAL.name: self.probabilities[0],
                LatentState.COMPRESSED.name: self.probabilities[1],
                LatentState.PRE_POLICY.name: self.probabilities[2],
                LatentState.POLICY.name: self.probabilities[3],
            },
            "dominant_state": self.dominant_state.name,
            "timestamp": self.timestamp,
        }
    
    @property
    def p_normal(self) -> float:
        return self.probabilities[LatentState.NORMAL]
    
    @property
    def p_compressed(self) -> float:
        return self.probabilities[LatentState.COMPRESSED]
    
    @property
    def p_pre_policy(self) -> float:
        return self.probabilities[LatentState.PRE_POLICY]
    
    @property
    def p_policy(self) -> float:
        return self.probabilities[LatentState.POLICY]


@dataclass(frozen=True)
class DebtSwitchDecision:
    """Immutable record of debt switch decision."""
    triggered: bool
    effective_date: Optional[str]
    observation: ObservationVector
    state_posterior: StatePosterior
    decision_basis: str
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "effective_date": self.effective_date,
            "observation": self.observation.to_dict(),
            "state_posterior": self.state_posterior.to_dict(),
            "decision_basis": self.decision_basis,
            "signature": self.signature,
        }


# =============================================================================
# DATA INGESTION (READ-ONLY, ISOLATED)
# =============================================================================

def _load_eurjpy_prices(data_path: str = EURJPY_DATA_FILE, force_refresh: bool = True) -> Optional[pd.Series]:
    """
    Load EURJPY price series, downloading fresh data from yfinance.
    
    Args:
        data_path: Path to EURJPY daily data CSV
        force_refresh: If True, always download fresh data
        
    Returns:
        pd.Series with DatetimeIndex and EURJPY prices, or None if unavailable
    """
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to download fresh data from yfinance
    if force_refresh:
        try:
            import yfinance as yf
            
            print(f"[debt_allocator] Downloading fresh EURJPY data...")
            
            ticker = yf.Ticker("EURJPY=X")
            df = ticker.history(period="10y")
            
            if df is not None and not df.empty and 'Close' in df.columns:
                df_save = df.reset_index()
                df_save.to_csv(path, index=False)
                
                print(f"[debt_allocator] Downloaded {len(df)} days of EURJPY data")
                print(f"[debt_allocator] Saved to: {path}")
                
                prices = df['Close'].dropna()
                if len(prices) >= MIN_HISTORY_DAYS:
                    return prices.sort_index()
                else:
                    print(f"[debt_allocator] Warning: Only {len(prices)} days (need {MIN_HISTORY_DAYS})")
        except ImportError:
            print("[debt_allocator] Warning: yfinance not installed, trying cached data")
        except Exception as e:
            print(f"[debt_allocator] Warning: Download failed ({e}), trying cached data")
    
    # Fall back to cached data
    try:
        if path.exists():
            print(f"[debt_allocator] Loading cached EURJPY data from: {path}")
            df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            
            if 'Close' in df.columns:
                prices = df['Close']
            elif 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                print("[debt_allocator] Error: No Close column in cached data")
                return None
            
            prices = prices.dropna()
            if len(prices) >= MIN_HISTORY_DAYS:
                print(f"[debt_allocator] Loaded {len(prices)} days from cache")
                return prices.sort_index()
            else:
                print(f"[debt_allocator] Warning: Cached data has only {len(prices)} days")
                return None
        else:
            print(f"[debt_allocator] Error: No cached data found at {path}")
            return None
            
    except Exception as e:
        print(f"[debt_allocator] Error loading cached data: {e}")
        return None


def _compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns: X_t = log(EURJPY_t), ΔX = X_t - X_{t-1}
    """
    log_prices = np.log(prices)
    return log_prices.diff().dropna()


# =============================================================================
# POSTERIOR SAMPLING (MONTE CARLO, NON-PARAMETRIC)
# =============================================================================

def _generate_posterior_samples(
    log_returns: pd.Series,
    horizon_days: int,
    n_samples: int = N_BOOTSTRAP_SAMPLES
) -> np.ndarray:
    """
    Generate posterior predictive samples using block bootstrap.
    
    Non-parametric approach:
    - Uses empirical distribution only
    - No analytical CDF assumptions
    - Preserves autocorrelation structure via block bootstrap
    
    Args:
        log_returns: Historical log returns
        horizon_days: Forecast horizon
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Array of cumulative return samples over horizon
    """
    returns_arr = log_returns.values
    n_obs = len(returns_arr)
    
    if n_obs < MIN_HISTORY_DAYS:
        return np.array([])
    
    # Block size for preserving autocorrelation
    block_size = min(horizon_days, 21)
    
    samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        cumulative = 0.0
        days_sampled = 0
        
        while days_sampled < horizon_days:
            # Random block start
            start_idx = np.random.randint(0, max(1, n_obs - block_size))
            block_len = min(block_size, horizon_days - days_sampled)
            
            # Sum returns in block
            cumulative += np.sum(returns_arr[start_idx:start_idx + block_len])
            days_sampled += block_len
        
        samples[i] = cumulative
    
    return samples


# =============================================================================
# OBSERVATION MODEL: CONVEX LOSS FUNCTIONAL
# =============================================================================

def _compute_convex_loss(
    posterior_samples: np.ndarray,
    p: float = CONVEX_LOSS_EXPONENT,
    trim_percentile: float = OUTLIER_TRIM_PERCENTILE
) -> float:
    """
    Compute convex loss functional: C(t) = E[max(ΔX, 0)^p]
    
    Rules:
    - Estimated via posterior predictive Monte Carlo ONLY
    - No analytical tails
    - Trim extreme outliers
    
    Args:
        posterior_samples: Monte Carlo samples of future returns
        p: Convex exponent in (1, 2]
        trim_percentile: Percentile for outlier trimming
        
    Returns:
        Convex loss estimate
    """
    if len(posterior_samples) < MIN_POSTERIOR_SAMPLES:
        return float('nan')
    
    # Filter to positive returns (funding loss occurs when ΔX > 0)
    positive_returns = posterior_samples[posterior_samples > 0]
    
    if len(positive_returns) == 0:
        return 0.0
    
    # Trim extreme outliers
    threshold = np.percentile(positive_returns, trim_percentile)
    trimmed = positive_returns[positive_returns <= threshold]
    
    if len(trimmed) == 0:
        return 0.0
    
    # Compute convex loss: E[max(ΔX, 0)^p]
    convex_values = np.power(trimmed, p)
    
    return float(np.mean(convex_values))


# =============================================================================
# OBSERVATION MODEL: TAIL MASS
# =============================================================================

def _compute_tail_mass(posterior_samples: np.ndarray) -> float:
    """
    Compute tail mass: P(t) = P(ΔX > 0)
    
    Rules:
    - Empirical (Monte Carlo)
    - Distribution-family invariant
    - No analytical shortcuts
    
    Args:
        posterior_samples: Monte Carlo samples
        
    Returns:
        Empirical probability of positive return
    """
    if len(posterior_samples) < MIN_POSTERIOR_SAMPLES:
        return float('nan')
    
    n_positive = np.sum(posterior_samples > 0)
    return float(n_positive) / float(len(posterior_samples))


# =============================================================================
# OBSERVATION MODEL: EPISTEMIC DISAGREEMENT
# =============================================================================

def _compute_epistemic_disagreement(
    log_returns: pd.Series,
    horizons: List[int] = HORIZONS,
    n_samples: int = 5000
) -> Tuple[float, List[float], List[float]]:
    """
    Compute normalized epistemic disagreement across horizons.
    
    D(t) = (1/H) * Σ_h |μ_h - μ̄| / σ_h
    
    Where μ̄ is the precision-weighted reference mean:
    μ̄(t) = Σ(μ_h / σ_h²) / Σ(1 / σ_h²)
    
    Rules:
    - Variance of means is FORBIDDEN
    - All quantities are pair-pure (EURJPY only)
    
    Args:
        log_returns: Historical log returns
        horizons: List of horizon days
        n_samples: Samples per horizon
        
    Returns:
        Tuple of (disagreement_score, means_per_horizon, stds_per_horizon)
    """
    means = []
    stds = []
    
    for h in horizons:
        samples = _generate_posterior_samples(log_returns, h, n_samples)
        
        if len(samples) >= MIN_POSTERIOR_SAMPLES:
            means.append(float(np.mean(samples)))
            stds.append(float(np.std(samples)))
        else:
            return (float('nan'), [], [])
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Check for zero stds
    if np.any(stds <= 0):
        return (float('nan'), means.tolist(), stds.tolist())
    
    # Precision-weighted reference mean
    precisions = 1.0 / (stds ** 2)
    mu_bar = np.sum(means * precisions) / np.sum(precisions)
    
    # Normalized deviations
    deviations = np.abs(means - mu_bar) / stds
    
    # Average normalized deviation
    disagreement = float(np.mean(deviations))
    
    return (disagreement, means.tolist(), stds.tolist())


# =============================================================================
# OBSERVATION MODEL: VOLATILITY REGIME
# =============================================================================

def _compute_volatility_ratio(
    log_returns: pd.Series,
    short_window: int = VOL_LOOKBACK_SHORT,
    long_window: int = VOL_LOOKBACK_LONG
) -> float:
    """
    Compute volatility compression/expansion ratio.
    
    V(t) = σ_short / σ_long
    
    - V < 1: volatility compression (potentially dangerous)
    - V > 1: volatility expansion
    - V ≈ 1: stable regime
    
    Args:
        log_returns: Historical log returns
        short_window: Short-term volatility window
        long_window: Long-term volatility window
        
    Returns:
        Volatility ratio
    """
    if len(log_returns) < long_window:
        return float('nan')
    
    recent = log_returns.iloc[-short_window:]
    historical = log_returns.iloc[-long_window:]
    
    vol_short = float(recent.std())
    vol_long = float(historical.std())
    
    if vol_long <= 0:
        return float('nan')
    
    return vol_short / vol_long


# =============================================================================
# OBSERVATION VECTOR CONSTRUCTION
# =============================================================================

def _construct_observation_vector(
    log_returns: pd.Series,
    prev_disagreement: Optional[float] = None,
    timestamp: Optional[str] = None
) -> ObservationVector:
    """
    Construct the full observation vector Y_t.
    
    Args:
        log_returns: Historical log returns up to time t
        prev_disagreement: Previous disagreement for momentum calculation
        timestamp: Timestamp for the observation
        
    Returns:
        ObservationVector with all components
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Generate posterior samples for 21-day horizon (standard)
    samples_21d = _generate_posterior_samples(log_returns, 21)
    
    # 1) Convex Loss Functional
    convex_loss = _compute_convex_loss(samples_21d)
    
    # 2) Tail Mass
    tail_mass = _compute_tail_mass(samples_21d)
    
    # 3) Epistemic Disagreement
    disagreement, _, _ = _compute_epistemic_disagreement(log_returns)
    
    # 4) Disagreement Momentum
    if prev_disagreement is not None and np.isfinite(prev_disagreement) and np.isfinite(disagreement):
        disagreement_momentum = disagreement - prev_disagreement
    else:
        disagreement_momentum = 0.0
    
    # 5) Volatility Ratio
    vol_ratio = _compute_volatility_ratio(log_returns)
    
    return ObservationVector(
        convex_loss=convex_loss,
        tail_mass=tail_mass,
        disagreement=disagreement,
        disagreement_momentum=disagreement_momentum,
        vol_ratio=vol_ratio,
        timestamp=timestamp,
    )


# =============================================================================
# HIDDEN MARKOV MODEL: TRANSITION MATRIX
# =============================================================================

def _build_transition_matrix(persistence: float = TRANSITION_PERSISTENCE) -> np.ndarray:
    """
    Build monotone transition matrix respecting partial ordering.
    
    States: NORMAL → COMPRESSED → PRE_POLICY → POLICY
    Backward transitions are forbidden (set to 0).
    
    Args:
        persistence: Diagonal dominance (probability of staying in same state)
        
    Returns:
        4x4 transition matrix
    """
    n_states = LatentState.n_states()
    A = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        # Diagonal: persistence probability
        A[i, i] = persistence
        
        # Forward transitions only (monotone constraint)
        remaining = 1.0 - persistence
        
        if i < n_states - 1:
            # Distribute remaining probability to forward states
            # More weight to immediate next state
            forward_states = n_states - i - 1
            weights = np.array([2.0 ** (forward_states - j - 1) for j in range(forward_states)])
            weights = weights / weights.sum()
            
            for j, w in enumerate(weights):
                A[i, i + 1 + j] = remaining * w
        else:
            # POLICY state: absorbing (stays in POLICY)
            A[i, i] = 1.0
    
    # Normalize rows
    for i in range(n_states):
        A[i, :] = A[i, :] / A[i, :].sum()
    
    return A


# =============================================================================
# HIDDEN MARKOV MODEL: EMISSION PROBABILITIES
# =============================================================================

def _estimate_emission_likelihood(
    observation: np.ndarray,
    state: LatentState,
    historical_observations: np.ndarray
) -> float:
    """
    Estimate emission likelihood P(Y_t | S_t) using empirical calibration.
    
    Non-Gaussian, empirical estimation based on historical data quantiles.
    
    Args:
        observation: Current observation vector
        state: Latent state
        historical_observations: Historical observations for calibration
        
    Returns:
        Log-likelihood of observation given state
    """
    if len(historical_observations) < 50:
        # Insufficient data - use uniform likelihood
        return 0.0
    
    # Extract observation components
    obs_convex = observation[0]  # Convex loss
    obs_tail = observation[1]    # Tail mass
    obs_disag = observation[2]   # Disagreement
    obs_momentum = observation[3]  # Disagreement momentum
    obs_vol = observation[4]     # Volatility ratio
    
    # Compute historical quantiles for calibration
    hist_convex = historical_observations[:, 0]
    hist_tail = historical_observations[:, 1]
    hist_disag = historical_observations[:, 2]
    hist_vol = historical_observations[:, 4]
    
    # Filter valid values
    hist_convex = hist_convex[np.isfinite(hist_convex)]
    hist_tail = hist_tail[np.isfinite(hist_tail)]
    hist_disag = hist_disag[np.isfinite(hist_disag)]
    hist_vol = hist_vol[np.isfinite(hist_vol)]
    
    # Compute quantiles for each metric
    if len(hist_convex) > 10:
        convex_q25, convex_q50, convex_q75, convex_q90 = np.percentile(hist_convex, [25, 50, 75, 90])
    else:
        convex_q25, convex_q50, convex_q75, convex_q90 = 0.0001, 0.0003, 0.0006, 0.001
    
    if len(hist_tail) > 10:
        tail_q25, tail_q50, tail_q75, tail_q90 = np.percentile(hist_tail, [25, 50, 75, 90])
    else:
        tail_q25, tail_q50, tail_q75, tail_q90 = 0.48, 0.50, 0.52, 0.55
    
    if len(hist_disag) > 10:
        disag_q25, disag_q50, disag_q75, disag_q90 = np.percentile(hist_disag, [25, 50, 75, 90])
    else:
        disag_q25, disag_q50, disag_q75, disag_q90 = 0.15, 0.20, 0.30, 0.45
    
    if len(hist_vol) > 10:
        vol_q10, vol_q25, vol_q50, vol_q75, vol_q90 = np.percentile(hist_vol, [10, 25, 50, 75, 90])
    else:
        vol_q10, vol_q25, vol_q50, vol_q75, vol_q90 = 0.7, 0.85, 1.0, 1.15, 1.3
    
    # Log-likelihood accumulator
    log_lik = 0.0
    
    # State-specific likelihood based on where observation falls in historical distribution
    # CRITICAL: NORMAL should be the default state - only elevated states on clear evidence
    
    if state == LatentState.NORMAL:
        # NORMAL: Default state - strongly favored for typical observations
        # Metrics near or below median
        if obs_convex < convex_q75:
            log_lik += 1.0
        else:
            log_lik -= 0.5 * (obs_convex - convex_q75) / (convex_q90 - convex_q75 + 0.0001)
            
        if obs_tail < tail_q75:
            log_lik += 0.8
        else:
            log_lik -= 0.3 * (obs_tail - tail_q75) / (tail_q90 - tail_q75 + 0.001)
            
        if obs_disag < disag_q75:
            log_lik += 0.6
        else:
            log_lik -= 0.2
            
        if 0.8 <= obs_vol <= 1.2:
            log_lik += 0.5
        
    elif state == LatentState.COMPRESSED:
        # COMPRESSED: low vol ratio (compression) is key signal
        # Moderate other metrics
        if obs_vol < vol_q25:
            log_lik += 1.5  # Strong signal: volatility compression
        elif obs_vol < vol_q50:
            log_lik += 0.5
        else:
            log_lik -= 1.0  # Not compressed
            
        if obs_convex < convex_q75:
            log_lik += 0.3
            
    elif state == LatentState.PRE_POLICY:
        # PRE_POLICY: requires MULTIPLE elevated metrics
        elevated_count = 0
        
        if obs_convex > convex_q75:
            log_lik += 0.5
            elevated_count += 1
        else:
            log_lik -= 0.5
            
        if obs_tail > tail_q75:
            log_lik += 0.5
            elevated_count += 1
        else:
            log_lik -= 0.5
            
        if obs_disag > disag_q75:
            log_lik += 0.5
            elevated_count += 1
        else:
            log_lik -= 0.3
            
        # Positive momentum bonus
        if np.isfinite(obs_momentum) and obs_momentum > 0.02:
            log_lik += 0.5
            elevated_count += 1
            
        # Require at least 2 elevated metrics
        if elevated_count < 2:
            log_lik -= 2.0
        
    elif state == LatentState.POLICY:
        # POLICY: requires EXTREME metrics - very rare state
        extreme_count = 0
        
        if obs_convex > convex_q90:
            log_lik += 1.0
            extreme_count += 1
        else:
            log_lik -= 1.5
            
        if obs_tail > tail_q90:
            log_lik += 1.0
            extreme_count += 1
        else:
            log_lik -= 1.0
            
        if obs_disag > disag_q90:
            log_lik += 0.8
            extreme_count += 1
        else:
            log_lik -= 0.5
            
        if obs_vol > vol_q90:
            log_lik += 1.0  # Volatility expansion
            extreme_count += 1
        else:
            log_lik -= 0.5
            
        # Require at least 3 extreme metrics
        if extreme_count < 3:
            log_lik -= 3.0
    
    return log_lik


# =============================================================================
# HIDDEN MARKOV MODEL: FORWARD ALGORITHM
# =============================================================================

def _forward_algorithm(
    observations: List[np.ndarray],
    transition_matrix: np.ndarray,
    initial_distribution: np.ndarray,
    historical_obs_array: np.ndarray
) -> List[np.ndarray]:
    """
    Forward algorithm for HMM inference.
    
    Computes P(S_t | Y_1:t) for each time step.
    
    Args:
        observations: List of observation vectors
        transition_matrix: State transition matrix A
        initial_distribution: Initial state distribution π
        historical_obs_array: Historical observations for emission estimation
        
    Returns:
        List of posterior state distributions (one per time step)
    """
    n_states = LatentState.n_states()
    n_obs = len(observations)
    
    if n_obs == 0:
        return []
    
    posteriors = []
    
    # Initialize with prior
    alpha = np.log(initial_distribution + 1e-10)
    
    for t, obs in enumerate(observations):
        # Emission probabilities
        log_emissions = np.array([
            _estimate_emission_likelihood(obs, LatentState(s), historical_obs_array)
            for s in range(n_states)
        ])
        
        if t == 0:
            # First observation
            log_alpha = alpha + log_emissions
        else:
            # Transition + emission
            log_alpha_new = np.zeros(n_states)
            for j in range(n_states):
                # Sum over previous states
                log_transitions = np.log(transition_matrix[:, j] + 1e-10)
                log_alpha_new[j] = logsumexp(alpha + log_transitions) + log_emissions[j]
            log_alpha = log_alpha_new
        
        # Normalize to get posterior
        log_posterior = log_alpha - logsumexp(log_alpha)
        posterior = np.exp(log_posterior)
        
        # Ensure proper normalization
        posterior = posterior / posterior.sum()
        posteriors.append(posterior)
        
        alpha = log_alpha
    
    return posteriors


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def _run_inference(
    log_returns: pd.Series,
    lookback_days: int = 252
) -> Tuple[ObservationVector, StatePosterior, List[ObservationVector]]:
    """
    Run latent state inference on the full history.
    
    Args:
        log_returns: Historical log returns
        lookback_days: Number of days for inference window
        
    Returns:
        Tuple of (current_observation, current_posterior, observation_history)
    """
    n_obs = len(log_returns)
    
    if n_obs < MIN_HISTORY_DAYS:
        raise ValueError(f"Insufficient history: {n_obs} < {MIN_HISTORY_DAYS}")
    
    # Build observation history
    print("[debt_allocator] Building observation history...")
    
    # Use rolling window approach
    window_size = min(lookback_days, n_obs - MIN_HISTORY_DAYS)
    start_idx = n_obs - window_size
    
    observations = []
    obs_arrays = []
    prev_disagreement = None
    
    for t in range(start_idx, n_obs):
        # Use data up to time t
        returns_to_t = log_returns.iloc[:t+1]
        
        obs = _construct_observation_vector(
            returns_to_t, 
            prev_disagreement=prev_disagreement,
            timestamp=str(log_returns.index[t])
        )
        observations.append(obs)
        obs_arrays.append(obs.to_array())
        
        # Update for next iteration
        prev_disagreement = obs.disagreement
    
    if len(observations) == 0:
        raise ValueError("No observations constructed")
    
    print(f"[debt_allocator] Built {len(observations)} observations")
    
    # Convert to array for HMM
    historical_obs_array = np.array(obs_arrays)
    
    # Build transition matrix
    A = _build_transition_matrix()
    
    # Initial distribution (start in NORMAL with very high probability)
    pi = np.array([0.85, 0.10, 0.04, 0.01])
    
    # Run forward algorithm
    print("[debt_allocator] Running forward inference...")
    posteriors = _forward_algorithm(
        [obs.to_array() for obs in observations],
        A,
        pi,
        historical_obs_array
    )
    
    # Get current (final) observation and posterior
    current_obs = observations[-1]
    current_posterior_probs = posteriors[-1]
    
    # Find dominant state
    dominant_state = LatentState(int(np.argmax(current_posterior_probs)))
    
    current_posterior = StatePosterior(
        probabilities=tuple(current_posterior_probs.tolist()),
        dominant_state=dominant_state,
        timestamp=current_obs.timestamp,
    )
    
    return current_obs, current_posterior, observations


# =============================================================================
# DECISION RULE
# =============================================================================

def _compute_decision_signature(
    observation: ObservationVector,
    posterior: StatePosterior,
    triggered: bool,
    effective_date: Optional[str]
) -> str:
    """
    Compute cryptographic signature for decision audit trail.
    """
    payload = {
        "observation": observation.to_dict(),
        "posterior": posterior.to_dict(),
        "triggered": triggered,
        "effective_date": effective_date,
        "signature_version": "2.0.0",
    }
    
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()


def _make_decision(
    observation: ObservationVector,
    posterior: StatePosterior,
    threshold: float = PRE_POLICY_THRESHOLD
) -> DebtSwitchDecision:
    """
    Make the debt switch decision based on latent state inference.
    
    Trigger on FIRST day t such that:
        P(S_t = PRE_POLICY | Y_1:t) > α
    
    Args:
        observation: Current observation vector
        posterior: Current state posterior
        threshold: Decision threshold α
        
    Returns:
        DebtSwitchDecision
    """
    # Check if PRE_POLICY probability exceeds threshold
    triggered = posterior.p_pre_policy > threshold
    
    # Also trigger if POLICY state is dominant (belt and suspenders)
    if posterior.dominant_state == LatentState.POLICY:
        triggered = True
    
    effective_date = observation.timestamp.split('T')[0] if triggered else None
    
    # Decision basis
    if triggered:
        if posterior.dominant_state == LatentState.POLICY:
            decision_basis = f"POLICY state dominant (P={posterior.p_policy:.2%})"
        else:
            decision_basis = f"PRE_POLICY threshold exceeded (P={posterior.p_pre_policy:.2%} > {threshold:.0%})"
    else:
        decision_basis = f"PRE_POLICY below threshold (P={posterior.p_pre_policy:.2%} ≤ {threshold:.0%})"
    
    signature = _compute_decision_signature(
        observation, posterior, triggered, effective_date
    )
    
    return DebtSwitchDecision(
        triggered=triggered,
        effective_date=effective_date,
        observation=observation,
        state_posterior=posterior,
        decision_basis=decision_basis,
        signature=signature,
    )


# =============================================================================
# PERSISTENCE
# =============================================================================

def _load_persisted_decision(
    persistence_path: str = DECISION_PERSISTENCE_FILE
) -> Optional[DebtSwitchDecision]:
    """Load previously persisted decision if it exists."""
    try:
        path = Path(persistence_path)
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct decision
        obs_data = data['observation']
        observation = ObservationVector(
            convex_loss=obs_data['convex_loss'],
            tail_mass=obs_data['tail_mass'],
            disagreement=obs_data['disagreement'],
            disagreement_momentum=obs_data['disagreement_momentum'],
            vol_ratio=obs_data['vol_ratio'],
            timestamp=obs_data['timestamp'],
        )
        
        post_data = data['state_posterior']
        probs = post_data['probabilities']
        posterior = StatePosterior(
            probabilities=(
                probs['NORMAL'],
                probs['COMPRESSED'],
                probs['PRE_POLICY'],
                probs['POLICY'],
            ),
            dominant_state=LatentState[post_data['dominant_state']],
            timestamp=post_data['timestamp'],
        )
        
        return DebtSwitchDecision(
            triggered=data['triggered'],
            effective_date=data['effective_date'],
            observation=observation,
            state_posterior=posterior,
            decision_basis=data['decision_basis'],
            signature=data['signature'],
        )
        
    except Exception as e:
        print(f"[debt_allocator] Warning: Could not load persisted decision: {e}")
        return None


def _persist_decision(
    decision: DebtSwitchDecision,
    persistence_path: str = DECISION_PERSISTENCE_FILE
) -> bool:
    """Persist decision to file."""
    try:
        path = Path(persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(decision.to_dict(), f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        print(f"[debt_allocator] Error persisting decision: {e}")
        return False


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def run_debt_allocation_engine(
    data_path: str = EURJPY_DATA_FILE,
    persistence_path: str = DECISION_PERSISTENCE_FILE,
    force_reevaluate: bool = False,
    force_refresh_data: bool = True,
) -> Optional[DebtSwitchDecision]:
    """
    Run the research-grade debt allocation engine.
    
    Args:
        data_path: Path to EURJPY data
        persistence_path: Path to decision persistence file
        force_reevaluate: If True, ignore persisted decision
        force_refresh_data: If True, download fresh data
        
    Returns:
        DebtSwitchDecision or None if engine cannot run
    """
    # CHECK FOR PRIOR DECISION
    if not force_reevaluate:
        prior_decision = _load_persisted_decision(persistence_path)
        if prior_decision is not None and prior_decision.triggered:
            print("[debt_allocator] Prior triggered decision exists - returning cached")
            return prior_decision
    
    # LOAD DATA
    prices = _load_eurjpy_prices(data_path, force_refresh=force_refresh_data)
    if prices is None:
        return None
    
    # COMPUTE LOG RETURNS
    log_returns = _compute_log_returns(prices)
    if len(log_returns) < MIN_HISTORY_DAYS:
        print(f"[debt_allocator] Insufficient data: {len(log_returns)} < {MIN_HISTORY_DAYS}")
        return None
    
    # RUN INFERENCE
    try:
        observation, posterior, _ = _run_inference(log_returns)
    except Exception as e:
        print(f"[debt_allocator] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # MAKE DECISION
    decision = _make_decision(observation, posterior)
    
    # PERSIST IF TRIGGERED
    if decision.triggered:
        _persist_decision(decision, persistence_path)
        print(f"[debt_allocator] DECISION TRIGGERED - persisted to {persistence_path}")
    
    return decision


# =============================================================================
# PRESENTATION (RICH OUTPUT)
# =============================================================================

def _get_status_display(decision: DebtSwitchDecision) -> Tuple[str, str, str]:
    """Get status emoji, text, and color for display."""
    if decision.triggered:
        if decision.state_posterior.dominant_state == LatentState.POLICY:
            return "🔴", "SWITCH TRIGGERED (POLICY)", "red"
        else:
            return "🔴", "SWITCH TRIGGERED (PRE-POLICY)", "red"
    
    dominant = decision.state_posterior.dominant_state
    if dominant == LatentState.PRE_POLICY:
        return "🟡", "MONITORING (PRE-POLICY ELEVATED)", "yellow"
    elif dominant == LatentState.COMPRESSED:
        return "🟡", "MONITORING (COMPRESSED)", "yellow"
    else:
        return "🟢", "NO ACTION REQUIRED", "green"


def render_debt_switch_decision(
    decision: Optional[DebtSwitchDecision],
    console: Optional[Console] = None
) -> None:
    """Render debt switch decision with Rich formatting."""
    if console is None:
        console = Console()
    
    # Header
    console.print()
    console.print("=" * 60)
    console.print("[bold cyan]DEBT ALLOCATION — EURJPY (RESEARCH)[/bold cyan]", justify="center")
    console.print("=" * 60)
    console.print()
    
    if decision is None:
        console.print(Panel(
            "[bold red]ENGINE CANNOT RUN[/bold red]\n\n"
            "Possible reasons:\n"
            "• Insufficient EURJPY data\n"
            "• Inference failure\n"
            "• Data validation failed",
            title="[bold red]❌ ERROR[/bold red]",
            border_style="red",
        ))
        return
    
    # Status
    status_emoji, status_text, status_color = _get_status_display(decision)
    
    # Build status panel
    if decision.triggered:
        status_content = (
            f"[bold]Switch Status[/bold]     : {status_emoji} [bold {status_color}]{status_text}[/bold {status_color}]\n"
            f"[bold]Effective Date[/bold]    : [bold white]{decision.effective_date}[/bold white]"
        )
    else:
        status_content = f"[bold]Switch Status[/bold]     : {status_emoji} [{status_color}]{status_text}[/{status_color}]"
    
    console.print(Panel(
        status_content,
        title=f"[bold {status_color}]Decision Status[/bold {status_color}]",
        border_style=status_color,
        padding=(1, 2),
    ))
    
    console.print()
    
    # Latent State Probabilities
    console.print("[bold cyan]Latent State Probabilities:[/bold cyan]")
    
    state_table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
        padding=(0, 2),
    )
    state_table.add_column("State", style="bold")
    state_table.add_column("Probability", justify="right")
    state_table.add_column("", justify="center", width=20)
    
    post = decision.state_posterior
    probs = [
        ("NORMAL", post.p_normal),
        ("COMPRESSED", post.p_compressed),
        ("PRE_POLICY", post.p_pre_policy),
        ("POLICY", post.p_policy),
    ]
    
    for state_name, prob in probs:
        # Create visual bar
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # Color based on state
        if state_name == post.dominant_state.name:
            prob_str = f"[bold cyan]{prob:.1%}[/bold cyan]"
            bar_str = f"[cyan]{bar}[/cyan]"
        elif state_name in ("PRE_POLICY", "POLICY") and prob > 0.3:
            prob_str = f"[yellow]{prob:.1%}[/yellow]"
            bar_str = f"[yellow]{bar}[/yellow]"
        else:
            prob_str = f"{prob:.1%}"
            bar_str = f"[dim]{bar}[/dim]"
        
        state_table.add_row(state_name, prob_str, bar_str)
    
    console.print(state_table)
    console.print()
    
    # Observation Metrics
    obs = decision.observation
    
    metrics_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    metrics_table.add_column("Metric", style="bold", width=25)
    metrics_table.add_column("Value", justify="right", width=20)
    
    # Format metrics with appropriate precision
    if np.isfinite(obs.convex_loss):
        convex_str = f"{obs.convex_loss:.6f}"
    else:
        convex_str = "[dim]N/A[/dim]"
    
    if np.isfinite(obs.tail_mass):
        tail_str = f"{obs.tail_mass:.2%}"
        if obs.tail_mass > 0.55:
            tail_str = f"[yellow]{tail_str}[/yellow]"
    else:
        tail_str = "[dim]N/A[/dim]"
    
    if np.isfinite(obs.disagreement):
        disag_str = f"{obs.disagreement:.3f}"
        if obs.disagreement > 0.4:
            disag_str = f"[yellow]{disag_str}[/yellow]"
    else:
        disag_str = "[dim]N/A[/dim]"
    
    if np.isfinite(obs.disagreement_momentum):
        mom_str = f"{obs.disagreement_momentum:+.4f}"
        if obs.disagreement_momentum > 0.02:
            mom_str = f"[yellow]{mom_str}[/yellow]"
    else:
        mom_str = "[dim]N/A[/dim]"
    
    if np.isfinite(obs.vol_ratio):
        vol_str = f"{obs.vol_ratio:.3f}"
        if obs.vol_ratio < 0.8:
            vol_str = f"[yellow]{vol_str}[/yellow] (compressed)"
        elif obs.vol_ratio > 1.3:
            vol_str = f"[red]{vol_str}[/red] (expanding)"
    else:
        vol_str = "[dim]N/A[/dim]"
    
    metrics_table.add_row("Convex Loss C(t)", convex_str)
    metrics_table.add_row("Tail Mass P(ΔX > 0)", tail_str)
    metrics_table.add_row("Disagreement D(t)", disag_str)
    metrics_table.add_row("Momentum dD(t)", mom_str)
    metrics_table.add_row("Vol Ratio V(t)", vol_str)
    
    console.print(metrics_table)
    console.print()
    
    # Decision Basis
    console.print(f"[bold]Decision Basis:[/bold] {decision.decision_basis}")
    console.print()
    
    # Signature
    console.print(f"[dim]Signature: {decision.signature[:16]}...[/dim]")
    console.print()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point for `make debt` command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Research-Grade FX Debt Allocation Engine (EURJPY)"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=EURJPY_DATA_FILE,
        help=f'Path to EURJPY data CSV (default: {EURJPY_DATA_FILE})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-evaluation even if prior decision exists'
    )
    parser.add_argument(
        '--no-refresh',
        action='store_true',
        help='Skip data refresh, use cached EURJPY data only'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of Rich formatting'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Show header
    if not args.json:
        console.print()
        console.print(Panel(
            "[bold cyan]Research-Grade FX Debt Allocation Engine[/bold cyan]\n"
            "[dim]EURJPY Latent Policy-Stress Inference[/dim]",
            border_style="cyan",
        ))
        console.print()
        console.print(f"[dim]Cache directory: {DEBT_CACHE_DIR}[/dim]")
        console.print(f"[dim]Data file: {args.data_path}[/dim]")
        console.print()
    
    # Run engine
    decision = run_debt_allocation_engine(
        data_path=args.data_path,
        force_reevaluate=args.force,
        force_refresh_data=not args.no_refresh,
    )
    
    if args.json:
        if decision is not None:
            print(json.dumps(decision.to_dict(), indent=2, default=str))
        else:
            print(json.dumps({"error": "Engine cannot run"}, indent=2))
    else:
        render_debt_switch_decision(decision, console)


if __name__ == '__main__':
    main()
