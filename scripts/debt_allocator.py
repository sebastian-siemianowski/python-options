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
Version: 3.0.0

UPGRADES (v3.0.0):
    1. Endogenous state transitions: P(S_t|S_{t-1}, Y_{t-1})
    2. PRE_POLICY split into A (silent stress) and B (unstable compression)
    3. Explicit tail skew dynamics: Skew(t) = E[(ΔX-μ)³]/σ³
    4. Meta-uncertainty with model averaging: P(S_t|Y) = Σ P(S_t|M_k,Y)P(M_k|Y)
    5. Dynamic decision boundary: α(t) = f(convexity, leverage, carry)
    6. Explicit "do nothing" dominance check: E[Loss_stay] - E[Loss_switch]
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
    
    Partially ordered: NORMAL → COMPRESSED → PRE_POLICY_A → PRE_POLICY_B → POLICY
    
    PRE_POLICY split (v3.0.0):
        - PRE_POLICY_A: Silent stress accumulation (carry still works)
        - PRE_POLICY_B: Unstable compression (carry looks safe but isn't)
    
    Backward transitions are forbidden except via explicit reset.
    """
    NORMAL = 0
    COMPRESSED = 1
    PRE_POLICY_A = 2  # Silent stress accumulation
    PRE_POLICY_B = 3  # Unstable compression
    POLICY = 4
    
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
        skew: Directional skew dynamics (v3.0.0) - E[(ΔX-μ)³]/σ³
        skew_momentum: Change in skew (v3.0.0)
    """
    convex_loss: float      # C(t)
    tail_mass: float        # P(t) = P(ΔX > 0)
    disagreement: float     # D(t)
    disagreement_momentum: float  # dD(t)
    vol_ratio: float        # V(t) = σ_short / σ_long
    skew: float             # Skew(t) = E[(ΔX-μ)³]/σ³ (v3.0.0)
    skew_momentum: float    # dSkew(t) (v3.0.0)
    timestamp: str
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for HMM processing."""
        return np.array([
            self.convex_loss,
            self.tail_mass,
            self.disagreement,
            self.disagreement_momentum,
            self.vol_ratio,
            self.skew,
            self.skew_momentum,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "convex_loss": self.convex_loss,
            "tail_mass": self.tail_mass,
            "disagreement": self.disagreement,
            "disagreement_momentum": self.disagreement_momentum,
            "vol_ratio": self.vol_ratio,
            "skew": self.skew,
            "skew_momentum": self.skew_momentum,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class StatePosterior:
    """
    Posterior state probabilities P(S_t | Y_1:t).
    
    v3.0.0: Includes model-averaged posterior and model weights for meta-uncertainty.
    """
    probabilities: Tuple[float, float, float, float, float]  # One per state (5 states)
    dominant_state: LatentState
    timestamp: str
    model_weights: Optional[Tuple[float, ...]] = None  # P(M_k | Y_1:t) (v3.0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "probabilities": {
                LatentState.NORMAL.name: self.probabilities[0],
                LatentState.COMPRESSED.name: self.probabilities[1],
                LatentState.PRE_POLICY_A.name: self.probabilities[2],
                LatentState.PRE_POLICY_B.name: self.probabilities[3],
                LatentState.POLICY.name: self.probabilities[4],
            },
            "dominant_state": self.dominant_state.name,
            "timestamp": self.timestamp,
        }
        if self.model_weights is not None:
            result["model_weights"] = list(self.model_weights)
        return result
    
    @property
    def p_normal(self) -> float:
        return self.probabilities[LatentState.NORMAL]
    
    @property
    def p_compressed(self) -> float:
        return self.probabilities[LatentState.COMPRESSED]
    
    @property
    def p_pre_policy_a(self) -> float:
        return self.probabilities[LatentState.PRE_POLICY_A]
    
    @property
    def p_pre_policy_b(self) -> float:
        return self.probabilities[LatentState.PRE_POLICY_B]
    
    @property
    def p_pre_policy(self) -> float:
        """Combined PRE_POLICY probability (A + B)."""
        return self.p_pre_policy_a + self.p_pre_policy_b
    
    @property
    def p_policy(self) -> float:
        return self.probabilities[LatentState.POLICY]


@dataclass(frozen=True)
class DebtSwitchDecision:
    """
    Immutable record of debt switch decision.
    
    v3.0.0 additions:
        - dynamic_alpha: Computed decision threshold α(t) based on context
        - expected_loss_delta: E[Loss_stay] - E[Loss_switch] (dominance check)
        - dominance_margin: Minimum required margin for switching
    """
    triggered: bool
    effective_date: Optional[str]
    observation: ObservationVector
    state_posterior: StatePosterior
    decision_basis: str
    signature: str
    dynamic_alpha: float = 0.60  # v3.0.0: Dynamic threshold
    expected_loss_delta: Optional[float] = None  # v3.0.0: E[Loss_stay] - E[Loss_switch]
    dominance_margin: float = 0.0  # v3.0.0: Required margin for dominance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "effective_date": self.effective_date,
            "observation": self.observation.to_dict(),
            "state_posterior": self.state_posterior.to_dict(),
            "decision_basis": self.decision_basis,
            "signature": self.signature,
            "dynamic_alpha": self.dynamic_alpha,
            "expected_loss_delta": self.expected_loss_delta,
            "dominance_margin": self.dominance_margin,
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
# OBSERVATION MODEL: SKEW DYNAMICS (v3.0.0 - Improvement 3)
# =============================================================================

def _compute_skew(
    log_returns: pd.Series,
    window: int = VOL_LOOKBACK_SHORT
) -> float:
    """
    Compute directional skew dynamics.
    
    Skew(t) = E[(ΔX - μ)³] / σ³
    
    Why this matters (v3.0.0):
    - JPY stress events show rapid skew sign changes
    - This often precedes volatility expansion
    - Helps discriminate between "good JPY strength" and "dangerous JPY strength"
    
    Args:
        log_returns: Historical log returns
        window: Lookback window for skew calculation
        
    Returns:
        Standardized skewness
    """
    if len(log_returns) < window:
        return float('nan')
    
    recent = log_returns.iloc[-window:].values
    
    if len(recent) < 10:
        return float('nan')
    
    mu = np.mean(recent)
    sigma = np.std(recent)
    
    if sigma <= 1e-10:
        return float('nan')
    
    # Compute standardized third moment
    centered = recent - mu
    skew = np.mean(centered ** 3) / (sigma ** 3)
    
    return float(skew)


def _compute_skew_momentum(
    log_returns: pd.Series,
    window: int = VOL_LOOKBACK_SHORT,
    lag: int = 5
) -> Tuple[float, float]:
    """
    Compute skew and its momentum (rate of change).
    
    v3.0.0: Tracks how rapidly skew is changing, which is a key
    early warning signal for stress events.
    
    Args:
        log_returns: Historical log returns
        window: Lookback window for skew calculation
        lag: Days to look back for momentum calculation
        
    Returns:
        Tuple of (current_skew, skew_momentum)
    """
    if len(log_returns) < window + lag:
        current_skew = _compute_skew(log_returns, window)
        return (current_skew, 0.0)
    
    current_skew = _compute_skew(log_returns, window)
    lagged_skew = _compute_skew(log_returns.iloc[:-lag], window)
    
    if np.isfinite(current_skew) and np.isfinite(lagged_skew):
        skew_momentum = current_skew - lagged_skew
    else:
        skew_momentum = 0.0
    
    return (current_skew, skew_momentum)


# =============================================================================
# OBSERVATION VECTOR CONSTRUCTION
# =============================================================================

def _construct_observation_vector(
    log_returns: pd.Series,
    prev_disagreement: Optional[float] = None,
    prev_skew: Optional[float] = None,
    timestamp: Optional[str] = None
) -> ObservationVector:
    """
    Construct the full observation vector Y_t.
    
    v3.0.0: Added skew and skew_momentum for directional asymmetry tracking.
    
    Args:
        log_returns: Historical log returns up to time t
        prev_disagreement: Previous disagreement for momentum calculation
        prev_skew: Previous skew for momentum calculation (v3.0.0)
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
    
    # 6) Skew Dynamics (v3.0.0 - Improvement 3)
    skew, skew_momentum = _compute_skew_momentum(log_returns)
    
    # Override skew momentum if we have explicit previous skew
    if prev_skew is not None and np.isfinite(prev_skew) and np.isfinite(skew):
        skew_momentum = skew - prev_skew
    
    return ObservationVector(
        convex_loss=convex_loss,
        tail_mass=tail_mass,
        disagreement=disagreement,
        disagreement_momentum=disagreement_momentum,
        vol_ratio=vol_ratio,
        skew=skew,
        skew_momentum=skew_momentum,
        timestamp=timestamp,
    )


# =============================================================================
# HIDDEN MARKOV MODEL: TRANSITION MATRIX (v3.0.0 - Endogenous)
# =============================================================================

def _build_base_transition_matrix(persistence: float = TRANSITION_PERSISTENCE) -> np.ndarray:
    """
    Build base monotone transition matrix respecting partial ordering.
    
    States: NORMAL → COMPRESSED → PRE_POLICY_A → PRE_POLICY_B → POLICY
    Backward transitions are forbidden (set to 0).
    
    v3.0.0: Extended to 5 states with PRE_POLICY split.
    
    Args:
        persistence: Diagonal dominance (probability of staying in same state)
        
    Returns:
        5x5 transition matrix
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


def _compute_endogenous_transition_matrix(
    base_matrix: np.ndarray,
    observation: np.ndarray,
    historical_quantiles: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute observation-dependent transition matrix.
    
    v3.0.0 - Improvement 1: Endogenize state transitions
    
    P(S_t = s_j | S_{t-1} = s_i, Y_{t-1})
    
    Key insight: Policy stress does not accumulate at a constant rate.
    When fragility accelerates, transition to PRE_POLICY becomes more likely.
    When volatility compresses, POLICY becomes more likely.
    
    This turns the model into a controlled latent process, not a passive one.
    
    Args:
        base_matrix: Base transition matrix (without observation conditioning)
        observation: Current observation vector Y_t
        historical_quantiles: Dict with quantile arrays for each metric
        
    Returns:
        Observation-conditioned 5x5 transition matrix
    """
    n_states = LatentState.n_states()
    A = base_matrix.copy()
    
    # Extract observation components
    obs_convex = observation[0]  # Convex loss
    obs_tail = observation[1]    # Tail mass
    obs_disag = observation[2]   # Disagreement
    obs_vol = observation[4]     # Volatility ratio
    obs_skew = observation[5] if len(observation) > 5 else 0.0  # Skew
    obs_skew_mom = observation[6] if len(observation) > 6 else 0.0  # Skew momentum
    
    # Get quantiles
    convex_q75 = historical_quantiles.get('convex_q75', 0.0006)
    convex_q90 = historical_quantiles.get('convex_q90', 0.001)
    tail_q75 = historical_quantiles.get('tail_q75', 0.52)
    disag_q75 = historical_quantiles.get('disag_q75', 0.30)
    vol_q25 = historical_quantiles.get('vol_q25', 0.85)
    
    # Compute stress acceleration factor
    # Higher values = faster transition to stressed states
    stress_factor = 1.0
    
    # Convex loss elevation increases forward transition probability
    if np.isfinite(obs_convex) and obs_convex > convex_q75:
        stress_factor += 0.5 * min((obs_convex - convex_q75) / (convex_q90 - convex_q75 + 1e-6), 2.0)
    
    # High tail mass increases stress
    if np.isfinite(obs_tail) and obs_tail > tail_q75:
        stress_factor += 0.3
    
    # Elevated disagreement increases uncertainty → faster transitions
    if np.isfinite(obs_disag) and obs_disag > disag_q75:
        stress_factor += 0.4
    
    # Volatility compression is a warning sign
    if np.isfinite(obs_vol) and obs_vol < vol_q25:
        stress_factor += 0.5
    
    # Rapid skew sign change is an early warning (v3.0.0)
    if np.isfinite(obs_skew_mom) and abs(obs_skew_mom) > 0.5:
        stress_factor += 0.6
    
    # Negative skew with momentum toward more negative = dangerous JPY strength
    if np.isfinite(obs_skew) and np.isfinite(obs_skew_mom):
        if obs_skew < -0.5 and obs_skew_mom < -0.1:
            stress_factor += 0.4
    
    # Clamp stress factor
    stress_factor = max(0.5, min(stress_factor, 3.0))
    
    # Modify transition probabilities based on stress factor
    for i in range(n_states - 1):  # Don't modify POLICY (absorbing)
        # Reduce persistence when stress is high
        adjusted_persistence = A[i, i] ** (1.0 / stress_factor)
        
        # Redistribute to forward states
        remaining = 1.0 - adjusted_persistence
        current_remaining = 1.0 - A[i, i]
        
        if current_remaining > 1e-6:
            scale = remaining / current_remaining
            for j in range(i + 1, n_states):
                A[i, j] = A[i, j] * scale
        
        A[i, i] = adjusted_persistence
    
    # Normalize rows
    for i in range(n_states):
        row_sum = A[i, :].sum()
        if row_sum > 0:
            A[i, :] = A[i, :] / row_sum
    
    return A


def _build_transition_matrix(persistence: float = TRANSITION_PERSISTENCE) -> np.ndarray:
    """
    Build monotone transition matrix respecting partial ordering.
    
    States: NORMAL → COMPRESSED → PRE_POLICY_A → PRE_POLICY_B → POLICY
    Backward transitions are forbidden (set to 0).
    
    v3.0.0: Extended to 5 states with PRE_POLICY split.
    
    Args:
        persistence: Diagonal dominance (probability of staying in same state)
        
    Returns:
        5x5 transition matrix
    """
    return _build_base_transition_matrix(persistence)


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
    
    v3.0.0: Extended to 5 states with PRE_POLICY_A/B split and skew dynamics.
    
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
    obs_skew = observation[5] if len(observation) > 5 else 0.0  # Skew (v3.0.0)
    obs_skew_mom = observation[6] if len(observation) > 6 else 0.0  # Skew momentum (v3.0.0)
    
    # Compute historical quantiles for calibration
    hist_convex = historical_observations[:, 0]
    hist_tail = historical_observations[:, 1]
    hist_disag = historical_observations[:, 2]
    hist_vol = historical_observations[:, 4]
    hist_skew = historical_observations[:, 5] if historical_observations.shape[1] > 5 else np.zeros(len(historical_observations))
    
    # Filter valid values
    hist_convex = hist_convex[np.isfinite(hist_convex)]
    hist_tail = hist_tail[np.isfinite(hist_tail)]
    hist_disag = hist_disag[np.isfinite(hist_disag)]
    hist_vol = hist_vol[np.isfinite(hist_vol)]
    hist_skew = hist_skew[np.isfinite(hist_skew)]
    
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
    
    # Skew quantiles (v3.0.0)
    if len(hist_skew) > 10:
        skew_q10, skew_q25, skew_q75, skew_q90 = np.percentile(hist_skew, [10, 25, 75, 90])
    else:
        skew_q10, skew_q25, skew_q75, skew_q90 = -0.5, -0.2, 0.2, 0.5
    
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
        
        # Skew near zero in NORMAL state (v3.0.0)
        if np.isfinite(obs_skew) and skew_q25 < obs_skew < skew_q75:
            log_lik += 0.3
        
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
            
    elif state == LatentState.PRE_POLICY_A:
        # PRE_POLICY_A (v3.0.0): Silent stress accumulation
        # Carry still appears to work, but subtle signs of stress
        elevated_count = 0
        
        # Moderately elevated convex loss
        if convex_q50 < obs_convex <= convex_q75:
            log_lik += 0.6
            elevated_count += 1
        elif obs_convex > convex_q75:
            log_lik += 0.3  # Too elevated for A
        else:
            log_lik -= 0.3
            
        # Tail mass starting to rise
        if tail_q50 < obs_tail <= tail_q75:
            log_lik += 0.5
            elevated_count += 1
        else:
            log_lik -= 0.2
            
        # Disagreement beginning to build
        if disag_q50 < obs_disag <= disag_q75:
            log_lik += 0.4
            elevated_count += 1
            
        # Vol ratio still moderate (not yet compressed/expanding)
        if vol_q25 < obs_vol < vol_q75:
            log_lik += 0.3
        
        # Skew starting to become negative (v3.0.0)
        if np.isfinite(obs_skew) and obs_skew < skew_q25:
            log_lik += 0.5
            elevated_count += 1
            
        # Require at least 2 subtle signals
        if elevated_count < 2:
            log_lik -= 1.5
            
    elif state == LatentState.PRE_POLICY_B:
        # PRE_POLICY_B (v3.0.0): Unstable compression
        # Carry looks safe but isn't - most regret happens here
        elevated_count = 0
        
        # Clearly elevated convex loss
        if obs_convex > convex_q75:
            log_lik += 0.7
            elevated_count += 1
        else:
            log_lik -= 0.5
            
        # High tail mass
        if obs_tail > tail_q75:
            log_lik += 0.6
            elevated_count += 1
        else:
            log_lik -= 0.4
            
        # Elevated disagreement
        if obs_disag > disag_q75:
            log_lik += 0.5
            elevated_count += 1
        else:
            log_lik -= 0.3
            
        # Positive momentum - things getting worse
        if np.isfinite(obs_momentum) and obs_momentum > 0.02:
            log_lik += 0.6
            elevated_count += 1
        
        # Strongly negative skew OR rapid skew change (v3.0.0)
        if np.isfinite(obs_skew) and obs_skew < skew_q10:
            log_lik += 0.7
            elevated_count += 1
        if np.isfinite(obs_skew_mom) and abs(obs_skew_mom) > 0.3:
            log_lik += 0.5
            elevated_count += 1
            
        # Require at least 3 elevated metrics for B state
        if elevated_count < 3:
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
        
        # Extreme skew (either direction) in POLICY (v3.0.0)
        if np.isfinite(obs_skew) and (obs_skew < skew_q10 or obs_skew > skew_q90):
            log_lik += 0.8
            extreme_count += 1
            
        # Require at least 3 extreme metrics
        if extreme_count < 3:
            log_lik -= 3.0
    
    return log_lik


# =============================================================================
# HIDDEN MARKOV MODEL: FORWARD ALGORITHM (v3.0.0 - Endogenous Transitions)
# =============================================================================

def _compute_historical_quantiles(historical_obs_array: np.ndarray) -> Dict[str, float]:
    """
    Compute historical quantiles for endogenous transition computation.
    
    v3.0.0: Used to condition transition probabilities on observations.
    """
    quantiles = {}
    
    if len(historical_obs_array) < 20:
        # Return defaults
        return {
            'convex_q75': 0.0006,
            'convex_q90': 0.001,
            'tail_q75': 0.52,
            'disag_q75': 0.30,
            'vol_q25': 0.85,
        }
    
    hist_convex = historical_obs_array[:, 0]
    hist_tail = historical_obs_array[:, 1]
    hist_disag = historical_obs_array[:, 2]
    hist_vol = historical_obs_array[:, 4]
    
    hist_convex = hist_convex[np.isfinite(hist_convex)]
    hist_tail = hist_tail[np.isfinite(hist_tail)]
    hist_disag = hist_disag[np.isfinite(hist_disag)]
    hist_vol = hist_vol[np.isfinite(hist_vol)]
    
    if len(hist_convex) > 10:
        quantiles['convex_q75'] = float(np.percentile(hist_convex, 75))
        quantiles['convex_q90'] = float(np.percentile(hist_convex, 90))
    else:
        quantiles['convex_q75'] = 0.0006
        quantiles['convex_q90'] = 0.001
    
    if len(hist_tail) > 10:
        quantiles['tail_q75'] = float(np.percentile(hist_tail, 75))
    else:
        quantiles['tail_q75'] = 0.52
    
    if len(hist_disag) > 10:
        quantiles['disag_q75'] = float(np.percentile(hist_disag, 75))
    else:
        quantiles['disag_q75'] = 0.30
    
    if len(hist_vol) > 10:
        quantiles['vol_q25'] = float(np.percentile(hist_vol, 25))
    else:
        quantiles['vol_q25'] = 0.85
    
    return quantiles


def _forward_algorithm(
    observations: List[np.ndarray],
    transition_matrix: np.ndarray,
    initial_distribution: np.ndarray,
    historical_obs_array: np.ndarray
) -> List[np.ndarray]:
    """
    Forward algorithm for HMM inference.
    
    Computes P(S_t | Y_1:t) for each time step.
    
    v3.0.0: Uses observation-dependent (endogenous) transition matrices.
    
    Args:
        observations: List of observation vectors
        transition_matrix: Base state transition matrix A
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
    
    # Compute historical quantiles for endogenous transitions (v3.0.0)
    hist_quantiles = _compute_historical_quantiles(historical_obs_array)
    
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
            # v3.0.0: Compute observation-dependent transition matrix
            A_t = _compute_endogenous_transition_matrix(
                transition_matrix, 
                observations[t-1],  # Condition on Y_{t-1}
                hist_quantiles
            )
            
            # Transition + emission
            log_alpha_new = np.zeros(n_states)
            for j in range(n_states):
                # Sum over previous states
                log_transitions = np.log(A_t[:, j] + 1e-10)
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
# META-MODEL UNCERTAINTY (v3.0.0 - Improvement 4)
# =============================================================================

@dataclass
class ModelSpecification:
    """
    Specification for an alternative latent stress model.
    
    v3.0.0: Used for Bayesian model averaging to prevent false certainty.
    """
    name: str
    persistence: float  # Transition persistence parameter
    convex_weight: float  # Weight for convex loss in emissions
    vol_weight: float  # Weight for volatility ratio in emissions
    skew_weight: float  # Weight for skew in emissions
    
    
def _get_model_specifications() -> List[ModelSpecification]:
    """
    Define alternative latent stress models for meta-uncertainty.
    
    v3.0.0 - Improvement 4: These represent different hypotheses about
    how stress accumulates and manifests in observations.
    """
    return [
        ModelSpecification(
            name="baseline",
            persistence=0.85,
            convex_weight=1.0,
            vol_weight=1.0,
            skew_weight=1.0,
        ),
        ModelSpecification(
            name="sticky",
            persistence=0.92,  # Higher persistence - slower transitions
            convex_weight=1.0,
            vol_weight=1.0,
            skew_weight=0.8,
        ),
        ModelSpecification(
            name="volatile",
            persistence=0.75,  # Lower persistence - faster transitions
            convex_weight=1.2,
            vol_weight=1.0,
            skew_weight=1.2,
        ),
        ModelSpecification(
            name="vol_driven",
            persistence=0.85,
            convex_weight=0.8,
            vol_weight=1.5,  # Emphasize volatility
            skew_weight=1.0,
        ),
        ModelSpecification(
            name="skew_driven",
            persistence=0.85,
            convex_weight=0.9,
            vol_weight=0.9,
            skew_weight=1.5,  # Emphasize skew
        ),
    ]


def _run_model_inference(
    observations: List[np.ndarray],
    historical_obs_array: np.ndarray,
    model_spec: ModelSpecification,
    initial_distribution: np.ndarray
) -> Tuple[List[np.ndarray], float]:
    """
    Run inference for a specific model specification.
    
    Returns:
        Tuple of (posteriors, marginal_likelihood)
    """
    # Build transition matrix with model's persistence
    A = _build_base_transition_matrix(model_spec.persistence)
    
    n_states = LatentState.n_states()
    n_obs = len(observations)
    
    if n_obs == 0:
        return [], 0.0
    
    posteriors = []
    marginal_log_lik = 0.0
    
    hist_quantiles = _compute_historical_quantiles(historical_obs_array)
    alpha = np.log(initial_distribution + 1e-10)
    
    for t, obs in enumerate(observations):
        # Weight observations according to model spec
        weighted_obs = obs.copy()
        weighted_obs[0] *= model_spec.convex_weight  # Convex loss
        weighted_obs[4] *= model_spec.vol_weight     # Vol ratio
        if len(weighted_obs) > 5:
            weighted_obs[5] *= model_spec.skew_weight  # Skew
        
        log_emissions = np.array([
            _estimate_emission_likelihood(weighted_obs, LatentState(s), historical_obs_array)
            for s in range(n_states)
        ])
        
        if t == 0:
            log_alpha = alpha + log_emissions
        else:
            A_t = _compute_endogenous_transition_matrix(
                A, observations[t-1], hist_quantiles
            )
            
            log_alpha_new = np.zeros(n_states)
            for j in range(n_states):
                log_transitions = np.log(A_t[:, j] + 1e-10)
                log_alpha_new[j] = logsumexp(alpha + log_transitions) + log_emissions[j]
            log_alpha = log_alpha_new
        
        # Accumulate marginal likelihood
        marginal_log_lik += logsumexp(log_alpha)
        
        # Normalize
        log_posterior = log_alpha - logsumexp(log_alpha)
        posterior = np.exp(log_posterior)
        posterior = posterior / posterior.sum()
        posteriors.append(posterior)
        
        alpha = log_alpha
    
    return posteriors, marginal_log_lik


def _compute_model_averaged_posterior(
    observations: List[np.ndarray],
    historical_obs_array: np.ndarray,
    initial_distribution: np.ndarray
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Compute model-averaged posterior using Bayesian model averaging.
    
    v3.0.0 - Improvement 4: Meta-uncertainty over the latent model
    
    P(S_t | Y_{1:t}) = Σ_k P(S_t | M_k, Y_{1:t}) P(M_k | Y_{1:t})
    
    This prevents false certainty when the world changes.
    
    Args:
        observations: List of observation vectors
        historical_obs_array: Historical observations for calibration
        initial_distribution: Initial state distribution
        
    Returns:
        Tuple of (model_averaged_posterior, model_weights)
    """
    models = _get_model_specifications()
    n_models = len(models)
    n_states = LatentState.n_states()
    
    # Uniform prior over models
    log_model_prior = -np.log(n_models)
    
    # Run inference for each model
    model_posteriors = []
    model_marginal_liks = []
    
    for spec in models:
        posteriors, marginal_lik = _run_model_inference(
            observations, historical_obs_array, spec, initial_distribution
        )
        model_posteriors.append(posteriors[-1] if posteriors else np.ones(n_states) / n_states)
        model_marginal_liks.append(marginal_lik)
    
    # Compute model posterior weights P(M_k | Y_{1:t})
    log_model_posteriors = np.array(model_marginal_liks) + log_model_prior
    log_normalizer = logsumexp(log_model_posteriors)
    model_weights = np.exp(log_model_posteriors - log_normalizer)
    
    # Compute model-averaged state posterior
    averaged_posterior = np.zeros(n_states)
    for k in range(n_models):
        averaged_posterior += model_weights[k] * model_posteriors[k]
    
    # Ensure normalization
    averaged_posterior = averaged_posterior / averaged_posterior.sum()
    
    return averaged_posterior, tuple(model_weights.tolist())


# =============================================================================
# DYNAMIC DECISION BOUNDARY (v3.0.0 - Improvement 5)
# =============================================================================

def _compute_dynamic_alpha(
    observation: ObservationVector,
    base_alpha: float = PRE_POLICY_THRESHOLD
) -> float:
    """
    Compute dynamic decision boundary α(t) based on context.
    
    v3.0.0 - Improvement 5: Decision boundary as a function, not a constant
    
    α(t) = f(convexity, leverage, carry)
    
    Meaning:
    - When convexity is extreme → lower α (trigger earlier)
    - When carry benefit is large → higher α (more conservative)
    
    This is dynamic regret minimization, not static.
    
    Args:
        observation: Current observation vector
        base_alpha: Base threshold (default PRE_POLICY_THRESHOLD)
        
    Returns:
        Dynamic threshold α(t)
    """
    alpha = base_alpha
    
    # Convexity adjustment
    # Higher convex loss → lower threshold (more aggressive)
    if np.isfinite(observation.convex_loss):
        # Typical convex loss range is 0.0001 to 0.002
        convex_normalized = min(observation.convex_loss / 0.001, 3.0)
        if convex_normalized > 1.0:
            # Reduce threshold when convexity is elevated
            alpha -= 0.05 * (convex_normalized - 1.0)
    
    # Volatility adjustment
    # Compressed volatility (pre-explosion) → lower threshold
    if np.isfinite(observation.vol_ratio):
        if observation.vol_ratio < 0.7:
            # Severe compression - be more aggressive
            alpha -= 0.10
        elif observation.vol_ratio < 0.85:
            # Moderate compression
            alpha -= 0.05
        elif observation.vol_ratio > 1.3:
            # Already expanding - might be too late, but also less "quiet danger"
            alpha += 0.05
    
    # Disagreement adjustment
    # High disagreement → more uncertainty → lower threshold
    if np.isfinite(observation.disagreement):
        if observation.disagreement > 0.5:
            alpha -= 0.08
        elif observation.disagreement > 0.35:
            alpha -= 0.04
    
    # Skew adjustment (v3.0.0)
    # Strong negative skew → lower threshold (danger signal)
    if np.isfinite(observation.skew):
        if observation.skew < -1.0:
            alpha -= 0.10
        elif observation.skew < -0.5:
            alpha -= 0.05
    
    # Rapid skew change → lower threshold
    if np.isfinite(observation.skew_momentum):
        if abs(observation.skew_momentum) > 0.5:
            alpha -= 0.07
    
    # Momentum adjustment
    # Rising disagreement → lower threshold
    if np.isfinite(observation.disagreement_momentum):
        if observation.disagreement_momentum > 0.05:
            alpha -= 0.05
    
    # Clamp to reasonable range
    alpha = max(0.35, min(alpha, 0.80))
    
    return alpha


# =============================================================================
# DOMINANCE CHECK (v3.0.0 - Improvement 6)
# =============================================================================

def _compute_expected_loss_delta(
    observation: ObservationVector,
    posterior: StatePosterior,
    log_returns: pd.Series,
    n_samples: int = 5000
) -> Tuple[float, float]:
    """
    Compute E[Loss_stay] - E[Loss_switch] for dominance check.
    
    v3.0.0 - Improvement 6: Explicit "do nothing" dominance check
    
    Before switching, compute:
        E[Loss_stay] - E[Loss_switch]
    
    If this difference is not strictly positive with margin, do nothing.
    
    Why:
    - Prevents switching on knife-edge probabilities
    - Adds robustness under noisy inference
    - Reduces false positives significantly
    
    Args:
        observation: Current observation
        posterior: Current state posterior
        log_returns: Historical log returns for Monte Carlo
        n_samples: Number of samples for expected loss computation
        
    Returns:
        Tuple of (loss_delta, required_margin)
    """
    # Generate forward samples
    samples_21d = _generate_posterior_samples(log_returns, 21, n_samples)
    
    if len(samples_21d) < MIN_POSTERIOR_SAMPLES:
        return (0.0, 0.0)
    
    # Expected loss if staying in JPY debt (funding loss when EUR/JPY rises)
    # Loss occurs when ΔX > 0 (EUR strengthens against JPY)
    positive_returns = samples_21d[samples_21d > 0]
    if len(positive_returns) > 0:
        # Convex loss: E[max(ΔX, 0)^p]
        expected_loss_stay = float(np.mean(positive_returns ** CONVEX_LOSS_EXPONENT))
    else:
        expected_loss_stay = 0.0
    
    # Expected loss if switching to EUR debt
    # Loss occurs when ΔX < 0 (JPY strengthens against EUR)
    # If we switch and JPY continues to strengthen, we "missed" the opportunity
    # But there's also switching cost (transaction, spread, etc.)
    switching_cost = 0.001  # Approximate transaction cost (10 bps)
    
    negative_returns = samples_21d[samples_21d < 0]
    if len(negative_returns) > 0:
        # Opportunity cost of switching too early
        expected_opportunity_cost = float(np.mean(np.abs(negative_returns)))
    else:
        expected_opportunity_cost = 0.0
    
    expected_loss_switch = switching_cost + 0.3 * expected_opportunity_cost
    
    # Scale by stress probability to weight by how likely stress is
    p_stress = posterior.p_pre_policy + posterior.p_policy
    
    # Loss delta: positive means staying is worse than switching
    loss_delta = expected_loss_stay * p_stress - expected_loss_switch * (1 - p_stress)
    
    # Required margin depends on posterior uncertainty
    # Higher uncertainty (flatter posterior) → higher required margin
    posterior_entropy = -np.sum([
        p * np.log(p + 1e-10) for p in posterior.probabilities
    ])
    max_entropy = np.log(LatentState.n_states())
    
    # Higher entropy → higher required margin
    uncertainty_ratio = posterior_entropy / max_entropy
    required_margin = 0.0001 * (1 + 2 * uncertainty_ratio)  # Base margin + uncertainty adjustment
    
    return (loss_delta, required_margin)


# =============================================================================
# INFERENCE ENGINE (v3.0.0 - Enhanced with Model Averaging)
# =============================================================================

def _run_inference(
    log_returns: pd.Series,
    lookback_days: int = 252,
    use_model_averaging: bool = True
) -> Tuple[ObservationVector, StatePosterior, List[ObservationVector]]:
    """
    Run latent state inference on the full history.
    
    v3.0.0: Enhanced with model averaging for meta-uncertainty.
    
    Args:
        log_returns: Historical log returns
        lookback_days: Number of days for inference window
        use_model_averaging: Whether to use Bayesian model averaging (v3.0.0)
        
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
    prev_skew = None  # v3.0.0: Track skew momentum
    
    for t in range(start_idx, n_obs):
        # Use data up to time t
        returns_to_t = log_returns.iloc[:t+1]
        
        obs = _construct_observation_vector(
            returns_to_t, 
            prev_disagreement=prev_disagreement,
            prev_skew=prev_skew,
            timestamp=str(log_returns.index[t])
        )
        observations.append(obs)
        obs_arrays.append(obs.to_array())
        
        # Update for next iteration
        prev_disagreement = obs.disagreement
        prev_skew = obs.skew  # v3.0.0
    
    if len(observations) == 0:
        raise ValueError("No observations constructed")
    
    print(f"[debt_allocator] Built {len(observations)} observations")
    
    # Convert to array for HMM
    historical_obs_array = np.array(obs_arrays)
    
    # Build transition matrix
    A = _build_transition_matrix()
    
    # Initial distribution (start in NORMAL with very high probability)
    # v3.0.0: Extended to 5 states
    pi = np.array([0.80, 0.12, 0.05, 0.02, 0.01])
    
    # Run inference
    print("[debt_allocator] Running forward inference...")
    
    model_weights = None
    if use_model_averaging:
        # v3.0.0: Use Bayesian model averaging
        print("[debt_allocator] Using model averaging (v3.0.0)...")
        current_posterior_probs, model_weights = _compute_model_averaged_posterior(
            [obs.to_array() for obs in observations],
            historical_obs_array,
            pi
        )
    else:
        # Standard forward algorithm
        posteriors = _forward_algorithm(
            [obs.to_array() for obs in observations],
            A,
            pi,
            historical_obs_array
        )
        current_posterior_probs = posteriors[-1]
    
    # Get current (final) observation and posterior
    current_obs = observations[-1]
    
    # Find dominant state
    dominant_state = LatentState(int(np.argmax(current_posterior_probs)))
    
    current_posterior = StatePosterior(
        probabilities=tuple(current_posterior_probs.tolist()),
        dominant_state=dominant_state,
        timestamp=current_obs.timestamp,
        model_weights=model_weights,  # v3.0.0
    )
    
    return current_obs, current_posterior, observations


# =============================================================================
# DECISION RULE (v3.0.0 - Enhanced with Dynamic α and Dominance Check)
# =============================================================================

def _compute_decision_signature(
    observation: ObservationVector,
    posterior: StatePosterior,
    triggered: bool,
    effective_date: Optional[str],
    dynamic_alpha: float = PRE_POLICY_THRESHOLD,
    expected_loss_delta: Optional[float] = None
) -> str:
    """
    Compute cryptographic signature for decision audit trail.
    
    v3.0.0: Includes dynamic_alpha and expected_loss_delta in signature.
    """
    payload = {
        "observation": observation.to_dict(),
        "posterior": posterior.to_dict(),
        "triggered": triggered,
        "effective_date": effective_date,
        "dynamic_alpha": dynamic_alpha,
        "expected_loss_delta": expected_loss_delta,
        "signature_version": "3.0.0",
    }
    
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()


def _make_decision(
    observation: ObservationVector,
    posterior: StatePosterior,
    log_returns: pd.Series,
    threshold: float = PRE_POLICY_THRESHOLD,
    use_dynamic_alpha: bool = True,
    use_dominance_check: bool = True
) -> DebtSwitchDecision:
    """
    Make the debt switch decision based on latent state inference.
    
    v3.0.0 Enhancements:
    - Dynamic decision boundary α(t) (Improvement 5)
    - Explicit dominance check (Improvement 6)
    - PRE_POLICY_A/B discrimination
    
    Trigger logic:
    1. Compute dynamic α(t) based on observation context
    2. Check if P(PRE_POLICY_B) > α(t) OR P(POLICY) dominant
    3. Verify dominance: E[Loss_stay] - E[Loss_switch] > margin
    
    Args:
        observation: Current observation vector
        posterior: Current state posterior
        log_returns: Historical log returns for dominance check
        threshold: Base decision threshold α
        use_dynamic_alpha: Whether to use dynamic threshold (v3.0.0)
        use_dominance_check: Whether to use dominance check (v3.0.0)
        
    Returns:
        DebtSwitchDecision
    """
    # v3.0.0 - Improvement 5: Dynamic threshold
    if use_dynamic_alpha:
        dynamic_alpha = _compute_dynamic_alpha(observation, threshold)
    else:
        dynamic_alpha = threshold
    
    # Initial trigger check
    # PRE_POLICY_B is the "unstable compression" state - most regret happens here
    # We trigger on PRE_POLICY_B crossing threshold, not PRE_POLICY_A
    prob_pre_policy_b = posterior.p_pre_policy_b
    prob_policy = posterior.p_policy
    prob_stress = posterior.p_pre_policy + posterior.p_policy  # Combined stress
    
    # Trigger conditions
    triggered = False
    decision_basis = ""
    
    # Primary trigger: PRE_POLICY_B exceeds threshold
    if prob_pre_policy_b > dynamic_alpha:
        triggered = True
        decision_basis = f"PRE_POLICY_B threshold exceeded (P={prob_pre_policy_b:.2%} > α={dynamic_alpha:.0%})"
    
    # Belt and suspenders: POLICY state is dominant
    elif posterior.dominant_state == LatentState.POLICY:
        triggered = True
        decision_basis = f"POLICY state dominant (P={prob_policy:.2%})"
    
    # Combined stress check (PRE_POLICY_A + B + POLICY)
    elif prob_stress > 0.75:
        triggered = True
        decision_basis = f"Combined stress high (P_stress={prob_stress:.2%} > 75%)"
    
    # v3.0.0 - Improvement 6: Dominance check
    expected_loss_delta = None
    dominance_margin = 0.0
    
    if triggered and use_dominance_check:
        loss_delta, required_margin = _compute_expected_loss_delta(
            observation, posterior, log_returns
        )
        expected_loss_delta = loss_delta
        dominance_margin = required_margin
        
        # Verify dominance: switching should be strictly better with margin
        if loss_delta < required_margin:
            # Loss delta not positive enough - do not trigger
            triggered = False
            decision_basis = (
                f"Dominance check failed: "
                f"E[Loss_stay]-E[Loss_switch]={loss_delta:.6f} < margin={required_margin:.6f}"
            )
    
    # If not triggered, provide status
    if not triggered and not decision_basis:
        decision_basis = (
            f"Below threshold: P(PRE_POLICY_B)={prob_pre_policy_b:.2%} ≤ α={dynamic_alpha:.0%}, "
            f"P(stress)={prob_stress:.2%}"
        )
    
    effective_date = observation.timestamp.split('T')[0] if triggered else None
    
    signature = _compute_decision_signature(
        observation, posterior, triggered, effective_date, 
        dynamic_alpha, expected_loss_delta
    )
    
    return DebtSwitchDecision(
        triggered=triggered,
        effective_date=effective_date,
        observation=observation,
        state_posterior=posterior,
        decision_basis=decision_basis,
        signature=signature,
        dynamic_alpha=dynamic_alpha,
        expected_loss_delta=expected_loss_delta,
        dominance_margin=dominance_margin,
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
            skew=obs_data.get('skew', 0.0),  # v3.0.0
            skew_momentum=obs_data.get('skew_momentum', 0.0),  # v3.0.0
            timestamp=obs_data['timestamp'],
        )
        
        post_data = data['state_posterior']
        probs = post_data['probabilities']
        
        # Handle both v2 (4 states) and v3 (5 states) formats
        if 'PRE_POLICY_A' in probs:
            # v3.0.0 format
            posterior = StatePosterior(
                probabilities=(
                    probs['NORMAL'],
                    probs['COMPRESSED'],
                    probs['PRE_POLICY_A'],
                    probs['PRE_POLICY_B'],
                    probs['POLICY'],
                ),
                dominant_state=LatentState[post_data['dominant_state']],
                timestamp=post_data['timestamp'],
                model_weights=tuple(post_data['model_weights']) if 'model_weights' in post_data else None,
            )
        else:
            # v2.0.0 format - convert to v3
            pre_policy_total = probs.get('PRE_POLICY', 0.0)
            posterior = StatePosterior(
                probabilities=(
                    probs['NORMAL'],
                    probs['COMPRESSED'],
                    pre_policy_total * 0.4,  # Approximate split: A gets 40%
                    pre_policy_total * 0.6,  # B gets 60%
                    probs['POLICY'],
                ),
                dominant_state=LatentState.NORMAL,  # Safe default
                timestamp=post_data['timestamp'],
                model_weights=None,
            )
        
        return DebtSwitchDecision(
            triggered=data['triggered'],
            effective_date=data['effective_date'],
            observation=observation,
            state_posterior=posterior,
            decision_basis=data['decision_basis'],
            signature=data['signature'],
            dynamic_alpha=data.get('dynamic_alpha', PRE_POLICY_THRESHOLD),  # v3.0.0
            expected_loss_delta=data.get('expected_loss_delta'),  # v3.0.0
            dominance_margin=data.get('dominance_margin', 0.0),  # v3.0.0
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
# HIGH-LEVEL API (v3.0.0 - Enhanced)
# =============================================================================

def run_debt_allocation_engine(
    data_path: str = EURJPY_DATA_FILE,
    persistence_path: str = DECISION_PERSISTENCE_FILE,
    force_reevaluate: bool = False,
    force_refresh_data: bool = True,
    use_model_averaging: bool = True,
    use_dynamic_alpha: bool = True,
    use_dominance_check: bool = True,
) -> Optional[DebtSwitchDecision]:
    """
    Run the research-grade debt allocation engine.
    
    v3.0.0 Enhancements:
    - Model averaging for meta-uncertainty (Improvement 4)
    - Dynamic decision boundary (Improvement 5)
    - Dominance check (Improvement 6)
    
    Args:
        data_path: Path to EURJPY data
        persistence_path: Path to decision persistence file
        force_reevaluate: If True, ignore persisted decision
        force_refresh_data: If True, download fresh data
        use_model_averaging: Use Bayesian model averaging (v3.0.0)
        use_dynamic_alpha: Use dynamic threshold α(t) (v3.0.0)
        use_dominance_check: Use expected loss dominance check (v3.0.0)
        
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
        observation, posterior, _ = _run_inference(
            log_returns, 
            use_model_averaging=use_model_averaging
        )
    except Exception as e:
        print(f"[debt_allocator] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # MAKE DECISION (v3.0.0: with dynamic alpha and dominance check)
    decision = _make_decision(
        observation, 
        posterior, 
        log_returns,
        use_dynamic_alpha=use_dynamic_alpha,
        use_dominance_check=use_dominance_check,
    )
    
    # PERSIST IF TRIGGERED
    if decision.triggered:
        _persist_decision(decision, persistence_path)
        print(f"[debt_allocator] DECISION TRIGGERED - persisted to {persistence_path}")
    
    return decision


# =============================================================================
# PRESENTATION (RICH OUTPUT) - v3.0.0 Enhanced
# =============================================================================

def _get_status_display(decision: DebtSwitchDecision) -> Tuple[str, str, str]:
    """Get status emoji, text, and color for display."""
    if decision.triggered:
        if decision.state_posterior.dominant_state == LatentState.POLICY:
            return "🔴", "SWITCH TRIGGERED (POLICY)", "red"
        elif decision.state_posterior.dominant_state == LatentState.PRE_POLICY_B:
            return "🔴", "SWITCH TRIGGERED (PRE-POLICY-B)", "red"
        else:
            return "🔴", "SWITCH TRIGGERED", "red"
    
    dominant = decision.state_posterior.dominant_state
    if dominant == LatentState.PRE_POLICY_B:
        return "🟡", "MONITORING (PRE-POLICY-B: UNSTABLE)", "yellow"
    elif dominant == LatentState.PRE_POLICY_A:
        return "🟡", "MONITORING (PRE-POLICY-A: SILENT STRESS)", "yellow"
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
    console.print("[bold cyan]DEBT ALLOCATION — EURJPY (RESEARCH v3.0.0)[/bold cyan]", justify="center")
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
            f"[bold]Effective Date[/bold]    : [bold white]{decision.effective_date}[/bold white]\n"
            f"[bold]Dynamic α(t)[/bold]      : {decision.dynamic_alpha:.2%}"
        )
    else:
        status_content = (
            f"[bold]Switch Status[/bold]     : {status_emoji} [{status_color}]{status_text}[/{status_color}]\n"
            f"[bold]Dynamic α(t)[/bold]      : {decision.dynamic_alpha:.2%}"
        )
    
    console.print(Panel(
        status_content,
        title=f"[bold {status_color}]Decision Status[/bold {status_color}]",
        border_style=status_color,
        padding=(1, 2),
    ))
    
    console.print()
    
    # Latent State Probabilities (v3.0.0: 5 states)
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
        ("PRE_POLICY_A", post.p_pre_policy_a),
        ("PRE_POLICY_B", post.p_pre_policy_b),
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
        elif state_name in ("PRE_POLICY_A", "PRE_POLICY_B", "POLICY") and prob > 0.2:
            prob_str = f"[yellow]{prob:.1%}[/yellow]"
            bar_str = f"[yellow]{bar}[/yellow]"
        else:
            prob_str = f"{prob:.1%}"
            bar_str = f"[dim]{bar}[/dim]"
        
        state_table.add_row(state_name, prob_str, bar_str)
    
    console.print(state_table)
    console.print()
    
    # Model weights (v3.0.0)
    if post.model_weights is not None:
        console.print("[bold cyan]Model Weights (Meta-Uncertainty):[/bold cyan]")
        model_names = ["baseline", "sticky", "volatile", "vol_driven", "skew_driven"]
        for name, weight in zip(model_names, post.model_weights):
            bar_len = int(weight * 20)
            bar = "▓" * bar_len + "░" * (20 - bar_len)
            console.print(f"  {name:12s}: {weight:.1%} {bar}")
        console.print()
    
    # Observation Metrics (v3.0.0: including skew)
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
    
    # Skew metrics (v3.0.0)
    if np.isfinite(obs.skew):
        skew_str = f"{obs.skew:+.3f}"
        if obs.skew < -0.5:
            skew_str = f"[yellow]{skew_str}[/yellow] (negative)"
        elif obs.skew > 0.5:
            skew_str = f"[yellow]{skew_str}[/yellow] (positive)"
    else:
        skew_str = "[dim]N/A[/dim]"
    
    if np.isfinite(obs.skew_momentum):
        skew_mom_str = f"{obs.skew_momentum:+.4f}"
        if abs(obs.skew_momentum) > 0.3:
            skew_mom_str = f"[yellow]{skew_mom_str}[/yellow] (rapid change)"
    else:
        skew_mom_str = "[dim]N/A[/dim]"
    
    metrics_table.add_row("Convex Loss C(t)", convex_str)
    metrics_table.add_row("Tail Mass P(ΔX > 0)", tail_str)
    metrics_table.add_row("Disagreement D(t)", disag_str)
    metrics_table.add_row("Momentum dD(t)", mom_str)
    metrics_table.add_row("Vol Ratio V(t)", vol_str)
    metrics_table.add_row("Skew(t)", skew_str)
    metrics_table.add_row("Skew Momentum dSkew(t)", skew_mom_str)
    
    console.print(metrics_table)
    console.print()
    
    # Dominance check info (v3.0.0)
    if decision.expected_loss_delta is not None:
        console.print("[bold cyan]Dominance Check (v3.0.0):[/bold cyan]")
        delta_str = f"{decision.expected_loss_delta:.6f}"
        margin_str = f"{decision.dominance_margin:.6f}"
        if decision.expected_loss_delta > decision.dominance_margin:
            console.print(f"  E[Loss_stay] - E[Loss_switch]: [green]{delta_str}[/green]")
            console.print(f"  Required margin: {margin_str}")
            console.print("  [green]✓ Switching dominates[/green]")
        else:
            console.print(f"  E[Loss_stay] - E[Loss_switch]: [yellow]{delta_str}[/yellow]")
            console.print(f"  Required margin: {margin_str}")
            console.print("  [yellow]✗ Switching does not dominate[/yellow]")
        console.print()
    
    # Decision Basis
    console.print(f"[bold]Decision Basis:[/bold] {decision.decision_basis}")
    console.print()
    
    # Signature
    console.print(f"[dim]Signature: {decision.signature[:16]}...[/dim]")
    console.print()


# =============================================================================
# CLI ENTRY POINT (v3.0.0 - Enhanced)
# =============================================================================

def main():
    """Main entry point for `make debt` command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Research-Grade FX Debt Allocation Engine (EURJPY) v3.0.0"
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
    # v3.0.0 options
    parser.add_argument(
        '--no-model-averaging',
        action='store_true',
        help='Disable Bayesian model averaging (v3.0.0)'
    )
    parser.add_argument(
        '--no-dynamic-alpha',
        action='store_true',
        help='Use fixed threshold instead of dynamic α(t) (v3.0.0)'
    )
    parser.add_argument(
        '--no-dominance-check',
        action='store_true',
        help='Disable expected loss dominance check (v3.0.0)'
    )
    parser.add_argument(
        '--v2-mode',
        action='store_true',
        help='Run in v2.0.0 compatibility mode (disables all v3.0.0 features)'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Determine v3.0.0 feature flags
    if args.v2_mode:
        use_model_averaging = False
        use_dynamic_alpha = False
        use_dominance_check = False
    else:
        use_model_averaging = not args.no_model_averaging
        use_dynamic_alpha = not args.no_dynamic_alpha
        use_dominance_check = not args.no_dominance_check
    
    # Show header
    if not args.json:
        console.print()
        console.print(Panel(
            "[bold cyan]Research-Grade FX Debt Allocation Engine[/bold cyan]\n"
            "[dim]EURJPY Latent Policy-Stress Inference[/dim]\n"
            "[bold green]Version 3.0.0[/bold green]",
            border_style="cyan",
        ))
        console.print()
        console.print(f"[dim]Cache directory: {DEBT_CACHE_DIR}[/dim]")
        console.print(f"[dim]Data file: {args.data_path}[/dim]")
        
        # Show v3.0.0 feature status
        console.print()
        console.print("[bold]v3.0.0 Features:[/bold]")
        console.print(f"  Model Averaging:    {'[green]ON[/green]' if use_model_averaging else '[yellow]OFF[/yellow]'}")
        console.print(f"  Dynamic α(t):       {'[green]ON[/green]' if use_dynamic_alpha else '[yellow]OFF[/yellow]'}")
        console.print(f"  Dominance Check:    {'[green]ON[/green]' if use_dominance_check else '[yellow]OFF[/yellow]'}")
        console.print()
    
    # Run engine
    decision = run_debt_allocation_engine(
        data_path=args.data_path,
        force_reevaluate=args.force,
        force_refresh_data=not args.no_refresh,
        use_model_averaging=use_model_averaging,
        use_dynamic_alpha=use_dynamic_alpha,
        use_dominance_check=use_dominance_check,
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
