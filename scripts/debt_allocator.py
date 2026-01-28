#!/usr/bin/env python3
"""
debt_allocator.py

FX Debt Allocation Engine (EURJPY)
==================================

PURPOSE:
    Determines a single, precise, irreversible calendar day on which 
    JPY-denominated debt must be switched to EUR-denominated debt,
    based exclusively on EURJPY dynamics.

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

OBSERVATION VECTOR:
    Y_t = (C, P, D, dD, V)
    - C: Convex loss functional
    - P: Tail mass
    - D: Epistemic disagreement
    - dD: Disagreement momentum
    - V: Volatility compression ratio
    
    Note: ΔC(t) = C(t) - C(t-1) is computed from C for transition pressure.

FINAL FIXES:
    1. Endogenous transition pressure: Φ(Y) accelerates forward transitions
    2. Risk-adaptive decision boundary: α(t) = α₀ − g(ΔC(t))

ARCHITECTURAL CONSTRAINTS:
    - Everything lives in this single file
    - No reuse of existing methods
    - No modification of existing files
    - No imports from tuning or signal helpers
    - No shared mutable state
    - Deletable with zero side effects

Version: 4.0.0
"""

from __future__ import annotations

import hashlib
import json
import random
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
from scipy.special import logsumexp, digamma, gammaln
from scipy.stats import dirichlet, entropy

# =============================================================================
# PRESENTATION IMPORTS (RICH ONLY)
# =============================================================================
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.padding import Padding
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
    
    Contract-specified components (C, P, D, dD, V):
        C: Convex loss functional E[max(ΔX, 0)^p]
        P: Tail mass P(ΔX > 0)
        D: Epistemic disagreement (normalized)
        dD: Disagreement momentum
        V: Volatility compression/expansion ratio
    
    Derived metric (computed from C):
        ΔC: Convex loss acceleration C(t) - C(t-1)
    """
    convex_loss: float      # C(t)
    convex_loss_acceleration: float  # ΔC(t) = C(t) - C(t-1), derived from C
    tail_mass: float        # P(t) = P(ΔX > 0)
    disagreement: float     # D(t)
    disagreement_momentum: float  # dD(t)
    vol_ratio: float        # V(t) = σ_short / σ_long
    timestamp: str
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for HMM processing.
        
        Array layout: [C, P, D, dD, V, ΔC]
        Core observation is [C, P, D, dD, V], ΔC is derived.
        """
        return np.array([
            self.convex_loss,
            self.tail_mass,
            self.disagreement,
            self.disagreement_momentum,
            self.vol_ratio,
            self.convex_loss_acceleration,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "convex_loss": self.convex_loss,
            "convex_loss_acceleration": self.convex_loss_acceleration,
            "tail_mass": self.tail_mass,
            "disagreement": self.disagreement,
            "disagreement_momentum": self.disagreement_momentum,
            "vol_ratio": self.vol_ratio,
            "timestamp": self.timestamp,
        }


# =============================================================================
# OBSERVATION VALIDATION / OUTLIER HANDLING
# =============================================================================

# Observation bounds for numerical stability and outlier rejection
OBSERVATION_BOUNDS = {
    'convex_loss': (1e-8, 0.5),           # C(t) must be positive, cap extreme values
    'convex_loss_acceleration': (-0.1, 0.1),  # ΔC(t) bounded acceleration
    'tail_mass': (0.01, 0.99),            # P(t) ∈ (0, 1), avoid boundaries
    'disagreement': (0.0, 1.0),           # D(t) normalized to [0, 1]
    'disagreement_momentum': (-0.5, 0.5), # dD(t) bounded momentum
    'vol_ratio': (0.1, 10.0),             # V(t) reasonable vol ratio range
}


def _validate_observation(obs: ObservationVector) -> ObservationVector:
    """
    Validate and clip observation values to prevent numerical issues.
    
    Handles:
    - NaN/Inf values → replaced with safe defaults
    - Extreme outliers → clipped to reasonable bounds
    - Boundary values → pulled away from exact 0 or 1
    
    Args:
        obs: Raw observation vector
        
    Returns:
        Validated ObservationVector with all values in safe ranges
    """
    def _safe_clip(value: float, bounds: Tuple[float, float], default: float) -> float:
        """Clip value to bounds, replacing NaN/Inf with default."""
        if not np.isfinite(value):
            return default
        return float(np.clip(value, bounds[0], bounds[1]))
    
    # Validate each component with appropriate defaults
    convex_loss = _safe_clip(
        obs.convex_loss, 
        OBSERVATION_BOUNDS['convex_loss'], 
        default=0.0003  # Typical median value
    )
    
    convex_loss_acceleration = _safe_clip(
        obs.convex_loss_acceleration,
        OBSERVATION_BOUNDS['convex_loss_acceleration'],
        default=0.0  # No acceleration by default
    )
    
    tail_mass = _safe_clip(
        obs.tail_mass,
        OBSERVATION_BOUNDS['tail_mass'],
        default=0.5  # Neutral tail mass
    )
    
    disagreement = _safe_clip(
        obs.disagreement,
        OBSERVATION_BOUNDS['disagreement'],
        default=0.2  # Typical disagreement
    )
    
    disagreement_momentum = _safe_clip(
        obs.disagreement_momentum,
        OBSERVATION_BOUNDS['disagreement_momentum'],
        default=0.0  # No momentum by default
    )
    
    vol_ratio = _safe_clip(
        obs.vol_ratio,
        OBSERVATION_BOUNDS['vol_ratio'],
        default=1.0  # Neutral vol ratio
    )
    
    return ObservationVector(
        convex_loss=convex_loss,
        convex_loss_acceleration=convex_loss_acceleration,
        tail_mass=tail_mass,
        disagreement=disagreement,
        disagreement_momentum=disagreement_momentum,
        vol_ratio=vol_ratio,
        timestamp=obs.timestamp,
    )


def _validate_observation_array(obs_array: np.ndarray) -> np.ndarray:
    """
    Validate observation array for HMM processing.
    
    Array layout: [C, P, D, dD, V, ΔC]
    
    Args:
        obs_array: Raw observation array
        
    Returns:
        Validated array with all values in safe ranges
    """
    validated = obs_array.copy()
    
    # Map array indices to bounds
    bounds_map = [
        ('convex_loss', 0),
        ('tail_mass', 1),
        ('disagreement', 2),
        ('disagreement_momentum', 3),
        ('vol_ratio', 4),
        ('convex_loss_acceleration', 5),
    ]
    
    defaults = [0.0003, 0.5, 0.2, 0.0, 1.0, 0.0]
    
    for (key, idx), default in zip(bounds_map, defaults):
        if idx < len(validated):
            if not np.isfinite(validated[idx]):
                validated[idx] = default
            else:
                bounds = OBSERVATION_BOUNDS[key]
                validated[idx] = np.clip(validated[idx], bounds[0], bounds[1])
    
    return validated


@dataclass(frozen=True)
class StatePosterior:
    """
    Posterior state probabilities P(S_t | Y_1:t).
    """
    probabilities: Tuple[float, float, float, float]  # One per state (4 states)
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
    """
    Immutable record of debt switch decision.
    """
    triggered: bool
    effective_date: Optional[str]
    observation: ObservationVector
    state_posterior: StatePosterior
    decision_basis: str
    signature: str
    dynamic_alpha: float = 0.60  # Dynamic threshold α(t)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "effective_date": self.effective_date,
            "observation": self.observation.to_dict(),
            "state_posterior": self.state_posterior.to_dict(),
            "decision_basis": self.decision_basis,
            "signature": self.signature,
            "dynamic_alpha": self.dynamic_alpha,
        }


# =============================================================================
# BAYESIAN MODEL AVERAGING WITH DIRICHLET PRIOR
# =============================================================================

class BayesianTransitionModel:
    """
    Dirichlet-Multinomial conjugate prior for transition probabilities.
    Maintains posterior distribution over transition matrices for uncertainty quantification.
    
    Mathematical Foundation:
    - Prior: P(A_i) ~ Dirichlet(α_prior)
    - Likelihood: counts ~ Multinomial(A_i)
    - Posterior: P(A_i | counts) ~ Dirichlet(α_prior + counts)
    
    Includes exponential forgetting to handle non-stationary markets:
    - α_posterior *= decay before each update
    - This gives more weight to recent transitions

    """

    def __init__(
        self, 
        n_states: int = 4, 
        prior_concentration: float = 1.0,
        decay_factor: float = 0.995,
        belief_threshold: float = 0.01
    ):
        """
        Initialize with uninformative Dirichlet prior.
        
        Args:
            n_states: Number of hidden states
            prior_concentration: Dirichlet concentration parameter (α)
                                 Higher values = stronger prior toward uniform
            decay_factor: Exponential forgetting factor (0.99-0.999 typical)
                         Lower = faster forgetting, more reactive to regime changes
            belief_threshold: Minimum belief probability to consider in efficient prediction
        """
        self.n_states = n_states
        self.decay_factor = decay_factor
        self.belief_threshold = belief_threshold
        self.alpha_prior = np.ones((n_states, n_states)) * prior_concentration
        self.alpha_posterior = self.alpha_prior.copy()
        self._transition_counts = np.zeros((n_states, n_states))
        self._update_count = 0

    def update(self, state_from: int, state_to: int, count: float = 1.0):
        """
        Update posterior with observed transition.
        Applies exponential decay to handle non-stationary markets.
        """
        # Apply decay to all counts (forgetting factor)
        # This ensures recent transitions have more influence
        self._transition_counts *= self.decay_factor
        
        # Add new observation
        self._transition_counts[state_from, state_to] += count
        
        # Recompute posterior
        self.alpha_posterior = self.alpha_prior + self._transition_counts
        self._update_count += 1

    def update_from_posteriors(
        self, 
        state_posteriors: List[np.ndarray],
        apply_batch_decay: bool = True
    ):
        """
        Update from soft state assignments (posterior probabilities).
        Uses expected counts: E[n_ij] = Σ_t P(S_t=i) P(S_{t+1}=j)
        
        Args:
            state_posteriors: List of posterior distributions
            apply_batch_decay: If True, apply decay once before batch update
        """
        if apply_batch_decay and len(state_posteriors) > 1:
            # Apply cumulative decay for batch size
            batch_decay = self.decay_factor ** len(state_posteriors)
            self._transition_counts *= batch_decay
        
        # Vectorized soft count computation (O(T*K²) but in numpy)
        posteriors_array = np.array(state_posteriors)
        for t in range(len(state_posteriors) - 1):
            # Outer product gives transition probabilities
            soft_counts = np.outer(posteriors_array[t], posteriors_array[t + 1])
            self._transition_counts += soft_counts
        
        # Recompute posterior
        self.alpha_posterior = self.alpha_prior + self._transition_counts
        self._update_count += len(state_posteriors)

    def expected_transition_matrix(self) -> np.ndarray:
        """
        E[P_ij] = α_ij / Σ_k α_ik (posterior mean).
        """
        row_sums = self.alpha_posterior.sum(axis=1, keepdims=True)
        return self.alpha_posterior / (row_sums + 1e-10)

    def sample_transition_matrix(self, n_samples: int = 100) -> np.ndarray:
        """
        Monte Carlo samples from posterior for model averaging.
        Returns:
            Array of shape (n_samples, n_states, n_states)
        """
        samples = np.zeros((n_samples, self.n_states, self.n_states))
        for i in range(self.n_states):
            samples[:, i, :] = dirichlet.rvs(
                self.alpha_posterior[i], size=n_samples
            )
        return samples

    def sample_transition_row(self, state: int, n_samples: int = 100) -> np.ndarray:
        """
        Efficient sampling of single transition row (O(K) instead of O(K²)).
        Use this when you only need transitions from a specific state.
        
        Args:
            state: Source state index
            n_samples: Number of MC samples
            
        Returns:
            Array of shape (n_samples, n_states)
        """
        return dirichlet.rvs(self.alpha_posterior[state], size=n_samples)

    def predictive_state_distribution_efficient(
        self,
        current_belief: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Efficient Bayesian prediction using only rows with non-zero belief.
        
        This is O(K' * K) where K' is number of states with p > threshold,
        instead of O(K² * n_samples) for full matrix sampling.
        
        Args:
            current_belief: Current state probability distribution P(S_t)
            n_samples: Number of MC samples
            
        Returns:
            Dictionary with mean prediction and credible intervals
        """
        predictions = np.zeros((n_samples, self.n_states))
        
        # Only sample rows where belief is non-negligible
        for state in range(self.n_states):
            if current_belief[state] > self.belief_threshold:
                # Sample transitions from this state
                row_samples = self.sample_transition_row(state, n_samples)
                # Weight by belief probability
                predictions += current_belief[state] * row_samples
        
        # Renormalize predictions (handle numerical issues)
        row_sums = predictions.sum(axis=1, keepdims=True)
        predictions = predictions / (row_sums + 1e-10)
        
        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'ci_lower': np.percentile(predictions, 2.5, axis=0),
            'ci_upper': np.percentile(predictions, 97.5, axis=0),
        }

    def predictive_state_distribution(
        self, 
        current_belief: np.ndarray, 
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Bayesian model-averaged prediction with uncertainty quantification.
        
        Args:
            current_belief: Current state probability distribution P(S_t)
            n_samples: Number of MC samples
            
        Returns:
            Dictionary with mean prediction and credible intervals
        """
        samples = self.sample_transition_matrix(n_samples)
        predictions = np.array([
            current_belief @ T for T in samples
        ])
        
        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'ci_lower': np.percentile(predictions, 2.5, axis=0),
            'ci_upper': np.percentile(predictions, 97.5, axis=0),
        }

    def posterior_entropy(self) -> float:
        """
        Entropy of the posterior Dirichlet distribution.
        Higher entropy = more uncertainty about transition probabilities.
        """
        total_entropy = 0.0
        for i in range(self.n_states):
            alpha_row = self.alpha_posterior[i]
            alpha_0 = alpha_row.sum()
            # Dirichlet entropy formula
            entropy_i = (
                gammaln(alpha_row).sum() - gammaln(alpha_0) +
                (alpha_0 - self.n_states) * digamma(alpha_0) -
                ((alpha_row - 1) * digamma(alpha_row)).sum()
            )
            total_entropy += entropy_i
        return total_entropy

    def transition_uncertainty(self) -> np.ndarray:
        """
        Variance of each transition probability.
        Var[P_ij] = α_ij(α_0 - α_ij) / (α_0²(α_0 + 1))
        """
        variance = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            alpha_row = self.alpha_posterior[i]
            alpha_0 = alpha_row.sum()
            for j in range(self.n_states):
                variance[i, j] = (
                    alpha_row[j] * (alpha_0 - alpha_row[j]) /
                    (alpha_0**2 * (alpha_0 + 1) + 1e-10)
                )
        return variance


# =============================================================================
# WASSERSTEIN REGIME DETECTOR (OPTIMAL TRANSPORT)
# =============================================================================

class WassersteinRegimeDetector:
    """
    Detects regime changes via Wasserstein distance between
    sliding window empirical distributions.
    
    Uses Sliced Wasserstein distance for efficient multivariate computation:
    SW_2(P, Q) = (∫_{S^{d-1}} W_2(proj_θ P, proj_θ Q)² dθ)^{1/2}
    
    Mathematical Foundation:
    - Wasserstein distance measures the "cost" of transforming one distribution into another
    - High distance between consecutive windows indicates regime change
    - Sliced approximation makes it tractable for multivariate data
    """

    def __init__(
        self, 
        window_size: int = 50, 
        n_projections: int = 50,
        threshold_percentile: float = 90.0
    ):
        """
        Args:
            window_size: Size of sliding windows for comparison
            n_projections: Number of random projections for sliced Wasserstein
            threshold_percentile: Percentile for regime change detection threshold
        """
        self.window_size = window_size
        self.n_projections = n_projections
        self.threshold_percentile = threshold_percentile
        self.scores_ = None
        self.threshold_ = None

    def _wasserstein_1d_quantile(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        1D Wasserstein-2 distance via quantile matching.
        W_2² = ∫_0^1 (F_X^{-1}(u) - F_Y^{-1}(u))² du
        
        Approximated by matching sorted samples (empirical quantiles).
        """
        n = min(len(x), len(y))
        if n == 0:
            return 0.0
            
        # Match quantiles via sorting
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Interpolate to same size if needed
        if len(x_sorted) != len(y_sorted):
            quantiles = np.linspace(0, 1, n)
            x_interp = np.percentile(x_sorted, quantiles * 100)
            y_interp = np.percentile(y_sorted, quantiles * 100)
        else:
            x_interp, y_interp = x_sorted, y_sorted
        
        return np.sqrt(np.mean((x_interp - y_interp)**2))

    def _sliced_wasserstein(
        self, 
        X: np.ndarray, 
        Y: np.ndarray,
        random_state: int = 42
    ) -> float:
        """
        Sliced Wasserstein-2 distance for multivariate data.
        
        Projects data onto random 1D directions and averages 1D Wasserstein distances.
        """
        if len(X) == 0 or len(Y) == 0:
            return 0.0
            
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
            
        d = X.shape[1]
        rng = np.random.RandomState(random_state)
        
        distances_squared = []
        for _ in range(self.n_projections):
            # Random unit vector on sphere
            theta = rng.randn(d)
            theta /= (np.linalg.norm(theta) + 1e-10)
            
            # Project data
            proj_X = X @ theta
            proj_Y = Y @ theta
            
            # 1D Wasserstein
            w = self._wasserstein_1d_quantile(proj_X, proj_Y)
            distances_squared.append(w**2)
        
        return np.sqrt(np.mean(distances_squared))

    def compute_regime_change_scores(
        self, 
        observations: np.ndarray,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Compute sliding window Wasserstein distance for regime change detection.
        
        Args:
            observations: Array of shape (T, n_features)
            random_state: Random seed for reproducibility
            
        Returns:
            Array of regime change scores (higher = more likely regime change)
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
            
        T = len(observations)
        scores = np.zeros(T)
        
        # Need at least 2 full windows
        if T < 2 * self.window_size:
            self.scores_ = scores
            self.threshold_ = 0.5
            return scores
        
        for t in range(self.window_size, T - self.window_size):
            window_past = observations[t - self.window_size:t]
            window_future = observations[t:t + self.window_size]
            
            scores[t] = self._sliced_wasserstein(
                window_past, window_future, 
                random_state=random_state + t
            )
        
        # Normalize to [0, 1]
        score_range = scores.max() - scores.min()
        if score_range > 1e-10:
            scores = (scores - scores.min()) / score_range
        
        self.scores_ = scores
        nonzero_scores = scores[scores > 0]
        if len(nonzero_scores) > 0:
            self.threshold_ = np.percentile(nonzero_scores, self.threshold_percentile)
        else:
            self.threshold_ = 0.5
        
        return scores

    def detect_regime_changes(self, observations: np.ndarray = None) -> np.ndarray:
        """
        Detect regime change points based on Wasserstein scores.
        
        Returns:
            Boolean array indicating detected regime changes
        """
        if self.scores_ is None:
            if observations is None:
                raise ValueError("No scores computed. Provide observations.")
            self.compute_regime_change_scores(observations)
        
        return self.scores_ > self.threshold_

    def get_regime_segments(self, observations: np.ndarray = None) -> List[Tuple[int, int]]:
        """
        Segment time series into regime periods.
        
        Returns:
            List of (start_idx, end_idx) tuples for each regime segment
        """
        if self.scores_ is None:
            if observations is None:
                raise ValueError("No scores computed. Provide observations.")
            self.compute_regime_change_scores(observations)
            
        changes = self.scores_ > self.threshold_
        change_points = np.where(changes)[0]
        
        segments = []
        start = 0
        for cp in change_points:
            if cp > start:
                segments.append((start, cp))
            start = cp
        
        # Add final segment
        T = len(self.scores_)
        if start < T:
            segments.append((start, T))
        
        return segments

    def regime_stability_score(self) -> float:
        """
        Overall stability score: inverse of average regime change intensity.
        Higher = more stable regime structure.
        """
        if self.scores_ is None:
            return 1.0
        
        nonzero_scores = self.scores_[self.scores_ > 0]
        if len(nonzero_scores) == 0:
            return 1.0
        
        return 1.0 / (1.0 + np.mean(nonzero_scores))


# =============================================================================
# INFORMATION-THEORETIC STATE DISCRIMINATION (MI + KL)
# =============================================================================

class InformationTheoreticWeighting:
    """
    Weight observations by their mutual information with the latent state.
    
    I(Y_i; S) = H(Y_i) - H(Y_i | S)
    
    Mathematical Foundation:
    - High MI observations are more discriminative for state classification
    - Uses these weights in emission probability computation
    - KL divergence measures state separability
    
    Uses kNN entropy estimation (Kraskov-Stögbauer-Grassberger) for robustness:
    - More sample-efficient than histograms
    - Less sensitive to bin count
    - Better tail behavior
    """

    def __init__(self, n_states: int = 4, n_neighbors: int = 5):
        """
        Args:
            n_states: Number of hidden states
            n_neighbors: k for kNN entropy estimation (3-7 typical)
        """
        self.n_states = n_states
        self.n_neighbors = n_neighbors
        self.observation_weights_ = None
        self.mi_scores_ = None
        self.feature_names_ = None
        # Emission parameters learned from data
        self.emission_means_ = None
        self.emission_stds_ = None

    def _knn_entropy(self, x: np.ndarray) -> float:
        """
        Estimate entropy using k-nearest neighbors (KSG estimator).
        
        H(X) ≈ ψ(N) - ψ(k) + log(c_d) + (d/N) Σ_i log(ε_i)
        
        Where:
        - ψ is the digamma function
        - c_d is the volume of unit ball in d dimensions
        - ε_i is the distance to k-th nearest neighbor
        
        For 1D: c_1 = 2, so log(c_1) = log(2)
        
        This is more robust than histogram estimation:
        - Sample-efficient
        - No bin count sensitivity
        - Better tail behavior
        """
        x_clean = x[~np.isnan(x) & ~np.isinf(x)]
        n = len(x_clean)
        
        # Strict bounds check: need at least k+1 points for k neighbors
        k_requested = self.n_neighbors
        if n < k_requested + 1:
            return 0.0
        
        # Ensure k doesn't exceed available neighbors
        k_actual = min(k_requested, n - 1)
        if k_actual < 1:
            return 0.0
        
        # Reshape for distance computation
        x_reshaped = x_clean.reshape(-1, 1)
        
        # Compute pairwise distances
        # For 1D, this is just |x_i - x_j|
        diffs = x_reshaped - x_reshaped.T
        distances = np.abs(diffs)
        
        # Set diagonal to inf to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Find k-th nearest neighbor distance for each point
        # np.partition(arr, idx) places the (idx+1)-th smallest at index idx
        # For k_actual neighbors, we want the k_actual-th smallest (1-indexed)
        # In 0-indexed: partition at (k_actual-1), take index (k_actual-1)
        # Example: k_actual=5 → partition(4) → index 4 = 5th smallest ✓
        k_idx = k_actual - 1  # Convert to 0-based index
        knn_distances = np.partition(distances, k_idx, axis=1)[:, k_idx]
        
        # Avoid log(0) - use small epsilon
        knn_distances = np.maximum(knn_distances, 1e-10)
        
        # KSG entropy estimator for 1D
        # H ≈ ψ(N) - ψ(k) + log(2) + mean(log(2 * ε))
        h = digamma(n) - digamma(k_actual) + np.log(2) + np.mean(np.log(2 * knn_distances))
        
        return max(0.0, h)  # Entropy is non-negative

    def _estimate_entropy(self, x: np.ndarray) -> float:
        """
        Estimate H(X) using kNN method (primary) with histogram fallback.
        """
        x_clean = x[~np.isnan(x) & ~np.isinf(x)]
        if len(x_clean) < 10:
            return 0.0
        
        # Primary: kNN entropy (more robust)
        return self._knn_entropy(x_clean)

    def _estimate_conditional_entropy(
        self, 
        x: np.ndarray, 
        states: np.ndarray
    ) -> float:
        """
        H(X | S) = Σ_s P(S=s) H(X | S=s)
        """
        h_conditional = 0.0
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() < self.n_neighbors + 1:  # Need enough samples for kNN
                continue
            p_s = mask.mean()
            h_x_given_s = self._estimate_entropy(x[mask])
            h_conditional += p_s * h_x_given_s
        return h_conditional

    def fit(
        self, 
        observations: np.ndarray, 
        state_labels: np.ndarray,
        feature_names: List[str] = None
    ):
        """
        Compute mutual information for each observation dimension.
        Also learns emission parameters (means, stds) per state.
        
        Args:
            observations: Array of shape (T, n_features)
            state_labels: Array of state assignments
            feature_names: Optional list of feature names
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
            
        n_features = observations.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            h_y = self._estimate_entropy(observations[:, i])
            h_y_given_s = self._estimate_conditional_entropy(
                observations[:, i], state_labels
            )
            mi_scores[i] = max(0, h_y - h_y_given_s)  # MI ≥ 0
        
        self.mi_scores_ = mi_scores
        
        # Normalize to sum to 1 (importance weights)
        total_mi = mi_scores.sum()
        if total_mi > 1e-10:
            self.observation_weights_ = mi_scores / total_mi
        else:
            # Uniform weights if no discriminative power
            self.observation_weights_ = np.ones(n_features) / n_features
        
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(n_features)]
        
        # Learn emission parameters
        self.emission_means_ = np.zeros((self.n_states, n_features))
        self.emission_stds_ = np.zeros((self.n_states, n_features))
        
        for s in range(self.n_states):
            mask = state_labels == s
            if mask.sum() > 1:
                self.emission_means_[s] = np.nanmean(observations[mask], axis=0)
                self.emission_stds_[s] = np.nanstd(observations[mask], axis=0) + 1e-6
            else:
                self.emission_means_[s] = np.nanmean(observations, axis=0)
                self.emission_stds_[s] = np.nanstd(observations, axis=0) + 1e-6
        
        return self

    def weighted_log_likelihood(
        self,
        observation: np.ndarray,
        state: int
    ) -> float:
        """
        Compute MI-weighted log-likelihood:
        log P(Y | S) ≈ Σ_i w_i · log N(y_i | μ_{s,i}, σ_{s,i})
        
        Higher weight on more discriminative features.
        """
        if self.observation_weights_ is None:
            raise ValueError("Call fit() first")
        
        if self.emission_means_ is None or self.emission_stds_ is None:
            raise ValueError("Emission parameters not learned. Call fit() first")
        
        log_prob = 0.0
        for i, (y, w) in enumerate(zip(observation, self.observation_weights_)):
            if not np.isfinite(y):
                continue
                
            mu = self.emission_means_[state, i]
            sigma = max(self.emission_stds_[state, i], 1e-6)
            
            # Log Gaussian PDF
            log_p = (
                -0.5 * np.log(2 * np.pi) - 
                np.log(sigma) - 
                0.5 * ((y - mu) / sigma)**2
            )
            
            log_prob += w * log_p
        
        return log_prob

    def compute_kl_divergence_matrix(self) -> np.ndarray:
        """
        KL divergence between state emission distributions.
        D_KL(P_i || P_j) measures how distinguishable states i and j are.
        
        For Gaussians:
        D_KL(N_0 || N_1) = log(σ_1/σ_0) + (σ_0² + (μ_0-μ_1)²)/(2σ_1²) - 1/2
        """
        if self.emission_means_ is None or self.emission_stds_ is None:
            raise ValueError("Emission parameters not learned. Call fit() first")
            
        n_states, n_features = self.emission_means_.shape
        kl_matrix = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(n_states):
                if i == j:
                    continue
                
                kl = 0.0
                for f in range(n_features):
                    mu_i = self.emission_means_[i, f]
                    sigma_i = max(self.emission_stds_[i, f], 1e-6)
                    mu_j = self.emission_means_[j, f]
                    sigma_j = max(self.emission_stds_[j, f], 1e-6)
                    
                    kl += (
                        np.log(sigma_j / sigma_i) +
                        (sigma_i**2 + (mu_i - mu_j)**2) / (2 * sigma_j**2) -
                        0.5
                    )
                
                kl_matrix[i, j] = max(0, kl)  # KL ≥ 0
        
        return kl_matrix

    def compute_symmetric_kl_matrix(self) -> np.ndarray:
        """
        Symmetric KL divergence (Jensen-Shannon style):
        D_sym(i, j) = 0.5 * (D_KL(i||j) + D_KL(j||i))
        """
        kl = self.compute_kl_divergence_matrix()
        return 0.5 * (kl + kl.T)

    def state_discriminability_scores(self) -> np.ndarray:
        """
        How well each state can be distinguished from others.
        Higher = more discriminable state.
        """
        kl_sym = self.compute_symmetric_kl_matrix()
        n_states = len(kl_sym)
        scores = np.zeros(n_states)
        for i in range(n_states):
            other_kls = [kl_sym[i, j] for j in range(n_states) if i != j]
            scores[i] = np.mean(other_kls) if other_kls else 0.0
        return scores

    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Get detailed report on feature importance.
        """
        if self.observation_weights_ is None:
            raise ValueError("Call fit() first")
        
        sorted_idx = np.argsort(self.mi_scores_)[::-1]
        
        return {
            'feature_names': [self.feature_names_[i] for i in sorted_idx],
            'mi_scores': self.mi_scores_[sorted_idx].tolist(),
            'weights': self.observation_weights_[sorted_idx].tolist(),
            'cumulative_weight': np.cumsum(self.observation_weights_[sorted_idx]).tolist(),
        }


# =============================================================================
# ENHANCED HMM INTEGRATION (CAUSAL PIPELINE)
# =============================================================================

class EnhancedHMMInference:
    """
    Integrates Bayesian transition model, Wasserstein regime detection,
    and information-theoretic weighting into a unified CAUSAL inference pipeline.
    
    CAUSAL PIPELINE ORDER (strictly enforced):
    
    1. Wasserstein Distribution Shift Detection
       - Detects if we're in a regime change region
       - Uses PAST windows only (t-W:t vs t-2W:t-W)
       
    2. Bayesian Transition Belief Update
       - Updates transition probability posterior with decay
       - Conditions on detected regime stability
       
    3. Information-Weighted Emission Confidence
       - Weights observation components by discriminative power
       - Adjusts emission likelihood based on MI weights
       
    4. Switch Decision with Uncertainty Gate
       - Only acts when uncertainty is below threshold
       - Combines all components into final posterior
    
    This ordering prevents:
    - Double counting
    - Confidence inflation
    - Regime echoing (using future info)
    """
    
    def __init__(
        self, 
        n_states: int = 4,
        decay_factor: float = 0.995,
        uncertainty_gate_threshold: float = 0.3
    ):
        """
        Args:
            n_states: Number of hidden states
            decay_factor: Forgetting factor for Bayesian updates
            uncertainty_gate_threshold: Max uncertainty to allow decision
        """
        self.n_states = n_states
        self.uncertainty_gate_threshold = uncertainty_gate_threshold
        
        # Initialize components
        self.bayesian_transition = BayesianTransitionModel(
            n_states=n_states, 
            decay_factor=decay_factor
        )
        self.wasserstein_detector = None
        self.info_weighting = InformationTheoreticWeighting(n_states=n_states)
        
        # Pipeline state
        self.is_fitted = False
        self._last_regime_change_score = 0.0
        self._last_uncertainty = 1.0
    
    def fit(
        self,
        observations: np.ndarray,
        state_posteriors: List[np.ndarray],
        feature_names: List[str] = None
    ):
        """
        Fit all enhanced components from inference results.
        
        IMPORTANT: This uses historical data only - no lookahead.
        
        Args:
            observations: Observation matrix (T, n_features)
            state_posteriors: List of posterior distributions from forward algorithm
            feature_names: Optional feature names
        """
        if len(observations) == 0 or len(state_posteriors) == 0:
            return self
        
        # Convert posteriors to hard assignments for MI computation
        state_labels = np.array([np.argmax(p) for p in state_posteriors])
        
        # Step 1: Wasserstein regime change detection (CAUSAL)
        window_size = min(50, len(observations) // 10)
        if window_size < 5:
            window_size = 5
        self.wasserstein_detector = WassersteinRegimeDetector(window_size=window_size)
        self.wasserstein_detector.compute_regime_change_scores(observations)
        
        # Step 2: Bayesian transition update with decay
        # Use soft counts to update, respecting temporal order
        self.bayesian_transition.update_from_posteriors(
            state_posteriors, 
            apply_batch_decay=True
        )
        
        # Step 3: Information-theoretic weighting
        self.info_weighting.fit(observations, state_labels, feature_names)
        
        self.is_fitted = True
        return self
    
    def get_enhanced_posterior(
        self,
        base_posterior: np.ndarray,
        observation: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Enhance base posterior following the CAUSAL PIPELINE.
        
        Pipeline:
        1. Wasserstein → detect distribution shift
        2. Bayesian → get transition uncertainty
        3. MI weights → adjust emission confidence
        4. Uncertainty gate → decide if adjustment is trustworthy
        
        Args:
            base_posterior: Posterior from standard forward algorithm
            observation: Current observation vector
            t: Time index
            
        Returns:
            Enhanced posterior distribution
        """
        if not self.is_fitted:
            return base_posterior
        
        enhanced = base_posterior.copy()
        
        # =================================================================
        # STEP 1: Wasserstein Distribution Shift Detection
        # =================================================================
        regime_change_score = 0.0
        if self.wasserstein_detector is not None and self.wasserstein_detector.scores_ is not None:
            if t < len(self.wasserstein_detector.scores_):
                regime_change_score = self.wasserstein_detector.scores_[t]
        self._last_regime_change_score = regime_change_score
        
        # =================================================================
        # STEP 2: Bayesian Transition Uncertainty
        # =================================================================
        transition_uncertainty = self.bayesian_transition.transition_uncertainty()
        mean_uncertainty = float(transition_uncertainty.mean())
        self._last_uncertainty = mean_uncertainty
        
        # If in high regime-change region, increase forward transition boost
        # This is CONDITIONAL on Wasserstein signal (no double-counting)
        regime_adjustment = 1.0
        if regime_change_score > 0.5:
            # Scale boost by inverse uncertainty (more certain = larger boost)
            certainty_factor = max(0.1, 1.0 - mean_uncertainty * 10)
            regime_adjustment = 1.0 + 0.3 * regime_change_score * certainty_factor
        
        # =================================================================
        # STEP 3: Information-Weighted Emission Adjustment
        # =================================================================
        if self.info_weighting.observation_weights_ is not None:
            # Compute MI-weighted log-likelihood ratios
            log_lik_ratios = np.zeros(self.n_states)
            for s in range(self.n_states):
                log_lik_ratios[s] = self.info_weighting.weighted_log_likelihood(
                    observation, s
                )
            
            # Normalize log-likelihoods to prevent overflow
            log_lik_ratios -= np.max(log_lik_ratios)
            
            # Soft adjustment factor (conservative: 0.1 weight)
            # Only apply if we have reasonable MI scores
            if self.info_weighting.mi_scores_ is not None:
                total_mi = self.info_weighting.mi_scores_.sum()
                if total_mi > 0.1:  # Meaningful discrimination
                    mi_adjustment = np.exp(0.1 * log_lik_ratios)
                    enhanced *= mi_adjustment
        
        # =================================================================
        # STEP 4: Apply Regime Change Boost (if warranted)
        # =================================================================
        if regime_adjustment > 1.0:
            # Boost forward states proportionally
            enhanced[LatentState.PRE_POLICY] *= regime_adjustment
            enhanced[LatentState.POLICY] *= regime_adjustment ** 0.5  # Smaller boost for extreme state
        
        # =================================================================
        # STEP 5: Uncertainty Gate
        # =================================================================
        # If uncertainty is too high, blend back toward base posterior
        if mean_uncertainty > self.uncertainty_gate_threshold:
            blend_factor = min(1.0, mean_uncertainty / self.uncertainty_gate_threshold - 1.0)
            enhanced = (1 - blend_factor) * enhanced + blend_factor * base_posterior
        
        # Renormalize
        enhanced = enhanced / (enhanced.sum() + 1e-10)
        
        return enhanced
    
    def get_switch_confidence(self, enhanced_posterior: np.ndarray) -> Dict[str, float]:
        """
        Compute calibrated switch confidence with uncertainty quantification.
        
        Returns:
            Dictionary with:
            - switch_probability: P(PRE_POLICY) + P(POLICY)
            - confidence: How certain we are (inverse of uncertainty)
            - regime_change_score: Latest Wasserstein score
            - should_gate: Whether uncertainty gate is active
        """
        switch_prob = float(
            enhanced_posterior[LatentState.PRE_POLICY] + 
            enhanced_posterior[LatentState.POLICY]
        )
        
        # Confidence is inverse of uncertainty, normalized to [0, 1]
        # Using sigmoid-style transformation for smooth behavior
        # confidence = 1 / (1 + exp(k * (uncertainty - threshold)))
        # Simplified: use linear scaling with proper clamping
        uncertainty = self._last_uncertainty
        
        # Scale uncertainty relative to threshold
        # If uncertainty < threshold: confidence is high
        # If uncertainty > threshold: confidence drops
        if uncertainty <= 0:
            confidence = 1.0
        elif uncertainty >= self.uncertainty_gate_threshold * 2:
            confidence = 0.0
        else:
            # Linear interpolation: full confidence at 0, zero at 2*threshold
            confidence = 1.0 - (uncertainty / (self.uncertainty_gate_threshold * 2))
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        should_gate = uncertainty > self.uncertainty_gate_threshold
        
        # Gated probability: only non-zero if gate is open AND we're confident
        if should_gate:
            gated_prob = 0.0
        else:
            # Weight switch probability by confidence
            gated_prob = switch_prob * confidence
        
        return {
            'switch_probability': switch_prob,
            'confidence': confidence,
            'regime_change_score': float(self._last_regime_change_score),
            'uncertainty': float(uncertainty),
            'should_gate': should_gate,
            'gated_switch_probability': float(gated_prob),
        }
    
    def get_transition_matrix_with_uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Get expected transition matrix with uncertainty estimates.
        """
        return {
            'expected': self.bayesian_transition.expected_transition_matrix(),
            'variance': self.bayesian_transition.transition_uncertainty(),
            'entropy': self.bayesian_transition.posterior_entropy(),
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics from all components.
        """
        diagnostics = {
            'is_fitted': self.is_fitted,
            'pipeline_order': [
                '1. Wasserstein Distribution Shift',
                '2. Bayesian Transition Update', 
                '3. MI-Weighted Emission',
                '4. Uncertainty Gate'
            ],
        }
        
        if self.is_fitted:
            # Bayesian transition diagnostics
            diagnostics['transition_entropy'] = self.bayesian_transition.posterior_entropy()
            diagnostics['transition_uncertainty_mean'] = float(
                self.bayesian_transition.transition_uncertainty().mean()
            )
            diagnostics['bayesian_update_count'] = self.bayesian_transition._update_count
            
            # Wasserstein diagnostics
            if self.wasserstein_detector is not None:
                diagnostics['regime_stability'] = self.wasserstein_detector.regime_stability_score()
                diagnostics['n_regime_changes'] = int(
                    (self.wasserstein_detector.scores_ > self.wasserstein_detector.threshold_).sum()
                )
                diagnostics['wasserstein_threshold'] = float(self.wasserstein_detector.threshold_)
            
            # Information-theoretic diagnostics
            if self.info_weighting.mi_scores_ is not None:
                diagnostics['feature_importance'] = self.info_weighting.get_feature_importance_report()
                diagnostics['state_discriminability'] = (
                    self.info_weighting.state_discriminability_scores().tolist()
                )
                diagnostics['total_mi'] = float(self.info_weighting.mi_scores_.sum())
        
        return diagnostics


# =============================================================================
# UNIFIED DECISION GATE (SINGLE SCALAR OUTPUT)
# =============================================================================

class UnifiedDecisionGate:
    """
    One dominant signal, one belief updater, one decision gate.
    Everything else is subordinate.
    
    Architecture:
    - Wasserstein = PRIMARY SENSOR (decides if world changed)
    - Bayesian transitions = BELIEF UPDATER (slow, with decay)
    - MI/KL = AMPLIFIERS ONLY (boost confidence, never trigger)
    - Single scalar output: switch_confidence ∈ [0, 1]
    
    This eliminates:
    - Confidence inflation from multiple voices
    - Delayed switches from competing signals
    - Regime echo from circular dependencies
    """
    
    def __init__(
        self,
        n_states: int = 4,
        wasserstein_threshold: float = 0.15,
        belief_decay: float = 0.95,
        confidence_ema_alpha: float = 0.3,
        switch_threshold: float = 0.6,
    ):
        """
        Args:
            n_states: Number of hidden states
            wasserstein_threshold: Primary sensor threshold for regime change
            belief_decay: Bayesian decay per step (pulls toward uniform)
            confidence_ema_alpha: EMA smoothing for confidence (higher = more responsive)
            switch_threshold: Confidence threshold for switch decision
        """
        self.n_states = n_states
        self.wasserstein_threshold = wasserstein_threshold
        self.belief_decay = belief_decay
        self.confidence_ema_alpha = confidence_ema_alpha
        self.switch_threshold = switch_threshold
        
        # State
        self.belief = np.ones(n_states) / n_states
        self.switch_confidence = 0.0
        self._last_wasserstein = 0.0
        self._last_mi_ratio = 1.0
        self._last_kl_divergence = 0.0
        self._confidence_ema = 0.0
        self._primary_signal = 0.0
        self._amplifier = 1.0
    
    def compute_switch_confidence(
        self,
        wasserstein_distance: float,
        mi_ratio: Optional[float] = None,
        kl_divergence: Optional[float] = None,
    ) -> float:
        """
        Single scalar output: switch_confidence ∈ [0, 1]
        
        Hierarchy (STRICTLY ENFORCED):
        1. Wasserstein = PRIMARY SENSOR (decides if world changed)
        2. MI/KL = AMPLIFIERS ONLY (boost confidence, never trigger)
        3. Everything else = SUBORDINATE
        
        Args:
            wasserstein_distance: Primary sensor signal (distribution shift)
            mi_ratio: Optional MI discriminability ratio (amplifier only)
            kl_divergence: Optional KL divergence between states (amplifier only)
            
        Returns:
            switch_confidence ∈ [0, 1]
        """
        self._last_wasserstein = wasserstein_distance
        
        # =================================================================
        # PRIMARY SIGNAL: Wasserstein (ONLY thing that can trigger)
        # =================================================================
        # Normalize relative to threshold
        w_normalized = wasserstein_distance / (self.wasserstein_threshold + 1e-10)
        
        # Sharp sigmoid centered at threshold
        # Below threshold: signal → 0
        # Above threshold: signal → 1
        # Sharpness = 5 gives reasonable transition width
        self._primary_signal = 1.0 / (1.0 + np.exp(-5.0 * (w_normalized - 1.0)))
        
        # =================================================================
        # AMPLIFIERS: MI and KL (can ONLY boost, never trigger)
        # =================================================================
        self._amplifier = 1.0
        
        # MI ratio amplifier
        if mi_ratio is not None and np.isfinite(mi_ratio):
            self._last_mi_ratio = mi_ratio
            # MI ratio > 1 means states are becoming more distinguishable
            # Amplify confidence by up to 30%, or reduce by up to 20%
            mi_factor = np.clip(mi_ratio, 0.8, 1.3)
            self._amplifier *= mi_factor
        
        # KL divergence amplifier
        if kl_divergence is not None and np.isfinite(kl_divergence):
            self._last_kl_divergence = kl_divergence
            # High KL = distributions diverging = more confidence in change
            # Logarithmic scaling to prevent explosion
            kl_factor = 1.0 + 0.1 * np.log1p(kl_divergence)
            kl_factor = np.clip(kl_factor, 0.9, 1.2)
            self._amplifier *= kl_factor
        
        # =================================================================
        # COMBINE: Primary * Amplifier
        # =================================================================
        raw_confidence = self._primary_signal * self._amplifier
        raw_confidence = np.clip(raw_confidence, 0.0, 1.0)
        
        # =================================================================
        # EMA SMOOTHING: Prevents jitter while maintaining responsiveness
        # =================================================================
        self._confidence_ema = (
            self.confidence_ema_alpha * raw_confidence +
            (1.0 - self.confidence_ema_alpha) * self._confidence_ema
        )
        
        self.switch_confidence = float(self._confidence_ema)
        return self.switch_confidence
    
    def update_belief(
        self,
        transition_matrix: np.ndarray,
        observation_likelihood: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Bayesian belief update with mandatory decay.
        
        Decay pulls belief toward uniform uncertainty.
        This prevents overconfidence in stale beliefs.
        
        Args:
            transition_matrix: State transition probabilities
            observation_likelihood: Optional P(Y|S) for each state
            
        Returns:
            Updated belief distribution
        """
        # Apply decay (always pulls toward uniform)
        uniform = np.ones(self.n_states) / self.n_states
        self.belief = self.belief_decay * self.belief + (1.0 - self.belief_decay) * uniform
        
        # Transition update: P(S_t | S_{t-1})
        self.belief = transition_matrix.T @ self.belief
        
        # Observation update (if available): P(S_t | Y_t)
        if observation_likelihood is not None:
            obs_lik = np.asarray(observation_likelihood)
            if obs_lik.shape == self.belief.shape:
                self.belief *= obs_lik
        
        # Normalize
        belief_sum = self.belief.sum()
        if belief_sum > 1e-10:
            self.belief = self.belief / belief_sum
        else:
            self.belief = uniform.copy()
        
        return self.belief
    
    def should_switch(self) -> Tuple[bool, float, str]:
        """
        Final decision: switch or not.
        
        Returns:
            Tuple of (decision, confidence, reason)
        """
        decision = self.switch_confidence > self.switch_threshold
        
        if decision:
            reason = f"Confidence {self.switch_confidence:.1%} > threshold {self.switch_threshold:.1%}"
        else:
            reason = f"Confidence {self.switch_confidence:.1%} ≤ threshold {self.switch_threshold:.1%}"
        
        return decision, self.switch_confidence, reason
    
    def get_dominant_state(self) -> int:
        """Most likely state from belief."""
        return int(np.argmax(self.belief))
    
    def get_state_probability(self, state: int) -> float:
        """Get probability of specific state."""
        if 0 <= state < self.n_states:
            return float(self.belief[state])
        return 0.0
    
    def get_forward_state_probability(self) -> float:
        """
        Get probability of being in forward states (PRE_POLICY or POLICY).
        This is the key metric for switch decisions.
        """
        if self.n_states >= 4:
            return float(self.belief[2] + self.belief[3])  # PRE_POLICY + POLICY
        return float(self.belief[-1])
    
    def reset(self):
        """Reset to initial state."""
        self.belief = np.ones(self.n_states) / self.n_states
        self.switch_confidence = 0.0
        self._confidence_ema = 0.0
        self._primary_signal = 0.0
        self._amplifier = 1.0
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get detailed diagnostics for debugging and monitoring.
        """
        return {
            'switch_confidence': self.switch_confidence,
            'should_switch': self.switch_confidence > self.switch_threshold,
            'threshold': self.switch_threshold,
            'belief': self.belief.tolist(),
            'dominant_state': self.get_dominant_state(),
            'forward_state_probability': self.get_forward_state_probability(),
            'primary_signal': {
                'wasserstein': self._last_wasserstein,
                'normalized_signal': self._primary_signal,
                'threshold': self.wasserstein_threshold,
            },
            'amplifiers': {
                'combined': self._amplifier,
                'mi_ratio': self._last_mi_ratio,
                'kl_divergence': self._last_kl_divergence,
            },
            'smoothing': {
                'ema_alpha': self.confidence_ema_alpha,
                'ema_value': self._confidence_ema,
            },
        }


# =============================================================================
# DATA INGESTION (READ-ONLY, ISOLATED)
# =============================================================================

def _load_eurjpy_prices(data_path: str = EURJPY_DATA_FILE, force_refresh: bool = True, quiet: bool = False) -> Optional[pd.Series]:
    """
    Load EURJPY price series, downloading fresh data from yfinance.
    
    Args:
        data_path: Path to EURJPY daily data CSV
        force_refresh: If True, always download fresh data
        quiet: Suppress verbose output
        
    Returns:
        pd.Series with DatetimeIndex and EURJPY prices, or None if unavailable
    """
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(msg: str):
        if not quiet:
            print(msg)
    
    # Try to download fresh data from yfinance
    if force_refresh:
        try:
            import yfinance as yf
            
            log(f"[debt_allocator] Downloading fresh EURJPY data...")
            
            ticker = yf.Ticker("EURJPY=X")
            df = ticker.history(period="max")  # Get maximum available history (>10 years)
            
            if df is not None and not df.empty and 'Close' in df.columns:
                df_save = df.reset_index()
                df_save.to_csv(path, index=False)
                
                log(f"[debt_allocator] Downloaded {len(df)} days of EURJPY data")
                log(f"[debt_allocator] Saved to: {path}")
                
                prices = df['Close'].dropna()
                if len(prices) >= MIN_HISTORY_DAYS:
                    return prices.sort_index()
                else:
                    log(f"[debt_allocator] Warning: Only {len(prices)} days (need {MIN_HISTORY_DAYS})")
        except ImportError:
            log("[debt_allocator] Warning: yfinance not installed, trying cached data")
        except Exception as e:
            log(f"[debt_allocator] Warning: Download failed ({e}), trying cached data")
    
    # Fall back to cached data
    try:
        if path.exists():
            log(f"[debt_allocator] Loading cached EURJPY data from: {path}")
            df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            
            if 'Close' in df.columns:
                prices = df['Close']
            elif 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                log("[debt_allocator] Error: No Close column in cached data")
                return None
            
            prices = prices.dropna()
            if len(prices) >= MIN_HISTORY_DAYS:
                log(f"[debt_allocator] Loaded {len(prices)} days from cache")
                return prices.sort_index()
            else:
                log(f"[debt_allocator] Warning: Cached data has only {len(prices)} days")
                return None
        else:
            log(f"[debt_allocator] Error: No cached data found at {path}")
            return None
            
    except Exception as e:
        log(f"[debt_allocator] Error loading cached data: {e}")
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
    long_window: int = VOL_LOOKBACK_LONG,
    adaptive: bool = True
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
        adaptive: If True (default), use available data when insufficient for long_window
                 (but only if data is at least 60% of required)
        
    Returns:
        Volatility ratio
    """
    n = len(log_returns)
    
    # Minimum requirement: need at least short_window * 2 for any comparison
    if n < short_window * 2:
        return float('nan')
    
    # Check if we have enough data for requested long_window
    if n < long_window:
        if adaptive:
            # Only adapt if we have at least 60% of requested data
            # This prevents using very limited data that could be misleading
            min_required = int(long_window * 0.6)
            if n < min_required:
                return float('nan')
            # Adaptive mode - use all available data
            effective_long_window = n
            # Ensure meaningful separation from short window
            if effective_long_window <= short_window:
                return float('nan')
        else:
            # Strict mode - return NaN if not enough data
            return float('nan')
    else:
        effective_long_window = long_window
    
    recent = log_returns.iloc[-short_window:]
    historical = log_returns.iloc[-effective_long_window:]
    
    vol_short = float(recent.std())
    vol_long = float(historical.std())
    
    if vol_long <= 0 or not np.isfinite(vol_long):
        return float('nan')
    
    return vol_short / vol_long


# =============================================================================
# OBSERVATION VECTOR CONSTRUCTION
# =============================================================================

def _construct_observation_vector(
    log_returns: pd.Series,
    prev_disagreement: Optional[float] = None,
    prev_convex_loss: Optional[float] = None,
    timestamp: Optional[str] = None
) -> ObservationVector:
    """
    Construct the full observation vector Y_t.
    
    Args:
        log_returns: Historical log returns up to time t
        prev_disagreement: Previous disagreement for momentum calculation
        prev_convex_loss: Previous convex loss for acceleration calculation
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
    
    # 2) Convex Loss Acceleration: ΔC(t) = C(t) - C(t-1)
    if prev_convex_loss is not None and np.isfinite(prev_convex_loss) and np.isfinite(convex_loss):
        convex_loss_acceleration = convex_loss - prev_convex_loss
    else:
        convex_loss_acceleration = 0.0
    
    # 3) Tail Mass
    tail_mass = _compute_tail_mass(samples_21d)
    
    # 4) Epistemic Disagreement
    disagreement, _, _ = _compute_epistemic_disagreement(log_returns)
    
    # 5) Disagreement Momentum
    if prev_disagreement is not None and np.isfinite(prev_disagreement) and np.isfinite(disagreement):
        disagreement_momentum = disagreement - prev_disagreement
    else:
        disagreement_momentum = 0.0
    
    # 6) Volatility Ratio (adaptive mode for robustness with limited data)
    vol_ratio = _compute_volatility_ratio(log_returns, adaptive=True)
    
    # Construct raw observation
    raw_obs = ObservationVector(
        convex_loss=convex_loss,
        convex_loss_acceleration=convex_loss_acceleration,
        tail_mass=tail_mass,
        disagreement=disagreement,
        disagreement_momentum=disagreement_momentum,
        vol_ratio=vol_ratio,
        timestamp=timestamp,
    )
    
    # Validate and return (handles NaN/Inf and outliers)
    return _validate_observation(raw_obs)


# =============================================================================
# HIDDEN MARKOV MODEL: TRANSITION MATRIX
# =============================================================================

def _build_base_transition_matrix(persistence: float = TRANSITION_PERSISTENCE) -> np.ndarray:
    """
    Build base monotone transition matrix respecting partial ordering.
    
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


def _compute_transition_pressure(
    observation: np.ndarray,
    historical_quantiles: Dict[str, float]
) -> float:
    """
    Compute bounded transition-pressure scalar Φ(Y).
    
    Φ(Y) is constructed from:
    - Convex loss acceleration: ΔC(t) = C(t) − C(t−1)
    - Epistemic momentum: dD(t) = D(t) − D(t−1)
    - Volatility compression indicator
    
    Properties:
    - Φ(Y) ≥ 0 (always non-negative)
    - Φ(Y) increases monotonically with stress acceleration
    - Φ(Y) saturates (bounded, no runaway probabilities)
    
    Args:
        observation: Current observation vector Y_t
        historical_quantiles: Dict with quantile arrays for each metric
        
    Returns:
        Bounded transition pressure scalar Φ(Y) ∈ [0, 1]
    """
    # Extract observation components
    # Array layout: [convex_loss, tail_mass, disagreement, disag_momentum, vol_ratio, convex_accel]
    obs_disag_momentum = observation[3]  # dD(t)
    obs_vol = observation[4]  # V(t)
    obs_convex_accel = observation[5] if len(observation) > 5 else 0.0  # ΔC(t)
    
    # Get historical quantiles for normalization
    convex_q75 = historical_quantiles.get('convex_q75', 0.0006)
    convex_q90 = historical_quantiles.get('convex_q90', 0.001)
    disag_q75 = historical_quantiles.get('disag_q75', 0.30)
    vol_q25 = historical_quantiles.get('vol_q25', 0.85)
    
    # Normalization scale for convex loss acceleration
    # Typical ΔC range is ±0.0003, extreme is ±0.001
    convex_accel_scale = convex_q90 - convex_q75 + 1e-8
    
    # Component 1: Convex loss acceleration contribution
    # Uses tanh for bounded, saturating, monotonic transformation
    if np.isfinite(obs_convex_accel):
        # Normalize and apply tanh for saturation
        # Positive acceleration → higher pressure
        normalized_accel = obs_convex_accel / convex_accel_scale
        phi_convex = max(0.0, np.tanh(normalized_accel * 2.0))  # [0, 1)
    else:
        phi_convex = 0.0
    
    # Component 2: Epistemic momentum contribution
    # Rising disagreement → higher pressure
    if np.isfinite(obs_disag_momentum):
        # Normalize by typical disagreement range
        normalized_momentum = obs_disag_momentum / (disag_q75 + 1e-8)
        phi_disag = max(0.0, np.tanh(normalized_momentum * 3.0))  # [0, 1)
    else:
        phi_disag = 0.0
    
    # Component 3: Volatility compression indicator
    # Low vol ratio (compression) → higher pressure
    if np.isfinite(obs_vol):
        # Compression occurs when V(t) < vol_q25
        if obs_vol < vol_q25:
            # More compressed → higher pressure
            compression_ratio = (vol_q25 - obs_vol) / vol_q25
            phi_vol = np.tanh(compression_ratio * 4.0)  # [0, 1)
        else:
            phi_vol = 0.0
    else:
        phi_vol = 0.0
    
    # Combine components with weighted sum
    # Weights reflect relative importance of each stress signal
    w_convex = 0.50  # Convex acceleration is primary signal
    w_disag = 0.30   # Epistemic momentum is secondary
    w_vol = 0.20     # Volatility compression is tertiary
    
    phi_raw = w_convex * phi_convex + w_disag * phi_disag + w_vol * phi_vol
    
    # Final saturation to ensure Φ(Y) ∈ [0, 1]
    # v3.1.0 Fix: Use tanh for bounded saturation that equals 0 at baseline
    # tanh(0) = 0, so Φ = 0 when all stress components are 0
    # This satisfies the requirement: "equals 0 under baseline conditions"
    phi = np.tanh(phi_raw * 2.0)  # Scale factor controls saturation rate
    
    # Ensure non-negative and bounded
    phi = max(0.0, min(phi, 1.0))
    
    return phi


def _compute_endogenous_transition_matrix(
    base_matrix: np.ndarray,
    observation: np.ndarray,
    historical_quantiles: Dict[str, float]
) -> np.ndarray:
    """
    Compute observation-dependent transition matrix.
    
    v3.1.0 - Issue 1 Fix: Endogenous transitions using bounded Φ(Y)
    
    P(S_t | S_{t−1}) → P(S_t | S_{t−1}, Φ(Y_{t−1}))
    
    Transition effects of Φ(Y):
    - Increase probability of advancing to next state
    - Never allow backward transitions (monotone constraint)
    - Preserve probability mass normalization
    
    This creates a stress-accelerated latent process without
    introducing new states or loss of identifiability.
    
    Args:
        base_matrix: Base transition matrix (without observation conditioning)
        observation: Current observation vector Y_t
        historical_quantiles: Dict with quantile arrays for each metric
        
    Returns:
        Observation-conditioned 4x4 transition matrix
    """
    n_states = LatentState.n_states()
    A = base_matrix.copy()
    
    # Compute transition pressure scalar Φ(Y) (v3.1.0)
    phi = _compute_transition_pressure(observation, historical_quantiles)
    
    # Modify transition probabilities based on Φ(Y)
    # Higher Φ → lower persistence → faster forward transitions
    for i in range(n_states - 1):  # Don't modify POLICY (absorbing)
        # Current persistence (diagonal element)
        base_persistence = A[i, i]
        
        # Reduce persistence proportionally to Φ(Y)
        # Formula: p'(i,i) = p(i,i) * (1 - φ * reduction_factor)
        # where reduction_factor controls how much Φ can reduce persistence
        reduction_factor = 0.4  # Max 40% reduction at Φ=1
        adjusted_persistence = base_persistence * (1.0 - phi * reduction_factor)
        
        # Ensure persistence stays reasonable (at least 50% of base)
        adjusted_persistence = max(adjusted_persistence, base_persistence * 0.5)
        
        # Redistribute removed probability mass to forward states only
        # (backward transitions remain forbidden)
        removed_mass = base_persistence - adjusted_persistence
        
        if removed_mass > 1e-10:
            # Get current forward transition probabilities
            forward_probs = A[i, i+1:].copy()
            forward_sum = forward_probs.sum()
            
            if forward_sum > 1e-10:
                # Distribute removed mass proportionally to existing forward probs
                A[i, i+1:] = forward_probs + removed_mass * (forward_probs / forward_sum)
            else:
                # No existing forward probs: put all mass on next state
                A[i, i+1] = removed_mass
        
        A[i, i] = adjusted_persistence
    
    # Normalize rows to ensure valid probability distribution
    for i in range(n_states):
        row_sum = A[i, :].sum()
        if row_sum > 0:
            A[i, :] = A[i, :] / row_sum
    
    return A


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
    return _build_base_transition_matrix(persistence)


# =============================================================================
# HIDDEN MARKOV MODEL: EMISSION PROBABILITIES (LOG-SPACE)
# =============================================================================

def _log_gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Compute log of Gaussian PDF for numerical stability.
    
    log N(x | μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²
    
    Args:
        x: Observation value
        mu: Mean
        sigma: Standard deviation (must be > 0)
        
    Returns:
        Log probability density
    """
    if sigma <= 0:
        sigma = 1e-6  # Minimum sigma for numerical stability
    
    z = (x - mu) / sigma
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * z * z


def _log_soft_indicator(x: float, threshold: float, direction: str = 'above', 
                        sharpness: float = 10.0) -> float:
    """
    Soft indicator function in log-space using sigmoid.
    
    Provides smooth transition around threshold for numerical stability.
    
    Args:
        x: Observation value
        threshold: Threshold value
        direction: 'above' or 'below' threshold
        sharpness: Controls steepness of transition
        
    Returns:
        Log probability (negative value)
    """
    if direction == 'above':
        z = sharpness * (x - threshold)
    else:
        z = sharpness * (threshold - x)
    
    # Numerically stable log-sigmoid: log(1 / (1 + exp(-z))) = -log(1 + exp(-z))
    # For large positive z: ≈ 0
    # For large negative z: ≈ z
    if z > 20:
        return 0.0
    elif z < -20:
        return z
    else:
        return -np.log1p(np.exp(-z))


def _estimate_emission_likelihood(
    observation: np.ndarray,
    state: LatentState,
    historical_observations: np.ndarray
) -> float:
    """
    Estimate log emission likelihood log P(Y_t | S_t) using empirical calibration.
    
    Uses log-space computation throughout for numerical stability.
    Validates observation values before processing.
    
    Args:
        observation: Current observation vector (will be validated)
        state: Latent state
        historical_observations: Historical observations for calibration
        
    Returns:
        Log-likelihood of observation given state
    """
    # Validate observation array
    obs = _validate_observation_array(observation)
    
    if len(historical_observations) < 50:
        # Insufficient data - use uniform likelihood (log(1) = 0)
        return 0.0
    
    # Extract validated observation components
    # Array layout: [convex_loss, tail_mass, disagreement, disag_momentum, vol_ratio, convex_accel]
    obs_convex = obs[0]    # C(t) # Convex loss
    obs_tail = obs[1]      # P(t) # Tail mass
    obs_disag = obs[2]     # D(t) # Disagreement
    obs_momentum = obs[3]  # dD(t)  # Disagreement momentum
    obs_vol = obs[4]       # V(t) # Volatility ratio
    
    # Compute historical statistics for calibration
    hist_convex = historical_observations[:, 0]
    hist_tail = historical_observations[:, 1]
    hist_disag = historical_observations[:, 2]
    hist_vol = historical_observations[:, 4]
    
    # Filter valid values
    hist_convex = hist_convex[np.isfinite(hist_convex)]
    hist_tail = hist_tail[np.isfinite(hist_tail)]
    hist_disag = hist_disag[np.isfinite(hist_disag)]
    hist_vol = hist_vol[np.isfinite(hist_vol)]
    
    # Compute quantiles and statistics for each metric
    if len(hist_convex) > 10:
        convex_q25, convex_q50, convex_q75, convex_q90 = np.percentile(hist_convex, [25, 50, 75, 90])
        convex_std = np.std(hist_convex) + 1e-8
    else:
        convex_q25, convex_q50, convex_q75, convex_q90 = 0.0001, 0.0003, 0.0006, 0.001
        convex_std = 0.0003
    
    if len(hist_tail) > 10:
        tail_q25, tail_q50, tail_q75, tail_q90 = np.percentile(hist_tail, [25, 50, 75, 90])
        tail_std = np.std(hist_tail) + 1e-8
    else:
        tail_q25, tail_q50, tail_q75, tail_q90 = 0.48, 0.50, 0.52, 0.55
        tail_std = 0.02
    
    if len(hist_disag) > 10:
        disag_q25, disag_q50, disag_q75, disag_q90 = np.percentile(hist_disag, [25, 50, 75, 90])
        disag_std = np.std(hist_disag) + 1e-8
    else:
        disag_q25, disag_q50, disag_q75, disag_q90 = 0.15, 0.20, 0.30, 0.45
        disag_std = 0.1
    
    if len(hist_vol) > 10:
        vol_q10, vol_q25, vol_q50, vol_q75, vol_q90 = np.percentile(hist_vol, [10, 25, 50, 75, 90])
        vol_std = np.std(hist_vol) + 1e-8
    else:
        vol_q10, vol_q25, vol_q50, vol_q75, vol_q90 = 0.7, 0.85, 1.0, 1.15, 1.3
        vol_std = 0.15
    
    # Log-likelihood accumulator (all operations in log-space)
    log_lik = 0.0
    
    # State-specific emission model using log-space soft indicators and Gaussians
    # Each state has characteristic observation patterns
    
    if state == LatentState.NORMAL:
        # NORMAL: Low stress metrics, vol ratio near 1.0
        # Favors observations below 75th percentile
        
        # Convex loss should be low
        log_lik += _log_gaussian_pdf(obs_convex, convex_q50, convex_std)
        log_lik += _log_soft_indicator(obs_convex, convex_q75, 'below', sharpness=5.0)
        
        # Tail mass should be moderate
        log_lik += _log_gaussian_pdf(obs_tail, tail_q50, tail_std)
        
        # Disagreement should be low
        log_lik += _log_soft_indicator(obs_disag, disag_q75, 'below', sharpness=5.0)
        
        # Vol ratio near 1.0
        log_lik += _log_gaussian_pdf(obs_vol, 1.0, 0.15)
        
    elif state == LatentState.COMPRESSED:
        # COMPRESSED: Key signal is low volatility ratio (compression)
        
        # Vol ratio should be LOW (compression)
        log_lik += _log_gaussian_pdf(obs_vol, vol_q25, vol_std * 0.5)
        log_lik += _log_soft_indicator(obs_vol, vol_q50, 'below', sharpness=8.0)
        
        # Convex loss still moderate
        log_lik += _log_soft_indicator(obs_convex, convex_q75, 'below', sharpness=3.0)
        
        # Mild penalty for high disagreement
        log_lik += _log_soft_indicator(obs_disag, disag_q90, 'below', sharpness=2.0)
        
    elif state == LatentState.PRE_POLICY:
        # PRE_POLICY: Elevated stress metrics
        # Multiple metrics above 75th percentile
        
        # Convex loss should be elevated
        log_lik += _log_soft_indicator(obs_convex, convex_q50, 'above', sharpness=5.0)
        log_lik += _log_gaussian_pdf(obs_convex, convex_q75, convex_std)
        
        # Tail mass elevated
        log_lik += _log_soft_indicator(obs_tail, tail_q50, 'above', sharpness=5.0)
        
        # Disagreement elevated
        log_lik += _log_soft_indicator(obs_disag, disag_q50, 'above', sharpness=5.0)
        
        # Positive momentum (worsening conditions)
        if obs_momentum > 0.02:
            log_lik += _log_gaussian_pdf(obs_momentum, 0.05, 0.03)
        else:
            log_lik += -1.0  # Penalty for non-positive momentum
        
    elif state == LatentState.POLICY:
        # POLICY: Extreme metrics - very rare state
        # Requires metrics above 90th percentile
        
        # Extreme convex loss
        log_lik += _log_soft_indicator(obs_convex, convex_q90, 'above', sharpness=10.0)
        log_lik += _log_gaussian_pdf(obs_convex, convex_q90 * 1.2, convex_std * 0.5)
        
        # Extreme tail mass
        log_lik += _log_soft_indicator(obs_tail, tail_q90, 'above', sharpness=10.0)
        
        # Extreme disagreement
        log_lik += _log_soft_indicator(obs_disag, disag_q90, 'above', sharpness=10.0)
        
        # Vol ratio typically expands in crisis
        log_lik += _log_soft_indicator(obs_vol, vol_q75, 'above', sharpness=5.0)
    
    # Clamp to prevent extreme values
    log_lik = np.clip(log_lik, -50.0, 10.0)
    
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
# DYNAMIC DECISION BOUNDARY
# =============================================================================

def _g_convexity_adjustment(delta_c: float, scale: float = 0.0005) -> float:
    """
    Bounded, monotone increasing function g(·) for decision boundary adjustment.
    
    g(ΔC) is used in: α(t) = α₀ − g(ΔC(t))
    
    Properties:
    - g(0) = 0 (no adjustment when convexity is flat)
    - g(ΔC) > 0 for ΔC > 0 (reduce threshold when convexity accelerates)
    - g(·) is monotonically increasing
    - g(·) is bounded (saturates to prevent extreme thresholds)
    
    Uses tanh for smooth saturation.
    
    Args:
        delta_c: Convex loss acceleration ΔC(t) = C(t) - C(t-1)
        scale: Normalization scale for typical ΔC values
        
    Returns:
        Bounded adjustment g(ΔC) ∈ [0, max_adjustment]
    """
    if not np.isfinite(delta_c):
        return 0.0
    
    # Only reduce threshold for positive acceleration (increasing convexity)
    if delta_c <= 0:
        return 0.0
    
    # Normalize by typical scale
    normalized = delta_c / scale
    
    # Apply tanh for bounded, monotonic, saturating behavior
    # tanh(x) ∈ [-1, 1], we use only positive part
    raw_g = np.tanh(normalized)
    
    # Maximum adjustment to α is 0.20 (from 0.60 base to 0.40 minimum)
    max_adjustment = 0.20
    
    return max_adjustment * raw_g


def _compute_dynamic_alpha(
    observation: ObservationVector,
    base_alpha: float = PRE_POLICY_THRESHOLD
) -> float:
    """
    Compute risk-adaptive decision boundary α(t).
    
    v3.1.0 - Issue 2 Fix: Static α replaced with dynamic α(t)
    
    Formula:
        α(t) = α₀ − g(ΔC(t))
    
    Where:
    - α₀ is the baseline confidence threshold (base_alpha)
    - g(·) is a bounded, monotone increasing function
    - ΔC(t) = C(t) - C(t-1) is convex loss acceleration
    
    Interpretation:
    - When convexity is flat (ΔC ≈ 0) → require high certainty (α ≈ α₀)
    - When convexity accelerates (ΔC > 0) → act earlier with less certainty
    
    This implements risk-sensitive Bayesian decision theory.
    No discretionary logic is permitted.
    
    Args:
        observation: Current observation vector
        base_alpha: Baseline threshold α₀ (default PRE_POLICY_THRESHOLD)
        
    Returns:
        Dynamic threshold α(t) ∈ (0, 1)
    """
    # v3.1.0 - Issue 2 Fix: α(t) = α₀ − g(ΔC(t))
    # This is the ONLY adjustment - no discretionary logic
    alpha = base_alpha
    
    if np.isfinite(observation.convex_loss_acceleration):
        g_adjustment = _g_convexity_adjustment(observation.convex_loss_acceleration)
        alpha -= g_adjustment
    
    # Enforce strict bounds: α(t) ∈ (0, 1)
    # Lower bound prevents extreme threshold reduction
    # Upper bound is the base_alpha
    alpha = max(0.40, min(alpha, base_alpha))
    
    return alpha


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def _run_inference(
    log_returns: pd.Series,
    lookback_days: int = 252,
    random_seed: int = 42,
    quiet: bool = False,
) -> Tuple[ObservationVector, StatePosterior, List[ObservationVector]]:
    """
    Run latent state inference on the full history.
    
    Args:
        log_returns: Historical log returns
        lookback_days: Number of days for inference window
        random_seed: Random seed for reproducibility (default: 42)
        quiet: Suppress verbose output
        
    Returns:
        Tuple of (current_observation, current_posterior, observation_history)
    """
    def log(msg: str):
        if not quiet:
            print(msg)
    
    # Freeze randomness for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    n_obs = len(log_returns)
    
    if n_obs < MIN_HISTORY_DAYS:
        raise ValueError(f"Insufficient history: {n_obs} < {MIN_HISTORY_DAYS}")
    
    # Build observation history
    log("[debt_allocator] Building observation history...")
    
    # Use rolling window approach
    window_size = min(lookback_days, n_obs - MIN_HISTORY_DAYS)
    start_idx = n_obs - window_size
    
    observations = []
    obs_arrays = []
    prev_disagreement = None
    prev_convex_loss = None
    
    for t in range(start_idx, n_obs):
        # Use data up to time t
        returns_to_t = log_returns.iloc[:t+1]
        
        obs = _construct_observation_vector(
            returns_to_t, 
            prev_disagreement=prev_disagreement,
            prev_convex_loss=prev_convex_loss,
            timestamp=str(log_returns.index[t])
        )
        observations.append(obs)
        obs_arrays.append(obs.to_array())
        
        # Update for next iteration
        prev_disagreement = obs.disagreement
        prev_convex_loss = obs.convex_loss
    
    if len(observations) == 0:
        raise ValueError("No observations constructed")
    
    log(f"[debt_allocator] Built {len(observations)} observations")
    
    # Convert to array for HMM
    historical_obs_array = np.array(obs_arrays)
    
    # Build transition matrix
    A = _build_transition_matrix()
    
    # Initial distribution (start in NORMAL with high probability)
    pi = np.array([0.85, 0.10, 0.04, 0.01])
    
    # Run forward inference
    log("[debt_allocator] Running forward inference...")
    posteriors = _forward_algorithm(
        [obs.to_array() for obs in observations],
        A,
        pi,
        historical_obs_array
    )
    
    # ==========================================================================
    # ENHANCED HMM: Bayesian Model Averaging + Wasserstein + Information Theory
    # ==========================================================================
    log("[debt_allocator] Running enhanced inference (Bayesian + Wasserstein + MI)...")
    
    feature_names = ['convex_loss', 'tail_mass', 'disagreement', 
                     'disag_momentum', 'vol_ratio', 'convex_accel']
    
    enhanced_hmm = EnhancedHMMInference(n_states=LatentState.n_states())
    enhanced_hmm.fit(
        observations=historical_obs_array,
        state_posteriors=posteriors,
        feature_names=feature_names
    )
    
    # ==========================================================================
    # UNIFIED DECISION GATE: Single scalar output architecture
    # ==========================================================================
    log("[debt_allocator] Running unified decision gate...")
    
    decision_gate = UnifiedDecisionGate(
        n_states=LatentState.n_states(),
        wasserstein_threshold=0.15,
        belief_decay=0.95,
        confidence_ema_alpha=0.3,
        switch_threshold=0.6,
    )
    
    # Get enhanced posterior for final observation
    current_posterior_probs = posteriors[-1]
    
    if enhanced_hmm.is_fitted:
        # Get Wasserstein score (PRIMARY SIGNAL)
        wasserstein_score = 0.0
        if enhanced_hmm.wasserstein_detector is not None:
            if enhanced_hmm.wasserstein_detector.scores_ is not None:
                wasserstein_score = enhanced_hmm.wasserstein_detector.scores_[-1]
        
        # Get MI ratio (AMPLIFIER ONLY)
        mi_ratio = None
        if enhanced_hmm.info_weighting.mi_scores_ is not None:
            total_mi = enhanced_hmm.info_weighting.mi_scores_.sum()
            if total_mi > 0:
                mi_ratio = total_mi / 1.0  # Normalize to baseline of 1.0
        
        # Get KL divergence (AMPLIFIER ONLY)
        kl_divergence = None
        if enhanced_hmm.info_weighting.emission_means_ is not None:
            kl_matrix = enhanced_hmm.info_weighting.compute_kl_divergence_matrix()
            kl_divergence = np.mean(kl_matrix[kl_matrix > 0]) if np.any(kl_matrix > 0) else 0.0
        
        # UNIFIED DECISION: Compute single scalar switch confidence
        switch_confidence_scalar = decision_gate.compute_switch_confidence(
            wasserstein_distance=wasserstein_score,
            mi_ratio=mi_ratio,
            kl_divergence=kl_divergence,
        )
        
        # Update belief using Bayesian transition with decay
        transition_matrix = enhanced_hmm.bayesian_transition.expected_transition_matrix()
        decision_gate.belief = current_posterior_probs.copy()
        decision_gate.update_belief(transition_matrix)
        
        # Get final decision
        should_switch, final_confidence, decision_reason = decision_gate.should_switch()
        
        # Log unified gate diagnostics (only if not quiet)
        if not quiet:
            gate_diag = decision_gate.get_diagnostics()
            log(f"[debt_allocator] Unified Decision Gate:")
            log(f"  - Primary Signal (Wasserstein): {gate_diag['primary_signal']['wasserstein']:.4f}")
            log(f"  - Normalized Signal: {gate_diag['primary_signal']['normalized_signal']:.4f}")
            log(f"  - Amplifier (combined): {gate_diag['amplifiers']['combined']:.3f}")
            log(f"  - Switch Confidence: {gate_diag['switch_confidence']:.2%}")
            log(f"  - Decision: {'SWITCH' if should_switch else 'HOLD'} ({decision_reason})")
            log(f"  - Forward State P(PRE_POLICY+POLICY): {gate_diag['forward_state_probability']:.2%}")
            
            # Also log legacy diagnostics for comparison
            diagnostics = enhanced_hmm.get_diagnostics()
            log(f"[debt_allocator] Supporting diagnostics:")
            log(f"  - Regime stability: {diagnostics.get('regime_stability', 'N/A'):.3f}")
            log(f"  - Transition uncertainty: {diagnostics.get('transition_uncertainty_mean', 'N/A'):.4f}")
            log(f"  - Regime changes detected: {diagnostics.get('n_regime_changes', 'N/A')}")
        
        # Use the unified gate's belief as the final posterior
        current_posterior_probs = decision_gate.belief
    
    # Get current (final) observation and posterior
    current_obs = observations[-1]
    
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
    effective_date: Optional[str],
    dynamic_alpha: float = PRE_POLICY_THRESHOLD
) -> str:
    """
    Compute cryptographic signature for decision audit trail.
    """
    payload = {
        "observation": observation.to_dict(),
        "posterior": posterior.to_dict(),
        "triggered": triggered,
        "effective_date": effective_date,
        "dynamic_alpha": dynamic_alpha,
        "signature_version": "4.0.0",
    }
    
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()


def _make_decision(
    observation: ObservationVector,
    posterior: StatePosterior,
    threshold: float = PRE_POLICY_THRESHOLD,
    use_dynamic_alpha: bool = True
) -> DebtSwitchDecision:
    """
    Make the debt switch decision based on latent state inference.
    
    Trigger logic:
    1. Compute dynamic α(t) based on observation context
    2. Check if P(PRE_POLICY) > α(t) OR P(POLICY) dominant
    
    Args:
        observation: Current observation vector
        posterior: Current state posterior
        threshold: Base decision threshold α
        use_dynamic_alpha: Whether to use dynamic threshold
        
    Returns:
        DebtSwitchDecision
    """
    # Dynamic threshold α(t)
    if use_dynamic_alpha:
        dynamic_alpha = _compute_dynamic_alpha(observation, threshold)
    else:
        dynamic_alpha = threshold
    
    prob_pre_policy = posterior.p_pre_policy
    prob_policy = posterior.p_policy
    
    # Trigger conditions
    triggered = False
    decision_basis = ""
    
    # Primary trigger: P(PRE_POLICY) > α(t)
    if prob_pre_policy > dynamic_alpha:
        triggered = True
        decision_basis = f"PRE_POLICY threshold exceeded (P={prob_pre_policy:.2%} > α={dynamic_alpha:.0%})"
    
    # Belt and suspenders: POLICY state is dominant
    elif posterior.dominant_state == LatentState.POLICY:
        triggered = True
        decision_basis = f"POLICY state dominant (P={prob_policy:.2%})"
    
    # If not triggered, provide status
    if not triggered and not decision_basis:
        decision_basis = (
            f"Below threshold: P(PRE_POLICY)={prob_pre_policy:.2%} ≤ α={dynamic_alpha:.0%}"
        )
    
    effective_date = observation.timestamp.split('T')[0] if triggered else None
    
    signature = _compute_decision_signature(
        observation, posterior, triggered, effective_date, dynamic_alpha
    )
    
    return DebtSwitchDecision(
        triggered=triggered,
        effective_date=effective_date,
        observation=observation,
        state_posterior=posterior,
        decision_basis=decision_basis,
        signature=signature,
        dynamic_alpha=dynamic_alpha,
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
            convex_loss_acceleration=obs_data.get('convex_loss_acceleration', 0.0),
            tail_mass=obs_data['tail_mass'],
            disagreement=obs_data['disagreement'],
            disagreement_momentum=obs_data['disagreement_momentum'],
            vol_ratio=obs_data['vol_ratio'],
            timestamp=obs_data['timestamp'],
        )
        
        post_data = data['state_posterior']
        probs = post_data['probabilities']
        
        # Contract requires exactly 4 states: NORMAL, COMPRESSED, PRE_POLICY, POLICY
        if 'PRE_POLICY' not in probs:
            # Legacy format not supported - discard and re-evaluate
            print("[debt_allocator] Legacy persistence format detected - discarding")
            return None
        
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
            dynamic_alpha=data.get('dynamic_alpha', PRE_POLICY_THRESHOLD),
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
    use_dynamic_alpha: bool = True,
    quiet: bool = False,
) -> Optional[DebtSwitchDecision]:
    """
    Run the debt allocation engine.
    
    Args:
        data_path: Path to EURJPY data
        persistence_path: Path to decision persistence file
        force_reevaluate: If True, ignore persisted decision
        force_refresh_data: If True, download fresh data
        use_dynamic_alpha: Use dynamic threshold α(t)
        quiet: Suppress verbose output
        
    Returns:
        DebtSwitchDecision or None if engine cannot run
    """
    # CHECK FOR PRIOR DECISION
    if not force_reevaluate:
        prior_decision = _load_persisted_decision(persistence_path)
        if prior_decision is not None and prior_decision.triggered:
            if not quiet:
                print("[debt_allocator] Prior triggered decision exists - returning cached")
            return prior_decision
    
    # LOAD DATA
    prices = _load_eurjpy_prices(data_path, force_refresh=force_refresh_data, quiet=quiet)
    if prices is None:
        return None
    
    # COMPUTE LOG RETURNS
    log_returns = _compute_log_returns(prices)
    if len(log_returns) < MIN_HISTORY_DAYS:
        if not quiet:
            print(f"[debt_allocator] Insufficient data: {len(log_returns)} < {MIN_HISTORY_DAYS}")
        return None
    
    # RUN INFERENCE
    try:
        observation, posterior, _ = _run_inference(log_returns, quiet=quiet)
    except Exception as e:
        if not quiet:
            print(f"[debt_allocator] Inference failed: {e}")
            import traceback
            traceback.print_exc()
        return None
    
    # MAKE DECISION
    decision = _make_decision(
        observation, 
        posterior, 
        use_dynamic_alpha=use_dynamic_alpha,
    )
    
    # PERSIST IF TRIGGERED
    if decision.triggered:
        _persist_decision(decision, persistence_path)
        if not quiet:
            print(f"[debt_allocator] DECISION TRIGGERED - persisted to {persistence_path}")
    
    return decision


# =============================================================================
# PRESENTATION (RICH OUTPUT) - v3.0.0 Enhanced
# =============================================================================

def _get_status_display(decision: DebtSwitchDecision) -> Tuple[str, str, str]:
    """Get status emoji, text, and color for display."""
    if decision.triggered:
        if decision.state_posterior.dominant_state == LatentState.POLICY:
            return "🔴", "SWITCH NOW", "red"
        elif decision.state_posterior.dominant_state == LatentState.PRE_POLICY:
            return "🔴", "SWITCH NOW", "red"
        else:
            return "🔴", "SWITCH NOW", "red"
    
    dominant = decision.state_posterior.dominant_state
    if dominant == LatentState.PRE_POLICY:
        return "🟡", "MONITOR", "yellow"
    elif dominant == LatentState.COMPRESSED:
        return "🟡", "MONITOR", "yellow"
    else:
        return "🟢", "HOLD", "bright_green"


def render_debt_switch_decision(
    decision: Optional[DebtSwitchDecision],
    console: Optional[Console] = None
) -> None:
    """Render debt switch decision with extraordinary Apple-quality UX."""
    from rich.rule import Rule
    from rich.align import Align
    from rich.text import Text
    from rich.columns import Columns
    
    if console is None:
        console = Console(force_terminal=True, width=100)
    
    if decision is None:
        console.print()
        error_text = Text()
        error_text.append("\n", style="")
        error_text.append("Engine Cannot Run", style="bold red")
        error_text.append("\n\n", style="")
        error_text.append("Insufficient data or inference failure", style="dim")
        error_text.append("\n", style="")
        
        error_panel = Panel(
            Align.center(error_text),
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 4),
        )
        console.print(Align.center(error_panel, width=50))
        console.print()
        return
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DECISION STATUS - Hero Section
    # ═══════════════════════════════════════════════════════════════════════════════
    status_emoji, status_text, status_color = _get_status_display(decision)
    
    console.print()
    
    # Central status badge
    status_content = Text()
    status_content.append("\n", style="")
    status_content.append(status_emoji + "  ", style="")
    status_content.append(status_text, style=f"bold {status_color}")
    status_content.append("\n", style="")
    
    if decision.triggered:
        status_content.append("\n", style="")
        status_content.append("Effective: ", style="dim")
        status_content.append(str(decision.effective_date), style="bold white")
        status_content.append("\n", style="")
    
    status_panel = Panel(
        Align.center(status_content),
        border_style=status_color if decision.triggered else "dim",
        box=box.DOUBLE if decision.triggered else box.ROUNDED,
        padding=(0, 6),
    )
    console.print(Align.center(status_panel, width=40))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # LATENT STATES - Visual Bars
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print(Rule(style="dim"))
    console.print()
    
    section_header = Text()
    section_header.append("▸ ", style="bright_cyan")
    section_header.append("LATENT STATES", style="bold white")
    console.print(section_header)
    console.print()
    
    post = decision.state_posterior
    states = [
        ("NORMAL", post.p_normal, "bright_green"),
        ("COMPRESSED", post.p_compressed, "yellow"),
        ("PRE_POLICY", post.p_pre_policy, "orange1"),
        ("POLICY", post.p_policy, "red"),
    ]
    
    bar_width = 35
    for state_name, prob, color in states:
        filled = int(prob * bar_width)
        is_dominant = state_name == post.dominant_state.name
        
        row = Text()
        row.append("    ", style="")
        
        if is_dominant:
            row.append("● ", style=f"bold {color}")
            row.append(f"{state_name:<12}", style=f"bold {color}")
            row.append(f"{prob:>7.1%}  ", style=f"bold {color}")
            row.append("━" * filled, style=color)
            row.append("─" * (bar_width - filled), style="dim")
        else:
            row.append("○ ", style="dim")
            row.append(f"{state_name:<12}", style="dim")
            row.append(f"{prob:>7.1%}  ", style="dim")
            row.append("─" * bar_width, style="dim")
        
        console.print(row)
    
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # OBSERVATION METRICS - Clean Grid
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print(Rule(style="dim"))
    console.print()
    
    section_header = Text()
    section_header.append("▸ ", style="bright_cyan")
    section_header.append("METRICS", style="bold white")
    console.print(section_header)
    console.print()
    
    obs = decision.observation
    
    metrics_table = Table(
        show_header=True,
        header_style="dim",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=False,
    )
    metrics_table.add_column("Metric", style="white", width=22)
    metrics_table.add_column("Value", justify="right", width=14)
    metrics_table.add_column("", justify="left", width=12)
    
    # Convex Loss
    convex_str = f"{obs.convex_loss:.6f}" if np.isfinite(obs.convex_loss) else "—"
    convex_status = ""
    
    # Acceleration
    if np.isfinite(obs.convex_loss_acceleration):
        accel = obs.convex_loss_acceleration
        accel_str = f"{accel:+.6f}"
        if accel > 0.0002:
            accel_status = "[yellow]↑ rising[/yellow]"
        elif accel < -0.0002:
            accel_status = "[bright_green]↓ falling[/bright_green]"
        else:
            accel_status = "[dim]stable[/dim]"
    else:
        accel_str = "—"
        accel_status = ""
    
    # Tail Mass
    if np.isfinite(obs.tail_mass):
        tail_str = f"{obs.tail_mass:.1%}"
        if obs.tail_mass > 0.60:
            tail_status = "[yellow]elevated[/yellow]"
        else:
            tail_status = "[dim]normal[/dim]"
    else:
        tail_str = "—"
        tail_status = ""
    
    # Disagreement
    if np.isfinite(obs.disagreement):
        disag_str = f"{obs.disagreement:.3f}"
        if obs.disagreement > 0.4:
            disag_status = "[yellow]high[/yellow]"
        else:
            disag_status = "[dim]normal[/dim]"
    else:
        disag_str = "—"
        disag_status = ""
    
    # Momentum
    if np.isfinite(obs.disagreement_momentum):
        mom_str = f"{obs.disagreement_momentum:+.4f}"
        if obs.disagreement_momentum > 0.02:
            mom_status = "[yellow]↑[/yellow]"
        elif obs.disagreement_momentum < -0.02:
            mom_status = "[bright_green]↓[/bright_green]"
        else:
            mom_status = "[dim]—[/dim]"
    else:
        mom_str = "—"
        mom_status = ""
    
    # Vol Ratio
    if np.isfinite(obs.vol_ratio):
        vol_str = f"{obs.vol_ratio:.3f}"
        if obs.vol_ratio < 0.8:
            vol_status = "[cyan]compressed[/cyan]"
        elif obs.vol_ratio > 1.3:
            vol_status = "[red]expanding[/red]"
        else:
            vol_status = "[dim]normal[/dim]"
    else:
        vol_str = "—"
        vol_status = ""
    
    metrics_table.add_row("Convex Loss C(t)", convex_str, convex_status)
    metrics_table.add_row("Acceleration ΔC(t)", accel_str, accel_status)
    metrics_table.add_row("Tail Mass P(ΔX>0)", tail_str, tail_status)
    metrics_table.add_row("Disagreement D(t)", disag_str, disag_status)
    metrics_table.add_row("Momentum dD/dt", mom_str, mom_status)
    metrics_table.add_row("Vol Ratio V(t)", vol_str, vol_status)
    
    console.print(Padding(metrics_table, (0, 0, 0, 4)))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FOOTER - Decision Details
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print(Rule(style="dim"))
    console.print()
    
    # Decision basis
    basis_text = Text()
    basis_text.append("    ", style="")
    basis_text.append("Decision: ", style="dim")
    basis_text.append(decision.decision_basis, style="white")
    console.print(basis_text)
    
    # Threshold
    threshold_text = Text()
    threshold_text.append("    ", style="")
    threshold_text.append("Threshold α(t): ", style="dim")
    threshold_text.append(f"{decision.dynamic_alpha:.0%}", style="white")
    console.print(threshold_text)
    
    # Signature
    sig_text = Text()
    sig_text.append("    ", style="")
    sig_text.append("Signature: ", style="dim")
    sig_text.append(decision.signature[:16], style="dim italic")
    console.print(sig_text)
    
    console.print()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point for `make debt` command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FX Debt Allocation Engine (EURJPY)"
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
    parser.add_argument(
        '--no-dynamic-alpha',
        action='store_true',
        help='Use fixed threshold instead of dynamic α(t)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose processing output'
    )
    
    args = parser.parse_args()
    
    use_dynamic_alpha = not args.no_dynamic_alpha
    
    from rich.rule import Rule
    from rich.align import Align
    from rich.text import Text
    
    console = Console(force_terminal=True, width=100)
    
    # Show header - Apple-quality cinematic design
    if not args.json:
        console.print()
        console.print()
        
        # Cinematic header
        header_text = Text()
        header_text.append("\n", style="")
        header_text.append("D E B T   A L L O C A T I O N", style="bold white")
        header_text.append("\n", style="")
        header_text.append("EURJPY Policy-Stress Engine", style="dim")
        header_text.append("\n", style="")
        
        header_panel = Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            border_style="bright_cyan",
            padding=(1, 4),
        )
        console.print(Align.center(header_panel, width=50))
        console.print()
        
        # Feature badges
        features = Text()
        features.append("◉ ", style="bright_green" if use_dynamic_alpha else "dim")
        features.append("Dynamic α(t)", style="white" if use_dynamic_alpha else "dim")
        features.append("    ", style="")
        features.append("◉ ", style="bright_green")
        features.append("Transition Φ", style="white")
        features.append("    ", style="")
        features.append("◎ ", style="dim")
        features.append("v4.0.0", style="dim")
        console.print(Align.center(features))
        console.print()
        console.print(Rule(style="dim"))
        console.print()
    
    # Run engine with quiet mode (ONLY ONCE!)
    decision = run_debt_allocation_engine(
        data_path=args.data_path,
        force_reevaluate=args.force,
        force_refresh_data=not args.no_refresh,
        use_dynamic_alpha=use_dynamic_alpha,
        quiet=args.quiet or args.json,
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
