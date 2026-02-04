"""
===============================================================================
ONLINE BAYESIAN PARAMETER UPDATES — Sequential Monte Carlo for Adaptive Signals
===============================================================================

This module implements Online Bayesian Parameter Updates as recommended by the
Chinese Staff Professor Panel (February 2026).

PROFESSOR LIU XIAOMING (Score: 9/10):
    "The current architecture's fundamental limitation is its batch nature.
     Parameters are estimated offline, cached, and used until the next tuning run.
     Markets evolve continuously—volatility clusters, correlations break down,
     regime transitions occur mid-day. Online updating transforms the Kalman filter
     from a static estimator to a living, adaptive system."

-------------------------------------------------------------------------------
ARCHITECTURE
-------------------------------------------------------------------------------

The online update layer sits between tune.py (batch priors) and signals.py
(real-time consumption):

    tune.py → Batch Priors (q, c, φ, ν)
                    ↓
    online_update.py → Particle Filter Updates
                    ↓
    signals.py → Time-Varying Parameters

-------------------------------------------------------------------------------
CORE ALGORITHM: Rao-Blackwellized Particle Filter (RBPF)
-------------------------------------------------------------------------------

For the state-space model:
    x_t = φ·x_{t-1} + w_t,  w_t ~ N(0, q)       # Latent drift state
    y_t = x_t + v_t,        v_t ~ t_ν(0, c·σ_t²) # Observation

We use Rao-Blackwellization to marginalize the linear state x_t:
    - Particles track: θ = (q, c, φ, ν)          # Non-linear parameters
    - Kalman filter tracks: p(x_t | y_{1:t}, θ)  # Linear state (marginalized)

This reduces variance compared to naive particle filtering.

-------------------------------------------------------------------------------
KEY FEATURES
-------------------------------------------------------------------------------

1. PARTICLE-BASED POSTERIOR:
   - Maintains N particles, each with (θ, weight)
   - Resampling when ESS < N/2

2. ANCHORED TO BATCH PRIORS:
   - Random walk proposals centered on batch estimates
   - Prevents runaway adaptation

3. PIT-TRIGGERED ACCELERATION:
   - When streaming PIT degrades, increase proposal variance
   - Enables faster adaptation during regime transitions

4. AUDIT TRAIL:
   - Full history of parameter trajectories
   - Regulatory compliance

5. GRACEFUL FALLBACK:
   - Returns cached parameters if online estimation becomes unstable

-------------------------------------------------------------------------------
COMPUTATIONAL BUDGET
-------------------------------------------------------------------------------

Target: < 10ms per asset per observation (Acceptance Criteria #5)

Achieved via:
    - Sufficient statistics (no full history storage)
    - Vectorized Kalman updates
    - Systematic resampling (O(N) not O(N log N))

===============================================================================
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import os
import warnings

# Numerical stability constants
EPS = 1e-12
LOG_EPS = -27.6  # log(1e-12)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OnlineUpdateConfig:
    """Configuration for online Bayesian parameter updates."""
    
    # Number of particles
    n_particles: int = 100
    
    # Effective sample size threshold for resampling (fraction of n_particles)
    ess_threshold_fraction: float = 0.5
    
    # Random walk proposal standard deviations (as fraction of parameter value)
    proposal_std_q: float = 0.05      # Process noise
    proposal_std_c: float = 0.03      # Observation noise multiplier
    proposal_std_phi: float = 0.02    # AR(1) coefficient
    proposal_std_nu: float = 0.05     # Student-t degrees of freedom
    
    # Anchoring strength to batch priors (0 = no anchoring, 1 = full anchoring)
    batch_anchor_strength: float = 0.3
    
    # PIT-triggered acceleration
    pit_acceleration_threshold: float = 0.05  # KS p-value threshold
    pit_acceleration_factor: float = 2.0      # Multiply proposal std by this
    
    # Stability bounds
    q_min: float = 1e-12
    q_max: float = 1e-2
    c_min: float = 0.3
    c_max: float = 3.0
    phi_min: float = -0.99
    phi_max: float = 0.99
    nu_min: float = 3.0
    nu_max: float = 100.0
    
    # Convergence monitoring
    convergence_window: int = 50  # Observations for convergence check
    convergence_threshold: float = 0.01  # Parameter change threshold
    
    # Fallback settings
    max_unstable_steps: int = 10  # Consecutive unstable steps before fallback
    min_ess_for_validity: float = 5.0  # Minimum ESS to consider valid
    
    # Audit trail
    enable_audit_trail: bool = True
    max_audit_history: int = 1000  # Rolling window for audit


@dataclass
class ParticleState:
    """State of a single particle in the SMC sampler."""
    
    # Parameters
    q: float
    c: float
    phi: float
    nu: float
    
    # Kalman filter sufficient statistics (Rao-Blackwellized)
    mu_filtered: float = 0.0  # E[x_t | y_{1:t}]
    P_filtered: float = 1.0   # Var[x_t | y_{1:t}]
    
    # Log-weight (unnormalized)
    log_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "q": self.q,
            "c": self.c,
            "phi": self.phi,
            "nu": self.nu,
            "mu_filtered": self.mu_filtered,
            "P_filtered": self.P_filtered,
            "log_weight": self.log_weight,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'ParticleState':
        return cls(
            q=d["q"],
            c=d["c"],
            phi=d["phi"],
            nu=d["nu"],
            mu_filtered=d.get("mu_filtered", 0.0),
            P_filtered=d.get("P_filtered", 1.0),
            log_weight=d.get("log_weight", 0.0),
        )


@dataclass
class OnlineUpdateResult:
    """Result of an online update step."""
    
    # Posterior mean parameters
    q_mean: float
    c_mean: float
    phi_mean: float
    nu_mean: float
    
    # Posterior standard deviations (uncertainty)
    q_std: float
    c_std: float
    phi_std: float
    nu_std: float
    
    # Diagnostics
    effective_sample_size: float
    resampled: bool
    step_count: int
    timestamp: str
    
    # Stability indicators
    is_stable: bool = True
    fallback_to_batch: bool = False
    
    # PIT-based acceleration active
    pit_acceleration_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_mean": self.q_mean,
            "c_mean": self.c_mean,
            "phi_mean": self.phi_mean,
            "nu_mean": self.nu_mean,
            "q_std": self.q_std,
            "c_std": self.c_std,
            "phi_std": self.phi_std,
            "nu_std": self.nu_std,
            "effective_sample_size": self.effective_sample_size,
            "resampled": self.resampled,
            "step_count": self.step_count,
            "timestamp": self.timestamp,
            "is_stable": self.is_stable,
            "fallback_to_batch": self.fallback_to_batch,
            "pit_acceleration_active": self.pit_acceleration_active,
        }


@dataclass
class AuditRecord:
    """Single entry in the parameter trajectory audit trail."""
    
    timestamp: str
    step: int
    observation: float
    volatility: float
    
    # Parameter trajectory
    q_mean: float
    c_mean: float
    phi_mean: float
    nu_mean: float
    
    # Diagnostics
    ess: float
    resampled: bool
    pit_acceleration: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "observation": self.observation,
            "volatility": self.volatility,
            "q_mean": self.q_mean,
            "c_mean": self.c_mean,
            "phi_mean": self.phi_mean,
            "nu_mean": self.nu_mean,
            "ess": self.ess,
            "resampled": self.resampled,
            "pit_acceleration": self.pit_acceleration,
        }


# =============================================================================
# STUDENT-T LOG-PDF (Vectorized)
# =============================================================================

def student_t_logpdf(x: np.ndarray, nu: float, mu: float, scale: float) -> np.ndarray:
    """
    Vectorized Student-t log-pdf.
    
    Args:
        x: Observations (array)
        nu: Degrees of freedom
        mu: Location
        scale: Scale parameter
        
    Returns:
        Log-pdf values
    """
    from scipy.special import gammaln
    
    if scale <= 0 or nu <= 0:
        return np.full_like(x, -1e12)
    
    z = (x - mu) / scale
    log_norm = (
        gammaln((nu + 1.0) / 2.0) 
        - gammaln(nu / 2.0) 
        - 0.5 * np.log(nu * np.pi * (scale ** 2))
    )
    log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
    
    return log_norm + log_kernel


def gaussian_logpdf(x: float, mu: float, var: float) -> float:
    """Gaussian log-pdf for single observation."""
    if var <= 0:
        return -1e12
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


# =============================================================================
# ONLINE BAYESIAN UPDATER
# =============================================================================

class OnlineBayesianUpdater:
    """
    Sequential Monte Carlo updater for Kalman filter parameters.
    
    This class maintains a particle approximation to the posterior:
        p(q, c, φ, ν | y_{1:t})
    
    and updates it incrementally as new observations arrive.
    
    Usage:
        # Initialize from batch-tuned parameters
        updater = OnlineBayesianUpdater.from_batch_params(tuned_params)
        
        # Update with new observation
        result = updater.update(y_t, sigma_t)
        
        # Get current parameter estimates
        q, c, phi, nu = updater.get_current_params()
    """
    
    def __init__(
        self,
        batch_params: Dict[str, float],
        config: Optional[OnlineUpdateConfig] = None,
    ):
        """
        Initialize the online updater.
        
        Args:
            batch_params: Batch-estimated parameters from tune.py
                Expected keys: q, c, phi, nu
            config: Configuration for online updates
        """
        self.config = config or OnlineUpdateConfig()
        
        # Random state for reproducibility - MUST be initialized FIRST
        # because _initialize_particles() calls _sample_truncated_normal() which uses self.rng
        self.rng = np.random.default_rng(42)
        self.batch_params = batch_params
        
        # Extract batch priors
        self.q_batch = float(batch_params.get("q", 1e-6))
        self.c_batch = float(batch_params.get("c", 1.0))
        self.phi_batch = float(batch_params.get("phi", 0.0) or 0.0)
        self.nu_batch = float(batch_params.get("nu", 8.0) or 8.0)
        
        # Initialize particles around batch estimates
        self.particles: List[ParticleState] = self._initialize_particles()
        
        # State tracking
        self.step_count = 0
        self.consecutive_unstable = 0
        self.using_fallback = False
        
        # Recent PIT values for acceleration trigger
        self.recent_pit_values: List[float] = []
        self.pit_acceleration_active = False
        
        # Audit trail
        self.audit_trail: List[AuditRecord] = []
        
    
    def _initialize_particles(self) -> List[ParticleState]:
        """Initialize particles around batch estimates with small perturbations."""
        particles = []
        n = self.config.n_particles
        
        for i in range(n):
            # Sample from prior centered on batch estimates
            # Use truncated normal to respect bounds
            q = self._sample_truncated_normal(
                self.q_batch, 
                self.q_batch * 0.1,
                self.config.q_min,
                self.config.q_max
            )
            c = self._sample_truncated_normal(
                self.c_batch,
                self.c_batch * 0.1,
                self.config.c_min,
                self.config.c_max
            )
            phi = self._sample_truncated_normal(
                self.phi_batch,
                0.1,
                self.config.phi_min,
                self.config.phi_max
            )
            nu = self._sample_truncated_normal(
                self.nu_batch,
                self.nu_batch * 0.1,
                self.config.nu_min,
                self.config.nu_max
            )
            
            particles.append(ParticleState(
                q=q, c=c, phi=phi, nu=nu,
                log_weight=0.0  # Uniform initial weights
            ))
        
        return particles
    
    def _sample_truncated_normal(
        self, 
        mean: float, 
        std: float, 
        lower: float, 
        upper: float
    ) -> float:
        """Sample from truncated normal distribution."""
        for _ in range(100):  # Rejection sampling with limit
            sample = self.rng.normal(mean, std)
            if lower <= sample <= upper:
                return sample
        # Fallback: clip to bounds
        return np.clip(self.rng.normal(mean, std), lower, upper)
    
    def update(
        self,
        y: float,
        sigma: float,
        pit_pvalue: Optional[float] = None,
    ) -> OnlineUpdateResult:
        """
        Update parameter posterior with a new observation.
        
        This implements one step of the Sequential Monte Carlo algorithm:
        1. Propagate particles (random walk proposal)
        2. Compute likelihood weights
        3. Normalize weights
        4. Resample if ESS < threshold
        
        Args:
            y: New observation (return)
            sigma: Observation volatility (EWMA or GARCH)
            pit_pvalue: Optional PIT KS p-value for acceleration trigger
            
        Returns:
            OnlineUpdateResult with posterior statistics
        """
        self.step_count += 1
        timestamp = datetime.now().isoformat()
        
        # Check for NaN/Inf inputs
        if not np.isfinite(y) or not np.isfinite(sigma) or sigma <= 0:
            return self._create_fallback_result(timestamp, "invalid_input")
        
        # Check PIT-triggered acceleration
        if pit_pvalue is not None:
            self.recent_pit_values.append(pit_pvalue)
            if len(self.recent_pit_values) > 20:
                self.recent_pit_values.pop(0)
            
            # Activate acceleration if recent PIT is poor
            recent_mean_pit = np.mean(self.recent_pit_values[-5:]) if len(self.recent_pit_values) >= 5 else 1.0
            self.pit_acceleration_active = recent_mean_pit < self.config.pit_acceleration_threshold
        
        # Determine proposal standard deviations
        accel = self.config.pit_acceleration_factor if self.pit_acceleration_active else 1.0
        
        # Step 1: Propagate particles
        new_particles = []
        log_weights = []
        
        for particle in self.particles:
            # Random walk proposal with anchoring
            new_q, new_c, new_phi, new_nu = self._propose_parameters(particle, accel)
            
            # Kalman filter prediction step
            mu_pred = new_phi * particle.mu_filtered
            P_pred = (new_phi ** 2) * particle.P_filtered + new_q
            
            # Observation variance
            R = new_c * (sigma ** 2)
            
            # Innovation
            innovation = y - mu_pred
            S = P_pred + R
            
            # Compute log-likelihood
            if new_nu > 30:
                # Approximate with Gaussian for large nu
                log_lik = gaussian_logpdf(y, mu_pred, S)
            else:
                # Student-t likelihood
                scale = np.sqrt(S * (new_nu - 2) / new_nu) if new_nu > 2 else np.sqrt(S)
                log_lik = float(student_t_logpdf(np.array([y]), new_nu, mu_pred, scale)[0])
            
            # Kalman update
            K = P_pred / max(S, EPS)
            mu_updated = mu_pred + K * innovation
            P_updated = max((1 - K) * P_pred, EPS)
            
            # Create new particle
            new_particle = ParticleState(
                q=new_q,
                c=new_c,
                phi=new_phi,
                nu=new_nu,
                mu_filtered=mu_updated,
                P_filtered=P_updated,
                log_weight=particle.log_weight + log_lik
            )
            new_particles.append(new_particle)
            log_weights.append(new_particle.log_weight)
        
        self.particles = new_particles
        
        # Step 2: Normalize weights
        log_weights = np.array(log_weights)
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        weights_sum = np.sum(weights)
        
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            return self._create_fallback_result(timestamp, "weight_degeneracy")
        
        normalized_weights = weights / weights_sum
        
        # Step 3: Compute ESS
        ess = 1.0 / np.sum(normalized_weights ** 2)
        
        # Check stability
        if ess < self.config.min_ess_for_validity:
            self.consecutive_unstable += 1
            if self.consecutive_unstable >= self.config.max_unstable_steps:
                self.using_fallback = True
                return self._create_fallback_result(timestamp, "ess_collapse")
        else:
            self.consecutive_unstable = 0
            self.using_fallback = False
        
        # Step 4: Resample if needed
        resampled = False
        if ess < self.config.n_particles * self.config.ess_threshold_fraction:
            self.particles = self._systematic_resample(normalized_weights)
            resampled = True
            # Reset weights after resampling
            for p in self.particles:
                p.log_weight = 0.0
        
        # Compute posterior statistics
        result = self._compute_posterior_statistics(normalized_weights, ess, resampled, timestamp)
        
        # Audit trail
        if self.config.enable_audit_trail:
            self._record_audit(y, sigma, result)
        
        return result
    
    def _propose_parameters(
        self, 
        particle: ParticleState, 
        acceleration_factor: float
    ) -> Tuple[float, float, float, float]:
        """
        Propose new parameters via anchored random walk.
        
        The proposal combines:
        1. Random walk from current particle
        2. Pull toward batch priors (anchoring)
        """
        cfg = self.config
        alpha = cfg.batch_anchor_strength
        
        # Anchored mean: blend of particle value and batch prior
        q_anchor = alpha * self.q_batch + (1 - alpha) * particle.q
        c_anchor = alpha * self.c_batch + (1 - alpha) * particle.c
        phi_anchor = alpha * self.phi_batch + (1 - alpha) * particle.phi
        nu_anchor = alpha * self.nu_batch + (1 - alpha) * particle.nu
        
        # Proposal standard deviations (scaled by acceleration)
        std_q = cfg.proposal_std_q * self.q_batch * acceleration_factor
        std_c = cfg.proposal_std_c * self.c_batch * acceleration_factor
        std_phi = cfg.proposal_std_phi * acceleration_factor
        std_nu = cfg.proposal_std_nu * self.nu_batch * acceleration_factor
        
        # Sample new parameters
        new_q = np.clip(
            self.rng.normal(q_anchor, std_q),
            cfg.q_min, cfg.q_max
        )
        new_c = np.clip(
            self.rng.normal(c_anchor, std_c),
            cfg.c_min, cfg.c_max
        )
        new_phi = np.clip(
            self.rng.normal(phi_anchor, std_phi),
            cfg.phi_min, cfg.phi_max
        )
        new_nu = np.clip(
            self.rng.normal(nu_anchor, std_nu),
            cfg.nu_min, cfg.nu_max
        )
        
        return new_q, new_c, new_phi, new_nu
    
    def _systematic_resample(self, weights: np.ndarray) -> List[ParticleState]:
        """
        Systematic resampling (O(N) complexity).
        
        This is more efficient than multinomial resampling and produces
        lower variance estimates.
        """
        n = len(weights)
        positions = (self.rng.random() + np.arange(n)) / n
        
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Ensure exact sum
        
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)
        
        # Create new particles (deep copy)
        new_particles = []
        for idx in indices:
            old = self.particles[idx]
            new_particles.append(ParticleState(
                q=old.q,
                c=old.c,
                phi=old.phi,
                nu=old.nu,
                mu_filtered=old.mu_filtered,
                P_filtered=old.P_filtered,
                log_weight=0.0  # Reset weights after resampling
            ))
        
        return new_particles
    
    def _compute_posterior_statistics(
        self,
        weights: np.ndarray,
        ess: float,
        resampled: bool,
        timestamp: str
    ) -> OnlineUpdateResult:
        """Compute weighted mean and std of parameters."""
        # Extract parameter arrays
        q_vals = np.array([p.q for p in self.particles])
        c_vals = np.array([p.c for p in self.particles])
        phi_vals = np.array([p.phi for p in self.particles])
        nu_vals = np.array([p.nu for p in self.particles])
        
        # Weighted means
        q_mean = np.sum(weights * q_vals)
        c_mean = np.sum(weights * c_vals)
        phi_mean = np.sum(weights * phi_vals)
        nu_mean = np.sum(weights * nu_vals)
        
        # Weighted standard deviations
        q_std = np.sqrt(np.sum(weights * (q_vals - q_mean) ** 2))
        c_std = np.sqrt(np.sum(weights * (c_vals - c_mean) ** 2))
        phi_std = np.sqrt(np.sum(weights * (phi_vals - phi_mean) ** 2))
        nu_std = np.sqrt(np.sum(weights * (nu_vals - nu_mean) ** 2))
        
        return OnlineUpdateResult(
            q_mean=float(q_mean),
            c_mean=float(c_mean),
            phi_mean=float(phi_mean),
            nu_mean=float(nu_mean),
            q_std=float(q_std),
            c_std=float(c_std),
            phi_std=float(phi_std),
            nu_std=float(nu_std),
            effective_sample_size=float(ess),
            resampled=resampled,
            step_count=self.step_count,
            timestamp=timestamp,
            is_stable=True,
            fallback_to_batch=False,
            pit_acceleration_active=self.pit_acceleration_active,
        )
    
    def _create_fallback_result(self, timestamp: str, reason: str) -> OnlineUpdateResult:
        """Create result using batch parameters as fallback."""
        return OnlineUpdateResult(
            q_mean=self.q_batch,
            c_mean=self.c_batch,
            phi_mean=self.phi_batch,
            nu_mean=self.nu_batch,
            q_std=0.0,
            c_std=0.0,
            phi_std=0.0,
            nu_std=0.0,
            effective_sample_size=0.0,
            resampled=False,
            step_count=self.step_count,
            timestamp=timestamp,
            is_stable=False,
            fallback_to_batch=True,
            pit_acceleration_active=self.pit_acceleration_active,
        )
    
    def _record_audit(self, y: float, sigma: float, result: OnlineUpdateResult) -> None:
        """Record audit entry for regulatory compliance."""
        record = AuditRecord(
            timestamp=result.timestamp,
            step=result.step_count,
            observation=y,
            volatility=sigma,
            q_mean=result.q_mean,
            c_mean=result.c_mean,
            phi_mean=result.phi_mean,
            nu_mean=result.nu_mean,
            ess=result.effective_sample_size,
            resampled=result.resampled,
            pit_acceleration=result.pit_acceleration_active,
        )
        
        self.audit_trail.append(record)
        
        # Rolling window
        if len(self.audit_trail) > self.config.max_audit_history:
            self.audit_trail.pop(0)
    
    def get_current_params(self) -> Dict[str, float]:
        """
        Get current parameter estimates for use in signal generation.
        
        Returns:
            Dictionary with q, c, phi, nu
        """
        if self.using_fallback or not self.particles:
            return {
                "q": self.q_batch,
                "c": self.c_batch,
                "phi": self.phi_batch,
                "nu": self.nu_batch,
                "online_updated": False,
            }
        
        # Compute current weighted means using log-sum-exp for numerical stability
        log_weights = np.array([p.log_weight for p in self.particles])
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        weights_sum = np.sum(weights)
        
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            weights = np.ones(len(self.particles)) / len(self.particles)
        else:
            weights = weights / weights_sum
        
        return {
            "q": float(np.sum(weights * np.array([p.q for p in self.particles]))),
            "c": float(np.sum(weights * np.array([p.c for p in self.particles]))),
            "phi": float(np.sum(weights * np.array([p.phi for p in self.particles]))),
            "nu": float(np.sum(weights * np.array([p.nu for p in self.particles]))),
            "online_updated": True,
        }
    
    def get_kalman_state(self) -> Dict[str, float]:
        """
        Get current Kalman filter state (marginalized over parameters).
        
        Returns weighted average of filtered state across particles.
        """
        if not self.particles:
            return {"mu_filtered": 0.0, "P_filtered": 1.0}
        
        # Use log-sum-exp for numerical stability
        log_weights = np.array([p.log_weight for p in self.particles])
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        weights_sum = np.sum(weights)
        
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            weights = np.ones(len(self.particles)) / len(self.particles)
        else:
            weights = weights / weights_sum
        
        mu = float(np.sum(weights * np.array([p.mu_filtered for p in self.particles])))
        P = float(np.sum(weights * np.array([p.P_filtered for p in self.particles])))
        
        return {"mu_filtered": mu, "P_filtered": P}
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail for regulatory compliance."""
        return [r.to_dict() for r in self.audit_trail]
    
    def reset_to_batch(self) -> None:
        """Reset to batch parameters (e.g., after regime change)."""
        self.particles = self._initialize_particles()
        self.step_count = 0
        self.consecutive_unstable = 0
        self.using_fallback = False
        self.recent_pit_values = []
        self.pit_acceleration_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize updater state for persistence."""
        return {
            "batch_params": self.batch_params,
            "config": {
                "n_particles": self.config.n_particles,
                "ess_threshold_fraction": self.config.ess_threshold_fraction,
                "batch_anchor_strength": self.config.batch_anchor_strength,
            },
            "particles": [p.to_dict() for p in self.particles],
            "step_count": self.step_count,
            "consecutive_unstable": self.consecutive_unstable,
            "using_fallback": self.using_fallback,
            "pit_acceleration_active": self.pit_acceleration_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnlineBayesianUpdater':
        """Deserialize updater state."""
        config = OnlineUpdateConfig(**data.get("config", {}))
        updater = cls(data["batch_params"], config)
        
        if "particles" in data:
            updater.particles = [ParticleState.from_dict(p) for p in data["particles"]]
        
        updater.step_count = data.get("step_count", 0)
        updater.consecutive_unstable = data.get("consecutive_unstable", 0)
        updater.using_fallback = data.get("using_fallback", False)
        updater.pit_acceleration_active = data.get("pit_acceleration_active", False)
        
        return updater
    
    @classmethod
    def from_batch_params(
        cls,
        tuned_params: Dict[str, Any],
        config: Optional[OnlineUpdateConfig] = None,
    ) -> 'OnlineBayesianUpdater':
        """
        Create updater from tune.py output structure.
        
        Handles both flat and BMA nested structures.
        """
        # Extract parameters from BMA structure
        if "global" in tuned_params:
            global_data = tuned_params["global"]
            
            # Get best model parameters
            models = global_data.get("models", {})
            model_posterior = global_data.get("model_posterior", {})
            
            if model_posterior:
                best_model = max(model_posterior.keys(), key=lambda m: model_posterior.get(m, 0))
                params = models.get(best_model, {})
            else:
                # Fallback to first available model
                params = next(iter(models.values()), {}) if models else global_data
            
            batch_params = {
                "q": params.get("q", global_data.get("q", 1e-6)),
                "c": params.get("c", global_data.get("c", 1.0)),
                "phi": params.get("phi", global_data.get("phi", 0.0)),
                "nu": params.get("nu", global_data.get("nu", 8.0)),
            }
        else:
            # Flat structure (legacy)
            batch_params = {
                "q": tuned_params.get("q", 1e-6),
                "c": tuned_params.get("c", 1.0),
                "phi": tuned_params.get("phi", 0.0),
                "nu": tuned_params.get("nu", 8.0),
            }
        
        return cls(batch_params, config)


# =============================================================================
# CACHE FOR ONLINE UPDATERS (Per-Asset)
# =============================================================================

_ONLINE_UPDATER_CACHE: Dict[str, OnlineBayesianUpdater] = {}


def get_or_create_updater(
    asset: str,
    tuned_params: Dict[str, Any],
    config: Optional[OnlineUpdateConfig] = None,
) -> OnlineBayesianUpdater:
    """
    Get or create an online updater for an asset.
    
    Maintains a cache of updaters to preserve state across calls.
    
    Args:
        asset: Asset symbol
        tuned_params: Batch-tuned parameters from tune.py
        config: Optional configuration
        
    Returns:
        OnlineBayesianUpdater instance
    """
    global _ONLINE_UPDATER_CACHE
    
    if asset not in _ONLINE_UPDATER_CACHE:
        _ONLINE_UPDATER_CACHE[asset] = OnlineBayesianUpdater.from_batch_params(
            tuned_params, config
        )
    
    return _ONLINE_UPDATER_CACHE[asset]


def clear_updater_cache(asset: Optional[str] = None) -> None:
    """Clear online updater cache."""
    global _ONLINE_UPDATER_CACHE
    
    if asset is None:
        _ONLINE_UPDATER_CACHE.clear()
    elif asset in _ONLINE_UPDATER_CACHE:
        del _ONLINE_UPDATER_CACHE[asset]


def get_online_params(
    asset: str,
    tuned_params: Dict[str, Any],
    y: float,
    sigma: float,
    pit_pvalue: Optional[float] = None,
    config: Optional[OnlineUpdateConfig] = None,
) -> Dict[str, float]:
    """
    Convenience function: update and get current parameters.
    
    This is the main entry point for signals.py integration.
    
    Args:
        asset: Asset symbol
        tuned_params: Batch-tuned parameters
        y: Current observation (return)
        sigma: Current volatility
        pit_pvalue: Optional PIT p-value for acceleration
        config: Optional configuration
        
    Returns:
        Dictionary with current parameter estimates
    """
    updater = get_or_create_updater(asset, tuned_params, config)
    result = updater.update(y, sigma, pit_pvalue)
    return updater.get_current_params()


# =============================================================================
# INTEGRATION HELPER FOR SIGNALS.PY
# =============================================================================

def compute_adaptive_kalman_params(
    asset: str,
    returns: np.ndarray,
    volatility: np.ndarray,
    tuned_params: Dict[str, Any],
    enable_online: bool = True,
    config: Optional[OnlineUpdateConfig] = None,
) -> Dict[str, Any]:
    """
    Compute adaptive Kalman parameters using online updates.
    
    This function processes a history of returns and volatility to produce
    time-varying parameters that adapt to recent market conditions.
    
    Args:
        asset: Asset symbol
        returns: Array of historical returns
        volatility: Array of historical volatility
        tuned_params: Batch-tuned parameters from tune.py
        enable_online: Whether to use online updates (False = use batch only)
        config: Online update configuration
        
    Returns:
        Dictionary with:
        - 'current_params': Current parameter estimates
        - 'kalman_state': Current filtered state
        - 'online_active': Whether online updates were applied
        - 'update_result': Last OnlineUpdateResult
    """
    if not enable_online:
        # Return batch parameters directly
        if "global" in tuned_params:
            global_data = tuned_params["global"]
            return {
                "current_params": {
                    "q": global_data.get("q", 1e-6),
                    "c": global_data.get("c", 1.0),
                    "phi": global_data.get("phi", 0.0),
                    "nu": global_data.get("nu", 8.0),
                    "online_updated": False,
                },
                "kalman_state": {"mu_filtered": 0.0, "P_filtered": 1.0},
                "online_active": False,
                "update_result": None,
            }
        else:
            return {
                "current_params": {
                    "q": tuned_params.get("q", 1e-6),
                    "c": tuned_params.get("c", 1.0),
                    "phi": tuned_params.get("phi", 0.0),
                    "nu": tuned_params.get("nu", 8.0),
                    "online_updated": False,
                },
                "kalman_state": {"mu_filtered": 0.0, "P_filtered": 1.0},
                "online_active": False,
                "update_result": None,
            }
    
    # Get or create updater
    updater = get_or_create_updater(asset, tuned_params, config)
    
    # Process observations
    n = min(len(returns), len(volatility))
    last_result = None
    
    for i in range(n):
        y = float(returns[i])
        sigma = float(volatility[i])
        
        if np.isfinite(y) and np.isfinite(sigma) and sigma > 0:
            last_result = updater.update(y, sigma)
    
    return {
        "current_params": updater.get_current_params(),
        "kalman_state": updater.get_kalman_state(),
        "online_active": True,
        "update_result": last_result.to_dict() if last_result else None,
    }


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_ONLINE_CONFIG = OnlineUpdateConfig(
    n_particles=100,
    ess_threshold_fraction=0.5,
    proposal_std_q=0.05,
    proposal_std_c=0.03,
    proposal_std_phi=0.02,
    proposal_std_nu=0.05,
    batch_anchor_strength=0.3,
    pit_acceleration_threshold=0.05,
    pit_acceleration_factor=2.0,
    enable_audit_trail=True,
)


# =============================================================================
# PERSISTENCE — Save/Load Online Update State to Disk
# =============================================================================
# Online update state is persisted to src/data/online_update/ for:
#   1. Faster restarts (no need to re-process history)
#   2. Continuity across sessions
#   3. Audit trail preservation
# =============================================================================

import pathlib

# Persistence directory
ONLINE_UPDATE_CACHE_DIR = pathlib.Path(__file__).parent.parent / "data" / "online_update"


def _ensure_cache_dir() -> pathlib.Path:
    """Ensure the online update cache directory exists."""
    ONLINE_UPDATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ONLINE_UPDATE_CACHE_DIR


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol for filesystem-safe filenames."""
    normalized = symbol.strip().upper()
    for char in ["-", "=", "/", ".", "^", " ", ":"]:
        normalized = normalized.replace(char, "_")
    normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
    if normalized and normalized[0].isdigit():
        normalized = "_" + normalized
    return normalized


def _get_state_path(symbol: str) -> pathlib.Path:
    """Get the state file path for a symbol."""
    return _ensure_cache_dir() / f"{_normalize_symbol(symbol)}_online.json"


def save_updater_state(symbol: str, updater: OnlineBayesianUpdater) -> str:
    """
    Save online updater state to disk.
    
    Args:
        symbol: Asset symbol
        updater: OnlineBayesianUpdater instance
        
    Returns:
        Path to saved state file
    """
    state_path = _get_state_path(symbol)
    state_data = {
        "symbol": symbol,
        "normalized_symbol": _normalize_symbol(symbol),
        "saved_at": datetime.now().isoformat(),
        "updater_state": updater.to_dict(),
    }
    
    # Atomic write
    temp_path = state_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(state_data, f, indent=2, default=_json_serializer)
        temp_path.replace(state_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e
    
    return str(state_path)


def load_updater_state(symbol: str) -> Optional[OnlineBayesianUpdater]:
    """
    Load online updater state from disk.
    
    Args:
        symbol: Asset symbol
        
    Returns:
        OnlineBayesianUpdater instance or None if not found
    """
    state_path = _get_state_path(symbol)
    if not state_path.exists():
        return None
    
    try:
        with open(state_path, "r") as f:
            state_data = json.load(f)
        
        updater_state = state_data.get("updater_state")
        if updater_state:
            return OnlineBayesianUpdater.from_dict(updater_state)
    except Exception as e:
        warnings.warn(f"Failed to load online state for {symbol}: {e}")
    
    return None


def delete_updater_state(symbol: str) -> bool:
    """Delete persisted state for a symbol."""
    state_path = _get_state_path(symbol)
    if state_path.exists():
        state_path.unlink()
        return True
    return False


def list_persisted_symbols() -> List[str]:
    """List all symbols with persisted online update state."""
    _ensure_cache_dir()
    symbols = []
    for path in ONLINE_UPDATE_CACHE_DIR.glob("*_online.json"):
        # Extract symbol from filename
        name = path.stem.replace("_online", "")
        symbols.append(name)
    return sorted(symbols)


def get_persistence_stats() -> Dict[str, Any]:
    """Get statistics about persisted online update states."""
    _ensure_cache_dir()
    files = list(ONLINE_UPDATE_CACHE_DIR.glob("*_online.json"))
    total_size = sum(f.stat().st_size for f in files)
    
    return {
        "n_symbols": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(ONLINE_UPDATE_CACHE_DIR),
    }


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# BULK PROCESSING — Process Multiple Assets in Parallel
# =============================================================================
# For efficiency, process multiple assets simultaneously using:
#   1. Thread pool for I/O-bound operations
#   2. Batch loading of tuned parameters
#   3. Parallel particle filter updates
# =============================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


@dataclass
class BulkUpdateResult:
    """Result of bulk online update processing."""
    
    symbol: str
    success: bool
    online_params: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    update_count: int = 0
    final_ess: float = 0.0
    persisted: bool = False


def process_asset_online_update(
    symbol: str,
    returns: np.ndarray,
    volatility: np.ndarray,
    tuned_params: Dict[str, Any],
    config: Optional[OnlineUpdateConfig] = None,
    persist: bool = True,
    load_existing: bool = True,
) -> BulkUpdateResult:
    """
    Process online updates for a single asset.
    
    Args:
        symbol: Asset symbol
        returns: Array of returns
        volatility: Array of volatility values
        tuned_params: Batch-tuned parameters
        config: Online update configuration
        persist: Whether to persist state to disk
        load_existing: Whether to load existing state from disk
        
    Returns:
        BulkUpdateResult with processing outcome
    """
    try:
        # Try to load existing state
        updater = None
        if load_existing:
            updater = load_updater_state(symbol)
        
        # Create new updater if not loaded
        if updater is None:
            updater = OnlineBayesianUpdater.from_batch_params(tuned_params, config)
        
        # Process observations
        n = min(len(returns), len(volatility))
        update_count = 0
        last_result = None
        
        for i in range(n):
            y = float(returns[i])
            sigma = float(volatility[i])
            
            if np.isfinite(y) and np.isfinite(sigma) and sigma > 0:
                last_result = updater.update(y, sigma)
                update_count += 1
        
        # Get final parameters
        online_params = updater.get_current_params()
        final_ess = last_result.effective_sample_size if last_result else 0.0
        
        # Persist state
        persisted = False
        if persist:
            try:
                save_updater_state(symbol, updater)
                persisted = True
            except Exception as e:
                warnings.warn(f"Failed to persist state for {symbol}: {e}")
        
        # Update global cache
        _ONLINE_UPDATER_CACHE[symbol] = updater
        
        return BulkUpdateResult(
            symbol=symbol,
            success=True,
            online_params=online_params,
            update_count=update_count,
            final_ess=final_ess,
            persisted=persisted,
        )
        
    except Exception as e:
        return BulkUpdateResult(
            symbol=symbol,
            success=False,
            error=str(e),
        )


def bulk_online_update(
    assets_data: Dict[str, Dict[str, Any]],
    tuned_params_loader: Callable[[str], Optional[Dict[str, Any]]],
    config: Optional[OnlineUpdateConfig] = None,
    max_workers: int = 8,
    persist: bool = True,
    load_existing: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, BulkUpdateResult]:
    """
    Process online updates for multiple assets in parallel.
    
    Args:
        assets_data: Dict mapping symbol -> {'returns': array, 'volatility': array}
        tuned_params_loader: Function to load tuned params for a symbol
        config: Online update configuration
        max_workers: Maximum parallel workers
        persist: Whether to persist states to disk
        load_existing: Whether to load existing states
        progress_callback: Optional callback(symbol, completed, total)
        
    Returns:
        Dict mapping symbol -> BulkUpdateResult
    """
    results: Dict[str, BulkUpdateResult] = {}
    symbols = list(assets_data.keys())
    total = len(symbols)
    completed = 0
    
    def process_one(symbol: str) -> BulkUpdateResult:
        data = assets_data[symbol]
        returns = data.get("returns", np.array([]))
        volatility = data.get("volatility", np.array([]))
        
        # Load tuned parameters
        tuned_params = tuned_params_loader(symbol)
        if tuned_params is None:
            return BulkUpdateResult(
                symbol=symbol,
                success=False,
                error="No tuned parameters found",
            )
        
        return process_asset_online_update(
            symbol=symbol,
            returns=returns,
            volatility=volatility,
            tuned_params=tuned_params,
            config=config,
            persist=persist,
            load_existing=load_existing,
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(process_one, sym): sym 
            for sym in symbols
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
            except Exception as e:
                result = BulkUpdateResult(
                    symbol=symbol,
                    success=False,
                    error=str(e),
                )
            
            results[symbol] = result
            completed += 1
            
            if progress_callback:
                progress_callback(symbol, completed, total)
    
    return results


def bulk_load_and_update_from_cache(
    symbols: List[str],
    price_loader: Callable[[str], Optional[pd.DataFrame]],
    tuned_params_loader: Callable[[str], Optional[Dict[str, Any]]],
    config: Optional[OnlineUpdateConfig] = None,
    max_workers: int = 8,
    persist: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, BulkUpdateResult]:
    """
    Load price data from cache and run online updates for multiple symbols.
    
    This is the main entry point for bulk online parameter estimation.
    
    Args:
        symbols: List of asset symbols
        price_loader: Function to load price DataFrame for a symbol
        tuned_params_loader: Function to load tuned params for a symbol
        config: Online update configuration
        max_workers: Maximum parallel workers
        persist: Whether to persist states to disk
        log_fn: Optional logging function
        
    Returns:
        Dict mapping symbol -> BulkUpdateResult
    """
    log = log_fn or (lambda x: None)
    
    log(f"Loading price data for {len(symbols)} symbols...")
    
    # Prepare data for all symbols
    assets_data: Dict[str, Dict[str, Any]] = {}
    skipped = []
    
    for symbol in symbols:
        try:
            price_df = price_loader(symbol)
            if price_df is None or price_df.empty:
                skipped.append(symbol)
                continue
            
            # Compute returns and volatility
            if "Close" in price_df.columns:
                close = price_df["Close"]
            elif "Adj Close" in price_df.columns:
                close = price_df["Adj Close"]
            else:
                skipped.append(symbol)
                continue
            
            close = close.dropna()
            if len(close) < 50:
                skipped.append(symbol)
                continue
            
            returns = np.log(close / close.shift(1)).dropna().values
            vol = pd.Series(returns).ewm(span=21).std().values
            
            # Take last 200 observations for online update
            n = min(200, len(returns))
            assets_data[symbol] = {
                "returns": returns[-n:],
                "volatility": vol[-n:],
            }
            
        except Exception as e:
            skipped.append(symbol)
            continue
    
    if skipped:
        log(f"Skipped {len(skipped)} symbols (no data or insufficient history)")
    
    log(f"Running online updates for {len(assets_data)} symbols with {max_workers} workers...")
    
    # Progress tracking
    def progress_cb(symbol: str, completed: int, total: int):
        if completed % 10 == 0 or completed == total:
            log(f"  Progress: {completed}/{total} ({100*completed//total}%)")
    
    # Run bulk update
    results = bulk_online_update(
        assets_data=assets_data,
        tuned_params_loader=tuned_params_loader,
        config=config,
        max_workers=max_workers,
        persist=persist,
        load_existing=True,
        progress_callback=progress_cb,
    )
    
    # Summary
    success_count = sum(1 for r in results.values() if r.success)
    persisted_count = sum(1 for r in results.values() if r.persisted)
    avg_ess = np.mean([r.final_ess for r in results.values() if r.success and r.final_ess > 0])
    
    log(f"Completed: {success_count}/{len(results)} successful, {persisted_count} persisted")
    if np.isfinite(avg_ess):
        log(f"Average ESS: {avg_ess:.1f}")
    
    return results


# =============================================================================
# CONVENIENCE FUNCTIONS FOR MAKE TARGETS
# =============================================================================

def run_bulk_online_update(
    symbols: Optional[List[str]] = None,
    max_workers: int = 8,
    persist: bool = True,
) -> Dict[str, BulkUpdateResult]:
    """
    Run bulk online update using default loaders.
    
    This is designed to be called from Makefile targets.
    
    Args:
        symbols: List of symbols (None = use default universe)
        max_workers: Number of parallel workers
        persist: Whether to persist to disk
        
    Returns:
        Dict of results
    """
    # Import here to avoid circular imports
    from ingestion.data_utils import get_default_asset_universe, get_price_series
    from tuning.kalman_cache import load_tuned_params
    
    if symbols is None:
        symbols = get_default_asset_universe()
    
    def price_loader(symbol: str) -> Optional[pd.DataFrame]:
        """Load price data from cache."""
        from ingestion.data_utils import _load_disk_prices
        return _load_disk_prices(symbol)
    
    def tuned_loader(symbol: str) -> Optional[Dict[str, Any]]:
        """Load tuned parameters."""
        return load_tuned_params(symbol)
    
    return bulk_load_and_update_from_cache(
        symbols=symbols,
        price_loader=price_loader,
        tuned_params_loader=tuned_loader,
        max_workers=max_workers,
        persist=persist,
        log_fn=print,
    )


def clear_persisted_states() -> int:
    """Clear all persisted online update states. Returns count deleted."""
    _ensure_cache_dir()
    count = 0
    for path in ONLINE_UPDATE_CACHE_DIR.glob("*_online.json"):
        path.unlink()
        count += 1
    return count
