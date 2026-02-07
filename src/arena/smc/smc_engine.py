"""
===============================================================================
SMC ENGINE — Sequential Monte Carlo Model Selection Engine
===============================================================================

The main engine for running SMC-based model selection in the arena.

Workflow:
    1. Initialize particles across all models (standard + experimental)
    2. For each time step (observation):
        a. Each particle makes a prediction
        b. Compute CRPS for each prediction
        c. Update particle weights based on CRPS
        d. Check ESS and resample if needed
    3. Report posterior model probabilities

Key Features:
    - CRPS-based scoring (proper scoring rule)
    - Systematic resampling for low variance
    - Tempered updates to prevent degeneracy
    - Persistence of particle state across runs

Reference: Liu Xiaoming Panel Recommendation (Score 92/100)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from pathlib import Path
import numpy as np
import json

from .particle import Particle, ParticleCloud, create_initial_particles
from .resampling import (
    systematic_resample,
    effective_sample_size,
    should_resample,
    compute_resampling_statistics,
)


@dataclass
class SMCConfig:
    """
    Configuration for SMC engine.
    
    Attributes:
        n_particles_per_model: Particles per model
        ess_threshold_ratio: ESS/N ratio for resampling trigger
        temperature_schedule: Tempering schedule (list of temperatures)
        initial_temperature: Starting temperature (>1 for flat start)
        perturbation_noise: Parameter noise after resampling
        pit_hard_constraint: Zero weight for uncalibrated particles
        pit_threshold: PIT p-value threshold
        save_history: Whether to save particle history
        state_file: Path for persisting particle state
    """
    n_particles_per_model: int = 100
    ess_threshold_ratio: float = 0.5
    temperature_schedule: Optional[List[float]] = None
    initial_temperature: float = 2.0
    perturbation_noise: float = 0.05
    pit_hard_constraint: bool = True
    pit_threshold: float = 0.05
    save_history: bool = True
    state_file: str = "src/data/arena/smc_state.json"
    
    def get_temperature(self, step: int, n_steps: int) -> float:
        """Get temperature for given step."""
        if self.temperature_schedule:
            idx = min(step, len(self.temperature_schedule) - 1)
            return self.temperature_schedule[idx]
        
        # Linear annealing from initial_temperature to 1.0
        if n_steps <= 1:
            return 1.0
        progress = step / (n_steps - 1)
        return self.initial_temperature * (1 - progress) + 1.0 * progress


# Default configuration
DEFAULT_SMC_CONFIG = SMCConfig()


@dataclass
class SMCResult:
    """
    Result of SMC model selection.
    
    Attributes:
        model_probabilities: Posterior probability for each model
        best_model: Model with highest probability
        best_probability: Probability of best model
        ess_final: Final effective sample size
        ess_history: History of ESS values
        n_resamples: Number of resampling events
        crps_by_model: Average CRPS for each model
        parameter_estimates: Estimated parameters for each model
        timestamp: When the run completed
    """
    model_probabilities: Dict[str, float]
    best_model: str
    best_probability: float
    ess_final: float
    ess_history: List[float]
    n_resamples: int
    crps_by_model: Dict[str, float]
    parameter_estimates: Dict[str, Dict[str, float]]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_probabilities": self.model_probabilities,
            "best_model": self.best_model,
            "best_probability": self.best_probability,
            "ess_final": self.ess_final,
            "ess_history": self.ess_history,
            "n_resamples": self.n_resamples,
            "crps_by_model": self.crps_by_model,
            "parameter_estimates": self.parameter_estimates,
            "timestamp": self.timestamp,
        }
    
    def save(self, filepath: str) -> None:
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SMCEngine:
    """
    Sequential Monte Carlo engine for model selection.
    
    Maintains a particle cloud and updates it sequentially as new
    observations arrive. Provides posterior probabilities over models.
    """
    
    def __init__(
        self,
        model_specs: Dict[str, Dict[str, Any]],
        config: Optional[SMCConfig] = None,
        prediction_fn: Optional[Callable] = None,
    ):
        """
        Initialize SMC engine.
        
        Args:
            model_specs: Dict mapping model name to specification
                         Each spec should have 'default_params' and optionally 'model_class'
            config: SMC configuration
            prediction_fn: Function to compute predictions for a particle
                          Signature: (particle, returns, vol) -> (mu, sigma, nu)
        """
        self.model_specs = model_specs
        self.config = config or DEFAULT_SMC_CONFIG
        self.prediction_fn = prediction_fn
        
        # Initialize particle cloud
        self.cloud = create_initial_particles(
            model_specs,
            n_particles_per_model=self.config.n_particles_per_model,
            parameter_noise=0.1,
        )
        
        # Tracking
        self.step = 0
        self.n_resamples = 0
        self.ess_history: List[float] = []
        
        # Try to load existing state
        self._load_state()
    
    def _load_state(self) -> bool:
        """Load particle state from file if exists."""
        state_path = Path(self.config.state_file)
        if state_path.exists():
            try:
                self.cloud = ParticleCloud.load(str(state_path))
                return True
            except Exception as e:
                print(f"Warning: Could not load SMC state: {e}")
        return False
    
    def _save_state(self) -> None:
        """Save particle state to file."""
        state_path = Path(self.config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self.cloud.timestamp = datetime.now().isoformat()
        self.cloud.save(str(state_path))
    
    def update(
        self,
        observation: float,
        predictions: Dict[int, Tuple[float, float, Optional[float]]],
        pit_values: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        Update particle weights based on new observation.
        
        Args:
            observation: Actual observed value
            predictions: Dict mapping particle_id to (mu, sigma, nu)
                        nu is optional (None for Gaussian)
            pit_values: Optional dict of PIT p-values per particle
            
        Returns:
            Current effective sample size
        """
        from ..scoring.crps import compute_crps_gaussian, compute_crps_student_t
        
        n_steps = max(1, self.step + 1)
        temperature = self.config.get_temperature(self.step, n_steps)
        
        # Update each particle
        for particle in self.cloud.particles:
            pid = particle.particle_id
            
            if pid not in predictions:
                # No prediction available, use neutral update
                particle.update_weight(0.1, temperature)
                continue
            
            mu, sigma, nu = predictions[pid]
            
            # Compute CRPS
            obs_arr = np.array([observation])
            mu_arr = np.array([mu])
            sigma_arr = np.array([max(sigma, 1e-6)])
            
            if nu is not None and nu > 1:
                crps_result = compute_crps_student_t(obs_arr, mu_arr, sigma_arr, nu)
            else:
                crps_result = compute_crps_gaussian(obs_arr, mu_arr, sigma_arr)
            
            crps = crps_result.crps
            
            # Apply PIT hard constraint if enabled
            if self.config.pit_hard_constraint and pit_values:
                pit_pvalue = pit_values.get(pid, 1.0)
                if pit_pvalue < self.config.pit_threshold:
                    crps = 10.0  # Heavy penalty for uncalibrated
            
            # Update weight
            particle.update_weight(crps, temperature)
        
        # Normalize weights
        self.cloud.normalize_weights()
        
        # Compute ESS
        ess = self.cloud.effective_sample_size()
        self.ess_history.append(ess)
        self.cloud.ess_history.append(ess)
        
        # Check if resampling needed
        if should_resample(ess, self.cloud.n_particles, self.config.ess_threshold_ratio):
            self._resample()
        
        self.step += 1
        
        return ess
    
    def _resample(self) -> None:
        """Perform resampling to address particle degeneracy."""
        weights = self.cloud.weights
        
        # Systematic resampling
        indices = systematic_resample(weights)
        
        # Create new particles
        new_particles = []
        for new_id, old_idx in enumerate(indices):
            old_particle = self.cloud.particles[old_idx]
            new_particle = old_particle.clone(new_id)
            
            # Add perturbation to maintain diversity
            new_particle.perturb_parameters(self.config.perturbation_noise)
            new_particle.reset_weight()
            
            new_particles.append(new_particle)
        
        self.cloud.particles = new_particles
        self.n_resamples += 1
    
    def run_batch(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        model_filters: Dict[str, Callable],
    ) -> SMCResult:
        """
        Run SMC on a batch of observations.
        
        This runs the full SMC algorithm over a time series.
        
        Args:
            returns: Array of log returns
            vol: Array of volatility estimates
            model_filters: Dict mapping model name to filter function
                          Filter signature: (returns, vol, params) -> (mu, sigma, nu)
        
        Returns:
            SMCResult with model probabilities and diagnostics
        """
        n = len(returns)
        
        for t in range(1, n):
            # Generate predictions from all particles
            predictions = {}
            
            for particle in self.cloud.particles:
                model_name = particle.model_name
                
                if model_name in model_filters:
                    filter_fn = model_filters[model_name]
                    try:
                        # Run filter up to time t-1 to predict t
                        mu_arr, sigma_arr, nu = filter_fn(
                            returns[:t],
                            vol[:t],
                            particle.parameters,
                        )
                        # One-step-ahead prediction
                        mu_pred = mu_arr[-1] if len(mu_arr) > 0 else 0.0
                        sigma_pred = sigma_arr[-1] if len(sigma_arr) > 0 else vol[t-1]
                        
                        predictions[particle.particle_id] = (mu_pred, sigma_pred, nu)
                    except Exception:
                        # Filter failed, use neutral prediction
                        predictions[particle.particle_id] = (0.0, vol[t-1], None)
                else:
                    predictions[particle.particle_id] = (0.0, vol[t-1], None)
            
            # Update with actual observation
            self.update(returns[t], predictions)
        
        # Compute final results
        return self._compile_results()
    
    def _compile_results(self) -> SMCResult:
        """Compile final results from particle cloud."""
        model_probs = self.cloud.get_model_probabilities()
        best_model, best_prob = self.cloud.get_best_model()
        
        # Compute average CRPS per model
        crps_by_model: Dict[str, List[float]] = {}
        for particle in self.cloud.particles:
            model_name = particle.model_name
            if particle.crps_history:
                if model_name not in crps_by_model:
                    crps_by_model[model_name] = []
                crps_by_model[model_name].extend(particle.crps_history)
        
        avg_crps = {
            model: np.mean(crps_list) if crps_list else 0.0
            for model, crps_list in crps_by_model.items()
        }
        
        # Parameter estimates
        param_estimates = {}
        for model_name in self.model_specs.keys():
            param_estimates[model_name] = self.cloud.get_parameter_estimates(model_name)
        
        # Save state
        self._save_state()
        
        return SMCResult(
            model_probabilities=model_probs,
            best_model=best_model,
            best_probability=best_prob,
            ess_final=self.cloud.effective_sample_size(),
            ess_history=self.ess_history,
            n_resamples=self.n_resamples,
            crps_by_model=avg_crps,
            parameter_estimates=param_estimates,
            timestamp=datetime.now().isoformat(),
        )
    
    def get_promotion_candidates(
        self,
        standard_models: List[str],
        threshold: float = 0.05,
    ) -> List[str]:
        """
        Get experimental models that should be promoted.
        
        Criteria:
        - Posterior probability > threshold
        - Higher probability than best standard model
        
        Args:
            standard_models: List of standard model names
            threshold: Minimum posterior probability
            
        Returns:
            List of experimental model names to promote
        """
        model_probs = self.cloud.get_model_probabilities()
        
        # Find best standard model probability
        best_std_prob = 0.0
        for model in standard_models:
            if model in model_probs:
                best_std_prob = max(best_std_prob, model_probs[model])
        
        # Find experimental models that beat standards
        candidates = []
        for model, prob in model_probs.items():
            if model not in standard_models:
                if prob > threshold and prob > best_std_prob:
                    candidates.append(model)
        
        return candidates
    
    def reset(self) -> None:
        """Reset engine to initial state."""
        self.cloud = create_initial_particles(
            self.model_specs,
            n_particles_per_model=self.config.n_particles_per_model,
            parameter_noise=0.1,
        )
        self.step = 0
        self.n_resamples = 0
        self.ess_history = []
        
        # Remove saved state
        state_path = Path(self.config.state_file)
        if state_path.exists():
            state_path.unlink()
