"""
===============================================================================
PARTICLE — Particle Data Structures for SMC
===============================================================================

Defines the particle representation for Sequential Monte Carlo model selection.

A particle represents a hypothesis about the true data-generating process:
    - Which model class (Gaussian, Student-t, experimental, etc.)
    - What parameters (q, c, phi, nu, etc.)
    - Current weight (posterior probability)

Particles form a cloud that approximates the posterior over models.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import json
from pathlib import Path


@dataclass
class Particle:
    """
    A single particle in the SMC system.
    
    Attributes:
        particle_id: Unique identifier
        model_name: Name of the model this particle represents
        parameters: Model parameters (q, c, phi, nu, etc.)
        weight: Current particle weight (unnormalized log-weight internally)
        log_weight: Log of weight for numerical stability
        crps_history: History of CRPS scores for this particle
        prediction_cache: Cached predictions for efficiency
        generation: How many times this particle has been resampled
    """
    particle_id: int
    model_name: str
    parameters: Dict[str, float]
    weight: float = 1.0
    log_weight: float = 0.0
    crps_history: List[float] = field(default_factory=list)
    prediction_cache: Optional[Dict[str, Any]] = None
    generation: int = 0
    
    def update_weight(self, crps: float, temperature: float = 1.0) -> None:
        """
        Update particle weight based on CRPS score.
        
        Lower CRPS is better, so we use negative CRPS as log-likelihood.
        
        Args:
            crps: CRPS score for this particle's prediction
            temperature: Tempering parameter (1.0 = standard, >1 = flatter)
        """
        # CRPS is a proper scoring rule: lower is better
        # Convert to log-weight increment (negative because lower is better)
        log_increment = -crps / temperature
        
        self.log_weight += log_increment
        self.crps_history.append(crps)
        
        # Update weight (will be normalized across particles)
        self.weight = np.exp(self.log_weight)
    
    def reset_weight(self) -> None:
        """Reset weight after resampling."""
        self.log_weight = 0.0
        self.weight = 1.0
        self.generation += 1
    
    def clone(self, new_id: int) -> 'Particle':
        """Create a clone of this particle with a new ID."""
        return Particle(
            particle_id=new_id,
            model_name=self.model_name,
            parameters=dict(self.parameters),
            weight=1.0,
            log_weight=0.0,
            crps_history=list(self.crps_history),
            prediction_cache=None,
            generation=self.generation,
        )
    
    def perturb_parameters(self, noise_scale: float = 0.1) -> None:
        """
        Add small perturbation to parameters (for diversity after resampling).
        
        Args:
            noise_scale: Scale of Gaussian noise relative to parameter value
        """
        for key, value in self.parameters.items():
            if isinstance(value, (int, float)) and key != 'nu':
                # Add multiplicative noise
                noise = np.random.normal(0, noise_scale * abs(value) + 1e-10)
                self.parameters[key] = value + noise
                
                # Enforce constraints
                if key == 'q':
                    self.parameters[key] = max(1e-10, self.parameters[key])
                elif key == 'c':
                    self.parameters[key] = max(0.1, min(5.0, self.parameters[key]))
                elif key == 'phi':
                    self.parameters[key] = max(-0.99, min(0.99, self.parameters[key]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "particle_id": self.particle_id,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "weight": self.weight,
            "log_weight": self.log_weight,
            "crps_history": self.crps_history,
            "generation": self.generation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Particle':
        """Create from dictionary."""
        return cls(
            particle_id=data["particle_id"],
            model_name=data["model_name"],
            parameters=data["parameters"],
            weight=data.get("weight", 1.0),
            log_weight=data.get("log_weight", 0.0),
            crps_history=data.get("crps_history", []),
            generation=data.get("generation", 0),
        )


@dataclass
class ParticleCloud:
    """
    Collection of particles representing posterior over models.
    
    Attributes:
        particles: List of particles
        n_particles: Number of particles
        timestamp: Last update timestamp
        ess_history: History of effective sample sizes
    """
    particles: List[Particle]
    n_particles: int = 0
    timestamp: str = ""
    ess_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.n_particles = len(self.particles)
    
    @property
    def weights(self) -> np.ndarray:
        """Get normalized weights."""
        w = np.array([p.weight for p in self.particles])
        w_sum = np.sum(w)
        if w_sum > 0:
            return w / w_sum
        return np.ones(len(self.particles)) / len(self.particles)
    
    @property
    def log_weights(self) -> np.ndarray:
        """Get log weights."""
        return np.array([p.log_weight for p in self.particles])
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
    
    def effective_sample_size(self) -> float:
        """
        Compute effective sample size (ESS).
        
        ESS = 1 / Σ w_i²
        
        ESS close to N means diverse particles.
        ESS close to 1 means particle degeneracy.
        """
        w = self.weights
        ess = 1.0 / np.sum(w ** 2)
        return float(ess)
    
    def get_model_probabilities(self) -> Dict[str, float]:
        """
        Compute posterior probability for each model.
        
        Aggregates weights across particles by model name.
        
        Returns:
            Dict mapping model name to posterior probability
        """
        w = self.weights
        model_weights: Dict[str, float] = {}
        
        for i, particle in enumerate(self.particles):
            model_name = particle.model_name
            if model_name not in model_weights:
                model_weights[model_name] = 0.0
            model_weights[model_name] += w[i]
        
        return model_weights
    
    def get_best_model(self) -> Tuple[str, float]:
        """Get model with highest posterior probability."""
        probs = self.get_model_probabilities()
        if not probs:
            return "", 0.0
        best = max(probs.items(), key=lambda x: x[1])
        return best
    
    def get_parameter_estimates(self, model_name: str) -> Dict[str, float]:
        """
        Get weighted average parameters for a specific model.
        
        Args:
            model_name: Model to get parameters for
            
        Returns:
            Dict of parameter estimates
        """
        w = self.weights
        
        # Filter particles for this model
        model_particles = [(p, w[i]) for i, p in enumerate(self.particles) if p.model_name == model_name]
        
        if not model_particles:
            return {}
        
        # Compute weighted average of each parameter
        param_sums: Dict[str, float] = {}
        weight_sum = 0.0
        
        for particle, weight in model_particles:
            weight_sum += weight
            for key, value in particle.parameters.items():
                if isinstance(value, (int, float)):
                    if key not in param_sums:
                        param_sums[key] = 0.0
                    param_sums[key] += weight * value
        
        if weight_sum > 0:
            return {k: v / weight_sum for k, v in param_sums.items()}
        return {}
    
    def save(self, filepath: str) -> None:
        """Save particle cloud to JSON file."""
        data = {
            "n_particles": self.n_particles,
            "timestamp": self.timestamp,
            "ess_history": self.ess_history,
            "particles": [p.to_dict() for p in self.particles],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ParticleCloud':
        """Load particle cloud from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        particles = [Particle.from_dict(p) for p in data["particles"]]
        
        return cls(
            particles=particles,
            n_particles=data.get("n_particles", len(particles)),
            timestamp=data.get("timestamp", ""),
            ess_history=data.get("ess_history", []),
        )


def create_initial_particles(
    model_specs: Dict[str, Dict[str, Any]],
    n_particles_per_model: int = 100,
    parameter_noise: float = 0.2,
) -> ParticleCloud:
    """
    Create initial particle cloud with uniform weights.
    
    Creates n_particles_per_model particles for each model, with
    parameters sampled around default values with noise.
    
    Args:
        model_specs: Dict mapping model name to spec with default_params
        n_particles_per_model: Number of particles per model
        parameter_noise: Scale of parameter variation (relative)
        
    Returns:
        ParticleCloud with initialized particles
    """
    particles = []
    particle_id = 0
    
    for model_name, spec in model_specs.items():
        default_params = spec.get("default_params", {})
        
        for _ in range(n_particles_per_model):
            # Sample parameters around defaults
            params = {}
            for key, value in default_params.items():
                if isinstance(value, (int, float)):
                    # Add Gaussian noise
                    noise = np.random.normal(0, parameter_noise * abs(value) + 1e-10)
                    params[key] = value + noise
                    
                    # Enforce constraints
                    if key == 'q':
                        params[key] = max(1e-10, min(1e-2, params[key]))
                    elif key == 'c':
                        params[key] = max(0.1, min(5.0, params[key]))
                    elif key == 'phi':
                        params[key] = max(-0.99, min(0.99, params[key]))
                    elif key == 'nu':
                        params[key] = int(value)  # Keep nu fixed at discrete values
                else:
                    params[key] = value
            
            particle = Particle(
                particle_id=particle_id,
                model_name=model_name,
                parameters=params,
                weight=1.0,
                log_weight=0.0,
            )
            particles.append(particle)
            particle_id += 1
    
    return ParticleCloud(particles=particles)
