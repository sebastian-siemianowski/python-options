"""
===============================================================================
SMC — Sequential Monte Carlo Model Selection
===============================================================================

Implements particle filtering for adaptive model selection in the arena.

Key Features:
1. Maintains posterior distribution over (model, parameter) configurations
2. Updates weights based on predictive performance (CRPS)
3. Resamples when effective sample size drops
4. Handles non-stationarity via sequential adaptation

Architecture:
    Particle = (model_name, parameters, weight)
    
    At each time step:
    1. Predict: each particle forecasts next observation
    2. Update: weight particles by CRPS of their predictions
    3. Resample: if ESS < threshold, resample particles
    4. Report: aggregate weights by model for selection

Reference: Liu Xiaoming Panel Recommendation (Score 92/100)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from .particle import (
    Particle,
    ParticleCloud,
    create_initial_particles,
)

from .smc_engine import (
    SMCEngine,
    SMCConfig,
    SMCResult,
    DEFAULT_SMC_CONFIG,
)

from .resampling import (
    systematic_resample,
    multinomial_resample,
    effective_sample_size,
)

__all__ = [
    # Particles
    'Particle',
    'ParticleCloud',
    'create_initial_particles',
    # Engine
    'SMCEngine',
    'SMCConfig',
    'SMCResult',
    'DEFAULT_SMC_CONFIG',
    # Resampling
    'systematic_resample',
    'multinomial_resample',
    'effective_sample_size',
]
