"""
===============================================================================
ARENA MODELS — Model Definitions and Registry for Competition
===============================================================================

This module provides the interface between the arena competition engine
and both standard and experimental models.

Standard models are imported from the main models/ package.
Experimental models are imported from arena/experimental_models/.

STANDARD MODELS (Baselines):
    - kalman_gaussian_unified
    - kalman_phi_gaussian_unified
    - phi_student_t_nu_{4,8,20}_momentum

EXPERIMENTAL MODELS (in arena/experimental_models/):
    - momentum_student_t_v2: Upgraded Student-t with adaptive tail coupling
    - momentum_student_t_regime_coupled: Regime-aware tail dynamics

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, List, Any

# Import experimental models from dedicated package
from .experimental_models import (
    EXPERIMENTAL_MODELS,
    EXPERIMENTAL_MODEL_SPECS,
    ExperimentalModelSpec,
    ExperimentalModelFamily,
    get_experimental_model_specs,
    create_experimental_model,
)


# =============================================================================
# STANDARD MODEL SPECS (for reference/comparison)
# =============================================================================

STANDARD_MODELS = [
    "kalman_gaussian_unified",
    "kalman_phi_gaussian_unified",
    "phi_student_t_nu_4_momentum",
    "phi_student_t_nu_8_momentum",
    "phi_student_t_nu_20_momentum",
]

# Legacy alias for backward compatibility
STANDARD_MOMENTUM_MODELS = STANDARD_MODELS


def get_standard_model_specs() -> List[Dict[str, Any]]:
    """Get specifications for standard models (unified Gaussian + Student-t momentum)."""
    specs = []
    
    # Unified Gaussian - momentum + GAS-Q are internal stages
    specs.append({
        "name": "kalman_gaussian_unified",
        "family": "gaussian_unified",
        "n_params": 6,
        "param_names": ("q", "c", "beta", "garch", "momentum_weight", "gas_q"),
        "description": "Unified Gaussian Kalman filter (internal momentum + GAS-Q)",
    })
    
    # Unified phi-Gaussian - momentum + GAS-Q are internal stages
    specs.append({
        "name": "kalman_phi_gaussian_unified",
        "family": "gaussian_unified",
        "n_params": 7,
        "param_names": ("q", "c", "phi", "beta", "garch", "momentum_weight", "gas_q"),
        "description": "Unified AR(1) Gaussian Kalman filter (internal momentum + GAS-Q)",
    })
    
    # Student-t momentum (discrete nu grid)
    for nu in [4, 8, 20]:
        specs.append({
            "name": f"phi_student_t_nu_{nu}_momentum",
            "family": "student_t",
            "n_params": 4,
            "param_names": ("q", "c", "phi", "nu"),
            "nu": nu,
            "description": f"Momentum-augmented Student-t(ν={nu}) Kalman filter",
        })
    
    return specs


__all__ = [
    # Standard models
    'STANDARD_MOMENTUM_MODELS',
    'get_standard_model_specs',
    # Experimental models (re-exported from experimental_models package)
    'EXPERIMENTAL_MODELS',
    'EXPERIMENTAL_MODEL_SPECS',
    'ExperimentalModelSpec',
    'ExperimentalModelFamily',
    'STANDARD_MODELS',
        'get_experimental_model_specs',
    'create_experimental_model',
]
