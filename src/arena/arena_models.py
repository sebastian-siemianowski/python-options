"""
===============================================================================
ARENA MODELS — Model Definitions and Registry for Competition
===============================================================================

This module provides the interface between the arena competition engine
and both standard and experimental models.

Standard models are imported from the main models/ package.
Experimental models are imported from arena/experimental_models/.

STANDARD MODELS (Baselines):
    - kalman_gaussian_momentum
    - kalman_phi_gaussian_momentum
    - phi_student_t_nu_{4,6,8,12,20}_momentum

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

STANDARD_MOMENTUM_MODELS = [
    "kalman_gaussian_momentum",
    "kalman_phi_gaussian_momentum",
    "phi_student_t_nu_4_momentum",
    "phi_student_t_nu_6_momentum",
    "phi_student_t_nu_8_momentum",
    "phi_student_t_nu_12_momentum",
    "phi_student_t_nu_20_momentum",
]


def get_standard_model_specs() -> List[Dict[str, Any]]:
    """Get specifications for standard momentum models."""
    specs = []
    
    # Gaussian momentum
    specs.append({
        "name": "kalman_gaussian_momentum",
        "family": "gaussian",
        "n_params": 2,
        "param_names": ("q", "c"),
        "description": "Momentum-augmented Gaussian Kalman filter",
    })
    
    # Phi-Gaussian momentum
    specs.append({
        "name": "kalman_phi_gaussian_momentum",
        "family": "phi_gaussian",
        "n_params": 3,
        "param_names": ("q", "c", "phi"),
        "description": "Momentum-augmented AR(1) Gaussian Kalman filter",
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
    'MomentumStudentTV2',
    'MomentumStudentTRegimeCoupled',
    'get_experimental_model_specs',
    'create_experimental_model',
]
