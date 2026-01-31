"""
===============================================================================
MODELS — Modular Kalman Drift Distribution Models
===============================================================================

This package contains the distributional models used in the tuning layer:

    - gaussian.py: GaussianDriftModel (Kalman with Gaussian noise)
    - phi_gaussian.py: PhiGaussianDriftModel (Kalman with AR(1) drift, Gaussian noise)
    - phi_student_t.py: PhiStudentTDriftModel (Kalman with AR(1) drift, Student-t noise)
    - phi_skew_t.py: PhiSkewTDriftModel (Kalman with AR(1) drift, Skew-t noise)

DESIGN PHILOSOPHY:
    Each model class is SELF-CONTAINED with no cross-dependencies.
    Each implements:
    - filter(): Run Kalman filter with model-specific dynamics
    - optimize_params(): Joint parameter optimization via CV-MLE
    - pit_ks(): PIT/KS calibration test

BMA ARCHITECTURE:
    The Bayesian Model Averaging framework treats each model as a competing hypothesis.
    Model weights are computed via BIC approximation to marginal likelihood:
    
        p(M_k | data) ∝ exp(-0.5 * BIC_k)
    
    The φ-Skew-t model (Fernández-Steel parameterization) captures:
    - Fat tails via ν (degrees of freedom)
    - Asymmetry via γ (skewness parameter)
    
    CORE PRINCIPLE: "Skewness is a hypothesis, not a certainty."
    Skewness is introduced ONLY when supported by data.

USAGE:
    from models import GaussianDriftModel, PhiGaussianDriftModel, PhiStudentTDriftModel
    from models import PhiSkewTDriftModel, SKEW_T_GAMMA_GRID
    
    # Or import individual components
    from models.gaussian import GaussianDriftModel
    from models.phi_student_t import PhiStudentTDriftModel, STUDENT_T_NU_GRID
    from models.phi_skew_t import PhiSkewTDriftModel, SKEW_T_GAMMA_GRID
"""

from models.gaussian import GaussianDriftModel
from models.phi_gaussian import PhiGaussianDriftModel
from models.phi_student_t import PhiStudentTDriftModel

# Import φ-Skew-t model for BMA ensemble (Fernández-Steel parameterization)
from models.phi_skew_t import (
    PhiSkewTDriftModel,
    SKEW_T_NU_GRID,
    SKEW_T_GAMMA_GRID,
    GAMMA_MIN,
    GAMMA_MAX,
    GAMMA_DEFAULT,
    is_skew_t_model,
    get_skew_t_model_name,
    parse_skew_t_model_name,
)

# Re-export constants from phi_student_t (the canonical source for Student-t config)
from models.phi_student_t import (
    PHI_SHRINKAGE_TAU_MIN,
    PHI_SHRINKAGE_GLOBAL_DEFAULT,
    PHI_SHRINKAGE_LAMBDA_DEFAULT,
    STUDENT_T_NU_GRID,
)

__all__ = [
    # Models
    'GaussianDriftModel',
    'PhiGaussianDriftModel',
    'PhiStudentTDriftModel',
    'PhiSkewTDriftModel',
    # Constants (from phi_student_t)
    'PHI_SHRINKAGE_TAU_MIN',
    'PHI_SHRINKAGE_GLOBAL_DEFAULT',
    'PHI_SHRINKAGE_LAMBDA_DEFAULT',
    'STUDENT_T_NU_GRID',
    # Constants (from phi_skew_t)
    'SKEW_T_NU_GRID',
    'SKEW_T_GAMMA_GRID',
    'GAMMA_MIN',
    'GAMMA_MAX',
    'GAMMA_DEFAULT',
    # Skew-t utilities
    'is_skew_t_model',
    'get_skew_t_model_name',
    'parse_skew_t_model_name',
]