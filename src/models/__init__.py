"""
===============================================================================
MODELS — Modular Kalman Drift Distribution Models
===============================================================================

This package contains the distributional models used in the tuning layer:

    - gaussian.py: GaussianDriftModel (Kalman with Gaussian noise)
    - phi_gaussian.py: PhiGaussianDriftModel (Kalman with AR(1) drift, Gaussian noise)
    - phi_student_t.py: PhiStudentTDriftModel (Kalman with AR(1) drift, Student-t noise)

DESIGN PHILOSOPHY:
    Each model class is SELF-CONTAINED with no cross-dependencies.
    Each implements:
    - filter(): Run Kalman filter with model-specific dynamics
    - optimize_params(): Joint parameter optimization via CV-MLE
    - pit_ks(): PIT/KS calibration test

USAGE:
    from models import GaussianDriftModel, PhiGaussianDriftModel, PhiStudentTDriftModel
    
    # Or import individual components
    from models.gaussian import GaussianDriftModel
    from models.phi_student_t import PhiStudentTDriftModel, STUDENT_T_NU_GRID
"""

from models.gaussian import GaussianDriftModel
from models.phi_gaussian import PhiGaussianDriftModel
from models.phi_student_t import PhiStudentTDriftModel

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
    # Constants (from phi_student_t)
    'PHI_SHRINKAGE_TAU_MIN',
    'PHI_SHRINKAGE_GLOBAL_DEFAULT',
    'PHI_SHRINKAGE_LAMBDA_DEFAULT',
    'STUDENT_T_NU_GRID',
]