"""
===============================================================================
MODELS — Modular Kalman Drift Distribution Models
===============================================================================

This package contains the distributional models used in the tuning layer:

    - gaussian.py: GaussianDriftModel (Kalman with Gaussian noise)
    - phi_gaussian.py: PhiGaussianDriftModel (Kalman with AR(1) drift, Gaussian noise)
    - phi_student_t.py: PhiStudentTDriftModel (Kalman with AR(1) drift, Student-t noise)
    - phi_skew_t.py: PhiSkewTDriftModel (Kalman with AR(1) drift, Skew-t noise)
    - phi_nig.py: PhiNIGDriftModel (Kalman with AR(1) drift, NIG noise)
    - gaussian_mixture.py: GaussianMixtureModel (2-State GMM for Monte Carlo)
    - hansen_skew_t.py: Hansen's Skew-t distribution (regime-conditional asymmetry)

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
    
    The distributional models capture different aspects of return dynamics:
    - Gaussian: Baseline, symmetric, light tails
    - Student-t: Symmetric, heavy tails (controlled by ν)
    - Skew-t (Fernández-Steel): Asymmetric, heavy tails (γ parameter)
    - Skew-t (Hansen): Asymmetric, heavy tails (λ parameter, regime-conditional)
    - NIG: Asymmetric, semi-heavy tails (α for tails, β for skewness)
    - GMM: Bimodal distribution (2 Gaussian components for momentum/reversal)
    
    CORE PRINCIPLE: "Heavy tails, asymmetry, and bimodality are hypotheses, not certainties."
    Complex distributions are introduced ONLY when supported by data.

USAGE:
    from models import GaussianDriftModel, PhiGaussianDriftModel, PhiStudentTDriftModel
    from models import PhiSkewTDriftModel, PhiNIGDriftModel, GaussianMixtureModel
    from models import HansenSkewTParams, fit_hansen_skew_t_mle
    
    # Or import individual components
    from models.hansen_skew_t import hansen_skew_t_cdf, hansen_skew_t_rvs
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

# Import φ-NIG model for BMA ensemble (Normal-Inverse Gaussian)
from models.phi_nig import (
    PhiNIGDriftModel,
    NIG_ALPHA_GRID,
    NIG_BETA_RATIO_GRID,
    NIG_ALPHA_MIN,
    NIG_ALPHA_MAX,
    NIG_ALPHA_DEFAULT,
    NIG_BETA_DEFAULT,
    NIG_DELTA_MIN,
    NIG_DELTA_MAX,
    NIG_DELTA_DEFAULT,
    is_nig_model,
    get_nig_model_name,
    parse_nig_model_name,
)

# Import 2-State Gaussian Mixture Model for Monte Carlo proposal distribution
from models.gaussian_mixture import (
    GaussianMixtureModel,
    fit_gmm_to_returns,
    compute_gmm_pit,
    get_gmm_model_name,
    is_gmm_model,
    GMM_MIN_OBS,
    GMM_MIN_SEPARATION,
    GMM_DEGENERATE_THRESHOLD,
)

# Import Hansen's Skew-t for regime-conditional asymmetric tails
from models.hansen_skew_t import (
    HansenSkewTParams,
    hansen_skew_t_pdf,
    hansen_skew_t_logpdf,
    hansen_skew_t_cdf,
    hansen_skew_t_ppf,
    hansen_skew_t_rvs,
    fit_hansen_skew_t_mle,
    compute_hansen_skew_t_pit,
    compare_symmetric_vs_hansen,
    hansen_skew_t_expected_shortfall,
    get_hansen_skew_t_model_name,
    is_hansen_skew_t_model,
    parse_hansen_skew_t_model_name,
    HANSEN_NU_MIN,
    HANSEN_NU_MAX,
    HANSEN_NU_DEFAULT,
    HANSEN_LAMBDA_MIN,
    HANSEN_LAMBDA_MAX,
    HANSEN_LAMBDA_DEFAULT,
    HANSEN_MLE_MIN_OBS,
)

# Import Contaminated Student-t Mixture for regime-dependent tails
from models.contaminated_student_t import (
    ContaminatedStudentTParams,
    contaminated_student_t_pdf,
    contaminated_student_t_logpdf,
    contaminated_student_t_rvs,
    fit_contaminated_student_t_profile,
    compute_contaminated_pit,
    compare_contaminated_vs_single,
    compute_crisis_probability_from_vol,
    compute_crisis_probability_from_drawdown,
    CST_NU_MIN,
    CST_NU_MAX,
    CST_NU_NORMAL_DEFAULT,
    CST_NU_CRISIS_DEFAULT,
    CST_EPSILON_MIN,
    CST_EPSILON_MAX,
    CST_EPSILON_DEFAULT,
    CST_MIN_OBS,
)


# Re-export constants from phi_student_t (the canonical source for Student-t config)
from models.phi_student_t import (
    PHI_SHRINKAGE_TAU_MIN,
    PHI_SHRINKAGE_GLOBAL_DEFAULT,
    PHI_SHRINKAGE_LAMBDA_DEFAULT,
    STUDENT_T_NU_GRID,
    # Enhanced Student-t constants (February 2026)
    NU_LEFT_GRID,
    NU_RIGHT_GRID,
    TWO_PIECE_BMA_PENALTY,
    GAMMA_VOV_GRID,
    VOV_BMA_PENALTY,
    NU_CALM_GRID,
    NU_STRESS_GRID,
    MIXTURE_BMA_PENALTY,
    MIXTURE_WEIGHT_K,
    MIXTURE_WEIGHT_DEFAULT,
    # Enhanced Mixture Weight Dynamics (February 2026 - Expert Panel)
    MIXTURE_WEIGHT_A_SHOCK,
    MIXTURE_WEIGHT_B_VOL_ACCEL,
    MIXTURE_WEIGHT_C_MOMENTUM,
)

__all__ = [
    # Models
    'GaussianDriftModel',
    'PhiGaussianDriftModel',
    'PhiStudentTDriftModel',
    'PhiSkewTDriftModel',
    'PhiNIGDriftModel',
    'GaussianMixtureModel',
    # Constants (from phi_student_t)
    'PHI_SHRINKAGE_TAU_MIN',
    'PHI_SHRINKAGE_GLOBAL_DEFAULT',
    'PHI_SHRINKAGE_LAMBDA_DEFAULT',
    'STUDENT_T_NU_GRID',
    # Enhanced Student-t constants
    'NU_LEFT_GRID',
    'NU_RIGHT_GRID',
    'TWO_PIECE_BMA_PENALTY',
    'GAMMA_VOV_GRID',
    'VOV_BMA_PENALTY',
    'NU_CALM_GRID',
    'NU_STRESS_GRID',
    'MIXTURE_BMA_PENALTY',
    'MIXTURE_WEIGHT_K',
    'MIXTURE_WEIGHT_DEFAULT',
    # Enhanced Mixture Weight Dynamics (February 2026)
    'MIXTURE_WEIGHT_A_SHOCK',
    'MIXTURE_WEIGHT_B_VOL_ACCEL',
    'MIXTURE_WEIGHT_C_MOMENTUM',
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
    # Constants (from phi_nig)
    'NIG_ALPHA_GRID',
    'NIG_BETA_RATIO_GRID',
    'NIG_ALPHA_MIN',
    'NIG_ALPHA_MAX',
    'NIG_ALPHA_DEFAULT',
    'NIG_BETA_DEFAULT',
    'NIG_DELTA_MIN',
    'NIG_DELTA_MAX',
    'NIG_DELTA_DEFAULT',
    # NIG utilities
    'is_nig_model',
    'get_nig_model_name',
    'parse_nig_model_name',
    # GMM utilities
    'fit_gmm_to_returns',
    'compute_gmm_pit',
    'get_gmm_model_name',
    'is_gmm_model',
    'GMM_MIN_OBS',
    'GMM_MIN_SEPARATION',
    'GMM_DEGENERATE_THRESHOLD',
    # Hansen Skew-t (regime-conditional asymmetry)
    'HansenSkewTParams',
    'hansen_skew_t_pdf',
    'hansen_skew_t_logpdf',
    'hansen_skew_t_cdf',
    'hansen_skew_t_ppf',
    'hansen_skew_t_rvs',
    'fit_hansen_skew_t_mle',
    'compute_hansen_skew_t_pit',
    'compare_symmetric_vs_hansen',
    'hansen_skew_t_expected_shortfall',
    'get_hansen_skew_t_model_name',
    'is_hansen_skew_t_model',
    'parse_hansen_skew_t_model_name',
    'HANSEN_NU_MIN',
    'HANSEN_NU_MAX',
    'HANSEN_NU_DEFAULT',
    'HANSEN_LAMBDA_MIN',
    'HANSEN_LAMBDA_MAX',
    'HANSEN_LAMBDA_DEFAULT',
    'HANSEN_MLE_MIN_OBS',
    # Contaminated Student-t Mixture (regime-dependent tails)
    'ContaminatedStudentTParams',
    'contaminated_student_t_pdf',
    'contaminated_student_t_logpdf',
    'contaminated_student_t_rvs',
    'fit_contaminated_student_t_profile',
    'compute_contaminated_pit',
    'compare_contaminated_vs_single',
    'compute_crisis_probability_from_vol',
    'compute_crisis_probability_from_drawdown',
    'CST_NU_MIN',
    'CST_NU_MAX',
    'CST_NU_NORMAL_DEFAULT',
    'CST_NU_CRISIS_DEFAULT',
    'CST_EPSILON_MIN',
    'CST_EPSILON_MAX',
    'CST_EPSILON_DEFAULT',
    'CST_MIN_OBS',
]