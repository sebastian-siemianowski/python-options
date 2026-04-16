"""
Story 3.3: phi-nu Joint Identifiability Guard
==============================================

Computes the joint Hessian of the Student-t log-likelihood w.r.t. (phi, nu)
and monitors the condition number to detect identifiability issues.

When phi and nu trade off (both affect tail behavior), the Hessian becomes
ill-conditioned. This module detects that and applies regularization.

Architecture:
  1. Numerically compute 2x2 Hessian at (phi*, nu*)
  2. Log condition number for diagnostics
  3. If kappa(H) > 100: flag as near-singular, apply regularization
  4. Regularization: ||phi - phi_0||^2/lambda_phi + ||nu - nu_0||^2/lambda_nu
"""
import os
import sys
import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Condition number thresholds
KAPPA_WARNING = 100.0       # Log warning if exceeded
KAPPA_CRITICAL = 1000.0     # Apply regularization if exceeded

# Regularization defaults
DEFAULT_PHI_0 = 0.0         # Default phi prior center (overridden by Story 3.1)
DEFAULT_NU_0 = 8.0          # Default nu prior center
DEFAULT_LAMBDA_PHI = 0.1    # phi regularization strength
DEFAULT_LAMBDA_NU = 2.0     # nu regularization strength (looser -- nu has wider range)

# Finite difference step sizes
FD_STEP_PHI = 1e-4          # Step for phi finite difference
FD_STEP_NU = 0.05           # Step for nu finite difference (nu is larger scale)


@dataclass
class IdentifiabilityResult:
    """Result of phi-nu identifiability check."""
    phi: float
    nu: float
    hessian: np.ndarray         # 2x2 Hessian matrix
    condition_number: float     # Condition number kappa(H)
    is_warning: bool            # kappa > KAPPA_WARNING
    is_critical: bool           # kappa > KAPPA_CRITICAL
    phi_regularized: float      # Post-regularization phi (same if no action needed)
    nu_regularized: float       # Post-regularization nu (same if no action needed)
    regularization_applied: bool


# ---------------------------------------------------------------------------
# Joint Hessian Computation
# ---------------------------------------------------------------------------

def _compute_log_likelihood(returns: np.ndarray, vol: np.ndarray,
                            q: float, c: float, phi: float, nu: float) -> float:
    """Compute Student-t Kalman filter log-likelihood at (phi, nu)."""
    try:
        from models.phi_student_t_unified import UnifiedPhiStudentTModel
        _, _, ll = UnifiedPhiStudentTModel.filter_phi(returns, vol, q, c, phi, nu)
        if np.isfinite(ll):
            return float(ll)
    except Exception:
        pass
    return -1e18


def compute_phi_nu_hessian(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    h_phi: float = FD_STEP_PHI,
    h_nu: float = FD_STEP_NU,
) -> np.ndarray:
    """
    Compute 2x2 Hessian of log-likelihood w.r.t. (phi, nu) via central finite differences.

    H = [[d2L/dphi2,   d2L/dphi_dnu],
         [d2L/dnu_dphi, d2L/dnu2    ]]

    Parameters
    ----------
    returns, vol : arrays
        Return and volatility series.
    q, c : float
        Fixed process noise and observation noise parameters.
    phi, nu : float
        Point at which to evaluate the Hessian.
    h_phi, h_nu : float
        Finite difference step sizes.

    Returns
    -------
    2x2 numpy array
    """
    # Central finite differences for second derivatives
    # d2L/dphi2
    ll_pp = _compute_log_likelihood(returns, vol, q, c, phi + h_phi, nu)
    ll_pm = _compute_log_likelihood(returns, vol, q, c, phi, nu)
    ll_mm = _compute_log_likelihood(returns, vol, q, c, phi - h_phi, nu)
    d2_phi2 = (ll_pp - 2.0 * ll_pm + ll_mm) / (h_phi ** 2)

    # d2L/dnu2
    ll_np = _compute_log_likelihood(returns, vol, q, c, phi, nu + h_nu)
    ll_nm = _compute_log_likelihood(returns, vol, q, c, phi, nu - h_nu)
    d2_nu2 = (ll_np - 2.0 * ll_pm + ll_nm) / (h_nu ** 2)

    # d2L/dphi_dnu (cross derivative)
    ll_pp_np = _compute_log_likelihood(returns, vol, q, c, phi + h_phi, nu + h_nu)
    ll_pm_np = _compute_log_likelihood(returns, vol, q, c, phi - h_phi, nu + h_nu)
    ll_pp_nm = _compute_log_likelihood(returns, vol, q, c, phi + h_phi, nu - h_nu)
    ll_pm_nm = _compute_log_likelihood(returns, vol, q, c, phi - h_phi, nu - h_nu)
    d2_phi_nu = (ll_pp_np - ll_pm_np - ll_pp_nm + ll_pm_nm) / (4.0 * h_phi * h_nu)

    hessian = np.array([
        [d2_phi2, d2_phi_nu],
        [d2_phi_nu, d2_nu2],
    ])

    return hessian


def compute_condition_number(hessian: np.ndarray) -> float:
    """Compute condition number of a 2x2 matrix."""
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        abs_eig = np.abs(eigenvalues)
        if abs_eig.min() < 1e-15:
            return 1e12  # Effectively singular
        return float(abs_eig.max() / abs_eig.min())
    except np.linalg.LinAlgError:
        return 1e12


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def apply_phi_nu_regularization(
    phi: float,
    nu: float,
    condition_number: float,
    phi_0: float = DEFAULT_PHI_0,
    nu_0: float = DEFAULT_NU_0,
    lambda_phi: float = DEFAULT_LAMBDA_PHI,
    lambda_nu: float = DEFAULT_LAMBDA_NU,
) -> Tuple[float, float]:
    """
    Apply shrinkage regularization when (phi, nu) are ill-conditioned.

    Regularized estimates:
      phi_reg = (1 - alpha) * phi + alpha * phi_0
      nu_reg  = (1 - alpha) * nu  + alpha * nu_0

    where alpha increases with condition number severity.

    Parameters
    ----------
    phi, nu : float
        Current estimates.
    condition_number : float
        Condition number of the joint Hessian.
    phi_0, nu_0 : float
        Prior centers for regularization.
    lambda_phi, lambda_nu : float
        Regularization strengths (used for alpha scaling).

    Returns
    -------
    (phi_regularized, nu_regularized)
    """
    if condition_number <= KAPPA_WARNING:
        return phi, nu

    # Scale alpha based on severity: 0 at KAPPA_WARNING, 0.5 at KAPPA_CRITICAL
    log_ratio = math.log(condition_number / KAPPA_WARNING) / math.log(KAPPA_CRITICAL / KAPPA_WARNING)
    alpha = min(0.5, max(0.0, 0.5 * log_ratio))

    # Strength-weighted shrinkage
    phi_weight = min(1.0, lambda_phi)
    nu_weight = min(1.0, lambda_nu / 10.0)  # nu has wider scale

    phi_reg = (1.0 - alpha * phi_weight) * phi + alpha * phi_weight * phi_0
    nu_reg = (1.0 - alpha * nu_weight) * nu + alpha * nu_weight * nu_0

    # Clamp to valid ranges
    phi_reg = float(np.clip(phi_reg, -0.80, 0.99))
    nu_reg = float(np.clip(nu_reg, 2.1, 30.0))

    return phi_reg, nu_reg


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def check_phi_nu_identifiability(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    phi_0: Optional[float] = None,
    nu_0: float = DEFAULT_NU_0,
    lambda_phi: float = DEFAULT_LAMBDA_PHI,
    lambda_nu: float = DEFAULT_LAMBDA_NU,
    asset_symbol: Optional[str] = None,
) -> IdentifiabilityResult:
    """
    Check identifiability of (phi, nu) and apply regularization if needed.

    This is called after Stage 5 (nu selection) in the unified pipeline.

    Parameters
    ----------
    returns, vol : arrays
        Return and volatility data.
    q, c, phi, nu : float
        Optimized parameters.
    phi_0 : float, optional
        phi prior center (from Story 3.1 compute_phi_prior). Uses default if None.
    nu_0 : float
        nu prior center.
    lambda_phi, lambda_nu : float
        Regularization strengths.
    asset_symbol : str, optional
        For logging.

    Returns
    -------
    IdentifiabilityResult
    """
    if phi_0 is None:
        phi_0 = DEFAULT_PHI_0
        try:
            from models.phi_student_t_unified import compute_phi_prior
            phi_0, _ = compute_phi_prior(asset_symbol, returns=None)
        except ImportError:
            pass

    # Compute joint Hessian
    hessian = compute_phi_nu_hessian(returns, vol, q, c, phi, nu)
    kappa = compute_condition_number(hessian)

    is_warning = kappa > KAPPA_WARNING
    is_critical = kappa > KAPPA_CRITICAL

    # Log
    asset_tag = f" [{asset_symbol}]" if asset_symbol else ""
    if is_critical:
        logger.warning(
            "phi-nu CRITICAL identifiability%s: kappa=%.1f > %.0f, "
            "phi=%.4f, nu=%.2f. Applying regularization.",
            asset_tag, kappa, KAPPA_CRITICAL, phi, nu,
        )
    elif is_warning:
        logger.info(
            "phi-nu identifiability warning%s: kappa=%.1f > %.0f, phi=%.4f, nu=%.2f",
            asset_tag, kappa, KAPPA_WARNING, phi, nu,
        )

    # Apply regularization if needed
    phi_reg, nu_reg = phi, nu
    reg_applied = False
    if is_warning:
        phi_reg, nu_reg = apply_phi_nu_regularization(
            phi, nu, kappa, phi_0, nu_0, lambda_phi, lambda_nu,
        )
        reg_applied = (abs(phi_reg - phi) > 1e-6 or abs(nu_reg - nu) > 1e-4)

    return IdentifiabilityResult(
        phi=phi,
        nu=nu,
        hessian=hessian,
        condition_number=kappa,
        is_warning=is_warning,
        is_critical=is_critical,
        phi_regularized=phi_reg,
        nu_regularized=nu_reg,
        regularization_applied=reg_applied,
    )
