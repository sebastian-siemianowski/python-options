"""Signal frozen dataclass definition.

Extracted from signals.py - Story 8.1.
Contains the Signal dataclass with 80+ fields for signal generation output.
"""

from dataclasses import dataclass
from typing import Dict, Optional


# NOTE: ExpectedUtilityResult dataclass removed - was only used by the legacy
# compute_expected_utility() function. EU computation is now done inline in
# latest_signals() from BMA r_samples.


@dataclass(frozen=True)
class Signal:
    horizon_days: int
    score: float          # edge in z units (mu_H/sigma_H with filters)
    p_up: float           # P(return>0) - UNIFIED posterior predictive MC probability (THE ONLY TRADING PROBABILITY)
    exp_ret: float        # expected log return over horizon
    ci_low: float         # lower bound of expected log return CI (68% or user-specified)
    ci_high: float        # upper bound of expected log return CI (68% or user-specified)
    ci_low_90: float      # lower bound of 90% CI (Story 3.2: quantile-based)
    ci_high_90: float     # upper bound of 90% CI (Story 3.2: quantile-based)
    profit_pln: float     # expected profit in PLN for NOTIONAL_PLN invested
    profit_ci_low_pln: float  # low CI bound for profit in PLN
    profit_ci_high_pln: float # high CI bound for profit in PLN
    position_strength: float  # EU-based position sizing: EU / max(E[loss], ε), scaled by drift_weight
    vol_mean: float       # mean volatility forecast (stochastic vol posterior)
    vol_ci_low: float     # lower bound of volatility CI
    vol_ci_high: float    # upper bound of volatility CI
    regime: str               # detected regime label
    label: str                # BUY/HOLD/SELL or STRONG BUY/SELL
    # Expected Utility fields (THE BASIS FOR POSITION SIZING):
    expected_utility: float = 0.0     # EU = p × E[gain] - (1-p) × E[loss]
    expected_gain: float = 0.0        # E[R_H | R_H > 0]
    expected_loss: float = 0.0        # E[-R_H | R_H < 0] (positive value)
    gain_loss_ratio: float = 1.0      # E[gain] / E[loss] - asymmetry
    eu_position_size: float = 0.0     # Position size from EU / max(E[loss], ε)
    # Contaminated Student-t Mixture fields (regime-dependent tails):
    cst_enabled: bool = False         # Whether contaminated mixture was used in MC
    cst_nu_normal: Optional[float] = None   # ν for normal regime (lighter tails)
    cst_nu_crisis: Optional[float] = None   # ν for crisis regime (heavier tails)
    cst_epsilon: Optional[float] = None     # Crisis contamination probability
    # Hansen Skew-t fields (asymmetric return distribution):
    hansen_enabled: bool = False            # Whether Hansen skew-t was fitted
    hansen_lambda: Optional[float] = None   # Skewness parameter λ ∈ (-1, 1)
    hansen_nu: Optional[float] = None       # Degrees of freedom ν
    hansen_skew_direction: Optional[str] = None  # "left", "right", or "symmetric"
    # Diagnostics only (NOT used for trading decisions):
    drift_uncertainty: float = 0.0  # P_t × drift_var_factor: uncertainty in drift estimate propagated to horizon
    p_analytical: float = 0.5       # DIAGNOSTIC ONLY: analytical posterior predictive P(r>0|D) 
    p_empirical: float = 0.5        # DIAGNOSTIC ONLY: raw empirical MC probability P(r>0) from simulations
    # STEP 7: Regime audit trace - tracks which regime params were used
    regime_used: Optional[int] = None        # Integer regime index (0-4) used for parameter selection
    regime_source: str = "global"            # "regime_tuned" or "global" - source of parameters
    regime_collapse_warning: bool = False    # True if regime params collapsed to global
    # STEP 8: Bayesian Model Averaging audit trace
    bma_method: str = "legacy"               # "bayesian_model_averaging_full" or "legacy"
    bma_has_model_posterior: bool = False    # True if BMA with model posteriors was used
    bma_borrowed_from_global: bool = False   # True if regime used hierarchical fallback
    # DUAL-SIDED TREND EXHAUSTION (0-100% scale, multi-timeframe weighted EMA deviation):
    ue_up: float = 0.0    # Price above weighted EMA equilibrium (0-1 scale)
    ue_down: float = 0.0  # Price below weighted EMA equilibrium (0-1 scale)
    # RISK TEMPERATURE MODULATION (cross-asset stress scaling):
    risk_temperature: float = 0.0      # Global risk temperature (0-2 scale)
    risk_scale_factor: float = 1.0     # Position scale factor from risk temperature
    overnight_budget_applied: bool = False  # True if overnight budget constraint was applied
    overnight_max_position: Optional[float] = None  # Max position from overnight budget
    pos_strength_pre_risk_temp: float = 0.0  # Position strength before risk temperature scaling
    # EVT (Extreme Value Theory) tail risk fields:
    expected_loss_empirical: float = 0.0    # Empirical expected loss (before EVT)
    evt_enabled: bool = False               # Whether EVT was used for tail estimation
    evt_xi: Optional[float] = None          # GPD shape parameter ξ
    evt_sigma: Optional[float] = None       # GPD scale parameter σ
    evt_threshold: Optional[float] = None   # POT threshold
    evt_n_exceedances: int = 0              # Number of threshold exceedances
    evt_fit_method: Optional[str] = None    # EVT fitting method used
    # PIT Violation EXIT Signal (February 2026):
    # When belief cannot be trusted, the only correct signal is EXIT.
    pit_exit_triggered: bool = False        # True if PIT violation requires EXIT
    pit_exit_reason: Optional[str] = None   # Human-readable exit reason
    pit_violation_severity: float = 0.0     # V ∈ [0, 1], 0 = no violation
    pit_penalty_effective: float = 1.0      # P ∈ (0, 1], 1 = no penalty
    pit_selected_model: Optional[str] = None  # Model that triggered the EXIT check
    # Volatility Estimator (February 2026):
    volatility_estimator: Optional[str] = None  # "GK", "HAR-GK", "EWMA", etc.
    # Enhanced Mixture (February 2026):
    mixture_enhanced: bool = False          # True if enhanced multi-factor mixture weights used
    # VIX-based ν adjustment (February 2026):
    vix_nu_adjustment_applied: bool = False  # True if VIX-based ν adjustment was applied
    nu_original: Optional[float] = None      # Original ν before VIX adjustment
    nu_adjusted: Optional[float] = None      # ν after VIX adjustment
    # CRPS-based model selection (February 2026):
    crps_score: Optional[float] = None       # Model's CRPS score (lower is better)
    scoring_weights: Optional[Dict[str, float]] = None  # Regime-aware weights used {bic, hyvarinen, crps}
    scoring_method: Optional[str] = None     # "regime_aware_bic_hyv_crps" or "bic_only"
    # Story 3.6: MC Path Diagnostics
    mc_diagnostics: Optional[Dict[str, float]] = None  # diagnose_mc_paths() output
    # Story 5.2: Balanced EU fields
    eu_asymmetric: float = 0.0         # Legacy EU with asymmetric EVT correction
    eu_balanced: float = 0.0           # Balanced EU with symmetric tail treatment
    # Story 5.6: Kelly sizing fields
    kelly_full: float = 0.0            # Full Kelly fraction
    kelly_half: float = 0.0            # Half-Kelly (conservative)

