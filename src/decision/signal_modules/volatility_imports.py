from __future__ import annotations
"""
Volatility framework imports: conditional dependencies for vol models,
enhanced distributional models, EVT, model registry, and scoring.

Extracted from signals.py (Story 5.2). Contains all conditional import
blocks for volatility estimation, distributional models, and scoring
frameworks. Each import is guarded with try/except and provides a
*_AVAILABLE flag.
"""
import os
import sys
import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, t as student_t

# Ensure src directory is in path for imports
_VOLIMPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DECISION_DIR = os.path.dirname(_VOLIMPORTS_DIR)
_SRC_DIR = os.path.dirname(_DECISION_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# =============================================================================
# GARMAN-KLASS REALIZED VOLATILITY (February 2026)
# =============================================================================
# Range-based volatility estimator using OHLC data.
# 7.4x more efficient than close-to-close EWMA.
# Improves PIT calibration without adding parameters.
# =============================================================================
try:
    from calibration.realized_volatility import (
        compute_gk_volatility,
        compute_hybrid_volatility,
        compute_hybrid_volatility_har,
        compute_volatility_from_df,
        VolatilityEstimator,
        HAR_WEIGHT_DAILY,
        HAR_WEIGHT_WEEKLY,
        HAR_WEIGHT_MONTHLY,
    )
    GK_VOLATILITY_AVAILABLE = True
    HAR_VOLATILITY_AVAILABLE = True
except ImportError:
    GK_VOLATILITY_AVAILABLE = False
    HAR_VOLATILITY_AVAILABLE = False


# =============================================================================
# CRPS COMPUTATION FOR MODEL SELECTION (February 2026)
# =============================================================================
# CRPS is a strictly proper scoring rule for calibration + sharpness.
# Used for regime-aware model selection in conjunction with BIC and Hyvärinen.
# =============================================================================
try:
    from tuning.diagnostics import (
        compute_crps_gaussian_inline,
        compute_crps_student_t_inline,
        compute_regime_aware_model_weights,
        REGIME_SCORING_WEIGHTS,
        CRPS_SCORING_ENABLED,
        # LFO-CV for out-of-sample model selection (February 2026)
        compute_lfo_cv_score_gaussian,
        compute_lfo_cv_score_student_t,
        LFO_CV_ENABLED,
    )
    CRPS_AVAILABLE = True
    LFO_CV_AVAILABLE = True
except ImportError:
    CRPS_AVAILABLE = False
    CRPS_SCORING_ENABLED = False
    LFO_CV_AVAILABLE = False
    LFO_CV_ENABLED = False


class StudentTDriftModel:
    """Minimal Student-t helper used for Kalman log-likelihood and mapping."""

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        if scale <= 0 or nu <= 0:
            return -1e12
        z = (x - mu) / scale
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)


# =============================================================================
# ENHANCED STUDENT-T MODELS (February 2026)
# =============================================================================
# Import PhiStudentTDriftModel for Vol-of-Vol, Two-Piece, and Mixture variants.
# These provide improved PIT/Hyvärinen calibration through:
#   1. Vol-of-Vol: R_t = c × σ² × (1 + γ × |Δlog(σ)|)
#   2. Two-Piece: νL ≠ νR for asymmetric crash/recovery tails
#   3. Mixture: w·t(νcalm) + (1-w)·t(νstress) with dynamic weights
# =============================================================================
try:
    from models.phi_student_t import PhiStudentTDriftModel
    ENHANCED_STUDENT_T_AVAILABLE = True
except ImportError:
    ENHANCED_STUDENT_T_AVAILABLE = False


# =============================================================================
# EVT (EXTREME VALUE THEORY) FOR POSITION SIZING
# =============================================================================
# Import POT/GPD tail modeling for EVT-corrected expected loss estimation.
# This provides principled extrapolation of tail losses beyond observed data.
#
# THEORETICAL FOUNDATION:
#   Pickands–Balkema–de Haan theorem: exceedances over high threshold → GPD
#   CTE = E[Loss | Loss > u] = u + σ/(1-ξ)  for ξ < 1
#
# INTEGRATION:
#   - Used in Expected Utility calculation to replace naive E[loss]
#   - Produces more conservative (larger) loss estimates for heavy-tailed assets
#   - Falls back to empirical × 1.5 if GPD fitting fails
# =============================================================================
try:
    from calibration.evt_tail import (
        compute_evt_expected_loss,
        compute_evt_var,
        fit_gpd_pot,
        GPDFitResult,
        EVT_THRESHOLD_PERCENTILE_DEFAULT,
        EVT_MIN_EXCEEDANCES,
        EVT_FALLBACK_MULTIPLIER,
        check_student_t_consistency,
    )
    EVT_AVAILABLE = True
except ImportError:
    EVT_AVAILABLE = False
    EVT_THRESHOLD_PERCENTILE_DEFAULT = 0.90
    EVT_MIN_EXCEEDANCES = 30
    EVT_FALLBACK_MULTIPLIER = 1.15

# Story 5.2: EVT inflation cap and balanced EU constants
EVT_MAX_INFLATION = 1.5   # E_loss_evt <= EVT_MAX_INFLATION * E_loss_empirical
EVT_GAIN_FACTOR = 0.5     # Symmetric gain correction (conservative: half the loss factor)

# Story 5.3: Forecast-magnitude-aware position sizing blend weights
SIZE_EU_WEIGHT = 0.6   # Weight on EU-based (risk-adjusted) sizing
SIZE_MAG_WEIGHT = 0.4  # Weight on forecast-magnitude (conviction) sizing


# =============================================================================
# CONTAMINATED STUDENT-T DISTRIBUTION
# =============================================================================
# Regime-indexed contaminated Student-t mixture for crisis tail modeling.
# Models returns as: (1-ε)·t(ν_normal) + ε·t(ν_crisis) where ε is contamination.
# =============================================================================
try:
    from models import (
        contaminated_student_t_rvs,
        ContaminatedStudentTParams,
    )
    CONTAMINATED_ST_AVAILABLE = True
except ImportError:
    CONTAMINATED_ST_AVAILABLE = False
    
    # Fallback: simple contaminated sampling without the module
    def contaminated_student_t_rvs(
        size: int,
        nu_normal: float,
        nu_crisis: float,
        epsilon: float,
        mu: float = 0.0,
        sigma: float = 1.0,
        random_state=None
    ):
        """Fallback contaminated student-t sampling."""
        import numpy as np
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Sample component indicators
        from_crisis = rng.random(size) < epsilon
        n_crisis = from_crisis.sum()
        n_normal = size - n_crisis
        
        samples = np.empty(size)
        if n_normal > 0:
            samples[~from_crisis] = rng.standard_t(df=nu_normal, size=n_normal)
        if n_crisis > 0:
            samples[from_crisis] = rng.standard_t(df=nu_crisis, size=n_crisis)
        
        return mu + sigma * samples


# =============================================================================
# HANSEN SKEW-T DISTRIBUTION (Asymmetric Tails)
# =============================================================================
# Hansen (1994) skew-t captures directional asymmetry via λ parameter:
#   - λ > 0: Right-skewed (recovery potential, heavier right tail)
#   - λ < 0: Left-skewed (crash risk, heavier left tail)
#   - λ = 0: Reduces to symmetric Student-t
#
# CRITICAL: This must be imported for signals.py to use hansen_lambda in MC sampling.
# Without this import, hansen_lambda is accepted but IGNORED - silent bug!
# =============================================================================
try:
    from models import (
        hansen_skew_t_rvs,
        hansen_skew_t_cdf,
        HansenSkewTParams,
    )
    HANSEN_SKEW_T_AVAILABLE = True
except ImportError:
    HANSEN_SKEW_T_AVAILABLE = False
    
    # Fallback: stub that falls back to symmetric Student-t (with warning)
    def hansen_skew_t_rvs(
        size: int,
        nu: float,
        lambda_: float,
        loc: float = 0.0,
        scale: float = 1.0,
        random_state=None
    ) -> np.ndarray:
        """Fallback Hansen skew-t sampling - uses symmetric Student-t with warning."""
        import warnings
        warnings.warn(
            f"Hansen skew-t not available, using symmetric Student-t (ignoring λ={lambda_:.3f})",
            RuntimeWarning
        )
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Fallback to symmetric Student-t
        samples = rng.standard_t(df=nu, size=size)
        return loc + scale * samples


# =============================================================================
# ENHANCED MIXTURE WEIGHT DYNAMICS (February 2026 - Expert Panel)
# =============================================================================
# Multi-factor weight conditioning for mixture Student-t models.
# w_t = sigmoid(a × z_t + b × Δσ_t + c × M_t)
# =============================================================================
ENHANCED_MIXTURE_ENABLED = True  # Set to False to use standard vol-only mixture

try:
    from models import (
        MIXTURE_WEIGHT_A_SHOCK,
        MIXTURE_WEIGHT_B_VOL_ACCEL,
        MIXTURE_WEIGHT_C_MOMENTUM,
    )
    ENHANCED_MIXTURE_AVAILABLE = True
except ImportError:
    ENHANCED_MIXTURE_AVAILABLE = False
    MIXTURE_WEIGHT_A_SHOCK = 1.0
    MIXTURE_WEIGHT_B_VOL_ACCEL = 0.5
    MIXTURE_WEIGHT_C_MOMENTUM = 0.3


# =============================================================================
# MARKOV-SWITCHING PROCESS NOISE (MS-q) — February 2026
# =============================================================================
# Proactive regime-switching q based on volatility structure.
# Unlike GAS-Q (reactive), MS-q shifts BEFORE errors materialize.
# =============================================================================
MS_Q_ENABLED = True  # Set to False to disable MS-q in signal generation

try:
    from models import (
        MS_Q_CALM_DEFAULT,
        MS_Q_STRESS_DEFAULT,
        MS_Q_SENSITIVITY,
        MS_Q_THRESHOLD,
        filter_phi_ms_q,
    )
    MS_Q_AVAILABLE = True
except ImportError:
    MS_Q_AVAILABLE = False
    MS_Q_CALM_DEFAULT = 1e-6
    MS_Q_STRESS_DEFAULT = 1e-4
    MS_Q_SENSITIVITY = 2.0
    MS_Q_THRESHOLD = 1.3
    filter_phi_ms_q = None


# =============================================================================
# MODEL REGISTRY — Single Source of Truth for Model Synchronization
# =============================================================================
# The model registry ensures tune.py and signals.py are ALWAYS synchronised.
# This prevents the #1 silent failure: model name mismatch → dropped from BMA.
#
# ARCHITECTURAL LAW: Top funds REFUSE TO TRADE without this assertion.
# =============================================================================
try:
    from models.model_registry import (
        MODEL_REGISTRY,
        ModelFamily,
        SupportType,
        get_model_spec,
        assert_models_synchronised,
    )
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    MODEL_REGISTRY_AVAILABLE = False


# =============================================================================
# RISK TEMPERATURE MODULATION LAYER (Expert Panel Solution 1 + 4)
# =============================================================================
# Risk temperature scales position sizes based on cross-asset stress indicators
# WITHOUT modifying distributional beliefs (Kalman state, BMA weights, GARCH).
#
# DESIGN PRINCIPLE:
#   "FX, futures, and commodities don't tell you WHERE to go.
#    They tell you HOW FAST you're allowed to drive."
#
# INTEGRATION:
#   - Computed AFTER EU-based sizing
#   - Applied BEFORE final position output
#   - Uses smooth sigmoid scaling (no cliff effects)
#   - Overnight budget constraint when temp > 1.0
#
# STRESS CATEGORIES (weighted sum):
#   - FX Stress (40%): AUDJPY, USDJPY z-scores — risk-on/off proxy
#   - Futures Stress (30%): ES/NQ overnight returns — equity sentiment
#   - Rates Stress (20%): TLT volatility — macro stress
#   - Commodity Stress (10%): Copper, gold/copper ratio — growth fear
#
# SCALING FUNCTION:
#   scale_factor(temp) = 1.0 / (1.0 + exp(3.0 × (temp - 1.0)))
#
# OVERNIGHT BUDGET:
#   When temp > 1.0: cap position such that position × gap ≤ budget
# =============================================================================
try:
    from decision.risk_temperature import (
        compute_risk_temperature,
        apply_risk_temperature_scaling,
        get_cached_risk_temperature,
        clear_risk_temperature_cache,
        RiskTemperatureResult,
        SIGMOID_THRESHOLD,
        OVERNIGHT_BUDGET_ACTIVATION_TEMP,
    )
    RISK_TEMPERATURE_AVAILABLE = True
except ImportError:
    RISK_TEMPERATURE_AVAILABLE = False
    SIGMOID_THRESHOLD = 1.0
    OVERNIGHT_BUDGET_ACTIVATION_TEMP = 1.0

# =============================================================================
# GAS-Q SCORE-DRIVEN PARAMETER DYNAMICS (February 2026)
# =============================================================================
# Implements Creal, Koopman & Lucas (2013) GAS dynamics for process noise q.
# q_t = omega + alpha * s_{t-1} + beta * q_{t-1}
# where s_t is the score (derivative of log-likelihood with respect to q).
#
# When gas_q_augmented=True in tuned params, the Kalman filter uses:
#   - Dynamic q_t that adapts to recent forecast errors
#   - Larger q during uncertainty spikes, smaller q during stable periods
#
# Expected Impact:
#   - 15-20% improvement in adaptive forecasting during regime transitions
#   - Better PIT calibration in volatile periods
# =============================================================================
try:
    from models.gas_q import (
        GASQConfig,
        GASQResult,
        DEFAULT_GAS_Q_CONFIG,
        gas_q_filter_gaussian,
        gas_q_filter_student_t,
        is_gas_q_enabled,
    )
    GAS_Q_AVAILABLE = True
except ImportError:
    GAS_Q_AVAILABLE = False
    # Stub definitions when GAS-Q module is unavailable
    class GASQConfig:
        def __init__(self, omega=1e-6, alpha=0.1, beta=0.5):
            self.omega = omega
            self.alpha = alpha
            self.beta = beta
    class GASQResult:
        pass
    DEFAULT_GAS_Q_CONFIG = None
    def gas_q_filter_gaussian(*args, **kwargs):
        return None
    def gas_q_filter_student_t(*args, **kwargs):
        return None
    def is_gas_q_enabled(*args, **kwargs):
        return False

# =============================================================================
# RV-ADAPTIVE PROCESS NOISE (Tune.md Epic 1)
# =============================================================================
# Proactive process noise q_t driven by realized volatility changes.
#   q_t = q_base * exp(gamma * delta_log(sigma_t^2))
# Unlike GAS-Q (reactive to errors), RV-Q responds to vol changes immediately.
# =============================================================================
try:
    from models.rv_adaptive_q import (
        RVAdaptiveQConfig,
        RVAdaptiveQResult,
        rv_adaptive_q_filter_gaussian,
        rv_adaptive_q_filter_student_t,
    )
    from models.model_registry import is_rv_q_model
    RV_Q_AVAILABLE = True
except ImportError:
    RV_Q_AVAILABLE = False
    def rv_adaptive_q_filter_gaussian(*args, **kwargs):
        return None
    def rv_adaptive_q_filter_student_t(*args, **kwargs):
        return None
    def is_rv_q_model(name):
        return False

# =============================================================================
# UNIFIED RISK CONTEXT (February 2026)
# =============================================================================
# Integrates all temperature modules (risk, metals, market) with copula-based
# tail dependence for institutional-grade crash risk estimation.
#
# PROFESSOR CHEN WEI-LIN (Score: 9.0/10):
#   "Copula models capture tail dependencies that Pearson correlation misses."
#
# PROFESSOR ZHANG XIN-YU (Score: 9.0/10):
#   "Unified architecture ensures risk signals translate to position sizing."
# =============================================================================
try:
    from calibration.copula_correlation import (
        compute_unified_risk_context,
        UnifiedRiskContext,
        compute_smooth_scale_factor,
        COPULA_CORRELATION_AVAILABLE,
    )
    UNIFIED_RISK_CONTEXT_AVAILABLE = True
except ImportError:
    UNIFIED_RISK_CONTEXT_AVAILABLE = False
    COPULA_CORRELATION_AVAILABLE = False


# =============================================================================
# ASSET-LEVEL CRASH RISK (February 2026 - Chinese Staff Professor Panel)
# =============================================================================
# Computes per-asset crash risk using multi-factor momentum analysis.
# Uses multiprocessing for parallel computation across assets.
# =============================================================================
try:
    from decision.asset_crash_risk import (
        compute_asset_crash_risk,
        compute_crash_risk_bulk,
        AssetCrashRiskResult,
        CrashRiskFactors,
        format_crash_risk_display,
        get_cached_crash_risk,
        cache_crash_risk,
        clear_crash_risk_cache,
    )
    ASSET_CRASH_RISK_AVAILABLE = True
except ImportError:
    ASSET_CRASH_RISK_AVAILABLE = False


# =============================================================================
# EPIC 8: SIGNAL ENRICHMENT MODULES (April 2026)
# =============================================================================
# Post-processing enrichment for conviction scoring, Kelly sizing, signal decay,
# earnings/macro event adjustments, and volatility surface context.
# All imports are guarded — missing modules degrade gracefully.
# =============================================================================

try:
    from decision.conviction_scoring import compute_conviction, rank_by_conviction
    CONVICTION_SCORING_AVAILABLE = True
except ImportError:
    CONVICTION_SCORING_AVAILABLE = False

try:
    from decision.kelly_sizing import (
        recommend_position_size,
        compute_kelly_from_quantiles,
        normalize_portfolio,
    )
    KELLY_SIZING_AVAILABLE = True
except ImportError:
    KELLY_SIZING_AVAILABLE = False

try:
    from decision.signal_decay import decay_signal, compute_half_life
    SIGNAL_DECAY_AVAILABLE = True
except ImportError:
    SIGNAL_DECAY_AVAILABLE = False

try:
    from decision.earnings_signal import (
        detect_earnings_window,
        adjust_for_earnings,
        find_nearest_earnings,
    )
    EARNINGS_SIGNAL_AVAILABLE = True
except ImportError:
    EARNINGS_SIGNAL_AVAILABLE = False

try:
    from decision.macro_events import (
        MacroEventCalendar,
        detect_event_proximity,
        adjust_for_macro_event,
    )
    MACRO_EVENTS_AVAILABLE = True
except ImportError:
    MACRO_EVENTS_AVAILABLE = False

try:
    from decision.vol_surface import compute_iv_context, adjust_forecast_with_iv
    VOL_SURFACE_AVAILABLE = True
except ImportError:
    VOL_SURFACE_AVAILABLE = False

try:
    from decision.sector_rotation import generate_rotation_signal, rank_sectors
    SECTOR_ROTATION_AVAILABLE = True
except ImportError:
    SECTOR_ROTATION_AVAILABLE = False

try:
    from decision.pair_trading import screen_pairs
    PAIR_TRADING_AVAILABLE = True
except ImportError:
    PAIR_TRADING_AVAILABLE = False

try:
    from models.quant_db import QuantDB
    QUANT_DB_AVAILABLE = True
except ImportError:
    QUANT_DB_AVAILABLE = False

try:
    from models.vectorized_ops import (
        vectorized_phi_forecast,
        vectorized_phi_variance,
        vectorized_bma_weights as vec_bma_weights,
        batch_monte_carlo_sample,
    )
    VECTORIZED_OPS_AVAILABLE = True
except ImportError:
    VECTORIZED_OPS_AVAILABLE = False

try:
    from models.computation_cache import ComputationCache
    COMPUTATION_CACHE_AVAILABLE = True
except ImportError:
    COMPUTATION_CACHE_AVAILABLE = False

# =============================================================================
# IMPORT PIT RECALIBRATION (Epic 26 - Enhanced Isotonic Recalibration)
# =============================================================================
try:
    from calibration.pit_recalibration import (
        isotonic_recalibrate,
        recalibration_schedule,
    )
    PIT_RECALIBRATION_AVAILABLE = True
except ImportError:
    PIT_RECALIBRATION_AVAILABLE = False

# =============================================================================
# IMPORT MISSING DATA HANDLING (Epic 29 - Gap-Aware Kalman Predict)
# =============================================================================
try:
    from calibration.missing_data import data_quality_score
    MISSING_DATA_AVAILABLE = True
except ImportError:
    MISSING_DATA_AVAILABLE = False

# =============================================================================
# IMPORT FORECAST ATTRIBUTION (Epic 27 - Drift/Vol Diagnostics)
# =============================================================================
try:
    from calibration.forecast_attribution import drift_attribution, volatility_attribution
    FORECAST_ATTRIBUTION_AVAILABLE = True
except ImportError:
    FORECAST_ATTRIBUTION_AVAILABLE = False

# =============================================================================
# IMPORT INTEGRATION TESTING (Epic 30 - Pipeline Output Validation)
# =============================================================================
try:
    from calibration.integration_testing import validate_pipeline_output
    INTEGRATION_TESTING_AVAILABLE = True
except ImportError:
    INTEGRATION_TESTING_AVAILABLE = False

# =============================================================================
# ONLINE C UPDATE (Story 2.3)
# =============================================================================
try:
    from calibration.online_c_update import run_online_c_update, OnlineCResult
    ONLINE_C_UPDATE_AVAILABLE = True
except ImportError:
    ONLINE_C_UPDATE_AVAILABLE = False

# =============================================================================
# SIGN PROBABILITY WITH UNCERTAINTY (Stories 5.1, 5.2, 5.3)
# =============================================================================
try:
    from calibration.sign_probability import (
        sign_prob_with_uncertainty,
        sign_prob_skewed,
        multi_horizon_sign_prob,
    )
    SIGN_PROBABILITY_AVAILABLE = True
except ImportError:
    SIGN_PROBABILITY_AVAILABLE = False

# =============================================================================
# LAPLACE POSTERIOR APPROXIMATION (Story 10.1)
# =============================================================================
try:
    from calibration.laplace_posterior import (
        laplace_posterior,
        sample_from_laplace,
        LaplaceResult,
    )
    LAPLACE_POSTERIOR_AVAILABLE = True
except ImportError:
    LAPLACE_POSTERIOR_AVAILABLE = False

# =============================================================================
# MC VARIANCE REDUCTION (Stories 10.2, 10.3)
# =============================================================================
try:
    from calibration.mc_variance_reduction import (
        importance_mc_student_t,
        antithetic_mc_sample,
        ImportanceSamplingResult,
        AntitheticResult,
    )
    MC_VARIANCE_REDUCTION_AVAILABLE = True
except ImportError:
    MC_VARIANCE_REDUCTION_AVAILABLE = False

# =============================================================================
# DIRECTIONAL CONFIDENCE CALIBRATION (Story 11.1)
# =============================================================================
try:
    from calibration.directional_confidence import platt_calibrate
    PLATT_CALIBRATE_AVAILABLE = True
except ImportError:
    PLATT_CALIBRATE_AVAILABLE = False

# =============================================================================
# UNCERTAINTY DECOMPOSITION (Story 11.2)
# =============================================================================
try:
    from calibration.uncertainty_decomposition import (
        decompose_uncertainty,
        UncertaintyDecomposition,
    )
    UNCERTAINTY_DECOMPOSITION_AVAILABLE = True
except ImportError:
    UNCERTAINTY_DECOMPOSITION_AVAILABLE = False

# =============================================================================
# REGIME CONFIDENCE SCALING (Story 11.3)
# =============================================================================
try:
    from calibration.regime_confidence import (
        regime_confidence_scale,
        RegimeConfidenceResult,
        RegimeHitRates,
    )
    REGIME_CONFIDENCE_AVAILABLE = True
except ImportError:
    REGIME_CONFIDENCE_AVAILABLE = False

# =============================================================================
# ADAPTIVE MOMENTUM & CROSS-ASSET (Stories 12.1, 12.2, 12.3)
# =============================================================================
try:
    from calibration.multi_timeframe_fusion import (
        adaptive_momentum_weights,
        momentum_mr_regime_indicator,
        cross_asset_confirmation,
        AdaptiveMomentumResult,
        MomentumMRRegime,
        CrossAssetConfirmation,
    )
    MULTI_TIMEFRAME_FUSION_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_FUSION_AVAILABLE = False

# =============================================================================
# KELLY SIZING (Stories 13.1, 13.2, 13.3)
# =============================================================================
try:
    from calibration.kelly_sizing import (
        kelly_fraction,
        drawdown_adjusted_kelly,
        auto_tune_kelly_frac,
    )
    KELLY_SIZING_CALIBRATION_AVAILABLE = True
except ImportError:
    KELLY_SIZING_CALIBRATION_AVAILABLE = False

# =============================================================================
# TRANSACTION COSTS (Stories 14.1, 14.2, 14.3)
# =============================================================================
try:
    from calibration.transaction_costs import (
        transaction_cost,
        turnover_filter,
        optimal_rebalance_freq,
        TransactionCostResult,
    )
    TRANSACTION_COSTS_AVAILABLE = True
except ImportError:
    TRANSACTION_COSTS_AVAILABLE = False

# =============================================================================
# REGIME POSITION SIZING (Stories 15.1, 15.2, 15.3)
# =============================================================================
try:
    from calibration.regime_position_sizing import (
        regime_position_limit,
        dynamic_leverage,
        vol_target_weight,
        RegimePositionResult,
        VolTargetResult,
    )
    REGIME_POSITION_SIZING_AVAILABLE = True
except ImportError:
    REGIME_POSITION_SIZING_AVAILABLE = False

# =============================================================================
# GARCH VARIANCE FORECAST (Story 16.3)
# =============================================================================
try:
    from models.gjr_garch import garch_variance_forecast, GARCHForecastResult
    GARCH_FORECAST_AVAILABLE = True
except ImportError:
    GARCH_FORECAST_AVAILABLE = False

# =============================================================================
# HANSEN SKEW-T DIRECTION (Story 17.3)
# =============================================================================
try:
    from models.hansen_skew_t import skew_adjusted_direction, SkewAdjustedDirectionResult
    SKEW_ADJUSTED_DIRECTION_AVAILABLE = True
except ImportError:
    SKEW_ADJUSTED_DIRECTION_AVAILABLE = False

# =============================================================================
# CONTAMINATED STUDENT-T SIGNALS (Stories 18.2, 18.3)
# =============================================================================
try:
    from models.contaminated_student_t import (
        cst_jump_probability,
        cst_prediction_interval,
        JumpProbabilityResult,
        CSTPredictionInterval,
    )
    CST_SIGNALS_AVAILABLE = True
except ImportError:
    CST_SIGNALS_AVAILABLE = False

# =============================================================================
# MEAN REVERSION SIGNAL STRENGTH (Story 20.3)
# =============================================================================
try:
    from models.ou_mean_reversion import mr_signal_strength, MRSignalResult
    MR_SIGNAL_STRENGTH_AVAILABLE = True
except ImportError:
    MR_SIGNAL_STRENGTH_AVAILABLE = False

# =============================================================================
# FACTOR-AUGMENTED CROSS-ASSET (Stories 22.1, 22.2, 22.3)
# =============================================================================
try:
    from calibration.factor_augmented import (
        extract_market_factors,
        factor_adjusted_R,
        granger_test,
        FactorExtractionResult,
        FactorAdjustedRResult,
        GrangerTestResult,
    )
    FACTOR_AUGMENTED_AVAILABLE = True
except ImportError:
    FACTOR_AUGMENTED_AVAILABLE = False

# =============================================================================
# VIX FORECAST ADJUSTMENTS (Stories 23.1, 23.2, 23.3)
# =============================================================================
try:
    from calibration.vix_forecast_adjustment import (
        vix_drift_adjustment,
        vix_term_structure_vol,
        detect_correlation_spike,
        VIXDriftAdjustmentResult,
        VIXTermStructureResult,
        CorrelationSpikeResult,
    )
    VIX_FORECAST_ADJUSTMENT_AVAILABLE = True
except ImportError:
    VIX_FORECAST_ADJUSTMENT_AVAILABLE = False

# =============================================================================
# ENSEMBLE FORECASTS (Stories 24.1, 24.2, 24.3)
# =============================================================================
try:
    from calibration.ensemble_forecast import (
        equal_weight_ensemble,
        trimmed_ensemble,
        online_prediction_pool,
        EqualWeightResult,
        TrimmedEnsembleResult,
        OnlinePredictionPoolResult,
    )
    ENSEMBLE_FORECAST_AVAILABLE = True
except ImportError:
    ENSEMBLE_FORECAST_AVAILABLE = False

# =============================================================================
# LOCATION-SCALE CORRECTION (Story 26.2)
# =============================================================================
try:
    from calibration.pit_recalibration import location_scale_correction, LocationScaleResult
    LOCATION_SCALE_CORRECTION_AVAILABLE = True
except ImportError:
    LOCATION_SCALE_CORRECTION_AVAILABLE = False

# =============================================================================
# BMA ATTRIBUTION (Story 27.3)
# =============================================================================
try:
    from calibration.forecast_attribution import bma_attribution, BMAAttributionResult
    BMA_ATTRIBUTION_AVAILABLE = True
except ImportError:
    BMA_ATTRIBUTION_AVAILABLE = False

# =============================================================================
# GAP-AWARE PREDICT (Story 29.1)
# =============================================================================
try:
    from calibration.missing_data import gap_aware_predict
    GAP_AWARE_PREDICT_AVAILABLE = True
except ImportError:
    GAP_AWARE_PREDICT_AVAILABLE = False


