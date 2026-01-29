#!/usr/bin/env python3
"""
===============================================================================
ISOTONIC RECALIBRATION — PROBABILITY TRANSPORT OPERATOR
===============================================================================

This module implements isotonic regression-based recalibration as a first-class
probabilistic transport operator in the Bayesian Model Averaging system.

CORE DOCTRINE:
    "Inference generates beliefs. Regimes provide context. 
     Calibration aligns beliefs with reality. Trust is updated continuously."

Calibration is NOT a validator, patch, or rejection criterion.
Calibration IS a learned probability transport map applied BEFORE regimes 
affect trust, weighting, or decisions.

ARCHITECTURE:
    Model Inference → Raw PIT → Transport Map g_model → Calibrated PIT
                                      ↓
                   Regime-Conditioned Diagnostics → Model Weight Updates

KEY ENFORCEMENT RULE:
    Raw PIT is never used directly by regimes, diagnostics, or escalation logic.
    Regimes see calibrated probability, not raw belief.

TRANSPORT MAP:
    g: [0,1] → [0,1]
    
    Properties:
    - Monotone (preserves probability ranking)
    - Learned from data (not assumed)
    - Persisted and versioned with model parameters
    
    Implementation: Isotonic regression
    - Fit on sorted (raw_pit, empirical_quantile) pairs
    - Empirical quantile = (rank - 0.5) / n
    - Clipped to [0.001, 0.999] for numerical stability

NON-DEGRADATION GUARANTEE:
    Under monotone probability loss, isotonic regression is guaranteed to not
    worsen in-sample calibration. Out-of-sample, we use validation split and
    fall back to identity mapping if fit fails.

REFERENCE:
    Zadrozny & Elkan (2002) "Transforming Classifier Scores into Accurate
    Multiclass Probability Estimates"

===============================================================================
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy import stats
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IsotonicRecalibrationConfig:
    """Configuration for isotonic recalibration transport map."""
    
    # Enable/disable recalibration
    enabled: bool = True
    
    # PIT value bounds (for numerical stability)
    pit_min: float = 0.001
    pit_max: float = 0.999
    
    # Minimum observations required for fitting
    min_observations: int = 50
    
    # Validation split for out-of-sample check
    validation_split: float = 0.2
    
    # Minimum improvement required to accept recalibration
    # (KS statistic reduction in validation set)
    min_ks_improvement: float = 0.0  # Accept any non-degradation
    
    # Maximum complexity (number of isotonic segments)
    # None = no limit
    max_segments: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary."""
        return {
            'enabled': self.enabled,
            'pit_min': self.pit_min,
            'pit_max': self.pit_max,
            'min_observations': self.min_observations,
            'validation_split': self.validation_split,
            'min_ks_improvement': self.min_ks_improvement,
            'max_segments': self.max_segments,
        }


# Default configuration
DEFAULT_RECALIBRATION_CONFIG = IsotonicRecalibrationConfig()


# =============================================================================
# TRANSPORT MAP RESULT
# =============================================================================

@dataclass
class TransportMapResult:
    """
    Result of fitting an isotonic transport map.
    
    The transport map g: [0,1] → [0,1] is represented by:
    - x_knots: Input knot points (raw PIT values)
    - y_knots: Output knot points (calibrated PIT values)
    
    For interpolation between knots, use linear interpolation.
    """
    
    # Transport map representation
    x_knots: np.ndarray  # Input knots (raw PIT)
    y_knots: np.ndarray  # Output knots (calibrated PIT)
    
    # Fit diagnostics
    n_observations: int
    n_segments: int
    
    # Calibration improvement metrics
    raw_ks_statistic: float
    raw_ks_pvalue: float
    calibrated_ks_statistic: float
    calibrated_ks_pvalue: float
    
    # Validation metrics (out-of-sample)
    validation_ks_statistic: Optional[float] = None
    validation_ks_pvalue: Optional[float] = None
    
    # Status
    fit_success: bool = True
    is_identity: bool = False  # True if g(x) ≈ x (already calibrated)
    fallback_to_identity: bool = False  # True if fitting failed
    warning_message: Optional[str] = None
    
    @property
    def ks_improvement(self) -> float:
        """KS statistic improvement (positive = better)."""
        return self.raw_ks_statistic - self.calibrated_ks_statistic
    
    @property
    def pvalue_improvement_ratio(self) -> float:
        """P-value improvement ratio (>1 = better)."""
        if self.raw_ks_pvalue <= 0 or self.calibrated_ks_pvalue <= 0:
            return 1.0
        return self.calibrated_ks_pvalue / self.raw_ks_pvalue
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result to dictionary for JSON serialization."""
        return {
            'x_knots': self.x_knots.tolist() if isinstance(self.x_knots, np.ndarray) else list(self.x_knots),
            'y_knots': self.y_knots.tolist() if isinstance(self.y_knots, np.ndarray) else list(self.y_knots),
            'n_observations': int(self.n_observations),
            'n_segments': int(self.n_segments),
            'raw_ks_statistic': float(self.raw_ks_statistic),
            'raw_ks_pvalue': float(self.raw_ks_pvalue),
            'calibrated_ks_statistic': float(self.calibrated_ks_statistic),
            'calibrated_ks_pvalue': float(self.calibrated_ks_pvalue),
            'validation_ks_statistic': float(self.validation_ks_statistic) if self.validation_ks_statistic is not None else None,
            'validation_ks_pvalue': float(self.validation_ks_pvalue) if self.validation_ks_pvalue is not None else None,
            'fit_success': bool(self.fit_success),
            'is_identity': bool(self.is_identity),
            'fallback_to_identity': bool(self.fallback_to_identity),
            'warning_message': self.warning_message,
            'ks_improvement': float(self.ks_improvement),
            'pvalue_improvement_ratio': float(self.pvalue_improvement_ratio),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TransportMapResult':
        """Load result from dictionary."""
        return cls(
            x_knots=np.array(d['x_knots']),
            y_knots=np.array(d['y_knots']),
            n_observations=d['n_observations'],
            n_segments=d['n_segments'],
            raw_ks_statistic=d['raw_ks_statistic'],
            raw_ks_pvalue=d['raw_ks_pvalue'],
            calibrated_ks_statistic=d['calibrated_ks_statistic'],
            calibrated_ks_pvalue=d['calibrated_ks_pvalue'],
            validation_ks_statistic=d.get('validation_ks_statistic'),
            validation_ks_pvalue=d.get('validation_ks_pvalue'),
            fit_success=d.get('fit_success', True),
            is_identity=d.get('is_identity', False),
            fallback_to_identity=d.get('fallback_to_identity', False),
            warning_message=d.get('warning_message'),
        )
    
    @classmethod
    def identity_map(cls, n_obs: int = 0, raw_ks_stat: float = 0.0, raw_ks_pval: float = 1.0) -> 'TransportMapResult':
        """Create identity transport map (no recalibration)."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        return cls(
            x_knots=x,
            y_knots=y,
            n_observations=n_obs,
            n_segments=1,
            raw_ks_statistic=raw_ks_stat,
            raw_ks_pvalue=raw_ks_pval,
            calibrated_ks_statistic=raw_ks_stat,
            calibrated_ks_pvalue=raw_ks_pval,
            fit_success=True,
            is_identity=True,
            fallback_to_identity=False,
        )


# =============================================================================
# ISOTONIC RECALIBRATOR
# =============================================================================

class IsotonicRecalibrator:
    """
    Isotonic regression-based probability transport map.
    
    This class learns a monotonic transformation g: [0,1] → [0,1] that maps
    raw PIT values to calibrated PIT values.
    
    Usage:
        # Fit during tuning
        recalibrator = IsotonicRecalibrator(config)
        result = recalibrator.fit(raw_pit_values)
        
        # Apply during inference
        calibrated_pit = recalibrator.transform(new_raw_pit)
        
        # Persist
        result_dict = result.to_dict()
        
        # Load
        recalibrator.load(TransportMapResult.from_dict(result_dict))
    """
    
    def __init__(self, config: Optional[IsotonicRecalibrationConfig] = None):
        """Initialize recalibrator with configuration."""
        self.config = config or DEFAULT_RECALIBRATION_CONFIG
        self.result: Optional[TransportMapResult] = None
        self._isotonic = None
    
    def fit(self, raw_pit: np.ndarray) -> TransportMapResult:
        """
        Fit isotonic transport map on raw PIT values.
        
        The transport map is learned by fitting isotonic regression on:
            (raw_pit_sorted, empirical_quantile)
        
        where empirical_quantile = (rank - 0.5) / n
        
        Args:
            raw_pit: Array of raw PIT values from base model
            
        Returns:
            TransportMapResult containing the learned transport map
        """
        from sklearn.isotonic import IsotonicRegression
        
        n = len(raw_pit)
        
        # Validate input
        if n < self.config.min_observations:
            # Insufficient data - return identity map
            ks_stat, ks_pval = self._compute_ks(raw_pit)
            return TransportMapResult.identity_map(
                n_obs=n,
                raw_ks_stat=ks_stat,
                raw_ks_pval=ks_pval,
            )
        
        # Clip raw PIT to valid range
        raw_pit_clipped = np.clip(raw_pit, self.config.pit_min, self.config.pit_max)
        
        # Compute raw calibration metrics
        raw_ks_stat, raw_ks_pval = self._compute_ks(raw_pit_clipped)
        
        # Check if already well-calibrated
        if raw_ks_pval >= 0.10:  # Very well calibrated
            self.result = TransportMapResult.identity_map(
                n_obs=n,
                raw_ks_stat=raw_ks_stat,
                raw_ks_pval=raw_ks_pval,
            )
            return self.result
        
        # Split into training and validation sets
        n_val = max(int(n * self.config.validation_split), 10)
        n_train = n - n_val
        
        if n_train < self.config.min_observations // 2:
            # Not enough for train/val split - use all data
            indices = np.arange(n)
            np.random.shuffle(indices)
            train_idx = indices
            val_idx = None
        else:
            indices = np.arange(n)
            np.random.shuffle(indices)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
        
        train_pit = raw_pit_clipped[train_idx]
        
        # Sort training PIT values
        sort_idx = np.argsort(train_pit)
        sorted_pit = train_pit[sort_idx]
        
        # Compute empirical quantiles (what PIT should be if calibrated)
        n_train_actual = len(sorted_pit)
        empirical_quantiles = (np.arange(1, n_train_actual + 1) - 0.5) / n_train_actual
        
        # Fit isotonic regression
        try:
            self._isotonic = IsotonicRegression(
                y_min=self.config.pit_min,
                y_max=self.config.pit_max,
                out_of_bounds='clip'
            )
            self._isotonic.fit(sorted_pit, empirical_quantiles)
            
            # Extract knots for persistence
            # Isotonic regression stores the mapping as (X_, y_)
            x_knots = self._isotonic.X_thresholds_ if hasattr(self._isotonic, 'X_thresholds_') else sorted_pit
            y_knots = self._isotonic.y_thresholds_ if hasattr(self._isotonic, 'y_thresholds_') else self._isotonic.predict(sorted_pit)
            
            # If sklearn version doesn't expose thresholds, create from unique points
            if not hasattr(self._isotonic, 'X_thresholds_'):
                # Get unique prediction points
                predictions = self._isotonic.predict(sorted_pit)
                unique_mask = np.concatenate([[True], np.diff(predictions) != 0])
                x_knots = sorted_pit[unique_mask]
                y_knots = predictions[unique_mask]
                
                # Add endpoints if missing
                if x_knots[0] > self.config.pit_min:
                    x_knots = np.concatenate([[self.config.pit_min], x_knots])
                    y_knots = np.concatenate([[self.config.pit_min], y_knots])
                if x_knots[-1] < self.config.pit_max:
                    x_knots = np.concatenate([x_knots, [self.config.pit_max]])
                    y_knots = np.concatenate([y_knots, [self.config.pit_max]])
            
        except Exception as e:
            # Fitting failed - return identity map with warning
            self.result = TransportMapResult.identity_map(
                n_obs=n,
                raw_ks_stat=raw_ks_stat,
                raw_ks_pval=raw_ks_pval,
            )
            self.result.fallback_to_identity = True
            self.result.warning_message = f"Isotonic fitting failed: {e}"
            return self.result
        
        # Apply recalibration to compute calibrated metrics
        calibrated_pit = self._isotonic.predict(raw_pit_clipped)
        cal_ks_stat, cal_ks_pval = self._compute_ks(calibrated_pit)
        
        # Validation metrics (if available)
        val_ks_stat = None
        val_ks_pval = None
        if val_idx is not None and len(val_idx) >= 10:
            val_pit = raw_pit_clipped[val_idx]
            val_calibrated = self._isotonic.predict(val_pit)
            val_ks_stat, val_ks_pval = self._compute_ks(val_calibrated)
            
            # Check for degradation in validation set
            val_raw_ks, _ = self._compute_ks(val_pit)
            if val_ks_stat > val_raw_ks + 0.05:  # Significant degradation
                # Fall back to identity
                self.result = TransportMapResult.identity_map(
                    n_obs=n,
                    raw_ks_stat=raw_ks_stat,
                    raw_ks_pval=raw_ks_pval,
                )
                self.result.fallback_to_identity = True
                self.result.warning_message = "Validation degradation detected, using identity map"
                return self.result
        
        # Count segments
        n_segments = len(x_knots) - 1 if len(x_knots) > 1 else 1
        
        # Check if effectively identity (all knots on diagonal)
        is_identity = np.allclose(x_knots, y_knots, atol=0.01)
        
        self.result = TransportMapResult(
            x_knots=np.array(x_knots),
            y_knots=np.array(y_knots),
            n_observations=n,
            n_segments=n_segments,
            raw_ks_statistic=raw_ks_stat,
            raw_ks_pvalue=raw_ks_pval,
            calibrated_ks_statistic=cal_ks_stat,
            calibrated_ks_pvalue=cal_ks_pval,
            validation_ks_statistic=val_ks_stat,
            validation_ks_pvalue=val_ks_pval,
            fit_success=True,
            is_identity=is_identity,
            fallback_to_identity=False,
        )
        
        return self.result
    
    def transform(self, raw_pit: np.ndarray) -> np.ndarray:
        """
        Apply learned transport map to raw PIT values.
        
        Args:
            raw_pit: Array of raw PIT values
            
        Returns:
            Array of calibrated PIT values
        """
        if self.result is None:
            raise ValueError("Recalibrator not fitted. Call fit() first or load().")
        
        if self.result.is_identity or self.result.fallback_to_identity:
            # Identity map - return clipped input
            return np.clip(raw_pit, self.config.pit_min, self.config.pit_max)
        
        # Use sklearn isotonic if available
        if self._isotonic is not None:
            return self._isotonic.predict(np.clip(raw_pit, self.config.pit_min, self.config.pit_max))
        
        # Otherwise, interpolate from knots
        return np.interp(
            np.clip(raw_pit, self.config.pit_min, self.config.pit_max),
            self.result.x_knots,
            self.result.y_knots
        )
    
    def load(self, result: TransportMapResult) -> None:
        """
        Load a previously fitted transport map.
        
        Args:
            result: TransportMapResult from previous fit
        """
        self.result = result
        
        # Rebuild sklearn isotonic for transform (if not identity)
        if not result.is_identity and not result.fallback_to_identity:
            from sklearn.isotonic import IsotonicRegression
            self._isotonic = IsotonicRegression(
                y_min=self.config.pit_min,
                y_max=self.config.pit_max,
                out_of_bounds='clip'
            )
            # Fit on knots to rebuild internal state
            self._isotonic.fit(result.x_knots, result.y_knots)
    
    def _compute_ks(self, pit: np.ndarray) -> Tuple[float, float]:
        """Compute KS test statistic and p-value against uniform."""
        try:
            result = stats.kstest(pit, 'uniform')
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return 1.0, 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_raw_pit_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    vol: np.ndarray,
    P_filtered: np.ndarray,
    c: float = 1.0
) -> np.ndarray:
    """
    Compute raw PIT values for Gaussian model.
    
    PIT(r_t) = Φ((r_t - μ_t) / σ_t)
    
    where σ_t = √(c · vol_t² + P_t)
    """
    # Observation variance
    R = c * (vol ** 2) + P_filtered
    sigma = np.sqrt(np.maximum(R, 1e-10))
    
    # Standardized residuals
    z = (returns - mu_filtered) / sigma
    
    # PIT via standard normal CDF
    pit = stats.norm.cdf(z)
    
    return np.clip(pit, 0.001, 0.999)


def compute_raw_pit_student_t(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    vol: np.ndarray,
    P_filtered: np.ndarray,
    c: float,
    nu: float
) -> np.ndarray:
    """
    Compute raw PIT values for Student-t model.
    
    PIT(r_t) = F_t((r_t - μ_t) / σ_t; ν)
    
    where σ_t = √(c · vol_t² + P_t) and F_t is Student-t CDF.
    """
    # Observation variance
    R = c * (vol ** 2) + P_filtered
    sigma = np.sqrt(np.maximum(R, 1e-10))
    
    # Standardized residuals
    z = (returns - mu_filtered) / sigma
    
    # PIT via Student-t CDF
    pit = stats.t.cdf(z, df=nu)
    
    return np.clip(pit, 0.001, 0.999)


def fit_recalibrator_for_asset(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    vol: np.ndarray,
    P_filtered: np.ndarray,
    c: float,
    nu: Optional[float] = None,
    config: Optional[IsotonicRecalibrationConfig] = None
) -> TransportMapResult:
    """
    Fit isotonic recalibrator for a single asset.
    
    This is the main entry point for fitting during tuning.
    
    Args:
        returns: Return series
        mu_filtered: Kalman-filtered drift estimates
        vol: EWMA volatility
        P_filtered: Kalman state variance
        c: Observation variance multiplier
        nu: Degrees of freedom (None for Gaussian)
        config: Recalibration configuration
        
    Returns:
        TransportMapResult containing the fitted transport map
    """
    config = config or DEFAULT_RECALIBRATION_CONFIG
    
    if not config.enabled:
        # Return identity map
        return TransportMapResult.identity_map(n_obs=len(returns))
    
    # Compute raw PIT based on model type
    if nu is not None and nu > 2:
        raw_pit = compute_raw_pit_student_t(returns, mu_filtered, vol, P_filtered, c, nu)
    else:
        raw_pit = compute_raw_pit_gaussian(returns, mu_filtered, vol, P_filtered, c)
    
    # Fit recalibrator
    recalibrator = IsotonicRecalibrator(config)
    result = recalibrator.fit(raw_pit)
    
    return result


def apply_recalibration(
    raw_pit: np.ndarray,
    transport_map: TransportMapResult
) -> np.ndarray:
    """
    Apply a fitted transport map to raw PIT values.
    
    This is the main entry point for applying recalibration during inference.
    
    Args:
        raw_pit: Array of raw PIT values
        transport_map: Previously fitted TransportMapResult
        
    Returns:
        Array of calibrated PIT values
    """
    recalibrator = IsotonicRecalibrator()
    recalibrator.load(transport_map)
    return recalibrator.transform(raw_pit)


# =============================================================================
# ENHANCED DIAGNOSTICS
# =============================================================================

def compute_calibration_diagnostics(
    pit_values: np.ndarray,
    returns: Optional[np.ndarray] = None,
    vol_proxy: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive calibration diagnostics.
    
    These diagnostics identify HOW calibration fails, not just THAT it fails.
    They are used for classification and research, never for triggering
    escalation or model changes.
    
    Args:
        pit_values: PIT values (should be calibrated)
        returns: Original returns (for context)
        vol_proxy: Volatility proxy for regime analysis
        
    Returns:
        Dictionary of diagnostic metrics
    """
    n = len(pit_values)
    diagnostics = {'n_observations': n}
    
    if n < 20:
        diagnostics['insufficient_data'] = True
        return diagnostics
    
    # 1. Basic PIT distribution statistics
    diagnostics['pit_mean'] = float(np.mean(pit_values))
    diagnostics['pit_std'] = float(np.std(pit_values))
    diagnostics['pit_skew'] = float(stats.skew(pit_values))
    diagnostics['pit_kurtosis'] = float(stats.kurtosis(pit_values))
    
    # Expected values for uniform distribution
    expected_mean = 0.5
    expected_std = 1 / np.sqrt(12)  # ≈ 0.289
    
    diagnostics['pit_mean_deviation'] = diagnostics['pit_mean'] - expected_mean
    diagnostics['pit_std_ratio'] = diagnostics['pit_std'] / expected_std
    
    # 2. Quantile deviations
    expected_quantiles = np.linspace(0.1, 0.9, 9)
    observed_quantiles = np.array([np.mean(pit_values <= q) for q in expected_quantiles])
    quantile_deviations = observed_quantiles - expected_quantiles
    
    diagnostics['quantile_deviations'] = {
        f'q{int(q*100)}': float(dev) 
        for q, dev in zip(expected_quantiles, quantile_deviations)
    }
    diagnostics['max_quantile_deviation'] = float(np.max(np.abs(quantile_deviations)))
    
    # 3. Tail calibration
    left_tail_coverage = float(np.mean(pit_values <= 0.05))
    right_tail_coverage = float(np.mean(pit_values >= 0.95))
    
    diagnostics['left_tail_coverage'] = left_tail_coverage
    diagnostics['right_tail_coverage'] = right_tail_coverage
    diagnostics['left_tail_bias'] = left_tail_coverage - 0.05
    diagnostics['right_tail_bias'] = right_tail_coverage - 0.05
    
    # 4. KS test
    ks_result = stats.kstest(pit_values, 'uniform')
    diagnostics['ks_statistic'] = float(ks_result.statistic)
    diagnostics['ks_pvalue'] = float(ks_result.pvalue)
    diagnostics['is_calibrated'] = ks_result.pvalue >= 0.05
    
    # 5. Conditional calibration by volatility regime
    if vol_proxy is not None and len(vol_proxy) == n:
        vol_median = np.median(vol_proxy)
        low_vol_mask = vol_proxy <= vol_median
        high_vol_mask = vol_proxy > vol_median
        
        if np.sum(low_vol_mask) >= 20 and np.sum(high_vol_mask) >= 20:
            ks_low = stats.kstest(pit_values[low_vol_mask], 'uniform')
            ks_high = stats.kstest(pit_values[high_vol_mask], 'uniform')
            
            diagnostics['low_vol_ks_statistic'] = float(ks_low.statistic)
            diagnostics['low_vol_ks_pvalue'] = float(ks_low.pvalue)
            diagnostics['high_vol_ks_statistic'] = float(ks_high.statistic)
            diagnostics['high_vol_ks_pvalue'] = float(ks_high.pvalue)
            diagnostics['worse_regime'] = 'high_vol' if ks_high.pvalue < ks_low.pvalue else 'low_vol'
    
    # 6. PIT autocorrelation (temporal dependence)
    if n > 20:
        diagnostics['pit_autocorr_lag1'] = float(np.corrcoef(pit_values[:-1], pit_values[1:])[0, 1])
        if n > 10:
            diagnostics['pit_autocorr_lag5'] = float(np.corrcoef(pit_values[:-5], pit_values[5:])[0, 1])
    
    # 7. Failure category classification
    diagnostics['failure_category'] = classify_calibration_failure(diagnostics)
    
    return diagnostics


def classify_calibration_failure(diag: Dict[str, Any]) -> str:
    """
    Classify the type of calibration failure.
    
    This classification is for RESEARCH AND LOGGING ONLY.
    It NEVER triggers escalation or model changes.
    
    Args:
        diag: Diagnostic dictionary
        
    Returns:
        Pipe-separated string of failure categories
    """
    categories = []
    
    # Tail miscalibration
    left_bias = diag.get('left_tail_bias', 0)
    right_bias = diag.get('right_tail_bias', 0)
    
    if left_bias > 0.02:
        categories.append('LEFT_TAIL_UNDERESTIMATED')
    elif left_bias < -0.02:
        categories.append('LEFT_TAIL_OVERESTIMATED')
    
    if right_bias > 0.02:
        categories.append('RIGHT_TAIL_UNDERESTIMATED')
    elif right_bias < -0.02:
        categories.append('RIGHT_TAIL_OVERESTIMATED')
    
    # Location bias
    pit_mean = diag.get('pit_mean', 0.5)
    if pit_mean < 0.45:
        categories.append('DRIFT_OVERESTIMATED')
    elif pit_mean > 0.55:
        categories.append('DRIFT_UNDERESTIMATED')
    
    # Scale bias
    pit_std_ratio = diag.get('pit_std_ratio', 1.0)
    if pit_std_ratio < 0.85:
        categories.append('VOLATILITY_OVERESTIMATED')
    elif pit_std_ratio > 1.15:
        categories.append('VOLATILITY_UNDERESTIMATED')
    
    # Regime-dependent miscalibration
    worse_regime = diag.get('worse_regime')
    if worse_regime == 'high_vol' and diag.get('high_vol_ks_pvalue', 1.0) < 0.05:
        categories.append('HIGH_VOL_REGIME_MISCALIBRATED')
    elif worse_regime == 'low_vol' and diag.get('low_vol_ks_pvalue', 1.0) < 0.05:
        categories.append('LOW_VOL_REGIME_MISCALIBRATED')
    
    # Temporal dependence
    if abs(diag.get('pit_autocorr_lag1', 0)) > 0.1:
        categories.append('TEMPORAL_DEPENDENCE_UNMODELED')
    
    return '|'.join(categories) if categories else 'CALIBRATED'


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'IsotonicRecalibrationConfig',
    'DEFAULT_RECALIBRATION_CONFIG',
    'TransportMapResult',
    'IsotonicRecalibrator',
    'compute_raw_pit_gaussian',
    'compute_raw_pit_student_t',
    'fit_recalibrator_for_asset',
    'apply_recalibration',
    'compute_calibration_diagnostics',
    'classify_calibration_failure',
]
