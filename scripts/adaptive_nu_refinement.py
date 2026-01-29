#!/usr/bin/env python3
"""
===============================================================================
ADAPTIVE ν REFINEMENT FOR φ-t DISTRIBUTION CALIBRATION
===============================================================================

This module implements adaptive degrees-of-freedom refinement for the φ-t
distribution calibration pipeline. It addresses PIT calibration failures
by locally refining the ν grid for specific assets, without globally
expanding the model space.

CORE PRINCIPLE:
    Add resolution only where truth demands it.
    
    Do not expand the global grid. Refine locally based on diagnostic signals.
    Treat calibration failures as information, not embarrassment.

DESIGN PHILOSOPHY:
    - Most assets do NOT need finer ν resolution
    - The ~2-5% of failing assets tell us WHERE resolution is insufficient
    - Refinement is diagnostic, not optimization toward passing tests
    - Full auditability of refinement decisions

DETECTION CRITERION:
    An asset is flagged as "ν-resolution limited" when ALL conditions hold:
    1. Best ν is at grid boundary (12 or 20)
    2. PIT KS p-value < 0.05 (calibration failure)
    3. Model is φ-t variant (not Gaussian or mixture)
    4. Likelihood is locally flat in ν (ν not well identified)

LOCAL REFINEMENT CANDIDATES:
    - ν = 12 → test [10, 14] (between 8-12 and 12-20)
    - ν = 20 → test [16] only (asymmetric: ν > 30 ≈ Gaussian)

COMPUTATIONAL BUDGET:
    - Expected flagged assets: ~2-5% of universe
    - Additional likelihood evaluations: 1-2 per flagged asset
    - Total compute increase: ≤5%

===============================================================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AdaptiveNuConfig:
    """Configuration for adaptive ν refinement."""
    
    # Enable/disable adaptive refinement
    enabled: bool = True
    
    # Boundary ν values that trigger refinement check
    boundary_nu_values: Tuple[float, ...] = (12.0, 20.0)
    
    # PIT p-value threshold for calibration failure
    pit_threshold: float = 0.05
    
    # Log-likelihood flatness threshold
    # Refinement only triggered if |LL_best - LL_second| < threshold
    likelihood_flatness_threshold: float = 1.0
    
    # Refinement candidates for each boundary ν
    # Asymmetric design: ν=20 only tests downward (ν > 30 ≈ Gaussian)
    refinement_candidates: Dict[float, List[float]] = field(
        default_factory=lambda: {
            12.0: [10.0, 14.0],  # Test between 8-12 and 12-20
            20.0: [16.0],        # One-sided refinement only (downward)
        }
    )
    
    # Models eligible for refinement (must be φ-t variants)
    eligible_model_prefixes: Tuple[str, ...] = ('φ-T', 'phi_student_t')
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary."""
        return {
            'enabled': self.enabled,
            'boundary_nu_values': list(self.boundary_nu_values),
            'pit_threshold': self.pit_threshold,
            'likelihood_flatness_threshold': self.likelihood_flatness_threshold,
            'refinement_candidates': {str(k): v for k, v in self.refinement_candidates.items()},
            'eligible_model_prefixes': list(self.eligible_model_prefixes),
        }


# Default configuration
DEFAULT_ADAPTIVE_NU_CONFIG = AdaptiveNuConfig()


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def is_nu_likelihood_flat(
    result: Dict[str, Any],
    threshold: float = 1.0
) -> bool:
    """
    Return True if likelihood difference between best ν and second-best ν
    is small (i.e., ν not well identified).
    
    This prevents refinement when boundary ν is genuinely correct
    and PIT failure stems from other sources.
    
    Args:
        result: Calibration result dictionary
        threshold: Maximum log-likelihood difference for "flat" classification
        
    Returns:
        True if likelihood is locally flat in ν
    """
    # Try to get likelihood values from model_comparison
    model_comparison = result.get('model_comparison', {})
    
    # Find all Student-t models
    student_t_models = {
        k: v for k, v in model_comparison.items()
        if k.startswith('phi_student_t_nu_')
    }
    
    if len(student_t_models) < 2:
        # Not enough models to assess flatness
        return True  # Conservative: allow refinement
    
    # Sort by log-likelihood (descending)
    sorted_models = sorted(
        student_t_models.items(),
        key=lambda x: x[1].get('ll', -1e12),
        reverse=True
    )
    
    # Get best and second-best log-likelihoods
    ll_best = sorted_models[0][1].get('ll', -1e12)
    ll_second = sorted_models[1][1].get('ll', -1e12)
    
    loglik_diff = abs(ll_best - ll_second)
    
    return loglik_diff < threshold


def is_phi_t_model(model_name: str, config: AdaptiveNuConfig = None) -> bool:
    """
    Check if model is a φ-t variant eligible for refinement.
    
    Args:
        model_name: Model name string
        config: Configuration with eligible prefixes
        
    Returns:
        True if model is φ-t variant
    """
    config = config or DEFAULT_ADAPTIVE_NU_CONFIG
    
    if not model_name:
        return False
    
    for prefix in config.eligible_model_prefixes:
        if model_name.startswith(prefix):
            return True
    
    return False


def needs_nu_refinement(
    result: Dict[str, Any],
    config: AdaptiveNuConfig = None
) -> bool:
    """
    Flag asset as ν-resolution limited when ALL conditions hold:
    - Best ν is at grid boundary (12 or 20)
    - PIT KS p-value < 0.05 (calibration failure)
    - Model is φ-t variant (not Gaussian or mixture)
    - Likelihood is locally flat in ν (ν not well identified)
    
    The flatness check prevents over-refinement when boundary ν
    is actually correct and PIT failure has other causes.
    
    Args:
        result: Calibration result dictionary containing:
            - model: Model name (e.g., "φ-T(ν=12)")
            - nu: Degrees of freedom
            - pit_ks_pvalue: PIT KS test p-value
            - model_comparison: Dict of all fitted models
        config: Adaptive refinement configuration
        
    Returns:
        True if asset needs ν refinement
    """
    config = config or DEFAULT_ADAPTIVE_NU_CONFIG
    
    if not config.enabled:
        return False
    
    # Extract relevant fields
    model = result.get('model') or result.get('noise_model', '')
    nu = result.get('nu')
    pit_p = result.get('pit_ks_pvalue')
    
    # Condition 1: Model is φ-t variant
    is_phi_t = is_phi_t_model(model, config)
    if not is_phi_t:
        return False
    
    # Condition 2: ν is at grid boundary
    if nu is None:
        return False
    is_boundary = float(nu) in config.boundary_nu_values
    if not is_boundary:
        return False
    
    # Condition 3: PIT failure
    if pit_p is None:
        return False
    is_pit_failure = float(pit_p) < config.pit_threshold
    if not is_pit_failure:
        return False
    
    # Condition 4: Likelihood is flat (ν not well identified)
    is_flat = is_nu_likelihood_flat(result, config.likelihood_flatness_threshold)
    if not is_flat:
        return False
    
    return True


def get_refinement_candidates(
    nu_current: float,
    config: AdaptiveNuConfig = None
) -> List[float]:
    """
    Return intermediate ν values for local refinement.
    
    Asymmetric design for ν=20: only test downward because
    ν > ~30 is effectively Gaussian. Force the system to
    prove it needs higher ν through accumulated evidence.
    
    Args:
        nu_current: Current best ν value
        config: Configuration with refinement candidates
        
    Returns:
        List of ν values to test
    """
    config = config or DEFAULT_ADAPTIVE_NU_CONFIG
    return config.refinement_candidates.get(float(nu_current), [])


# =============================================================================
# REFINEMENT RESULT DATACLASS
# =============================================================================

@dataclass
class NuRefinementResult:
    """Result of ν refinement for a single asset."""
    
    asset: str
    refinement_attempted: bool
    nu_original: float
    nu_candidates_tested: List[float]
    nu_final: Optional[float] = None
    improvement_achieved: bool = False
    pit_before: Optional[float] = None
    pit_after: Optional[float] = None
    bic_before: Optional[float] = None
    bic_after: Optional[float] = None
    likelihood_flatness: Optional[float] = None
    refinement_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result to dictionary."""
        return {
            'asset': self.asset,
            'refinement_attempted': self.refinement_attempted,
            'nu_original': self.nu_original,
            'nu_candidates_tested': self.nu_candidates_tested,
            'nu_final': self.nu_final,
            'improvement_achieved': self.improvement_achieved,
            'pit_before': self.pit_before,
            'pit_after': self.pit_after,
            'bic_before': self.bic_before,
            'bic_after': self.bic_after,
            'likelihood_flatness': self.likelihood_flatness,
            'refinement_reason': self.refinement_reason,
            'timestamp': self.timestamp,
        }


# =============================================================================
# REFINEMENT ENGINE
# =============================================================================

class AdaptiveNuRefiner:
    """
    Engine for adaptive ν refinement of φ-t models.
    
    This class coordinates the refinement process:
    1. Identifies assets needing refinement
    2. Evaluates additional ν candidates
    3. Updates calibration results
    4. Logs all decisions for auditability
    """
    
    def __init__(self, config: AdaptiveNuConfig = None):
        """Initialize refiner with configuration."""
        self.config = config or DEFAULT_ADAPTIVE_NU_CONFIG
        self.refinement_log: List[NuRefinementResult] = []
    
    def identify_candidates(
        self,
        calibration_results: Dict[str, Dict]
    ) -> List[str]:
        """
        Identify assets that need ν refinement.
        
        Args:
            calibration_results: Dictionary mapping asset -> calibration result
            
        Returns:
            List of asset names needing refinement
        """
        candidates = []
        
        for asset, result in calibration_results.items():
            if needs_nu_refinement(result, self.config):
                candidates.append(asset)
        
        return candidates
    
    def compute_likelihood_flatness(
        self,
        result: Dict[str, Any]
    ) -> float:
        """
        Compute the likelihood flatness metric for logging.
        
        Args:
            result: Calibration result dictionary
            
        Returns:
            Log-likelihood difference between best and second-best ν
        """
        model_comparison = result.get('model_comparison', {})
        
        student_t_models = {
            k: v for k, v in model_comparison.items()
            if k.startswith('phi_student_t_nu_')
        }
        
        if len(student_t_models) < 2:
            return 0.0
        
        sorted_models = sorted(
            student_t_models.items(),
            key=lambda x: x[1].get('ll', -1e12),
            reverse=True
        )
        
        ll_best = sorted_models[0][1].get('ll', -1e12)
        ll_second = sorted_models[1][1].get('ll', -1e12)
        
        return abs(ll_best - ll_second)
    
    def refine_single_asset(
        self,
        asset: str,
        result: Dict[str, Any],
        fit_function: callable
    ) -> NuRefinementResult:
        """
        Attempt ν refinement for a single asset.
        
        Args:
            asset: Asset name
            result: Current calibration result
            fit_function: Function to fit model at specific ν
                          Signature: fit_function(asset, nu) -> Dict with 'll', 'bic', 'pit_ks_pvalue'
            
        Returns:
            NuRefinementResult with refinement outcome
        """
        nu_original = float(result.get('nu', 0))
        pit_before = result.get('pit_ks_pvalue')
        bic_before = result.get('bic')
        likelihood_flatness = self.compute_likelihood_flatness(result)
        
        # Get refinement candidates
        candidates = get_refinement_candidates(nu_original, self.config)
        
        if not candidates:
            return NuRefinementResult(
                asset=asset,
                refinement_attempted=False,
                nu_original=nu_original,
                nu_candidates_tested=[],
                nu_final=nu_original,
                improvement_achieved=False,
                pit_before=pit_before,
                pit_after=pit_before,
                bic_before=bic_before,
                bic_after=bic_before,
                likelihood_flatness=likelihood_flatness,
                refinement_reason="No candidates for this ν value"
            )
        
        # Evaluate candidates
        best_nu = nu_original
        best_bic = bic_before or float('inf')
        best_pit = pit_before or 0.0
        
        candidate_results = []
        
        for nu_candidate in candidates:
            try:
                fit_result = fit_function(asset, nu_candidate)
                
                cand_bic = fit_result.get('bic', float('inf'))
                cand_pit = fit_result.get('pit_ks_pvalue', 0.0)
                
                candidate_results.append({
                    'nu': nu_candidate,
                    'bic': cand_bic,
                    'pit': cand_pit,
                })
                
                # Select if BIC improves (lower is better)
                if cand_bic < best_bic:
                    best_nu = nu_candidate
                    best_bic = cand_bic
                    best_pit = cand_pit
                    
            except Exception as e:
                candidate_results.append({
                    'nu': nu_candidate,
                    'error': str(e),
                })
        
        # Determine if improvement was achieved
        improvement_achieved = (
            best_nu != nu_original and
            best_bic < (bic_before or float('inf'))
        )
        
        refinement_reason = (
            f"Refined from ν={nu_original} to ν={best_nu}" if improvement_achieved
            else f"No improvement found (tested {candidates})"
        )
        
        return NuRefinementResult(
            asset=asset,
            refinement_attempted=True,
            nu_original=nu_original,
            nu_candidates_tested=candidates,
            nu_final=best_nu,
            improvement_achieved=improvement_achieved,
            pit_before=pit_before,
            pit_after=best_pit,
            bic_before=bic_before,
            bic_after=best_bic,
            likelihood_flatness=likelihood_flatness,
            refinement_reason=refinement_reason
        )
    
    def run_refinement(
        self,
        calibration_results: Dict[str, Dict],
        fit_function: callable
    ) -> Dict[str, NuRefinementResult]:
        """
        Run adaptive ν refinement on all eligible assets.
        
        Args:
            calibration_results: Dictionary mapping asset -> calibration result
            fit_function: Function to fit model at specific ν
            
        Returns:
            Dictionary mapping asset -> NuRefinementResult
        """
        # Identify candidates
        candidates = self.identify_candidates(calibration_results)
        
        if not candidates:
            return {}
        
        results = {}
        
        for asset in candidates:
            result = calibration_results[asset]
            refinement_result = self.refine_single_asset(
                asset, result, fit_function
            )
            results[asset] = refinement_result
            self.refinement_log.append(refinement_result)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all refinement attempts.
        
        Returns:
            Summary dictionary with statistics
        """
        total = len(self.refinement_log)
        attempted = sum(1 for r in self.refinement_log if r.refinement_attempted)
        improved = sum(1 for r in self.refinement_log if r.improvement_achieved)
        
        pit_improvements = []
        bic_improvements = []
        
        for r in self.refinement_log:
            if r.improvement_achieved:
                if r.pit_before is not None and r.pit_after is not None:
                    pit_improvements.append(r.pit_after - r.pit_before)
                if r.bic_before is not None and r.bic_after is not None:
                    bic_improvements.append(r.bic_before - r.bic_after)
        
        return {
            'total_candidates': total,
            'refinement_attempted': attempted,
            'improvement_achieved': improved,
            'improvement_rate': improved / attempted if attempted > 0 else 0.0,
            'mean_pit_improvement': float(np.mean(pit_improvements)) if pit_improvements else None,
            'mean_bic_improvement': float(np.mean(bic_improvements)) if bic_improvements else None,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat(),
        }
    
    def save_log(self, filepath: str) -> None:
        """
        Save refinement log to JSON file.
        
        Args:
            filepath: Path to output file
        """
        log_data = {
            'summary': self.get_summary(),
            'refinements': [r.to_dict() for r in self.refinement_log],
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)


# =============================================================================
# INTEGRATION HELPER FUNCTIONS
# =============================================================================

def create_nu_fit_function(
    returns_arr: np.ndarray,
    vol_arr: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> callable:
    """
    Create a fit function for use with AdaptiveNuRefiner.
    
    This factory creates a closure that fits a φ-Student-t model
    at a specific ν value.
    
    Args:
        returns_arr: Return series
        vol_arr: Volatility series
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        
    Returns:
        Fit function with signature: fit(asset, nu) -> Dict
    """
    # Import here to avoid circular dependency
    from tune_q_mle import (
        PhiStudentTDriftModel,
        kalman_filter_drift_phi_student_t,
        compute_pit_ks_pvalue_student_t,
        compute_bic
    )
    
    def fit_at_nu(asset: str, nu: float) -> Dict[str, Any]:
        """Fit φ-Student-t model at specific ν."""
        
        # Optimize q, c, phi with FIXED nu
        q, c, phi, ll_cv, diag = PhiStudentTDriftModel.optimize_params_fixed_nu(
            returns_arr, vol_arr,
            nu=nu,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full φ-Student-t Kalman filter
        mu, P, ll_full = kalman_filter_drift_phi_student_t(
            returns_arr, vol_arr, q, c, phi, nu
        )
        
        # Compute Student-t PIT calibration
        ks_stat, pit_p = compute_pit_ks_pvalue_student_t(
            returns_arr, mu, vol_arr, P, c, nu
        )
        
        # Compute BIC (3 params: q, c, φ; ν is fixed)
        n_obs = len(returns_arr)
        bic = compute_bic(ll_full, n_params=3, n_obs=n_obs)
        
        return {
            'asset': asset,
            'nu': nu,
            'q': float(q),
            'c': float(c),
            'phi': float(phi),
            'll': float(ll_full),
            'bic': float(bic),
            'ks_statistic': float(ks_stat),
            'pit_ks_pvalue': float(pit_p),
        }
    
    return fit_at_nu


def analyze_calibration_failures(
    calibration_file: str,
    config: AdaptiveNuConfig = None
) -> Dict[str, Any]:
    """
    Analyze calibration failures to identify refinement candidates.
    
    Args:
        calibration_file: Path to calibration_failures.json
        config: Adaptive refinement configuration
        
    Returns:
        Analysis results including candidate counts and breakdown
    """
    config = config or DEFAULT_ADAPTIVE_NU_CONFIG
    
    with open(calibration_file, 'r') as f:
        data = json.load(f)
    
    issues = data.get('issues', [])
    
    # Categorize by ν value
    nu_distribution = {}
    refinement_candidates = []
    non_candidates = []
    
    for issue in issues:
        nu = issue.get('nu')
        model = issue.get('model', '')
        pit_p = issue.get('pit_ks_pvalue')
        
        if nu is not None:
            nu_key = float(nu)
            nu_distribution[nu_key] = nu_distribution.get(nu_key, 0) + 1
        
        # Check if this would be a refinement candidate
        # (Simplified check without likelihood flatness, which requires full result)
        is_phi_t = is_phi_t_model(model, config)
        is_boundary = nu is not None and float(nu) in config.boundary_nu_values
        is_pit_failure = pit_p is not None and float(pit_p) < config.pit_threshold
        
        if is_phi_t and is_boundary and is_pit_failure:
            refinement_candidates.append(issue)
        else:
            non_candidates.append(issue)
    
    return {
        'total_issues': len(issues),
        'nu_distribution': nu_distribution,
        'potential_refinement_candidates': len(refinement_candidates),
        'non_candidates': len(non_candidates),
        'candidate_breakdown': {
            'boundary_nu_12': sum(1 for c in refinement_candidates if c.get('nu') == 12.0),
            'boundary_nu_20': sum(1 for c in refinement_candidates if c.get('nu') == 20.0),
        },
        'config': config.to_dict(),
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'AdaptiveNuConfig',
    'DEFAULT_ADAPTIVE_NU_CONFIG',
    'NuRefinementResult',
    'AdaptiveNuRefiner',
    'needs_nu_refinement',
    'get_refinement_candidates',
    'is_nu_likelihood_flat',
    'is_phi_t_model',
    'create_nu_fit_function',
    'analyze_calibration_failures',
]
