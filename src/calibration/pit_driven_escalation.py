#!/usr/bin/env python3
"""
===============================================================================
PIT-DRIVEN DISTRIBUTION ESCALATION (PDDE)
===============================================================================

This module implements hierarchical model escalation based on PIT diagnostics.
It orchestrates the full calibration pipeline with automatic fallback when
simpler models fail to achieve calibration.

DESIGN PHILOSOPHY:
    Escalate model complexity only when diagnostics demand it.
    Treat PIT failure as information — not error.
    
    Do NOT expand the global model grid blindly.
    Refine locally, conditionally, and reversibly.

ESCALATION CHAIN:
    Level 0: φ-Gaussian
        ↓ (PIT fail)
    Level 1: φ-Student-t (coarse ν grid: 4, 6, 8, 12, 20)
        ↓ (PIT fail at boundary ν)
    Level 2: Adaptive ν Refinement (local grid expansion)
        ↓ (ν-refinement fails)
    Level 3: K=2 Scale Mixture (σ dispersion for regime heterogeneity)
        ↓ (mixture fails, extreme kurtosis)
    Level 4: EVT Tail Splice (GPD beyond threshold, rare)

ESCALATION TRIGGERS:
    - PIT KS p-value < α (default 0.05)
    - Each level requires ALL lower levels to have failed
    - Level 4 only for top 5-10% kurtosis assets

OUTPUT CONTRACT:
    Each asset records:
    {
        "final_model": "gaussian | phi-gaussian | phi-t | phi-t-refined | mixture | evt",
        "escalation_level": 0-4,
        "pit_ks_pvalue": float,
        "escalation_path": ["level0", "level1", ...],
        "justification": "diagnostic-driven"
    }

ARCHITECTURAL CONSTRAINTS:
    - Preserves tune → signal separation
    - Drop-in compatible with existing pipeline
    - All escalation decisions are auditable
    - Parameter growth is strictly controlled

===============================================================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import IntEnum

import numpy as np


# =============================================================================
# ESCALATION LEVELS
# =============================================================================

class EscalationLevel(IntEnum):
    """Enumeration of model complexity levels."""
    GAUSSIAN = 0
    PHI_STUDENT_T = 1
    NU_REFINED = 2
    MIXTURE_K2 = 3
    EVT_SPLICE = 4


LEVEL_NAMES = {
    EscalationLevel.GAUSSIAN: "φ-Gaussian",
    EscalationLevel.PHI_STUDENT_T: "φ-Student-t",
    EscalationLevel.NU_REFINED: "φ-Student-t (ν-refined)",
    EscalationLevel.MIXTURE_K2: "K=2 Scale Mixture",
    EscalationLevel.EVT_SPLICE: "EVT Tail Splice",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PDDEConfig:
    """Configuration for PIT-Driven Distribution Escalation."""
    
    # Enable/disable PDDE
    enabled: bool = True
    
    # PIT threshold for escalation trigger
    pit_alpha: float = 0.05
    
    # Severe miscalibration threshold (for logging)
    pit_severe_alpha: float = 0.01
    
    # Maximum escalation level allowed
    max_escalation_level: EscalationLevel = EscalationLevel.MIXTURE_K2
    
    # Kurtosis threshold for EVT eligibility (top 5-10%)
    evt_kurtosis_threshold: float = 10.0
    
    # Minimum observations for mixture model
    min_obs_for_mixture: int = 100
    
    # Minimum observations for EVT
    min_obs_for_evt: int = 500
    
    # BIC penalty for model selection (prefer simpler models)
    bic_complexity_penalty: float = 2.0
    
    # Enable verbose logging
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary."""
        return {
            'enabled': self.enabled,
            'pit_alpha': self.pit_alpha,
            'pit_severe_alpha': self.pit_severe_alpha,
            'max_escalation_level': int(self.max_escalation_level),
            'evt_kurtosis_threshold': self.evt_kurtosis_threshold,
            'min_obs_for_mixture': self.min_obs_for_mixture,
            'min_obs_for_evt': self.min_obs_for_evt,
            'bic_complexity_penalty': self.bic_complexity_penalty,
            'verbose': self.verbose,
        }


# Default configuration
DEFAULT_PDDE_CONFIG = PDDEConfig()


# =============================================================================
# ESCALATION RESULT
# =============================================================================

@dataclass
class EscalationResult:
    """Result of PIT-driven escalation for a single asset."""
    
    # Final model selection
    final_model: str
    escalation_level: EscalationLevel
    
    # Calibration metrics
    pit_ks_pvalue: float
    ks_statistic: float
    kurtosis: float
    
    # Escalation history
    escalation_path: List[str]
    levels_attempted: List[int]
    levels_failed: List[int]
    
    # Model parameters (subset)
    q: Optional[float] = None
    c: Optional[float] = None
    phi: Optional[float] = None
    nu: Optional[float] = None
    
    # Mixture-specific (if applicable)
    mixture_selected: bool = False
    mixture_sigma_ratio: Optional[float] = None
    mixture_weight: Optional[float] = None
    
    # ν-refinement specific (if applicable)
    nu_refinement_attempted: bool = False
    nu_refinement_improved: bool = False
    nu_original: Optional[float] = None
    nu_final: Optional[float] = None
    
    # Momentum-specific (February 2026)
    momentum_enabled: bool = False
    momentum_weight: Optional[float] = None
    
    # EVT-specific (if applicable)
    evt_fitted: bool = False
    evt_significant: bool = False
    evt_xi: Optional[float] = None
    evt_severity: str = ""
    
    # Justification
    justification: str = "diagnostic-driven"
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def is_calibrated(self) -> bool:
        """Check if final model passes calibration."""
        return self.pit_ks_pvalue >= 0.05
    
    @property
    def calibration_status(self) -> str:
        """Human-readable calibration status."""
        if self.pit_ks_pvalue >= 0.05:
            return "calibrated"
        elif self.pit_ks_pvalue >= 0.01:
            return "warning"
        else:
            return "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result to dictionary."""
        return {
            'final_model': self.final_model,
            'escalation_level': int(self.escalation_level),
            'escalation_level_name': LEVEL_NAMES.get(self.escalation_level, "unknown"),
            'pit_ks_pvalue': float(self.pit_ks_pvalue),
            'ks_statistic': float(self.ks_statistic),
            'kurtosis': float(self.kurtosis),
            'escalation_path': self.escalation_path,
            'levels_attempted': self.levels_attempted,
            'levels_failed': self.levels_failed,
            'is_calibrated': self.is_calibrated,
            'calibration_status': self.calibration_status,
            'q': float(self.q) if self.q is not None else None,
            'c': float(self.c) if self.c is not None else None,
            'phi': float(self.phi) if self.phi is not None else None,
            'nu': float(self.nu) if self.nu is not None else None,
            'mixture_selected': self.mixture_selected,
            'mixture_sigma_ratio': float(self.mixture_sigma_ratio) if self.mixture_sigma_ratio else None,
            'mixture_weight': float(self.mixture_weight) if self.mixture_weight else None,
            'nu_refinement_attempted': self.nu_refinement_attempted,
            'nu_refinement_improved': self.nu_refinement_improved,
            'nu_original': float(self.nu_original) if self.nu_original else None,
            'nu_final': float(self.nu_final) if self.nu_final else None,
            # Momentum fields (February 2026)
            'momentum_enabled': self.momentum_enabled,
            'momentum_weight': float(self.momentum_weight) if self.momentum_weight else None,
            # EVT fields
            'evt_fitted': self.evt_fitted,
            'evt_significant': self.evt_significant,
            'evt_xi': float(self.evt_xi) if self.evt_xi is not None else None,
            'evt_severity': self.evt_severity,
            'justification': self.justification,
            'timestamp': self.timestamp,
        }


# =============================================================================
# ESCALATION ORCHESTRATOR
# =============================================================================

class PDDEOrchestrator:
    """
    Orchestrates PIT-Driven Distribution Escalation.
    
    This class coordinates the escalation chain, ensuring that:
    1. Each level is attempted in order
    2. Escalation only occurs when diagnostics demand it
    3. All decisions are auditable
    4. Simpler models are preferred when calibrated
    
    Usage:
        orchestrator = PDDEOrchestrator(config)
        result = orchestrator.escalate(
            returns=returns,
            vol=vol,
            baseline_result=baseline_fit_result
        )
    """
    
    def __init__(self, config: Optional[PDDEConfig] = None):
        """Initialize orchestrator with configuration."""
        self.config = config or DEFAULT_PDDE_CONFIG
        self._log_buffer: List[str] = []
    
    def _log(self, message: str) -> None:
        """Log message if verbose mode enabled."""
        self._log_buffer.append(message)
        if self.config.verbose:
            print(f"  [PDDE] {message}")
    
    def should_escalate(
        self,
        pit_ks_pvalue: float,
        current_level: EscalationLevel,
        kurtosis: Optional[float] = None,
        n_obs: int = 0
    ) -> Tuple[bool, str]:
        """
        Determine if escalation should occur.
        
        Args:
            pit_ks_pvalue: Current PIT KS p-value
            current_level: Current escalation level
            kurtosis: Excess kurtosis (for EVT decision)
            n_obs: Number of observations
            
        Returns:
            (should_escalate, reason)
        """
        # Check if already at max level
        if current_level >= self.config.max_escalation_level:
            return False, "at_max_level"
        
        # Check calibration status
        if pit_ks_pvalue >= self.config.pit_alpha:
            return False, "calibrated"
        
        # Level-specific checks
        next_level = EscalationLevel(current_level + 1)
        
        if next_level == EscalationLevel.MIXTURE_K2:
            if n_obs < self.config.min_obs_for_mixture:
                return False, f"insufficient_obs_for_mixture ({n_obs} < {self.config.min_obs_for_mixture})"
        
        if next_level == EscalationLevel.EVT_SPLICE:
            if n_obs < self.config.min_obs_for_evt:
                return False, f"insufficient_obs_for_evt ({n_obs} < {self.config.min_obs_for_evt})"
            if kurtosis is not None and kurtosis < self.config.evt_kurtosis_threshold:
                return False, f"kurtosis_below_evt_threshold ({kurtosis:.2f} < {self.config.evt_kurtosis_threshold})"
        
        return True, "pit_failure"
    
    def build_escalation_result(
        self,
        baseline_result: Dict[str, Any],
        escalation_path: List[str],
        levels_attempted: List[int],
        levels_failed: List[int],
        final_level: EscalationLevel,
        final_pit_pvalue: float,
        final_ks_stat: float,
    ) -> EscalationResult:
        """
        Build EscalationResult from baseline result and escalation history.
        
        Args:
            baseline_result: Original calibration result dictionary
            escalation_path: List of escalation steps taken
            levels_attempted: List of levels attempted
            levels_failed: List of levels that failed
            final_level: Final escalation level reached
            final_pit_pvalue: Final PIT p-value
            final_ks_stat: Final KS statistic
            
        Returns:
            EscalationResult
        """
        # Extract common fields
        global_result = baseline_result.get('global', baseline_result)
        
        # Determine final model name
        if global_result.get('mixture_selected'):
            final_model = "mixture"
        elif global_result.get('nu_refinement', {}).get('improvement_achieved'):
            final_model = "phi-t-refined"
        elif global_result.get('noise_model', '').startswith('phi_student_t'):
            final_model = "phi-t"
        elif global_result.get('noise_model') == 'kalman_phi_gaussian':
            final_model = "phi-gaussian"
        else:
            final_model = "gaussian"
        
        # Extract mixture info
        mixture_model = global_result.get('mixture_model', {})
        
        # Extract ν refinement info
        nu_refinement = global_result.get('nu_refinement', {})
        
        return EscalationResult(
            final_model=final_model,
            escalation_level=final_level,
            pit_ks_pvalue=final_pit_pvalue,
            ks_statistic=final_ks_stat,
            kurtosis=global_result.get('std_residual_kurtosis', global_result.get('excess_kurtosis', 0.0)),
            escalation_path=escalation_path,
            levels_attempted=levels_attempted,
            levels_failed=levels_failed,
            q=global_result.get('q'),
            c=global_result.get('c'),
            phi=global_result.get('phi'),
            nu=global_result.get('nu'),
            mixture_selected=global_result.get('mixture_selected', False),
            mixture_sigma_ratio=mixture_model.get('sigma_ratio'),
            mixture_weight=mixture_model.get('weight'),
            nu_refinement_attempted=nu_refinement.get('refinement_attempted', False),
            nu_refinement_improved=nu_refinement.get('improvement_achieved', False),
            nu_original=nu_refinement.get('nu_original'),
            nu_final=nu_refinement.get('nu_final'),
            justification="diagnostic-driven",
        )
    
    def summarize_escalation(
        self,
        results: Dict[str, EscalationResult]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for escalation results.
        
        Args:
            results: Dict mapping asset names to EscalationResult
            
        Returns:
            Summary dictionary
        """
        total = len(results)
        if total == 0:
            return {'total': 0}
        
        # Count by level
        level_counts = {level: 0 for level in EscalationLevel}
        for result in results.values():
            level_counts[result.escalation_level] += 1
        
        # Count calibration status
        calibrated = sum(1 for r in results.values() if r.is_calibrated)
        warnings = sum(1 for r in results.values() if r.calibration_status == 'warning')
        critical = sum(1 for r in results.values() if r.calibration_status == 'critical')
        
        # Count by model type
        model_counts: Dict[str, int] = {}
        for result in results.values():
            model_counts[result.final_model] = model_counts.get(result.final_model, 0) + 1
        
        # Escalation statistics
        escalations_triggered = sum(1 for r in results.values() if r.escalation_level > EscalationLevel.GAUSSIAN)
        mixture_attempts = sum(1 for r in results.values() if r.mixture_selected or r.escalation_level >= EscalationLevel.MIXTURE_K2)
        mixture_successes = sum(1 for r in results.values() if r.mixture_selected and r.is_calibrated)
        nu_refinement_attempts = sum(1 for r in results.values() if r.nu_refinement_attempted)
        nu_refinement_successes = sum(1 for r in results.values() if r.nu_refinement_improved)
        
        # Momentum statistics (February 2026)
        momentum_enabled_count = sum(1 for r in results.values() if r.momentum_enabled)
        momentum_calibrated = sum(1 for r in results.values() if r.momentum_enabled and r.is_calibrated)
        
        # EVT statistics
        evt_fitted_count = sum(1 for r in results.values() if r.evt_fitted)
        evt_significant_count = sum(1 for r in results.values() if r.evt_significant)
        
        return {
            'total': total,
            'calibrated': calibrated,
            'calibrated_pct': calibrated / total * 100,
            'warnings': warnings,
            'critical': critical,
            'level_counts': {LEVEL_NAMES[k]: v for k, v in level_counts.items()},
            'model_counts': model_counts,
            'escalations_triggered': escalations_triggered,
            'escalation_rate': escalations_triggered / total * 100,
            'mixture_attempts': mixture_attempts,
            'mixture_successes': mixture_successes,
            'mixture_success_rate': mixture_successes / mixture_attempts * 100 if mixture_attempts > 0 else 0,
            'nu_refinement_attempts': nu_refinement_attempts,
            'nu_refinement_successes': nu_refinement_successes,
            'nu_refinement_success_rate': nu_refinement_successes / nu_refinement_attempts * 100 if nu_refinement_attempts > 0 else 0,
            # Momentum statistics (February 2026)
            'momentum_enabled': momentum_enabled_count,
            'momentum_enabled_pct': momentum_enabled_count / total * 100 if total > 0 else 0,
            'momentum_calibrated': momentum_calibrated,
            'momentum_calibration_rate': momentum_calibrated / momentum_enabled_count * 100 if momentum_enabled_count > 0 else 0,
            # EVT statistics
            'evt_fitted': evt_fitted_count,
            'evt_significant': evt_significant_count,
            'evt_significant_pct': evt_significant_count / total * 100 if total > 0 else 0,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_escalation_from_result(result: Dict[str, Any]) -> EscalationResult:
    """
    Extract EscalationResult from a tune_asset_q result dictionary.
    
    This function is used to convert existing results to the PDDE format.
    
    Args:
        result: Result dictionary from tune_asset_q
        
    Returns:
        EscalationResult
    """
    global_result = result.get('global', result)
    
    # Determine escalation level
    if global_result.get('mixture_selected'):
        level = EscalationLevel.MIXTURE_K2
    elif global_result.get('nu_refinement', {}).get('improvement_achieved'):
        level = EscalationLevel.NU_REFINED
    elif global_result.get('noise_model', '').startswith('phi_student_t'):
        level = EscalationLevel.PHI_STUDENT_T
    else:
        level = EscalationLevel.GAUSSIAN
    
    # Build escalation path
    path = []
    levels_attempted = []
    levels_failed = []
    
    # Level 0 always attempted
    levels_attempted.append(0)
    path.append("baseline_fit")
    
    # Check if Student-t was selected
    if global_result.get('noise_model', '').startswith('phi_student_t'):
        levels_attempted.append(1)
        path.append("student_t_selected")
    
    # Check ν refinement
    nu_ref = global_result.get('nu_refinement', {})
    if nu_ref.get('refinement_attempted'):
        levels_attempted.append(2)
        path.append("nu_refinement_attempted")
        if not nu_ref.get('improvement_achieved'):
            levels_failed.append(2)
            path.append("nu_refinement_failed")
        else:
            path.append("nu_refinement_improved")
    
    # Check mixture (DEPRECATED - K=2 mixture removed after 206 attempts, 0 selections)
    # Kept for backward compatibility with cached results
    if global_result.get('mixture_attempted'):
        levels_attempted.append(3)
        path.append("mixture_attempted_legacy")
        if not global_result.get('mixture_selected'):
            levels_failed.append(3)
            path.append("mixture_rejected_legacy")
        else:
            path.append("mixture_selected_legacy")
    
    # Determine final model name (including momentum variants - February 2026)
    noise_model = global_result.get('noise_model', '')
    is_momentum = global_result.get('momentum_enabled', False) or 'momentum' in noise_model.lower() or '+mom' in noise_model.lower()
    momentum_suffix = '+Momentum' if is_momentum else ''
    
    # K=2 mixture check kept for backward compat with cached results
    if global_result.get('mixture_selected'):
        final_model = "mixture_legacy"
    elif nu_ref.get('improvement_achieved'):
        final_model = f"phi-t-refined{momentum_suffix}"
    elif noise_model.startswith('phi_student_t') or 'student_t' in noise_model.lower():
        final_model = f"phi-t{momentum_suffix}"
    elif noise_model == 'kalman_phi_gaussian' or 'phi_gaussian' in noise_model.lower():
        final_model = f"phi-gaussian{momentum_suffix}"
    else:
        if is_momentum:
            final_model = f"gaussian{momentum_suffix}"
        else:
            final_model = "gaussian"
    
    mixture_model = global_result.get('mixture_model', {})
    
    # Extract momentum data (February 2026)
    momentum_data = global_result.get('momentum', {})
    momentum_weight = momentum_data.get('weight') or global_result.get('momentum_weight')
    
    # Extract EVT data
    evt_data = global_result.get('evt') or {}
    evt_diag = global_result.get('evt_diagnostics') or {}
    evt_fitted = evt_diag.get('fit_success', False)
    evt_xi = evt_data.get('xi')
    evt_severity = evt_data.get('severity', '')
    evt_significant = evt_fitted and evt_xi is not None and evt_severity.upper() in ['HIGH', 'MEDIUM']
    
    return EscalationResult(
        final_model=final_model,
        escalation_level=level,
        pit_ks_pvalue=global_result.get('pit_ks_pvalue', 0.0),
        ks_statistic=global_result.get('ks_statistic', 1.0),
        kurtosis=global_result.get('std_residual_kurtosis', global_result.get('excess_kurtosis', 0.0)),
        escalation_path=path,
        levels_attempted=levels_attempted,
        levels_failed=levels_failed,
        q=global_result.get('q'),
        c=global_result.get('c'),
        phi=global_result.get('phi'),
        nu=global_result.get('nu'),
        mixture_selected=global_result.get('mixture_selected', False),
        mixture_sigma_ratio=mixture_model.get('sigma_ratio'),
        mixture_weight=mixture_model.get('weight'),
        nu_refinement_attempted=nu_ref.get('refinement_attempted', False),
        nu_refinement_improved=nu_ref.get('improvement_achieved', False),
        nu_original=nu_ref.get('nu_original'),
        nu_final=nu_ref.get('nu_final'),
        # Momentum fields (February 2026)
        momentum_enabled=is_momentum,
        momentum_weight=momentum_weight,
        # EVT fields
        evt_fitted=evt_fitted,
        evt_significant=evt_significant,
        evt_xi=evt_xi,
        evt_severity=evt_severity,
    )


def get_escalation_summary_from_cache(cache: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Generate PDDE summary from existing cache.
    
    Args:
        cache: Cache dictionary mapping asset names to results
        
    Returns:
        Summary dictionary with escalation statistics
    """
    results = {}
    skipped = 0
    
    for asset, result in cache.items():
        try:
            results[asset] = extract_escalation_from_result(result)
        except Exception as e:
            # Log but continue - don't silently skip assets
            skipped += 1
            continue
    
    # If too many assets were skipped, fall back to direct counting
    if skipped > len(cache) * 0.1:  # More than 10% skipped
        return _compute_summary_directly_from_cache(cache)
    
    orchestrator = PDDEOrchestrator()
    summary = orchestrator.summarize_escalation(results)
    summary['skipped_assets'] = skipped
    return summary


def _compute_summary_directly_from_cache(cache: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compute escalation summary directly from cache without EscalationResult conversion.
    
    This is a fallback when extract_escalation_from_result fails for too many assets.
    """
    total = len(cache)
    if total == 0:
        return {'total': 0}
    
    # Initialize counters
    calibrated = 0
    warnings = 0
    critical = 0
    
    # Model type counters (base models)
    gaussian_count = 0
    phi_gaussian_count = 0
    student_t_count = 0
    student_t_refined_count = 0
    mixture_count = 0
    gh_count = 0
    tvvm_count = 0
    
    # Momentum model counters (February 2026)
    momentum_gaussian_count = 0
    momentum_phi_gaussian_count = 0
    momentum_student_t_count = 0
    
    # Escalation counters
    mixture_attempts = 0
    mixture_successes = 0
    nu_refinement_attempts = 0
    nu_refinement_successes = 0
    gh_attempts = 0
    gh_successes = 0
    tvvm_attempts = 0
    tvvm_successes = 0
    evt_attempts = 0
    evt_successes = 0
    
    for asset, data in cache.items():
        global_data = data.get('global', data)
        
        # Get PIT p-value and calibration status
        pit_p = global_data.get('pit_ks_pvalue', 0)
        calibration_warning = global_data.get('calibration_warning', False)
        
        if pit_p >= 0.05 and not calibration_warning:
            calibrated += 1
        elif pit_p >= 0.01:
            warnings += 1
        else:
            critical += 1
        
        # Count model types (including momentum variants)
        noise_model = global_data.get('noise_model', '')
        nu_ref = global_data.get('nu_refinement') or {}
        is_momentum = global_data.get('momentum_enabled', False) or 'momentum' in noise_model.lower() or '+mom' in noise_model.lower()
        
        if global_data.get('tvvm_selected'):
            tvvm_count += 1
        elif global_data.get('gh_selected'):
            gh_count += 1
        elif global_data.get('mixture_selected'):
            mixture_count += 1
        elif nu_ref.get('improvement_achieved'):
            student_t_refined_count += 1
        elif noise_model.startswith('phi_student_t') or 'student_t' in noise_model.lower():
            if is_momentum:
                momentum_student_t_count += 1
            student_t_count += 1
        elif noise_model == 'kalman_phi_gaussian' or 'phi_gaussian' in noise_model.lower() or global_data.get('phi') is not None:
            if is_momentum:
                momentum_phi_gaussian_count += 1
                momentum_gaussian_count += 1  # Also count in momentum_gaussian for backwards compat
            phi_gaussian_count += 1
        else:
            if is_momentum:
                momentum_gaussian_count += 1
            gaussian_count += 1
        
        # Count escalation attempts
        if global_data.get('mixture_attempted'):
            mixture_attempts += 1
            if global_data.get('mixture_selected'):
                mixture_successes += 1
        
        if nu_ref.get('refinement_attempted'):
            nu_refinement_attempts += 1
            if nu_ref.get('improvement_achieved'):
                nu_refinement_successes += 1
        
        if global_data.get('gh_attempted'):
            gh_attempts += 1
            if global_data.get('gh_selected'):
                gh_successes += 1
        
        if global_data.get('tvvm_attempted'):
            tvvm_attempts += 1
            if global_data.get('tvvm_selected'):
                tvvm_successes += 1
        
        # Count EVT tail splice attempts (EVT is always attempted when data is available)
        evt_diag = global_data.get('evt_diagnostics') or {}
        if evt_diag.get('fit_success'):
            evt_attempts += 1
            # EVT is considered successful if it provides valid tail parameters with significant severity
            evt_data = global_data.get('evt') or {}
            if evt_data.get('xi') is not None and evt_data.get('severity', '').upper() in ['HIGH', 'MEDIUM']:
                evt_successes += 1
    
    escalations = student_t_count + student_t_refined_count + mixture_count + gh_count + tvvm_count
    
    return {
        'total': total,
        'calibrated': calibrated,
        'calibrated_pct': calibrated / total * 100 if total > 0 else 0,
        'warnings': warnings,
        'critical': critical,
        'level_counts': {
            'Gaussian': gaussian_count - momentum_gaussian_count,
            'Gaussian+Momentum': momentum_gaussian_count - momentum_phi_gaussian_count,
            'φ-Gaussian': phi_gaussian_count - momentum_phi_gaussian_count,
            'φ-Gaussian+Momentum': momentum_phi_gaussian_count,
            'φ-Student-t': student_t_count - momentum_student_t_count,
            'φ-Student-t+Momentum': momentum_student_t_count,
            'φ-Student-t (ν-refined)': student_t_refined_count,
            'K=2 Scale Mixture': mixture_count,
            'Generalized Hyperbolic': gh_count,
            'Time-Varying Vol Multiplier': tvvm_count,
            'EVT Tail Splice': 0,
        },
        'model_counts': {
            'gaussian': gaussian_count - momentum_gaussian_count,
            'gaussian_momentum': momentum_gaussian_count - momentum_phi_gaussian_count,
            'phi-gaussian': phi_gaussian_count - momentum_phi_gaussian_count,
            'phi-gaussian_momentum': momentum_phi_gaussian_count,
            'phi-t': student_t_count - momentum_student_t_count,
            'phi-t_momentum': momentum_student_t_count,
            'phi-t-refined': student_t_refined_count,
            'mixture': mixture_count,
            'gh': gh_count,
            'tvvm': tvvm_count,
        },
        'momentum_counts': {
            'total_momentum': momentum_gaussian_count + momentum_student_t_count,
            'gaussian_momentum': momentum_gaussian_count,
            'phi_gaussian_momentum': momentum_phi_gaussian_count,
            'student_t_momentum': momentum_student_t_count,
        },
        'escalations_triggered': escalations,
        'escalation_rate': escalations / total * 100 if total > 0 else 0,
        'mixture_attempts': mixture_attempts,
        'mixture_successes': mixture_successes,
        'mixture_success_rate': mixture_successes / mixture_attempts * 100 if mixture_attempts > 0 else 0,
        'nu_refinement_attempts': nu_refinement_attempts,
        'nu_refinement_successes': nu_refinement_successes,
        'nu_refinement_success_rate': nu_refinement_successes / nu_refinement_attempts * 100 if nu_refinement_attempts > 0 else 0,
        'gh_attempts': gh_attempts,
        'gh_successes': gh_successes,
        'gh_success_rate': gh_successes / gh_attempts * 100 if gh_attempts > 0 else 0,
        'tvvm_attempts': tvvm_attempts,
        'tvvm_successes': tvvm_successes,
        'tvvm_success_rate': tvvm_successes / tvvm_attempts * 100 if tvvm_attempts > 0 else 0,
        'evt_attempts': evt_attempts,
        'evt_successes': evt_successes,
        'evt_success_rate': evt_successes / evt_attempts * 100 if evt_attempts > 0 else 0,
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'EscalationLevel',
    'LEVEL_NAMES',
    'PDDEConfig',
    'DEFAULT_PDDE_CONFIG',
    'EscalationResult',
    'PDDEOrchestrator',
    'extract_escalation_from_result',
    'get_escalation_summary_from_cache',
]
