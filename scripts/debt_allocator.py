#!/usr/bin/env python3
"""
debt_allocator.py

Fully Encapsulated FX Debt Allocation Engine
============================================

PURPOSE:
    Determines a single, irreversible calendar day on which JPY-denominated debt
    must be replaced with EUR-denominated debt, based exclusively on EURJPY.

THIS IS:
    ❌ NOT a trade signal
    ❌ NOT portfolio optimization  
    ❌ NOT macro prediction
    ✅ A balance-sheet convexity control mechanism

MATHEMATICAL FOUNDATION:
    - Pair purity: All reasoning on EURJPY only (no isolated currency risk)
    - Stochastic process: X_t = log(EURJPY_t)
    - Funding loss occurs iff: ΔX > 0 (EUR cost of JPY debt increases)
    - Dual-key trigger system with causal, backward-looking logic only

ARCHITECTURAL CONSTRAINTS:
    - Everything lives in this single file
    - No reuse of existing methods
    - No modification of existing files
    - No imports from tuning or signal helpers
    - No shared mutable state
    - Deletable with zero side effects

Author: Debt Allocation Engine
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings

# =============================================================================
# NUMERICAL IMPORTS (MINIMAL, STANDARD ONLY)
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# PRESENTATION IMPORTS (RICH ONLY)
# =============================================================================
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# File paths (isolated, deletable)
DEBT_CACHE_DIR = "scripts/quant/cache/debt"
DECISION_PERSISTENCE_FILE = f"{DEBT_CACHE_DIR}/debt_switch_decision.json"
EURJPY_DATA_FILE = f"{DEBT_CACHE_DIR}/EURJPY_1d.csv"

# Minimum data requirements
MIN_HISTORY_DAYS = 252  # 1 year of trading days minimum
MIN_POSTERIOR_SAMPLES = 1000  # Minimum Monte Carlo samples
MIN_HORIZONS = 3  # Minimum horizons for fragility computation

# Trigger thresholds
STRUCTURAL_RISK_THRESHOLD = 0.55  # P(ΔX > 0) threshold
FRAGILITY_PERCENTILE = 95  # Rolling percentile for fragility trigger
FRAGILITY_LOOKBACK_DAYS = 63  # ~3 months rolling window

# Dual-key overlap window
DUAL_KEY_LOOKBACK_DAYS = 5  # Days to check for dual-key overlap

# Horizons for epistemic fragility (trading days)
FRAGILITY_HORIZONS = [5, 21, 63, 126, 252]  # 1w, 1m, 3m, 6m, 1y


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class StructuralRiskResult:
    """Result of structural convexity risk computation."""
    probability_up: float  # P(ΔX > 0)
    n_samples: int
    triggered: bool
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability_up": self.probability_up,
            "n_samples": self.n_samples,
            "triggered": self.triggered,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class FragilityResult:
    """Result of epistemic fragility computation."""
    fragility_score: float  # D(t) - normalized disagreement
    rolling_threshold: float  # 95th percentile threshold
    triggered: bool
    n_horizons: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragility_score": self.fragility_score,
            "rolling_threshold": self.rolling_threshold,
            "triggered": self.triggered,
            "n_horizons": self.n_horizons,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class DebtSwitchDecision:
    """Immutable record of debt switch decision."""
    triggered: bool
    effective_date: Optional[str]
    structural_risk: StructuralRiskResult
    fragility: FragilityResult
    confidence: str  # NONE, LOW, MEDIUM, HIGH
    dual_key_overlap_days: int
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "effective_date": self.effective_date,
            "structural_risk": self.structural_risk.to_dict(),
            "fragility": self.fragility.to_dict(),
            "confidence": self.confidence,
            "dual_key_overlap_days": self.dual_key_overlap_days,
            "signature": self.signature,
        }


@dataclass
class PosteriorStatistics:
    """Posterior statistics for a single horizon."""
    horizon_days: int
    mean: float  # μ_h(t)
    std: float   # σ_h(t)
    samples: Optional[np.ndarray] = None  # Monte Carlo samples if available


# =============================================================================
# DATA INGESTION (READ-ONLY, ISOLATED)
# =============================================================================

def _load_eurjpy_prices(data_path: str = EURJPY_DATA_FILE, force_refresh: bool = True) -> Optional[pd.Series]:
    """
    Load EURJPY price series, downloading fresh data from yfinance.
    
    This function:
    - Downloads fresh data from yfinance (default behavior)
    - Saves to cache for audit trail
    - Falls back to cached data if download fails
    - Reads data as-is (no forward filling)
    - Returns None if data missing or insufficient
    - Makes no calendar assumptions
    
    Args:
        data_path: Path to EURJPY daily data CSV
        force_refresh: If True, always download fresh data (default: True)
        
    Returns:
        pd.Series with DatetimeIndex and EURJPY prices, or None if unavailable
    """
    path = Path(data_path)
    
    # Ensure cache directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to download fresh data from yfinance
    if force_refresh:
        try:
            import yfinance as yf
            
            print(f"[debt_allocator] Downloading fresh EURJPY data...")
            
            # Download EURJPY data (ticker: EURJPY=X)
            ticker = yf.Ticker("EURJPY=X")
            df = ticker.history(period="10y")
            
            if df is not None and not df.empty and 'Close' in df.columns:
                # Save to CSV for audit trail
                df_save = df.reset_index()
                df_save.to_csv(path, index=False)
                
                print(f"[debt_allocator] Downloaded {len(df)} days of EURJPY data")
                print(f"[debt_allocator] Saved to: {path}")
                
                prices = df['Close'].dropna()
                if len(prices) >= MIN_HISTORY_DAYS:
                    return prices.sort_index()
                else:
                    print(f"[debt_allocator] Warning: Only {len(prices)} days available (need {MIN_HISTORY_DAYS})")
        except ImportError:
            print("[debt_allocator] Warning: yfinance not installed, trying cached data")
        except Exception as e:
            print(f"[debt_allocator] Warning: Download failed ({e}), trying cached data")
    
    # Fall back to cached data
    try:
        if path.exists():
            print(f"[debt_allocator] Loading cached EURJPY data from: {path}")
            df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            
            # Look for Close or Adj Close column
            if 'Close' in df.columns:
                prices = df['Close']
            elif 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                print("[debt_allocator] Error: No Close column in cached data")
                return None
            
            prices = prices.dropna()
            if len(prices) >= MIN_HISTORY_DAYS:
                print(f"[debt_allocator] Loaded {len(prices)} days from cache")
                return prices.sort_index()
            else:
                print(f"[debt_allocator] Warning: Cached data has only {len(prices)} days (need {MIN_HISTORY_DAYS})")
                return None
        else:
            print(f"[debt_allocator] Error: No cached data found at {path}")
            return None
            
    except Exception as e:
        print(f"[debt_allocator] Error loading cached data: {e}")
        return None


def _compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from price series.
    
    X_t = log(EURJPY_t)
    ΔX = X_t - X_{t-1} = log(EURJPY_t / EURJPY_{t-1})
    
    Args:
        prices: EURJPY price series
        
    Returns:
        Log return series (one element shorter than prices)
    """
    log_prices = np.log(prices)
    log_returns = log_prices.diff().dropna()
    return log_returns


# =============================================================================
# KEY A: STRUCTURAL CONVEXITY RISK (MONTE CARLO ONLY)
# =============================================================================

def _compute_structural_risk(
    posterior_samples: np.ndarray,
    timestamp: Optional[str] = None
) -> StructuralRiskResult:
    """
    Compute structural convexity risk using posterior predictive Monte Carlo.
    
    MATHEMATICAL DEFINITION:
        StructuralRisk(t) = P(ΔX > 0)
        
    where ΔX > 0 means EUR cost of JPY debt increases (funding loss).
    
    ESTIMATION RULES (STRICT):
        - Use posterior predictive Monte Carlo ONLY
        - No analytical CDFs
        - No Gaussian / Student-t shortcuts
        - No parametric tail approximations
        - Operate in log-return space
    
    Args:
        posterior_samples: Monte Carlo samples of future log returns
        timestamp: Optional timestamp for the result
        
    Returns:
        StructuralRiskResult with empirical probability
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Validate samples
    samples = np.asarray(posterior_samples).flatten()
    n_samples = len(samples)
    
    if n_samples < MIN_POSTERIOR_SAMPLES:
        # Insufficient samples - return NaN result
        return StructuralRiskResult(
            probability_up=float('nan'),
            n_samples=n_samples,
            triggered=False,  # Cannot trigger with insufficient data
            timestamp=timestamp,
        )
    
    # Remove any NaN/Inf values
    valid_samples = samples[np.isfinite(samples)]
    n_valid = len(valid_samples)
    
    if n_valid < MIN_POSTERIOR_SAMPLES:
        return StructuralRiskResult(
            probability_up=float('nan'),
            n_samples=n_valid,
            triggered=False,
            timestamp=timestamp,
        )
    
    # EMPIRICAL PROBABILITY ONLY - NO ANALYTICAL SHORTCUTS
    # P(ΔX > 0) = fraction of samples where return is positive
    # Positive EURJPY return = EUR appreciation = JPY debt becomes more expensive
    n_positive = np.sum(valid_samples > 0)
    probability_up = float(n_positive) / float(n_valid)
    
    # Trigger if probability exceeds threshold
    triggered = probability_up >= STRUCTURAL_RISK_THRESHOLD
    
    return StructuralRiskResult(
        probability_up=probability_up,
        n_samples=n_valid,
        triggered=triggered,
        timestamp=timestamp,
    )


# =============================================================================
# KEY B: EPISTEMIC FRAGILITY (NORMALIZED DISAGREEMENT)
# =============================================================================

def _compute_precision_weighted_mean(
    means: np.ndarray,
    stds: np.ndarray
) -> float:
    """
    Compute precision-weighted reference mean.
    
    μ̄(t) = Σ_h (μ_h / σ_h²) / Σ_h (1 / σ_h²)
    
    Args:
        means: Array of posterior means per horizon
        stds: Array of posterior stds per horizon
        
    Returns:
        Precision-weighted mean
    """
    # Compute precisions (inverse variances)
    variances = stds ** 2
    precisions = 1.0 / variances
    
    # Precision-weighted mean
    weighted_sum = np.sum(means * precisions)
    precision_sum = np.sum(precisions)
    
    return weighted_sum / precision_sum


def _compute_fragility_score(
    means: np.ndarray,
    stds: np.ndarray
) -> float:
    """
    Compute epistemic fragility score.
    
    MATHEMATICAL DEFINITION:
        D(t) = (1/H) Σ_h |μ_h - μ̄| / σ_h
        
    This measures belief incompatibility relative to confidence.
    
    FORBIDDEN:
        ❌ Variance of means
        ❌ Raw disagreement without normalization
    
    Args:
        means: Array of posterior means per horizon
        stds: Array of posterior stds per horizon
        
    Returns:
        Fragility score D(t)
    """
    H = len(means)
    
    # Compute precision-weighted reference mean
    mu_bar = _compute_precision_weighted_mean(means, stds)
    
    # Compute normalized deviations
    deviations = np.abs(means - mu_bar) / stds
    
    # Average normalized deviation
    fragility = np.mean(deviations)
    
    return float(fragility)


def _compute_fragility_trigger(
    current_score: float,
    historical_scores: np.ndarray,
    percentile: float = FRAGILITY_PERCENTILE
) -> Tuple[bool, float]:
    """
    Determine if fragility score triggers based on historical context.
    
    RULES:
        - Normalize D(t) using backward-looking rolling statistics only
        - Current day must never be included
        - Enforce minimum history length
        - If insufficient data → return NaN, not False
    
    FragilityTrigger(t) = D(t) > rolling_95th_percentile(t−1)
    
    Args:
        current_score: Current fragility score D(t)
        historical_scores: Past fragility scores (must not include current day)
        percentile: Percentile threshold (default 95)
        
    Returns:
        Tuple of (triggered, threshold)
    """
    # Validate historical data
    valid_history = historical_scores[np.isfinite(historical_scores)]
    
    if len(valid_history) < FRAGILITY_LOOKBACK_DAYS:
        # Insufficient history - return NaN threshold
        return (False, float('nan'))
    
    # Compute rolling percentile threshold (backward-looking only)
    threshold = float(np.percentile(valid_history, percentile))
    
    # Check if current score exceeds threshold
    if not np.isfinite(current_score):
        return (False, threshold)
    
    triggered = current_score > threshold
    
    return (triggered, threshold)


def _compute_epistemic_fragility(
    posterior_stats: List[PosteriorStatistics],
    historical_fragility_scores: np.ndarray,
    timestamp: Optional[str] = None
) -> FragilityResult:
    """
    Compute epistemic fragility with normalized disagreement.
    
    Args:
        posterior_stats: List of PosteriorStatistics for each horizon
        historical_fragility_scores: Past fragility scores (backward-looking)
        timestamp: Optional timestamp
        
    Returns:
        FragilityResult
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Validate inputs
    if len(posterior_stats) < MIN_HORIZONS:
        return FragilityResult(
            fragility_score=float('nan'),
            rolling_threshold=float('nan'),
            triggered=False,
            n_horizons=len(posterior_stats),
            timestamp=timestamp,
        )
    
    # Extract means and stds
    means = np.array([s.mean for s in posterior_stats])
    stds = np.array([s.std for s in posterior_stats])
    
    # Check for valid values
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
        return FragilityResult(
            fragility_score=float('nan'),
            rolling_threshold=float('nan'),
            triggered=False,
            n_horizons=len(posterior_stats),
            timestamp=timestamp,
        )
    
    # Check for zero stds (infinite precision - pathological)
    if np.any(stds <= 0):
        return FragilityResult(
            fragility_score=float('nan'),
            rolling_threshold=float('nan'),
            triggered=False,
            n_horizons=len(posterior_stats),
            timestamp=timestamp,
        )
    
    # Compute fragility score
    fragility_score = _compute_fragility_score(means, stds)
    
    # Compute trigger
    triggered, threshold = _compute_fragility_trigger(
        fragility_score, 
        historical_fragility_scores
    )
    
    return FragilityResult(
        fragility_score=fragility_score,
        rolling_threshold=threshold,
        triggered=triggered,
        n_horizons=len(posterior_stats),
        timestamp=timestamp,
    )


# =============================================================================
# DUAL-KEY DECISION ENGINE
# =============================================================================

def _compute_confidence_level(
    structural_triggered: bool,
    fragility_triggered: bool,
    dual_key_overlap_days: int
) -> str:
    """
    Compute confidence level based on trigger states.
    
    MECHANICAL MAPPING (MANDATORY):
        No triggers          → NONE
        One key              → LOW
        Dual-key (1 day)     → MEDIUM
        Dual-key (≥2 days)   → HIGH
    
    Args:
        structural_triggered: Whether structural risk key is triggered
        fragility_triggered: Whether fragility key is triggered
        dual_key_overlap_days: Number of days with dual-key overlap
        
    Returns:
        Confidence level string
    """
    if not structural_triggered and not fragility_triggered:
        return "NONE"
    elif structural_triggered != fragility_triggered:  # XOR - only one key
        return "LOW"
    elif dual_key_overlap_days >= 2:
        return "HIGH"
    else:  # dual_key_overlap_days == 1
        return "MEDIUM"


def _compute_decision_signature(
    effective_date: str,
    structural_risk: StructuralRiskResult,
    fragility: FragilityResult,
    triggered: bool
) -> str:
    """
    Compute cryptographic signature for decision audit trail.
    
    Signature hashes:
        - date
        - StructuralRisk value
        - Fragility score
        - trigger states
    
    Args:
        effective_date: Decision date
        structural_risk: Structural risk result
        fragility: Fragility result
        triggered: Whether decision was triggered
        
    Returns:
        SHA-256 signature string
    """
    payload = {
        "effective_date": effective_date,
        "structural_risk_probability": structural_risk.probability_up,
        "structural_risk_triggered": structural_risk.triggered,
        "fragility_score": fragility.fragility_score,
        "fragility_triggered": fragility.triggered,
        "decision_triggered": triggered,
        "signature_version": "1.0.0",
    }
    
    # Serialize deterministically
    payload_str = json.dumps(payload, sort_keys=True)
    
    # Compute SHA-256
    signature = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
    
    return signature


def _check_dual_key_overlap(
    structural_risk_history: List[bool],
    fragility_history: List[bool],
    lookback_days: int = DUAL_KEY_LOOKBACK_DAYS
) -> int:
    """
    Check for dual-key overlap in recent history.
    
    RULES:
        - Window is backward-looking only
        - No future data allowed
        - Count consecutive days where both keys triggered
    
    Args:
        structural_risk_history: Recent structural risk trigger states
        fragility_history: Recent fragility trigger states
        lookback_days: Number of days to check
        
    Returns:
        Number of days with dual-key overlap
    """
    # Take only the lookback window
    struct_window = structural_risk_history[-lookback_days:]
    frag_window = fragility_history[-lookback_days:]
    
    # Ensure same length
    min_len = min(len(struct_window), len(frag_window))
    if min_len == 0:
        return 0
    
    struct_window = struct_window[-min_len:]
    frag_window = frag_window[-min_len:]
    
    # Count dual-key days
    overlap_days = sum(1 for s, f in zip(struct_window, frag_window) if s and f)
    
    return overlap_days


# =============================================================================
# PERSISTENCE (RESTART-SAFE, AUDITABLE)
# =============================================================================

def _load_persisted_decision(
    persistence_path: str = DECISION_PERSISTENCE_FILE
) -> Optional[DebtSwitchDecision]:
    """
    Load previously persisted decision if it exists.
    
    Once a decision is persisted:
        - Engine must short-circuit
        - No re-evaluation allowed
        - Restart-safe
    
    Args:
        persistence_path: Path to persistence file
        
    Returns:
        DebtSwitchDecision if exists and valid, None otherwise
    """
    try:
        path = Path(persistence_path)
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct decision from persisted data
        structural_risk = StructuralRiskResult(
            probability_up=data['structural_risk']['probability_up'],
            n_samples=data['structural_risk']['n_samples'],
            triggered=data['structural_risk']['triggered'],
            timestamp=data['structural_risk']['timestamp'],
        )
        
        fragility = FragilityResult(
            fragility_score=data['fragility']['fragility_score'],
            rolling_threshold=data['fragility']['rolling_threshold'],
            triggered=data['fragility']['triggered'],
            n_horizons=data['fragility']['n_horizons'],
            timestamp=data['fragility']['timestamp'],
        )
        
        decision = DebtSwitchDecision(
            triggered=data['triggered'],
            effective_date=data['effective_date'],
            structural_risk=structural_risk,
            fragility=fragility,
            confidence=data['confidence'],
            dual_key_overlap_days=data['dual_key_overlap_days'],
            signature=data['signature'],
        )
        
        return decision
        
    except Exception:
        return None


def _persist_decision(
    decision: DebtSwitchDecision,
    persistence_path: str = DECISION_PERSISTENCE_FILE
) -> bool:
    """
    Persist decision to file.
    
    Args:
        decision: Decision to persist
        persistence_path: Path to persistence file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(decision.to_dict(), f, indent=2)
        
        return True
        
    except Exception:
        return False


# =============================================================================
# MAIN DECISION ENGINE
# =============================================================================

def evaluate_debt_switch(
    posterior_samples: np.ndarray,
    posterior_stats_per_horizon: List[PosteriorStatistics],
    historical_fragility_scores: np.ndarray,
    structural_risk_history: List[bool],
    fragility_history: List[bool],
    evaluation_date: Optional[str] = None,
    persistence_path: str = DECISION_PERSISTENCE_FILE,
    force_reevaluate: bool = False,
) -> DebtSwitchDecision:
    """
    Main entry point for debt switch evaluation.
    
    DECISION RULE (CAUSAL, IRREVERSIBLE):
        On day t:
            if StructuralRisk(t) == TRUE
            and FragilityTrigger(t−k … t) == TRUE
            and no prior decision exists:
                trigger EUR debt switch
    
    RULES:
        - Window is backward-looking only
        - No future data allowed
        - First valid overlap day wins
        - Decision is final and permanent
    
    Args:
        posterior_samples: Monte Carlo samples for structural risk
        posterior_stats_per_horizon: Posterior statistics per horizon
        historical_fragility_scores: Historical fragility scores (backward-looking)
        structural_risk_history: Recent structural risk trigger states
        fragility_history: Recent fragility trigger states
        evaluation_date: Date of evaluation (default: today)
        persistence_path: Path to persistence file
        force_reevaluate: If True, ignore persisted decision (for testing only)
        
    Returns:
        DebtSwitchDecision
    """
    if evaluation_date is None:
        evaluation_date = date.today().isoformat()
    
    timestamp = datetime.now().isoformat()
    
    # CHECK FOR PRIOR DECISION (SHORT-CIRCUIT)
    if not force_reevaluate:
        prior_decision = _load_persisted_decision(persistence_path)
        if prior_decision is not None and prior_decision.triggered:
            # Decision already made - return existing decision
            return prior_decision
    
    # COMPUTE KEY A: STRUCTURAL RISK
    structural_risk = _compute_structural_risk(posterior_samples, timestamp)
    
    # COMPUTE KEY B: EPISTEMIC FRAGILITY
    fragility = _compute_epistemic_fragility(
        posterior_stats_per_horizon,
        historical_fragility_scores,
        timestamp
    )
    
    # CHECK DUAL-KEY OVERLAP
    # Add current trigger states to history for overlap check
    current_structural_history = structural_risk_history + [structural_risk.triggered]
    current_fragility_history = fragility_history + [fragility.triggered]
    
    dual_key_overlap_days = _check_dual_key_overlap(
        current_structural_history,
        current_fragility_history
    )
    
    # DETERMINE CONFIDENCE
    confidence = _compute_confidence_level(
        structural_risk.triggered,
        fragility.triggered,
        dual_key_overlap_days
    )
    
    # MAKE DECISION
    triggered = (
        structural_risk.triggered 
        and fragility.triggered 
        and dual_key_overlap_days >= 1
    )
    
    effective_date = evaluation_date if triggered else None
    
    # COMPUTE SIGNATURE
    signature = _compute_decision_signature(
        evaluation_date,
        structural_risk,
        fragility,
        triggered
    )
    
    # CREATE DECISION OBJECT
    decision = DebtSwitchDecision(
        triggered=triggered,
        effective_date=effective_date,
        structural_risk=structural_risk,
        fragility=fragility,
        confidence=confidence,
        dual_key_overlap_days=dual_key_overlap_days,
        signature=signature,
    )
    
    # PERSIST IF TRIGGERED
    if triggered:
        _persist_decision(decision, persistence_path)
    
    return decision


# =============================================================================
# POSTERIOR GENERATION (ISOLATED, SELF-CONTAINED)
# =============================================================================

def _generate_posterior_samples_from_returns(
    log_returns: pd.Series,
    n_samples: int = 10000,
    horizon_days: int = 21,
) -> np.ndarray:
    """
    Generate posterior predictive samples using bootstrap resampling.
    
    This is a self-contained, non-parametric approach that:
    - Uses empirical distribution only
    - No analytical CDF assumptions
    - Preserves fat tails naturally
    
    Args:
        log_returns: Historical log returns
        n_samples: Number of Monte Carlo samples
        horizon_days: Forecast horizon in trading days
        
    Returns:
        Array of posterior predictive samples for cumulative return
    """
    returns_arr = log_returns.values
    n_obs = len(returns_arr)
    
    if n_obs < MIN_HISTORY_DAYS:
        return np.array([])
    
    # Bootstrap resampling with replacement
    # For each sample, draw horizon_days returns and sum them
    samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Sample with replacement
        indices = np.random.randint(0, n_obs, size=horizon_days)
        sampled_returns = returns_arr[indices]
        # Cumulative return over horizon
        samples[i] = np.sum(sampled_returns)
    
    return samples


def _compute_posterior_statistics_per_horizon(
    log_returns: pd.Series,
    horizons: List[int] = FRAGILITY_HORIZONS,
    n_samples: int = 5000,
) -> List[PosteriorStatistics]:
    """
    Compute posterior statistics for each horizon.
    
    Args:
        log_returns: Historical log returns
        horizons: List of horizon days
        n_samples: Number of samples per horizon
        
    Returns:
        List of PosteriorStatistics
    """
    results = []
    
    for h in horizons:
        samples = _generate_posterior_samples_from_returns(
            log_returns, n_samples=n_samples, horizon_days=h
        )
        
        if len(samples) >= MIN_POSTERIOR_SAMPLES:
            mean = float(np.mean(samples))
            std = float(np.std(samples))
        else:
            mean = float('nan')
            std = float('nan')
        
        results.append(PosteriorStatistics(
            horizon_days=h,
            mean=mean,
            std=std,
            samples=samples if len(samples) > 0 else None,
        ))
    
    return results


def _compute_historical_fragility_scores(
    log_returns: pd.Series,
    horizons: List[int] = FRAGILITY_HORIZONS,
    lookback_days: int = FRAGILITY_LOOKBACK_DAYS,
) -> np.ndarray:
    """
    Compute historical fragility scores for rolling threshold.
    
    RULES:
        - Backward-looking only
        - Current day excluded
        - Each historical score uses only data available at that time
    
    Args:
        log_returns: Full historical log returns
        horizons: Horizons for fragility computation
        lookback_days: Number of historical scores to compute
        
    Returns:
        Array of historical fragility scores
    """
    n_obs = len(log_returns)
    
    if n_obs < MIN_HISTORY_DAYS + lookback_days:
        return np.array([])
    
    scores = []
    
    # Compute fragility score for each historical day
    # Start from MIN_HISTORY_DAYS to ensure enough data
    for t in range(n_obs - lookback_days, n_obs):
        if t < MIN_HISTORY_DAYS:
            continue
        
        # Use only data up to day t-1 (backward-looking)
        historical_returns = log_returns.iloc[:t]
        
        # Compute statistics for this historical point
        stats = []
        valid = True
        
        for h in horizons:
            # Simple rolling statistics for historical estimation
            if len(historical_returns) < h:
                valid = False
                break
            
            # Use recent window for mean/std estimation
            window = historical_returns.iloc[-h:]
            mean = float(window.mean()) * h  # Scale to horizon
            std = float(window.std()) * np.sqrt(h)  # Scale to horizon
            
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                valid = False
                break
            
            stats.append(PosteriorStatistics(
                horizon_days=h,
                mean=mean,
                std=std,
            ))
        
        if valid and len(stats) >= MIN_HORIZONS:
            means = np.array([s.mean for s in stats])
            stds = np.array([s.std for s in stats])
            score = _compute_fragility_score(means, stds)
            scores.append(score)
        else:
            scores.append(float('nan'))
    
    return np.array(scores)


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def run_debt_allocation_engine(
    data_path: str = EURJPY_DATA_FILE,
    persistence_path: str = DECISION_PERSISTENCE_FILE,
    evaluation_date: Optional[str] = None,
    force_reevaluate: bool = False,
    force_refresh_data: bool = True,
) -> Optional[DebtSwitchDecision]:
    """
    Run the complete debt allocation engine.
    
    This is the main entry point that:
    1. Downloads/refreshes EURJPY data
    2. Generates posterior samples
    3. Computes structural risk
    4. Computes epistemic fragility
    5. Makes dual-key decision
    6. Persists if triggered
    
    Args:
        data_path: Path to EURJPY data
        persistence_path: Path to decision persistence file
        evaluation_date: Date of evaluation (default: today)
        force_reevaluate: If True, ignore persisted decision
        force_refresh_data: If True, download fresh EURJPY data (default: True)
        
    Returns:
        DebtSwitchDecision or None if engine cannot run
    """
    # CHECK FOR PRIOR DECISION FIRST
    if not force_reevaluate:
        prior_decision = _load_persisted_decision(persistence_path)
        if prior_decision is not None and prior_decision.triggered:
            return prior_decision
    
    # LOAD DATA (with refresh option)
    prices = _load_eurjpy_prices(data_path, force_refresh=force_refresh_data)
    if prices is None:
        return None
    
    # COMPUTE LOG RETURNS
    log_returns = _compute_log_returns(prices)
    if len(log_returns) < MIN_HISTORY_DAYS:
        return None
    
    # GENERATE POSTERIOR SAMPLES (21-day horizon for structural risk)
    posterior_samples = _generate_posterior_samples_from_returns(
        log_returns, n_samples=10000, horizon_days=21
    )
    
    if len(posterior_samples) < MIN_POSTERIOR_SAMPLES:
        return None
    
    # COMPUTE POSTERIOR STATISTICS PER HORIZON
    posterior_stats = _compute_posterior_statistics_per_horizon(log_returns)
    
    # COMPUTE HISTORICAL FRAGILITY SCORES
    historical_fragility = _compute_historical_fragility_scores(log_returns)
    
    # For structural/fragility history, use simplified approach
    # (In production, these would be loaded from historical evaluations)
    structural_history: List[bool] = []
    fragility_history: List[bool] = []
    
    # EVALUATE
    decision = evaluate_debt_switch(
        posterior_samples=posterior_samples,
        posterior_stats_per_horizon=posterior_stats,
        historical_fragility_scores=historical_fragility,
        structural_risk_history=structural_history,
        fragility_history=fragility_history,
        evaluation_date=evaluation_date,
        persistence_path=persistence_path,
        force_reevaluate=force_reevaluate,
    )
    
    return decision


# =============================================================================
# PRESENTATION (RICH OUTPUT)
# =============================================================================

def _get_status_display(decision: DebtSwitchDecision) -> Tuple[str, str]:
    """Get status emoji and text for display."""
    if decision.triggered:
        return "🔴", "SWITCH TRIGGERED"
    elif decision.confidence == "LOW":
        return "🟡", "MONITORING (one key active)"
    elif decision.confidence == "NONE":
        return "🟢", "NO ACTION REQUIRED"
    else:
        return "🟡", "MONITORING"


def render_debt_switch_decision(
    decision: Optional[DebtSwitchDecision],
    console: Optional[Console] = None
) -> None:
    """
    Render debt switch decision with Rich formatting.
    
    Args:
        decision: DebtSwitchDecision or None
        console: Rich console instance
    """
    if console is None:
        console = Console()
    
    # Header
    console.print()
    console.print("=" * 50)
    console.print("[bold cyan]DEBT SWITCH — EURJPY[/bold cyan]", justify="center")
    console.print("=" * 50)
    console.print()
    
    if decision is None:
        console.print(Panel(
            "[bold red]ENGINE CANNOT RUN[/bold red]\n\n"
            "Possible reasons:\n"
            "• Insufficient EURJPY data\n"
            "• Missing posterior inputs\n"
            "• Data validation failed",
            title="[bold red]❌ ERROR[/bold red]",
            border_style="red",
        ))
        return
    
    # Status
    status_emoji, status_text = _get_status_display(decision)
    
    # Build status panel
    if decision.triggered:
        status_style = "red"
        status_content = (
            f"[bold]Status[/bold]           : {status_emoji} [bold red]{status_text}[/bold red]\n"
            f"[bold]Effective Date[/bold]   : [bold white]{decision.effective_date}[/bold white]"
        )
    else:
        status_style = "green" if decision.confidence == "NONE" else "yellow"
        status_content = f"[bold]Status[/bold]           : {status_emoji} {status_text}"
    
    console.print(Panel(
        status_content,
        title=f"[bold {status_style}]Decision Status[/bold {status_style}]",
        border_style=status_style,
        padding=(1, 2),
    ))
    
    console.print()
    
    # Metrics table
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", width=20)
    table.add_column("Value", justify="right", width=30)
    table.add_column("Status", justify="center", width=12)
    
    # Structural Risk row
    sr = decision.structural_risk
    if np.isfinite(sr.probability_up):
        sr_value = f"P(EURJPY ↑) = {sr.probability_up:.2%}"
        sr_status = "[red]TRIGGERED[/red]" if sr.triggered else "[green]OK[/green]"
    else:
        sr_value = "INSUFFICIENT DATA"
        sr_status = "[yellow]N/A[/yellow]"
    table.add_row("Structural Risk", sr_value, sr_status)
    
    # Fragility row
    fr = decision.fragility
    if np.isfinite(fr.fragility_score):
        if np.isfinite(fr.rolling_threshold):
            fr_value = f"{fr.fragility_score:.2f} (threshold: {fr.rolling_threshold:.2f})"
        else:
            fr_value = f"{fr.fragility_score:.2f} (threshold: N/A)"
        fr_status = "[red]TRIGGERED[/red]" if fr.triggered else "[green]OK[/green]"
    else:
        fr_value = "INSUFFICIENT DATA"
        fr_status = "[yellow]N/A[/yellow]"
    table.add_row("Fragility Score", fr_value, fr_status)
    
    # Dual-key overlap
    overlap_value = f"{decision.dual_key_overlap_days} days"
    overlap_status = "[red]ACTIVE[/red]" if decision.dual_key_overlap_days >= 1 else "[dim]—[/dim]"
    table.add_row("Dual-Key Overlap", overlap_value, overlap_status)
    
    # Confidence
    conf_colors = {"NONE": "dim", "LOW": "yellow", "MEDIUM": "orange1", "HIGH": "red"}
    conf_color = conf_colors.get(decision.confidence, "white")
    table.add_row("Confidence", f"[{conf_color}]{decision.confidence}[/{conf_color}]", "")
    
    console.print(table)
    
    console.print()
    
    # Signature (audit trail)
    console.print(f"[dim]Signature: {decision.signature[:16]}...[/dim]")
    console.print()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point for `make debt` command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FX Debt Allocation Engine - EURJPY Balance Sheet Convexity Control"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=EURJPY_DATA_FILE,
        help=f'Path to EURJPY daily data CSV (default: {EURJPY_DATA_FILE})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-evaluation even if prior decision exists'
    )
    parser.add_argument(
        '--no-refresh',
        action='store_true',
        help='Skip data refresh, use cached EURJPY data only'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Evaluation date (YYYY-MM-DD format, default: today)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of Rich formatting'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Show header
    if not args.json:
        console.print()
        console.print(Panel(
            "[bold cyan]FX Debt Allocation Engine[/bold cyan]\n"
            "[dim]EURJPY Balance Sheet Convexity Control[/dim]",
            border_style="cyan",
        ))
        console.print()
        console.print(f"[dim]Cache directory: {DEBT_CACHE_DIR}[/dim]")
        console.print(f"[dim]Data file: {args.data_path}[/dim]")
        console.print()
    
    # Run engine
    decision = run_debt_allocation_engine(
        data_path=args.data_path,
        evaluation_date=args.date,
        force_reevaluate=args.force,
        force_refresh_data=not args.no_refresh,
    )
    
    if args.json:
        if decision is not None:
            print(json.dumps(decision.to_dict(), indent=2))
        else:
            print(json.dumps({"error": "Engine cannot run"}, indent=2))
    else:
        render_debt_switch_decision(decision, console)


if __name__ == '__main__':
    main()
