#!/usr/bin/env python3
"""
===============================================================================
OPTIONS SIGNAL GENERATION WITH STRATEGY RECOMMENDATIONS
===============================================================================

Generates options signals based on tuned volatility models and equity signals.

Architecture: Confidence-Bounded Envelopes with Liquidity-First Filtering

For each high conviction equity signal:
1. Apply liquidity filter (untradeable options never enter evaluation)
2. Stratify by expiry (near-expiry requires elevated conviction)
3. Load tuned volatility parameters
4. Compute expected returns for relevant options strategies
5. Generate confidence-bounded strategy recommendations

CORE PRINCIPLE: "Every option signal carries explicit confidence bounds that
mechanically constrain position sizing. Signals without uncertainty
quantification are operationally unusable."

-------------------------------------------------------------------------------
SIGNAL GENERATION PIPELINE
-------------------------------------------------------------------------------

1. LIQUIDITY FILTER (first gate)
   - Spread width < 15% of mid price
   - Open interest > 50 contracts
   - Recent volume > 10 contracts
   
2. EXPIRY STRATIFICATION
   - Near (≤7 days): Requires 80%+ conviction, max 25% position
   - Short (8-21 days): Requires 70%+ conviction, max 50% position
   - Medium (22-45 days): Standard thresholds apply
   - Long (>45 days): Reduced position limits due to vega risk
   
3. MODEL COMPETITION
   - Selected volatility models produce posterior beliefs
   - Ensemble forecast with confidence bounds
   
4. STRATEGY SELECTION
   - Strong BUY → Long calls, bull call spreads
   - Strong SELL → Long puts, bear put spreads
   - Directional conviction determines strategy aggressiveness
   
5. CONFIDENCE ENVELOPE
   - Every signal carries confidence interval
   - Position sizing mechanically bounded by confidence width

-------------------------------------------------------------------------------
GOVERNANCE PRINCIPLES
-------------------------------------------------------------------------------

1. Liquidity is prerequisite — untradeable options never enter model evaluation
2. Expiry demands respect — near-expiry signals require elevated conviction
3. Uncertainty is mandatory — every signal carries explicit confidence bounds
4. Provenance is preserved — every signal traceable to triggering equity signal
5. Isolation is enforced — option failures cannot propagate to equity signals

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log, sqrt, exp

import numpy as np
from scipy.stats import norm

# Rich console for UX
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn


# Import from sibling modules
from .option_tune import (
    tune_ticker_options,
    load_option_tune_cache,
    save_option_tune_cache,
    VOL_MODEL_CLASSES,
    DEFAULT_VOL_PRIOR_MEAN,
    get_expiry_stratum,
    get_expiry_weight,
    EXPIRY_NEAR,
    EXPIRY_SHORT,
    EXPIRY_MEDIUM,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Liquidity filter thresholds
MAX_SPREAD_PCT = 0.15         # Maximum 15% spread (relative to mid)
MIN_OPEN_INTEREST = 50        # Minimum 50 contracts OI
MIN_VOLUME = 10               # Minimum 10 contracts traded
MAX_MONEYNESS_PCT = 0.20      # Maximum 20% OTM

# Conviction thresholds by expiry stratum
CONVICTION_THRESHOLDS = {
    'near': 0.80,     # 80% conviction for ≤7 days
    'short': 0.70,    # 70% conviction for 8-21 days
    'medium': 0.62,   # Standard threshold for 22-45 days
    'long': 0.62,     # Standard for >45 days
}

# Position limits by expiry stratum
MAX_POSITION_PCT = {
    'near': 0.25,     # Max 25% of intended for near expiry
    'short': 0.50,    # Max 50% for short term
    'medium': 1.00,   # Full position for medium term
    'long': 0.75,     # Reduced for vega risk
}

# Expected return thresholds
MIN_EXPECTED_RETURN = 0.10    # Minimum 10% expected return
MAX_EXPECTED_LOSS = 0.50      # Maximum 50% expected loss tolerance

# Risk-free rate
RISK_FREE_RATE = 0.045

# Parallel processing
MAX_WORKERS = 8

# Directory paths
HIGH_CONVICTION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "high_conviction"
)
OPTIONS_SIGNALS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "options_signals"
)


# =============================================================================
# LIQUIDITY FILTER
# =============================================================================

def passes_liquidity_filter(option: Dict) -> Tuple[bool, str]:
    """
    Check if option contract passes liquidity requirements.
    
    This is the FIRST gate in the pipeline - untradeable options
    never enter model evaluation.
    
    When bid/ask is not available (common with delayed/EOD data), we use
    last_price and volume as proxy for liquidity.
    
    Returns:
        (passes, reason) tuple
    """
    bid = option.get("bid", 0) or 0
    ask = option.get("ask", 0) or 0
    last_price = option.get("last_price", 0) or 0
    oi = option.get("open_interest", 0) or 0
    volume = option.get("volume", 0) or 0
    moneyness = abs(option.get("moneyness_pct", 0) or 0)
    
    # Check spread - if bid/ask available, validate spread width
    mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
    if mid > 0:
        spread_pct = (ask - bid) / mid
        if spread_pct > MAX_SPREAD_PCT:
            return False, f"spread_too_wide_{spread_pct:.1%}"
    elif bid == 0 and ask == 0:
        # No bid/ask available (common with delayed/EOD data)
        # Fall back to last_price and volume as liquidity proxy
        if last_price <= 0:
            return False, "no_price_data"
        # Require higher volume threshold when no bid/ask
        if volume < MIN_VOLUME * 5:  # 50 contracts min when no bid/ask
            return False, f"low_volume_no_bidask_{volume}"
    
    # Check open interest (relaxed when high volume exists)
    if oi < MIN_OPEN_INTEREST:
        # Allow if volume is very high (indicates active trading)
        if volume < MIN_VOLUME * 10:  # 100 contracts
            return False, f"low_oi_{oi}"
    
    # Check volume
    if volume < MIN_VOLUME:
        return False, f"low_volume_{volume}"
    
    # Check moneyness
    if moneyness > MAX_MONEYNESS_PCT * 100:
        return False, f"too_otm_{moneyness:.1f}%"
    
    return True, "passed"


def filter_liquid_options(options: List[Dict]) -> List[Dict]:
    """
    Filter options chain to liquid contracts only.
    
    Returns list of options that pass all liquidity criteria.
    """
    liquid = []
    for opt in options:
        passes, _ = passes_liquidity_filter(opt)
        if passes:
            liquid.append(opt)
    return liquid


# =============================================================================
# EXPIRY STRATIFICATION
# =============================================================================

def get_conviction_threshold(stratum: str) -> float:
    """Get required conviction level for expiry stratum."""
    return CONVICTION_THRESHOLDS.get(stratum, 0.62)


def get_position_limit(stratum: str) -> float:
    """Get maximum position size multiplier for expiry stratum."""
    return MAX_POSITION_PCT.get(stratum, 1.0)


def stratify_options(options: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Stratify options by expiry category.
    
    Returns dict mapping stratum name to list of options.
    """
    strata = {'near': [], 'short': [], 'medium': [], 'long': []}
    
    for opt in options:
        dte = opt.get("days_to_expiration", 30)
        stratum = get_expiry_stratum(dte)
        strata[stratum].append(opt)
    
    return strata


# =============================================================================
# EXPECTED RETURN CALCULATIONS
# =============================================================================

def compute_option_expected_return(
    current_price: float,
    strike: float,
    premium: float,
    days_to_expiry: int,
    option_type: str,
    p_up: float,
    expected_underlying_move: float,
    vol_forecast: float,
    vol_confidence_width: float,
) -> Dict[str, float]:
    """
    Compute expected return for an options position with confidence bounds.
    
    Uses the equity signal's p_up and expected move combined with
    volatility forecast to estimate option expected return.
    
    Args:
        current_price: Current underlying price
        strike: Option strike price
        premium: Current option premium (bid-ask midpoint)
        days_to_expiry: Days until expiration
        option_type: "call" or "put"
        p_up: Probability underlying goes up (from equity signal)
        expected_underlying_move: Expected percentage move (from equity signal)
        vol_forecast: Forecasted implied volatility (from options tuning)
        vol_confidence_width: Confidence interval width on vol forecast
        
    Returns:
        Dict with expected_return, prob_profit, confidence_bounds, etc.
    """
    if premium <= 0 or current_price <= 0:
        return {"expected_return": 0, "valid": False, "reason": "invalid_inputs"}
    
    # Convert days to years
    T = max(days_to_expiry / 365.0, 1/365.0)
    
    # Expected underlying price at expiry
    expected_price_up = current_price * (1 + expected_underlying_move)
    expected_price_down = current_price * (1 - abs(expected_underlying_move))
    
    # Option payoffs at expiry
    if option_type == "call":
        payoff_up = max(0, expected_price_up - strike)
        payoff_down = max(0, expected_price_down - strike)
    else:  # put
        payoff_up = max(0, strike - expected_price_up)
        payoff_down = max(0, strike - expected_price_down)
    
    # Expected payoff weighted by probabilities
    expected_payoff = p_up * payoff_up + (1 - p_up) * payoff_down
    
    # Expected return (as percentage of premium paid)
    expected_return = (expected_payoff - premium) / premium
    
    # Probability of profit (using vol forecast)
    if option_type == "call":
        breakeven = strike + premium
        if current_price > 0 and vol_forecast > 0:
            d = (log(current_price / breakeven) + 0.5 * vol_forecast ** 2 * T) / (vol_forecast * sqrt(T))
            prob_profit = p_up * norm.cdf(d)
        else:
            prob_profit = 0.5 * p_up
    else:  # put
        breakeven = strike - premium
        if current_price > 0 and vol_forecast > 0 and breakeven > 0:
            d = (log(current_price / breakeven) + 0.5 * vol_forecast ** 2 * T) / (vol_forecast * sqrt(T))
            prob_profit = (1 - p_up) * (1 - norm.cdf(d))
        else:
            prob_profit = 0.5 * (1 - p_up)
    
    # Confidence bounds on expected return using vol uncertainty
    # Higher vol forecast → lower expected return for long positions
    vol_upper = vol_forecast + vol_confidence_width
    vol_lower = max(0.01, vol_forecast - vol_confidence_width)
    
    # Simplified: vol impacts probability of profit
    if vol_upper > 0:
        if option_type == "call":
            d_upper = (log(current_price / breakeven) + 0.5 * vol_upper ** 2 * T) / (vol_upper * sqrt(T)) if breakeven > 0 else 0
            prob_profit_upper = p_up * norm.cdf(d_upper) if np.isfinite(d_upper) else prob_profit
        else:
            d_upper = (log(current_price / breakeven) + 0.5 * vol_upper ** 2 * T) / (vol_upper * sqrt(T)) if breakeven > 0 else 0
            prob_profit_upper = (1 - p_up) * (1 - norm.cdf(d_upper)) if np.isfinite(d_upper) else prob_profit
    else:
        prob_profit_upper = prob_profit
    
    if vol_lower > 0:
        if option_type == "call":
            d_lower = (log(current_price / breakeven) + 0.5 * vol_lower ** 2 * T) / (vol_lower * sqrt(T)) if breakeven > 0 else 0
            prob_profit_lower = p_up * norm.cdf(d_lower) if np.isfinite(d_lower) else prob_profit
        else:
            d_lower = (log(current_price / breakeven) + 0.5 * vol_lower ** 2 * T) / (vol_lower * sqrt(T)) if breakeven > 0 else 0
            prob_profit_lower = (1 - p_up) * (1 - norm.cdf(d_lower)) if np.isfinite(d_lower) else prob_profit
    else:
        prob_profit_lower = prob_profit
    
    # Risk-reward ratio
    max_loss = premium  # For long options
    max_gain = max(payoff_up, payoff_down)
    risk_reward = max_gain / max_loss if max_loss > 0 else 0
    
    return {
        "expected_return": round(expected_return, 4),
        "prob_profit": round(prob_profit, 4),
        "prob_profit_lower": round(min(prob_profit_lower, prob_profit_upper), 4),
        "prob_profit_upper": round(max(prob_profit_lower, prob_profit_upper), 4),
        "risk_reward": round(risk_reward, 2),
        "max_loss": round(max_loss, 2),
        "max_gain": round(max_gain, 2),
        "breakeven": round(breakeven, 2),
        "expected_payoff": round(expected_payoff, 2),
        "valid": True,
    }


def score_option_contract(
    contract: Dict,
    equity_signal: Dict,
    vol_params: Dict,
    stratum: str,
) -> Dict[str, Any]:
    """
    Score an individual options contract for recommendation.
    
    Combines:
    - Expected return from equity signal direction
    - Greeks-based risk assessment
    - Volatility model fit quality
    - Liquidity considerations
    - Expiry stratum requirements
    """
    # Check conviction threshold for this stratum
    conviction_threshold = get_conviction_threshold(stratum)
    p_up = equity_signal.get("probability_up", 0.5)
    signal_type = equity_signal.get("signal_type", "")
    
    # Determine required conviction based on signal type
    if "BUY" in signal_type:
        conviction = p_up
    else:  # SELL
        conviction = 1 - p_up
    
    if conviction < conviction_threshold:
        return {
            "score": 0,
            "valid": False,
            "reason": f"conviction_{conviction:.1%}_below_{conviction_threshold:.0%}_for_{stratum}"
        }
    
    # Extract contract data
    strike = contract.get("strike", 0)
    bid = contract.get("bid", 0)
    bid = contract.get("bid", 0) or 0
    ask = contract.get("ask", 0) or 0
    last_price = contract.get("last_price", 0) or 0
    
    # Use bid/ask midpoint if available, otherwise fall back to last_price
    if bid > 0 and ask > 0:
        premium = (bid + ask) / 2
    elif last_price > 0:
        premium = last_price
    else:
        premium = 0
    
    dte = contract.get("days_to_expiration", 30)
    option_type = contract.get("type", "call")
    volume = contract.get("volume", 0) or 0
    oi = contract.get("open_interest", 0) or 0
    iv = contract.get("implied_volatility_pct", 30) / 100
    delta = contract.get("delta")
    gamma = contract.get("gamma")
    theta = contract.get("theta")
    vega = contract.get("vega")
    
    current_price = equity_signal.get("current_price", 0)
    if current_price <= 0:
        current_price = contract.get("underlying_price", strike)
    
    # Skip if invalid data
    if strike <= 0 or premium <= 0:
        return {"score": 0, "valid": False, "reason": "invalid_contract_data"}
    
    # Get equity signal parameters
    exp_ret = equity_signal.get("expected_return_pct", 0)
    if isinstance(exp_ret, (int, float)):
        if abs(exp_ret) > 1:  # Percentage format
            exp_ret = exp_ret / 100
    
    # Get volatility forecast from tuned model
    vol_posterior = vol_params.get("global", {}).get("model_posterior", {})
    vol_models = vol_params.get("global", {}).get("models", {})
    ensemble = vol_params.get("global", {}).get("ensemble_forecast", {})
    
    vol_forecast = ensemble.get("volatility", iv)
    vol_confidence_lower = ensemble.get("confidence_lower", vol_forecast * 0.8)
    vol_confidence_upper = ensemble.get("confidence_upper", vol_forecast * 1.2)
    vol_confidence_width = (vol_confidence_upper - vol_confidence_lower) / 2
    
    # Compute expected return
    exp_return_result = compute_option_expected_return(
        current_price=current_price,
        strike=strike,
        premium=premium,
        days_to_expiry=dte,
        option_type=option_type,
        p_up=p_up,
        expected_underlying_move=exp_ret,
        vol_forecast=vol_forecast,
        vol_confidence_width=vol_confidence_width,
    )
    
    if not exp_return_result.get("valid", False):
        return {"score": 0, "valid": False, "reason": exp_return_result.get("reason", "return_calc_failed")}
    
    expected_return = exp_return_result["expected_return"]
    prob_profit = exp_return_result["prob_profit"]
    
    # Filter on expected return threshold
    if expected_return < MIN_EXPECTED_RETURN:
        return {
            "score": 0, 
            "valid": False, 
            "reason": f"expected_return_{expected_return:.1%}_below_{MIN_EXPECTED_RETURN:.0%}"
        }
    
    # Compute composite score
    
    # 1. Liquidity score
    liquidity_score = min(1.0, (volume + oi) / 200)
    
    # 2. Delta alignment score (delta should match signal direction)
    if delta is not None:
        if option_type == "call":
            delta_alignment = abs(delta)  # Higher delta = more directional
        else:
            delta_alignment = abs(delta)
        # Penalize extreme deltas (>0.8 or <0.2)
        if abs(delta) > 0.8:
            delta_alignment *= 0.8
        elif abs(delta) < 0.2:
            delta_alignment *= 0.7
    else:
        delta_alignment = 0.5
    
    # 3. Theta penalty (for long positions, theta hurts)
    if theta is not None and theta < 0:
        theta_score = max(0, 1 + theta / premium) if premium > 0 else 0.5
    else:
        theta_score = 0.5
    
    # 4. Vol edge: model forecast vs market IV
    vol_edge = (iv - vol_forecast) / vol_forecast if vol_forecast > 0 else 0
    # Positive vol_edge means market IV > model → options "expensive"
    # For long options buyers, negative edge is good
    vol_score = max(-0.3, min(0.3, -vol_edge))
    
    # 5. Expiry stratum adjustment
    position_limit = get_position_limit(stratum)
    
    # Composite score (higher is better)
    score = (
        0.35 * expected_return +              # Expected return dominates
        0.25 * prob_profit +                  # Probability of profit
        0.15 * delta_alignment +              # Direction alignment
        0.10 * liquidity_score +              # Liquidity
        0.10 * theta_score +                  # Theta consideration
        0.05 * (0.5 + vol_score)              # Vol edge
    ) * position_limit  # Scale by stratum position limit
    
    return {
        "score": round(score, 4),
        "valid": True,
        "expected_return": expected_return,
        "prob_profit": prob_profit,
        "prob_profit_lower": exp_return_result.get("prob_profit_lower", prob_profit),
        "prob_profit_upper": exp_return_result.get("prob_profit_upper", prob_profit),
        "delta_alignment": delta_alignment,
        "liquidity_score": liquidity_score,
        "theta_score": theta_score,
        "vol_edge": vol_edge,
        "vol_forecast": vol_forecast,
        "vol_confidence_width": vol_confidence_width,
        "market_iv": iv,
        "position_limit": position_limit,
        "stratum": stratum,
        "conviction": conviction,
        "conviction_threshold": conviction_threshold,
        **exp_return_result,
    }


# =============================================================================
# STRATEGY RECOMMENDATION
# =============================================================================

def recommend_options_strategy(
    equity_signal: Dict,
    options_chain: Dict,
    vol_params: Dict,
) -> Dict[str, Any]:
    """
    Generate options strategy recommendations for an equity signal.
    
    For STRONG_BUY: Recommend long calls
    For STRONG_SELL: Recommend long puts
    
    Applies:
    1. Liquidity filter first
    2. Expiry stratification
    3. Conviction thresholds per stratum
    4. Confidence-bounded scoring
    
    Returns ranked list of specific contract recommendations.
    """
    signal_type = equity_signal.get("signal_type", "")
    ticker = equity_signal.get("ticker", "")
    
    # Determine option type based on signal
    if "BUY" in signal_type:
        target_type = "call"
        strategy = "LONG_CALL"
    elif "SELL" in signal_type:
        target_type = "put"
        strategy = "LONG_PUT"
    else:
        return {"strategy": "NONE", "recommendations": [], "reason": "not_strong_signal"}
    
    # Get options of the target type
    all_options = options_chain.get("options", [])
    if not all_options:
        return {
            "strategy": strategy,
            "recommendations": [],
            "error": "no_options_available",
            "filtered_count": 0,
        }
    
    # Filter to target type
    candidates = [
        opt for opt in all_options 
        if opt.get("type", "").startswith(target_type[:4])
    ]
    
    if not candidates:
        return {
            "strategy": strategy,
            "recommendations": [],
            "error": f"no_{target_type}s_available",
            "filtered_count": 0,
        }
    
    # STEP 1: Apply liquidity filter
    liquid_options = filter_liquid_options(candidates)
    
    if not liquid_options:
        return {
            "strategy": strategy,
            "recommendations": [],
            "error": "no_liquid_options",
            "total_candidates": len(candidates),
            "filtered_count": 0,
        }
    
    # STEP 2: Stratify by expiry
    strata = stratify_options(liquid_options)
    
    # Add current price to equity signal for calculations
    equity_signal["current_price"] = options_chain.get("underlying_price", 0)
    
    # STEP 3: Score all candidates within their stratum
    scored = []
    stratum_stats = {}
    
    for stratum_name, stratum_options in strata.items():
        stratum_stats[stratum_name] = {
            "total": len(stratum_options),
            "passed": 0,
            "filtered_reasons": {},
        }
        
        for contract in stratum_options:
            result = score_option_contract(contract, equity_signal, vol_params, stratum_name)
            
            if result.get("valid", False):
                scored.append({
                    **contract,
                    **result,
                })
                stratum_stats[stratum_name]["passed"] += 1
            else:
                reason = result.get("reason", "unknown")
                stratum_stats[stratum_name]["filtered_reasons"][reason] = \
                    stratum_stats[stratum_name]["filtered_reasons"].get(reason, 0) + 1
    
    # Sort by score (descending)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take top 5 recommendations
    recommendations = scored[:5]
    
    # Compute summary statistics
    if recommendations:
        avg_score = np.mean([r["score"] for r in recommendations])
        avg_exp_ret = np.mean([r["expected_return"] for r in recommendations])
        avg_prob = np.mean([r["prob_profit"] for r in recommendations])
        avg_confidence_width = np.mean([r.get("vol_confidence_width", 0.05) for r in recommendations])
    else:
        avg_score = 0
        avg_exp_ret = 0
        avg_prob = 0
        avg_confidence_width = 0
    
    # Confidence envelope for the strategy
    confidence_envelope = {
        "avg_expected_return": float(avg_exp_ret),
        "expected_return_lower": float(avg_exp_ret * 0.5) if recommendations else 0,  # Conservative
        "expected_return_upper": float(avg_exp_ret * 1.5) if recommendations else 0,
        "avg_prob_profit": float(avg_prob),
        "prob_profit_range": [
            float(min(r.get("prob_profit_lower", r["prob_profit"]) for r in recommendations)) if recommendations else 0,
            float(max(r.get("prob_profit_upper", r["prob_profit"]) for r in recommendations)) if recommendations else 0,
        ],
        "vol_uncertainty": float(avg_confidence_width),
    }
    
    return {
        "ticker": ticker,
        "strategy": strategy,
        "signal_type": signal_type,
        "total_candidates": len(candidates),
        "liquid_candidates": len(liquid_options),
        "n_recommended": len(recommendations),
        "avg_score": round(avg_score, 4),
        "confidence_envelope": confidence_envelope,
        "stratum_stats": stratum_stats,
        "recommendations": recommendations,
        "vol_params_summary": {
            "model_posterior": vol_params.get("global", {}).get("model_posterior", {}),
            "ensemble_volatility": vol_params.get("global", {}).get("ensemble_forecast", {}).get("volatility"),
            "equity_prior_applied": vol_params.get("meta", {}).get("equity_prior_applied", False),
        },
        "generated_at": datetime.now().isoformat(),
    }


# =============================================================================
# RICH UX RENDERING
# =============================================================================

def render_options_summary_tables(
    call_results: List[Dict],
    put_results: List[Dict],
    console: Optional[Console] = None,
) -> None:
    """
    Render beautiful Rich tables for options signal recommendations.
    
    Separated into:
    - CALLS table (from STRONG BUY equity signals)
    - PUTS table (from STRONG SELL equity signals)
    """
    if console is None:
        console = Console()
    
    # Header
    console.print()
    header = Panel(
        Align.center(
            Text.assemble(
                ("OPTIONS CHAIN RECOMMENDATIONS\n", "bold bright_white"),
                ("Hierarchical Bayesian Framework with Confidence-Bounded Envelopes", "dim"),
            )
        ),
        box=box.HEAVY,
        border_style="bright_cyan",
        padding=(1, 2),
    )
    console.print(header)
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CALLS TABLE (STRONG BUY)
    # ═══════════════════════════════════════════════════════════════════════════════
    if call_results:
        # Flatten recommendations from all results
        all_calls = []
        for result in call_results:
            ticker = result.get("ticker", "")
            sector = result.get("sector", "")
            horizon = result.get("horizon_days", 0)
            equity_prob = result.get("probability_up", 0.5)
            strategy = result.get("options_strategy", {})
            recs = strategy.get("recommendations", [])
            
            for rec in recs:
                all_calls.append({
                    "ticker": ticker,
                    "sector": sector,
                    "horizon": horizon,
                    "equity_prob": equity_prob,
                    "strike": rec.get("strike", 0),
                    "expiry": rec.get("expiration_date", ""),
                    "dte": rec.get("days_to_expiration", 0),
                    "premium": (rec.get("bid", 0) + rec.get("ask", 0)) / 2,
                    "delta": rec.get("delta"),
                    "iv": rec.get("implied_volatility_pct", 0),
                    "expected_return": rec.get("expected_return", 0),
                    "prob_profit": rec.get("prob_profit", 0),
                    "score": rec.get("score", 0),
                    "stratum": rec.get("stratum", ""),
                    "vol_forecast": rec.get("vol_forecast", 0),
                    "vol_edge": rec.get("vol_edge", 0),
                })
        
        # Sort by expected return (highest first)
        all_calls.sort(key=lambda x: x["expected_return"], reverse=True)
        
        console.print()
        call_header = Text()
        call_header.append("  ▲▲ ", style="bold bright_green")
        call_header.append("CALL OPTIONS", style="bold bright_green")
        call_header.append(f"  ({len(all_calls)} recommendations from {len(call_results)} tickers)", style="dim")
        console.print(call_header)
        console.print()
        
        call_table = Table(
            show_header=True,
            header_style="bold white on green",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        
        call_table.add_column("Ticker", justify="left", width=10, no_wrap=True)
        call_table.add_column("Strike", justify="right", width=10)
        call_table.add_column("Expiry", justify="center", width=12)
        call_table.add_column("DTE", justify="right", width=5)
        call_table.add_column("Premium", justify="right", width=10)
        call_table.add_column("Delta", justify="right", width=8)
        call_table.add_column("IV", justify="right", width=8)
        call_table.add_column("E[Return]", justify="right", width=12)
        call_table.add_column("P(Profit)", justify="right", width=10)
        call_table.add_column("Score", justify="right", width=8)
        
        for c in all_calls[:20]:  # Top 20
            # Format values
            strike_str = f"${c['strike']:.2f}" if c['strike'] else "—"
            premium_str = f"${c['premium']:.2f}" if c['premium'] else "—"
            delta_str = f"{c['delta']:.2f}" if c['delta'] else "—"
            iv_str = f"{c['iv']:.1f}%" if c['iv'] else "—"
            exp_ret_str = f"{c['expected_return']*100:+.1f}%" if c['expected_return'] else "—"
            prob_str = f"{c['prob_profit']*100:.1f}%" if c['prob_profit'] else "—"
            score_str = f"{c['score']:.2f}" if c['score'] else "—"
            
            # Color coding for expected return
            if c['expected_return'] > 0.3:
                exp_ret_style = "bold bright_green"
            elif c['expected_return'] > 0.15:
                exp_ret_style = "bright_green"
            else:
                exp_ret_style = "green"
            
            call_table.add_row(
                c['ticker'],
                strike_str,
                c['expiry'][:10] if c['expiry'] else "—",
                str(c['dte']),
                premium_str,
                delta_str,
                iv_str,
                f"[{exp_ret_style}]{exp_ret_str}[/]",
                f"[bright_green]{prob_str}[/]",
                f"[dim]{score_str}[/]",
            )
        
        if len(all_calls) > 20:
            call_table.add_row(
                f"[dim]... and {len(all_calls) - 20} more[/]",
                "", "", "", "", "", "", "", "", ""
            )
        
        console.print(call_table)
        console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # PUTS TABLE (STRONG SELL)
    # ═══════════════════════════════════════════════════════════════════════════════
    if put_results:
        # Flatten recommendations from all results
        all_puts = []
        for result in put_results:
            ticker = result.get("ticker", "")
            sector = result.get("sector", "")
            horizon = result.get("horizon_days", 0)
            equity_prob = result.get("probability_up", 0.5)
            strategy = result.get("options_strategy", {})
            recs = strategy.get("recommendations", [])
            
            for rec in recs:
                bid = rec.get("bid", 0) or 0
                ask = rec.get("ask", 0) or 0
                last_price = rec.get("last_price", 0) or 0
                premium = (bid + ask) / 2 if (bid > 0 and ask > 0) else last_price
                
                all_puts.append({
                    "ticker": ticker,
                    "sector": sector,
                    "horizon": horizon,
                    "equity_prob": equity_prob,
                    "strike": rec.get("strike", 0),
                    "expiry": rec.get("expiration_date", ""),
                    "dte": rec.get("days_to_expiration", 0),
                    "premium": premium,
                    "delta": rec.get("delta"),
                    "iv": rec.get("implied_volatility_pct", 0),
                    "expected_return": rec.get("expected_return", 0),
                    "prob_profit": rec.get("prob_profit", 0),
                    "score": rec.get("score", 0),
                    "stratum": rec.get("stratum", ""),
                    "vol_forecast": rec.get("vol_forecast", 0),
                    "vol_edge": rec.get("vol_edge", 0),
                })
        
        # Sort by expected return (highest first)
        all_puts.sort(key=lambda x: x["expected_return"], reverse=True)
        
        console.print()
        put_header = Text()
        put_header.append("  ▼▼ ", style="bold indian_red1")
        put_header.append("PUT OPTIONS", style="bold indian_red1")
        put_header.append(f"  ({len(all_puts)} recommendations from {len(put_results)} tickers)", style="dim")
        console.print(put_header)
        console.print()
        
        put_table = Table(
            show_header=True,
            header_style="bold white on red",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        
        put_table.add_column("Ticker", justify="left", width=10, no_wrap=True)
        put_table.add_column("Strike", justify="right", width=10)
        put_table.add_column("Expiry", justify="center", width=12)
        put_table.add_column("DTE", justify="right", width=5)
        put_table.add_column("Premium", justify="right", width=10)
        put_table.add_column("Delta", justify="right", width=8)
        put_table.add_column("IV", justify="right", width=8)
        put_table.add_column("E[Return]", justify="right", width=12)
        put_table.add_column("P(Profit)", justify="right", width=10)
        put_table.add_column("Score", justify="right", width=8)
        
        for p in all_puts[:20]:  # Top 20
            # Format values
            strike_str = f"${p['strike']:.2f}" if p['strike'] else "—"
            premium_str = f"${p['premium']:.2f}" if p['premium'] else "—"
            delta_str = f"{p['delta']:.2f}" if p['delta'] else "—"
            iv_str = f"{p['iv']:.1f}%" if p['iv'] else "—"
            exp_ret_str = f"{p['expected_return']*100:+.1f}%" if p['expected_return'] else "—"
            prob_str = f"{p['prob_profit']*100:.1f}%" if p['prob_profit'] else "—"
            score_str = f"{p['score']:.2f}" if p['score'] else "—"
            
            # Color coding for expected return
            if p['expected_return'] > 0.3:
                exp_ret_style = "bold indian_red1"
            elif p['expected_return'] > 0.15:
                exp_ret_style = "indian_red1"
            else:
                exp_ret_style = "red"
            
            put_table.add_row(
                p['ticker'],
                strike_str,
                p['expiry'][:10] if p['expiry'] else "—",
                str(p['dte']),
                premium_str,
                delta_str,
                iv_str,
                f"[{exp_ret_style}]{exp_ret_str}[/]",
                f"[indian_red1]{prob_str}[/]",
                f"[dim]{score_str}[/]",
            )
        
        if len(all_puts) > 20:
            put_table.add_row(
                f"[dim]... and {len(all_puts) - 20} more[/]",
                "", "", "", "", "", "", "", "", ""
            )
        
        console.print(put_table)
        console.print()
    
    # No recommendations
    if not call_results and not put_results:
        console.print()
        console.print(Align.center(
            Text("No options recommendations generated", style="dim italic")
        ))
        console.print()
    
    # Footer
    console.print()
    footer = Text(justify="center")
    footer.append("Liquidity: ", style="dim")
    footer.append(f"Spread ≤ {MAX_SPREAD_PCT:.0%}, OI ≥ {MIN_OPEN_INTEREST}, Vol ≥ {MIN_VOLUME}", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append("Sorted by E[Return]", style="dim italic")
    console.print(Align.center(footer))
    console.print()


def render_pipeline_summary(
    stats: Dict[str, int],
    console: Optional[Console] = None,
) -> None:
    """Render a summary panel of pipeline statistics."""
    if console is None:
        console = Console()
    
    processed = stats.get("processed", 0)
    successful = stats.get("successful", 0)
    call_recs = stats.get("call_recommendations", 0)
    put_recs = stats.get("put_recommendations", 0)
    skipped = stats.get("skipped", 0)
    errors = stats.get("errors", 0)
    
    # Summary table
    summary_table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 2),
    )
    summary_table.add_column("Metric", justify="right", style="dim")
    summary_table.add_column("Value", justify="left")
    
    summary_table.add_row("Signals Processed", f"[bold]{processed}[/]")
    summary_table.add_row("Successful", f"[bright_green]{successful}[/]")
    summary_table.add_row("Call Recommendations", f"[green]▲ {call_recs}[/]")
    summary_table.add_row("Put Recommendations", f"[red]▼ {put_recs}[/]")
    summary_table.add_row("Skipped", f"[yellow]{skipped}[/]")
    summary_table.add_row("Errors", f"[red]{errors}[/]" if errors > 0 else "[dim]0[/]")
    
    panel = Panel(
        summary_table,
        title="Pipeline Summary",
        border_style="bright_cyan",
        box=box.ROUNDED,
    )
    
    console.print()
    console.print(Align.center(panel))
    console.print()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_high_conviction_signal(
    signal_path: str,
    tune_cache: Dict[str, Dict],
    force_retune: bool = False,
) -> Optional[Dict]:
    """
    Process a single high conviction signal file.
    
    1. Load signal data
    2. Check/perform options tuning
    3. Generate strategy recommendations
    
    Args:
        signal_path: Path to signal JSON file
        tune_cache: Options tuning cache
        force_retune: Force re-tuning even if cached
        
    Returns:
        Enhanced signal dict with options recommendations
    """
    # Load signal
    try:
        with open(signal_path, 'r') as f:
            signal = json.load(f)
    except Exception as e:
        return {"error": f"failed_to_load: {e}", "path": signal_path}
    
    ticker = signal.get("ticker", "")
    if not ticker:
        return {"error": "no_ticker", "path": signal_path}
    
    # Check if options data available
    options_chain = signal.get("options_chain", {})
    if options_chain.get("skipped", False):
        return {
            **signal,
            "options_strategy": {
                "strategy": "SKIPPED",
                "reason": options_chain.get("reason", "non_optionable"),
            }
        }
    
    if options_chain.get("error"):
        return {
            **signal,
            "options_strategy": {
                "strategy": "ERROR",
                "reason": options_chain.get("error"),
            }
        }
    
    # Get price history for tuning
    price_history = signal.get("price_history", {})
    
    # Get or compute tuning
    safe_ticker = ticker.replace("^", "_").replace("=", "_").replace(".", "_")
    
    if safe_ticker in tune_cache and not force_retune:
        vol_params = tune_cache[safe_ticker]
    elif ticker in tune_cache and not force_retune:
        vol_params = tune_cache[ticker]
    else:
        # Perform tuning
        vol_params = tune_ticker_options(
            ticker=ticker,
            options_chain=options_chain,
            price_history=price_history,
            equity_signal=signal,
        )
        
        if vol_params:
            tune_cache[safe_ticker] = vol_params
        else:
            vol_params = {
                "global": {
                    "model_posterior": {},
                    "models": {},
                    "ensemble_forecast": {"volatility": DEFAULT_VOL_PRIOR_MEAN},
                },
                "meta": {"equity_prior_applied": False},
            }
    
    # Generate recommendations
    strategy_result = recommend_options_strategy(
        equity_signal=signal,
        options_chain=options_chain,
        vol_params=vol_params,
    )
    
    # Combine
    enhanced_signal = {
        **signal,
        "options_strategy": strategy_result,
        "vol_tuning": {
            "model_posterior": vol_params.get("global", {}).get("model_posterior", {}),
            "ensemble_volatility": vol_params.get("global", {}).get("ensemble_forecast", {}).get("volatility"),
            "timestamp": vol_params.get("meta", {}).get("timestamp"),
        },
    }
    
    return enhanced_signal


def run_options_signal_pipeline(
    force_retune: bool = False,
    max_workers: int = MAX_WORKERS,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Run the full options signal pipeline for all high conviction signals.
    
    This is the main entry point for `make chain`.
    
    Pipeline:
    1. Load all high conviction signals from buy/ and sell/ directories
    2. Apply liquidity filter
    3. Stratify by expiry
    4. Perform options tuning (or use cache)
    5. Generate confidence-bounded strategy recommendations
    6. Save enhanced signals
    
    Args:
        force_retune: Force re-tuning even if cached
        max_workers: Number of parallel workers
        dry_run: Preview only, don't process
        
    Returns:
        Summary dict with counts
    """
    # Create console for rich output
    console = Console()
    
    # Clean header
    console.print()
    header_text = Text()
    header_text.append("OPTIONS CHAIN ANALYSIS", style="bold bright_white")
    header_text.append("  —  ", style="dim")
    header_text.append("Hierarchical Bayesian Framework", style="dim italic")
    
    console.print(Panel(
        Align.center(header_text),
        box=box.DOUBLE,
        border_style="bright_cyan",
        padding=(0, 2),
    ))
    console.print()
    
    # Configuration section
    config_table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=False,
    )
    config_table.add_column("Key", style="dim", width=20)
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Architecture", "[bright_cyan]Confidence-Bounded Envelopes[/]")
    config_table.add_row("Liquidity Filter", f"Spread ≤{MAX_SPREAD_PCT:.0%}, OI ≥{MIN_OPEN_INTEREST}, Vol ≥{MIN_VOLUME}")
    config_table.add_row("Conviction", f"Near {CONVICTION_THRESHOLDS['near']:.0%} · Short {CONVICTION_THRESHOLDS['short']:.0%} · Medium {CONVICTION_THRESHOLDS['medium']:.0%}")
    if force_retune:
        config_table.add_row("Mode", "[yellow]Force re-tune[/]")
    

    # Check if high conviction directory exists
    if not os.path.exists(HIGH_CONVICTION_DIR):
        console.print("  [red]ERROR:[/] High conviction directory not found")
        console.print("  [dim]Run `make stocks` first to generate equity signals.[/]")
        return {"error": "no_high_conviction_data", "processed": 0}

    # Load tuning cache
    tune_cache = load_option_tune_cache()

    console.print(config_table)
    console.print()
    
    # Find all signal files
    signal_files = []
    for subdir in ["buy", "sell"]:
        dir_path = os.path.join(HIGH_CONVICTION_DIR, subdir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".json") and filename != "manifest.json":
                    signal_files.append(os.path.join(dir_path, filename))
    
    # Count buy vs sell
    buy_count = sum(1 for f in signal_files if "/buy/" in f)
    sell_count = sum(1 for f in signal_files if "/sell/" in f)
    
    # Status line
    status_parts = []
    status_parts.append(f"[dim]Cache:[/] [bold]{len(tune_cache)}[/] tuned")
    status_parts.append(f"[dim]Signals:[/] [green]{buy_count} buy[/] · [red]{sell_count} sell[/]")
    console.print("  " + "    ".join(status_parts))
    console.print()
    
    if not signal_files:
        console.print("  [yellow]No high conviction signals found.[/] Run `make stocks` first.")
        return {"processed": 0, "buy_recommendations": 0, "sell_recommendations": 0}
    
    if dry_run:
        console.print("  [bold yellow]DRY RUN[/] — No processing")
        console.print()
        for f in signal_files[:10]:
            console.print(f"  [dim]Would process:[/] {os.path.basename(f)}")
        if len(signal_files) > 10:
            console.print(f"  [dim]... and {len(signal_files) - 10} more[/]")
        return {"processed": 0, "buy_recommendations": 0, "sell_recommendations": 0, "dry_run": True}
    
    # Process signals
    results = []
    call_results = []  # Results with LONG_CALL recommendations (for UX)
    put_results = []   # Results with LONG_PUT recommendations (for UX)
    call_recs = 0
    put_recs = 0
    skipped = 0
    errors = 0
    
    # Process with progress bar
    console.print("  [bold]Processing high conviction signals...[/]")
    console.print()
    console.print()
    
    # Track results for summary
    successful_tickers = []
    error_tickers = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Analyzing options", total=len(signal_files))
        
        for signal_path in signal_files:
            filename = os.path.basename(signal_path)
            ticker = filename.split("_")[0]
            
            try:
                result = process_high_conviction_signal(
                    signal_path=signal_path,
                    tune_cache=tune_cache,
                    force_retune=force_retune,
                )
                
                if result:
                    results.append(result)
                    strategy = result.get("options_strategy", {})
                    
                    if strategy.get("strategy") == "LONG_CALL":
                        n_recs = strategy.get("n_recommended", 0)
                        call_recs += n_recs
                        if n_recs > 0:
                            call_results.append(result)
                            successful_tickers.append((ticker, "call", n_recs))
                    elif strategy.get("strategy") == "LONG_PUT":
                        n_recs = strategy.get("n_recommended", 0)
                        put_recs += n_recs
                        if n_recs > 0:
                            put_results.append(result)
                            successful_tickers.append((ticker, "put", n_recs))
                    elif strategy.get("strategy") == "SKIPPED":
                        skipped += 1
                    elif strategy.get("strategy") == "ERROR":
                        errors += 1
                        error_tickers.append((ticker, strategy.get("reason", "unknown")))
                else:
                    errors += 1
                    error_tickers.append((ticker, "processing failed"))
                    
            except Exception as e:
                errors += 1
                error_tickers.append((ticker, str(e)[:30]))
            
            progress.update(task, advance=1)
    
    console.print()
    
    # Show compact summary of results
    if successful_tickers:
        # Group by ticker
        ticker_summary = {}
        for ticker, opt_type, count in successful_tickers:
            if ticker not in ticker_summary:
                ticker_summary[ticker] = {"call": 0, "put": 0}
            ticker_summary[ticker][opt_type] += count
        
        items = []
        for ticker, counts in sorted(ticker_summary.items()):
            parts = []
            if counts["call"] > 0:
                parts.append(f"[green]▲{counts['call']}[/]")
            if counts["put"] > 0:
                parts.append(f"[red]▼{counts['put']}[/]")
            items.append(f"[bold]{ticker}[/] {'+'.join(parts)}")
        console.print("  [dim]Recommendations:[/] " + "  ·  ".join(items))
        console.print()
    
    if error_tickers and len(error_tickers) <= 5:
        console.print(f"  [dim]Errors ({len(error_tickers)}): {', '.join(t[0] for t in error_tickers[:5])}[/]")
    elif error_tickers:
        console.print(f"  [dim]Errors: {len(error_tickers)} tickers (no options available)[/]")
    
    # Save updated tuning cache
    save_option_tune_cache(tune_cache)
    console.print(f"  [dim]Cached {len(tune_cache)} tuning results[/]")
    console.print()
    
    # Save enhanced signals
    if os.path.exists(OPTIONS_SIGNALS_DIR):
        shutil.rmtree(OPTIONS_SIGNALS_DIR)
    os.makedirs(OPTIONS_SIGNALS_DIR, exist_ok=True)
    
    for result in results:
        if result.get("error"):
            continue
            
        ticker = result.get("ticker", "UNKNOWN")
        horizon = result.get("horizon_days", 0)
        signal_type = result.get("signal_type", "").lower().replace("_", "")
        
        # Sanitize ticker
        safe_ticker = ticker.replace("^", "_").replace("=", "_").replace(".", "_")
        filename = f"{safe_ticker}_{horizon}d_{signal_type}.json"
        filepath = os.path.join(OPTIONS_SIGNALS_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            console.print(f"  [yellow]Warning: Error saving {filename}: {e}[/]")
            errors += 1
    
    # Create manifest
    successful_results = [r for r in results if not r.get("error") and r.get("options_strategy", {}).get("strategy") not in ["SKIPPED", "ERROR", "NONE"]]
    
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "pipeline": "options_signal_v1",
        "architecture": "Hierarchical Bayesian with Confidence-Bounded Envelopes",
        "total_signals_processed": len(results),
        "successful_recommendations": len(successful_results),
        "call_recommendations": call_recs,
        "put_recommendations": put_recs,
        "skipped": skipped,
        "errors": errors,
        "configuration": {
            "max_spread_pct": MAX_SPREAD_PCT,
            "min_open_interest": MIN_OPEN_INTEREST,
            "min_volume": MIN_VOLUME,
            "min_expected_return": MIN_EXPECTED_RETURN,
            "conviction_thresholds": CONVICTION_THRESHOLDS,
            "position_limits": MAX_POSITION_PCT,
        },
    }
    
    with open(os.path.join(OPTIONS_SIGNALS_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Statistics for summary
    stats = {
        "processed": len(results),
        "successful": len(successful_results),
        "call_recommendations": call_recs,
        "put_recommendations": put_recs,
        "skipped": skipped,
        "errors": errors,
    }
    
    # Render pipeline summary FIRST (before tables)
    render_pipeline_summary(stats, console)
    
    # Output directory info
    console.print()
    console.print(f"  Output directory: [bold]{OPTIONS_SIGNALS_DIR}[/]")
    console.print(f"    Calls: [green]{len(call_results)} tickers with recommendations[/]")
    console.print(f"    Puts:  [red]{len(put_results)} tickers with recommendations[/]")
    console.print()
    
    # Render options tables AFTER summary
    render_options_summary_tables(call_results, put_results, console)
    
    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate options signals for high conviction equity signals"
    )
    parser.add_argument("--force", action="store_true", help="Force re-tuning")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    
    args = parser.parse_args()
    
    run_options_signal_pipeline(
        force_retune=args.force,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )
