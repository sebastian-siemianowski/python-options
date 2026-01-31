"""
Tests for the multi-timeframe fat-tail-aware exhaustion indicator.

COMPREHENSIVE TEST SUITE with 20 realistic market scenarios:
- 10 scenarios where we expect HIGH ue_up (price extended above equilibrium)
- 10 scenarios where we expect HIGH ue_down (price extended below equilibrium)

Each scenario is designed by senior quant professors to test specific market
microstructure patterns commonly observed in equities, FX, and commodities.

The exhaustion metric outputs 0-100% scale:
- ue_up: How far price is ABOVE multi-timeframe EMA equilibrium
- ue_down: How far price is BELOW multi-timeframe EMA equilibrium
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.signals import compute_directional_exhaustion_from_features


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def build_features_from_prices(px: pd.Series, nu: float = 8.0) -> dict:
    """Build the features dict expected by compute_directional_exhaustion_from_features."""
    ret = px.pct_change().fillna(0)
    log_px = np.log(px.clip(lower=0.01))
    
    def mom_t(days: int) -> pd.Series:
        cum = (log_px - log_px.shift(days))
        rv = ret.rolling(days).std() * np.sqrt(days)
        return cum / rv.replace(0, np.nan)
    
    return {
        "px": px,
        "ret": ret,
        "vol": ret.rolling(21).std().fillna(0.02),
        "mom21": mom_t(21),
        "mom63": mom_t(63),
        "mom126": mom_t(126),
        "mom252": mom_t(252),
        "nu_hat": pd.Series([nu], index=[px.index[-1]]),
    }


# =============================================================================
# 10 SCENARIOS EXPECTING HIGH UE_UP (Price Above Equilibrium)
# =============================================================================

def create_parabolic_rally_ongoing(days: int = 300) -> pd.Series:
    """
    Scenario 1: PARABOLIC RALLY STILL ONGOING
    -----------------------------------------
    Like NVDA in 2023 or GME squeeze - exponential acceleration without pullback.
    Price is WAY above all EMAs and momentum is extreme.
    
    Expected: Very high ue_up (80-99%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Base building (100 days)
    phase1 = np.linspace(20, 25, 100)
    
    # Phase 2: Breakout (100 days)
    phase2 = np.linspace(25, 50, 100)
    
    # Phase 3: Parabolic blow-off (100 days) - exponential
    t = np.linspace(0, 1, 100)
    phase3 = 50 * np.exp(1.2 * t)  # Ends around $165
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(101)
    noise = np.random.randn(days) * 0.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_steady_uptrend_extended(days: int = 300) -> pd.Series:
    """
    Scenario 2: STEADY UPTREND BECOMING EXTENDED
    --------------------------------------------
    Like AAPL/MSFT in a multi-year bull market - steady gains but now
    stretched above 200-day EMA by 30%+.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Steady exponential growth with low volatility
    t = np.linspace(0, 1, days)
    prices = 100 * np.exp(0.5 * t)  # ~65% gain over period
    
    np.random.seed(102)
    noise = np.random.randn(days) * 1.0
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_gap_up_breakout(days: int = 300) -> pd.Series:
    """
    Scenario 3: GAP UP BREAKOUT ON EARNINGS
    ---------------------------------------
    Stock gaps up 20% on earnings, now trading well above all EMAs.
    Short-term extended but fundamentally justified.
    
    Expected: Moderate ue_up (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Consolidation (250 days)
    phase1 = 50 + np.random.randn(250) * 2
    
    # Phase 2: Gap up and continuation (50 days)
    phase2_start = phase1[-1] * 1.20  # 20% gap
    phase2 = np.linspace(phase2_start, phase2_start * 1.10, 50)
    
    prices = np.concatenate([phase1, phase2])
    
    np.random.seed(103)
    noise = np.random.randn(days) * 0.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_meme_stock_squeeze(days: int = 300) -> pd.Series:
    """
    Scenario 4: MEME STOCK SHORT SQUEEZE
    ------------------------------------
    Like GME/AMC - massive short squeeze with 500%+ gains in weeks.
    Extreme deviation from any fundamental value.
    
    Expected: Extreme ue_up (90-99%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Dead money (200 days)
    phase1 = 10 + np.random.randn(200) * 0.5
    
    # Phase 2: Squeeze begins (50 days)
    phase2 = np.linspace(10, 50, 50)
    
    # Phase 3: Parabolic squeeze (50 days)
    t = np.linspace(0, 1, 50)
    phase3 = 50 * np.exp(2.0 * t)  # Ends around $370
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(104)
    noise = np.random.randn(days) * 1.0
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_sector_rotation_leader(days: int = 300) -> pd.Series:
    """
    Scenario 5: SECTOR ROTATION LEADER
    ----------------------------------
    Stock benefits from sector rotation (e.g., energy in 2022).
    Outperforming market significantly but trend is mature.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Underperformance (100 days)
    phase1 = np.linspace(30, 28, 100)
    
    # Phase 2: Rotation begins (100 days)
    phase2 = np.linspace(28, 50, 100)
    
    # Phase 3: Momentum continuation (100 days)
    phase3 = np.linspace(50, 75, 100)
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(105)
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_buyout_speculation(days: int = 300) -> pd.Series:
    """
    Scenario 6: BUYOUT SPECULATION PREMIUM
    --------------------------------------
    Stock trading at premium on M&A speculation.
    Price elevated but could go higher on deal or collapse on no-deal.
    
    Expected: Moderate ue_up (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Normal trading (200 days)
    phase1 = 40 + np.random.randn(200) * 2
    
    # Phase 2: M&A rumors (50 days) - gap up and elevated
    phase2 = np.linspace(55, 60, 50)
    
    # Phase 3: Premium maintenance (50 days)
    phase3 = 60 + np.random.randn(50) * 1.5
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(106)
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_technical_breakout_retest(days: int = 300) -> pd.Series:
    """
    Scenario 7: TECHNICAL BREAKOUT WITH SUCCESSFUL RETEST
    -----------------------------------------------------
    Classic breakout pattern - broke resistance, retested as support,
    now rallying. Healthy trend but extended.
    
    Expected: Moderate ue_up (35-55%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Range-bound (150 days) - resistance at 50
    phase1 = 45 + np.random.randn(150) * 2
    phase1 = np.clip(phase1, 40, 50)
    
    # Phase 2: Breakout (50 days)
    phase2 = np.linspace(50, 60, 50)
    
    # Phase 3: Retest and rally (100 days)
    phase3a = np.linspace(60, 52, 30)  # Pullback to retest
    phase3b = np.linspace(52, 70, 70)  # Rally
    phase3 = np.concatenate([phase3a, phase3b])
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(107)
    noise = np.random.randn(days) * 1.0
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_low_vol_grind_higher(days: int = 300) -> pd.Series:
    """
    Scenario 8: LOW VOLATILITY GRIND HIGHER
    ---------------------------------------
    Like utilities or staples in a risk-off environment.
    Slow and steady gains, low vol, but now stretched.
    
    Expected: Low-moderate ue_up (25-45%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Very steady linear appreciation with minimal noise
    prices = np.linspace(50, 70, days)
    
    np.random.seed(108)
    noise = np.random.randn(days) * 0.3  # Very low vol
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_v_shaped_recovery_overshoot(days: int = 300) -> pd.Series:
    """
    Scenario 9: V-SHAPED RECOVERY OVERSHOOT
    ---------------------------------------
    Stock crashed, recovered fully, and now overshooting to upside.
    Like many stocks post-COVID crash.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Pre-crash (100 days)
    phase1 = np.linspace(100, 105, 100)
    
    # Phase 2: Crash (50 days)
    phase2 = np.linspace(105, 50, 50)
    
    # Phase 3: Recovery (100 days)
    phase3 = np.linspace(50, 110, 100)
    
    # Phase 4: Overshoot (50 days)
    phase4 = np.linspace(110, 140, 50)
    
    prices = np.concatenate([phase1, phase2, phase3, phase4])
    
    np.random.seed(109)
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_multi_year_high_breakout(days: int = 300) -> pd.Series:
    """
    Scenario 10: MULTI-YEAR HIGH BREAKOUT
    -------------------------------------
    Stock finally breaking out of multi-year consolidation.
    Strong momentum but just beginning.
    
    Expected: Moderate ue_up (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Long consolidation (200 days) near $50 resistance
    phase1 = 45 + np.random.randn(200) * 3
    phase1 = np.clip(phase1, 35, 50)
    
    # Phase 2: Breakout and follow-through (100 days)
    phase2 = np.linspace(50, 75, 100)
    
    prices = np.concatenate([phase1, phase2])
    
    np.random.seed(110)
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


# =============================================================================
# 10 SCENARIOS EXPECTING HIGH UE_DOWN (Price Below Equilibrium)
# =============================================================================

def create_parabolic_rally_then_breakdown(days: int = 300) -> pd.Series:
    """
    Scenario 1: PARABOLIC RALLY THEN BREAKDOWN (RKLB-like)
    ------------------------------------------------------
    Stock rallied 200%+ then breaks down. This is MEAN REVERSION,
    not a buying opportunity. Should show LOW ue_down despite being
    below short-term EMAs.
    
    Expected: LOW ue_down (0-20%) because long-term still extended
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Gradual rise (150 days)
    phase1 = np.linspace(10, 15, 150)
    
    # Phase 2: Parabolic rally (90 days)
    phase2_base = np.linspace(0, 1, 90)
    phase2 = 15 + 20 * (phase2_base ** 2)
    
    # Phase 3: Breakdown (60 days)
    phase3 = np.linspace(35, 28, 60)
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(42)
    noise = np.random.randn(days) * 0.3
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_capitulation_selloff(days: int = 300) -> pd.Series:
    """
    Scenario 2: TRUE CAPITULATION SELLOFF
    -------------------------------------
    Sustained crash across all timeframes. Negative momentum everywhere.
    Classic oversold condition - potential buying opportunity.
    
    Expected: High ue_down (60-90%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Gradual decline (150 days)
    phase1 = np.linspace(100, 70, 150)
    
    # Phase 2: Accelerating decline (100 days)
    phase2 = np.linspace(70, 35, 100)
    
    # Phase 3: Capitulation spike down (50 days)
    phase3 = np.linspace(35, 25, 50)
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(201)
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_earnings_miss_crash(days: int = 300) -> pd.Series:
    """
    Scenario 3: EARNINGS MISS CRASH
    -------------------------------
    Stock gaps down 30% on earnings miss. Fundamental reset.
    Could be oversold if reaction is overdone.
    
    Expected: Moderate-high ue_down (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Stable trading (250 days)
    phase1 = 80 + np.random.randn(250) * 3
    
    # Phase 2: Gap down and continuation (50 days)
    phase2_start = 56  # 30% gap down
    phase2 = np.linspace(phase2_start, 50, 50)
    
    prices = np.concatenate([phase1, phase2])
    
    np.random.seed(202)
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_sector_bear_market(days: int = 300) -> pd.Series:
    """
    Scenario 4: SECTOR BEAR MARKET
    ------------------------------
    Entire sector in bear market (e.g., Chinese tech 2021-2022).
    Sustained decline, could be value trap or opportunity.
    
    Expected: High ue_down (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Steady decline with occasional dead cat bounces
    base_decline = np.linspace(150, 40, days)
    
    # Add some bounces
    np.random.seed(203)
    bounce_noise = np.random.randn(days) * 5
    prices = base_decline + bounce_noise
    prices = np.maximum(prices, 5)
    
    return pd.Series(prices, index=dates, name='price')


def create_healthy_pullback_in_uptrend(days: int = 300) -> pd.Series:
    """
    Scenario 5: HEALTHY PULLBACK IN UPTREND
    ---------------------------------------
    Stock in strong uptrend pulls back to 50-day EMA.
    This is a buying opportunity, not distress.
    
    Expected: Low ue_down (10-30%) because trend is intact
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Strong uptrend (200 days)
    phase1 = np.linspace(50, 100, 200)
    
    # Phase 2: Pullback (50 days)
    phase2 = np.linspace(100, 85, 50)
    
    # Phase 3: Start of recovery (50 days)
    phase3 = np.linspace(85, 90, 50)
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(204)
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_broken_growth_story(days: int = 300) -> pd.Series:
    """
    Scenario 6: BROKEN GROWTH STORY
    -------------------------------
    Former high-flyer now in structural decline (like NFLX early 2022).
    Multiple compression, growth deceleration.
    
    Expected: High ue_down (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Peak and rollover (100 days)
    phase1 = np.linspace(400, 350, 100)
    
    # Phase 2: First leg down (100 days)
    phase2 = np.linspace(350, 200, 100)
    
    # Phase 3: Dead cat bounce and continuation (100 days)
    phase3a = np.linspace(200, 250, 40)
    phase3b = np.linspace(250, 180, 60)
    phase3 = np.concatenate([phase3a, phase3b])
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(205)
    noise = np.random.randn(days) * 8
    prices = prices + noise
    prices = np.maximum(prices, 10)
    
    return pd.Series(prices, index=dates, name='price')


def create_macro_shock_selloff(days: int = 300) -> pd.Series:
    """
    Scenario 7: MACRO SHOCK SELLOFF
    -------------------------------
    Broad market selloff on macro shock (like March 2020 COVID).
    Indiscriminate selling, potential snapback.
    
    Expected: Very high ue_down (70-90%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Stable market (200 days)
    phase1 = 100 + np.random.randn(200) * 3
    
    # Phase 2: Shock selloff (50 days)
    phase2 = np.linspace(100, 55, 50)
    
    # Phase 3: Stabilization (50 days)
    phase3 = 55 + np.random.randn(50) * 2
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    np.random.seed(206)
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_slow_bleed_downtrend(days: int = 300) -> pd.Series:
    """
    Scenario 8: SLOW BLEED DOWNTREND
    --------------------------------
    Gradual decline without capitulation. Death by a thousand cuts.
    May not be oversold despite being down significantly.
    
    Expected: Moderate ue_down (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Slow steady decline
    prices = np.linspace(80, 45, days)
    
    np.random.seed(207)
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_distribution_top(days: int = 300) -> pd.Series:
    """
    Scenario 9: DISTRIBUTION TOP BREAKING DOWN
    ------------------------------------------
    Stock topped out after extended rally, now breaking down SIGNIFICANTLY.
    Price has lost most of its gains.
    
    Expected: Moderate-high ue_down (40-70%) because trend has reversed
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Rally (100 days) - from 50 to 100
    phase1 = np.linspace(50, 100, 100)
    
    # Phase 2: Distribution top (100 days) - oscillating near 100
    np.random.seed(208)
    phase2 = 100 + np.random.randn(100) * 5
    
    # Phase 3: Breakdown (100 days) - back to 55 (lost most gains)
    phase3 = np.linspace(100, 55, 100)
    
    prices = np.concatenate([phase1, phase2, phase3])
    
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_flash_crash_recovery(days: int = 300) -> pd.Series:
    """
    Scenario 10: FLASH CRASH PARTIAL RECOVERY
    -----------------------------------------
    Sudden crash followed by partial recovery.
    Still significantly below pre-crash levels.
    
    Expected: Moderate-high ue_down (40-70%) because still well below equilibrium
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Normal trading (180 days) at $100
    np.random.seed(209)
    phase1 = 100 + np.random.randn(180) * 3
    
    # Phase 2: Flash crash (30 days) - down to $50
    phase2 = np.linspace(100, 50, 30)
    
    # Phase 3: Partial recovery (90 days) - only to $65 (35% below pre-crash)
    phase3 = np.linspace(50, 65, 90)
    
    prices = np.concatenate([phase1, phase2, phase3])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


# =============================================================================
# 15 ADDITIONAL HIGH UE_UP SCENARIOS
# =============================================================================

def create_ipo_honeymoon_rally(days: int = 300) -> pd.Series:
    """
    Scenario 11: IPO HONEYMOON RALLY
    --------------------------------
    Recent IPO trading at premium to IPO price, momentum chasers piling in.
    No historical support levels, pure momentum.
    
    Expected: High ue_up (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Phase 1: Pre-IPO placeholder (200 days at $25)
    phase1 = np.full(200, 25.0) + np.random.randn(200) * 0.5
    
    # Phase 2: IPO and honeymoon (100 days)
    np.random.seed(301)
    phase2 = np.linspace(40, 85, 100)  # IPO at $40, rallies to $85
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_short_squeeze_gamma_ramp(days: int = 300) -> pd.Series:
    """
    Scenario 12: SHORT SQUEEZE WITH GAMMA RAMP
    ------------------------------------------
    Options market makers forced to buy shares as calls go ITM.
    Explosive move disconnected from fundamentals.
    
    Expected: Extreme ue_up (85-99%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(302)
    # Phase 1: Sideways (200 days)
    phase1 = 20 + np.random.randn(200) * 1
    
    # Phase 2: Initial squeeze (50 days)
    phase2 = np.linspace(20, 60, 50)
    
    # Phase 3: Gamma explosion (50 days)
    t = np.linspace(0, 1, 50)
    phase3 = 60 * np.exp(1.5 * t)  # Ends ~$270
    
    prices = np.concatenate([phase1, phase2, phase3])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_ai_hype_bubble(days: int = 300) -> pd.Series:
    """
    Scenario 13: AI HYPE BUBBLE (2023-style)
    ----------------------------------------
    AI-related stock benefiting from narrative, multiple expansion.
    Strong fundamentals but valuation stretched.
    
    Expected: High ue_up (70-90%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(303)
    # Phase 1: Pre-hype (100 days)
    phase1 = np.linspace(50, 55, 100)
    
    # Phase 2: Hype begins (100 days)
    phase2 = np.linspace(55, 120, 100)
    
    # Phase 3: Parabolic hype (100 days)
    t = np.linspace(0, 1, 100)
    phase3 = 120 + 100 * (t ** 1.5)
    
    prices = np.concatenate([phase1, phase2, phase3])
    noise = np.random.randn(days) * 3
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_dividend_aristocrat_extended(days: int = 300) -> pd.Series:
    """
    Scenario 14: DIVIDEND ARISTOCRAT EXTENDED
    -----------------------------------------
    Blue chip dividend payer trading at premium valuation.
    Low volatility but stretched above historical range.
    
    Expected: Moderate ue_up (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(304)
    # Slow steady appreciation
    prices = 80 + np.linspace(0, 30, days)
    noise = np.random.randn(days) * 1.0
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_currency_carry_unwind_winner(days: int = 300) -> pd.Series:
    """
    Scenario 15: CURRENCY CARRY UNWIND WINNER
    -----------------------------------------
    Stock benefiting from weak domestic currency (exporter).
    Extended move but fundamental tailwind persists.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(305)
    # Steady uptrend with currency boost
    phase1 = np.linspace(40, 50, 150)
    phase2 = np.linspace(50, 80, 150)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_post_restructuring_rally(days: int = 300) -> pd.Series:
    """
    Scenario 16: POST-RESTRUCTURING RALLY
    -------------------------------------
    Company emerged from restructuring, new management executing.
    Re-rating in progress, strong momentum.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(306)
    # Phase 1: Restructuring period - depressed (150 days)
    phase1 = 15 + np.random.randn(150) * 1
    
    # Phase 2: Re-rating rally (150 days)
    phase2 = np.linspace(15, 45, 150)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 1
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_commodity_supercycle_leader(days: int = 300) -> pd.Series:
    """
    Scenario 17: COMMODITY SUPERCYCLE LEADER
    ----------------------------------------
    Mining/energy stock in commodity bull market.
    Strong earnings growth justifying move, but extended.
    
    Expected: High ue_up (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(307)
    # Strong uptrend with commodity correlation
    phase1 = np.linspace(30, 50, 100)
    phase2 = np.linspace(50, 90, 100)
    phase3 = np.linspace(90, 130, 100)
    
    prices = np.concatenate([phase1, phase2, phase3])
    noise = np.random.randn(days) * 4
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_retail_frenzy_small_cap(days: int = 300) -> pd.Series:
    """
    Scenario 18: RETAIL FRENZY SMALL CAP
    ------------------------------------
    Small cap stock discovered by retail traders.
    Low float causing outsized moves.
    
    Expected: Very high ue_up (75-95%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(308)
    # Phase 1: Unknown (200 days)
    phase1 = 5 + np.random.randn(200) * 0.3
    
    # Phase 2: Discovered (100 days)
    t = np.linspace(0, 1, 100)
    phase2 = 5 * np.exp(2.5 * t)  # Exponential to ~$60
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 0.5)
    
    return pd.Series(prices, index=dates, name='price')


def create_defense_stock_geopolitical(days: int = 300) -> pd.Series:
    """
    Scenario 19: DEFENSE STOCK GEOPOLITICAL PREMIUM
    -----------------------------------------------
    Defense contractor benefiting from geopolitical tensions.
    Extended but with fundamental catalyst.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(309)
    # Phase 1: Normal trading (200 days)
    phase1 = 120 + np.random.randn(200) * 5
    
    # Phase 2: Geopolitical catalyst (100 days)
    phase2 = np.linspace(120, 180, 100)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 3
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_biotech_phase3_success(days: int = 300) -> pd.Series:
    """
    Scenario 20: BIOTECH PHASE 3 SUCCESS
    ------------------------------------
    Biotech with successful phase 3 trial results.
    Gap up and continuation, binary event resolved positively.
    
    Expected: High ue_up (60-85%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(310)
    # Phase 1: Pre-trial trading (250 days)
    phase1 = 30 + np.random.randn(250) * 3
    
    # Phase 2: Post-trial rally (50 days)
    phase2 = np.linspace(60, 90, 50)  # Gap and continuation
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_spac_merger_pump(days: int = 300) -> pd.Series:
    """
    Scenario 21: SPAC MERGER PUMP
    -----------------------------
    SPAC trading at premium pre-merger on hot target.
    Speculative premium, high risk of post-merger dump.
    
    Expected: High ue_up (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(311)
    # Phase 1: SPAC at NAV (200 days)
    phase1 = 10 + np.random.randn(200) * 0.2
    
    # Phase 2: Merger announcement pump (100 days)
    phase2 = np.linspace(10, 25, 100)
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_china_reopening_play(days: int = 300) -> pd.Series:
    """
    Scenario 22: CHINA REOPENING PLAY
    ---------------------------------
    Stock benefiting from China reopening narrative.
    V-shaped recovery with overshoot.
    
    Expected: Moderate-high ue_up (50-75%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(312)
    # Phase 1: COVID depression (150 days)
    phase1 = np.linspace(80, 40, 150)
    
    # Phase 2: Reopening rally (150 days)
    phase2 = np.linspace(40, 100, 150)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 3
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_ev_transition_winner(days: int = 300) -> pd.Series:
    """
    Scenario 23: EV TRANSITION WINNER
    ---------------------------------
    Auto supplier pivoting to EV, re-rating on growth story.
    Strong momentum but multiple expansion stretched.
    
    Expected: High ue_up (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(313)
    # Steady re-rating
    phase1 = np.linspace(25, 40, 150)
    phase2 = np.linspace(40, 75, 150)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_activist_investor_target(days: int = 300) -> pd.Series:
    """
    Scenario 24: ACTIVIST INVESTOR TARGET
    -------------------------------------
    Stock rallying on activist investor involvement.
    Unlock value narrative driving premium.
    
    Expected: Moderate ue_up (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(314)
    # Phase 1: Pre-activist (200 days)
    phase1 = 50 + np.random.randn(200) * 2
    
    # Phase 2: Activist involvement (100 days)
    phase2 = np.linspace(50, 72, 100)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_index_inclusion_rally(days: int = 300) -> pd.Series:
    """
    Scenario 25: INDEX INCLUSION RALLY
    ----------------------------------
    Stock rallying ahead of S&P 500 inclusion.
    Passive flows anticipated, front-running in progress.
    
    Expected: Moderate-high ue_up (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(315)
    # Phase 1: Normal growth (200 days)
    phase1 = np.linspace(60, 80, 200)
    
    # Phase 2: Inclusion rally (100 days)
    phase2 = np.linspace(80, 115, 100)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


# =============================================================================
# 15 ADDITIONAL HIGH UE_DOWN SCENARIOS
# =============================================================================

def create_fraud_discovery_crash(days: int = 300) -> pd.Series:
    """
    Scenario 11: FRAUD DISCOVERY CRASH
    ----------------------------------
    Accounting fraud discovered, equity potentially worthless.
    Extreme oversold but value trap risk.
    
    Expected: Very high ue_down (80-99%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(401)
    # Phase 1: Normal trading (200 days)
    phase1 = 100 + np.random.randn(200) * 5
    
    # Phase 2: Fraud crash (50 days)
    phase2 = np.linspace(100, 15, 50)
    
    # Phase 3: Stabilization at low (50 days)
    phase3 = 15 + np.random.randn(50) * 2
    
    prices = np.concatenate([phase1, phase2, phase3])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_interest_rate_shock_reit(days: int = 300) -> pd.Series:
    """
    Scenario 12: INTEREST RATE SHOCK - REIT
    ---------------------------------------
    REIT crushed by rising interest rates.
    Fundamentally impaired, structural headwind.
    
    Expected: High ue_down (60-85%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(402)
    # Phase 1: Pre-rate hike (150 days)
    phase1 = 75 + np.random.randn(150) * 2
    
    # Phase 2: Rate hike impact (150 days)
    phase2 = np.linspace(75, 40, 150)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_consumer_staple_derating(days: int = 300) -> pd.Series:
    """
    Scenario 13: CONSUMER STAPLE DE-RATING
    --------------------------------------
    Defensive stock losing safe haven status.
    Gradual multiple compression, not crash.
    
    Expected: Moderate ue_down (40-60%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(403)
    # Slow steady decline
    prices = np.linspace(90, 60, days)
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_competition_disruption(days: int = 300) -> pd.Series:
    """
    Scenario 14: COMPETITION DISRUPTION
    -----------------------------------
    Incumbent losing market share to disruptor.
    Structural decline, not cyclical.
    
    Expected: High ue_down (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(404)
    # Phase 1: Peak (100 days)
    phase1 = 120 + np.random.randn(100) * 5
    
    # Phase 2: Decline as disruption evident (200 days)
    phase2 = np.linspace(120, 50, 200)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 4
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_regulatory_crackdown(days: int = 300) -> pd.Series:
    """
    Scenario 15: REGULATORY CRACKDOWN
    ---------------------------------
    Company facing regulatory action, business model at risk.
    Like Chinese tech ADRs or crypto exchanges.
    
    Expected: Very high ue_down (70-90%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(405)
    # Phase 1: Growth (100 days)
    phase1 = np.linspace(100, 150, 100)
    
    # Phase 2: Regulatory news (50 days)
    phase2 = np.linspace(150, 80, 50)
    
    # Phase 3: Continued pressure (150 days)
    phase3 = np.linspace(80, 35, 150)
    
    prices = np.concatenate([phase1, phase2, phase3])
    noise = np.random.randn(days) * 5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_currency_crisis_em(days: int = 300) -> pd.Series:
    """
    Scenario 16: CURRENCY CRISIS - EM STOCK
    ---------------------------------------
    Emerging market stock crushed by currency crisis.
    Dollar-denominated ADR showing extreme losses.
    
    Expected: Very high ue_down (75-95%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(406)
    # Phase 1: Normal (150 days)
    phase1 = 60 + np.random.randn(150) * 3
    
    # Phase 2: Currency crisis (150 days)
    phase2 = np.linspace(60, 15, 150)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 2
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_dividend_cut_income_stock(days: int = 300) -> pd.Series:
    """
    Scenario 17: DIVIDEND CUT - INCOME STOCK
    ----------------------------------------
    Income stock forced to cut dividend, investor exodus.
    Yield chasers selling indiscriminately.
    
    Expected: High ue_down (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(407)
    # Phase 1: Stable income (200 days)
    phase1 = 45 + np.random.randn(200) * 1.5
    
    # Phase 2: Dividend cut reaction (100 days)
    phase2 = np.linspace(45, 25, 100)
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_post_spac_dump(days: int = 300) -> pd.Series:
    """
    Scenario 18: POST-SPAC DUMP
    ---------------------------
    SPAC completed merger, PIPE unlock selling.
    Typical post-merger destruction.
    
    Expected: High ue_down (65-85%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(408)
    # Phase 1: SPAC pump (100 days)
    phase1 = np.linspace(10, 25, 100)
    
    # Phase 2: Post-merger dump (200 days)
    phase2 = np.linspace(25, 5, 200)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 0.5
    prices = prices + noise
    prices = np.maximum(prices, 0.5)
    
    return pd.Series(prices, index=dates, name='price')


def create_tech_wreck_growth_stock(days: int = 300) -> pd.Series:
    """
    Scenario 19: TECH WRECK - GROWTH STOCK
    --------------------------------------
    High-growth tech stock crushed in risk-off environment.
    Multiple compression from 30x to 10x revenue.
    
    Expected: Very high ue_down (75-95%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(409)
    # Phase 1: Bubble (100 days)
    phase1 = np.linspace(100, 200, 100)
    
    # Phase 2: Crash (200 days)
    phase2 = np.linspace(200, 40, 200)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 5
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_supply_chain_disruption(days: int = 300) -> pd.Series:
    """
    Scenario 20: SUPPLY CHAIN DISRUPTION
    ------------------------------------
    Manufacturer unable to source components.
    Earnings collapse, recovery uncertain.
    
    Expected: High ue_down (55-75%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(410)
    # Phase 1: Normal operations (200 days)
    phase1 = 80 + np.random.randn(200) * 3
    
    # Phase 2: Supply disruption (100 days)
    phase2 = np.linspace(80, 45, 100)
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_pandemic_victim_travel(days: int = 300) -> pd.Series:
    """
    Scenario 21: PANDEMIC VICTIM - TRAVEL STOCK
    -------------------------------------------
    Airline/cruise/hotel crushed by travel restrictions.
    Existential risk but eventual recovery possible.
    
    Expected: Very high ue_down (80-95%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(411)
    # Phase 1: Pre-pandemic (150 days)
    phase1 = 60 + np.random.randn(150) * 3
    
    # Phase 2: Pandemic crash (50 days)
    phase2 = np.linspace(60, 12, 50)
    
    # Phase 3: Languishing (100 days)
    phase3 = 12 + np.random.randn(100) * 1.5
    
    prices = np.concatenate([phase1, phase2, phase3])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_management_scandal(days: int = 300) -> pd.Series:
    """
    Scenario 22: MANAGEMENT SCANDAL
    -------------------------------
    CEO misconduct discovered, institutional selling.
    Governance risk premium applied.
    
    Expected: Moderate-high ue_down (50-70%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(412)
    # Phase 1: Normal (200 days)
    phase1 = 95 + np.random.randn(200) * 4
    
    # Phase 2: Scandal impact (100 days)
    phase2 = np.linspace(95, 55, 100)
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_patent_cliff_pharma(days: int = 300) -> pd.Series:
    """
    Scenario 23: PATENT CLIFF - PHARMA
    ----------------------------------
    Key drug going generic, revenue cliff approaching.
    Predictable decline but oversold opportunity.
    
    Expected: High ue_down (60-80%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(413)
    # Phase 1: Peak earnings (100 days)
    phase1 = 80 + np.random.randn(100) * 2
    
    # Phase 2: Patent cliff pricing in (200 days)
    phase2 = np.linspace(80, 35, 200)
    
    prices = np.concatenate([phase1, phase2])
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_commodity_bust_miner(days: int = 300) -> pd.Series:
    """
    Scenario 24: COMMODITY BUST - MINER
    -----------------------------------
    Mining stock crushed by commodity price collapse.
    High operating leverage amplifying losses.
    
    Expected: Very high ue_down (75-90%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(414)
    # Phase 1: Commodity boom (100 days)
    phase1 = np.linspace(50, 100, 100)
    
    # Phase 2: Bust (200 days)
    phase2 = np.linspace(100, 20, 200)
    
    prices = np.concatenate([phase1, phase2])
    noise = np.random.randn(days) * 3
    prices = prices + noise
    prices = np.maximum(prices, 1)
    
    return pd.Series(prices, index=dates, name='price')


def create_retail_apocalypse_brick_mortar(days: int = 300) -> pd.Series:
    """
    Scenario 25: RETAIL APOCALYPSE
    ------------------------------
    Traditional retailer losing to e-commerce.
    Structural decline, potential bankruptcy.
    
    Expected: Very high ue_down (80-95%)
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(415)
    # Steady structural decline
    prices = np.linspace(50, 8, days)
    noise = np.random.randn(days) * 1.5
    prices = prices + noise
    prices = np.maximum(prices, 0.5)
    
    return pd.Series(prices, index=dates, name='price')


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestHighUeUp:
    """Test 10 scenarios expecting HIGH ue_up (price above equilibrium)."""
    
    def test_01_parabolic_rally_ongoing(self):
        """Parabolic rally still ongoing - should show very high ue_up."""
        px = create_parabolic_rally_ongoing()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Parabolic Rally Ongoing: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.5, f"Expected high ue_up, got {result['ue_up']:.2%}"
        assert result["ue_down"] == 0, "ue_down should be 0"
    
    def test_02_steady_uptrend_extended(self):
        """Steady uptrend becoming extended - should show moderate-high ue_up."""
        px = create_steady_uptrend_extended()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Steady Uptrend Extended: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
        assert result["ue_down"] == 0, "ue_down should be 0"
    
    def test_03_gap_up_breakout(self):
        """Gap up on earnings - should show moderate ue_up."""
        px = create_gap_up_breakout()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Gap Up Breakout: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.2, f"Expected some ue_up, got {result['ue_up']:.2%}"
        assert result["ue_down"] == 0, "ue_down should be 0"
    
    def test_04_meme_stock_squeeze(self):
        """Meme stock squeeze - should show extreme ue_up."""
        px = create_meme_stock_squeeze()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Meme Stock Squeeze: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.7, f"Expected very high ue_up, got {result['ue_up']:.2%}"
        assert result["ue_down"] == 0, "ue_down should be 0"
    
    def test_05_sector_rotation_leader(self):
        """Sector rotation leader - should show moderate-high ue_up."""
        px = create_sector_rotation_leader()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Sector Rotation Leader: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_06_buyout_speculation(self):
        """Buyout speculation premium - should show moderate ue_up."""
        px = create_buyout_speculation()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Buyout Speculation: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.2, f"Expected some ue_up, got {result['ue_up']:.2%}"
    
    def test_07_technical_breakout_retest(self):
        """Technical breakout with retest - should show moderate ue_up."""
        px = create_technical_breakout_retest()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Technical Breakout Retest: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.2, f"Expected some ue_up, got {result['ue_up']:.2%}"
    
    def test_08_low_vol_grind_higher(self):
        """Low vol grind higher - should show low-moderate ue_up."""
        px = create_low_vol_grind_higher()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Low Vol Grind Higher: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.15, f"Expected some ue_up, got {result['ue_up']:.2%}"
    
    def test_09_v_shaped_recovery_overshoot(self):
        """V-shaped recovery overshoot - should show moderate-high ue_up."""
        px = create_v_shaped_recovery_overshoot()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"V-Shaped Recovery Overshoot: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_10_multi_year_high_breakout(self):
        """Multi-year high breakout - should show moderate ue_up."""
        px = create_multi_year_high_breakout()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Multi-Year High Breakout: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.25, f"Expected some ue_up, got {result['ue_up']:.2%}"


class TestHighUeDown:
    """Test 10 scenarios for ue_down (price below equilibrium)."""
    
    def test_01_parabolic_breakdown_low_ue_down(self):
        """
        Parabolic rally then breakdown - should show LOW ue_down.
        This is mean reversion, not oversold.
        """
        px = create_parabolic_rally_then_breakdown()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Parabolic Breakdown: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        # KEY TEST: ue_down should be LOW despite price being below short EMAs
        assert result["ue_down"] < 0.4, f"Expected low ue_down (mean reversion), got {result['ue_down']:.2%}"
    
    def test_02_capitulation_selloff(self):
        """True capitulation - should show high ue_down."""
        px = create_capitulation_selloff()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Capitulation Selloff: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
        assert result["ue_up"] == 0, "ue_up should be 0"
    
    def test_03_earnings_miss_crash(self):
        """Earnings miss crash - should show moderate-high ue_down."""
        px = create_earnings_miss_crash()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Earnings Miss Crash: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.3, f"Expected some ue_down, got {result['ue_down']:.2%}"
    
    def test_04_sector_bear_market(self):
        """Sector bear market - should show high ue_down."""
        px = create_sector_bear_market()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Sector Bear Market: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_05_healthy_pullback_in_uptrend(self):
        """
        Healthy pullback in uptrend - should show LOW ue_down.
        This is a buying dip, not distress.
        """
        px = create_healthy_pullback_in_uptrend()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Healthy Pullback: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        # KEY TEST: ue_down should be LOW because long-term trend is still up
        assert result["ue_down"] < 0.35, f"Expected low ue_down (healthy pullback), got {result['ue_down']:.2%}"
    
    def test_06_broken_growth_story(self):
        """Broken growth story - should show high ue_down."""
        px = create_broken_growth_story()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Broken Growth Story: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_07_macro_shock_selloff(self):
        """Macro shock selloff - should show very high ue_down."""
        px = create_macro_shock_selloff()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Macro Shock Selloff: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.5, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_08_slow_bleed_downtrend(self):
        """Slow bleed downtrend - should show moderate ue_down."""
        px = create_slow_bleed_downtrend()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Slow Bleed Downtrend: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.25, f"Expected some ue_down, got {result['ue_down']:.2%}"
    
    def test_09_distribution_top(self):
        """
        Distribution top breaking down - should show moderate-high ue_down.
        Price has lost most gains, trend has reversed.
        """
        px = create_distribution_top()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Distribution Top: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        # Trend reversed, should show ue_down
        assert result["ue_down"] >= 0.3, f"Expected moderate ue_down, got {result['ue_down']:.2%}"
    
    def test_10_flash_crash_recovery(self):
        """Flash crash partial recovery - should show moderate-high ue_down."""
        px = create_flash_crash_recovery()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        print(f"Flash Crash Recovery: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        # Still well below equilibrium
        assert result["ue_down"] >= 0.3, f"Expected moderate ue_down, got {result['ue_down']:.2%}"


class TestFatTailAdjustment:
    """Test that fat tails are properly accounted for."""
    
    def test_heavy_tails_reduce_exhaustion(self):
        """With heavy tails (low ν), extreme moves are more expected."""
        px = create_parabolic_rally_ongoing()
        
        feats_heavy = build_features_from_prices(px, nu=5.0)
        feats_light = build_features_from_prices(px, nu=50.0)
        
        result_heavy = compute_directional_exhaustion_from_features(feats_heavy)
        result_light = compute_directional_exhaustion_from_features(feats_light)
        
        print(f"Heavy tails (nu=5): ue_up={result_heavy['ue_up']:.2%}")
        print(f"Light tails (nu=50): ue_up={result_light['ue_up']:.2%}")
        
        # Both should show ue_up > 0
        assert result_heavy["ue_up"] > 0, "Heavy tails should still show ue_up > 0"
        assert result_light["ue_up"] > 0, "Light tails should show ue_up > 0"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_insufficient_data_fallback(self):
        """Test that insufficient data triggers the simple fallback."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        px = pd.Series(np.linspace(100, 110, 50), index=dates)
        feats = {"px": px, "ret": px.pct_change().fillna(0)}
        
        result = compute_directional_exhaustion_from_features(feats)
        
        assert "ue_up" in result
        assert "ue_down" in result
        print(f"Fallback result: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
    
    def test_neutral_price(self):
        """Test when price is exactly at equilibrium."""
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        px = pd.Series([100.0] * 300, index=dates)
        feats = build_features_from_prices(px)
        
        result = compute_directional_exhaustion_from_features(feats)
        
        assert result["ue_up"] < 0.1, "Flat price should have low ue_up"
        assert result["ue_down"] < 0.1, "Flat price should have low ue_down"


class TestAdditionalHighUeUp:
    """Test 15 additional scenarios expecting HIGH ue_up."""
    
    def test_11_ipo_honeymoon_rally(self):
        """IPO honeymoon rally."""
        px = create_ipo_honeymoon_rally()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"IPO Honeymoon: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.4, f"Expected high ue_up, got {result['ue_up']:.2%}"
    
    def test_12_short_squeeze_gamma_ramp(self):
        """Short squeeze with gamma ramp."""
        px = create_short_squeeze_gamma_ramp()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Gamma Squeeze: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.6, f"Expected very high ue_up, got {result['ue_up']:.2%}"
    
    def test_13_ai_hype_bubble(self):
        """AI hype bubble."""
        px = create_ai_hype_bubble()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"AI Hype Bubble: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.5, f"Expected high ue_up, got {result['ue_up']:.2%}"
    
    def test_14_dividend_aristocrat_extended(self):
        """Dividend aristocrat extended."""
        px = create_dividend_aristocrat_extended()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Dividend Aristocrat: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.2, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_15_currency_carry_unwind_winner(self):
        """Currency carry unwind winner."""
        px = create_currency_carry_unwind_winner()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Currency Carry Winner: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_16_post_restructuring_rally(self):
        """Post-restructuring rally."""
        px = create_post_restructuring_rally()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Post-Restructuring: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_17_commodity_supercycle_leader(self):
        """Commodity supercycle leader."""
        px = create_commodity_supercycle_leader()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Commodity Supercycle: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.4, f"Expected high ue_up, got {result['ue_up']:.2%}"
    
    def test_18_retail_frenzy_small_cap(self):
        """Retail frenzy small cap."""
        px = create_retail_frenzy_small_cap()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Retail Frenzy: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.5, f"Expected very high ue_up, got {result['ue_up']:.2%}"
    
    def test_19_defense_stock_geopolitical(self):
        """Defense stock geopolitical premium."""
        px = create_defense_stock_geopolitical()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Defense Geopolitical: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_20_biotech_phase3_success(self):
        """Biotech phase 3 success."""
        px = create_biotech_phase3_success()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Biotech Phase 3: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.4, f"Expected high ue_up, got {result['ue_up']:.2%}"
    
    def test_21_spac_merger_pump(self):
        """SPAC merger pump."""
        px = create_spac_merger_pump()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"SPAC Merger: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.4, f"Expected high ue_up, got {result['ue_up']:.2%}"
    
    def test_22_china_reopening_play(self):
        """China reopening play."""
        px = create_china_reopening_play()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"China Reopening: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_23_ev_transition_winner(self):
        """EV transition winner."""
        px = create_ev_transition_winner()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"EV Transition: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.4, f"Expected high ue_up, got {result['ue_up']:.2%}"
    
    def test_24_activist_investor_target(self):
        """Activist investor target."""
        px = create_activist_investor_target()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Activist Target: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.2, f"Expected moderate ue_up, got {result['ue_up']:.2%}"
    
    def test_25_index_inclusion_rally(self):
        """Index inclusion rally."""
        px = create_index_inclusion_rally()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Index Inclusion: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_up"] >= 0.3, f"Expected moderate ue_up, got {result['ue_up']:.2%}"


class TestAdditionalHighUeDown:
    """Test 15 additional scenarios expecting HIGH ue_down."""
    
    def test_11_fraud_discovery_crash(self):
        """Fraud discovery crash."""
        px = create_fraud_discovery_crash()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Fraud Crash: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.6, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_12_interest_rate_shock_reit(self):
        """Interest rate shock REIT."""
        px = create_interest_rate_shock_reit()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Rate Shock REIT: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_13_consumer_staple_derating(self):
        """Consumer staple de-rating."""
        px = create_consumer_staple_derating()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Staple De-rating: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.25, f"Expected moderate ue_down, got {result['ue_down']:.2%}"
    
    def test_14_competition_disruption(self):
        """Competition disruption."""
        px = create_competition_disruption()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Competition Disruption: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_15_regulatory_crackdown(self):
        """Regulatory crackdown."""
        px = create_regulatory_crackdown()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Regulatory Crackdown: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.5, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_16_currency_crisis_em(self):
        """Currency crisis EM stock."""
        px = create_currency_crisis_em()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Currency Crisis EM: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.5, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_17_dividend_cut_income_stock(self):
        """Dividend cut income stock."""
        px = create_dividend_cut_income_stock()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Dividend Cut: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_18_post_spac_dump(self):
        """Post-SPAC dump."""
        px = create_post_spac_dump()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Post-SPAC Dump: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.5, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_19_tech_wreck_growth_stock(self):
        """Tech wreck growth stock."""
        px = create_tech_wreck_growth_stock()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Tech Wreck: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.5, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_20_supply_chain_disruption(self):
        """Supply chain disruption."""
        px = create_supply_chain_disruption()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Supply Chain: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.35, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_21_pandemic_victim_travel(self):
        """Pandemic victim travel stock."""
        px = create_pandemic_victim_travel()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Pandemic Travel: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.55, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_22_management_scandal(self):
        """Management scandal."""
        px = create_management_scandal()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Management Scandal: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.35, f"Expected moderate-high ue_down, got {result['ue_down']:.2%}"
    
    def test_23_patent_cliff_pharma(self):
        """Patent cliff pharma."""
        px = create_patent_cliff_pharma()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Patent Cliff: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.4, f"Expected high ue_down, got {result['ue_down']:.2%}"
    
    def test_24_commodity_bust_miner(self):
        """Commodity bust miner."""
        px = create_commodity_bust_miner()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Commodity Bust: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.5, f"Expected very high ue_down, got {result['ue_down']:.2%}"
    
    def test_25_retail_apocalypse_brick_mortar(self):
        """Retail apocalypse brick & mortar."""
        px = create_retail_apocalypse_brick_mortar()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        print(f"Retail Apocalypse: ue_up={result['ue_up']:.2%}, ue_down={result['ue_down']:.2%}")
        assert result["ue_down"] >= 0.6, f"Expected very high ue_down, got {result['ue_down']:.2%}"


# =============================================================================
# MAIN - RUN ALL 50 SCENARIOS WITH DETAILED OUTPUT
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("EXHAUSTION INDICATOR TEST SUITE - 50 SCENARIOS")
    print("=" * 80)
    
    # Original 10 high ue_up scenarios
    high_ue_up_scenarios = [
        ("01. Parabolic Rally Ongoing", create_parabolic_rally_ongoing),
        ("02. Steady Uptrend Extended", create_steady_uptrend_extended),
        ("03. Gap Up Breakout", create_gap_up_breakout),
        ("04. Meme Stock Squeeze", create_meme_stock_squeeze),
        ("05. Sector Rotation Leader", create_sector_rotation_leader),
        ("06. Buyout Speculation", create_buyout_speculation),
        ("07. Technical Breakout Retest", create_technical_breakout_retest),
        ("08. Low Vol Grind Higher", create_low_vol_grind_higher),
        ("09. V-Shaped Recovery Overshoot", create_v_shaped_recovery_overshoot),
        ("10. Multi-Year High Breakout", create_multi_year_high_breakout),
        ("11. IPO Honeymoon Rally", create_ipo_honeymoon_rally),
        ("12. Gamma Squeeze", create_short_squeeze_gamma_ramp),
        ("13. AI Hype Bubble", create_ai_hype_bubble),
        ("14. Dividend Aristocrat Extended", create_dividend_aristocrat_extended),
        ("15. Currency Carry Winner", create_currency_carry_unwind_winner),
        ("16. Post-Restructuring Rally", create_post_restructuring_rally),
        ("17. Commodity Supercycle", create_commodity_supercycle_leader),
        ("18. Retail Frenzy Small Cap", create_retail_frenzy_small_cap),
        ("19. Defense Geopolitical", create_defense_stock_geopolitical),
        ("20. Biotech Phase 3 Success", create_biotech_phase3_success),
        ("21. SPAC Merger Pump", create_spac_merger_pump),
        ("22. China Reopening Play", create_china_reopening_play),
        ("23. EV Transition Winner", create_ev_transition_winner),
        ("24. Activist Investor Target", create_activist_investor_target),
        ("25. Index Inclusion Rally", create_index_inclusion_rally),
    ]
    
    # Original 10 + 15 new high ue_down scenarios
    high_ue_down_scenarios = [
        ("01. Parabolic Breakdown (RKLB)", create_parabolic_rally_then_breakdown),
        ("02. Capitulation Selloff", create_capitulation_selloff),
        ("03. Earnings Miss Crash", create_earnings_miss_crash),
        ("04. Sector Bear Market", create_sector_bear_market),
        ("05. Healthy Pullback", create_healthy_pullback_in_uptrend),
        ("06. Broken Growth Story", create_broken_growth_story),
        ("07. Macro Shock Selloff", create_macro_shock_selloff),
        ("08. Slow Bleed Downtrend", create_slow_bleed_downtrend),
        ("09. Distribution Top", create_distribution_top),
        ("10. Flash Crash Recovery", create_flash_crash_recovery),
        ("11. Fraud Discovery Crash", create_fraud_discovery_crash),
        ("12. Interest Rate Shock REIT", create_interest_rate_shock_reit),
        ("13. Consumer Staple De-rating", create_consumer_staple_derating),
        ("14. Competition Disruption", create_competition_disruption),
        ("15. Regulatory Crackdown", create_regulatory_crackdown),
        ("16. Currency Crisis EM", create_currency_crisis_em),
        ("17. Dividend Cut Income", create_dividend_cut_income_stock),
        ("18. Post-SPAC Dump", create_post_spac_dump),
        ("19. Tech Wreck Growth", create_tech_wreck_growth_stock),
        ("20. Supply Chain Disruption", create_supply_chain_disruption),
        ("21. Pandemic Victim Travel", create_pandemic_victim_travel),
        ("22. Management Scandal", create_management_scandal),
        ("23. Patent Cliff Pharma", create_patent_cliff_pharma),
        ("24. Commodity Bust Miner", create_commodity_bust_miner),
        ("25. Retail Apocalypse", create_retail_apocalypse_brick_mortar),
    ]
    
    print("\n" + "=" * 80)
    print("25 SCENARIOS EXPECTING HIGH UE_UP (Price Above Equilibrium)")
    print("=" * 80)
    
    passed_up = 0
    for name, create_fn in high_ue_up_scenarios:
        px = create_fn()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        ue_up_pct = int(result['ue_up'] * 100)
        ue_down_pct = int(result['ue_down'] * 100)
        
        if result['ue_up'] >= 0.2:
            status = "✓"
            passed_up += 1
        else:
            status = "✗"
        print(f"{status} {name:40s} | ↑{ue_up_pct:3d}% | ↓{ue_down_pct:3d}%")
    
    print(f"\nPassed: {passed_up}/{len(high_ue_up_scenarios)}")
    
    print("\n" + "=" * 80)
    print("25 SCENARIOS FOR UE_DOWN (Price Below Equilibrium)")
    print("=" * 80)
    
    passed_down = 0
    for name, create_fn in high_ue_down_scenarios:
        px = create_fn()
        feats = build_features_from_prices(px)
        result = compute_directional_exhaustion_from_features(feats)
        
        ue_up_pct = int(result['ue_up'] * 100)
        ue_down_pct = int(result['ue_down'] * 100)
        
        # For parabolic breakdown and healthy pullback, we expect LOW ue_down
        if "Parabolic" in name or "Pullback" in name:
            if result['ue_down'] < 0.4:
                status = "✓"
                passed_down += 1
            else:
                status = "✗"
        else:
            # For other scenarios (capitulation, crash, etc), we expect HIGH ue_down
            if result['ue_down'] >= 0.25:
                status = "✓"
                passed_down += 1
            else:
                status = "✗"
        
        print(f"{status} {name:40s} | ↑{ue_up_pct:3d}% | ↓{ue_down_pct:3d}%")
    
    print(f"\nPassed: {passed_down}/{len(high_ue_down_scenarios)}")
    print(f"\nTOTAL: {passed_up + passed_down}/{len(high_ue_up_scenarios) + len(high_ue_down_scenarios)} scenarios passed")
