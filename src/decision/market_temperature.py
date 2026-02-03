"""
===============================================================================
MARKET TEMPERATURE MODULE
===============================================================================

Comprehensive US Equity Market Health Assessment (February 2026)

DESIGN PRINCIPLE:
    "The market tells you everything — if you listen to the right instruments."

This module computes a comprehensive market temperature from multiple stock
universe segments to give a complete picture of market health and risk:

UNIVERSE SEGMENTS:
    1. Top 100 US Stocks (40%): Mega-cap bellwethers (Apple, Microsoft, etc.)
       - Represents institutional flows and macro sentiment
       
    2. S&P 500 Proxy (30%): Broad market health via SPY components
       - Core market breadth indicator
       
    3. Russell 2000 Small-Caps (20%): Risk appetite indicator
       - Small caps lead in risk-on, lag in risk-off
       
    4. Growth vs Value Spread (10%): Rotation signals
       - QQQ/IWD spread reveals regime shifts

COMPUTED METRICS:
    - Market Temperature: Aggregate stress score ∈ [0, 2]
    - Crash Risk: Probability of significant drawdown (vol inversion, etc.)
    - Momentum: Multi-timeframe momentum across segments
    - Breadth: Advance/decline ratios and participation
    - Volatility Regime: Current vol vs historical percentiles
    - Correlation Stress: Rising correlations = systemic risk

REFERENCES:
    Expert Panel Design (February 2026)
    Mirrors metals_risk_temperature.py and risk_temperature.py patterns

===============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# COPULA-BASED CORRELATION STRESS (February 2026)
# =============================================================================
# Import copula correlation module for tail dependency analysis
# Falls back to Pearson correlation if copula module unavailable
# =============================================================================
try:
    from calibration.copula_correlation import (
        compute_copula_correlation_stress,
        compute_smooth_scale_factor,
        CopulaCorrelationStress,
        COPULA_CORRELATION_AVAILABLE,
    )
    USE_COPULA_CORRELATION = True
except ImportError:
    USE_COPULA_CORRELATION = False
    COPULA_CORRELATION_AVAILABLE = False

# =============================================================================
# MARKET TEMPERATURE CONSTANTS
# =============================================================================

# Universe weights (sum to 1.0)
MEGA_CAP_WEIGHT = 0.40       # Top 100 mega-caps
BROAD_MARKET_WEIGHT = 0.30   # S&P 500 proxy
SMALL_CAP_WEIGHT = 0.20      # Russell 2000
GROWTH_VALUE_WEIGHT = 0.10   # Growth vs Value rotation

# Z-score calculation lookback
ZSCORE_LOOKBACK_DAYS = 60

# Volatility percentile lookback
VOLATILITY_LOOKBACK_DAYS = 252  # 1 year

# Scaling function parameters
SIGMOID_K = 3.0
SIGMOID_THRESHOLD = 1.0

# Temperature bounds
TEMP_MIN = 0.0
TEMP_MAX = 2.0

# Cache TTL (seconds)
CACHE_TTL_SECONDS = 3600  # 1 hour

# MAD consistency constant for robust z-score
MAD_CONSISTENCY_CONSTANT = 1.4826

# Vol term structure inversion
VOL_TERM_STRUCTURE_SHORT_WINDOW = 5
VOL_TERM_STRUCTURE_LONG_WINDOW = 20
VOL_TERM_STRUCTURE_INVERSION_THRESHOLD = 1.5

# Breadth thresholds
BREADTH_WARNING_THRESHOLD = 0.40   # < 40% above 50-day MA = warning
BREADTH_DANGER_THRESHOLD = 0.25    # < 25% = danger

# Momentum thresholds
MOMENTUM_STRONG_BULL = 0.15    # > 15% in 21 days
MOMENTUM_MILD_BULL = 0.05      # > 5%
MOMENTUM_MILD_BEAR = -0.05     # < -5%
MOMENTUM_STRONG_BEAR = -0.15   # < -15%

# Correlation stress
CORRELATION_LOOKBACK = 60
CORRELATION_STRESS_THRESHOLD = 0.75  # Avg correlation > 0.75 = systemic risk


class AlertSeverity:
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UniverseMetrics:
    """Metrics for a single universe segment (e.g., Top 100, Small Caps)."""
    name: str
    weight: float
    
    # Price/Return metrics
    current_level: Optional[float] = None
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    return_63d: float = 0.0  # 3 months
    
    # Volatility metrics
    volatility_20d: float = 0.0
    volatility_percentile: float = 0.5
    vol_term_structure_ratio: float = 1.0
    vol_inverted: bool = False
    
    # Breadth (for indices with components)
    breadth_pct_above_50ma: Optional[float] = None
    breadth_pct_above_200ma: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    
    # Computed stress
    stress_level: float = 0.0
    stress_contribution: float = 0.0
    
    # Momentum signal
    momentum_signal: str = "→ Flat"
    
    # Data quality
    data_available: bool = False
    ticker_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "current_level": self.current_level,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "return_63d": self.return_63d,
            "volatility_20d": self.volatility_20d,
            "volatility_percentile": self.volatility_percentile,
            "vol_term_structure_ratio": self.vol_term_structure_ratio,
            "vol_inverted": self.vol_inverted,
            "breadth_pct_above_50ma": self.breadth_pct_above_50ma,
            "breadth_pct_above_200ma": self.breadth_pct_above_200ma,
            "advance_decline_ratio": self.advance_decline_ratio,
            "stress_level": self.stress_level,
            "stress_contribution": self.stress_contribution,
            "momentum_signal": self.momentum_signal,
            "data_available": self.data_available,
            "ticker_count": self.ticker_count,
        }


@dataclass
class SectorMetrics:
    """Metrics for a single sector (e.g., Technology, Financials)."""
    name: str
    ticker: str  # ETF ticker (e.g., XLK for Technology)
    
    # Price/Return metrics
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    
    # Volatility
    volatility_20d: float = 0.0
    volatility_percentile: float = 0.5
    
    # Momentum
    momentum_signal: str = "→ Flat"
    
    # Risk score (0-100)
    risk_score: int = 0
    
    # Data availability
    data_available: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ticker": self.ticker,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "volatility_20d": self.volatility_20d,
            "volatility_percentile": self.volatility_percentile,
            "momentum_signal": self.momentum_signal,
            "risk_score": self.risk_score,
            "data_available": self.data_available,
        }


@dataclass
class CurrencyMetrics:
    """Metrics for a currency pair (e.g., EUR/USD, GBP/USD)."""
    name: str
    ticker: str
    rate: float = 0.0
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    volatility_20d: float = 0.0
    momentum_signal: str = "→ Flat"
    risk_score: int = 0
    data_available: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ticker": self.ticker,
            "rate": self.rate,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "volatility_20d": self.volatility_20d,
            "momentum_signal": self.momentum_signal,
            "risk_score": self.risk_score,
            "data_available": self.data_available,
        }


@dataclass
class MarketBreadthMetrics:
    """Aggregated market breadth analysis."""
    pct_above_50ma: float = 0.5
    pct_above_200ma: float = 0.5
    new_highs: int = 0
    new_lows: int = 0
    advance_decline_ratio: float = 1.0
    breadth_thrust: bool = False  # Extreme breadth signal
    breadth_warning: bool = False
    interpretation: str = "Normal"
    
    def to_dict(self) -> Dict:
        return {
            "pct_above_50ma": self.pct_above_50ma,
            "pct_above_200ma": self.pct_above_200ma,
            "new_highs": self.new_highs,
            "new_lows": self.new_lows,
            "advance_decline_ratio": self.advance_decline_ratio,
            "breadth_thrust": self.breadth_thrust,
            "breadth_warning": self.breadth_warning,
            "interpretation": self.interpretation,
        }


@dataclass
class CorrelationStress:
    """Cross-asset correlation analysis."""
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_percentile: float = 0.5
    systemic_risk_elevated: bool = False
    interpretation: str = "Normal dispersion"
    
    def to_dict(self) -> Dict:
        return {
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "correlation_percentile": self.correlation_percentile,
            "systemic_risk_elevated": self.systemic_risk_elevated,
            "interpretation": self.interpretation,
        }


@dataclass
class MarketTemperatureResult:
    """Complete market temperature computation result."""
    temperature: float                       # Final temperature ∈ [0, 2]
    scale_factor: float                      # Position scaling factor ∈ (0, 1)
    universes: Dict[str, UniverseMetrics]    # Per-universe metrics
    breadth: MarketBreadthMetrics            # Market breadth analysis
    correlation: CorrelationStress           # Correlation stress analysis
    computed_at: str                         # ISO timestamp
    data_quality: float                      # Fraction of universes with data
    status: str                              # Calm, Elevated, Stressed, Extreme
    action_text: str                         # Position recommendation
    
    # Crash risk fields
    crash_risk_pct: float = 0.0
    crash_risk_level: str = "Low"
    vol_inversion_count: int = 0
    inverted_universes: Optional[List[str]] = None
    
    # Momentum summary
    overall_momentum: str = "→ Neutral"
    sector_rotation_signal: str = "Normal"
    
    # Exit signals
    exit_signal: bool = False
    exit_reason: Optional[str] = None
    
    # Sector-by-sector breakdown (February 2026)
    sectors: Dict[str, SectorMetrics] = field(default_factory=dict)
    
    # Currency pairs breakdown (February 2026)
    currencies: Dict[str, CurrencyMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "temperature": float(self.temperature),
            "scale_factor": float(self.scale_factor),
            "status": self.status,
            "action_text": self.action_text,
            "computed_at": self.computed_at,
            "data_quality": float(self.data_quality),
            "universes": {k: v.to_dict() for k, v in self.universes.items()},
            "breadth": self.breadth.to_dict(),
            "correlation": self.correlation.to_dict(),
            "crash_risk_pct": float(self.crash_risk_pct),
            "crash_risk_level": self.crash_risk_level,
            "vol_inversion_count": self.vol_inversion_count,
            "inverted_universes": self.inverted_universes,
            "overall_momentum": self.overall_momentum,
            "sector_rotation_signal": self.sector_rotation_signal,
            "exit_signal": self.exit_signal,
            "exit_reason": self.exit_reason,
            "sectors": {k: v.to_dict() for k, v in self.sectors.items()},
            "currencies": {k: v.to_dict() for k, v in self.currencies.items()},
        }
    
    @property
    def is_elevated(self) -> bool:
        return self.temperature > 0.5
    
    @property
    def is_stressed(self) -> bool:
        return self.temperature > 1.0
    
    @property
    def is_extreme(self) -> bool:
        return self.temperature > 1.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_robust_zscore(
    values: pd.Series,
    lookback: int = ZSCORE_LOOKBACK_DAYS
) -> float:
    """Compute robust z-score using Median Absolute Deviation (MAD)."""
    try:
        if values is None or len(values) == 0:
            return 0.0
        
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 1:
                values = values.iloc[:, 0]
            else:
                return 0.0
        
        values = values.dropna()
        
        if len(values) < lookback // 2:
            return 0.0
        
        recent = values.iloc[-lookback:] if len(values) >= lookback else values
        current = float(values.iloc[-1])
        
        median = float(recent.median())
        mad = float((recent - median).abs().median())
        
        scaled_mad = MAD_CONSISTENCY_CONSTANT * mad
        
        if scaled_mad < 1e-10:
            std = float(recent.std())
            if std < 1e-10:
                return 0.0
            zscore = (current - recent.mean()) / std
        else:
            zscore = (current - median) / scaled_mad
        
        return float(np.clip(zscore, -5.0, 5.0))
    except Exception:
        return 0.0


def _compute_volatility_percentile(
    prices: pd.Series,
    vol_window: int = 20,
    lookback: int = VOLATILITY_LOOKBACK_DAYS
) -> float:
    """Compute current volatility percentile over lookback period."""
    try:
        if prices is None or len(prices) < vol_window + 10:
            return 0.5
        
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return 0.5
        
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) < 20:
            return 0.5
        
        current_vol = float(rolling_vol.iloc[-1])
        historical_vol = rolling_vol.iloc[-lookback:] if len(rolling_vol) >= lookback else rolling_vol
        
        percentile = (historical_vol < current_vol).sum() / len(historical_vol)
        return float(percentile)
    except Exception:
        return 0.5


def _compute_vol_term_structure(
    prices: pd.Series,
    short_window: int = VOL_TERM_STRUCTURE_SHORT_WINDOW,
    long_window: int = VOL_TERM_STRUCTURE_LONG_WINDOW,
) -> Tuple[float, bool]:
    """Compute volatility term structure ratio. Returns (ratio, is_inverted)."""
    try:
        if prices is None or len(prices) < long_window + 5:
            return 1.0, False
        
        returns = prices.pct_change().dropna()
        
        if len(returns) < long_window:
            return 1.0, False
        
        short_vol = float(returns.iloc[-short_window:].std() * np.sqrt(252))
        long_vol = float(returns.iloc[-long_window:].std() * np.sqrt(252))
        
        if long_vol < 1e-10:
            return 1.0, False
        
        ratio = short_vol / long_vol
        is_inverted = ratio >= VOL_TERM_STRUCTURE_INVERSION_THRESHOLD
        
        return ratio, is_inverted
    except Exception:
        return 1.0, False


def _compute_returns(prices: pd.Series) -> Dict[str, float]:
    """Compute returns over multiple horizons."""
    results = {"1d": 0.0, "5d": 0.0, "21d": 0.0, "63d": 0.0}
    
    try:
        if prices is None or len(prices) < 2:
            return results
        
        prices = prices.dropna()
        current = float(prices.iloc[-1])
        
        if len(prices) >= 2:
            results["1d"] = (current / float(prices.iloc[-2]) - 1)
        if len(prices) >= 6:
            results["5d"] = (current / float(prices.iloc[-6]) - 1)
        if len(prices) >= 22:
            results["21d"] = (current / float(prices.iloc[-22]) - 1)
        if len(prices) >= 64:
            results["63d"] = (current / float(prices.iloc[-64]) - 1)
    except Exception:
        pass
    
    return results


def _compute_momentum_signal(return_5d: float, return_21d: float) -> str:
    """Compute momentum signal from returns."""
    avg_momentum = (return_5d + return_21d) / 2
    
    if avg_momentum >= MOMENTUM_STRONG_BULL:
        return "↑ Strong"
    elif avg_momentum >= MOMENTUM_MILD_BULL:
        return "↗ Rising"
    elif avg_momentum <= MOMENTUM_STRONG_BEAR:
        return "↓ Weak"
    elif avg_momentum <= MOMENTUM_MILD_BEAR:
        return "↘ Falling"
    else:
        return "→ Flat"


def _compute_scale_factor(temperature: float) -> float:
    """
    Compute position scale factor using smooth exponential decay.
    
    February 2026 Enhancement (Professor Zhang Xin-Yu, Score: 8.7/10):
    Replaces sigmoid-based scaling with smooth exponential decay that
    eliminates discontinuities and uses hysteresis to prevent oscillation.
    
    Formula:
        scale = exp(-decay_rate * max(0, temperature - threshold))
    
    Properties:
        - Continuous and differentiable everywhere
        - scale = 1.0 when temperature <= threshold
        - Smooth decay as temperature increases
    """
    if USE_COPULA_CORRELATION and COPULA_CORRELATION_AVAILABLE:
        return compute_smooth_scale_factor(
            temperature,
            threshold=SIGMOID_THRESHOLD,
            decay_rate=SIGMOID_K * 0.7,  # Slightly gentler than sigmoid equivalent
            hysteresis_band=0.05,
            state_key="market_temperature"
        )
    
    # Fallback to original sigmoid
    return 1.0 / (1.0 + math.exp(SIGMOID_K * (temperature - SIGMOID_THRESHOLD)))


def _get_status_and_action(temperature: float) -> Tuple[str, str]:
    """Get status label and action text based on temperature."""
    if temperature >= 1.5:
        return "Extreme", "EXIT POSITIONS - Capital preservation mode"
    elif temperature >= 1.0:
        return "Stressed", "REDUCE EXPOSURE - Risk management priority"
    elif temperature >= 0.7:
        return "Elevated", "CAUTION - Consider hedging"
    elif temperature >= 0.5:
        return "Warm", "MONITOR - Tighten stops"
    else:
        return "Calm", "NORMAL - Business as usual"


def _extract_close_series(df, ticker: str) -> Optional[pd.Series]:
    """Safely extract Close price series from yfinance DataFrame."""
    if df is None:
        return None
    
    try:
        if hasattr(df, 'empty') and df.empty:
            return None
    except ValueError:
        if len(df) == 0:
            return None
    
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                # Check if it's a multi-ticker DataFrame
                close_df = df['Close']
                if isinstance(close_df, pd.DataFrame):
                    # Try to get the specific ticker's column
                    if ticker in close_df.columns:
                        return close_df[ticker].dropna()
                    # Try without special characters (e.g., ^VIX -> VIX)
                    ticker_clean = ticker.replace('^', '').replace('=X', '')
                    if ticker_clean in close_df.columns:
                        return close_df[ticker_clean].dropna()
                    # Fallback to first column only if single column
                    if len(close_df.columns) == 1:
                        return close_df.iloc[:, 0].dropna()
                    # For multi-column, don't fallback - return None to avoid wrong data
                    return None
                else:
                    # It's already a Series
                    return close_df.dropna()
        
        if 'Close' in df.columns:
            series = df['Close']
            if isinstance(series, pd.DataFrame):
                # Same logic for multi-ticker case
                if ticker in series.columns:
                    return series[ticker].dropna()
                ticker_clean = ticker.replace('^', '').replace('=X', '')
                if ticker_clean in series.columns:
                    return series[ticker_clean].dropna()
                if len(series.columns) == 1:
                    return series.iloc[:, 0].dropna()
                return None
            return series.dropna()
        
        if 'close' in df.columns:
            return df['close'].dropna()
        
        # Only use first column fallback for single-column DataFrames
        if len(df.columns) == 1:
            return df.iloc[:, 0].dropna()
    except Exception:
        pass
    
    return None


# =============================================================================
# TOP 100 US STOCKS - Mega-Cap Universe
# =============================================================================

# Top 100 US stocks by market cap (static list - updated periodically)
TOP_100_TICKERS = [
    # Top 10 Mega-Caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    # 11-25
    "V", "XOM", "JPM", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO",
    # 26-50
    "TMO", "MCD", "CSCO", "ABT", "ACN", "WFC", "DHR", "ADBE", "CRM", "NKE",
    "TXN", "NEE", "PM", "BMY", "ORCL", "LIN", "UNP", "CMCSA", "UPS", "RTX",
    "AMD", "QCOM", "T", "LOW", "MS",
    # 51-75
    "HON", "INTC", "SPGI", "COP", "CAT", "BA", "GS", "ELV", "SBUX", "DE",
    "IBM", "INTU", "PLD", "AMGN", "GE", "AXP", "BKNG", "ISRG", "MDLZ", "GILD",
    "BLK", "ADI", "MMC", "REGN", "CVS",
    # 76-100
    "TJX", "VRTX", "SYK", "SCHW", "ADP", "C", "PGR", "ZTS", "LRCX", "CI",
    "CB", "NOW", "MO", "SO", "DUK", "SLB", "EOG", "PNC", "BDX", "ITW",
    "CL", "USB", "CME", "MCO", "APD",
]

# S&P 500 Representative Sample (50 stocks across sectors)
SP500_SAMPLE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "ACN",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrials
    "CAT", "HON", "UNP", "BA", "GE", "RTX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "VZ",
    # Materials/Utilities
    "LIN", "APD", "NEE", "DUK", "SO",
]

# Russell 2000 Representative Sample (30 small-cap stocks)
RUSSELL_2000_SAMPLE = [
    "AMC", "GME", "BBBY", "SPCE", "PLTR",  # High-volatility small caps
    "CROX", "FIVE", "BOOT", "WSM", "PRGS",  # Consumer/Retail
    "PRLB", "NMIH", "ESNT", "RDN", "MTG",   # Financials
    "CARG", "VCEL", "FOLD", "AXNX", "TGTX", # Healthcare
    "DOCN", "NEOG", "ZI", "AMBA", "CRNC",   # Tech
    "MATX", "ARCB", "WERN", "SAIA", "XPO",  # Industrials
]


# =============================================================================
# SECTOR ETFs - For sector-by-sector breakdown (February 2026)
# =============================================================================
# SPDR Select Sector ETFs tracking S&P 500 sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Comm. Svcs",
}


# =============================================================================
# MARKET DATA CACHE
# =============================================================================

# =============================================================================
# SECTOR ETFs - For sector-by-sector breakdown (February 2026)
# =============================================================================
# SPDR Select Sector ETFs tracking S&P 500 sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Comm. Svcs",
}

# =============================================================================
# CURRENCY PAIRS - For FX market breakdown (February 2026)
# =============================================================================
# Major currency pairs and cryptocurrencies with Yahoo Finance tickers
CURRENCY_PAIRS = {
    # Major FX pairs
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    "AUDJPY=X": "AUD/JPY",
    # Cryptocurrencies
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}


_market_data_cache: Dict[str, Tuple[datetime, Any]] = {}


class _SuppressOutput:
    """Context manager to fully suppress stdout and stderr at file descriptor level."""
    def __enter__(self):
        import os
        import sys
        # Save the actual stdout/stderr file descriptors
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        # Open /dev/null
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        # Redirect stdout/stderr to /dev/null
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)
        return self
        
    def __exit__(self, *args):
        import os
        # Restore stdout/stderr
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        # Close the duplicated file descriptors
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._null_fd)


def _fetch_etf_data(
    start_date: str,
    end_date: Optional[str] = None
) -> Dict[str, pd.Series]:
    """Fetch key ETF data for market assessment."""
    cache_key = f"etf_{start_date}_{end_date}"
    now = datetime.now()
    
    if cache_key in _market_data_cache:
        cached_time, cached_data = _market_data_cache[cache_key]
        if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not available")
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    etf_tickers = {
        'SPY': 'SPY',      # S&P 500
        'QQQ': 'QQQ',      # Nasdaq 100
        'IWM': 'IWM',      # Russell 2000
        'IWD': 'IWD',      # Russell 1000 Value
        'IWF': 'IWF',      # Russell 1000 Growth
        'VTI': 'VTI',      # Total Market
        'VIX': '^VIX',     # Volatility Index
        # Sector ETFs (SPDR Select Sector)
        'XLK': 'XLK',      # Technology
        'XLF': 'XLF',      # Financials
        'XLV': 'XLV',      # Healthcare
        'XLY': 'XLY',      # Consumer Discretionary
        'XLP': 'XLP',      # Consumer Staples
        'XLE': 'XLE',      # Energy
        'XLI': 'XLI',      # Industrials
        'XLB': 'XLB',      # Materials
        'XLU': 'XLU',      # Utilities
        'XLRE': 'XLRE',    # Real Estate
        'XLC': 'XLC',      # Communication Services
    }
    
    result = {}
    
    for name, ticker in etf_tickers.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            series = _extract_close_series(df, ticker)
            if series is not None and len(series) > 20:
                result[name] = series
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
    
    _market_data_cache[cache_key] = (now, result)
    return result


def _fetch_stock_sample_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    cache_key_prefix: str = "stocks"
) -> Dict[str, pd.Series]:
    """Fetch price data for a sample of stocks."""
    cache_key = f"{cache_key_prefix}_{start_date}_{end_date}_{len(tickers)}"
    now = datetime.now()
    
    if cache_key in _market_data_cache:
        cached_time, cached_data = _market_data_cache[cache_key]
        if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    result = {}
    
    # Batch download for efficiency (threads=False to avoid output issues)
    try:
        with warnings.catch_warnings(), _SuppressOutput():
            warnings.simplefilter("ignore")
            df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        
        if df is not None and not df.empty:
            # Handle multi-ticker DataFrame
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        if ('Close', ticker) in df.columns:
                            series = df[('Close', ticker)].dropna()
                            if len(series) > 20:
                                result[ticker] = series
                    except Exception:
                        pass
            else:
                # Single ticker case
                if len(tickers) == 1:
                    series = _extract_close_series(df, tickers[0])
                    if series is not None and len(series) > 20:
                        result[tickers[0]] = series
    except Exception as e:
        logger.debug(f"Batch download failed: {e}")
    
    _market_data_cache[cache_key] = (now, result)
    return result


# =============================================================================
# UNIVERSE METRICS COMPUTATION
# =============================================================================

def _compute_universe_metrics(
    name: str,
    weight: float,
    prices: pd.Series,
    ticker_count: int = 1
) -> UniverseMetrics:
    """Compute all metrics for a universe segment."""
    metrics = UniverseMetrics(
        name=name,
        weight=weight,
        ticker_count=ticker_count,
    )
    
    if prices is None or len(prices) < 20:
        return metrics
    
    metrics.data_available = True
    metrics.current_level = float(prices.iloc[-1])
    
    # Returns
    returns = _compute_returns(prices)
    metrics.return_1d = returns["1d"]
    metrics.return_5d = returns["5d"]
    metrics.return_21d = returns["21d"]
    metrics.return_63d = returns["63d"]
    
    # Volatility
    try:
        ret_series = prices.pct_change().dropna()
        metrics.volatility_20d = float(ret_series.iloc[-20:].std() * np.sqrt(252))
    except Exception:
        pass
    
    metrics.volatility_percentile = _compute_volatility_percentile(prices)
    
    # Vol term structure
    ratio, inverted = _compute_vol_term_structure(prices)
    metrics.vol_term_structure_ratio = ratio
    metrics.vol_inverted = inverted
    
    # Momentum signal
    metrics.momentum_signal = _compute_momentum_signal(metrics.return_5d, metrics.return_21d)
    
    # Compute stress level
    # Stress increases with: high volatility, negative momentum, vol inversion
    vol_stress = max(0, (metrics.volatility_percentile - 0.5) * 2)  # 0 to 1
    momentum_stress = max(0, -metrics.return_21d * 5)  # Negative returns = stress
    inversion_stress = 0.5 if inverted else 0.0
    
    metrics.stress_level = min(2.0, vol_stress + momentum_stress + inversion_stress)
    metrics.stress_contribution = metrics.stress_level * weight
    
    return metrics


def _compute_breadth_from_sample(
    stock_data: Dict[str, pd.Series]
) -> MarketBreadthMetrics:
    """Compute market breadth from a sample of stocks."""
    breadth = MarketBreadthMetrics()
    
    if not stock_data:
        return breadth
    
    above_50ma = 0
    above_200ma = 0
    advancing = 0
    declining = 0
    new_highs = 0
    new_lows = 0
    total = 0
    
    for ticker, prices in stock_data.items():
        try:
            if prices is None or len(prices) < 200:
                continue
            
            total += 1
            current = float(prices.iloc[-1])
            
            # 50-day MA
            ma_50 = float(prices.iloc[-50:].mean())
            if current > ma_50:
                above_50ma += 1
            
            # 200-day MA
            ma_200 = float(prices.iloc[-200:].mean())
            if current > ma_200:
                above_200ma += 1
            
            # 1-day change
            prev = float(prices.iloc[-2])
            if current > prev:
                advancing += 1
            else:
                declining += 1
            
            # 52-week highs/lows
            high_252 = float(prices.iloc[-252:].max())
            low_252 = float(prices.iloc[-252:].min())
            
            if current >= high_252 * 0.98:  # Within 2% of high
                new_highs += 1
            if current <= low_252 * 1.02:  # Within 2% of low
                new_lows += 1
                
        except Exception:
            pass
    
    if total > 0:
        breadth.pct_above_50ma = above_50ma / total
        breadth.pct_above_200ma = above_200ma / total
        breadth.advance_decline_ratio = advancing / max(1, declining)
        breadth.new_highs = new_highs
        breadth.new_lows = new_lows
        
        # Breadth thrust detection
        if breadth.pct_above_50ma > 0.90:
            breadth.breadth_thrust = True
        
        # Warning detection
        if breadth.pct_above_50ma < BREADTH_WARNING_THRESHOLD:
            breadth.breadth_warning = True
        
        # Interpretation
        if breadth.pct_above_50ma >= 0.70:
            breadth.interpretation = "Strong - Broad participation"
        elif breadth.pct_above_50ma >= 0.50:
            breadth.interpretation = "Healthy - Normal breadth"
        elif breadth.pct_above_50ma >= BREADTH_WARNING_THRESHOLD:
            breadth.interpretation = "Narrowing - Watch for divergence"
        elif breadth.pct_above_50ma >= BREADTH_DANGER_THRESHOLD:
            breadth.interpretation = "⚠️ Weak - Distribution phase"
        else:
            breadth.interpretation = "🚨 Critical - Capitulation risk"
    
    return breadth


def _compute_correlation_stress(
    stock_data: Dict[str, pd.Series],
    lookback: int = CORRELATION_LOOKBACK
) -> CorrelationStress:
    """Compute cross-asset correlation stress."""
    stress = CorrelationStress()
    
    if len(stock_data) < 5:
        return stress
    
    try:
        # Build returns DataFrame
        returns_dict = {}
        for ticker, prices in stock_data.items():
            if prices is not None and len(prices) >= lookback:
                ret = prices.pct_change().dropna().iloc[-lookback:]
                if len(ret) >= lookback // 2:
                    returns_dict[ticker] = ret
        
        if len(returns_dict) < 5:
            return stress
        
        returns_df = pd.DataFrame(returns_dict)
        
        # Compute correlation matrix
        corr_matrix = returns_df.corr()
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corrs = corr_matrix.where(mask).stack()
        
        if len(upper_corrs) > 0:
            stress.avg_correlation = float(upper_corrs.mean())
            stress.max_correlation = float(upper_corrs.max())
            
            # Compute historical percentile (simplified - compare to threshold)
            stress.correlation_percentile = min(1.0, stress.avg_correlation / 0.80)
            
            stress.systemic_risk_elevated = stress.avg_correlation > CORRELATION_STRESS_THRESHOLD
            
            if stress.avg_correlation >= 0.80:
                stress.interpretation = "🚨 Extreme - Systemic risk"
            elif stress.avg_correlation >= 0.65:
                stress.interpretation = "⚠️ High - Correlations rising"
            elif stress.avg_correlation >= 0.50:
                stress.interpretation = "Moderate - Normal correlation"
            else:
                stress.interpretation = "Low - Healthy dispersion"
                
    except Exception:
        pass
    
    return stress


def _compute_crash_risk(
    universes: Dict[str, UniverseMetrics],
    breadth: MarketBreadthMetrics,
    correlation: CorrelationStress,
    temperature: float,
) -> Tuple[float, str, int, List[str]]:
    """Compute crash risk probability and level."""
    
    # Count vol inversions
    inverted_universes = [
        name for name, u in universes.items()
        if u.vol_inverted
    ]
    vol_inversion_count = len(inverted_universes)
    
    # Base risk from vol inversions
    base_risk_by_count = {0: 0.02, 1: 0.08, 2: 0.18, 3: 0.30, 4: 0.45}
    base_risk = base_risk_by_count.get(vol_inversion_count, 0.45)
    
    # Breadth modifier
    if breadth.pct_above_50ma < BREADTH_DANGER_THRESHOLD:
        breadth_multiplier = 1.5
    elif breadth.pct_above_50ma < BREADTH_WARNING_THRESHOLD:
        breadth_multiplier = 1.25
    else:
        breadth_multiplier = 1.0
    
    # Correlation modifier
    if correlation.systemic_risk_elevated:
        corr_multiplier = 1.4
    elif correlation.avg_correlation > 0.60:
        corr_multiplier = 1.2
    else:
        corr_multiplier = 1.0
    
    # Temperature modifier
    if temperature > 1.5:
        temp_multiplier = 1.5
    elif temperature > 1.0:
        temp_multiplier = 1.25
    else:
        temp_multiplier = 1.0
    
    crash_risk_pct = min(0.75, base_risk * breadth_multiplier * corr_multiplier * temp_multiplier)
    
    # Determine level
    if crash_risk_pct >= 0.40:
        level = "Extreme"
    elif crash_risk_pct >= 0.25:
        level = "High"
    elif crash_risk_pct >= 0.15:
        level = "Elevated"
    elif crash_risk_pct >= 0.08:
        level = "Moderate"
    else:
        level = "Low"
    
    return crash_risk_pct, level, vol_inversion_count, inverted_universes


def _determine_exit_signal(
    temperature: float,
    crash_risk_pct: float,
    breadth: MarketBreadthMetrics,
    correlation: CorrelationStress,
) -> Tuple[bool, Optional[str]]:
    """Determine if an exit signal should be issued."""
    reasons = []
    
    if temperature >= 1.5:
        reasons.append(f"Temperature extreme ({temperature:.2f})")
    
    if crash_risk_pct >= 0.35:
        reasons.append(f"Crash risk high ({crash_risk_pct:.0%})")
    
    if breadth.pct_above_50ma < BREADTH_DANGER_THRESHOLD:
        reasons.append(f"Breadth critical ({breadth.pct_above_50ma:.0%} above 50MA)")
    
    if correlation.systemic_risk_elevated:
        reasons.append(f"Systemic risk ({correlation.avg_correlation:.0%} avg corr)")
    
    if len(reasons) >= 2:
        return True, "; ".join(reasons)
    
    return False, None


def _compute_overall_momentum(universes: Dict[str, UniverseMetrics]) -> str:
    """Compute aggregate momentum signal."""
    total_weight = sum(u.weight for u in universes.values() if u.data_available)
    if total_weight == 0:
        return "→ Neutral"
    
    weighted_return = sum(
        u.return_21d * u.weight
        for u in universes.values()
        if u.data_available
    ) / total_weight
    
    return _compute_momentum_signal(weighted_return, weighted_return)


def _compute_sector_rotation(etf_data: Dict[str, pd.Series]) -> str:
    """Compute growth vs value rotation signal."""
    if 'IWF' not in etf_data or 'IWD' not in etf_data:
        return "Unknown"
    
    try:
        growth_returns = _compute_returns(etf_data['IWF'])
        value_returns = _compute_returns(etf_data['IWD'])
        
        spread_21d = growth_returns['21d'] - value_returns['21d']
        
        if spread_21d > 0.05:
            return "Growth Leading"
        elif spread_21d < -0.05:
            return "Value Leading"
        else:
            return "Balanced"
    except Exception:
        return "Unknown"


def _compute_sector_metrics(etf_data: Dict[str, pd.Series]) -> Dict[str, SectorMetrics]:
    """
    Compute metrics for each S&P 500 sector using SPDR Select Sector ETFs.
    
    Returns dict mapping sector name to SectorMetrics.
    """
    sectors = {}
    
    for ticker, sector_name in SECTOR_ETFS.items():
        if ticker not in etf_data:
            # Create placeholder with no data
            sectors[sector_name] = SectorMetrics(
                name=sector_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        prices = etf_data[ticker]
        if prices is None or len(prices) < 30:
            sectors[sector_name] = SectorMetrics(
                name=sector_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        try:
            # Compute returns
            returns = _compute_returns(prices)
            ret_1d = returns.get('1d', 0.0)
            ret_5d = returns.get('5d', 0.0)
            ret_21d = returns.get('21d', 0.0)
            
            # Compute volatility
            daily_returns = prices.pct_change().dropna()
            if len(daily_returns) >= 20:
                vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252))
            else:
                vol_20d = 0.0
            
            # Volatility percentile
            vol_pctl = _compute_volatility_percentile(prices)
            
            # Momentum signal
            momentum = _compute_momentum_signal(ret_5d, ret_21d)
            
            # Risk score (0-100)
            # Based on: vol_percentile (0-40), drawdown (0-40), vol level (0-20)
            vol_pts = vol_pctl * 40
            drawdown_pts = min(max(0, -ret_5d) / 0.10, 1.0) * 40
            vol_level_pts = min(vol_20d / 0.50, 1.0) * 20
            risk_score = int(min(100, vol_pts + drawdown_pts + vol_level_pts))
            
            sectors[sector_name] = SectorMetrics(
                name=sector_name,
                ticker=ticker,
                return_1d=ret_1d,
                return_5d=ret_5d,
                return_21d=ret_21d,
                volatility_20d=vol_20d,
                volatility_percentile=vol_pctl,
                momentum_signal=momentum,
                risk_score=risk_score,
                data_available=True,
            )
        except Exception:
            sectors[sector_name] = SectorMetrics(
                name=sector_name,
                ticker=ticker,
                data_available=False,
            )
    
    return sectors


def _fetch_currency_data(
    start_date: str,
    end_date: Optional[str] = None
) -> Dict[str, pd.Series]:
    """Fetch currency pair data for FX market assessment."""
    cache_key = f"currency_{start_date}_{end_date}"
    now = datetime.now()
    
    if cache_key in _market_data_cache:
        cached_time, cached_data = _market_data_cache[cache_key]
        if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    result = {}
    
    for ticker in CURRENCY_PAIRS.keys():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            series = _extract_close_series(df, ticker)
            if series is not None and len(series) > 20:
                result[ticker] = series
        except Exception:
            pass
    
    _market_data_cache[cache_key] = (now, result)
    return result


def _compute_currency_metrics(currency_data: Dict[str, pd.Series]) -> Dict[str, CurrencyMetrics]:
    """Compute metrics for each currency pair."""
    currencies = {}
    
    for ticker, pair_name in CURRENCY_PAIRS.items():
        if ticker not in currency_data:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        prices = currency_data[ticker]
        if prices is None or len(prices) < 30:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        try:
            rate = float(prices.iloc[-1])
            returns = _compute_returns(prices)
            ret_1d = returns.get('1d', 0.0)
            ret_5d = returns.get('5d', 0.0)
            ret_21d = returns.get('21d', 0.0)
            
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            
            momentum = _compute_momentum_signal(ret_5d, ret_21d)
            
            # Risk score: vol (0-50) + recent moves (0-50)
            vol_pts = min(vol_20d / 0.15, 1.0) * 50
            move_pts = min(abs(ret_5d) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                rate=rate,
                return_1d=ret_1d,
                return_5d=ret_5d,
                return_21d=ret_21d,
                volatility_20d=vol_20d,
                momentum_signal=momentum,
                risk_score=risk_score,
                data_available=True,
            )
        except Exception:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
    
    return currencies


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_market_temperature(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> MarketTemperatureResult:
    """
    Compute comprehensive market temperature.
    
    This is the main entry point for market risk assessment.
    
    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)
        
    Returns:
        MarketTemperatureResult with complete market assessment
    """
    # Fetch ETF data
    etf_data = _fetch_etf_data(start_date, end_date)
    
    # Fetch stock samples
    top100_data = _fetch_stock_sample_data(
        TOP_100_TICKERS[:50],  # Use first 50 for speed
        start_date, end_date,
        "top100"
    )
    
    sp500_data = _fetch_stock_sample_data(
        SP500_SAMPLE,
        start_date, end_date,
        "sp500"
    )
    
    russell_data = _fetch_stock_sample_data(
        RUSSELL_2000_SAMPLE[:20],  # Use first 20 for speed
        start_date, end_date,
        "russell"
    )
    
    # Compute universe metrics
    universes = {}
    
    # Mega-cap (Top 100) - use aggregate of sample
    if top100_data:
        # Create equal-weighted index
        sample_returns = pd.DataFrame({
            t: p.pct_change() for t, p in top100_data.items()
        }).dropna()
        if not sample_returns.empty:
            # Build cumulative index
            equal_weight_return = sample_returns.mean(axis=1)
            index = (1 + equal_weight_return).cumprod() * 100
            universes["Mega-Cap 100"] = _compute_universe_metrics(
                "Mega-Cap 100", MEGA_CAP_WEIGHT, index, len(top100_data)
            )
    
    # Broad Market (S&P 500 via SPY)
    if 'SPY' in etf_data:
        universes["S&P 500"] = _compute_universe_metrics(
            "S&P 500", BROAD_MARKET_WEIGHT, etf_data['SPY'], 500
        )
    
    # Small Cap (Russell 2000 via IWM)
    if 'IWM' in etf_data:
        universes["Russell 2000"] = _compute_universe_metrics(
            "Russell 2000", SMALL_CAP_WEIGHT, etf_data['IWM'], 2000
        )
    
    # Growth vs Value
    if 'QQQ' in etf_data:
        universes["Growth (QQQ)"] = _compute_universe_metrics(
            "Growth (QQQ)", GROWTH_VALUE_WEIGHT, etf_data['QQQ'], 100
        )
    
    # Compute breadth from all available stocks
    all_stocks = {**top100_data, **sp500_data}
    breadth = _compute_breadth_from_sample(all_stocks)
    
    # Compute correlation stress
    correlation = _compute_correlation_stress(all_stocks)
    
    # Aggregate temperature
    total_contribution = sum(u.stress_contribution for u in universes.values())
    total_weight = sum(u.weight for u in universes.values() if u.data_available)
    
    if total_weight > 0:
        temperature = total_contribution / total_weight
    else:
        temperature = 1.0  # Default to elevated if no data
    
    # Add correlation and breadth stress
    if correlation.systemic_risk_elevated:
        temperature += 0.3
    if breadth.breadth_warning:
        temperature += 0.2
    
    temperature = max(TEMP_MIN, min(TEMP_MAX, temperature))
    
    # Compute scale factor
    scale_factor = _compute_scale_factor(temperature)
    
    # Status and action
    status, action_text = _get_status_and_action(temperature)
    
    # Crash risk
    crash_risk_pct, crash_risk_level, vol_inversion_count, inverted_universes = \
        _compute_crash_risk(universes, breadth, correlation, temperature)
    
    # Exit signal
    exit_signal, exit_reason = _determine_exit_signal(
        temperature, crash_risk_pct, breadth, correlation
    )
    
    # Momentum and rotation
    overall_momentum = _compute_overall_momentum(universes)
    sector_rotation = _compute_sector_rotation(etf_data)
    
    # Compute sector-by-sector metrics
    sectors = _compute_sector_metrics(etf_data)
    
    # Compute currency pair metrics
    currency_data = _fetch_currency_data(start_date, end_date)
    currencies = _compute_currency_metrics(currency_data)
    
    # Data quality
    data_quality = sum(1 for u in universes.values() if u.data_available) / max(1, len(universes))
    
    return MarketTemperatureResult(
        temperature=temperature,
        scale_factor=scale_factor,
        universes=universes,
        breadth=breadth,
        correlation=correlation,
        computed_at=datetime.now().isoformat(),
        data_quality=data_quality,
        status=status,
        action_text=action_text,
        crash_risk_pct=crash_risk_pct,
        crash_risk_level=crash_risk_level,
        vol_inversion_count=vol_inversion_count,
        inverted_universes=inverted_universes,
        overall_momentum=overall_momentum,
        sector_rotation_signal=sector_rotation,
        exit_signal=exit_signal,
        exit_reason=exit_reason,
        sectors=sectors,
        currencies=currencies,
    )


# =============================================================================
# RENDERING
# =============================================================================

def render_market_temperature(result: MarketTemperatureResult, console=None) -> None:
    """Render market temperature with premium Apple-quality UX."""
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
    
    if console is None:
        console = Console()
    
    # Temperature-based styling
    if result.temperature >= 1.5:
        temp_style = "bold red"
        status_color = "red"
        emoji = "🚨"
    elif result.temperature >= 1.0:
        temp_style = "bold orange1"
        status_color = "orange1"
        emoji = "🔥"
    elif result.temperature >= 0.7:
        temp_style = "bold yellow"
        status_color = "yellow"
        emoji = "⚠️"
    elif result.temperature >= 0.5:
        temp_style = "yellow"
        status_color = "yellow"
        emoji = "📊"
    else:
        temp_style = "bright_green"
        status_color = "bright_green"
        emoji = "✅"
    
    # Print clear separator and header
    console.print()
    console.print("  " + "═" * 76)
    
    header = Text()
    header.append(f"  {emoji} ", style="")
    header.append("MARKET TEMPERATURE", style="bold white")
    header.append("  —  ", style="dim")
    header.append(f"{result.temperature:.2f}", style=temp_style)
    header.append(f"  {result.status.upper()}", style=f"bold {status_color}")
    console.print(header)
    
    # Action text
    if result.exit_signal:
        action_line = Text()
        action_line.append("  🚨 EXIT SIGNAL: ", style="bold red")
        action_line.append(result.exit_reason or "", style="red")
        console.print(action_line)
    else:
        action_line = Text()
        action_line.append("  → ", style="dim")
        action_line.append(result.action_text, style="dim italic")
        console.print(action_line)
    
    console.print("  " + "═" * 76)
    console.print()
    
    # Universe breakdown
    console.print("  [dim]Universe Segments[/dim]")
    console.print()
    
    header_line = Text()
    header_line.append("  ")
    header_line.append("SEGMENT".ljust(18), style="bold dim")
    header_line.append("LEVEL".rjust(10), style="bold dim")
    header_line.append("1D".rjust(8), style="bold dim")
    header_line.append("5D".rjust(8), style="bold dim")
    header_line.append("21D".rjust(8), style="bold dim")
    header_line.append("VOL".rjust(8), style="bold dim")
    header_line.append("MOMENTUM".ljust(12), style="bold dim")
    console.print(header_line)
    console.print("  " + "─" * 78)
    
    for name, univ in result.universes.items():
        line = Text()
        line.append("  ")
        
        # Name with stress indicator
        if univ.stress_level > 1.0:
            line.append(f"🔴 {name}".ljust(18), style="red")
        elif univ.stress_level > 0.5:
            line.append(f"🟡 {name}".ljust(18), style="yellow")
        else:
            line.append(f"🟢 {name}".ljust(18), style="green")
        
        # Level
        if univ.current_level:
            line.append(f"{univ.current_level:,.0f}".rjust(10), style="white")
        else:
            line.append("—".rjust(10), style="dim")
        
        # Returns
        for ret in [univ.return_1d, univ.return_5d, univ.return_21d]:
            ret_style = "bright_green" if ret >= 0 else "indian_red1"
            line.append(f"{ret:+.1%}".rjust(8), style=ret_style)
        
        # Volatility
        vol_style = "red" if univ.volatility_percentile > 0.8 else ("yellow" if univ.volatility_percentile > 0.6 else "white")
        line.append(f"{univ.volatility_20d:.0%}".rjust(8), style=vol_style)
        
        # Momentum
        if "Strong" in univ.momentum_signal and "↑" in univ.momentum_signal:
            mom_style = "bold bright_green"
        elif "Rising" in univ.momentum_signal:
            mom_style = "bright_green"
        elif "Weak" in univ.momentum_signal or "↓" in univ.momentum_signal:
            mom_style = "bold indian_red1"
        elif "Falling" in univ.momentum_signal:
            mom_style = "indian_red1"
        else:
            mom_style = "dim"
        line.append(f"  {univ.momentum_signal}", style=mom_style)
        
        console.print(line)
    
    console.print()
    
    # Market Breadth
    console.print("  [dim]Market Breadth[/dim]")
    console.print()
    
    breadth_line = Text()
    breadth_line.append("  ")
    breadth_line.append("Above 50MA   ", style="dim")
    
    pct_50ma = result.breadth.pct_above_50ma
    if pct_50ma < BREADTH_DANGER_THRESHOLD:
        b_style = "bold red"
    elif pct_50ma < BREADTH_WARNING_THRESHOLD:
        b_style = "yellow"
    else:
        b_style = "green"
    breadth_line.append(f"{pct_50ma:.0%}", style=b_style)
    
    breadth_line.append("   Above 200MA   ", style="dim")
    breadth_line.append(f"{result.breadth.pct_above_200ma:.0%}", style="white")
    
    breadth_line.append("   A/D Ratio   ", style="dim")
    ad_style = "green" if result.breadth.advance_decline_ratio > 1 else "red"
    breadth_line.append(f"{result.breadth.advance_decline_ratio:.2f}", style=ad_style)
    console.print(breadth_line)
    
    interp_line = Text()
    interp_line.append("  ")
    interp_line.append(result.breadth.interpretation, style="dim italic")
    console.print(interp_line)
    
    console.print()
    
    # Correlation Stress
    console.print("  [dim]Correlation Stress[/dim]")
    console.print()
    
    corr_line = Text()
    corr_line.append("  ")
    corr_line.append("Avg Correlation   ", style="dim")
    
    if result.correlation.systemic_risk_elevated:
        c_style = "bold red"
    elif result.correlation.avg_correlation > 0.60:
        c_style = "yellow"
    else:
        c_style = "green"
    corr_line.append(f"{result.correlation.avg_correlation:.0%}", style=c_style)
    
    corr_line.append("   ", style="")
    corr_line.append(result.correlation.interpretation, style="dim italic")
    console.print(corr_line)
    
    console.print()
    
    # Crash Risk Panel
    if result.crash_risk_pct > 0.05:
        console.print("  [dim]Crash Risk Assessment[/dim]")
        console.print()
        
        # Risk gauge
        gauge_width = 40
        filled = int(min(1.0, result.crash_risk_pct / 0.50) * gauge_width)
        
        gauge_line = Text()
        gauge_line.append("  ")
        for i in range(gauge_width):
            if i < filled:
                if i < gauge_width * 0.4:
                    gauge_line.append("━", style="green")
                elif i < gauge_width * 0.7:
                    gauge_line.append("━", style="yellow")
                else:
                    gauge_line.append("━", style="red")
            else:
                gauge_line.append("─", style="bright_black")
        console.print(gauge_line)
        
        risk_line = Text()
        risk_line.append("  ")
        
        if result.crash_risk_level == "Extreme":
            r_style = "bold red"
        elif result.crash_risk_level == "High":
            r_style = "red"
        elif result.crash_risk_level == "Elevated":
            r_style = "yellow"
        else:
            r_style = "green"
        
        risk_line.append(f"{result.crash_risk_pct:.0%}", style=r_style)
        risk_line.append(f"  {result.crash_risk_level.upper()}", style=r_style)
        
        if result.vol_inversion_count > 0:
            risk_line.append(f"  ({result.vol_inversion_count} vol inversions)", style="dim")
        console.print(risk_line)
        
        console.print()
    
    # Summary metrics
    console.print("  [dim]Summary[/dim]")
    console.print()
    
    summary_line = Text()
    summary_line.append("  ")
    summary_line.append("Position Scale   ", style="dim")
    
    scale = result.scale_factor
    if scale > 0.9:
        s_style = "green"
    elif scale > 0.6:
        s_style = "yellow"
    else:
        s_style = "red"
    summary_line.append(f"{scale:.0%}", style=s_style)
    
    summary_line.append("   Momentum   ", style="dim")
    summary_line.append(result.overall_momentum, style="white")
    
    summary_line.append("   Rotation   ", style="dim")
    summary_line.append(result.sector_rotation_signal, style="white")
    console.print(summary_line)
    
    qual_line = Text()
    qual_line.append("  ")
    qual_line.append("Data Quality   ", style="dim")
    qual_line.append(f"{result.data_quality:.0%}", style="green" if result.data_quality > 0.8 else "yellow")
    qual_line.append(f"   Computed   ", style="dim")
    qual_line.append(result.computed_at[:19], style="dim italic")
    console.print(qual_line)
    
    console.print()
    console.print()


# =============================================================================
# STANDALONE CLI
# =============================================================================

if __name__ == "__main__":
    """Run market temperature computation and display."""
    import argparse
    import os
    import sys
    import time
    from rich.console import Console
    
    # Suppress yfinance output at environment level
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['YF_LOG_LEVEL'] = 'CRITICAL'
    
    # Suppress yfinance warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)
    
    parser = argparse.ArgumentParser(description="Market Temperature — US Equity Market Assessment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--start", default="2020-01-01", help="Start date for historical data")
    args = parser.parse_args()
    
    # Compute market temperature with full output suppression
    with _SuppressOutput():
        result = compute_market_temperature(start_date=args.start)
    
    # Wait for any lingering threads to complete their output
    time.sleep(1.0)
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Clear the line with carriage return and spaces
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()
    
    console = Console()
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        render_market_temperature(result, console=console)
