"""
===============================================================================
BACKTEST CONFIGURATION — Structural Backtest Arena Settings
===============================================================================

Defines the standardized multi-sector universe for structural backtesting:
    - ~50 equities spanning market caps and sectors
    - Consistent time coverage requirements
    - Strict column requirements (no derived features)
    
This configuration enforces the NON-OPTIMIZATION CONSTITUTION:
    - Behavioral validation only
    - No performance-based promotion
    - One-way flow: Tuning → Backtest → Integration Trial

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from pathlib import Path


# =============================================================================
# SECTOR AND CAP CATEGORY DEFINITIONS
# =============================================================================

class Sector(Enum):
    """Market sector classification."""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    DEFENCE = "defence"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    MATERIALS = "materials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"
    INDEX = "index"


class MarketCap(Enum):
    """Market capitalization classification."""
    SMALL_CAP = "small_cap"      # < $2B
    MID_CAP = "mid_cap"          # $2B - $10B
    LARGE_CAP = "large_cap"      # > $10B
    MEGA_CAP = "mega_cap"        # > $200B


# =============================================================================
# CANONICAL 50-TICKER BACKTEST UNIVERSE
# =============================================================================
# Selected for:
#   - Liquidity (tradeable in real portfolios)
#   - Sector representation (no single-sector bias)
#   - Market cap diversity (regime behavior varies by size)
#   - Historical coverage (5+ years required)
# =============================================================================

BACKTEST_UNIVERSE: Dict[str, Dict[str, any]] = {
    # ─────────────────────────────────────────────────────────────────────────
    # TECHNOLOGY (8 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "AAPL":  {"sector": Sector.TECHNOLOGY, "cap": MarketCap.MEGA_CAP, "name": "Apple Inc"},
    "MSFT":  {"sector": Sector.TECHNOLOGY, "cap": MarketCap.MEGA_CAP, "name": "Microsoft"},
    "NVDA":  {"sector": Sector.TECHNOLOGY, "cap": MarketCap.MEGA_CAP, "name": "NVIDIA"},
    "GOOGL": {"sector": Sector.TECHNOLOGY, "cap": MarketCap.MEGA_CAP, "name": "Alphabet"},
    "CRM":   {"sector": Sector.TECHNOLOGY, "cap": MarketCap.LARGE_CAP, "name": "Salesforce"},
    "ADBE":  {"sector": Sector.TECHNOLOGY, "cap": MarketCap.LARGE_CAP, "name": "Adobe"},
    "CRWD":  {"sector": Sector.TECHNOLOGY, "cap": MarketCap.MID_CAP, "name": "CrowdStrike"},
    "NET":   {"sector": Sector.TECHNOLOGY, "cap": MarketCap.MID_CAP, "name": "Cloudflare"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINANCE (6 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "JPM":   {"sector": Sector.FINANCE, "cap": MarketCap.MEGA_CAP, "name": "JPMorgan Chase"},
    "BAC":   {"sector": Sector.FINANCE, "cap": MarketCap.MEGA_CAP, "name": "Bank of America"},
    "GS":    {"sector": Sector.FINANCE, "cap": MarketCap.LARGE_CAP, "name": "Goldman Sachs"},
    "MS":    {"sector": Sector.FINANCE, "cap": MarketCap.LARGE_CAP, "name": "Morgan Stanley"},
    "SCHW":  {"sector": Sector.FINANCE, "cap": MarketCap.LARGE_CAP, "name": "Charles Schwab"},
    "AFRM":  {"sector": Sector.FINANCE, "cap": MarketCap.SMALL_CAP, "name": "Affirm"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEFENCE (4 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "LMT":   {"sector": Sector.DEFENCE, "cap": MarketCap.LARGE_CAP, "name": "Lockheed Martin"},
    "RTX":   {"sector": Sector.DEFENCE, "cap": MarketCap.LARGE_CAP, "name": "Raytheon"},
    "NOC":   {"sector": Sector.DEFENCE, "cap": MarketCap.LARGE_CAP, "name": "Northrop Grumman"},
    "GD":    {"sector": Sector.DEFENCE, "cap": MarketCap.LARGE_CAP, "name": "General Dynamics"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # HEALTHCARE (5 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "JNJ":   {"sector": Sector.HEALTHCARE, "cap": MarketCap.MEGA_CAP, "name": "Johnson & Johnson"},
    "UNH":   {"sector": Sector.HEALTHCARE, "cap": MarketCap.MEGA_CAP, "name": "UnitedHealth"},
    "PFE":   {"sector": Sector.HEALTHCARE, "cap": MarketCap.LARGE_CAP, "name": "Pfizer"},
    "ABBV":  {"sector": Sector.HEALTHCARE, "cap": MarketCap.LARGE_CAP, "name": "AbbVie"},
    "MRNA":  {"sector": Sector.HEALTHCARE, "cap": MarketCap.MID_CAP, "name": "Moderna"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # INDUSTRIALS (5 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "CAT":   {"sector": Sector.INDUSTRIALS, "cap": MarketCap.LARGE_CAP, "name": "Caterpillar"},
    "DE":    {"sector": Sector.INDUSTRIALS, "cap": MarketCap.LARGE_CAP, "name": "Deere"},
    "BA":    {"sector": Sector.INDUSTRIALS, "cap": MarketCap.LARGE_CAP, "name": "Boeing"},
    "UPS":   {"sector": Sector.INDUSTRIALS, "cap": MarketCap.LARGE_CAP, "name": "UPS"},
    "GE":    {"sector": Sector.INDUSTRIALS, "cap": MarketCap.LARGE_CAP, "name": "GE Aerospace"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENERGY (5 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "XOM":   {"sector": Sector.ENERGY, "cap": MarketCap.MEGA_CAP, "name": "Exxon Mobil"},
    "CVX":   {"sector": Sector.ENERGY, "cap": MarketCap.MEGA_CAP, "name": "Chevron"},
    "COP":   {"sector": Sector.ENERGY, "cap": MarketCap.LARGE_CAP, "name": "ConocoPhillips"},
    "SLB":   {"sector": Sector.ENERGY, "cap": MarketCap.LARGE_CAP, "name": "Schlumberger"},
    "OXY":   {"sector": Sector.ENERGY, "cap": MarketCap.MID_CAP, "name": "Occidental"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # MATERIALS (4 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "LIN":   {"sector": Sector.MATERIALS, "cap": MarketCap.LARGE_CAP, "name": "Linde"},
    "FCX":   {"sector": Sector.MATERIALS, "cap": MarketCap.LARGE_CAP, "name": "Freeport-McMoRan"},
    "NEM":   {"sector": Sector.MATERIALS, "cap": MarketCap.MID_CAP, "name": "Newmont"},
    "NUE":   {"sector": Sector.MATERIALS, "cap": MarketCap.MID_CAP, "name": "Nucor"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONSUMER DISCRETIONARY (5 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "AMZN":  {"sector": Sector.CONSUMER_DISCRETIONARY, "cap": MarketCap.MEGA_CAP, "name": "Amazon"},
    "TSLA":  {"sector": Sector.CONSUMER_DISCRETIONARY, "cap": MarketCap.MEGA_CAP, "name": "Tesla"},
    "HD":    {"sector": Sector.CONSUMER_DISCRETIONARY, "cap": MarketCap.MEGA_CAP, "name": "Home Depot"},
    "NKE":   {"sector": Sector.CONSUMER_DISCRETIONARY, "cap": MarketCap.LARGE_CAP, "name": "Nike"},
    "SBUX":  {"sector": Sector.CONSUMER_DISCRETIONARY, "cap": MarketCap.LARGE_CAP, "name": "Starbucks"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONSUMER STAPLES (4 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "PG":    {"sector": Sector.CONSUMER_STAPLES, "cap": MarketCap.MEGA_CAP, "name": "Procter & Gamble"},
    "KO":    {"sector": Sector.CONSUMER_STAPLES, "cap": MarketCap.MEGA_CAP, "name": "Coca-Cola"},
    "PEP":   {"sector": Sector.CONSUMER_STAPLES, "cap": MarketCap.MEGA_CAP, "name": "PepsiCo"},
    "COST":  {"sector": Sector.CONSUMER_STAPLES, "cap": MarketCap.MEGA_CAP, "name": "Costco"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMMUNICATION (4 stocks)
    # ─────────────────────────────────────────────────────────────────────────
    "META":  {"sector": Sector.COMMUNICATION, "cap": MarketCap.MEGA_CAP, "name": "Meta"},
    "NFLX":  {"sector": Sector.COMMUNICATION, "cap": MarketCap.LARGE_CAP, "name": "Netflix"},
    "DIS":   {"sector": Sector.COMMUNICATION, "cap": MarketCap.LARGE_CAP, "name": "Disney"},
    "SNAP":  {"sector": Sector.COMMUNICATION, "cap": MarketCap.SMALL_CAP, "name": "Snap"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # INDEX ETFs (3 ETFs) - Market benchmarks for regime analysis
    # ─────────────────────────────────────────────────────────────────────────
    "SPY":   {"sector": Sector.INDEX, "cap": MarketCap.MEGA_CAP, "name": "S&P 500 ETF"},
    "QQQ":   {"sector": Sector.INDEX, "cap": MarketCap.MEGA_CAP, "name": "Nasdaq 100 ETF"},
    "IWM":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Russell 2000 ETF"},
    # SECTOR ETFs (11 ETFs)
    "XLK":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Technology Select Sector SPDR"},
    "XLF":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Financial Select Sector SPDR"},
    "XLV":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Health Care Select Sector SPDR"},
    "XLI":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Industrial Select Sector SPDR"},
    "XLE":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Energy Select Sector SPDR"},
    "XLB":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Materials Select Sector SPDR"},
    "XLY":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Consumer Disc Select Sector SPDR"},
    "XLP":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Consumer Staples Select Sector SPDR"},
    "XLC":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Communication Services Select Sector SPDR"},
    "XLU":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Utilities Select Sector SPDR"},
    "XLRE":  {"sector": Sector.INDEX, "cap": MarketCap.MID_CAP, "name": "Real Estate Select Sector SPDR"},
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTOR ETFs (11 ETFs) - SPDR Select Sector ETFs for sector regime analysis
    # ─────────────────────────────────────────────────────────────────────────
    "XLK":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Technology Select Sector SPDR"},
    "XLF":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Financial Select Sector SPDR"},
    "XLV":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Health Care Select Sector SPDR"},
    "XLI":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Industrial Select Sector SPDR"},
    "XLE":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Energy Select Sector SPDR"},
    "XLB":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Materials Select Sector SPDR"},
    "XLY":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Consumer Discretionary Select Sector SPDR"},
    "XLP":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Consumer Staples Select Sector SPDR"},
    "XLC":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Communication Services Select Sector SPDR"},
    "XLU":   {"sector": Sector.INDEX, "cap": MarketCap.LARGE_CAP, "name": "Utilities Select Sector SPDR"},
    "XLRE":  {"sector": Sector.INDEX, "cap": MarketCap.MID_CAP, "name": "Real Estate Select Sector SPDR"},
}

# List of all tickers
BACKTEST_TICKERS = list(BACKTEST_UNIVERSE.keys())


def get_tickers_by_sector(sector: Sector) -> List[str]:
    """Get all tickers for a specific sector."""
    return [t for t, info in BACKTEST_UNIVERSE.items() if info["sector"] == sector]


def get_tickers_by_cap(cap: MarketCap) -> List[str]:
    """Get all tickers for a specific market cap category."""
    return [t for t, info in BACKTEST_UNIVERSE.items() if info["cap"] == cap]


# =============================================================================
# REQUIRED DATA COLUMNS (STRICT)
# =============================================================================
# These columns are REQUIRED in all backtest data.
# No derived features allowed at data layer.
# =============================================================================

REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Ticker",
]


# =============================================================================
# DECISION OUTCOMES
# =============================================================================

class DecisionOutcome(Enum):
    """
    Possible outcomes from Structural Backtest Arena.
    
    Decisions are based on BEHAVIORAL SAFETY, informed by diagnostics —
    NEVER by raw performance.
    """
    APPROVED = "APPROVED"           # Passed all behavioral gates
    RESTRICTED = "RESTRICTED"       # Passed with caveats (sector/regime limits)
    QUARANTINED = "QUARANTINED"     # Needs observation period before retry
    REJECTED = "REJECTED"           # Failed behavioral gates - no promotion


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Configuration for Structural Backtest Arena.
    
    NON-OPTIMIZATION CONSTITUTION:
        - Financial metrics are OBSERVATIONAL ONLY
        - No backtest result may influence tuning logic upstream
        - Decisions based on behavioral safety, not raw performance
    
    Attributes:
        tickers: List of tickers to backtest
        lookback_years: Years of historical data
        min_observations: Minimum data points required
        initial_capital: Starting capital for simulation ($)
        transaction_cost_bps: Transaction cost in basis points
        
        # Diagnostic thresholds (for WARNINGS, not optimization)
        max_drawdown_warning: Trigger warning if DD exceeds this
        max_leverage_warning: Trigger warning if leverage exceeds this
        volatility_warning_mult: Warn if vol > mult * benchmark vol
        
        # Paths
        data_dir: Directory for backtest data
        params_dir: Directory for tuned parameters
        results_dir: Directory for results
    """
    tickers: List[str] = field(default_factory=lambda: BACKTEST_TICKERS.copy())
    lookback_years: int = 5
    min_observations: int = 252 * 3  # 3 years minimum
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 2.0  # 2 bps (reduced for behavioral testing)
    
    # Diagnostic thresholds (WARNING triggers, not hard gates)
    max_drawdown_warning: float = 0.30       # 30% drawdown triggers warning
    max_leverage_warning: float = 2.0        # 2x leverage triggers warning
    volatility_warning_mult: float = 2.0     # 2x benchmark vol triggers warning
    tail_cluster_warning: int = 3            # Warn if 3+ tail losses cluster
    
    # Behavioral safety thresholds
    min_sharpe_for_approval: float = -0.5    # Sharpe floor (can be negative!)
    max_drawdown_for_approval: float = 0.50  # 50% DD = automatic rejection
    max_drawdown_duration_days: int = 365    # 1 year DD duration = concern
    
    # Paths (relative to repo root)
    data_dir: str = "src/arena/data/backtest_data"
    params_dir: str = "src/arena/backtest_tuned_params"
    results_dir: str = "src/arena/data/backtest_results"
    
    # Parallel processing
    parallel_workers: int = 4
    
    # Output control
    verbose: bool = True
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.tickers:
            raise ValueError("No tickers configured for backtest")
        if self.lookback_years < 1:
            raise ValueError("lookback_years must be >= 1")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        return True
    
    def get_tickers_by_sector(self, sector: Sector) -> List[str]:
        """Get configured tickers for a specific sector."""
        return [t for t in self.tickers if BACKTEST_UNIVERSE.get(t, {}).get("sector") == sector]
    
    def get_tickers_by_cap(self, cap: MarketCap) -> List[str]:
        """Get configured tickers for a specific market cap."""
        return [t for t in self.tickers if BACKTEST_UNIVERSE.get(t, {}).get("cap") == cap]


# Default configuration
DEFAULT_BACKTEST_CONFIG = BacktestConfig()


# =============================================================================
# CRISIS PERIODS FOR REGIME-CONDITIONED ANALYSIS
# =============================================================================
# Used for behavioral testing - how does the model act during stress?
# =============================================================================

CRISIS_PERIODS = {
    "covid_crash": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "COVID-19 market crash",
    },
    "covid_recovery": {
        "start": "2020-03-24",
        "end": "2020-08-31",
        "description": "Post-COVID recovery",
    },
    "inflation_shock_2022": {
        "start": "2022-01-01",
        "end": "2022-10-12",
        "description": "2022 inflation/rate shock bear market",
    },
    "banking_crisis_2023": {
        "start": "2023-03-08",
        "end": "2023-03-31",
        "description": "SVB/regional banking crisis",
    },
    "ai_rally_2023": {
        "start": "2023-01-01",
        "end": "2023-12-31",
        "description": "AI-driven tech rally",
    },
}


__all__ = [
    "Sector",
    "MarketCap", 
    "DecisionOutcome",
    "BACKTEST_UNIVERSE",
    "BACKTEST_TICKERS",
    "REQUIRED_COLUMNS",
    "BacktestConfig",
    "DEFAULT_BACKTEST_CONFIG",
    "CRISIS_PERIODS",
    "get_tickers_by_sector",
    "get_tickers_by_cap",
]
