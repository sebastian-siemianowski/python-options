"""
Story 8.1: Validation Universe & Benchmark Definition.

Formal definition of 12 validation symbols with expected characteristics
and quantitative profitability targets.
"""

VALIDATION_UNIVERSE = {
    # Large Cap (liquid, well-modeled)
    "AAPL": {"sector": "Technology", "expected_vol": 0.25, "class": "equity"},
    "NVDA": {"sector": "Technology", "expected_vol": 0.40, "class": "equity"},
    "MSFT": {"sector": "Technology", "expected_vol": 0.22, "class": "equity"},
    "GOOGL": {"sector": "Technology", "expected_vol": 0.28, "class": "equity"},
    # High Vol (challenging)
    "TSLA": {"sector": "Auto", "expected_vol": 0.55, "class": "equity"},
    "CRWD": {"sector": "Cybersecurity", "expected_vol": 0.45, "class": "equity"},
    "DKNG": {"sector": "Gaming", "expected_vol": 0.50, "class": "equity"},
    "COIN": {"sector": "Crypto", "expected_vol": 0.70, "class": "equity"},
    # Index/ETF (baseline)
    "SPY": {"sector": "Index", "expected_vol": 0.16, "class": "etf"},
    "QQQ": {"sector": "Index", "expected_vol": 0.22, "class": "etf"},
    "GLD": {"sector": "Metals", "expected_vol": 0.15, "class": "metal"},
    "TLT": {"sector": "Bonds", "expected_vol": 0.18, "class": "etf"},
}

PROFITABILITY_TARGETS = {
    "hit_rate_7d": 0.55,         # >55% directional accuracy at 1 week
    "hit_rate_21d": 0.53,        # >53% directional accuracy at 1 month
    "signal_rate": 0.15,         # >15% non-HOLD signals
    "sharpe_7d": 0.50,           # Sharpe > 0.5 at 7d horizon
    "crps_improvement": 0.20,    # >20% CRPS reduction vs baseline
    "ece_max": 0.03,             # Expected Calibration Error < 3%
    "max_drawdown_single": 0.30, # No single asset > 30% cumulative drawdown
}

# Measurement methodology for each metric:
METRIC_METHODOLOGY = {
    "hit_rate_7d": "Walk-forward: fraction of 7d forecasts where sign(forecast) == sign(realized)",
    "hit_rate_21d": "Walk-forward: fraction of 21d forecasts where sign(forecast) == sign(realized)",
    "signal_rate": "Fraction of (asset, horizon) pairs where label != HOLD",
    "sharpe_7d": "Annualized Sharpe of signal-following strategy at 7d horizon",
    "crps_improvement": "% reduction in CRPS vs naive climatology baseline",
    "ece_max": "Maximum ECE across all horizons (calibration buckets = 10)",
    "max_drawdown_single": "Maximum cumulative log-return drawdown for any single asset",
}

# Validation horizons
VALIDATION_HORIZONS = [1, 3, 7, 21, 63, 126, 252]

# Symbols by class for stratified testing
EQUITY_SYMBOLS = [s for s, m in VALIDATION_UNIVERSE.items() if m["class"] == "equity"]
ETF_SYMBOLS = [s for s, m in VALIDATION_UNIVERSE.items() if m["class"] == "etf"]
METAL_SYMBOLS = [s for s, m in VALIDATION_UNIVERSE.items() if m["class"] == "metal"]
HIGH_VOL_SYMBOLS = [s for s, m in VALIDATION_UNIVERSE.items() if m["expected_vol"] >= 0.40]
LOW_VOL_SYMBOLS = [s for s, m in VALIDATION_UNIVERSE.items() if m["expected_vol"] < 0.25]
