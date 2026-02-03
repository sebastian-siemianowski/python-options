#!/usr/bin/env python3
"""Test sector ETF data extraction."""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'src')

import yfinance as yf
import pandas as pd
from decision.market_temperature import _extract_close_series, _compute_returns

# Test individual downloads
tickers = ['XLK', 'XLV', 'XLP', 'XLE', 'XLI']
for ticker in tickers:
    df = yf.download(ticker, start='2024-01-01', progress=False, auto_adjust=True)
    series = _extract_close_series(df, ticker)
    if series is not None:
        returns = _compute_returns(series)
        print(f"{ticker}: 1d={returns['1d']:.2%}, 5d={returns['5d']:.2%}, 21d={returns['21d']:.2%}")
    else:
        print(f"{ticker}: NO DATA EXTRACTED")
