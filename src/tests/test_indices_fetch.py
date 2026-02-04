#!/usr/bin/env python3
"""Test fetching market indices from Yahoo Finance."""
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd

symbols = ['^SPX', '^VIX', '^RUT', '^NDX', '^DJX', '^GSPC', '^IXIC']

for sym in symbols:
    try:
        df = yf.download(sym, period='5d', progress=False, auto_adjust=True)
        if df is not None and len(df) > 0:
            # Handle both Series and DataFrame
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            last = float(close.iloc[-1])
            print(f'{sym}: {len(df)} rows, last close: {last:.2f}')
        else:
            print(f'{sym}: NO DATA')
    except Exception as e:
        print(f'{sym}: ERROR - {e}')
