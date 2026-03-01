"""
Adaptive Data Quality Filter (February 2026).

Detects and truncates phantom/synthetic data from asset price histories.
Designed for assets with structural data quality issues:

  - GPUS: Ultra-illiquid OTC micro-cap with ~780 rows of Volume=0 phantom
    quotes before genuine trading begins. Price collapsed $57M -> $0.18
    across multiple reverse splits.

  - HWM: Howmet Aerospace spun off from Arconic on April 1, 2020.
    Pre-spinoff data is synthetic Arconic pricing (different company).

  - ABTC: Crypto-adjacent stock spanning 4 orders of magnitude ($19,240->$1).
    First rows have Volume=1-21 (phantom quotes), +235% single-day spike.

  - General: Any asset with leading zero-volume blocks or sparse-volume
    periods that violate the continuous-distribution assumption.

Three-layer approach:

  Layer A — Leading Low-Volume Purge:
    Scan forward from row 0, find the first row with Volume >= 100,
    drop all preceding rows.  (Catches phantom pre-IPO/pre-spinoff data
    AND ultra-low-volume OTC quotes with Volume=1-50.)
    Threshold: MIN_GENUINE_VOLUME = 100 shares (floor of genuine price
    discovery for US equities).

  Layer B — Sparse Volume Adaptive Window:
    After purge, if the remaining data still has > 15% zero-volume days,
    apply a maximum history window (1260 trading days = 5 years) by
    keeping only the most recent rows.  (Catches GPUS-style data where
    2019 is still 64% Volume=0 even after leading-block removal.)

  Layer C — Price-Range Windowing:
    After Layer B, compute log(max(Close)/min(Close)).  If this exceeds
    PRICE_RANGE_LOG_THRESHOLD (≈55× range), keep only the last 1260 rows.
    (Catches ABTC-style multi-order-of-magnitude declines where even
    genuine-volume data spans an implausible price range for Kalman
    filtering.)

  Minimum data gate:
    Require >= 252 rows (1 year) after truncation.  If fewer, keep all
    available data and log a warning.

Reference:
  - Lo & MacKinlay (1990), "When Are Contrarian Profits Due to Stock
    Market Overreaction?" — Non-trading effects in illiquid assets.
  - Bai & Perron (1998), "Estimating and Testing Linear Models with
    Multiple Structural Changes" — Structural break detection.
  - Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask
    Spread" — Phantom-volume microstructure artifacts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

# ── Configuration ───────────────────────────────────────────────────
MIN_GENUINE_VOLUME = 100       # Floor of genuine price discovery
SPARSE_VOL_THRESHOLD = 0.15   # If >15% zero-vol after purge, apply window
MAX_QUALITY_WINDOW = 1260     # 5 years of trading days
PRICE_RANGE_LOG_THRESHOLD = 4.0  # log(55×) — implausible for Kalman filter
MIN_DATA_AFTER_TRUNCATION = 252  # 1 year minimum
# ────────────────────────────────────────────────────────────────────


def adaptive_data_quality(
    df: pd.DataFrame,
    asset: str = "",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and truncate phantom/synthetic data from price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw price DataFrame with columns including 'Volume' (case-insensitive).
    asset : str
        Asset symbol for logging.
    verbose : bool
        If True, print diagnostic messages.

    Returns
    -------
    (df_clean, report) where:
        df_clean : pd.DataFrame — truncated DataFrame
        report : dict — quality metrics:
            rows_original : int
            rows_purged_leading : int — rows dropped by Layer A
            zero_vol_frac_after_purge : float — Volume=0 fraction after Layer A
            window_applied : bool — whether Layer B window was applied
            rows_final : int
    """
    report: Dict[str, Any] = {
        "rows_original": len(df),
        "rows_purged_leading": 0,
        "zero_vol_frac_after_purge": 0.0,
        "window_applied": False,
        "price_range_log": 0.0,
        "price_range_window_applied": False,
        "rows_final": len(df),
    }

    if df is None or df.empty:
        return df, report

    # ── Resolve Volume column (case-insensitive) ────────────────────
    cols_map = {c.lower(): c for c in df.columns}
    vol_col = cols_map.get("volume")
    if vol_col is None:
        # No volume column — cannot apply quality filter
        return df, report

    volume = df[vol_col].values

    # ── FX pair detection ───────────────────────────────────────────
    # Yahoo Finance reports Volume=0 for OTC forex pairs (=X suffix)
    # and calculated indices (^ prefix, e.g. ^VIX).
    # Volume-based filters are meaningless for these — skip all volume layers.
    _skip_vol = (asset.endswith('=X') or asset.startswith('^')) if asset else False

    # ── Layer A: Leading Low-Volume Purge ───────────────────────────
    # Find first row where Volume >= MIN_GENUINE_VOLUME (100 shares)
    # This catches phantom OTC quotes with Volume=1-50 that bypass
    # the old Volume>0 check (e.g., ABTC first rows Volume=1,17,21)
    # Skipped for FX pairs and indices where Volume is always 0.
    first_genuine = 0
    if not _skip_vol:
        for i in range(len(volume)):
            v = volume[i]
            if v is not None and not np.isnan(v) and v >= MIN_GENUINE_VOLUME:
                first_genuine = i
                break
        else:
            # ALL rows below minimum volume — cannot truncate, return as-is
            if verbose and asset:
                print(f"     ⚠️  {asset}: ALL {len(df)} rows have Volume<{MIN_GENUINE_VOLUME} — no quality truncation possible")
            return df, report

    if first_genuine > 0:
        report["rows_purged_leading"] = first_genuine
        df = df.iloc[first_genuine:].copy()
        volume = df[vol_col].values
        if verbose and asset:
            print(f"     🔬  {asset}: Purged {first_genuine} leading low-volume rows (Layer A, Volume<{MIN_GENUINE_VOLUME})")

    # ── Layer B: Sparse Volume Adaptive Window ──────────────────────
    n_after_purge = len(df)
    if n_after_purge > 0:
        n_zero_vol = int(np.sum((volume == 0) | np.isnan(volume)))
        zero_vol_frac = n_zero_vol / n_after_purge
        report["zero_vol_frac_after_purge"] = zero_vol_frac

        if zero_vol_frac > SPARSE_VOL_THRESHOLD and n_after_purge > MAX_QUALITY_WINDOW:
            df = df.iloc[-MAX_QUALITY_WINDOW:].copy()
            report["window_applied"] = True
            if verbose and asset:
                print(
                    f"     🔬  {asset}: Applied {MAX_QUALITY_WINDOW}-day quality window "
                    f"(sparse vol: {100*zero_vol_frac:.1f}% > {100*SPARSE_VOL_THRESHOLD:.0f}% threshold, Layer B)"
                )

    # ── Layer C: Price-Range Windowing ──────────────────────────────
    # If the remaining data spans an implausible price range (>55×),
    # the Kalman filter can't model such non-stationarity.  Keep only
    # the most recent MAX_QUALITY_WINDOW rows.
    close_col = cols_map.get("close")
    if close_col is not None and len(df) > MAX_QUALITY_WINDOW:
        close_vals = df[close_col].values
        close_finite = close_vals[np.isfinite(close_vals) & (close_vals > 0)]
        if len(close_finite) > 10:
            price_range_log = float(np.log(np.max(close_finite) / np.min(close_finite)))
            report["price_range_log"] = price_range_log
            if price_range_log > PRICE_RANGE_LOG_THRESHOLD:
                df = df.iloc[-MAX_QUALITY_WINDOW:].copy()
                report["price_range_window_applied"] = True
                if verbose and asset:
                    print(
                        f"     🔬  {asset}: Applied {MAX_QUALITY_WINDOW}-day quality window "
                        f"(price range: {price_range_log:.1f} > {PRICE_RANGE_LOG_THRESHOLD:.1f} log-threshold, "
                        f"≈{np.exp(price_range_log):.0f}× range, Layer C)"
                    )

    # ── Minimum data gate ───────────────────────────────────────────
    if len(df) < MIN_DATA_AFTER_TRUNCATION:
        if verbose and asset:
            print(
                f"     ⚠️  {asset}: Only {len(df)} rows after quality filter "
                f"(< {MIN_DATA_AFTER_TRUNCATION} minimum) — keeping all available"
            )

    report["rows_final"] = len(df)
    return df, report
