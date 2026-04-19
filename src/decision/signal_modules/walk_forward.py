from __future__ import annotations

"""Walk-forward validation and backtesting infrastructure.

Extracted from signals.py - Story 7.3.
Contains: walk_forward_validation, WalkForwardRecord, WalkForwardResult,
          run_walk_forward_backtest, walkforward_result_to_dataframe,
          save_walkforward_csv, _WFFeatureCache, run_walk_forward_parallel.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys as _sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from decision.signal_modules.config import *  # noqa: F403
from decision.signal_modules.volatility_imports import *  # noqa: F403
from decision.signal_modules.regime_classification import *  # noqa: F403
from decision.signal_modules.data_fetching import *  # noqa: F403
from decision.signal_modules.feature_pipeline import *  # noqa: F403
from decision.signal_modules.parameter_loading import _load_tuned_kalman_params  # noqa: F401
from ingestion.data_utils import _ensure_float_series, _download_prices, safe_last  # noqa: F401

def walk_forward_validation(px: pd.Series, train_days: int = 504, test_days: int = 21, horizons: List[int] = [1, 21, 63]) -> Dict[str, pd.DataFrame]:
    """
    Perform walk-forward out-of-sample testing to validate predictive power.

    Splits data into non-overlapping train/test windows, fits model on train,
    predicts on test, and tracks hit rates and prediction errors.

    Args:
        px: Price series
        train_days: Training window size (days)
        test_days: Test window size (days)
        horizons: Forecast horizons to test

    Returns:
        Dictionary with out-of-sample performance metrics
    """
    px_clean = _ensure_float_series(px).dropna()
    if len(px_clean) < train_days + test_days + max(horizons):
        return {}

    log_px = np.log(px_clean)
    dates = px_clean.index

    # Define walk-forward windows
    windows = []
    start_idx = 0
    while start_idx + train_days + test_days <= len(dates):
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(dates))

        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        if len(test_dates) > 0:
            windows.append({
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            })

        # Move forward by test_days (non-overlapping)
        start_idx = test_end_idx

    if not windows:
        return {}

    # Track predictions and outcomes for each horizon
    results = {h: [] for h in horizons}

    for window in windows:
        # Fit features on training data
        train_px = px_clean.loc[window["train_start"]:window["train_end"]]

        try:
            train_feats = compute_features(train_px)

            # Get predictions at end of training window
            mu_now = safe_last(train_feats.get("mu_post", pd.Series([0.0])))
            vol_now = safe_last(train_feats.get("vol", pd.Series([1.0])))

            if not np.isfinite(mu_now):
                mu_now = 0.0
            if not np.isfinite(vol_now) or vol_now <= 0:
                vol_now = 1.0

            # For each horizon, predict and measure actual outcome
            test_log_px = log_px.loc[window["test_start"]:window["test_end"]]
            train_end_log_px = float(log_px.loc[window["train_end"]])

            for H in horizons:
                # Predicted return over H days
                pred_ret_H = mu_now * H
                pred_sign = np.sign(pred_ret_H) if pred_ret_H != 0 else 0

                # Actual return H days forward from train_end
                try:
                    forward_idx = dates.get_loc(window["train_end"]) + H
                    if forward_idx < len(dates):
                        forward_date = dates[forward_idx]
                        actual_log_px = float(log_px.loc[forward_date])
                        actual_ret_H = actual_log_px - train_end_log_px
                        actual_sign = np.sign(actual_ret_H) if actual_ret_H != 0 else 0

                        # Prediction error
                        pred_error = actual_ret_H - pred_ret_H

                        # Direction hit (1 if signs match, 0 otherwise)
                        hit = 1 if (pred_sign * actual_sign > 0) else 0

                        results[H].append({
                            "train_end": window["train_end"],
                            "forecast_date": forward_date,
                            "predicted_return": pred_ret_H,
                            "actual_return": actual_ret_H,
                            "prediction_error": pred_error,
                            "direction_hit": hit,
                        })
                except Exception:
                    continue

        except Exception:
            continue

    # Aggregate results into DataFrames
    oos_metrics = {}
    for H in horizons:
        if results[H]:
            df = pd.DataFrame(results[H]).set_index("train_end")

            # Compute cumulative statistics
            hit_rate = df["direction_hit"].mean() if len(df) > 0 else float("nan")
            mean_error = df["prediction_error"].mean() if len(df) > 0 else float("nan")
            rmse = np.sqrt((df["prediction_error"] ** 2).mean()) if len(df) > 0 else float("nan")

            oos_metrics[f"H{H}"] = {
                "predictions": df,
                "hit_rate": float(hit_rate),
                "mean_error": float(mean_error),
                "rmse": float(rmse),
                "n_forecasts": len(df),
            }

    return oos_metrics


# =========================================================================
# Story 2.1: Walk-Forward Backtest Infrastructure
# =========================================================================
# Full-pipeline walk-forward that uses latest_signals() to produce
# (forecast, realized) pairs for downstream calibration (EMOS, vol ratio).
#
# Design:
#   At each rebalance date t:
#     1. Truncate prices/OHLC to [:t]  (no look-ahead)
#     2. compute_features() on truncated data
#     3. latest_signals() with tune cache
#     4. Record forecast_ret, forecast_sig, p_up per horizon
#     5. After waiting for realized, compute realized_log_ret
#
# Output: DataFrame with columns:
#   [date, horizon, forecast_ret, forecast_p_up, forecast_sig,
#    realized_ret, hit, signed_error, calibration_bucket]
# =========================================================================

WF_HORIZONS = [1, 3, 7, 21, 63]
WF_REBALANCE_FREQ = 5
WF_RETUNE_FREQ = 63
WF_MIN_TRAIN_DAYS = 504   # ~2 years of trading days required before first forecast
WF_CALIBRATION_BUCKETS = 10


@dataclass
class WalkForwardRecord:
    """Single forecast-vs-realized observation."""
    date_idx: int
    date: object             # pd.Timestamp
    horizon: int
    forecast_ret: float      # E[log_ret] from latest_signals
    forecast_p_up: float     # P(r>0) from latest_signals
    forecast_sig: float      # sigma_H from latest_signals
    realized_ret: float      # actual log(P[t+H]/P[t])
    hit: bool                # (realized>0)==(forecast>0)
    signed_error: float      # realized - forecast
    calibration_bucket: int  # int in [0, WF_CALIBRATION_BUCKETS)


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward backtest results."""
    records: List['WalkForwardRecord'] = field(default_factory=list)
    hit_rate: Dict[int, float] = field(default_factory=dict)
    rmse: Dict[int, float] = field(default_factory=dict)
    mae: Dict[int, float] = field(default_factory=dict)
    mean_signed_error: Dict[int, float] = field(default_factory=dict)
    calibration_curve: Dict[int, List[Dict[str, float]]] = field(default_factory=dict)
    n_forecasts: Dict[int, int] = field(default_factory=dict)
    symbol: str = ""


def run_walk_forward_backtest(
    symbol: str,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    rebalance_freq: int = WF_REBALANCE_FREQ,
    horizons: Optional[List[int]] = None,
    retune_freq: int = WF_RETUNE_FREQ,
    n_mc_paths: int = 5000,
) -> WalkForwardResult:
    """
    Full-pipeline walk-forward backtest using latest_signals().

    At each rebalance date, truncates data to simulate real-time and calls
    the production signal pipeline.  Records (forecast, realized) pairs
    for downstream EMOS calibration and volatility ratio computation.

    Args:
        symbol: Asset ticker (e.g. "SPY").
        start_date: First date for price data fetch.
        end_date: Last date for price data fetch.
        rebalance_freq: Days between rebalance dates.
        horizons: Forecast horizons (default [1,3,7,21,63]).
        retune_freq: Re-tune model every N steps (placeholder).
        n_mc_paths: MC paths per call (reduced from 10k for speed).

    Returns:
        WalkForwardResult with per-record detail and aggregate stats.
    """
    from decision.signals import latest_signals, Signal  # lazy: avoid circular
    if horizons is None:
        horizons = list(WF_HORIZONS)
    h_max = max(horizons)

    # ------------------------------------------------------------------
    # 1. Load full price + OHLC data ONCE
    # ------------------------------------------------------------------
    px, _title = fetch_px_asset(symbol, start_date, end_date)
    px = _ensure_float_series(px).dropna()
    px = px[px > 0]
    n_total = len(px)
    if n_total < WF_MIN_TRAIN_DAYS + h_max + 1:
        return WalkForwardResult(symbol=symbol)

    ohlc_df = None
    if GK_VOLATILITY_AVAILABLE:
        try:
            ohlc_df = _download_prices(symbol, start_date, end_date)
        except Exception:
            ohlc_df = None

    log_px = np.log(px.values)

    # Load tune cache once (static params for entire backtest)
    tuned_params = _load_tuned_kalman_params(symbol)
    _asset_type = classify_asset_type(symbol)

    # Story 7.2: Smart caching for tune params within retune window
    _fc = _WFFeatureCache(retune_freq=retune_freq)

    # ------------------------------------------------------------------
    # 2. Walk forward
    # ------------------------------------------------------------------
    records: List[WalkForwardRecord] = []

    t = WF_MIN_TRAIN_DAYS
    while t + h_max < n_total:
        # Truncate data to [0, t]  — no look-ahead
        px_as_of_t = px.iloc[:t + 1]  # inclusive of t
        ohlc_as_of_t = None
        if ohlc_df is not None and not ohlc_df.empty:
            ohlc_as_of_t = ohlc_df.iloc[:t + 1]

        last_close = float(px_as_of_t.iloc[-1])

        # Story 7.2: Use cached tune params within retune window
        tuned_params = _fc.get_tune_params(symbol, t)

        # Compute features on truncated data (respects no look-ahead)
        try:
            feats = compute_features(
                px_as_of_t, asset_symbol=symbol, ohlc_df=ohlc_as_of_t,
            )
        except Exception:
            t += rebalance_freq
            continue

        # Call production signal pipeline
        try:
            sigs, _thr = latest_signals(
                feats, horizons, last_close=last_close, t_map=True,
                ci=0.68, tuned_params=tuned_params, asset_key=symbol,
                n_mc_paths=n_mc_paths, asset_type=_asset_type,
                _calibration_fast_mode=True,
            )
        except Exception:
            t += rebalance_freq
            continue

        # Map horizon -> Signal for quick lookup
        sig_map: Dict[int, Signal] = {}
        for s in sigs:
            sig_map[s.horizon_days] = s

        # Record (forecast, realized) for each horizon
        for H in horizons:
            if t + H >= n_total:
                continue
            sig = sig_map.get(H)
            if sig is None:
                continue

            realized_ret = float(log_px[t + H] - log_px[t])

            # Direction hit
            hit = (realized_ret > 0) == (sig.exp_ret > 0)

            # Calibration bucket: discretize p_up into WF_CALIBRATION_BUCKETS bins
            bucket = min(
                int(sig.p_up * WF_CALIBRATION_BUCKETS),
                WF_CALIBRATION_BUCKETS - 1,
            )

            records.append(WalkForwardRecord(
                date_idx=t,
                date=px.index[t],
                horizon=H,
                forecast_ret=float(sig.exp_ret),
                forecast_p_up=float(sig.p_up),
                forecast_sig=float(sig.vol_mean) if sig.vol_mean > 0 else 0.01,
                realized_ret=realized_ret,
                hit=hit,
                signed_error=realized_ret - float(sig.exp_ret),
                calibration_bucket=bucket,
            ))

        t += rebalance_freq

    # ------------------------------------------------------------------
    # 3. Compute aggregate statistics
    # ------------------------------------------------------------------
    result = WalkForwardResult(records=records, symbol=symbol)
    if not records:
        return result

    for H in horizons:
        h_recs = [r for r in records if r.horizon == H]
        n_h = len(h_recs)
        result.n_forecasts[H] = n_h
        if n_h == 0:
            continue
        hits = sum(1 for r in h_recs if r.hit)
        result.hit_rate[H] = hits / n_h
        errs = np.array([r.signed_error for r in h_recs])
        result.rmse[H] = float(np.sqrt(np.mean(errs ** 2)))
        result.mae[H] = float(np.mean(np.abs(errs)))
        result.mean_signed_error[H] = float(np.mean(errs))

        # Calibration curve: for each bucket, compute (mean_p_up, realized_hit_freq)
        buckets: List[Dict[str, float]] = []
        for b in range(WF_CALIBRATION_BUCKETS):
            b_recs = [r for r in h_recs if r.calibration_bucket == b]
            if b_recs:
                mean_p = np.mean([r.forecast_p_up for r in b_recs])
                hit_freq = np.mean([1.0 if r.realized_ret > 0 else 0.0 for r in b_recs])
                buckets.append({
                    "bucket": b,
                    "mean_forecast_p_up": float(mean_p),
                    "realized_hit_freq": float(hit_freq),
                    "n_obs": len(b_recs),
                })
            else:
                buckets.append({
                    "bucket": b,
                    "mean_forecast_p_up": (b + 0.5) / WF_CALIBRATION_BUCKETS,
                    "realized_hit_freq": float("nan"),
                    "n_obs": 0,
                })
        result.calibration_curve[H] = buckets

    return result


def walkforward_result_to_dataframe(wf: WalkForwardResult) -> 'pd.DataFrame':
    """Convert WalkForwardResult to a pandas DataFrame for analysis."""
    if not wf.records:
        return pd.DataFrame()
    rows = []
    for r in wf.records:
        rows.append({
            "date": r.date,
            "horizon": r.horizon,
            "forecast_ret": r.forecast_ret,
            "forecast_p_up": r.forecast_p_up,
            "forecast_sig": r.forecast_sig,
            "realized_ret": r.realized_ret,
            "hit": r.hit,
            "signed_error": r.signed_error,
            "calibration_bucket": r.calibration_bucket,
        })
    return pd.DataFrame(rows)


def save_walkforward_csv(wf: WalkForwardResult, output_dir: str = "src/data/calibration") -> str:
    """Save walk-forward results to CSV. Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"walkforward_{wf.symbol}.csv")
    df = walkforward_result_to_dataframe(wf)
    if not df.empty:
        df.to_csv(path, index=False)
    return path


# ─── Story 7.2: Parallel Walk-Forward with Smart Caching ────────────────

class _WFFeatureCache:
    """Caches computed features for incremental walk-forward reuse.

    Within a retune window, tune_params are constant so we cache them.
    Features at t+freq share most data with t, but compute_features() is
    stateless (expanding window), so we cache the last feature dict and
    skip recomputation when the new window only adds `rebalance_freq` days.
    """
    __slots__ = ("_tune_params", "_tune_valid_until", "_last_feats",
                 "_last_t", "_retune_freq")

    def __init__(self, retune_freq: int = WF_RETUNE_FREQ):
        self._tune_params: Optional[Dict] = None
        self._tune_valid_until: int = -1
        self._last_feats: Optional[Dict] = None
        self._last_t: int = -1
        self._retune_freq = retune_freq

    def get_tune_params(self, symbol: str, t: int) -> Dict:
        """Return cached tune params if within retune window, else reload."""
        if self._tune_params is not None and t <= self._tune_valid_until:
            return self._tune_params
        self._tune_params = _load_tuned_kalman_params(symbol) or {}
        self._tune_valid_until = t + self._retune_freq
        return self._tune_params

    def cache_hit(self, t: int, rebalance_freq: int) -> bool:
        """Check if previous features can be reused (same tune window)."""
        if self._last_feats is None or self._last_t < 0:
            return False
        return (t - self._last_t) == rebalance_freq

    def get_cached_feats(self) -> Optional[Dict]:
        return self._last_feats

    def store(self, t: int, feats: Dict) -> None:
        self._last_feats = feats
        self._last_t = t


def run_walk_forward_parallel(
    symbols: List[str],
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    rebalance_freq: int = WF_REBALANCE_FREQ,
    horizons: Optional[List[int]] = None,
    max_workers: Optional[int] = None,
    n_mc_paths: int = 5000,
) -> Dict[str, WalkForwardResult]:
    """Run walk-forward backtest across multiple assets in parallel.

    Uses ProcessPoolExecutor for CPU-bound work. Each asset is independent.

    Returns:
        Dict mapping symbol -> WalkForwardResult.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    def _run_single(sym: str) -> tuple:
        result = run_walk_forward_backtest(
            sym, start_date=start_date, end_date=end_date,
            rebalance_freq=rebalance_freq, horizons=horizons,
            n_mc_paths=n_mc_paths,
        )
        return (sym, result)

    results: Dict[str, WalkForwardResult] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_single, s): s for s in symbols}
        for future in futures:
            try:
                sym, wf = future.result()
                results[sym] = wf
            except Exception:
                results[futures[future]] = WalkForwardResult(symbol=futures[future])
    return results

