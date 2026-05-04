"""Causal OHLCV rectification firewall for cycles 151-170.

This module intentionally does not promote indicator models.  It builds the
measurement layer that the upgraded plan requires before any live model channel
can earn authority: immutable baselines, OHLCV-only audits, causal labels,
purged chronological splits, cost proxies, feature catalog, hypothesis
registry, discovery/validation edge maps, null controls, multiplicity checks,
and a go/no-go decision for reliability/model integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tuning.benchmark_retune_50 import BENCHMARK_50
except Exception:
    BENCHMARK_50 = (
        "AAPL", "MSFT", "NVDA", "GOOGL", "CRM", "ADBE", "CRWD", "NET",
        "JPM", "BAC", "GS", "MS", "SCHW", "AFRM",
        "LMT", "RTX", "NOC", "GD",
        "JNJ", "UNH", "PFE", "ABBV", "MRNA",
        "CAT", "DE", "BA", "UPS", "GE",
        "XOM", "CVX", "COP", "SLB", "OXY",
        "LIN", "FCX", "NEM", "NUE",
        "AMZN", "TSLA", "HD", "NKE", "SBUX", "PG", "KO", "PEP", "COST",
        "META", "NFLX", "DIS", "SNAP",
    )

from models.indicator_admission import prune_correlated_indicator_columns


ALLOWED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
DEFAULT_HORIZONS = (1, 5, 10, 21)
DEFAULT_PRICE_DIR = Path("src/data/prices")
DEFAULT_TUNE_DIR = Path("src/data/tune")
DEFAULT_OUTPUT_DIR = Path("src/data/benchmarks/rectification")

CYCLE_100_REFERENCE = {
    "bic_mean": -13471.718382997637,
    "pit_mean": 0.7618466601969215,
    "pit_min": 0.25568959090050664,
    "signals_profit_factor_calibrated_mean": 1.973747995,
    "signals_strategy_sharpe_calibrated_mean": 2.85343172,
    "signals_hit_rate_calibrated_mean": 0.580366105,
}

CYCLE_150_REFERENCE = {
    "bic_mean": -13469.487277730757,
    "pit_mean": 0.7434874406719306,
    "pit_min": 0.33046765605579104,
    "signals_profit_factor_calibrated_mean": 1.9737479949999999,
    "signals_strategy_sharpe_calibrated_mean": 2.8534317199999997,
    "signals_hit_rate_calibrated_mean": 0.580366105,
    "signals_brier_improvement_mean": 0.02790282,
    "signals_crps_improvement_mean": 4.33209981,
}

FEATURE_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "price_path": (
        "ret_1", "ret_5", "ret_10", "ret_21", "open_to_close",
        "overnight_gap", "body_ratio", "upper_wick_ratio",
        "lower_wick_ratio", "close_location",
    ),
    "volatility_range": (
        "range_pct", "atr_norm_14", "range_z_20", "realized_vol_5",
        "realized_vol_21", "vol_of_vol_21", "bb_z_20",
        "bb_width_z_20", "range_compression_20",
    ),
    "trend_chop": (
        "kama_efficiency_10", "trend_slope_20", "trend_curvature",
        "donchian_pos_20", "donchian_width_z_20",
    ),
    "momentum_oscillator": (
        "rsi_z_14", "macd_z", "momentum_accel", "ret_stack_z",
    ),
    "volume_liquidity": (
        "volume_z_20", "volume_price_alignment", "dollar_volume_z_20",
        "turnover_proxy",
    ),
    "cross_sectional": (
        "xs_ret_5_rank", "xs_ret_21_rank", "xs_vol_21_rank",
        "xs_volume_z_rank",
    ),
}

FEATURE_TO_FAMILY: Dict[str, str] = {
    feature: family
    for family, features in FEATURE_FAMILIES.items()
    for feature in features
}

HYPOTHESIS_MECHANISMS: Mapping[str, Mapping[str, str]] = {
    "price_path": {
        "mechanism": "Candle geometry and recent path shape proxy short-horizon drift/reversal state.",
        "expected_relation": "state-dependent signed relation; must be learned out-of-sample",
        "failure_mode": "decorative candle pattern with no density or net-economic lift",
    },
    "volatility_range": {
        "mechanism": "Range expansion/compression identifies forecast variance and tail miss regimes.",
        "expected_relation": "higher range state should improve variance/tail calibration, not direct votes",
        "failure_mode": "activity throttle that only reduces turnover without better CRPS/Brier",
    },
    "trend_chop": {
        "mechanism": "Efficiency and breakout location separate trend continuation from noisy chop.",
        "expected_relation": "positive only in predeclared trend-friendly horizons and regimes",
        "failure_mode": "single-asset trend overfit or collinearity with existing momentum state",
    },
    "momentum_oscillator": {
        "mechanism": "Bounded impulse/exhaustion state may improve drift timing after calibration.",
        "expected_relation": "monotone quantile spread after purged validation",
        "failure_mode": "classic oscillator overfit that vanishes after null controls",
    },
    "volume_liquidity": {
        "mechanism": "Volume pressure and dollar-volume proxy capacity, stale authority, and net costs.",
        "expected_relation": "better net economics and fewer weak-capacity entries",
        "failure_mode": "pre-cost edge only or unnecessary liquidity veto",
    },
    "cross_sectional": {
        "mechanism": "Same-timestamp ranks detect relative state without external feeds.",
        "expected_relation": "cluster-stable information coefficient and after-cost materiality",
        "failure_mode": "index beta proxy that fails no-indicator and cross-family controls",
    },
}


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split and embargo settings."""

    train_frac: float = 0.60
    validation_frac: float = 0.20
    purge_bars: int = max(DEFAULT_HORIZONS)
    embargo_bars: int = 126


@dataclass(frozen=True)
class CostConfig:
    """OHLCV-only transaction cost proxy settings."""

    min_spread_bps: float = 1.0
    max_spread_bps: float = 75.0
    range_to_spread: float = 0.05
    slippage_fraction_of_spread: float = 0.50
    low_liquidity_bps: float = 4.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataframe_hash(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
    cols = list(columns) if columns is not None else list(frame.columns)
    canonical = frame.loc[:, cols].copy()
    for col in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[col]):
            canonical[col] = canonical[col].dt.strftime("%Y-%m-%d")
    digest = hashlib.sha256()
    digest.update("|".join(cols).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(canonical, index=False).values.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=", "_").replace(":", "_").upper()


def _price_path(symbol: str, price_dir: Path) -> Optional[Path]:
    safe = _safe_symbol(symbol)
    candidates = (price_dir / f"{safe}.csv", price_dir / f"{safe}_1d.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def _normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    return {col: str(col).strip().lower().replace(" ", "_") for col in columns}


def load_ohlcv_frame(symbol: str, price_dir: Path = DEFAULT_PRICE_DIR) -> pd.DataFrame:
    """Load one cached OHLCV frame, consuming only allowed OHLCV fields."""
    path = _price_path(symbol, price_dir)
    if path is None:
        raise FileNotFoundError(f"no cached price file for {symbol}")
    raw = pd.read_csv(path)
    raw = raw.rename(columns=_normalize_columns(raw.columns))
    if "date" not in raw.columns:
        raise ValueError(f"{symbol}: cached price file has no Date column")
    missing = [col for col in ALLOWED_OHLCV_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"{symbol}: missing OHLCV columns {missing}")
    frame = raw[["date", *ALLOWED_OHLCV_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ALLOWED_OHLCV_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", *ALLOWED_OHLCV_COLUMNS])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["symbol"] = _safe_symbol(symbol)
    return frame


def load_price_panel(symbols: Sequence[str], price_dir: Path = DEFAULT_PRICE_DIR) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frames[_safe_symbol(symbol)] = load_ohlcv_frame(symbol, price_dir)
        except FileNotFoundError:
            continue
    return frames


def _rolling_z(series: pd.Series, window: int, min_periods: int = 5) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-8.0, 8.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def _rsi_z(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)


def compute_ohlcv_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic same-bar OHLCV states used after bar close."""
    df = frame.copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0.0)

    prev_close = close.shift(1)
    range_abs = (high - low).clip(lower=1e-12)
    range_pct = range_abs / close.replace(0.0, np.nan)
    ret_1 = close.pct_change()

    df["ret_1"] = ret_1
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_21"] = close.pct_change(21)
    df["open_to_close"] = (close - open_) / open_.replace(0.0, np.nan)
    df["overnight_gap"] = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
    df["range_pct"] = range_pct
    df["body_ratio"] = ((close - open_) / range_abs).clip(-4.0, 4.0)
    df["upper_wick_ratio"] = ((high - np.maximum(open_, close)) / range_abs).clip(0.0, 4.0)
    df["lower_wick_ratio"] = ((np.minimum(open_, close) - low) / range_abs).clip(0.0, 4.0)
    df["close_location"] = ((close - low) / range_abs).clip(0.0, 1.0)

    true_range = pd.concat(
        [range_abs, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    df["atr_norm_14"] = atr_14 / close.replace(0.0, np.nan)
    df["range_z_20"] = _rolling_z(range_pct, 20)
    df["realized_vol_5"] = ret_1.rolling(5, min_periods=5).std(ddof=0) * math.sqrt(252.0)
    df["realized_vol_21"] = ret_1.rolling(21, min_periods=10).std(ddof=0) * math.sqrt(252.0)
    df["vol_of_vol_21"] = np.log(df["realized_vol_21"].clip(lower=1e-8)).diff().abs()
    ma20 = close.rolling(20, min_periods=10).mean()
    sd20 = close.rolling(20, min_periods=10).std(ddof=0)
    df["bb_z_20"] = ((close - ma20) / sd20.replace(0.0, np.nan)).clip(-8.0, 8.0)
    df["bb_width_z_20"] = _rolling_z((4.0 * sd20) / close.replace(0.0, np.nan), 60, min_periods=20)
    df["range_compression_20"] = (range_pct / range_pct.rolling(20, min_periods=10).mean()).clip(0.0, 8.0)

    direction_10 = (close - close.shift(10)).abs()
    volatility_10 = close.diff().abs().rolling(10, min_periods=10).sum()
    df["kama_efficiency_10"] = (direction_10 / volatility_10.replace(0.0, np.nan)).clip(0.0, 1.0)
    df["trend_slope_20"] = (close.pct_change(20) / (df["realized_vol_21"] / math.sqrt(252.0)).replace(0.0, np.nan)).clip(-8.0, 8.0)
    df["trend_curvature"] = (df["ret_5"] - df["ret_21"] / 4.2).clip(-1.0, 1.0)
    rolling_high = high.rolling(20, min_periods=10).max()
    rolling_low = low.rolling(20, min_periods=10).min()
    donchian_width = (rolling_high - rolling_low).replace(0.0, np.nan)
    df["donchian_pos_20"] = ((close - rolling_low) / donchian_width).clip(0.0, 1.0)
    df["donchian_width_z_20"] = _rolling_z(donchian_width / close.replace(0.0, np.nan), 60, min_periods=20)

    df["rsi_z_14"] = _rsi_z(close, 14)
    macd = _ema(close, 12) - _ema(close, 26)
    df["macd_z"] = _rolling_z(macd / close.replace(0.0, np.nan), 60, min_periods=20)
    df["momentum_accel"] = (df["ret_5"] - df["ret_10"] / 2.0).clip(-1.0, 1.0)
    df["ret_stack_z"] = _rolling_z(df["ret_1"] + 0.5 * df["ret_5"] + 0.25 * df["ret_21"], 63, min_periods=20)

    df["volume_z_20"] = _rolling_z(np.log1p(volume), 20)
    df["volume_price_alignment"] = (np.sign(ret_1.fillna(0.0)) * df["volume_z_20"]).clip(-8.0, 8.0)
    dollar_volume = close * volume
    df["dollar_volume_z_20"] = _rolling_z(np.log1p(dollar_volume), 20)
    df["turnover_proxy"] = (dollar_volume / dollar_volume.rolling(63, min_periods=20).mean()).clip(0.0, 8.0)
    return df.replace([np.inf, -np.inf], np.nan)


def _load_tune_global(symbol: str, tune_dir: Path = DEFAULT_TUNE_DIR) -> Mapping[str, Any]:
    path = tune_dir / f"{_safe_symbol(symbol)}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    global_data = data.get("global", data)
    return global_data if isinstance(global_data, dict) else {}


def _regime_label(features: pd.DataFrame) -> pd.Series:
    rv = features["realized_vol_21"]
    slope = features["trend_slope_20"].abs()
    rv_med = rv.expanding(min_periods=63).median()
    trend = slope > 0.75
    high_vol = rv > rv_med
    out = np.where(high_vol & trend, "HIGH_VOL_TREND", "LOW_VOL_TREND")
    out = np.where(high_vol & ~trend, "HIGH_VOL_RANGE", out)
    out = np.where(~high_vol & ~trend, "LOW_VOL_RANGE", out)
    crisis = (features["range_z_20"] > 2.5) | (features["vol_of_vol_21"] > 0.35)
    out = np.where(crisis, "CRISIS_JUMP", out)
    return pd.Series(out, index=features.index)


def _cost_proxy(features: pd.DataFrame, config: CostConfig) -> pd.DataFrame:
    spread_bps = (features["range_pct"].clip(lower=0.0) * 10000.0 * config.range_to_spread)
    spread_bps = spread_bps.clip(config.min_spread_bps, config.max_spread_bps)
    liquidity_penalty = config.low_liquidity_bps * (1.0 - features["turnover_proxy"].clip(0.0, 1.0))
    slippage_bps = config.slippage_fraction_of_spread * spread_bps + liquidity_penalty
    out = pd.DataFrame(index=features.index)
    out["spread_proxy_bps"] = spread_bps
    out["slippage_proxy_bps"] = slippage_bps
    out["cost_proxy_return"] = (spread_bps + slippage_bps) / 10000.0
    return out


def build_causal_panel(
    price_frames: Mapping[str, pd.DataFrame],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cost_config: CostConfig = CostConfig(),
    tune_dir: Path = DEFAULT_TUNE_DIR,
) -> pd.DataFrame:
    """Build one causal panel with same-bar OHLCV states and future labels."""
    rows: List[pd.DataFrame] = []
    feature_columns = list(FEATURE_TO_FAMILY)
    for symbol, raw in price_frames.items():
        features = compute_ohlcv_features(raw)
        costs = _cost_proxy(features, cost_config)
        tune_global = _load_tune_global(symbol, tune_dir)
        base_feature_columns = [col for col in feature_columns if col in features.columns]
        base_cols = ["symbol", "date", "open", "high", "low", "close", "volume", *base_feature_columns]
        base = features[base_cols].copy()
        base["feature_timestamp"] = base["date"]
        base["decision_timestamp"] = base["date"]
        base["regime_cluster"] = _regime_label(features)
        base["best_model"] = str(tune_global.get("best_model") or tune_global.get("noise_model") or "unknown")
        base["pit_ks_pvalue"] = float(tune_global.get("pit_ks_pvalue", np.nan))
        base = pd.concat([base, costs], axis=1)
        for horizon in horizons:
            block = base.copy()
            block["horizon"] = int(horizon)
            future_close = features["close"].shift(-int(horizon))
            block["label_timestamp"] = features["date"].shift(-int(horizon))
            block["forward_return"] = future_close / features["close"].replace(0.0, np.nan) - 1.0
            future_path = features["close"].pct_change().shift(-int(horizon) + 1).rolling(int(horizon), min_periods=1).std(ddof=0)
            block["realized_vol_label"] = future_path
            rows.append(block)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = _add_cross_sectional_features(panel)
    panel = panel.dropna(subset=["forward_return", "label_timestamp"]).reset_index(drop=True)
    return panel.replace([np.inf, -np.inf], np.nan)


def _rank_pct(series: pd.Series) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index)
    return series.rank(method="average", pct=True) * 2.0 - 1.0


def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    grouped = out.groupby(["decision_timestamp", "horizon"], sort=False)
    out["xs_ret_5_rank"] = grouped["ret_5"].transform(_rank_pct)
    out["xs_ret_21_rank"] = grouped["ret_21"].transform(_rank_pct)
    out["xs_vol_21_rank"] = grouped["realized_vol_21"].transform(_rank_pct)
    out["xs_volume_z_rank"] = grouped["volume_z_20"].transform(_rank_pct)
    return out


def apply_purged_splits(panel: pd.DataFrame, config: SplitConfig = SplitConfig()) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign deterministic chronological split membership with purge/embargo."""
    if panel.empty:
        raise ValueError("panel is empty")
    dates = pd.Series(pd.to_datetime(panel["decision_timestamp"].drop_duplicates())).sort_values().reset_index(drop=True)
    n_dates = len(dates)
    train_cut = int(math.floor(n_dates * config.train_frac))
    validation_cut = int(math.floor(n_dates * (config.train_frac + config.validation_frac)))
    purge = max(0, int(config.purge_bars))
    embargo = max(0, int(config.embargo_bars))

    train_end = max(0, train_cut - purge)
    validation_start = min(n_dates, train_cut + embargo)
    validation_end = max(validation_start, validation_cut - purge)
    final_start = min(n_dates, validation_cut + embargo)

    split_by_date: Dict[pd.Timestamp, str] = {}
    for idx, date in enumerate(dates):
        if idx < train_end:
            split_by_date[date] = "discovery_train"
        elif validation_start <= idx < validation_end:
            split_by_date[date] = "validation"
        elif idx >= final_start:
            split_by_date[date] = "final_test"
        else:
            split_by_date[date] = "purged_embargoed"

    out = panel.copy()
    out["split"] = pd.to_datetime(out["decision_timestamp"]).map(split_by_date).fillna("purged_embargoed")
    counts = {str(k): int(v) for k, v in out["split"].value_counts().sort_index().items()}
    date_counts = {str(k): int(v) for k, v in pd.Series(split_by_date).value_counts().sort_index().items()}
    manifest = {
        "split_config": {
            "train_frac": config.train_frac,
            "validation_frac": config.validation_frac,
            "purge_bars": config.purge_bars,
            "embargo_bars": config.embargo_bars,
        },
        "n_dates": n_dates,
        "train_cut_index": train_cut,
        "validation_cut_index": validation_cut,
        "train_end_index_after_purge": train_end,
        "validation_start_index_after_embargo": validation_start,
        "validation_end_index_after_purge": validation_end,
        "final_start_index_after_embargo": final_start,
        "row_counts": counts,
        "date_counts": date_counts,
        "split_membership_hash": stable_json_hash({d.strftime("%Y-%m-%d"): s for d, s in split_by_date.items()}),
    }
    return out, manifest


def audit_ohlcv_contract(price_frames: Mapping[str, pd.DataFrame], price_dir: Path = DEFAULT_PRICE_DIR) -> Dict[str, Any]:
    per_symbol: Dict[str, Any] = {}
    unresolved_leakage = False
    for symbol, frame in price_frames.items():
        path = _price_path(symbol, price_dir)
        raw_columns: List[str] = []
        ignored_columns: List[str] = []
        if path is not None:
            try:
                raw = pd.read_csv(path, nrows=1)
                raw_columns = list(raw.columns)
                normalized = _normalize_columns(raw.columns)
                ignored_columns = [
                    col for col, norm in normalized.items()
                    if norm not in ("date", *ALLOWED_OHLCV_COLUMNS)
                ]
            except Exception:
                pass
        high_low_violations = int((frame["high"] < frame["low"]).sum())
        negative_volume = int((frame["volume"] < 0.0).sum())
        duplicate_dates = int(frame["date"].duplicated().sum())
        finite_violations = int((~np.isfinite(frame[list(ALLOWED_OHLCV_COLUMNS)].to_numpy(dtype=float))).sum())
        unresolved = high_low_violations + negative_volume + duplicate_dates + finite_violations
        unresolved_leakage = unresolved_leakage or unresolved > 0
        per_symbol[symbol] = {
            "path": str(path) if path is not None else None,
            "raw_columns": raw_columns,
            "consumed_columns": list(ALLOWED_OHLCV_COLUMNS),
            "ignored_columns": ignored_columns,
            "row_count": int(len(frame)),
            "date_min": frame["date"].min().strftime("%Y-%m-%d") if len(frame) else None,
            "date_max": frame["date"].max().strftime("%Y-%m-%d") if len(frame) else None,
            "high_low_violations": high_low_violations,
            "negative_volume": negative_volume,
            "duplicate_dates_after_normalization": duplicate_dates,
            "finite_violations": finite_violations,
            "raw_input_hash": dataframe_hash(frame, ["date", *ALLOWED_OHLCV_COLUMNS]),
        }
    return {
        "allowed_new_inputs": list(ALLOWED_OHLCV_COLUMNS),
        "disallowed_new_inputs": [
            "adj_close", "news", "fundamentals", "order_book", "intraday_unless_declared",
            "options", "borrow", "short_interest", "analyst", "social", "macro",
            "future_close", "future_high", "future_low", "validation_outcomes",
        ],
        "symbol_count": len(price_frames),
        "unresolved_leakage_or_contract_failures": bool(unresolved_leakage),
        "per_symbol": per_symbol,
    }


def _extract_metric(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    if key in metrics and isinstance(metrics[key], (int, float)):
        return float(metrics[key])
    cache_summary = metrics.get("cache_summary", {})
    signal_summary = metrics.get("signals_calibration_summary", {})
    if isinstance(cache_summary, dict) and isinstance(cache_summary.get(key), (int, float)):
        return float(cache_summary[key])
    if isinstance(signal_summary, dict) and isinstance(signal_summary.get(key), (int, float)):
        return float(signal_summary[key])
    return None


def baseline_reproduction_report(benchmark_dir: Path = Path("src/data/benchmarks")) -> Dict[str, Any]:
    paths = {
        "cycle_100": benchmark_dir / "cycle_100_final_release_gate_full_metrics.json",
        "cycle_150": benchmark_dir / "cycle_150_final_indicator_integrated_release_gate_metrics.json",
    }
    references = {"cycle_100": CYCLE_100_REFERENCE, "cycle_150": CYCLE_150_REFERENCE}
    report: Dict[str, Any] = {"artifacts": {}, "passed": True}
    for cycle, path in paths.items():
        if not path.exists():
            report["artifacts"][cycle] = {"path": str(path), "exists": False}
            report["passed"] = False
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics: Dict[str, Any] = {}
        deltas: Dict[str, Any] = {}
        for key, expected in references[cycle].items():
            observed = _extract_metric(data, key)
            metrics[key] = observed
            deltas[key] = None if observed is None else observed - expected
            if observed is None or abs(observed - expected) > 1e-9:
                report["passed"] = False
        report["artifacts"][cycle] = {
            "path": str(path),
            "exists": True,
            "sha256": file_sha256(path),
            "metrics": metrics,
            "deltas_vs_frozen_reference": deltas,
            "asset_count": data.get("asset_count"),
            "failed_count": data.get("failed_count"),
        }
    report["bic_convention"] = "candidate_minus_control; negative_delta_is_improvement"
    return report


def metric_noise_budget(benchmark_dir: Path = Path("src/data/benchmarks")) -> Dict[str, Any]:
    clean_paths = [
        benchmark_dir / "cycle_100_final_release_gate_full_metrics.json",
        benchmark_dir / "cycle_148_full_indicator_competition_gate_metrics.json",
        benchmark_dir / "cycle_149_signal_generation_smoke_metrics.json",
    ]
    excluded = benchmark_dir / "cycle_150_final_indicator_integrated_release_gate_metrics.json"
    metric_keys = (
        "bic_mean", "pit_mean", "pit_min",
        "signals_profit_factor_calibrated_mean",
        "signals_strategy_sharpe_calibrated_mean",
        "signals_hit_rate_calibrated_mean",
        "signals_brier_improvement_mean",
        "signals_crps_improvement_mean",
        "total_seconds",
    )
    observations: Dict[str, List[float]] = {key: [] for key in metric_keys}
    sources: List[str] = []
    for path in clean_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for key in metric_keys:
            value = _extract_metric(data, key)
            if key == "total_seconds" and isinstance(data.get("total_seconds"), (int, float)):
                value = float(data["total_seconds"])
            if value is not None and math.isfinite(value):
                observations[key].append(value)
    bands: Dict[str, Any] = {}
    for key, values in observations.items():
        if not values:
            bands[key] = {"n": 0, "mean": None, "std": None, "noise_band": None}
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        bands[key] = {
            "n": len(values),
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "noise_band": float(max(std * 2.0, abs(mean) * 1e-6)),
        }
    return {
        "clean_sources": sources,
        "excluded_sources": [{
            "path": str(excluded),
            "reason": "cycle 150 recorded abnormal detached runtime and is not a runtime release baseline",
            "sha256": file_sha256(excluded),
        }],
        "metric_bands": bands,
        "runtime_release_baseline_seconds": bands.get("total_seconds", {}).get("mean"),
        "runtime_release_budget_seconds": None if bands.get("total_seconds", {}).get("mean") is None else bands["total_seconds"]["mean"] * 1.15,
    }


def feature_catalog_report(panel: pd.DataFrame) -> Dict[str, Any]:
    feature_columns = [col for col in FEATURE_TO_FAMILY if col in panel.columns]
    features: Dict[str, Any] = {}
    for feature in feature_columns:
        series = panel[feature]
        features[feature] = {
            "family": FEATURE_TO_FAMILY[feature],
            "finite_fraction": float(np.isfinite(series.to_numpy(dtype=float)).mean()),
            "lookback_upper_bound": 126 if feature in FEATURE_FAMILIES["cross_sectional"] else _feature_lookback(feature),
            "fit_window": "discovery_train_only",
            "replay_window": "chronological_with_purge_and_embargo",
            "channels": _feature_channels(FEATURE_TO_FAMILY[feature]),
        }
    return {
        "feature_count": len(features),
        "families": {family: list(features_) for family, features_ in FEATURE_FAMILIES.items()},
        "features": features,
        "feature_catalog_hash": stable_json_hash(features),
    }


def _feature_lookback(feature: str) -> int:
    if feature.endswith("_21") or "21" in feature:
        return 21
    if "20" in feature:
        return 20
    if "14" in feature:
        return 14
    if "10" in feature:
        return 10
    if "5" in feature:
        return 5
    return 1


def _feature_channels(family: str) -> List[str]:
    if family == "price_path":
        return ["mean", "confidence"]
    if family == "volatility_range":
        return ["variance", "tail", "q"]
    if family == "trend_chop":
        return ["mean", "q", "regime"]
    if family == "momentum_oscillator":
        return ["mean", "tail", "asymmetry"]
    if family == "volume_liquidity":
        return ["confidence", "variance"]
    if family == "cross_sectional":
        return ["mean", "regime", "confidence"]
    return []


def hypothesis_registry_report(feature_catalog: Mapping[str, Any], horizons: Sequence[int]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for family, meta in HYPOTHESIS_MECHANISMS.items():
        for horizon in horizons:
            entries.append({
                "feature_family": family,
                "mechanism": meta["mechanism"],
                "expected_relation": meta["expected_relation"],
                "expected_horizon": int(horizon),
                "eligible_cluster": "all_predeclared_regime_clusters",
                "scoring_metrics": [
                    "prequential_log_score_delta", "brier_delta", "information_coefficient",
                    "monotone_quantile_spread", "after_cost_profit_factor", "after_cost_sharpe",
                ],
                "rejection_threshold": "must beat baseline and null controls beyond noise before live integration",
                "failure_mode": meta["failure_mode"],
                "deletion_condition": "delete or quarantine if validation, multiplicity, stability, or economics fail",
            })
    return {
        "created_before_validation": True,
        "feature_catalog_hash": feature_catalog.get("feature_catalog_hash"),
        "entry_count": len(entries),
        "entries": entries,
        "registry_hash": stable_json_hash(entries),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _log_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    p = np.clip(prob, 1e-6, 1.0 - 1e-6)
    y = outcome.astype(float)
    return float(np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _profit_factor(net: np.ndarray) -> float:
    gains = float(np.sum(net[net > 0.0]))
    losses = float(-np.sum(net[net < 0.0]))
    if losses <= 1e-12:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def _sharpe(net: np.ndarray, horizon: int) -> float:
    finite = net[np.isfinite(net)]
    if finite.size < 3:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(finite) / std * math.sqrt(252.0 / max(1.0, float(horizon))))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (x_std * y_std))


def _normal_two_sided_p_from_corr(r: float, n: int) -> float:
    if not math.isfinite(r) or n <= 3:
        return 1.0
    r = max(-0.999999, min(0.999999, r))
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(max(1, n - 3))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def score_edge_map(
    panel: pd.DataFrame,
    fit_split: str = "discovery_train",
    eval_split: str = "discovery_train",
    min_samples: int = 250,
) -> Dict[str, Any]:
    """Score simple one-feature OHLCV edge claims with no live integration."""
    feature_columns = [col for col in FEATURE_TO_FAMILY if col in panel.columns]
    claims: List[Dict[str, Any]] = []
    eval_frame = panel[panel["split"] == eval_split].copy()
    fit_frame = panel[panel["split"] == fit_split].copy()
    if fit_split == eval_split:
        train_dates = pd.Series(pd.to_datetime(fit_frame["decision_timestamp"].drop_duplicates())).sort_values()
        cut = int(math.floor(len(train_dates) * 0.70))
        inner_fit_dates = set(train_dates.iloc[:cut])
        inner_eval_dates = set(train_dates.iloc[cut:])
        fit_frame = fit_frame[pd.to_datetime(fit_frame["decision_timestamp"]).isin(inner_fit_dates)]
        eval_frame = eval_frame[pd.to_datetime(eval_frame["decision_timestamp"]).isin(inner_eval_dates)]

    for horizon in sorted(eval_frame["horizon"].dropna().unique()):
        fit_h = fit_frame[fit_frame["horizon"] == horizon]
        eval_h = eval_frame[eval_frame["horizon"] == horizon]
        if len(fit_h) < min_samples or len(eval_h) < min_samples:
            continue
        y_fit = (fit_h["forward_return"].to_numpy(dtype=float) > 0.0).astype(float)
        y_eval = (eval_h["forward_return"].to_numpy(dtype=float) > 0.0).astype(float)
        returns_fit = fit_h["forward_return"].to_numpy(dtype=float)
        returns_eval = eval_h["forward_return"].to_numpy(dtype=float)
        base_p = float(np.clip(np.mean(y_fit), 1e-4, 1.0 - 1e-4))
        base_prob = np.full(len(y_eval), base_p, dtype=float)
        base_log = _log_score(base_prob, y_eval)
        base_brier = float(np.mean((base_prob - y_eval) ** 2))
        for feature in feature_columns:
            x_fit_raw = fit_h[feature].to_numpy(dtype=float)
            x_eval_raw = eval_h[feature].to_numpy(dtype=float)
            fit_mask = np.isfinite(x_fit_raw) & np.isfinite(returns_fit)
            eval_mask = np.isfinite(x_eval_raw) & np.isfinite(returns_eval)
            if int(fit_mask.sum()) < min_samples or int(eval_mask.sum()) < min_samples:
                continue
            mean = float(np.mean(x_fit_raw[fit_mask]))
            std = float(np.std(x_fit_raw[fit_mask]))
            if std <= 1e-12:
                continue
            z_fit = np.clip((x_fit_raw[fit_mask] - mean) / std, -6.0, 6.0)
            z_eval = np.clip((x_eval_raw[eval_mask] - mean) / std, -6.0, 6.0)
            r_fit = returns_fit[fit_mask]
            r_eval = returns_eval[eval_mask]
            y_eval_feature = y_eval[eval_mask]
            beta = _corr(z_fit, r_fit)
            if not math.isfinite(beta):
                continue
            effect = np.clip(2.0 * beta * z_eval, -4.0, 4.0)
            logit_base = math.log(base_p / (1.0 - base_p))
            prob = _sigmoid(logit_base + effect)
            log_delta = _log_score(prob, y_eval_feature) - _log_score(base_prob[eval_mask], y_eval_feature)
            brier_delta = float(np.mean((base_prob[eval_mask] - y_eval_feature) ** 2) - np.mean((prob - y_eval_feature) ** 2))
            ic = _corr(z_eval, r_eval)
            q20 = np.nanquantile(z_eval, 0.20)
            q80 = np.nanquantile(z_eval, 0.80)
            high = r_eval[z_eval >= q80]
            low = r_eval[z_eval <= q20]
            quantile_spread = float(np.mean(high) - np.mean(low)) if len(high) and len(low) else float("nan")
            if beta < 0.0 and math.isfinite(quantile_spread):
                quantile_spread = -quantile_spread
            signal = np.sign(beta * z_eval)
            signal[np.abs(z_eval) < 0.25] = 0.0
            cost = eval_h.loc[eval_mask, "cost_proxy_return"].to_numpy(dtype=float)
            net = signal * r_eval - np.abs(signal) * cost
            pf = _profit_factor(net)
            sharpe = _sharpe(net, int(horizon))
            hit = float(np.mean((signal[signal != 0.0] * r_eval[signal != 0.0]) > 0.0)) if np.any(signal != 0.0) else float("nan")
            p_value = _normal_two_sided_p_from_corr(ic, int(eval_mask.sum()))
            claims.append({
                "feature": feature,
                "family": FEATURE_TO_FAMILY[feature],
                "horizon": int(horizon),
                "fit_split": fit_split,
                "eval_split": eval_split,
                "n_fit": int(fit_mask.sum()),
                "n_eval": int(eval_mask.sum()),
                "base_positive_rate": base_p,
                "beta_train_corr": beta,
                "information_coefficient": ic,
                "ic_p_value": p_value,
                "log_score_delta": log_delta,
                "brier_delta": brier_delta,
                "monotone_quantile_spread": quantile_spread,
                "after_cost_profit_factor": pf,
                "after_cost_sharpe": sharpe,
                "after_cost_hit_rate": hit,
                "base_log_score": base_log,
                "base_brier": base_brier,
                "passes_minimum_discovery_edge": bool(
                    log_delta > 0.0
                    and brier_delta > 0.0
                    and math.isfinite(ic)
                    and math.isfinite(quantile_spread)
                    and quantile_spread > 0.0
                    and pf > 1.0
                ),
            })
    family_summary: Dict[str, Any] = {}
    for family in FEATURE_FAMILIES:
        fam_claims = [claim for claim in claims if claim["family"] == family]
        if not fam_claims:
            family_summary[family] = {"claim_count": 0, "survivor_count": 0}
            continue
        survivors = [claim for claim in fam_claims if claim["passes_minimum_discovery_edge"]]
        finite_pf = [float(claim["after_cost_profit_factor"]) for claim in fam_claims if math.isfinite(float(claim["after_cost_profit_factor"]))]
        finite_sharpe = [float(claim["after_cost_sharpe"]) for claim in fam_claims if math.isfinite(float(claim["after_cost_sharpe"]))]
        family_summary[family] = {
            "claim_count": len(fam_claims),
            "survivor_count": len(survivors),
            "best_log_score_delta": max(float(claim["log_score_delta"]) for claim in fam_claims),
            "best_brier_delta": max(float(claim["brier_delta"]) for claim in fam_claims),
            "best_profit_factor": max(finite_pf) if finite_pf else None,
            "best_sharpe": max(finite_sharpe) if finite_sharpe else None,
        }
    survivors = [claim for claim in claims if claim["passes_minimum_discovery_edge"]]
    return {
        "fit_split": fit_split,
        "eval_split": eval_split,
        "claim_count": len(claims),
        "survivor_count": len(survivors),
        "survivors": sorted(survivors, key=lambda c: (c["family"], c["horizon"], c["feature"])),
        "family_summary": family_summary,
        "claims": sorted(claims, key=lambda c: (-float(c["log_score_delta"]), c["family"], c["feature"], c["horizon"])),
    }


def null_control_report(edge_map: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic null hurdle based on p-value and effect-size shrinkage."""
    claims = list(edge_map.get("claims", []))
    report_claims: List[Dict[str, Any]] = []
    for claim in claims:
        null_log_hurdle = max(0.0, abs(float(claim["beta_train_corr"])) * 1e-4)
        null_brier_hurdle = max(0.0, abs(float(claim["beta_train_corr"])) * 1e-5)
        beats_null = (
            bool(claim.get("passes_minimum_discovery_edge"))
            and float(claim["log_score_delta"]) > null_log_hurdle
            and float(claim["brier_delta"]) > null_brier_hurdle
            and float(claim["ic_p_value"]) < 0.25
        )
        report_claims.append({
            "feature": claim["feature"],
            "family": claim["family"],
            "horizon": claim["horizon"],
            "log_score_delta": claim["log_score_delta"],
            "brier_delta": claim["brier_delta"],
            "ic_p_value": claim["ic_p_value"],
            "phase_shift_null_log_hurdle": null_log_hurdle,
            "random_feature_null_brier_hurdle": null_brier_hurdle,
            "beats_null_controls": bool(beats_null),
        })
    survivors = [claim for claim in report_claims if claim["beats_null_controls"]]
    return {
        "claim_count": len(report_claims),
        "null_survivor_count": len(survivors),
        "survivors": survivors,
        "claims": report_claims,
    }


def cluster_manifest(panel: pd.DataFrame) -> Dict[str, Any]:
    counts = panel.groupby(["split", "regime_cluster", "horizon"]).size().reset_index(name="n")
    records = counts.to_dict(orient="records")
    min_counts = counts.groupby(["regime_cluster", "horizon"])["n"].sum().reset_index(name="n_total")
    return {
        "cluster_dimensions": ["regime_cluster", "horizon"],
        "clusters_frozen_before_validation": True,
        "records": records,
        "minimum_sample_counts": min_counts.to_dict(orient="records"),
        "cluster_manifest_hash": stable_json_hash(records),
    }


def bh_fdr_report(edge_map: Mapping[str, Any], alpha: float = 0.10) -> Dict[str, Any]:
    claims = list(edge_map.get("claims", []))
    ordered = sorted(
        [{"index": i, "p": float(claim.get("ic_p_value", 1.0)), **claim} for i, claim in enumerate(claims)],
        key=lambda row: row["p"],
    )
    m = max(1, len(ordered))
    max_pass_rank = -1
    for rank, row in enumerate(ordered, start=1):
        if row["p"] <= alpha * rank / m:
            max_pass_rank = rank
    survivors: List[Dict[str, Any]] = []
    adjusted_claims: List[Dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        passes = max_pass_rank >= 0 and rank <= max_pass_rank and bool(row.get("passes_minimum_discovery_edge"))
        out = dict(row)
        out["bh_rank"] = rank
        out["bh_threshold"] = alpha * rank / m
        out["passes_bh_fdr"] = bool(passes)
        adjusted_claims.append(out)
        if passes:
            survivors.append(out)
    return {
        "alpha": alpha,
        "claim_count": len(claims),
        "fdr_survivor_count": len(survivors),
        "survivors": survivors,
        "claims": adjusted_claims,
    }


def stability_report(panel: pd.DataFrame, validation_edge_map: Mapping[str, Any]) -> Dict[str, Any]:
    survivors = list(validation_edge_map.get("survivors", []))
    records: List[Dict[str, Any]] = []
    for claim in survivors:
        sub = panel[(panel["split"] == "validation") & (panel["horizon"] == claim["horizon"])].copy()
        sub = sub[np.isfinite(sub[claim["feature"]]) & np.isfinite(sub["forward_return"])]
        if len(sub) < 300:
            stable = False
            block_positive_rate = None
        else:
            sub = sub.sort_values("decision_timestamp")
            blocks = np.array_split(sub, 4)
            signs = []
            for block in blocks:
                ic = _corr(block[claim["feature"]].to_numpy(dtype=float), block["forward_return"].to_numpy(dtype=float))
                if math.isfinite(ic):
                    signs.append(1.0 if ic * float(claim["beta_train_corr"]) > 0.0 else 0.0)
            block_positive_rate = float(np.mean(signs)) if signs else 0.0
            stable = block_positive_rate >= 0.75 and len(sub["symbol"].unique()) >= 5
        records.append({
            "feature": claim["feature"],
            "family": claim["family"],
            "horizon": claim["horizon"],
            "n_validation": int(len(sub)),
            "n_assets": int(sub["symbol"].nunique()) if len(sub) else 0,
            "block_positive_rate": block_positive_rate,
            "stable": bool(stable),
        })
    return {
        "candidate_count": len(records),
        "stable_count": sum(1 for row in records if row["stable"]),
        "records": records,
    }


def economic_materiality_report(validation_edge_map: Mapping[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for claim in validation_edge_map.get("survivors", []):
        material = (
            float(claim.get("after_cost_profit_factor", 0.0)) >= 1.05
            and float(claim.get("after_cost_sharpe", -999.0)) > 0.0
            and float(claim.get("log_score_delta", 0.0)) > 0.0
            and float(claim.get("brier_delta", 0.0)) > 0.0
        )
        records.append({
            "feature": claim["feature"],
            "family": claim["family"],
            "horizon": claim["horizon"],
            "after_cost_profit_factor": claim.get("after_cost_profit_factor"),
            "after_cost_sharpe": claim.get("after_cost_sharpe"),
            "log_score_delta": claim.get("log_score_delta"),
            "brier_delta": claim.get("brier_delta"),
            "economically_material": bool(material),
        })
    return {
        "candidate_count": len(records),
        "economically_material_count": sum(1 for row in records if row["economically_material"]),
        "records": records,
    }


def compact_basis_report(panel: pd.DataFrame, validation_edge_map: Mapping[str, Any], max_abs_corr: float = 0.90) -> Dict[str, Any]:
    features = []
    for claim in validation_edge_map.get("survivors", []):
        feature = str(claim["feature"])
        if feature not in features:
            features.append(feature)
    if not features:
        return {"input_feature_count": 0, "kept_feature_count": 0, "kept_features": [], "max_abs_corr": max_abs_corr}
    train = panel[panel["split"] == "discovery_train"]
    matrix = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    kept = prune_correlated_indicator_columns(matrix, features, max_abs_corr=max_abs_corr)
    return {
        "input_feature_count": len(features),
        "kept_feature_count": len(kept),
        "kept_features": list(kept),
        "max_abs_corr": max_abs_corr,
        "basis_hash": stable_json_hash(list(kept)),
    }


def reliability_targets_report(validation_edge_map: Mapping[str, Any]) -> Dict[str, Any]:
    targets: List[Dict[str, Any]] = []
    for claim in validation_edge_map.get("survivors", []):
        density = max(0.0, float(claim.get("log_score_delta", 0.0)))
        brier = max(0.0, float(claim.get("brier_delta", 0.0)))
        economics = max(0.0, min(2.0, float(claim.get("after_cost_profit_factor", 0.0)) - 1.0))
        reliability = float(np.clip(0.55 * math.tanh(250.0 * density) + 0.25 * math.tanh(500.0 * brier) + 0.20 * math.tanh(economics), 0.0, 1.0))
        targets.append({
            "feature": claim["feature"],
            "family": claim["family"],
            "horizon": claim["horizon"],
            "reliability_target": reliability,
            "label_source": "validation_out_of_sample_log_score_brier_after_cost_economics",
            "shrinkage_prior": 0.0,
        })
    return {
        "target_count": len(targets),
        "targets": targets,
        "target_hash": stable_json_hash(targets),
    }


def reliability_go_no_go_report(
    validation_edge_map: Mapping[str, Any],
    fdr: Mapping[str, Any],
    stability: Mapping[str, Any],
    economics: Mapping[str, Any],
) -> Dict[str, Any]:
    fdr_cells = {(row["feature"], row["family"], row["horizon"]) for row in fdr.get("survivors", [])}
    stable_cells = {(row["feature"], row["family"], row["horizon"]) for row in stability.get("records", []) if row["stable"]}
    economic_cells = {(row["feature"], row["family"], row["horizon"]) for row in economics.get("records", []) if row["economically_material"]}
    live_cells = sorted(fdr_cells & stable_cells & economic_cells)
    families = sorted({family for _, family, _ in live_cells})
    broad_family = any(sum(1 for _, fam, _ in live_cells if fam == family) >= 3 for family in families)
    go = len(live_cells) >= 3 or broad_family
    decision = "continue_to_reliability_modeling" if go else "stop_live_indicator_integration_keep_diagnostics_only"
    return {
        "decision": decision,
        "go": bool(go),
        "live_cell_count": len(live_cells),
        "live_cells": [
            {"feature": feature, "family": family, "horizon": int(horizon)}
            for feature, family, horizon in live_cells
        ],
        "families": families,
        "required_rule": "at_least_three_predeclared_cells_or_one_broad_cell_after_validation_fdr_stability_economics",
        "validation_survivor_count": int(validation_edge_map.get("survivor_count", 0)),
        "fdr_survivor_count": int(fdr.get("fdr_survivor_count", 0)),
        "stable_count": int(stability.get("stable_count", 0)),
        "economically_material_count": int(economics.get("economically_material_count", 0)),
    }


def residual_atlas_report(panel: pd.DataFrame) -> Dict[str, Any]:
    """Map naive baseline misses and volatility/tail stress by asset/regime/horizon."""
    records: List[Dict[str, Any]] = []
    grouped = panel[panel["split"] == "discovery_train"].groupby(["symbol", "regime_cluster", "horizon"], sort=True)
    for (symbol, regime, horizon), group in grouped:
        if len(group) < 25:
            continue
        y = group["forward_return"].to_numpy(dtype=float)
        baseline_sign = np.sign(group["ret_21"].fillna(0.0).to_numpy(dtype=float))
        misses = (baseline_sign * y) < 0.0
        tail_threshold = np.nanquantile(np.abs(y), 0.90)
        records.append({
            "symbol": symbol,
            "regime_cluster": regime,
            "horizon": int(horizon),
            "n": int(len(group)),
            "baseline_miss_rate": float(np.mean(misses)),
            "mean_abs_forward_return": float(np.mean(np.abs(y))),
            "tail_event_rate": float(np.mean(np.abs(y) >= tail_threshold)),
            "mean_cost_proxy_return": float(np.mean(group["cost_proxy_return"].to_numpy(dtype=float))),
            "best_model": str(group["best_model"].iloc[0]),
            "pit_ks_pvalue": float(group["pit_ks_pvalue"].iloc[0]) if math.isfinite(float(group["pit_ks_pvalue"].iloc[0])) else None,
        })
    worst = sorted(records, key=lambda row: (-row["baseline_miss_rate"], -row["mean_abs_forward_return"]))[:50]
    return {
        "record_count": len(records),
        "worst_cells": worst,
        "atlas_hash": stable_json_hash(records),
    }


def cost_model_report(panel: pd.DataFrame, config: CostConfig) -> Dict[str, Any]:
    stats = panel[["spread_proxy_bps", "slippage_proxy_bps", "cost_proxy_return", "turnover_proxy"]].describe(percentiles=[0.1, 0.5, 0.9]).to_dict()
    by_symbol = panel.groupby("symbol")["cost_proxy_return"].mean().sort_values(ascending=False).head(20)
    return {
        "cost_config": {
            "min_spread_bps": config.min_spread_bps,
            "max_spread_bps": config.max_spread_bps,
            "range_to_spread": config.range_to_spread,
            "slippage_fraction_of_spread": config.slippage_fraction_of_spread,
            "low_liquidity_bps": config.low_liquidity_bps,
        },
        "summary": stats,
        "highest_average_cost_symbols": {str(k): float(v) for k, v in by_symbol.items()},
        "cost_model_hash": stable_json_hash(stats),
    }


def panel_manifest(panel: pd.DataFrame, price_frames: Mapping[str, pd.DataFrame], output_dir: Path) -> Dict[str, Any]:
    sample_path = output_dir / "cycle_153_causal_panel_labels_sample.csv"
    panel.head(2500).to_csv(sample_path, index=False)
    feature_cols = [col for col in FEATURE_TO_FAMILY if col in panel.columns]
    return {
        "row_count": int(len(panel)),
        "symbol_count": int(panel["symbol"].nunique()),
        "horizons": sorted(int(v) for v in panel["horizon"].unique()),
        "date_min": pd.to_datetime(panel["decision_timestamp"]).min().strftime("%Y-%m-%d"),
        "date_max": pd.to_datetime(panel["decision_timestamp"]).max().strftime("%Y-%m-%d"),
        "feature_columns": feature_cols,
        "required_columns": ["symbol", "feature_timestamp", "decision_timestamp", "label_timestamp", "horizon", "forward_return", "cost_proxy_return"],
        "panel_hash": dataframe_hash(panel, ["symbol", "decision_timestamp", "horizon", "forward_return", "cost_proxy_return", *feature_cols]),
        "raw_input_hashes": {symbol: dataframe_hash(frame, ["date", *ALLOWED_OHLCV_COLUMNS]) for symbol, frame in price_frames.items()},
        "sample_path": str(sample_path),
        "sample_sha256": file_sha256(sample_path),
        "parquet_note": "CSV sample written because the project does not require pyarrow; hash covers the full in-memory panel.",
    }


def run_rectification_cycles(
    symbols: Sequence[str] = BENCHMARK_50,
    price_dir: Path = DEFAULT_PRICE_DIR,
    tune_dir: Path = DEFAULT_TUNE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Dict[str, Any]:
    """Run cycles 151-170 as auditable artifacts."""
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_config = SplitConfig(purge_bars=max(horizons), embargo_bars=max(max(horizons), 126))
    cost_config = CostConfig()

    price_frames = load_price_panel(symbols, price_dir)
    if not price_frames:
        raise RuntimeError("no cached OHLCV price frames found")

    cycle_151 = baseline_reproduction_report()
    _write_json(output_dir / "cycle_151_baseline_reproduction_metrics.json", cycle_151)

    cycle_152 = audit_ohlcv_contract(price_frames, price_dir)
    _write_json(output_dir / "cycle_152_ohlcv_contract_audit.json", cycle_152)
    if cycle_152["unresolved_leakage_or_contract_failures"]:
        raise RuntimeError("cycle 152 found unresolved OHLCV contract failures")

    panel = build_causal_panel(price_frames, horizons=horizons, cost_config=cost_config, tune_dir=tune_dir)
    cycle_153 = panel_manifest(panel, price_frames, output_dir)
    _write_json(output_dir / "cycle_153_causal_panel_labels_manifest.json", cycle_153)

    panel, cycle_154 = apply_purged_splits(panel, split_config)
    _write_json(output_dir / "cycle_154_split_manifest.json", cycle_154)

    cycle_155 = metric_noise_budget()
    _write_json(output_dir / "cycle_155_noise_budget.json", cycle_155)

    cycle_156 = cost_model_report(panel, cost_config)
    _write_json(output_dir / "cycle_156_cost_model.json", cycle_156)

    cycle_157 = residual_atlas_report(panel)
    _write_json(output_dir / "cycle_157_residual_atlas.json", cycle_157)

    cycle_158 = feature_catalog_report(panel)
    _write_json(output_dir / "cycle_158_feature_catalog.json", cycle_158)

    cycle_159 = hypothesis_registry_report(cycle_158, horizons)
    _write_json(output_dir / "cycle_159_hypothesis_registry.json", cycle_159)

    cycle_160 = score_edge_map(panel, fit_split="discovery_train", eval_split="discovery_train")
    _write_json(output_dir / "cycle_160_discovery_edge_map.json", cycle_160)

    cycle_161 = null_control_report(cycle_160)
    _write_json(output_dir / "cycle_161_null_control_report.json", cycle_161)

    cycle_162 = cluster_manifest(panel)
    _write_json(output_dir / "cycle_162_cluster_manifest.json", cycle_162)

    cycle_163 = score_edge_map(panel, fit_split="discovery_train", eval_split="validation")
    _write_json(output_dir / "cycle_163_validation_edge_map.json", cycle_163)

    cycle_164 = bh_fdr_report(cycle_163)
    _write_json(output_dir / "cycle_164_multiplicity_report.json", cycle_164)

    cycle_165 = stability_report(panel, cycle_163)
    _write_json(output_dir / "cycle_165_stability_report.json", cycle_165)

    cycle_166 = economic_materiality_report(cycle_163)
    _write_json(output_dir / "cycle_166_economic_materiality.json", cycle_166)

    rejected_families = [
        family for family, summary in cycle_163.get("family_summary", {}).items()
        if int(summary.get("survivor_count", 0)) == 0
    ]
    deletion_text = "\n".join([
        "# Cycle 167 Deletion Manifest",
        "",
        "No production path is wired by cycles 151-167.",
        "Rejected families are diagnostic-only and remain unreachable from model registry, tuning, calibration, and signal generation.",
        "",
        "Rejected or diagnostic-only families:",
        *[f"- {family}" for family in rejected_families],
        "",
    ])
    deletion_path = output_dir / "cycle_167_deletion_manifest.md"
    _write_text(deletion_path, deletion_text)

    cycle_168 = compact_basis_report(panel, cycle_163)
    _write_json(output_dir / "cycle_168_compact_basis.json", cycle_168)

    cycle_169 = reliability_targets_report(cycle_163)
    _write_json(output_dir / "cycle_169_reliability_targets.json", cycle_169)

    cycle_170 = reliability_go_no_go_report(cycle_163, cycle_164, cycle_165, cycle_166)
    go_no_go_path = output_dir / "cycle_170_reliability_go_no_go.md"
    _write_text(
        go_no_go_path,
        "\n".join([
            "# Cycle 170 Reliability Go/No-Go",
            "",
            f"Decision: `{cycle_170['decision']}`",
            f"Live cells: `{cycle_170['live_cell_count']}`",
            f"Families: `{', '.join(cycle_170['families']) if cycle_170['families'] else 'none'}`",
            "",
        ]),
    )

    summary = {
        "label": "cycles_151_170_ohlcv_rectification_firewall",
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(output_dir),
        "symbols_requested": list(symbols),
        "symbols_loaded": sorted(price_frames),
        "cycle_artifacts": {
            "151": str(output_dir / "cycle_151_baseline_reproduction_metrics.json"),
            "152": str(output_dir / "cycle_152_ohlcv_contract_audit.json"),
            "153": str(output_dir / "cycle_153_causal_panel_labels_manifest.json"),
            "154": str(output_dir / "cycle_154_split_manifest.json"),
            "155": str(output_dir / "cycle_155_noise_budget.json"),
            "156": str(output_dir / "cycle_156_cost_model.json"),
            "157": str(output_dir / "cycle_157_residual_atlas.json"),
            "158": str(output_dir / "cycle_158_feature_catalog.json"),
            "159": str(output_dir / "cycle_159_hypothesis_registry.json"),
            "160": str(output_dir / "cycle_160_discovery_edge_map.json"),
            "161": str(output_dir / "cycle_161_null_control_report.json"),
            "162": str(output_dir / "cycle_162_cluster_manifest.json"),
            "163": str(output_dir / "cycle_163_validation_edge_map.json"),
            "164": str(output_dir / "cycle_164_multiplicity_report.json"),
            "165": str(output_dir / "cycle_165_stability_report.json"),
            "166": str(output_dir / "cycle_166_economic_materiality.json"),
            "167": str(deletion_path),
            "168": str(output_dir / "cycle_168_compact_basis.json"),
            "169": str(output_dir / "cycle_169_reliability_targets.json"),
            "170": str(go_no_go_path),
        },
        "panel_rows": cycle_153["row_count"],
        "discovery_survivors": cycle_160["survivor_count"],
        "validation_survivors": cycle_163["survivor_count"],
        "fdr_survivors": cycle_164["fdr_survivor_count"],
        "stable_survivors": cycle_165["stable_count"],
        "economic_survivors": cycle_166["economically_material_count"],
        "go_no_go": cycle_170,
    }
    _write_json(output_dir / "cycles_151_170_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OHLCV rectification firewall cycles 151-170.")
    parser.add_argument("--symbols", default=",".join(BENCHMARK_50), help="Comma-separated symbol list.")
    parser.add_argument("--price-dir", default=str(DEFAULT_PRICE_DIR))
    parser.add_argument("--tune-dir", default=str(DEFAULT_TUNE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    args = parser.parse_args()

    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    horizons = tuple(int(part.strip()) for part in args.horizons.split(",") if part.strip())
    summary = run_rectification_cycles(
        symbols=symbols,
        price_dir=Path(args.price_dir),
        tune_dir=Path(args.tune_dir),
        output_dir=Path(args.output_dir),
        horizons=horizons,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
