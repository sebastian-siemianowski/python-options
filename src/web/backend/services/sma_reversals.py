"""
SMA Reversal Detection Service — Elite Buy-Signal Quality Engine
=================================================================

Answers the question: *"Should I buy this stock right now, and if so, at
what size and with what expectation?"*

This module is a quant research engine, not a moving-average screener.
For every fresh cross of close vs. SMA (9 / 50 / 600) we compute:

**Regime / filters**
1. **Trend regime** — SMA-200 level *and slope*. Being above a falling
   SMA-200 is not a bull regime; it is a dead cat.
2. **Volatility regime** — realized-vol ratio (20d / 252d). Extreme vol
   dampens the score because mean-reversion edge decays in panics.
3. **Overextension** — |price − SMA| / ATR > 3 is a chase.
4. **Persistence** — K of last M closes on the new side (whipsaw reject).
5. **False break** — re-cross within 3 bars ⇒ 0.6× score penalty.

**Trade geometry (makes R:R informative)**
6. **Stop** — ``min(10-bar swing low, entry − 2·ATR)`` (bull).
7. **Structural target** — prior 50-bar swing high (bull) / low (bear)
   when meaningfully distant; otherwise 2R projection. R:R therefore
   varies per setup.
8. **Pullback quality** — distance (in ATRs) from the 20-bar peak when
   the cross fired. Healthy reversals come on a moderate pullback, not
   after a deep drawdown.
9. **Multi-timeframe alignment** — for each cross, how many of the
   *other* two SMAs confirm the direction.

**Honest historical edge (per-symbol backtest)**
10. Walk the symbol's own history, find every past crossing of this
    same period / direction. Crucially, we **condition on the same
    regime + overextension filters** we use today — so the edge
    reported is the true edge of *this grade of setup*, not a noisy
    unconditional average. We also report:
    - ``expectancy_r`` (mean return in 2-ATR-stop units — THE metric)
    - ``profit_factor`` (Σwins / |Σlosses|)
    - ``median_mae_atr`` (typical adverse excursion — stop survival)
    - ``stop_hit_rate`` (fraction of past winners that dipped through
      the 2-ATR stop before resolving)
    - ``recency_weighted_win_rate`` (2-year half-life — markets change)

**Sizing**
11. **Kelly fraction** — ``p − (1−p)/R``, capped at 25%.

**Composite score & grade**
12. Score is a weighted blend of seven components, dampened by the vol
    regime and boosted by MTF alignment.
13. **Grade A** — the elite gate:
    - regime_ok (price AND slope aligned)
    - persistence_ok AND !false_break AND !overextended
    - R:R ≥ 2.0
    - win_rate ≥ 55% (n ≥ 5)
    - expectancy_r ≥ 0.20
    - stop_hit_rate ≤ 0.45 (stop survives historically)
14. **Grade B** — tradeable: same base gates + R:R ≥ 1.5.
15. **Grade C** — watch-list: passes persistence.

Cached in-memory; invalidated when any underlying CSV mtime changes.
"""

from __future__ import annotations

import glob
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from web.backend.services.data_service import PRICES_DIR

# ── Tunable constants ─────────────────────────────────────────────────────
_SMA_PERIODS: Tuple[int, ...] = (9, 50, 600)
_LOOKBACK_BARS = 5
_PERSISTENCE_M = 3
_PERSISTENCE_K = 2
_FALSE_BREAK_WINDOW = 3
_ATR_PERIOD = 14
_VOLUME_BASELINE = 20
_SLOPE_WINDOW = 5

# Regime
_REGIME_PERIOD = 200
_REGIME_SLOPE_WINDOW = 20
_OVEREXTENDED_ATR = 3.0

# Volatility regime
_VOL_REGIME_WINDOW = 20
_VOL_REGIME_BASELINE = 252
_VOL_REGIME_HIGH = 1.30
_VOL_REGIME_EXTREME = 1.80

# Trade geometry
_STOP_ATR_MULT = 2.0
_TARGET_R_MULT = 2.0
_SWING_LOOKBACK = 10
_STRUCTURAL_TARGET_WINDOW = 50
_STRUCTURAL_TARGET_MIN_ATR = 0.5  # target must be >= this many ATRs away
_PULLBACK_WINDOW = 20

# Historical edge — horizon matched to SMA period
_EDGE_FORWARD_BY_PERIOD: Dict[int, int] = {9: 10, 50: 20, 600: 60}
_EDGE_DEFAULT_FORWARD = 10
_EDGE_MIN_SAMPLES = 5
_EDGE_HALF_LIFE_BARS = 504  # ~2 years

# Sizing
_KELLY_CAP = 0.25

# Grade gates
_GRADE_A_RR = 2.0
_GRADE_A_WINRATE = 0.55
_GRADE_A_EXPECTANCY_R = 0.20
_GRADE_A_STOP_HIT_MAX = 0.45
_GRADE_B_RR = 1.5

# Score weights (sum to 1.0)
_W_DISTANCE = 0.30
_W_SLOPE = 0.20
_W_VOLUME = 0.20
_W_PERSISTENCE = 0.15
_W_FRESHNESS = 0.15

# Score modifiers
_VOL_SCORE_MULT = {"normal": 1.0, "high": 0.9, "extreme": 0.7, "unknown": 1.0}
_MTF_SCORE_FLOOR = 0.95  # at alignment=0 we multiply by 0.95
_MTF_SCORE_CEIL = 1.05  # at alignment=1.0 we multiply by 1.05

_CACHE_TTL_SEC = 300

_cache: Dict[str, Any] = {
    "built_at": 0.0,
    "mtime_signature": None,
    "reversals": [],
    "counts": {},
}


# ── util ──────────────────────────────────────────────────────────────────

def _signature() -> Optional[float]:
    if not os.path.isdir(PRICES_DIR):
        return None
    sig = 0.0
    try:
        for fp in glob.iglob(os.path.join(PRICES_DIR, "*.csv")):
            try:
                sig += os.path.getmtime(fp)
            except OSError:
                continue
    except OSError:
        return None
    return sig


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _symbol_for(filepath: str) -> str:
    fname = os.path.basename(filepath)
    if fname.endswith("_1d.csv"):
        return fname[:-7]
    if fname.endswith(".csv"):
        return fname[:-4]
    return fname


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _series_to_list(s: Any) -> List[Optional[float]]:
    """Convert pandas Series to list[float|None], mapping NaN/Inf -> None."""
    out: List[Optional[float]] = []
    for v in s.tolist():
        if v is None:
            out.append(None)
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if math.isnan(fv) or math.isinf(fv):
            out.append(None)
        else:
            out.append(fv)
    return out


def _median_sorted(xs: Sequence[float]) -> float:
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


# ── Edge statistics (honest per-symbol backtest) ──────────────────────────

def _historical_edge(
    close: List[Optional[float]],
    sma: List[Optional[float]],
    atr: List[Optional[float]],
    regime: List[Optional[float]],
    direction: str,
    forward: int = _EDGE_DEFAULT_FORWARD,
    stop_mult: float = _STOP_ATR_MULT,
    overextended_atr: float = _OVEREXTENDED_ATR,
) -> Dict[str, Any]:
    """
    Walk the symbol's own history. For every past SMA crossing that
    would *also* have passed our regime + overextension filters, compute:
        - forward return over `forward` bars
        - R-multiple return assuming a `stop_mult`×ATR stop
        - maximum adverse excursion (in ATRs)
        - whether the 2-ATR stop was hit during the forward window

    Returns a dict with conditional stats (the true edge of this
    grade of setup) and an `unconditional_samples` count for context.
    """
    empty: Dict[str, Any] = {
        "samples": 0,
        "unconditional_samples": 0,
        "win_rate": None,
        "median_fwd_pct": None,
        "mean_fwd_pct": None,
        "std_fwd_pct": None,
        "expectancy_r": None,
        "profit_factor": None,
        "median_mae_atr": None,
        "stop_hit_rate": None,
        "recency_weighted_win_rate": None,
    }

    n = len(close)
    if n < forward + 2 or len(sma) != n or len(atr) != n or len(regime) != n:
        return empty

    rows: List[Dict[str, Any]] = []
    unconditional_samples = 0

    for i in range(1, n - forward):
        s_prev, s_cur = sma[i - 1], sma[i]
        c_prev, c_cur = close[i - 1], close[i]
        if s_prev is None or s_cur is None or c_prev is None or c_cur is None:
            continue

        prev_sign = 1 if c_prev > s_prev else (-1 if c_prev < s_prev else 0)
        cur_sign = 1 if c_cur > s_cur else (-1 if c_cur < s_cur else 0)

        if direction == "bull":
            if not (prev_sign <= 0 and cur_sign > 0):
                continue
        else:
            if not (prev_sign >= 0 and cur_sign < 0):
                continue

        c0 = c_cur
        c_end = close[i + forward]
        if c_end is None or c0 <= 0:
            continue

        unconditional_samples += 1

        # Conditional filters — same as live-setup gates
        r0 = regime[i]
        a0 = atr[i]
        if r0 is None or a0 is None or a0 <= 0:
            continue
        if direction == "bull" and c0 <= r0:
            continue
        if direction == "bear" and c0 >= r0:
            continue
        if abs((c0 - s_cur) / a0) > overextended_atr:
            continue

        # Forward window
        window = close[i + 1 : i + 1 + forward]
        if any(x is None for x in window):
            continue
        # Safe because we just verified non-None
        win_vals: List[float] = [x for x in window if x is not None]
        if len(win_vals) != forward:
            continue

        fwd_pct = (c_end - c0) / c0 * 100.0
        stop_dist = stop_mult * a0
        if stop_dist <= 0:
            continue

        if direction == "bull":
            worst = min([c0] + win_vals)
            mae = (c0 - worst) / a0  # >= 0
            r_mult = (c_end - c0) / stop_dist
        else:
            best = max([c0] + win_vals)
            mae = (best - c0) / a0
            r_mult = (c0 - c_end) / stop_dist

        stop_hit = mae >= stop_mult

        rows.append({
            "age_bars": n - 1 - i,
            "fwd_pct": fwd_pct,
            "r_mult": r_mult,
            "mae": mae,
            "stop_hit": stop_hit,
        })

    cs = len(rows)
    if cs < _EDGE_MIN_SAMPLES:
        out = dict(empty)
        out["samples"] = cs
        out["unconditional_samples"] = unconditional_samples
        return out

    # Win definition: positive return for bull, negative for bear
    if direction == "bull":
        wins = sum(1 for r in rows if r["fwd_pct"] > 0)
    else:
        wins = sum(1 for r in rows if r["fwd_pct"] < 0)
    win_rate = wins / cs

    r_mults = [r["r_mult"] for r in rows]
    expectancy_r = sum(r_mults) / cs

    pos_sum = sum(v for v in r_mults if v > 0)
    neg_sum = sum(v for v in r_mults if v < 0)
    if neg_sum < 0:
        profit_factor: Optional[float] = pos_sum / abs(neg_sum)
    else:
        profit_factor = None

    fwd_pct_list = [r["fwd_pct"] for r in rows]
    sorted_fwd = sorted(fwd_pct_list)
    median_fwd = _median_sorted(sorted_fwd)
    mean_fwd = sum(fwd_pct_list) / cs
    var = sum((x - mean_fwd) ** 2 for x in fwd_pct_list) / cs
    std_fwd = math.sqrt(var) if var > 0 else 0.0

    mae_list = sorted(r["mae"] for r in rows)
    median_mae = _median_sorted(mae_list)

    stop_hit_rate = sum(1 for r in rows if r["stop_hit"]) / cs

    # Recency-weighted win-rate — 2-year half-life
    weights = [0.5 ** (r["age_bars"] / _EDGE_HALF_LIFE_BARS) for r in rows]
    tot_w = sum(weights)
    if tot_w > 0:
        if direction == "bull":
            rw_win = sum(w for r, w in zip(rows, weights) if r["fwd_pct"] > 0)
        else:
            rw_win = sum(w for r, w in zip(rows, weights) if r["fwd_pct"] < 0)
        rw_win_rate: Optional[float] = rw_win / tot_w
    else:
        rw_win_rate = None

    return {
        "samples": cs,
        "unconditional_samples": unconditional_samples,
        "win_rate": round(win_rate, 3),
        "median_fwd_pct": round(median_fwd, 3),
        "mean_fwd_pct": round(mean_fwd, 3),
        "std_fwd_pct": round(std_fwd, 3),
        "expectancy_r": round(expectancy_r, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "median_mae_atr": round(median_mae, 3),
        "stop_hit_rate": round(stop_hit_rate, 3),
        "recency_weighted_win_rate": round(rw_win_rate, 3) if rw_win_rate is not None else None,
    }


# ── Regime / volatility helpers ───────────────────────────────────────────

def _vol_regime(close_series: Any) -> Tuple[Optional[float], str]:
    """Return (ratio_20d_over_252d, regime_str). Needs >= _VOL_REGIME_BASELINE bars."""
    import pandas as pd
    s = pd.Series(close_series).astype(float)
    rets = s.pct_change().dropna()
    if len(rets) < _VOL_REGIME_BASELINE:
        return None, "unknown"
    rv_short = rets.iloc[-_VOL_REGIME_WINDOW:].std()
    rv_long = rets.iloc[-_VOL_REGIME_BASELINE:].std()
    if rv_long is None or rv_long == 0 or not math.isfinite(float(rv_long)):
        return None, "unknown"
    if rv_short is None or not math.isfinite(float(rv_short)):
        return None, "unknown"
    ratio = float(rv_short) / float(rv_long)
    if ratio >= _VOL_REGIME_EXTREME:
        return ratio, "extreme"
    if ratio >= _VOL_REGIME_HIGH:
        return ratio, "high"
    return ratio, "normal"


def _regime_slope_pct(regime_sma_series: Any) -> float:
    """SMA-200 slope as % change over _REGIME_SLOPE_WINDOW bars. 0 if insufficient."""
    now = _safe_float(regime_sma_series.iloc[-1])
    if now is None or len(regime_sma_series) <= _REGIME_SLOPE_WINDOW:
        return 0.0
    ago = _safe_float(regime_sma_series.iloc[-1 - _REGIME_SLOPE_WINDOW])
    if ago is None or ago == 0.0:
        return 0.0
    return (now - ago) / abs(ago) * 100.0


def _pullback_atr(
    close_series: Any, cross_idx: int, atr_val: Optional[float], direction: str,
) -> Optional[float]:
    """Distance in ATRs from the 20-bar peak (bull) / trough (bear) at cross_idx."""
    if atr_val is None or atr_val <= 0 or cross_idx < 1:
        return None
    start = max(0, cross_idx - _PULLBACK_WINDOW + 1)
    window = close_series.iloc[start : cross_idx + 1]
    if len(window) == 0:
        return None
    c0 = _safe_float(close_series.iloc[cross_idx])
    if c0 is None:
        return None
    if direction == "bull":
        peak = _safe_float(window.max())
        if peak is None:
            return None
        return max(0.0, (peak - c0) / atr_val)
    trough = _safe_float(window.min())
    if trough is None:
        return None
    return max(0.0, (c0 - trough) / atr_val)


def _mtf_alignment(
    close_series: Any,
    latest_price: float,
    direction: str,
    all_periods: Sequence[int],
    current_period: int,
) -> Optional[float]:
    """Fraction of *other* SMAs whose price relationship confirms direction."""
    n = len(close_series)
    confirms = 0
    checks = 0
    for p in all_periods:
        if p == current_period or n < p:
            continue
        sma_now = _safe_float(close_series.rolling(p).mean().iloc[-1])
        if sma_now is None:
            continue
        checks += 1
        if direction == "bull" and latest_price > sma_now:
            confirms += 1
        elif direction == "bear" and latest_price < sma_now:
            confirms += 1
    if checks == 0:
        return None
    return confirms / checks


def _structural_target(
    close_series: Any,
    high_series: Any,
    low_series: Any,
    latest_price: float,
    stop_price: float,
    atr_val: float,
    direction: str,
) -> Tuple[float, bool]:
    """Return (target_price, used_structural_level).

    Structural level = 50-bar prior swing high (bull) or low (bear),
    used only if meaningfully distant (>= 0.5*ATR beyond entry).
    Otherwise fall back to 2R projection.
    """
    risk = abs(latest_price - stop_price)
    fallback = (latest_price + _TARGET_R_MULT * risk) if direction == "bull" \
        else (latest_price - _TARGET_R_MULT * risk)
    import pandas as pd
    window = _STRUCTURAL_TARGET_WINDOW
    # Exclude the current bar (don't use today's high/low as resistance)
    if direction == "bull":
        src = high_series if high_series is not None else close_series
        src = pd.Series(src).astype(float)
        if len(src) < 2:
            return fallback, False
        level = _safe_float(src.iloc[-window - 1 : -1].max()) if len(src) > 1 else None
        if level is None:
            return fallback, False
        if level > latest_price + _STRUCTURAL_TARGET_MIN_ATR * atr_val:
            return level, True
        return fallback, False
    src = low_series if low_series is not None else close_series
    src = pd.Series(src).astype(float)
    if len(src) < 2:
        return fallback, False
    level = _safe_float(src.iloc[-window - 1 : -1].min()) if len(src) > 1 else None
    if level is None:
        return fallback, False
    if level < latest_price - _STRUCTURAL_TARGET_MIN_ATR * atr_val:
        return level, True
    return fallback, False


# ── Detection ─────────────────────────────────────────────────────────────

def detect_reversals(
    close: "Any",
    high: "Optional[Any]" = None,
    low: "Optional[Any]" = None,
    volume: "Optional[Any]" = None,
    dates: "Optional[Any]" = None,
    periods: Tuple[int, ...] = _SMA_PERIODS,
) -> List[Dict[str, Any]]:
    """Pure detection function — returns one record per SMA period with a
    fresh cross in the last `_LOOKBACK_BARS` bars."""
    import pandas as pd

    close = pd.Series(close).astype(float).reset_index(drop=True)
    n = len(close)
    if n < 20:
        return []

    out: List[Dict[str, Any]] = []

    # ATR
    if high is not None and low is not None:
        h = pd.Series(high).astype(float).reset_index(drop=True)
        l = pd.Series(low).astype(float).reset_index(drop=True)
        prev_c = close.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(_ATR_PERIOD, min_periods=max(2, _ATR_PERIOD // 2)).mean()
    else:
        h = None
        l = None
        atr = close.pct_change().rolling(_ATR_PERIOD, min_periods=5).std() * close

    # Volume baseline
    if volume is not None:
        v = pd.Series(volume).astype(float).reset_index(drop=True)
        v_avg = v.rolling(_VOLUME_BASELINE, min_periods=5).mean()
    else:
        v = None
        v_avg = None

    # Regime SMA
    reg_period = min(_REGIME_PERIOD, max(20, n // 2))
    regime_sma = close.rolling(reg_period).mean()
    regime_slope_pct = _regime_slope_pct(regime_sma)

    # Vol regime
    vol_ratio, vol_regime_str = _vol_regime(close)

    # Swing levels for structural stops
    swing_low = (l if l is not None else close).rolling(_SWING_LOOKBACK).min()
    swing_high = (h if h is not None else close).rolling(_SWING_LOOKBACK).max()

    latest_idx = n - 1
    latest_price = _safe_float(close.iloc[-1])
    if latest_price is None:
        return []

    regime_now = _safe_float(regime_sma.iloc[-1])
    atr_val = _safe_float(atr.iloc[-1])
    swing_low_val = _safe_float(swing_low.iloc[-1])
    swing_high_val = _safe_float(swing_high.iloc[-1])

    # Precompute list forms once per symbol for _historical_edge
    close_list = _series_to_list(close)
    atr_list = _series_to_list(atr)
    regime_list = _series_to_list(regime_sma)

    for period in periods:
        if n < period + _LOOKBACK_BARS + 1:
            continue
        sma = close.rolling(period).mean()
        diff = close - sma

        def _sgn(x: Any) -> int:
            if x is None:
                return 0
            try:
                fx = float(x)
            except (TypeError, ValueError):
                return 0
            if math.isnan(fx):
                return 0
            return 1 if fx > 0 else (-1 if fx < 0 else 0)

        sign = diff.apply(_sgn)

        cross_idx: Optional[int] = None
        cross_dir: Optional[str] = None
        for i in range(latest_idx, max(latest_idx - _LOOKBACK_BARS, 0), -1):
            if i - 1 < 0:
                break
            prev = sign.iloc[i - 1]
            cur = sign.iloc[i]
            if prev <= 0 and cur > 0:
                cross_idx, cross_dir = i, "bull"
                break
            if prev >= 0 and cur < 0:
                cross_idx, cross_dir = i, "bear"
                break

        if cross_idx is None or cross_dir is None:
            continue

        days_since = latest_idx - cross_idx
        sma_now = _safe_float(sma.iloc[-1])
        if sma_now is None or sma_now == 0.0:
            continue

        distance_pct = (latest_price - sma_now) / sma_now * 100.0
        if atr_val is not None and atr_val > 0:
            atr_distance = (latest_price - sma_now) / atr_val
        else:
            atr_distance = None

        sma_5ago = _safe_float(sma.iloc[-1 - _SLOPE_WINDOW]) if n > _SLOPE_WINDOW else None
        if sma_5ago is not None and sma_5ago != 0.0:
            slope_pct = (sma_now - sma_5ago) / sma_5ago * 100.0
        else:
            slope_pct = 0.0

        if v is not None and v_avg is not None:
            vol_now = _safe_float(v.iloc[-1])
            vol_base = _safe_float(v_avg.iloc[-1])
            volume_ratio = vol_now / vol_base if (vol_now is not None and vol_base is not None and vol_base > 0) else None
        else:
            volume_ratio = None

        tail_signs = sign.iloc[max(0, latest_idx - _PERSISTENCE_M + 1): latest_idx + 1].tolist()
        if cross_dir == "bull":
            persistence = sum(1 for s in tail_signs if s > 0)
        else:
            persistence = sum(1 for s in tail_signs if s < 0)
        passes_persistence = persistence >= _PERSISTENCE_K

        false_break = False
        check_end = min(cross_idx + _FALSE_BREAK_WINDOW, latest_idx)
        post = sign.iloc[cross_idx + 1: check_end + 1].tolist() if check_end > cross_idx else []
        if cross_dir == "bull" and any(s < 0 for s in post):
            false_break = True
        if cross_dir == "bear" and any(s > 0 for s in post):
            false_break = True

        # Regime — price AND slope must align
        regime_price_ok = regime_now is not None and (
            (cross_dir == "bull" and latest_price > regime_now)
            or (cross_dir == "bear" and latest_price < regime_now)
        )
        regime_slope_ok = (
            (cross_dir == "bull" and regime_slope_pct > 0.0)
            or (cross_dir == "bear" and regime_slope_pct < 0.0)
        )
        regime_ok = bool(regime_price_ok and regime_slope_ok)

        overextended = atr_distance is not None and abs(atr_distance) > _OVEREXTENDED_ATR

        # Pullback & MTF alignment
        pullback_atr_val = _pullback_atr(close, cross_idx, atr_val, cross_dir)
        mtf_alignment_val = _mtf_alignment(close, latest_price, cross_dir, periods, period)

        # Stop / structural target
        stop_price: Optional[float] = None
        target_price: Optional[float] = None
        risk_reward: Optional[float] = None
        used_structural = False
        if atr_val is not None and atr_val > 0:
            if cross_dir == "bull":
                vol_stop = latest_price - _STOP_ATR_MULT * atr_val
                struct_stop = swing_low_val if swing_low_val is not None else vol_stop
                stop_price = min(vol_stop, struct_stop)
                if stop_price is not None and stop_price < latest_price:
                    target_price, used_structural = _structural_target(
                        close, h, l, latest_price, stop_price, atr_val, "bull",
                    )
                    risk = latest_price - stop_price
                    risk_reward = (target_price - latest_price) / risk if risk > 0 else None
                else:
                    stop_price = None
            else:
                vol_stop = latest_price + _STOP_ATR_MULT * atr_val
                struct_stop = swing_high_val if swing_high_val is not None else vol_stop
                stop_price = max(vol_stop, struct_stop)
                if stop_price is not None and stop_price > latest_price:
                    target_price, used_structural = _structural_target(
                        close, h, l, latest_price, stop_price, atr_val, "bear",
                    )
                    risk = stop_price - latest_price
                    risk_reward = (latest_price - target_price) / risk if risk > 0 else None
                else:
                    stop_price = None

        # Historical edge — horizon matched to SMA period
        forward_bars = _EDGE_FORWARD_BY_PERIOD.get(period, _EDGE_DEFAULT_FORWARD)
        # Clamp so we have data
        forward_bars = max(1, min(forward_bars, max(1, n // 4)))
        sma_list = _series_to_list(sma)
        edge = _historical_edge(
            close_list, sma_list, atr_list, regime_list,
            cross_dir, forward=forward_bars,
            stop_mult=_STOP_ATR_MULT, overextended_atr=_OVEREXTENDED_ATR,
        )

        # Kelly fraction (capped)
        win_rate = edge.get("win_rate")
        kelly_fraction: Optional[float] = None
        if win_rate is not None and risk_reward is not None and risk_reward > 0:
            p = float(win_rate)
            raw = p - (1.0 - p) / float(risk_reward)
            kelly_fraction = max(0.0, min(_KELLY_CAP, raw))

        # Composite score
        if atr_distance is not None:
            dist_comp = _clip01(abs(atr_distance) / 2.0)
        else:
            dist_comp = _clip01(abs(distance_pct) / 5.0)

        slope_comp = _clip01(abs(slope_pct) / 2.0)
        if volume_ratio is not None:
            vol_comp = _clip01((volume_ratio - 0.8) / 1.2)
        else:
            vol_comp = 0.5
        pers_comp = _clip01(persistence / _PERSISTENCE_M)
        fresh_comp = _clip01(1.0 - days_since / max(1, _LOOKBACK_BARS))

        score = 100.0 * (
            _W_DISTANCE * dist_comp
            + _W_SLOPE * slope_comp
            + _W_VOLUME * vol_comp
            + _W_PERSISTENCE * pers_comp
            + _W_FRESHNESS * fresh_comp
        )
        if false_break:
            score *= 0.6
        # Vol-regime dampening
        score *= _VOL_SCORE_MULT.get(vol_regime_str, 1.0)
        # MTF alignment bonus/malus
        if mtf_alignment_val is not None:
            score *= _MTF_SCORE_FLOOR + (_MTF_SCORE_CEIL - _MTF_SCORE_FLOOR) * mtf_alignment_val

        # Grade
        grade: Optional[str] = None
        grade_reasons: List[str] = []
        expectancy_r = edge.get("expectancy_r")
        stop_hit_rate = edge.get("stop_hit_rate")

        base_pass = (
            passes_persistence
            and not false_break
            and not overextended
            and regime_ok
            and risk_reward is not None
        )

        if base_pass and risk_reward is not None:
            gate_a = (
                risk_reward >= _GRADE_A_RR
                and win_rate is not None and win_rate >= _GRADE_A_WINRATE
                and expectancy_r is not None and expectancy_r >= _GRADE_A_EXPECTANCY_R
                and (stop_hit_rate is None or stop_hit_rate <= _GRADE_A_STOP_HIT_MAX)
            )
            if gate_a:
                grade = "A"
                grade_reasons = [
                    "regime+slope aligned", "persistence confirmed", "not overextended",
                    f"R:R {risk_reward:.1f}",
                    f"win {win_rate:.0%} (n={edge['samples']})",
                    f"E[R] {expectancy_r:.2f}",
                ]
                if stop_hit_rate is not None:
                    grade_reasons.append(f"stop-hit {stop_hit_rate:.0%}")
            elif risk_reward >= _GRADE_B_RR:
                grade = "B"
                grade_reasons = [
                    "regime+slope aligned", "persistence confirmed", "not overextended",
                    f"R:R {risk_reward:.1f}",
                ]
                if win_rate is not None:
                    grade_reasons.append(f"edge {win_rate:.0%} (n={edge['samples']})")
                if expectancy_r is not None:
                    grade_reasons.append(f"E[R] {expectancy_r:.2f}")
            else:
                grade = "C"
                grade_reasons = ["persistence ok", f"R:R {risk_reward:.1f}"]
        elif passes_persistence and not false_break:
            grade = "C"
            if not regime_ok:
                grade_reasons.append("against regime")
            if overextended:
                grade_reasons.append("overextended")

        cross_date = None
        if dates is not None:
            try:
                cross_date = str(pd.Series(dates).reset_index(drop=True).iloc[cross_idx])[:10]
            except Exception:
                cross_date = None

        rec: Dict[str, Any] = {
            "period": period,
            "direction": cross_dir,
            "price": latest_price,
            "sma": sma_now,
            "distance_pct": round(distance_pct, 4),
            "atr_distance": round(atr_distance, 4) if atr_distance is not None else None,
            "atr": round(atr_val, 4) if atr_val is not None else None,
            "slope_pct_5d": round(slope_pct, 4),
            "volume_ratio": round(volume_ratio, 3) if volume_ratio is not None else None,
            "days_since_cross": int(days_since),
            "persistence": int(persistence),
            "persistence_window": _PERSISTENCE_M,
            "persistence_threshold": _PERSISTENCE_K,
            "passes_persistence": bool(passes_persistence),
            "false_break": bool(false_break),
            "score": round(score, 2),
            "cross_date": cross_date,
            "cross_index_from_end": int(days_since),

            "regime_sma": round(regime_now, 4) if regime_now is not None else None,
            "regime_slope_pct": round(regime_slope_pct, 4),
            "regime_ok": bool(regime_ok),
            "vol_regime": vol_regime_str,
            "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
            "overextended": bool(overextended),
            "pullback_atr": round(pullback_atr_val, 3) if pullback_atr_val is not None else None,
            "mtf_alignment": round(mtf_alignment_val, 3) if mtf_alignment_val is not None else None,

            "stop_price": round(stop_price, 4) if stop_price is not None else None,
            "target_price": round(target_price, 4) if target_price is not None else None,
            "risk_reward": round(risk_reward, 2) if risk_reward is not None else None,
            "used_structural_target": bool(used_structural),
            "kelly_fraction": round(kelly_fraction, 3) if kelly_fraction is not None else None,

            "grade": grade,
            "grade_reasons": grade_reasons,
            "historical_edge": edge,
            "edge_forward_days": forward_bars,
        }
        out.append(rec)

    return out


# ── build from CSV ────────────────────────────────────────────────────────

def _compute_one(filepath: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        return []
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return []
    if df.empty:
        return []

    close_col = None
    for cand in ("Close", "close", "Adj Close", "adj_close"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        return []

    date_col = None
    for cand in ("Date", "date", "Datetime", "datetime"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is not None:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col).reset_index(drop=True)
        except Exception:
            pass

    close = df[close_col].astype(float).dropna()
    if len(close) < 30:
        return []

    df = df.loc[close.index].reset_index(drop=True)
    close = close.reset_index(drop=True)

    high = df.get("High", df.get("high"))
    low = df.get("Low", df.get("low"))
    volume = df.get("Volume", df.get("volume"))
    dates = df.get(date_col) if date_col else None

    return detect_reversals(close, high=high, low=low, volume=volume, dates=dates)


def _build() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not os.path.isdir(PRICES_DIR):
        return [], {}
    reversals: List[Dict[str, Any]] = []
    for fp in sorted(glob.glob(os.path.join(PRICES_DIR, "*.csv"))):
        recs = _compute_one(fp)
        if not recs:
            continue
        sym = _symbol_for(fp)
        for r in recs:
            r["symbol"] = sym
            reversals.append(r)

    counts: Dict[str, Dict[str, int]] = {}
    for p in _SMA_PERIODS:
        counts[str(p)] = {
            "bull": sum(1 for r in reversals if r["period"] == p and r["direction"] == "bull"),
            "bear": sum(1 for r in reversals if r["period"] == p and r["direction"] == "bear"),
        }

    grade_order = {"A": 0, "B": 1, "C": 2, None: 3}
    reversals.sort(key=lambda r: (
        grade_order.get(r.get("grade"), 4),
        -r["score"],
        r["symbol"],
        r["period"],
    ))
    return reversals, counts


def get_all_sma_reversals(force: bool = False) -> Dict[str, Any]:
    """Return full reversals snapshot (cached)."""
    now = time.time()
    sig = _signature()
    fresh = (
        not force
        and _cache["reversals"]
        and _cache["mtime_signature"] == sig
        and (now - _cache["built_at"]) < _CACHE_TTL_SEC
    )
    if not fresh:
        reversals, counts = _build()
        _cache["reversals"] = reversals
        _cache["counts"] = counts
        _cache["mtime_signature"] = sig
        _cache["built_at"] = now

    reversals = _cache["reversals"]
    grade_counts = {
        "A": sum(1 for r in reversals if r.get("grade") == "A"),
        "B": sum(1 for r in reversals if r.get("grade") == "B"),
        "C": sum(1 for r in reversals if r.get("grade") == "C"),
        "ungraded": sum(1 for r in reversals if r.get("grade") is None),
    }
    buy_setups = sum(1 for r in reversals
                     if r.get("direction") == "bull" and r.get("grade") in ("A", "B"))

    return {
        "reversals": reversals,
        "counts_by_period": _cache["counts"],
        "grade_counts": grade_counts,
        "buy_setups": buy_setups,
        "periods": list(_SMA_PERIODS),
        "lookback_bars": _LOOKBACK_BARS,
        "persistence_window": _PERSISTENCE_M,
        "persistence_threshold": _PERSISTENCE_K,
        "regime_period": _REGIME_PERIOD,
        "regime_slope_window": _REGIME_SLOPE_WINDOW,
        "overextended_atr": _OVEREXTENDED_ATR,
        "edge_forward_days": _EDGE_DEFAULT_FORWARD,
        "edge_forward_by_period": dict(_EDGE_FORWARD_BY_PERIOD),
        "vol_regime_window": _VOL_REGIME_WINDOW,
        "vol_regime_baseline": _VOL_REGIME_BASELINE,
        "pullback_window": _PULLBACK_WINDOW,
        "structural_target_window": _STRUCTURAL_TARGET_WINDOW,
        "grade_a_rr": _GRADE_A_RR,
        "grade_a_winrate": _GRADE_A_WINRATE,
        "grade_a_expectancy_r": _GRADE_A_EXPECTANCY_R,
        "grade_a_stop_hit_max": _GRADE_A_STOP_HIT_MAX,
        "grade_b_rr": _GRADE_B_RR,
        "kelly_cap": _KELLY_CAP,
        "total": len(reversals),
        "built_at": _cache["built_at"],
    }
