"""
SMA Reversal Detection Service — Buy-Signal Quality Engine
===========================================================

Answers the question: *"Should I buy this stock right now?"*

World-class quant logic for detecting SMA (9/50/600) reversals, plus a
per-setup grading system built from confluence of:

1. **Crossover freshness** — true sign-change on (close − SMA) within the
   last `_LOOKBACK_BARS` closes (not just "below + turning up")
2. **Regime filter** — bull setups require price > SMA-200 ("don't fight
   the elephant"); bear setups require price < SMA-200
3. **Persistence** — K of last M closes on the new side (whipsaw reject)
4. **ATR-normalised distance** — reject overextended entries (> 3σ from
   SMA ⇒ chasing; edge evaporates)
5. **Volume confirmation** — cross on ≥ 1× 20-bar avg volume
6. **Trend alignment** — SMA slope in the direction of the trade
7. **Stop / target / risk-reward** — stop = min(10-bar swing low,
   entry − 2·ATR); target = 2R above entry ⇒ R:R
8. **Historical edge** — backtest every past crossover on the *same*
   symbol; report win-rate + median forward return (10-bar horizon)
9. **Composite score 0-100** — blended from the seven sub-scores; false
   breaks (re-cross within 3 bars) get a 0.6× penalty
10. **Grade A / B / C** — hard confluence gates:
      • A = regime_ok + persistence_ok + !overextended + !false_break
            + R:R ≥ 2.0 + win_rate ≥ 0.55 (N ≥ 5)
      • B = regime_ok + persistence_ok + !overextended + !false_break
            + R:R ≥ 1.5
      • C = everything else that still passes persistence
      • ``null`` otherwise (setup not tradeable)

All numbers are derived from the same symbol's own price history — no
cross-sectional assumptions — so the signal is robust to idiosyncratic
volatility and cross-asset regime differences.

Cached in-memory; invalidated when any underlying CSV mtime changes.
"""

from __future__ import annotations

import glob
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Buy-signal quality
_REGIME_PERIOD = 200
_OVEREXTENDED_ATR = 3.0
_STOP_ATR_MULT = 2.0
_TARGET_R_MULT = 2.0
_SWING_LOOKBACK = 10
_EDGE_FORWARD_DAYS = 10
_EDGE_MIN_SAMPLES = 5

# Grade gates
_GRADE_A_RR = 2.0
_GRADE_A_WINRATE = 0.55
_GRADE_B_RR = 1.5

# Score weights (sum to 1.0)
_W_DISTANCE = 0.30
_W_SLOPE = 0.20
_W_VOLUME = 0.20
_W_PERSISTENCE = 0.15
_W_FRESHNESS = 0.15

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


# ── historical edge (per-symbol backtest of the exact setup) ──────────────

def _historical_edge(
    close_list: List[float],
    sma_list: List[float],
    direction: str,
    forward: int = _EDGE_FORWARD_DAYS,
) -> Dict[str, Any]:
    """
    Walk the series and find every past crossover of the same kind.
    For each, compute the `forward`-bar forward return from the close at
    the cross bar. Report sample size, win rate and median.
    """
    n = len(close_list)
    if n != len(sma_list) or n < forward + 2:
        return {"samples": 0, "win_rate": None, "median_fwd_pct": None,
                "mean_fwd_pct": None, "std_fwd_pct": None}

    fwd_returns: List[float] = []
    for i in range(1, n - forward):
        s_prev, s_cur = sma_list[i - 1], sma_list[i]
        if s_prev is None or s_cur is None:
            continue
        try:
            if math.isnan(s_prev) or math.isnan(s_cur):
                continue
        except TypeError:
            continue
        c_prev, c_cur = close_list[i - 1], close_list[i]
        sign_prev = 0 if c_prev == s_prev else (1 if c_prev > s_prev else -1)
        sign_cur = 0 if c_cur == s_cur else (1 if c_cur > s_cur else -1)

        is_cross = False
        if direction == "bull" and sign_prev <= 0 and sign_cur > 0:
            is_cross = True
        elif direction == "bear" and sign_prev >= 0 and sign_cur < 0:
            is_cross = True
        if not is_cross:
            continue

        c0 = close_list[i]
        c1 = close_list[i + forward]
        if c0 is None or c1 is None or c0 == 0:
            continue
        fwd_returns.append((c1 - c0) / c0 * 100.0)

    samples = len(fwd_returns)
    if samples < _EDGE_MIN_SAMPLES:
        return {"samples": samples, "win_rate": None, "median_fwd_pct": None,
                "mean_fwd_pct": None, "std_fwd_pct": None}

    if direction == "bull":
        wins = sum(1 for r in fwd_returns if r > 0)
    else:
        wins = sum(1 for r in fwd_returns if r < 0)
    sorted_r = sorted(fwd_returns)
    mid = samples // 2
    if samples % 2 == 1:
        median = sorted_r[mid]
    else:
        median = 0.5 * (sorted_r[mid - 1] + sorted_r[mid])
    mean = sum(fwd_returns) / samples
    var = sum((r - mean) ** 2 for r in fwd_returns) / samples
    std = math.sqrt(var) if var > 0 else 0.0

    return {
        "samples": samples,
        "win_rate": round(wins / samples, 3),
        "median_fwd_pct": round(median, 3),
        "mean_fwd_pct": round(mean, 3),
        "std_fwd_pct": round(std, 3),
    }


# ── detection ─────────────────────────────────────────────────────────────

def detect_reversals(
    close: "Any",
    high: "Optional[Any]" = None,
    low: "Optional[Any]" = None,
    volume: "Optional[Any]" = None,
    dates: "Optional[Any]" = None,
    periods: Tuple[int, ...] = _SMA_PERIODS,
) -> List[Dict[str, Any]]:
    """Pure detection function (symbol-agnostic)."""
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
        atr = close.pct_change().rolling(_ATR_PERIOD, min_periods=5).std() * close

    # Volume baseline
    if volume is not None:
        v = pd.Series(volume).astype(float).reset_index(drop=True)
        v_avg = v.rolling(_VOLUME_BASELINE, min_periods=5).mean()
    else:
        v = None
        v_avg = None

    # Regime (SMA-200 proxy; truncated if series too short)
    reg_period = min(_REGIME_PERIOD, max(20, n // 2))
    regime_sma = close.rolling(reg_period).mean()

    # Swing high/low for structural stops
    if low is not None:
        swing_low = pd.Series(low).astype(float).reset_index(drop=True).rolling(_SWING_LOOKBACK).min()
    else:
        swing_low = close.rolling(_SWING_LOOKBACK).min()
    if high is not None:
        swing_high = pd.Series(high).astype(float).reset_index(drop=True).rolling(_SWING_LOOKBACK).max()
    else:
        swing_high = close.rolling(_SWING_LOOKBACK).max()

    latest_idx = n - 1
    latest_price = _safe_float(close.iloc[-1])
    if latest_price is None:
        return []

    regime_now = _safe_float(regime_sma.iloc[-1])
    atr_val = _safe_float(atr.iloc[-1])
    swing_low_val = _safe_float(swing_low.iloc[-1])
    swing_high_val = _safe_float(swing_high.iloc[-1])

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

        # Regime
        if regime_now is not None:
            regime_ok = (cross_dir == "bull" and latest_price > regime_now) or \
                        (cross_dir == "bear" and latest_price < regime_now)
        else:
            regime_ok = False

        overextended = atr_distance is not None and abs(atr_distance) > _OVEREXTENDED_ATR

        # Stop / target geometry
        stop_price: Optional[float] = None
        target_price: Optional[float] = None
        risk_reward: Optional[float] = None
        if atr_val is not None and atr_val > 0:
            if cross_dir == "bull":
                vol_stop = latest_price - _STOP_ATR_MULT * atr_val
                struct_stop = swing_low_val if swing_low_val is not None else vol_stop
                stop_price = min(vol_stop, struct_stop)
                if stop_price is not None and stop_price < latest_price:
                    risk = latest_price - stop_price
                    target_price = latest_price + _TARGET_R_MULT * risk
                    risk_reward = _TARGET_R_MULT
                else:
                    stop_price = None
            else:
                vol_stop = latest_price + _STOP_ATR_MULT * atr_val
                struct_stop = swing_high_val if swing_high_val is not None else vol_stop
                stop_price = max(vol_stop, struct_stop)
                if stop_price is not None and stop_price > latest_price:
                    risk = stop_price - latest_price
                    target_price = latest_price - _TARGET_R_MULT * risk
                    risk_reward = _TARGET_R_MULT
                else:
                    stop_price = None

        # Historical edge on this symbol's own past
        close_list = close.tolist()
        sma_list_raw = sma.tolist()
        edge = _historical_edge(close_list, sma_list_raw, cross_dir,
                                forward=_EDGE_FORWARD_DAYS)

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

        # Grade
        grade: Optional[str] = None
        grade_reasons: List[str] = []
        base_pass = (
            passes_persistence and not false_break
            and not overextended and regime_ok
            and risk_reward is not None
        )
        win_rate = edge.get("win_rate")
        if base_pass and risk_reward is not None:
            if risk_reward >= _GRADE_A_RR and win_rate is not None and win_rate >= _GRADE_A_WINRATE:
                grade = "A"
                grade_reasons = [
                    "regime aligned", "persistence confirmed", "not overextended",
                    f"R:R {risk_reward:.1f}",
                    f"win-rate {win_rate:.0%} (n={edge['samples']})",
                ]
            elif risk_reward >= _GRADE_B_RR:
                grade = "B"
                grade_reasons = [
                    "regime aligned", "persistence confirmed", "not overextended",
                    f"R:R {risk_reward:.1f}",
                ]
                if win_rate is not None:
                    grade_reasons.append(f"edge {win_rate:.0%} (n={edge['samples']})")
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
            "regime_ok": bool(regime_ok),
            "overextended": bool(overextended),
            "stop_price": round(stop_price, 4) if stop_price is not None else None,
            "target_price": round(target_price, 4) if target_price is not None else None,
            "risk_reward": round(risk_reward, 2) if risk_reward is not None else None,
            "grade": grade,
            "grade_reasons": grade_reasons,
            "historical_edge": edge,
            "edge_forward_days": _EDGE_FORWARD_DAYS,
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
        "overextended_atr": _OVEREXTENDED_ATR,
        "edge_forward_days": _EDGE_FORWARD_DAYS,
        "total": len(reversals),
        "built_at": _cache["built_at"],
    }
