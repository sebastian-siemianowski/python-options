"""
Multi-metric scoring and ranking system for indicator strategies.
Produces a composite score for leaderboard ranking.
"""

import numpy as np


def compute_composite_score(agg: dict) -> float:
    """
    Compute composite score from aggregated backtest metrics.

    Weights:
        - Sharpe ratio:     30%
        - Sortino ratio:    15%
        - CAGR vs B&H:     15%
        - Max drawdown:     15%
        - Buy hit rate:     10%
        - Win rate:          5%
        - Profit factor:     5%
        - Exposure penalty:  5%

    Returns float in roughly [0, 100] range.
    """
    sharpe = agg.get("med_sharpe", 0) or 0
    sortino = agg.get("med_sortino", 0) or 0
    cagr_diff = agg.get("med_cagr_diff", 0) or 0
    max_dd = agg.get("med_max_dd", 0) or 0
    buy_hit = agg.get("med_buy_hit", 50) or 50
    win_rate = agg.get("med_win_rate", 50) or 50
    pf = agg.get("med_profit_factor", 1) or 1
    exposure = agg.get("med_exposure", 50) or 50

    # Normalize components to [0, 1] range
    sharpe_n = np.clip(sharpe / 2.0, -1, 1) * 0.5 + 0.5       # 0.5 at Sharpe=0, 1.0 at Sharpe=2
    sortino_n = np.clip(sortino / 3.0, -1, 1) * 0.5 + 0.5
    cagr_n = np.clip(cagr_diff / 20.0, -1, 1) * 0.5 + 0.5      # 0.5 at 0% diff, 1.0 at +20%
    dd_n = np.clip(1 + max_dd / 50.0, 0, 1)                     # max_dd is negative; -50% = 0, 0% = 1
    hit_n = np.clip((buy_hit - 40) / 30, 0, 1)                  # 40% = 0, 70% = 1
    win_n = np.clip((win_rate - 40) / 30, 0, 1)
    pf_n = np.clip((pf - 0.5) / 2.0, 0, 1)                     # 0.5 = 0, 2.5 = 1
    # Exposure: prefer 30-70%; penalize extremes
    exp_n = 1.0 - abs(exposure - 50) / 50.0

    composite = (
        0.30 * sharpe_n +
        0.15 * sortino_n +
        0.15 * cagr_n +
        0.15 * dd_n +
        0.10 * hit_n +
        0.05 * win_n +
        0.05 * pf_n +
        0.05 * exp_n
    ) * 100

    # Penalize strategies with zero or near-zero trades / exposure
    n_trades = agg.get("med_n_trades", 0) or 0
    if n_trades < 1 or exposure < 1:
        composite *= 0.1  # effectively disqualify

    return round(composite, 2)


def aggregate_results(per_asset: list[dict]) -> dict:
    """Aggregate per-asset backtest results into summary metrics."""
    if not per_asset:
        return {}

    def _med(key):
        vals = [r[key] for r in per_asset if r.get(key) is not None]
        return round(float(np.median(vals)), 3) if vals else None

    def _mean(key):
        vals = [r[key] for r in per_asset if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    n = len(per_asset)
    beats = sum(1 for r in per_asset if (r.get("sharpe") or 0) > (r.get("bh_sharpe") or 0))

    agg = {
        "n_assets": n,
        "med_sharpe": _med("sharpe"),
        "mean_sharpe": _mean("sharpe"),
        "med_sortino": _med("sortino"),
        "med_cagr": _med("cagr"),
        "med_bh_cagr": _med("bh_cagr"),
        "med_cagr_diff": None,
        "med_max_dd": _med("max_dd"),
        "med_buy_hit": _med("buy_hit"),
        "med_sell_hit": _med("sell_hit"),
        "med_win_rate": _med("win_rate"),
        "med_profit_factor": _med("profit_factor"),
        "med_exposure": _med("exposure"),
        "med_n_trades": _med("n_trades"),
        "sharpe_beat_bh": f"{beats}/{n}",
        "med_total_return": _med("total_return"),
        "med_bh_return": _med("bh_return"),
    }

    # CAGR difference
    if agg["med_cagr"] is not None and agg["med_bh_cagr"] is not None:
        agg["med_cagr_diff"] = round(agg["med_cagr"] - agg["med_bh_cagr"], 2)

    agg["composite"] = compute_composite_score(agg)

    return agg


def rank_strategies(strategy_results: dict) -> list[dict]:
    """
    Rank strategies by composite score.

    Args:
        strategy_results: dict of {strategy_id: {"name": ..., "aggregate": {...}, "per_asset": [...]}}

    Returns:
        Sorted list of dicts with rank, id, name, composite, key metrics.
    """
    rows = []
    for sid, data in strategy_results.items():
        agg = data.get("aggregate", {})
        if not agg:
            continue
        rows.append({
            "rank": 0,
            "id": sid,
            "name": data["name"],
            "family": data.get("family", ""),
            "composite": agg.get("composite", 0),
            "sharpe": agg.get("med_sharpe"),
            "sortino": agg.get("med_sortino"),
            "cagr": agg.get("med_cagr"),
            "max_dd": agg.get("med_max_dd"),
            "buy_hit": agg.get("med_buy_hit"),
            "win_rate": agg.get("med_win_rate"),
            "profit_factor": agg.get("med_profit_factor"),
            "exposure": agg.get("med_exposure"),
            "n_assets": agg.get("n_assets", 0),
            "sharpe_beat_bh": agg.get("sharpe_beat_bh", ""),
        })

    rows.sort(key=lambda x: x["composite"], reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    return rows
