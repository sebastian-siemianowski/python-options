"""
Indicators service — reads backtest results from JSON,
serves leaderboard, per-strategy detail, and per-asset breakdowns.
Also manages background backtest runs.
"""

import json
import os
import subprocess
import sys
import time
import threading

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
RESULTS_DIR = os.path.join(SRC_DIR, "data", "indicators")
RESULTS_FILE = os.path.join(RESULTS_DIR, "backtest_results.json")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "backtest_results_summary.json")

_cache = {}

# ── Backtest runner state ─────────────────────────────────────────────
_backtest_state = {
    "running": False,
    "pid": None,
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "progress": "",
    "error": None,
    "mode": None,  # "quick" or "full"
}
_backtest_lock = threading.Lock()


def _load_summary():
    if "summary" in _cache:
        return _cache["summary"]
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as f:
            data = json.load(f)
        _cache["summary"] = data
        return data
    return {}


def _load_full():
    if "full" in _cache:
        return _cache["full"]
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        _cache["full"] = data
        return data
    return {}


def clear_cache():
    _cache.clear()


def get_leaderboard(top_n: int = 0, family: str = None):
    """Return ranked leaderboard of strategies. top_n=0 means all."""
    summary = _load_summary()
    if not summary:
        return {"strategies": [], "total": 0}

    rows = []
    for sid, data in summary.items():
        agg = data.get("aggregate", {})
        if family and family.lower() not in data.get("family", "").lower():
            continue
        rows.append({
            "id": int(sid),
            "name": data["name"],
            "family": data.get("family", ""),
            "composite": agg.get("composite", 0) if agg else 0,
            "sharpe": agg.get("med_sharpe") if agg else None,
            "sortino": agg.get("med_sortino") if agg else None,
            "cagr": agg.get("med_cagr") if agg else None,
            "bh_cagr": agg.get("med_bh_cagr") if agg else None,
            "cagr_diff": agg.get("med_cagr_diff") if agg else None,
            "max_dd": agg.get("med_max_dd") if agg else None,
            "buy_hit": agg.get("med_buy_hit") if agg else None,
            "sell_hit": agg.get("med_sell_hit") if agg else None,
            "win_rate": agg.get("med_win_rate") if agg else None,
            "profit_factor": agg.get("med_profit_factor") if agg else None,
            "exposure": agg.get("med_exposure") if agg else None,
            "n_trades": agg.get("med_n_trades") if agg else None,
            "n_assets": data.get("n_assets", 0),
            "sharpe_beat_bh": agg.get("sharpe_beat_bh") if agg else None,
        })

    rows.sort(key=lambda x: x["composite"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    total = len(rows)
    if top_n and top_n > 0:
        rows = rows[:top_n]

    return {"strategies": rows, "total": total}


def get_strategy_detail(strategy_id: int):
    """Return full detail for a single strategy including per-asset results."""
    full = _load_full()
    sid_str = str(strategy_id)
    if sid_str not in full:
        return None

    data = full[sid_str]
    return {
        "id": strategy_id,
        "name": data["name"],
        "family": data.get("family", ""),
        "aggregate": data.get("aggregate", {}),
        "per_asset": data.get("per_asset", []),
    }


def get_families():
    """Return list of strategy families with counts."""
    summary = _load_summary()
    families = {}
    for sid, data in summary.items():
        fam = data.get("family", "Unknown")
        if fam not in families:
            families[fam] = {"name": fam, "count": 0, "avg_composite": 0, "ids": []}
        families[fam]["count"] += 1
        families[fam]["ids"].append(int(sid))
        agg = data.get("aggregate", {})
        families[fam]["avg_composite"] += agg.get("composite", 0)

    result = []
    for fam, info in families.items():
        if info["count"] > 0:
            info["avg_composite"] = round(info["avg_composite"] / info["count"], 2)
        info["ids"] = sorted(info["ids"])
        result.append(info)

    result.sort(key=lambda x: x["avg_composite"], reverse=True)
    return result


def get_top_10():
    """Return top 10 strategies (the elite selection)."""
    lb = get_leaderboard(top_n=10)
    return lb["strategies"]


def get_asset_heatmap(strategy_id: int):
    """Return per-asset performance for a strategy as a heatmap-ready structure."""
    detail = get_strategy_detail(strategy_id)
    if not detail:
        return None

    assets = []
    for r in detail.get("per_asset", []):
        assets.append({
            "symbol": r.get("symbol", ""),
            "sharpe": r.get("sharpe", 0),
            "cagr": r.get("cagr", 0),
            "max_dd": r.get("max_dd", 0),
            "total_return": r.get("total_return", 0),
            "win_rate": r.get("win_rate"),
            "n_trades": r.get("n_trades", 0),
        })

    assets.sort(key=lambda x: x["sharpe"] or 0, reverse=True)
    return {
        "id": strategy_id,
        "name": detail["name"],
        "assets": assets,
    }


# ── Backtest runner ──────────────────────────────────────────────────

def _run_backtest_thread(mode: str):
    """Run backtest as subprocess. Called in background thread."""
    global _backtest_state
    python = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    cmd = [python, "-m", "indicators.cli", "--top", "500"]
    if mode == "quick":
        cmd.append("--quick")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=SRC_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": SRC_DIR},
        )
        with _backtest_lock:
            _backtest_state["pid"] = proc.pid

        last_line = ""
        for line in proc.stdout:
            line = line.strip()
            if line:
                last_line = line
                with _backtest_lock:
                    _backtest_state["progress"] = line

        proc.wait()
        with _backtest_lock:
            _backtest_state["running"] = False
            _backtest_state["finished_at"] = time.time()
            _backtest_state["exit_code"] = proc.returncode
            if proc.returncode != 0:
                _backtest_state["error"] = last_line
            # Clear result cache so fresh data is served
            _cache.clear()

    except Exception as exc:
        with _backtest_lock:
            _backtest_state["running"] = False
            _backtest_state["finished_at"] = time.time()
            _backtest_state["exit_code"] = -1
            _backtest_state["error"] = str(exc)


def start_backtest(mode: str = "full") -> dict:
    """Start a background backtest run. Returns status dict."""
    global _backtest_state
    with _backtest_lock:
        if _backtest_state["running"]:
            return {
                "status": "already_running",
                "started_at": _backtest_state["started_at"],
                "mode": _backtest_state["mode"],
                "progress": _backtest_state["progress"],
            }

        _backtest_state = {
            "running": True,
            "pid": None,
            "started_at": time.time(),
            "finished_at": None,
            "exit_code": None,
            "progress": "Starting...",
            "error": None,
            "mode": mode,
        }

    t = threading.Thread(target=_run_backtest_thread, args=(mode,), daemon=True)
    t.start()

    return {"status": "started", "mode": mode}


def get_backtest_status() -> dict:
    """Return current backtest status."""
    with _backtest_lock:
        state = dict(_backtest_state)
    elapsed = None
    if state["started_at"]:
        end = state["finished_at"] or time.time()
        elapsed = round(end - state["started_at"], 1)
    state["elapsed_seconds"] = elapsed
    return state
