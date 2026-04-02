"""
Health / monitoring service — checks backend, Redis, Celery, data freshness.
"""

import os
import time
import psutil
from typing import Any, Dict, List

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DATA_DIR = os.path.join(SRC_DIR, "data")

# In-memory error log (kept small)
_error_log: List[Dict[str, Any]] = []
_MAX_ERRORS = 50
_START_TIME = time.time()


def log_error(source: str, message: str) -> None:
    """Record an error in the in-memory log."""
    global _error_log
    _error_log.append({
        "source": source,
        "message": str(message)[:500],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    if len(_error_log) > _MAX_ERRORS:
        _error_log = _error_log[-_MAX_ERRORS:]


def get_recent_errors(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent errors, newest first."""
    return list(reversed(_error_log[-limit:]))


def get_full_health() -> Dict[str, Any]:
    """Comprehensive health check across all subsystems."""
    return {
        "api": _check_api(),
        "signal_cache": _check_signal_cache(),
        "price_data": _check_price_data(),
        "workers": _check_workers(),
        "recent_errors": get_recent_errors(10),
    }


def _check_api() -> Dict[str, Any]:
    """API server stats."""
    uptime = time.time() - _START_TIME
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "uptime_human": _human_duration(uptime),
        "memory_mb": round(mem.rss / 1024 / 1024, 1),
        "cpu_percent": round(process.cpu_percent(interval=0.1), 1),
        "pid": os.getpid(),
    }


def _check_signal_cache() -> Dict[str, Any]:
    """Signal cache availability and age."""
    cache_path = os.path.join(DATA_DIR, "currencies", "fx_plnjpy.json")
    if not os.path.isfile(cache_path):
        return {"status": "missing", "exists": False, "age_seconds": None, "size_mb": 0}

    stat = os.stat(cache_path)
    age = time.time() - stat.st_mtime
    size_mb = round(stat.st_size / 1024 / 1024, 2)
    # Fresh < 4h, stale < 24h, old > 24h
    if age < 4 * 3600:
        status = "fresh"
    elif age < 24 * 3600:
        status = "stale"
    else:
        status = "old"

    return {
        "status": status,
        "exists": True,
        "age_seconds": round(age, 1),
        "age_human": _human_duration(age),
        "size_mb": size_mb,
        "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
    }


def _check_price_data() -> Dict[str, Any]:
    """Price data directory stats."""
    prices_dir = os.path.join(DATA_DIR, "prices")
    if not os.path.isdir(prices_dir):
        return {"status": "missing", "total_files": 0, "stale_files": 0}

    files = [f for f in os.listdir(prices_dir) if f.endswith(".csv")]
    total = len(files)
    if total == 0:
        return {"status": "empty", "total_files": 0, "stale_files": 0}

    now = time.time()
    ages = []
    total_size = 0
    for f in files:
        fp = os.path.join(prices_dir, f)
        st = os.stat(fp)
        ages.append(now - st.st_mtime)
        total_size += st.st_size

    stale = sum(1 for a in ages if a > 24 * 3600)
    freshest = min(ages) if ages else 0
    oldest = max(ages) if ages else 0
    status = "ok" if stale < total * 0.1 else "degraded" if stale < total * 0.5 else "stale"

    return {
        "status": status,
        "total_files": total,
        "stale_files": stale,
        "fresh_files": total - stale,
        "freshest_hours": round(freshest / 3600, 1),
        "oldest_hours": round(oldest / 3600, 1),
        "total_size_mb": round(total_size / 1024 / 1024, 1),
    }


def _check_workers() -> Dict[str, Any]:
    """Check Redis and Celery availability (both are optional)."""
    result = {"redis": _check_redis(), "celery": _check_celery()}
    redis_ok = result["redis"]["status"] == "ok"
    celery_ok = result["celery"]["status"] == "ok"
    redis_optional = result["redis"]["status"] == "not_running"
    celery_optional = result["celery"]["status"] in ("not_running", "no_workers")
    if redis_ok and celery_ok:
        result["status"] = "ok"
    elif redis_optional or celery_optional:
        # Optional services not running is fine — not an error
        result["status"] = "ok" if not (redis_ok or celery_ok) else "ok"
    elif redis_ok:
        result["status"] = "degraded"
    else:
        result["status"] = "unavailable"
    return result


def _check_redis() -> Dict[str, str]:
    """Ping Redis (optional service)."""
    try:
        import redis as redis_lib
    except ImportError:
        return {"status": "not_running", "message": "Optional — start with: make redis"}
    try:
        r = redis_lib.Redis(host="localhost", port=6379, db=0, socket_timeout=2)
        r.ping()
        info = r.info("memory")
        return {
            "status": "ok",
            "used_memory_human": info.get("used_memory_human", "?"),
        }
    except Exception:
        return {"status": "not_running", "message": "Optional — start with: make redis"}


def _check_celery() -> Dict[str, Any]:
    """Check Celery worker status (optional service)."""
    try:
        import celery as _celery_mod  # noqa: F401
    except ImportError:
        return {"status": "not_running", "workers": 0, "message": "Optional — start with: make web-worker"}
    try:
        import concurrent.futures
        from web.backend.celery_app import celery_app

        def _inspect():
            i = celery_app.control.inspect(timeout=1)
            return i.active()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_inspect)
            active = future.result(timeout=3)

        if active is None:
            return {"status": "not_running", "workers": 0, "message": "Optional — start with: make web-worker"}
        return {
            "status": "ok",
            "workers": len(active),
            "worker_names": list(active.keys()),
        }
    except Exception:
        return {"status": "not_running", "workers": 0, "message": "Optional — start with: make web-worker"}


def _human_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    if hours < 24:
        return f"{hours}h {mins}m"
    days = hours // 24
    hours = hours % 24
    return f"{days}d {hours}h"
