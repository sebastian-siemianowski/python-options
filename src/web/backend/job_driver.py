"""
Job driver: runs make <mode> and emits structured JSON events to stdout
purpose-built for the frontend, rather than relying on terminal output scraping.

Emits one-line JSON records prefixed with ``@@EVT@@`` for semantic events and
``@@LOG@@`` for raw subprocess log pass-through.

Event types:
    {event: "start",     mode, total_expected}
    {event: "phase",     title}
    {event: "asset",     symbol, status: "ok"|"fail", detail?}
    {event: "heartbeat", done, total, elapsed_s}
    {event: "done",      status: "ok"|"fail", done, total, elapsed_s, exit_code}

The driver watches the filesystem (price CSVs for stocks, tune JSONs for
tune/retune/calibrate) and synthesizes asset events from file mtime changes.
This yields a clean, mode-agnostic event stream that the UI can render as
structured components.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
PRICES_DIR = REPO_ROOT / "src" / "data" / "prices"
TUNE_DIR = REPO_ROOT / "src" / "data" / "tune"

# Only these modes are accepted
VALID_MODES = {"stocks", "tune", "retune", "calibrate"}


def _emit(payload: dict) -> None:
    """Write a semantic event as a single line to stdout."""
    sys.stdout.write("@@EVT@@ " + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_log(text: str) -> None:
    """Pass through a raw log line."""
    sys.stdout.write("@@LOG@@ " + text.rstrip() + "\n")
    sys.stdout.flush()


def _scan(dir_path: Path, suffix: str) -> Dict[str, Tuple[float, int]]:
    """Return {symbol: (mtime, size)} for files matching *.<suffix> in dir_path."""
    out: Dict[str, Tuple[float, int]] = {}
    if not dir_path.exists():
        return out
    try:
        for entry in os.scandir(dir_path):
            name = entry.name
            if not name.endswith(suffix):
                continue
            symbol = name[: -len(suffix)]
            # stocks prices use e.g. AAPL_1d.csv → strip _1d
            if symbol.endswith("_1d"):
                symbol = symbol[:-3]
            try:
                st = entry.stat()
                out[symbol] = (st.st_mtime, st.st_size)
            except OSError:
                continue
    except OSError:
        pass
    return out


def _estimate_total(mode: str) -> int:
    """Estimate how many symbols are expected to be processed."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from ingestion.data_utils import DEFAULT_ASSET_UNIVERSE  # type: ignore

        return len(DEFAULT_ASSET_UNIVERSE)
    except Exception:
        if mode == "stocks":
            return max(1, len(_scan(PRICES_DIR, ".csv")))
        return max(1, len(_scan(TUNE_DIR, ".json")))


def _watch_dir(
    dir_path: Path,
    suffix: str,
    baseline: Dict[str, Tuple[float, int]],
    done_symbols: Set[str],
    stop_flag: threading.Event,
    poll_interval: float = 0.8,
) -> None:
    """Emit asset events whenever a file in dir_path changes after baseline."""
    while not stop_flag.is_set():
        current = _scan(dir_path, suffix)
        for symbol, (mtime, size) in current.items():
            if symbol in done_symbols:
                continue
            prev = baseline.get(symbol)
            if prev is None or mtime > prev[0] + 0.01 or size != prev[1]:
                done_symbols.add(symbol)
                _emit(
                    {
                        "event": "asset",
                        "symbol": symbol,
                        "status": "ok",
                        "detail": f"{size // 1024}k" if size > 1024 else f"{size}B",
                    }
                )
        stop_flag.wait(poll_interval)


def _heartbeat(
    total: int,
    done_symbols: Set[str],
    start_ts: float,
    stop_flag: threading.Event,
    interval: float = 1.0,
) -> None:
    """Periodically emit heartbeat events with progress counts."""
    while not stop_flag.wait(interval):
        _emit(
            {
                "event": "heartbeat",
                "done": len(done_symbols),
                "total": total,
                "elapsed_s": round(time.time() - start_ts, 1),
            }
        )


def _run(mode: str) -> int:
    total = _estimate_total(mode)
    _emit({"event": "start", "mode": mode, "total_expected": total})

    # Pick watcher target based on mode
    if mode == "stocks":
        watch_dir, suffix, phase_title = PRICES_DIR, ".csv", "Downloading prices"
    else:
        watch_dir, suffix, phase_title = TUNE_DIR, ".json", "Fitting models"

    baseline = _scan(watch_dir, suffix)
    _emit({"event": "phase", "title": phase_title})

    done_symbols: Set[str] = set()
    stop_flag = threading.Event()
    start_ts = time.time()

    watcher = threading.Thread(
        target=_watch_dir,
        args=(watch_dir, suffix, baseline, done_symbols, stop_flag),
        daemon=True,
    )
    hb = threading.Thread(
        target=_heartbeat,
        args=(total, done_symbols, start_ts, stop_flag),
        daemon=True,
    )
    watcher.start()
    hb.start()

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "COLUMNS": "120",
    }

    proc = subprocess.Popen(
        ["make", mode],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                _emit_log(line)
        rc = proc.wait()
    finally:
        stop_flag.set()
        watcher.join(timeout=2.0)
        hb.join(timeout=2.0)

    # Final sweep to catch any files that landed in the last polling gap
    final = _scan(watch_dir, suffix)
    for symbol, (mtime, size) in final.items():
        if symbol in done_symbols:
            continue
        prev = baseline.get(symbol)
        if prev is None or mtime > prev[0] + 0.01 or size != prev[1]:
            done_symbols.add(symbol)
            _emit(
                {
                    "event": "asset",
                    "symbol": symbol,
                    "status": "ok",
                    "detail": f"{size // 1024}k" if size > 1024 else f"{size}B",
                }
            )

    _emit(
        {
            "event": "done",
            "status": "ok" if rc == 0 else "fail",
            "done": len(done_symbols),
            "total": total,
            "elapsed_s": round(time.time() - start_ts, 1),
            "exit_code": rc,
        }
    )
    return rc


def main(argv: Optional[list] = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) < 1 or argv[0] not in VALID_MODES:
        sys.stderr.write(f"usage: job_driver <mode>  where mode in {sorted(VALID_MODES)}\n")
        return 2
    return _run(argv[0])


if __name__ == "__main__":
    sys.exit(main())
