"""
Job driver: runs make <mode> and emits structured JSON events to stdout
purpose-built for the frontend, rather than relying on terminal output scraping.

Emits one-line JSON records prefixed with ``@@EVT@@`` for semantic events and
``@@LOG@@`` for raw subprocess log pass-through.

Event types:
    {event: "start",     mode, total_expected}
    {event: "phase",     title, step?, total_steps?, kind?}
    {event: "asset",     symbol, status: "ok"|"fail", detail?}
    {event: "heartbeat", done, total, elapsed_s, phase_done?, phase_total?}
    {event: "done",      status: "ok"|"fail", done, total, elapsed_s, exit_code}

The driver watches the filesystem (price CSVs for stocks, tune JSONs for
tune/retune/calibrate) and synthesizes asset events from file mtime changes.
For multi-step pipelines (like `retune` which chains refresh -> backup -> tune),
the driver parses Make-level phase markers from stdout and switches the
watcher target dynamically so progress always reflects the active stage.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
PRICES_DIR = REPO_ROOT / "src" / "data" / "prices"
TUNE_DIR = REPO_ROOT / "src" / "data" / "tune"

# Only these modes are accepted
VALID_MODES = {"stocks", "tune", "retune", "calibrate", "tune-stocks"}


def _emit(payload: dict) -> None:
    """Write a semantic event as a single line to stdout."""
    sys.stdout.write("@@EVT@@ " + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_log(text: str) -> None:
    """Pass through a raw log line."""
    sys.stdout.write("@@LOG@@ " + text.rstrip() + "\n")
    sys.stdout.flush()


def _parse_metric_float(value: str) -> Optional[float]:
    text = value.strip().replace(",", "")
    if not text or text in {"-", "—", "nan", "NaN"}:
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError:
        return None


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


# ---------------------------------------------------------------------------
# Phase plan: ordered list of expected stages for each mode.
#
# Each phase declares how to detect it (a regex pattern), which directory it
# writes to, and a human-readable title. The driver walks phases in order,
# advancing whenever the make/tune subprocess prints a matching marker. This
# gives the UI a truthful, per-step view (e.g. during `retune`, users see
# "Refreshing market data (1/3)" before "Fitting models (3/3)", instead of a
# stuck "Fitting models" bar while data is actually downloading).
# ---------------------------------------------------------------------------


class Phase:
    __slots__ = ("title", "kind", "dir", "suffix", "marker")

    def __init__(
        self,
        title: str,
        kind: str,
        dir_: Optional[Path],
        suffix: Optional[str],
        marker: Optional[re.Pattern],
    ) -> None:
        self.title = title
        self.kind = kind
        self.dir = dir_
        self.suffix = suffix
        self.marker = marker


def _phase_plan(mode: str) -> List[Phase]:
    if mode == "stocks":
        # Makefile `stocks` target prints `Step N/2: ...` markers. Include
        # fallbacks that match signals.py banners so the tune phase still
        # activates if the outer echo is filtered or delayed.
        return [
            Phase(
                "Refreshing market data",
                "download",
                PRICES_DIR,
                ".csv",
                re.compile(r"Step 1/2:\s*Refreshing", re.IGNORECASE),
            ),
            Phase(
                "Generating dashboard signals",
                "signals",
                None,
                None,
                re.compile(
                    r"Step 2/2:\s*Generating"
                    r"|▸\s*VALIDATION"
                    r"|▸\s*PROCESSING"
                    r"|\d+\s+assets\s+requested",
                    re.IGNORECASE,
                ),
            ),
        ]
    if mode == "tune":
        return [
            Phase("Fitting models", "tune", TUNE_DIR, ".json", None),
        ]
    if mode == "calibrate":
        return [
            Phase("Calibrating models", "tune", TUNE_DIR, ".json", None),
        ]
    if mode == "retune":
        # Makefile `retune` target prints these markers in order.
        return [
            Phase(
                "Refreshing market data",
                "download",
                PRICES_DIR,
                ".csv",
                re.compile(r"Step 1/3:\s*Refreshing", re.IGNORECASE),
            ),
            Phase(
                "Backing up previous tune",
                "backup",
                None,
                None,
                re.compile(r"Step 2/3:\s*Backing up", re.IGNORECASE),
            ),
            Phase(
                "Fitting models",
                "tune",
                TUNE_DIR,
                ".json",
                re.compile(r"Step 3/3:\s*Running tune", re.IGNORECASE),
            ),
        ]
    if mode == "tune-stocks":
        return [
            Phase(
                "Fitting models",
                "tune",
                TUNE_DIR,
                ".json",
                re.compile(r"Step 1/3:\s*Running tune", re.IGNORECASE),
            ),
            Phase(
                "Refreshing market data",
                "download",
                PRICES_DIR,
                ".csv",
                re.compile(r"Step 2/3:\s*Refreshing", re.IGNORECASE),
            ),
            Phase(
                "Generating dashboard signals",
                "signals",
                None,
                None,
                re.compile(
                    r"Step 3/3:\s*Generating"
                    r"|▸\s*VALIDATION"
                    r"|▸\s*PROCESSING"
                    r"|\d+\s+assets\s+requested",
                    re.IGNORECASE,
                ),
            ),
        ]
    return [Phase("Working", "work", None, None, None)]


class _WatcherState:
    """Mutable state shared with the background watcher thread.

    The main parser thread can swap the watched directory mid-run by updating
    ``dir``, ``suffix``, and ``baseline``. The watcher reads these under a lock.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dir: Optional[Path] = None
        self.suffix: Optional[str] = None
        self.baseline: Dict[str, Tuple[float, int]] = {}
        self.done_symbols: Set[str] = set()
        self.stop = threading.Event()


def _watch_loop(state: _WatcherState, poll_interval: float = 0.8) -> None:
    while not state.stop.is_set():
        with state.lock:
            dir_path = state.dir
            suffix = state.suffix
            baseline = state.baseline
            done = state.done_symbols
        if dir_path is not None and suffix is not None:
            current = _scan(dir_path, suffix)
            for symbol, (mtime, size) in current.items():
                if symbol in done:
                    continue
                prev = baseline.get(symbol)
                if prev is None or mtime > prev[0] + 0.01 or size != prev[1]:
                    with state.lock:
                        state.done_symbols.add(symbol)
                    _emit(
                        {
                            "event": "asset",
                            "symbol": symbol,
                            "status": "ok",
                            "detail": f"{size // 1024}k" if size > 1024 else f"{size}B",
                        }
                    )
        state.stop.wait(poll_interval)


def _heartbeat_loop(
    total: int,
    state: _WatcherState,
    start_ts: float,
    current_phase: List[Phase],
    current_idx: List[int],
    interval: float = 1.0,
) -> None:
    while not state.stop.wait(interval):
        with state.lock:
            phase_done = len(state.done_symbols)
        payload = {
            "event": "heartbeat",
            "done": phase_done,
            "total": total,
            "elapsed_s": round(time.time() - start_ts, 1),
        }
        if current_phase:
            payload["phase_step"] = current_idx[0] + 1
            payload["phase_title"] = current_phase[0].title
        _emit(payload)


def _run(mode: str) -> int:
    total = _estimate_total(mode)
    plan = _phase_plan(mode)
    visible_phase_count = 4 if mode == "retune" else len(plan)
    _emit({"event": "start", "mode": mode, "total_expected": total, "phase_count": visible_phase_count})

    state = _WatcherState()
    start_ts = time.time()
    current_phase_box: List[Phase] = []
    current_idx_box = [0]

    def activate_phase(index: int) -> None:
        """Switch to phase[index]: swap watcher dir + baseline, emit event."""
        if index < 0 or index >= len(plan):
            return
        phase = plan[index]
        with state.lock:
            # Take a fresh baseline snapshot so we only count files that change
            # during this phase, not pre-existing ones.
            if phase.dir is not None and phase.suffix is not None:
                state.dir = phase.dir
                state.suffix = phase.suffix
                state.baseline = _scan(phase.dir, phase.suffix)
                state.done_symbols = set()
            else:
                # Non-filesystem phase (e.g. "Backing up") - pause watcher.
                state.dir = None
                state.suffix = None
                state.baseline = {}
                state.done_symbols = set()
        current_phase_box.clear()
        current_phase_box.append(phase)
        current_idx_box[0] = index
        _emit(
            {
                "event": "phase",
                "title": phase.title,
                "kind": phase.kind,
                "step": index + 1,
                "total_steps": visible_phase_count,
            }
        )

    # Activate first phase immediately so the UI has a label from t=0.
    activate_phase(0)

    watcher = threading.Thread(target=_watch_loop, args=(state,), daemon=True)
    hb = threading.Thread(
        target=_heartbeat_loop,
        args=(total, state, start_ts, current_phase_box, current_idx_box),
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

    # Secondary markers inside the tune phase that give users a feel for
    # internal progress without having to parse asset-level output.
    sub_markers = [
        (re.compile(r"loading tuning cache|loading.*cache", re.IGNORECASE), "Loading tune cache"),
        (re.compile(r"assigning regimes|regime labels", re.IGNORECASE), "Assigning regimes"),
        (re.compile(r"computing bma|bayesian model averag", re.IGNORECASE), "Computing BMA weights"),
        (re.compile(r"writing cache|saving .*tune", re.IGNORECASE), "Writing cache"),
        (re.compile(r"pit calibration|calibrat(ing|ion)", re.IGNORECASE), "Calibrating PIT"),
    ]

    # Data-extraction regexes for log enrichment.
    # Tune logs emit the model pick in two ways:
    #   (a) Inline banner:  `  ✓ GBPJPY=X → φ-Gaussian-Unified`
    #   (b) Boxed symbol followed by a header+top-pick table, e.g.
    #         │    ALB    │
    #         │ Company · Sector │
    #         φ-T(ν=20)   45%   -11597   628
    #         Model     Weight    BIC      Hyv
    model_rx_inline = re.compile(r"^\s*✓\s*(\S+)\s+(?:→|->)\s+(.+?)\s*$")
    # Box content for a ticker (single token, no spaces) within │ … │.
    box_symbol_rx = re.compile(
        r"^\s*[│┃]\s+([A-Z0-9][A-Z0-9._=^\-]{0,14})\s+[│┃]\s*$"
    )
    # Header line that marks the previous non-empty line as the top-pick.
    model_header_rx = re.compile(
        r"^\s*Model\s+Weight\s+BIC\s+Hyv\s*$", re.IGNORECASE
    )
    # Top-pick line shape: "MODEL_NAME   NN%   -NNNN   NNNN" (flexible whitespace).
    model_pick_rx = re.compile(
        r"^\s*(.+?)\s{2,}(\d{1,3})%\s+(-?\d+)\s+(\d+)\s*$"
    )
    tune_result_symbol_rx = re.compile(r"^[A-Z0-9][A-Z0-9._=^\-]{0,14}$")
    # Refresh logs emit: `    Pass 2/5  (400 symbols)`, `    ● 192 ok  ·  305 pending`,
    # and `    ✓ 530/532 complete` when a pass finishes with no pending.
    pass_rx = re.compile(r"Pass\s+(\d+)\s*/\s*(\d+)")
    pass_progress_rx = re.compile(
        r"●\s*(\d+)\s*ok\s*[·|/•]\s*(\d+)\s*pending", re.IGNORECASE
    )
    pass_complete_rx = re.compile(
        r"✓\s*(\d+)\s*/\s*(\d+)\s*complete", re.IGNORECASE
    )
    signals_processed_rx = re.compile(
        r"✓\s*(\d+)\s+assets\s+processed", re.IGNORECASE
    )
    # Symbols known to have failed (parsed from refresh errors / tune fallbacks).
    fail_rx = re.compile(r"^\s*(?:✗|✘|FAILED|ERROR)[:\s]+(\S+)", re.IGNORECASE)
    # Python traceback capture so the UI can show the actual failure reason.
    traceback_start_rx = re.compile(r"^\s*Traceback\s*\(most recent call last\)")
    exception_line_rx = re.compile(
        r"^\s*([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt))\s*:\s*(.*)$"
    )

    last_sub_marker: Optional[str] = None
    seen_models: Set[str] = set()
    seen_asset_completions: Set[str] = set()
    current_pass: Tuple[int, int] = (0, 0)
    # Boxed-symbol/model-pick state for the two-line pattern above.
    last_boxed_symbol: Optional[str] = None
    prev_nonempty_line: str = ""
    # Traceback capture state.
    in_traceback: bool = False
    traceback_lines: list = []
    last_error: Optional[str] = None
    raw_log_line_count = 0
    latest_refresh_ok = 0
    latest_refresh_total = 0
    latest_signals_done = 0
    latest_signals_total = 0

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")

            # Phase transitions: check each remaining phase's marker.
            for i in range(current_idx_box[0] + 1, len(plan)):
                marker = plan[i].marker
                if marker is not None and marker.search(line):
                    activate_phase(i)
                    last_sub_marker = None
                    seen_models.clear()
                    current_pass = (0, 0)
                    last_boxed_symbol = None
                    prev_nonempty_line = ""
                    break

            current_phase = plan[current_idx_box[0]]

            if line:
                raw_log_line_count += 1
                # Market refresh can print a very large amount of output.
                # Keep semantic progress events accurate, but sample raw logs
                # during download phases so the router/browser are not fed a
                # line-by-line firehose while prices are refreshing.
                should_emit_raw_log = current_phase.kind != "download"
                if current_phase.kind == "download":
                    should_emit_raw_log = (
                        raw_log_line_count % 50 == 0
                        or pass_rx.search(line) is not None
                        or pass_progress_rx.search(line) is not None
                        or pass_complete_rx.search(line) is not None
                        or fail_rx.match(line) is not None
                    )
                elif current_phase.kind == "signals":
                    should_emit_raw_log = (
                        raw_log_line_count % 40 == 0
                        or "S I G N A L S" in line
                        or "Complete" in line
                        or "assets" in line.lower()
                        or "signals" in line.lower()
                        or fail_rx.match(line) is not None
                    )
                if should_emit_raw_log:
                    _emit_log(line)

            # Sub-phase hints (only within a tune-kind phase, where they fit).
            if current_phase.kind == "tune":
                for pat, label in sub_markers:
                    if pat.search(line) and label != last_sub_marker:
                        last_sub_marker = label
                        if label == "Calibrating PIT" and mode == "retune":
                            _emit(
                                {
                                    "event": "phase",
                                    "title": "Calibration",
                                    "kind": "calibration",
                                    "step": visible_phase_count,
                                    "total_steps": visible_phase_count,
                                }
                            )
                            break
                        _emit(
                            {
                                "event": "phase",
                                "title": f"{current_phase.title} — {label}",
                                "kind": "tune_sub",
                                "step": current_idx_box[0] + 1,
                                "total_steps": visible_phase_count,
                            }
                        )
                        break

                # Per-asset model selection. Two formats supported:
                #   (a) Inline banner: `✓ SYMBOL → ModelName`
                #   (b) Boxed symbol followed by top-pick + `Model Weight BIC Hyv` header.
                m = model_rx_inline.match(line)
                if m:
                    symbol = m.group(1)
                    model_name = m.group(2).strip()
                    key = f"{symbol}::{model_name}"
                    if key not in seen_models:
                        seen_models.add(key)
                        _emit(
                            {
                                "event": "model",
                                "symbol": symbol,
                                "model": model_name,
                            }
                        )
                        if symbol not in seen_asset_completions:
                            seen_asset_completions.add(symbol)
                            _emit(
                                {
                                    "event": "asset",
                                    "symbol": symbol,
                                    "status": "ok",
                                    "detail": model_name,
                                    "model": model_name,
                                }
                            )
                else:
                    # Track the most recent boxed ticker.
                    mb = box_symbol_rx.match(line)
                    if mb:
                        last_boxed_symbol = mb.group(1)
                    elif model_header_rx.match(line) and last_boxed_symbol and prev_nonempty_line:
                        mp_pick = model_pick_rx.match(prev_nonempty_line)
                        if mp_pick:
                            model_name = mp_pick.group(1).strip()
                            weight_pct = int(mp_pick.group(2))
                            bic = int(mp_pick.group(3))
                            hyv = int(mp_pick.group(4))
                            symbol = last_boxed_symbol
                            key = f"{symbol}::{model_name}"
                            if key not in seen_models:
                                seen_models.add(key)
                            _emit(
                                {
                                    "event": "model",
                                    "symbol": symbol,
                                    "model": model_name,
                                    "weight_pct": weight_pct,
                                    "bic": bic,
                                    "hyv": hyv,
                                }
                            )
                            if symbol not in seen_asset_completions:
                                seen_asset_completions.add(symbol)
                                # Also emit an asset-completion event so the
                                # overall processed counter advances during tune.
                                _emit(
                                    {
                                        "event": "asset",
                                        "symbol": symbol,
                                        "status": "ok",
                                        "detail": model_name,
                                        "model": model_name,
                                        "weight_pct": weight_pct,
                                        "bic": bic,
                                        "hyv": hyv,
                                    }
                                )

                    # Rich tune results table: │ Asset │ Model │ ... │ BIC │ Hyv │ CRPS │ PIT p │ St │
                    # These rows carry the premium metrics the UI should show as soon as they stream.
                    if "│" in line:
                        cols = [col.strip() for col in line.strip().strip("│").split("│")]
                        if len(cols) >= 13 and tune_result_symbol_rx.match(cols[0]) and cols[0].lower() != "asset":
                            symbol = cols[0]
                            model_name = cols[1]
                            bic_val = _parse_metric_float(cols[-5])
                            hyv_val = _parse_metric_float(cols[-4])
                            crps_val = _parse_metric_float(cols[-3])
                            pit_val = _parse_metric_float(cols[-2])
                            status_text = cols[-1]
                            if model_name and model_name.lower() != "model":
                                key = f"{symbol}::{model_name}"
                                if key not in seen_models:
                                    seen_models.add(key)
                                payload = {
                                    "event": "model",
                                    "symbol": symbol,
                                    "model": model_name,
                                    "bic": bic_val,
                                    "hyv": hyv_val,
                                    "crps": crps_val,
                                    "pit_p": pit_val,
                                    "fit_status": status_text,
                                }
                                _emit(payload)
                                if symbol not in seen_asset_completions:
                                    seen_asset_completions.add(symbol)
                                    _emit(
                                        {
                                            **payload,
                                            "event": "asset",
                                            "status": "ok",
                                            "detail": model_name,
                                        }
                                    )

            # Refresh-phase enrichment: pass number + ok/pending counters.
            if current_phase.kind == "download":
                mp = pass_rx.search(line)
                if mp:
                    p, total_p = int(mp.group(1)), int(mp.group(2))
                    if (p, total_p) != current_pass:
                        current_pass = (p, total_p)
                        _emit(
                            {
                                "event": "refresh",
                                "pass": p,
                                "total_passes": total_p,
                            }
                        )

            if current_phase.kind == "signals":
                msg = signals_processed_rx.search(line)
                if msg:
                    latest_signals_done = int(msg.group(1))
                    latest_signals_total = latest_signals_done
                    _emit(
                        {
                            "event": "progress",
                            "kind": "signals",
                            "done": latest_signals_done,
                            "fail": 0,
                            "total": latest_signals_total,
                        }
                    )
                mpp = pass_progress_rx.search(line)
                if mpp:
                    ok_n = int(mpp.group(1))
                    pending_n = int(mpp.group(2))
                    latest_refresh_ok = ok_n
                    latest_refresh_total = ok_n + pending_n
                    _emit(
                        {
                            "event": "refresh",
                            "pass": current_pass[0] or 1,
                            "total_passes": current_pass[1] or 1,
                            "ok": ok_n,
                            "pending": pending_n,
                        }
                    )
                else:
                    mpc = pass_complete_rx.search(line)
                    if mpc:
                        ok_n = int(mpc.group(1))
                        total_n = int(mpc.group(2))
                        latest_refresh_ok = ok_n
                        latest_refresh_total = total_n
                        _emit(
                            {
                                "event": "refresh",
                                "pass": current_pass[0] or 1,
                                "total_passes": current_pass[1] or 1,
                                "ok": ok_n,
                                "pending": max(0, total_n - ok_n),
                            }
                        )

            # Track the last non-empty rendered line for boxed-model lookups.
            if line.strip():
                prev_nonempty_line = line

            # Python traceback capture so the UI can show failure reason.
            if traceback_start_rx.match(line):
                in_traceback = True
                traceback_lines = [line]
            elif in_traceback:
                traceback_lines.append(line)
                mex = exception_line_rx.match(line)
                # An exception summary line ends the traceback.
                if mex:
                    last_error = f"{mex.group(1)}: {mex.group(2).strip()}"[:500]
                    in_traceback = False
                    _emit(
                        {
                            "event": "error",
                            "error_type": mex.group(1),
                            "message": mex.group(2).strip()[:500],
                        }
                    )
                    traceback_lines = []
                # Safety: don't accumulate unbounded if no summary arrives.
                elif len(traceback_lines) > 80:
                    in_traceback = False
                    traceback_lines = []

            # Failure detection (cross-phase).
            mf = fail_rx.match(line)
            if mf:
                sym = mf.group(1).rstrip(",:;")
                _emit(
                    {
                        "event": "asset",
                        "symbol": sym,
                        "status": "fail",
                        "detail": "error",
                    }
                )

        rc = proc.wait()
    finally:
        state.stop.set()
        watcher.join(timeout=2.0)
        hb.join(timeout=2.0)

    # Final sweep to catch files that landed in the last polling gap, using
    # whatever directory the final phase was watching.
    with state.lock:
        final_dir = state.dir
        final_suffix = state.suffix
        final_baseline = state.baseline
        final_done = state.done_symbols
    if final_dir is not None and final_suffix is not None:
        final = _scan(final_dir, final_suffix)
        for symbol, (mtime, size) in final.items():
            if symbol in final_done:
                continue
            prev = final_baseline.get(symbol)
            if prev is None or mtime > prev[0] + 0.01 or size != prev[1]:
                final_done.add(symbol)
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
            "done": max(len(final_done), latest_refresh_ok, latest_signals_done),
            "total": max(total, latest_refresh_total, latest_signals_total),
            "elapsed_s": round(time.time() - start_ts, 1),
            "exit_code": rc,
            "error": last_error,
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
