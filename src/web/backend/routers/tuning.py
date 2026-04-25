"""
Tuning router — tuning cache, PIT calibration status, model weights, retune SSE.
"""

import asyncio
import json
import os
import signal
import sys
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from web.backend.services.tune_service import (
    list_tuned_assets,
    get_tune_detail,
    get_pit_failures,
    get_tune_stats,
    _invalidate_tune_cache,
)

router = APIRouter()

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
_ACTIVE_RETUNE_PROCESSES: set[asyncio.subprocess.Process] = set()
_ACTIVE_RETUNE_LOCK = asyncio.Lock()
_RAW_LOG_FLUSH_INTERVAL_S = 0.35
_RAW_LOG_MAX_BATCH_LINES = 24
_RAW_LOG_MAX_BATCH_CHARS = 5000


async def _terminate_process_tree(process: asyncio.subprocess.Process, timeout: float = 5.0) -> bool:
    """Terminate a running job-driver process and its child process group."""
    if process.returncode is not None:
        return False

    def _signal_group(sig: int) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), sig)
            else:
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
        except ProcessLookupError:
            pass

    _signal_group(signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        _signal_group(signal.SIGKILL)
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            return False
    return True


@router.post("/retune/cancel")
async def cancel_retune():
    """Cancel any active retune/tune/stocks streaming subprocesses."""
    async with _ACTIVE_RETUNE_LOCK:
        processes = list(_ACTIVE_RETUNE_PROCESSES)

    stopped = 0
    for process in processes:
        if await _terminate_process_tree(process):
            stopped += 1

    return {"status": "ok", "active": len(processes), "stopped": stopped}


@router.get("/list")
async def tune_list():
    """List all tuned assets with summary info."""
    assets = list_tuned_assets()
    return {"assets": assets, "total": len(assets)}


@router.post("/refresh-cache")
async def refresh_tune_cache():
    """Invalidate the in-memory tune cache, forcing a reload on next request."""
    _invalidate_tune_cache()
    return {"status": "ok", "message": "Tune cache invalidated"}


@router.get("/stats")
async def tune_stats():
    """Tuning cache statistics."""
    return get_tune_stats()


@router.get("/pit-failures")
async def pit_failures():
    """Assets failing PIT calibration (AD test)."""
    failures = get_pit_failures()
    return {"failures": failures, "count": len(failures)}


@router.get("/detail/{symbol}")
async def tune_detail(symbol: str):
    """Full tuning detail for a single asset."""
    detail = get_tune_detail(symbol)
    if detail is None:
        return {"error": f"No tuning data for {symbol}"}
    return {"symbol": symbol, "data": detail}


@router.get("/retune/stream")
async def retune_stream(mode: str = "retune"):
    """
    Stream retune progress via Server-Sent Events.

    Runs `make retune` (or `make tune` / `make calibrate`) as a subprocess and
    streams stdout/stderr line by line.

    Query params:
      mode: "retune" (default), "tune", "calibrate"
    """
    valid_modes = {"retune", "tune", "calibrate", "stocks"}
    if mode not in valid_modes:
        mode = "retune"

    async def event_generator():
        process = None
        raw_log_buffer: list[str] = []
        last_raw_log_flush = asyncio.get_running_loop().time()

        def _buffer_raw_log(message: str) -> None:
            cleaned = _strip_ansi(message).strip()
            if cleaned:
                raw_log_buffer.append(cleaned[:800])

        def _pop_raw_log_payload() -> Optional[str]:
            nonlocal last_raw_log_flush
            if not raw_log_buffer:
                return None
            payload = "\n".join(raw_log_buffer)
            raw_log_buffer.clear()
            last_raw_log_flush = asyncio.get_running_loop().time()
            if len(payload) > _RAW_LOG_MAX_BATCH_CHARS:
                payload = payload[-_RAW_LOG_MAX_BATCH_CHARS:]
            return payload

        def _should_flush_raw_logs() -> bool:
            if len(raw_log_buffer) >= _RAW_LOG_MAX_BATCH_LINES:
                return True
            return asyncio.get_running_loop().time() - last_raw_log_flush >= _RAW_LOG_FLUSH_INTERVAL_S

        try:
            process_kwargs = {"start_new_session": True} if hasattr(os, "setsid") else {}
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "web.backend.job_driver", mode,
                cwd=REPO_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "TERM": "dumb",
                    "NO_COLOR": "1",
                    "COLUMNS": "120",
                    "PYTHONPATH": f"{os.path.join(REPO_ROOT, 'src')}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                },
                **process_kwargs,
            )
            async with _ACTIVE_RETUNE_LOCK:
                _ACTIVE_RETUNE_PROCESSES.add(process)

            done_count = 0
            fail_count = 0
            total_expected = 0
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue

                if text.startswith("@@EVT@@ "):
                    # Semantic event from driver — forward as typed SSE event
                    try:
                        payload = json.loads(text[len("@@EVT@@ "):])
                    except json.JSONDecodeError:
                        continue
                    ev = payload.get("event")
                    if ev == "start":
                        total_expected = int(payload.get("total_expected", 0))
                        yield f"data: {json.dumps({'type': 'start', 'mode': payload.get('mode'), 'total': total_expected, 'phase_count': payload.get('phase_count', 1)})}\n\n"
                    elif ev == "phase":
                        yield f"data: {json.dumps({'type': 'phase', 'title': payload.get('title', ''), 'step': payload.get('step'), 'total_steps': payload.get('total_steps'), 'kind': payload.get('kind')})}\n\n"
                    elif ev == "asset":
                        status = payload.get("status", "ok")
                        if status == "fail":
                            fail_count += 1
                        else:
                            done_count += 1
                        yield f"data: {json.dumps({'type': 'asset', 'symbol': payload.get('symbol'), 'status': status, 'detail': payload.get('detail'), 'done': done_count, 'fail': fail_count, 'total': total_expected})}\n\n"
                    elif ev == "model":
                        # Per-asset model selection (e.g. 'Student-t+EVTH').
                        yield f"data: {json.dumps({'type': 'model', 'symbol': payload.get('symbol'), 'model': payload.get('model')})}\n\n"
                    elif ev == "refresh":
                        yield f"data: {json.dumps({'type': 'refresh', 'pass': payload.get('pass'), 'total_passes': payload.get('total_passes'), 'ok': payload.get('ok'), 'pending': payload.get('pending')})}\n\n"
                    elif ev == "heartbeat":
                        yield f"data: {json.dumps({'type': 'heartbeat', 'done': payload.get('done', 0), 'total': payload.get('total', 0), 'elapsed_s': payload.get('elapsed_s', 0), 'phase_step': payload.get('phase_step'), 'phase_title': payload.get('phase_title')})}\n\n"
                    elif ev == "error":
                        # Structured error extracted from a Python traceback.
                        yield f"data: {json.dumps({'type': 'error', 'error_type': payload.get('error_type'), 'message': payload.get('message', '')})}\n\n"
                    elif ev == "done":
                        status = "completed" if payload.get("status") == "ok" else "failed"
                        yield f"data: {json.dumps({'type': status, 'done': payload.get('done', 0), 'total': payload.get('total', 0), 'elapsed_s': payload.get('elapsed_s', 0), 'exit_code': payload.get('exit_code', -1), 'error': payload.get('error')})}\n\n"
                elif text.startswith("@@LOG@@ "):
                    # Raw logs can be extremely chatty during data refreshes.
                    # Batch them before sending SSE so the browser's main thread
                    # is not starved before the Stop click can be handled.
                    _buffer_raw_log(text[len("@@LOG@@ "):])
                    if _should_flush_raw_logs():
                        payload = _pop_raw_log_payload()
                        if payload:
                            yield f"data: {json.dumps({'type': 'log', 'message': payload})}\n\n"
                else:
                    # Unstructured line (e.g. driver crash message)
                    _buffer_raw_log(text)
                    if _should_flush_raw_logs():
                        payload = _pop_raw_log_payload()
                        if payload:
                            yield f"data: {json.dumps({'type': 'log', 'message': payload})}\n\n"

            payload = _pop_raw_log_payload()
            if payload:
                yield f"data: {json.dumps({'type': 'log', 'message': payload})}\n\n"

            await process.wait()
            if process.returncode != 0:
                yield f"data: {json.dumps({'type': 'failed', 'exit_code': process.returncode})}\n\n"

        except asyncio.CancelledError:
            if process:
                await _terminate_process_tree(process)
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            if process and process.returncode is None:
                await _terminate_process_tree(process)
        finally:
            if process:
                async with _ACTIVE_RETUNE_LOCK:
                    _ACTIVE_RETUNE_PROCESSES.discard(process)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


import re as _re
_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\]\d+;[^\x07]*\x07|\x1b[^\[\]][^a-zA-Z]*[a-zA-Z]')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub('', text)


def _sse_json(event_type: str, message: str, count: int = 0, success: int = 0, fail: int = 0) -> str:
    """Format an SSE data payload with ANSI codes stripped."""
    import json
    return json.dumps({
        "type": event_type,
        "message": _strip_ansi(message),
        "count": count,
        "success": success,
        "fail": fail,
    })
