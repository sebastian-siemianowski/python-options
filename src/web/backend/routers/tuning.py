"""
Tuning router — tuning cache, PIT calibration status, model weights, retune SSE.
"""

import asyncio
import os
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
    valid_modes = {"retune", "tune", "calibrate"}
    if mode not in valid_modes:
        mode = "retune"

    async def event_generator():
        yield f"data: {_sse_json('start', f'Starting make {mode}...')}\n\n"

        try:
            process = await asyncio.create_subprocess_exec(
                "make", mode,
                cwd=REPO_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "TERM": "dumb",
                    "NO_COLOR": "1",
                    "COLUMNS": "120",
                    "TUNING_QUIET": "0",
                },
            )

            asset_count = 0
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue

                # Detect asset progress lines (tune_ux.py outputs progress)
                if any(marker in text for marker in ["✓", "✗", "tuning", "Tuning", "Processing"]):
                    asset_count += 1
                    yield f"data: {_sse_json('progress', text, asset_count)}\n\n"
                elif any(marker in text for marker in ["Step", "RETUNE", "═", "✅", "📥", "📦", "🎛"]):
                    yield f"data: {_sse_json('phase', text)}\n\n"
                else:
                    yield f"data: {_sse_json('log', text)}\n\n"

            await process.wait()
            status = "completed" if process.returncode == 0 else "failed"
            yield f"data: {_sse_json(status, f'make {mode} finished (exit code {process.returncode})', asset_count)}\n\n"

        except Exception as e:
            yield f"data: {_sse_json('error', str(e))}\n\n"

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


def _sse_json(event_type: str, message: str, count: int = 0) -> str:
    """Format an SSE data payload with ANSI codes stripped."""
    import json
    return json.dumps({"type": event_type, "message": _strip_ansi(message), "count": count})
