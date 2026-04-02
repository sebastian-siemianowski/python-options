"""
FastAPI application — Signal Engine Web Dashboard.

Start with:
  cd src && uvicorn web.backend.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure src/ is on the Python path for importing existing modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from web.backend.ws import manager
from web.backend.routers import (
    overview,
    signals,
    risk,
    charts,
    tuning,
    data,
    arena,
    tasks,
    services,
    diagnostics,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup: ensure output directories exist
    for subdir in ["plots/signals", "plots/sma", "plots/index",
                    "high_conviction/buy", "high_conviction/sell",
                    "tune", "prices"]:
        os.makedirs(os.path.join(SRC_DIR, "data", subdir), exist_ok=True)
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Signal Engine Dashboard",
    description="Quantitative signal engine with Bayesian Model Averaging",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount chart images as static files
plots_dir = os.path.join(SRC_DIR, "data", "plots")
if os.path.isdir(plots_dir):
    app.mount("/static/plots", StaticFiles(directory=plots_dir), name="plots")

# ── Register routers ─────────────────────────────────────────────────────────

app.include_router(overview.router, prefix="/api", tags=["overview"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(risk.router, prefix="/api/risk", tags=["risk"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(tuning.router, prefix="/api/tune", tags=["tuning"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(arena.router, prefix="/api/arena", tags=["arena"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(services.router, prefix="/api/services", tags=["services"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "signal-engine"}
