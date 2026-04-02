"""
Tasks router — trigger and monitor background Celery tasks.
"""

from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ComputeSignalsRequest(BaseModel):
    args: Optional[List[str]] = None


class RefreshDataRequest(BaseModel):
    symbols: Optional[List[str]] = None


class RunTuningRequest(BaseModel):
    symbols: Optional[List[str]] = None


@router.post("/signals/compute")
async def trigger_signals(request: ComputeSignalsRequest = ComputeSignalsRequest()):
    """Trigger signal computation in background."""
    from web.backend.tasks import compute_signals_task
    task = compute_signals_task.delay(args=request.args)
    return {"task_id": task.id, "task_type": "Signal Computation", "status": "queued"}


@router.post("/data/refresh")
async def trigger_data_refresh(request: RefreshDataRequest = RefreshDataRequest()):
    """Trigger data refresh from Yahoo Finance."""
    from web.backend.tasks import refresh_data_task
    task = refresh_data_task.delay(symbols=request.symbols)
    return {"task_id": task.id, "task_type": "Data Refresh", "status": "queued"}


@router.post("/tune/run")
async def trigger_tuning(request: RunTuningRequest = RunTuningRequest()):
    """Trigger model tuning in background."""
    from web.backend.tasks import run_tuning_task
    task = run_tuning_task.delay(symbols=request.symbols)
    return {"task_id": task.id, "task_type": "Model Tuning", "status": "queued"}


@router.post("/risk/compute")
async def trigger_risk():
    """Trigger risk dashboard computation in background."""
    from web.backend.tasks import compute_risk_task
    task = compute_risk_task.delay()
    return {"task_id": task.id, "task_type": "Risk Dashboard", "status": "queued"}


@router.post("/charts/generate")
async def trigger_charts():
    """Trigger chart generation in background."""
    from web.backend.tasks import generate_charts_task
    task = generate_charts_task.delay()
    return {"task_id": task.id, "task_type": "Chart Generation", "status": "queued"}


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    """Check status of a background task."""
    from web.backend.celery_app import celery_app
    result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.status == "PROGRESS":
        response["meta"] = result.info
    elif result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)

    return response
