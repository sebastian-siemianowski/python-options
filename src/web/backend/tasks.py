"""
Celery tasks — background computation for signals, risk, tuning, data refresh.

These tasks wrap the existing CLI entry points so the web UI can trigger
long-running operations without blocking the API.
"""

import os
import sys
import subprocess
import json
from typing import Any, Dict, Optional

from web.backend.celery_app import celery_app

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")

# Use venv python if available, otherwise sys.executable
PYTHON = VENV_PYTHON if os.path.isfile(VENV_PYTHON) else sys.executable


def _run_command(cmd: list, task, task_type: str, timeout: int = 3600) -> Dict[str, Any]:
    """
    Run a CLI command as a subprocess and report progress via task state.
    
    Args:
        cmd: Command list (e.g., [PYTHON, "-m", "decision.signals"])
        task: Celery task instance (for updating state)
        task_type: Human-readable task type
        timeout: Maximum seconds to wait
    
    Returns:
        Dict with status, stdout, stderr, returncode
    """
    task.update_state(state="PROGRESS", meta={
        "task_type": task_type,
        "progress": 0,
        "message": f"Starting {task_type}...",
    })

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SRC_DIR,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": SRC_DIR},
        )

        success = result.returncode == 0
        return {
            "status": "completed" if success else "failed",
            "task_type": task_type,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "task_type": task_type,
            "message": f"{task_type} timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "status": "error",
            "task_type": task_type,
            "message": str(e),
        }


@celery_app.task(bind=True, name="compute_signals")
def compute_signals_task(self, args: Optional[list] = None):
    """Run signal computation (equivalent to `make stocks`)."""
    cmd = [PYTHON, "-m", "decision.signals"]
    if args:
        cmd.extend(args)
    return _run_command(cmd, self, "Signal Computation")


@celery_app.task(bind=True, name="refresh_data")
def refresh_data_task(self, symbols: Optional[list] = None):
    """Refresh price data from Yahoo Finance."""
    cmd = [PYTHON, "-m", "data_ops.refresh_data"]
    if symbols:
        cmd.extend(["--symbols"] + symbols)
    return _run_command(cmd, self, "Data Refresh")


@celery_app.task(bind=True, name="run_tuning")
def run_tuning_task(self, symbols: Optional[list] = None):
    """Run model tuning (equivalent to `make tune`)."""
    cmd = [PYTHON, "-m", "tuning.tune"]
    if symbols:
        cmd.extend(["--symbols"] + symbols)
    return _run_command(cmd, self, "Model Tuning")


@celery_app.task(bind=True, name="compute_risk")
def compute_risk_task(self):
    """Compute risk dashboard (equivalent to `make risk`)."""
    from web.backend.services.risk_service import compute_risk_json
    self.update_state(state="PROGRESS", meta={
        "task_type": "Risk Dashboard",
        "progress": 0,
        "message": "Computing risk dashboard...",
    })
    result = compute_risk_json()
    return {
        "status": "completed",
        "task_type": "Risk Dashboard",
        "result": result,
    }


@celery_app.task(bind=True, name="generate_charts")
def generate_charts_task(self):
    """Generate signal chart PNGs."""
    cmd = [PYTHON, "-m", "decision.signal_charts"]
    return _run_command(cmd, self, "Chart Generation")
