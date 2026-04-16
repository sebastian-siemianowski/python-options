"""
Story 4.4: Task Pipeline Orchestration.

Orchestrates: prices -> incremental tune -> signals -> cache update
with proper dependency ordering and failure handling.

Usage:
    from decision.pipeline import run_pipeline, PipelineResult
    result = run_pipeline()
    if result.failed_phase:
        print(f"Failed at: {result.failed_phase}")
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(REPO_ROOT, "data")

PIPELINE_STATUS_PATH = os.path.join(DATA_DIR, "pipeline_status.json")

# Failure thresholds
TUNE_FAIL_WARN_PCT = 0.05   # Warn if >5% fail
TUNE_FAIL_STOP_PCT = 0.50   # Stop if >50% fail


@dataclass
class PhaseResult:
    """Result of a single pipeline phase."""
    phase: str
    status: str = "not_started"  # not_started, running, success, warning, error
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    assets_processed: int = 0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0
    phases: Dict[str, PhaseResult] = field(default_factory=dict)
    failed_phase: Optional[str] = None
    overall_status: str = "not_started"  # success, warning, error

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _run_phase(
    name: str,
    func: Callable,
    pipeline_result: PipelineResult,
    **kwargs,
) -> PhaseResult:
    """Execute a single pipeline phase with timing and error handling."""
    phase = PhaseResult(phase=name)
    phase.status = "running"
    phase.started_at = datetime.now(timezone.utc).isoformat()
    
    start = time.monotonic()
    try:
        result = func(**kwargs)
        phase.duration_seconds = time.monotonic() - start
        phase.completed_at = datetime.now(timezone.utc).isoformat()
        
        if isinstance(result, dict):
            phase.assets_processed = result.get("assets_processed", 0)
            phase.errors = result.get("errors", [])
            phase.details = result
        
        # Check failure thresholds for tune phase
        if name == "tune" and isinstance(result, dict):
            total = result.get("total_assets", 1)
            failed = len(result.get("errors", []))
            fail_pct = failed / max(total, 1)
            
            if fail_pct > TUNE_FAIL_STOP_PCT:
                phase.status = "error"
            elif fail_pct > TUNE_FAIL_WARN_PCT:
                phase.status = "warning"
            else:
                phase.status = "success"
        else:
            phase.status = "success" if not phase.errors else "warning"
    
    except Exception as e:
        phase.duration_seconds = time.monotonic() - start
        phase.completed_at = datetime.now(timezone.utc).isoformat()
        phase.status = "error"
        phase.errors.append(str(e))
    
    pipeline_result.phases[name] = phase
    return phase


def run_pipeline(
    phases: Optional[List[str]] = None,
    phase_functions: Optional[Dict[str, Callable]] = None,
) -> PipelineResult:
    """
    Run the full pipeline: prices -> tune -> signals.
    
    Args:
        phases: Override which phases to run (default: all)
        phase_functions: Map of phase name to callable.
            Each callable should return dict with {assets_processed, errors, total_assets}.
    
    Returns:
        PipelineResult with per-phase status.
    """
    all_phases = phases or ["prices", "tune", "signals"]
    phase_functions = phase_functions or {}
    
    result = PipelineResult()
    result.started_at = datetime.now(timezone.utc).isoformat()
    result.overall_status = "running"
    
    start = time.monotonic()
    
    for phase_name in all_phases:
        func = phase_functions.get(phase_name)
        if func is None:
            # No function provided -> skip
            pr = PhaseResult(phase=phase_name, status="skipped")
            result.phases[phase_name] = pr
            continue
        
        pr = _run_phase(phase_name, func, result)
        
        if pr.status == "error":
            result.failed_phase = phase_name
            result.overall_status = "error"
            break
    
    result.total_duration_seconds = time.monotonic() - start
    result.completed_at = datetime.now(timezone.utc).isoformat()
    
    if result.overall_status != "error":
        has_warnings = any(
            p.status == "warning" for p in result.phases.values()
        )
        result.overall_status = "warning" if has_warnings else "success"
    
    # Save status
    save_pipeline_status(result)
    
    return result


def save_pipeline_status(result: PipelineResult):
    """Save pipeline status to JSON."""
    os.makedirs(os.path.dirname(PIPELINE_STATUS_PATH), exist_ok=True)
    try:
        with open(PIPELINE_STATUS_PATH, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    except Exception:
        pass


def load_pipeline_status() -> Optional[dict]:
    """Load last pipeline status."""
    if not os.path.exists(PIPELINE_STATUS_PATH):
        return None
    with open(PIPELINE_STATUS_PATH, "r") as f:
        return json.load(f)
