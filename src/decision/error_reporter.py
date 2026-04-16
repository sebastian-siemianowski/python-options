"""
Story 4.8: Unified Error Reporting Pipeline.

All errors from tuning, signal generation, and data fetching flow
through a single pipeline accessible via terminal and API.

Usage:
    from decision.error_reporter import report_error, get_recent_errors, Severity
    report_error(Severity.ERROR, "tuning", "AAPL", "Convergence failed")
"""
import json
import os
import glob
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
ERRORS_DIR = os.path.join(REPO_ROOT, "data", "errors")

# Rolling retention
ERROR_RETENTION_DAYS = 7


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class ErrorRecord:
    """Single error record."""
    timestamp: str
    source: str
    severity: str
    asset: str
    message: str
    stack_trace: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# In-memory buffer for the current session
_error_buffer: List[ErrorRecord] = []


def report_error(
    severity: Severity,
    source: str,
    asset: str,
    message: str,
    stack_trace: str = "",
):
    """
    Record an error to the pipeline.
    
    Appends to in-memory buffer and flushes to daily JSON file.
    """
    record = ErrorRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
        severity=severity.name,
        asset=asset,
        message=message,
        stack_trace=stack_trace,
    )
    _error_buffer.append(record)
    _flush_to_file(record)
    return record


def _flush_to_file(record: ErrorRecord):
    """Append error to today's error log file."""
    os.makedirs(ERRORS_DIR, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(ERRORS_DIR, f"errors_{date_str}.json")
    
    # Load existing or start fresh
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                records = json.load(f)
        except (json.JSONDecodeError, IOError):
            records = []
    
    records.append(record.to_dict())
    
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def get_recent_errors(
    days: int = 1,
    severity_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[dict]:
    """
    Get recent errors from log files.
    
    Args:
        days: How many days back to look
        severity_filter: e.g. "ERROR" to filter
        source_filter: e.g. "tuning" to filter
    
    Returns list of error dicts, newest first.
    """
    if not os.path.exists(ERRORS_DIR):
        return []
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    
    all_errors = []
    pattern = os.path.join(ERRORS_DIR, "errors_*.json")
    
    for filepath in sorted(glob.glob(pattern), reverse=True):
        basename = os.path.basename(filepath)
        file_date = basename.replace("errors_", "").replace(".json", "")
        if file_date < cutoff_str:
            continue
        
        try:
            with open(filepath, "r") as f:
                records = json.load(f)
                all_errors.extend(records)
        except (json.JSONDecodeError, IOError):
            continue
    
    # Apply filters
    if severity_filter:
        all_errors = [e for e in all_errors if e.get("severity") == severity_filter]
    if source_filter:
        all_errors = [e for e in all_errors if e.get("source") == source_filter]
    
    # Sort by timestamp descending
    all_errors.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    
    return all_errors


def prune_old_errors(retention_days: int = ERROR_RETENTION_DAYS) -> int:
    """Remove error logs older than retention_days."""
    if not os.path.exists(ERRORS_DIR):
        return 0
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    
    removed = 0
    pattern = os.path.join(ERRORS_DIR, "errors_*.json")
    
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        file_date = basename.replace("errors_", "").replace(".json", "")
        if file_date < cutoff_str:
            os.remove(filepath)
            removed += 1
    
    return removed


def clear_buffer():
    """Clear in-memory error buffer."""
    _error_buffer.clear()


def get_buffer() -> List[ErrorRecord]:
    """Get current session errors."""
    return list(_error_buffer)
