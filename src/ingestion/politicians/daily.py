"""Daily politician disclosure sync orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from ingestion.politicians.parse import parse_local_raw_artifacts
from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso
from ingestion.politicians.sync import default_incremental_window, run_incremental_sync
from ingestion.politicians.validation import validate_trade_record


StepFn = Callable[..., dict[str, Any]]


def run_daily_politicians_sync(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    data_root: str | Path | None = None,
    sync_fn: StepFn | None = None,
    parse_fn: StepFn | None = None,
    validate_fn: StepFn | None = None,
    cache_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    console=None,
) -> dict[str, Any]:
    """Run source sync, parse, validation, and API-cache refresh as one task."""
    root = ensure_politicians_data_dirs(data_root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    offline_mode = _offline_mode_enabled()
    if date_from is None or date_to is None:
        default_from, default_to = default_incremental_window()
        date_from = date_from or default_from
        date_to = date_to or default_to

    steps: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    if offline_mode:
        steps["sync_sources"] = {
            "status": "skipped",
            "reason": "OFFLINE_MODE=1",
            "date_window": {"from": date_from, "to": date_to},
        }
    else:
        sync_step_fn = sync_fn or run_incremental_sync
        steps["sync_sources"] = _run_step(
            "sync_sources",
            lambda: sync_step_fn(date_from=date_from, date_to=date_to, data_root=root, console=console),
            errors,
        )

    parser_step_fn = parse_fn or parse_local_raw_artifacts
    steps["parse_artifacts"] = _run_step(
        "parse_artifacts",
        lambda: parser_step_fn(data_root=root),
        errors,
    )

    validation_step_fn = validate_fn or validate_normalized_output
    steps["validate_output"] = _run_step(
        "validate_output",
        lambda: validation_step_fn(data_root=root),
        errors,
    )

    refresh_fn = cache_refresh_fn or _refresh_api_cache
    steps["refresh_api_cache"] = _run_step("refresh_api_cache", refresh_fn, errors)

    status = _overall_status(steps, errors)
    summary = {
        "task": "politicians_daily_sync",
        "status": status,
        "offline_mode": offline_mode,
        "idempotency_key": f"{date_from}:{date_to}:offline={offline_mode}",
        "safe_to_rerun": True,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "date_window": {"from": date_from, "to": date_to},
        "steps": steps,
        "counts": _aggregate_counts(steps),
        "errors": errors,
    }
    _write_run_summary(summary, runs_dir)
    return summary


def validate_normalized_output(*, data_root: str | Path | None = None) -> dict[str, Any]:
    """Validate the current normalized trades JSONL without rewriting it."""
    root = ensure_politicians_data_dirs(data_root)
    trades_path = root / "trades.jsonl"
    parse_errors_path = root / "parse_errors.jsonl"
    if not trades_path.exists():
        return {
            "status": "degraded",
            "trade_count": 0,
            "valid_count": 0,
            "error_count": 1,
            "errors": ["missing_trades_jsonl"],
        }

    valid_count = 0
    error_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_jsonl(trades_path)):
        ok, row_errors = validate_trade_record(row)
        if ok:
            valid_count += 1
        else:
            error_rows.append({
                "row_index": idx,
                "source": row.get("source"),
                "report_id": row.get("report_id"),
                "errors": row_errors,
            })
    return {
        "status": "ok" if not error_rows else "valid_with_errors",
        "trade_count": valid_count + len(error_rows),
        "valid_count": valid_count,
        "error_count": len(error_rows),
        "parse_error_count": _count_jsonl(parse_errors_path),
        "errors": error_rows,
    }


def _run_step(name: str, fn: Callable[[], dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    try:
        result = fn()
    except Exception as exc:
        message = f"{name}:{type(exc).__name__}: {exc}"
        errors.append(message)
        return {"status": "error", "errors": [message]}
    if not isinstance(result, dict):
        result = {"status": "ok", "result": result}
    status = str(result.get("status", "ok"))
    if status in {"degraded", "error", "failed", "valid_with_errors"}:
        errors.extend(_step_errors(name, result))
    return result


def _step_errors(name: str, result: dict[str, Any]) -> list[str]:
    values = result.get("errors") or result.get("error_summary") or []
    if isinstance(values, dict):
        return [f"{name}:{key}={value}" for key, value in values.items()]
    if isinstance(values, list):
        return [f"{name}:{value}" for value in values]
    if values:
        return [f"{name}:{values}"]
    return [f"{name}:{result.get('status')}"]


def _overall_status(steps: dict[str, dict[str, Any]], errors: list[str]) -> str:
    if errors:
        return "degraded"
    statuses = {str(step.get("status", "ok")) for step in steps.values()}
    degraded = {"degraded", "error", "failed", "valid_with_errors"}
    if statuses & degraded:
        return "degraded"
    return "ok"


def _aggregate_counts(steps: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in steps.values():
        nested_counts = result.get("counts")
        if isinstance(nested_counts, dict):
            for key, value in nested_counts.items():
                _add_count(counts, key, value)
        for key, value in result.items():
            if key.endswith("_count") or key in {"trade_count", "valid_count", "error_count"}:
                _add_count(counts, key, value)
        result_rows = result.get("results", [])
        if not isinstance(result_rows, list):
            continue
        for row in result_rows:
            if isinstance(row, dict) and isinstance(row.get("counts"), dict):
                for key, value in row["counts"].items():
                    _add_count(counts, key, value)
    return dict(sorted(counts.items()))


def _add_count(counts: dict[str, int], key: str, value: Any) -> None:
    try:
        counts[key] = counts.get(key, 0) + int(value)
    except (TypeError, ValueError):
        return


def _write_run_summary(summary: dict[str, Any], runs_dir: Path) -> None:
    safe_started = str(summary["started_at"]).replace(":", "-")
    latest_path = runs_dir / "daily_sync_latest.json"
    history_path = runs_dir / f"daily_sync_{safe_started}.json"
    summary["summary_path"] = str(latest_path)
    summary["history_path"] = str(history_path)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    latest_path.write_text(payload, encoding="utf-8")
    history_path.write_text(payload, encoding="utf-8")


def _refresh_api_cache() -> dict[str, Any]:
    try:
        from web.backend.services.politicians_service import invalidate_politicians_cache
    except Exception as exc:
        return {"status": "degraded", "errors": [f"cache_refresh_unavailable:{type(exc).__name__}: {exc}"]}
    return invalidate_politicians_cache()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _offline_mode_enabled() -> bool:
    return os.getenv("OFFLINE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
