"""Sync orchestration and sync-state persistence for politician disclosures."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso
from ingestion.politicians.sources.house import download_house_ptr_pdfs, download_house_yearly_archive, load_house_filings_from_archives
from ingestion.politicians.sources.senate import search_senate_ptr_filings


def read_sync_state(data_root: str | Path | None = None) -> dict[str, Any]:
    """Read sync_state.json, returning the empty structure when missing."""
    root = ensure_politicians_data_dirs(data_root)
    path = root / "sync_state.json"
    if not path.exists():
        return {"updated_at": None, "sources": {}, "backfills": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": None, "sources": {}, "backfills": {}}
    state.setdefault("sources", {})
    state.setdefault("backfills", {})
    return state


def write_sync_state(state: dict[str, Any], data_root: str | Path | None = None) -> dict[str, Any]:
    """Persist sync_state.json."""
    root = ensure_politicians_data_dirs(data_root)
    path = root / "sync_state.json"
    state["updated_at"] = utc_now_iso()
    state.setdefault("sources", {})
    state.setdefault("backfills", {})
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state


def default_incremental_window(date_to: date | None = None, days: int = 14) -> tuple[str, str]:
    """Return the default inclusive incremental sync window."""
    end = date_to or datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def run_incremental_sync(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    data_root: str | Path | None = None,
    house_sync_fn: Callable[..., dict[str, Any]] | None = None,
    senate_sync_fn: Callable[..., dict[str, Any]] | None = None,
    post_source_fn: Callable[[str, dict[str, Any]], Any] | None = None,
    console=None,
) -> dict[str, Any]:
    """Run independent source syncs and update sync_state.json."""
    if date_to is None:
        to_date = datetime.now(timezone.utc).date()
    else:
        to_date = date.fromisoformat(date_to)
    if date_from is None:
        date_from, date_to = default_incremental_window(to_date)
    else:
        date_to = to_date.isoformat()

    root = ensure_politicians_data_dirs(data_root)
    results: list[dict[str, Any]] = []
    source_fns: list[tuple[str, Callable[..., dict[str, Any]]]] = [
        ("house", house_sync_fn or _default_house_incremental_sync),
        ("senate", senate_sync_fn or search_senate_ptr_filings),
    ]

    for source, sync_fn in source_fns:
        try:
            manifest = sync_fn(date_from=date_from, date_to=date_to, data_root=root)
            result = _result_from_manifest(source, date_from, date_to, manifest)
        except Exception as exc:
            result = {
                "source": source,
                "status": "degraded",
                "date_window": {"from": date_from, "to": date_to},
                "counts": {},
                "error_summary": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(result)
        if post_source_fn is not None:
            post_source_fn(source, result)

    state = read_sync_state(root)
    for result in results:
        _merge_source_sync_state(state, result)
    write_sync_state(state, root)
    render_sync_summary(results, console=console)
    return {"date_window": {"from": date_from, "to": date_to}, "results": results, "state": state}


def run_house_backfill(
    year: int,
    *,
    data_root: str | Path | None = None,
    backfill_fn: Callable[..., dict[str, Any]] | None = None,
    fetch_pdfs_fn: Callable[..., dict[str, Any]] | None = None,
    console=None,
) -> dict[str, Any]:
    """Run House yearly backfill and preserve unrelated backfill years."""
    root = ensure_politicians_data_dirs(data_root)
    backfill = backfill_fn or download_house_yearly_archive
    fetch_pdfs = fetch_pdfs_fn or download_house_ptr_pdfs
    errors: list[str] = []
    counts: dict[str, Any] = {}
    status = "ok"
    try:
        archive_manifest = backfill(year, data_root=root)
        counts.update(_counts_from_manifest(archive_manifest))
        pdf_manifest = fetch_pdfs(year, data_root=root)
        counts.update(_counts_from_manifest(pdf_manifest))
        if archive_manifest.get("status") != "ok" or pdf_manifest.get("status") != "ok":
            status = "degraded"
            errors.extend(archive_manifest.get("errors", []))
            errors.extend(pdf_manifest.get("errors", []))
    except Exception as exc:
        status = "degraded"
        errors.append(f"{type(exc).__name__}: {exc}")

    result = {
        "source": "house",
        "status": status,
        "year": year,
        "date_window": {"from": f"{year}-01-01", "to": f"{year}-12-31"},
        "counts": counts,
        "error_summary": errors,
    }
    state = read_sync_state(root)
    state.setdefault("backfills", {}).setdefault("house", {})[str(year)] = {
        "source": "house",
        "year": year,
        "last_successful_sync_at": utc_now_iso() if status == "ok" else None,
        "last_failed_sync_at": utc_now_iso() if status != "ok" else None,
        "date_window": result["date_window"],
        "counts": counts,
        "error_summary": errors,
        "status": status,
    }
    write_sync_state(state, root)
    render_sync_summary([result], console=console)
    return {"result": result, "state": state}


def render_sync_summary(results: list[dict[str, Any]], *, console=None) -> None:
    """Render a Rich summary table, falling back to plain output."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        for result in results:
            print(f"{result['source']}: {result['status']} {result.get('counts', {})}")
        return

    out = console or Console()
    table = Table(title="Politician Sync")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Window")
    table.add_column("Counts")
    table.add_column("Errors")
    for result in results:
        window = result.get("date_window", {})
        counts = result.get("counts", {})
        errors = result.get("error_summary", [])
        table.add_row(
            str(result.get("source", "")),
            str(result.get("status", "")),
            f"{window.get('from', '')} -> {window.get('to', '')}",
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-",
            str(len(errors)),
        )
    out.print(table)


def _default_house_incremental_sync(*, date_from: str, date_to: str, data_root: Path) -> dict[str, Any]:
    year = int(date_to[:4])
    archive_manifest = download_house_yearly_archive(year, data_root=data_root)
    filings = _filter_house_filings_for_window(
        load_house_filings_from_archives(year, data_root=data_root),
        date_from=date_from,
        date_to=date_to,
    )
    pdf_manifest = download_house_ptr_pdfs(year, filings=filings, data_root=data_root)
    return {
        "source": "house",
        "year": year,
        "status": "ok" if archive_manifest.get("status") == "ok" and pdf_manifest.get("status") == "ok" else "degraded",
        "artifact_count": archive_manifest.get("artifact_count", 0),
        "filing_count": pdf_manifest.get("filing_count", 0),
        "pdf_count": pdf_manifest.get("pdf_count", 0),
        "errors": [*archive_manifest.get("errors", []), *pdf_manifest.get("errors", [])],
    }


def _filter_house_filings_for_window(
    filings: list[Any],
    *,
    date_from: str,
    date_to: str,
) -> list[Any]:
    filtered = []
    for filing in filings:
        doc_type = str(getattr(filing, "document_type", ""))
        if "ptr" not in doc_type.lower() and "periodic" not in doc_type.lower():
            continue
        filed_date = getattr(filing, "filed_date", None)
        if filed_date and not (date_from <= filed_date <= date_to):
            continue
        filtered.append(filing)
    return filtered


def _result_from_manifest(source: str, date_from: str, date_to: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "status": str(manifest.get("status", "degraded")),
        "date_window": {"from": date_from, "to": date_to},
        "counts": _counts_from_manifest(manifest),
        "error_summary": list(manifest.get("errors", [])),
    }


def _counts_from_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    keys = ("artifact_count", "filing_count", "pdf_count", "missing_artifact_count")
    return {key: int(manifest[key]) for key in keys if key in manifest}


def _merge_source_sync_state(state: dict[str, Any], result: dict[str, Any]) -> None:
    source = str(result["source"])
    status = result["status"]
    entry = {
        "source": source,
        "status": status,
        "date_window": result["date_window"],
        "counts": result.get("counts", {}),
        "error_summary": result.get("error_summary", []),
    }
    if status == "ok":
        entry["last_successful_sync_at"] = utc_now_iso()
        previous = state.setdefault("sources", {}).get(source, {})
        if previous.get("last_failed_sync_at"):
            entry["last_failed_sync_at"] = previous["last_failed_sync_at"]
    else:
        entry["last_failed_sync_at"] = utc_now_iso()
        previous = state.setdefault("sources", {}).get(source, {})
        if previous.get("last_successful_sync_at"):
            entry["last_successful_sync_at"] = previous["last_successful_sync_at"]
    state.setdefault("sources", {})[source] = entry
