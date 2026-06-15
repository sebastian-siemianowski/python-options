"""Source-health persistence for politician disclosure ingestion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ensure_politicians_data_dirs


def utc_now_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_source_health(data_root: str | Path | None = None) -> dict[str, Any]:
    """Read source health, returning an empty structure when missing."""
    root = ensure_politicians_data_dirs(data_root)
    path = root / "source_health.json"
    if not path.exists():
        return {"updated_at": None, "sources": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": None, "sources": {}}


def write_source_health(
    source: str,
    *,
    status: str,
    message: str,
    data_root: str | Path | None = None,
    errors: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge and persist one source health entry."""
    root = ensure_politicians_data_dirs(data_root)
    path = root / "source_health.json"
    payload = read_source_health(root)
    now = utc_now_iso()
    entry: dict[str, Any] = {
        "status": status,
        "message": message,
        "updated_at": now,
        "errors": errors or [],
    }
    if extra:
        entry.update(extra)
    payload["updated_at"] = now
    payload.setdefault("sources", {})[source] = entry
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
