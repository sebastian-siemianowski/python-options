"""House Clerk financial disclosure archive downloader."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from zipfile import ZipFile

from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso, write_source_health


HOUSE_DISCLOSURE_REPORTS_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewReport"
HOUSE_ARCHIVE_URL_TEMPLATE = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
HOUSE_PTR_PDF_URL_TEMPLATE = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{report_id}.pdf"
HOUSE_FD_PDF_URL_TEMPLATE = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}/{report_id}.pdf"
USER_AGENT = "python-options-politician-monitor/0.1 (+official-source-research)"


@dataclass(frozen=True)
class DownloadedArtifact:
    """Manifest entry for a downloaded or reused House archive."""

    source: str
    year: int
    url: str
    filename: str
    path: str
    sha256: str
    size_bytes: int
    downloaded_at: str
    reused_existing: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "year": self.year,
            "url": self.url,
            "filename": self.filename,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "downloaded_at": self.downloaded_at,
            "reused_existing": self.reused_existing,
            "status": self.status,
        }


@dataclass(frozen=True)
class HouseFiling:
    """Normalized House filing row used before detailed PDF parsing."""

    report_id: str
    filing_year: int
    document_type: str
    filer_name: str
    document_url: str | None = None
    filed_date: str | None = None
    state_district: str | None = None
    filing_type_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "filing_year": self.filing_year,
            "document_type": self.document_type,
            "filer_name": self.filer_name,
            "document_url": self.document_url,
            "filed_date": self.filed_date,
            "state_district": self.state_district,
            "filing_type_code": self.filing_type_code,
        }


def discover_house_yearly_archive_urls(
    year: int,
    *,
    index_url: str = HOUSE_DISCLOSURE_REPORTS_URL,
) -> list[str]:
    """Discover House yearly archive ZIP URLs from the official reports page."""
    html = _read_url_bytes(index_url).decode("utf-8", errors="ignore")
    matches: list[str] = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
        lower = href.lower()
        if lower.endswith(".zip") and f"{year}" in lower and ("fd" in lower or "financial" in lower):
            matches.append(urljoin(index_url, href))
    if matches:
        return sorted(dict.fromkeys(matches))
    return [HOUSE_ARCHIVE_URL_TEMPLATE.format(year=year)]


def download_house_yearly_archive(
    year: int,
    *,
    archive_urls: list[str] | None = None,
    index_url: str = HOUSE_DISCLOSURE_REPORTS_URL,
    data_root: str | Path | None = None,
) -> dict[str, object]:
    """Download or reuse House yearly archive artifacts and write a manifest."""
    root = ensure_politicians_data_dirs(data_root)
    raw_dir = root / "raw" / "house" / str(year)
    manifest_dir = root / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"house_{year}.json"

    errors: list[str] = []
    artifacts: list[DownloadedArtifact] = []
    offline = os.getenv("OFFLINE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

    if offline:
        for path in sorted(raw_dir.glob("*")):
            if path.is_file():
                artifacts.append(_artifact_from_existing(year, path, url=f"offline://{path.name}"))
        status = "ok" if artifacts else "degraded"
        message = "House archive reused from local raw directory." if artifacts else "OFFLINE_MODE=1 and no local House archives found."
        manifest = _write_manifest(year, manifest_path, artifacts, errors, offline=offline)
        write_source_health(
            "house",
            status=status,
            message=message,
            data_root=root,
            errors=errors,
            extra={"year": year, "artifact_count": len(artifacts), "manifest_path": str(manifest_path)},
        )
        return manifest

    urls = archive_urls or []
    if not urls:
        try:
            urls = discover_house_yearly_archive_urls(year, index_url=index_url)
        except Exception as exc:
            errors.append(f"discover_failed: {type(exc).__name__}: {exc}")
            urls = [HOUSE_ARCHIVE_URL_TEMPLATE.format(year=year)]

    for url in urls:
        filename = _filename_from_url(url, default=f"{year}FD.zip")
        target = raw_dir / filename
        try:
            if target.exists():
                artifacts.append(_artifact_from_existing(year, target, url=url))
                continue
            content = _read_url_bytes(url)
            target.write_bytes(content)
            artifacts.append(_artifact_from_existing(year, target, url=url, reused_existing=False))
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    status = "ok" if artifacts and not errors else "degraded"
    message = "House yearly archive sync complete." if status == "ok" else "House yearly archive sync completed with errors."
    manifest = _write_manifest(year, manifest_path, artifacts, errors, offline=offline)
    write_source_health(
        "house",
        status=status,
        message=message,
        data_root=root,
        errors=errors,
        extra={"year": year, "artifact_count": len(artifacts), "manifest_path": str(manifest_path)},
    )
    return manifest


def build_house_pdf_url(filing: HouseFiling) -> str:
    """Build the official House PDF URL for a filing when not explicit."""
    if filing.document_url:
        return filing.document_url
    doc_type = filing.document_type.strip().lower()
    if "ptr" in doc_type or "periodic" in doc_type:
        return HOUSE_PTR_PDF_URL_TEMPLATE.format(year=filing.filing_year, report_id=filing.report_id)
    return HOUSE_FD_PDF_URL_TEMPLATE.format(year=filing.filing_year, report_id=filing.report_id)


def load_house_filings_from_archives(
    year: int,
    *,
    data_root: str | Path | None = None,
) -> list[HouseFiling]:
    """Load filing rows from previously downloaded House yearly archives."""
    root = ensure_politicians_data_dirs(data_root)
    manifest_path = root / "manifests" / f"house_{year}.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filings: list[HouseFiling] = []
    for artifact in manifest.get("artifacts", []):
        path = Path(str(artifact.get("path", "")))
        if path.suffix.lower() == ".zip" and path.exists():
            filings.extend(_load_house_filings_from_zip(year, path))
        elif path.suffix.lower() in {".txt", ".csv"} and path.exists():
            filings.extend(_parse_house_index_text(year, path.read_text(encoding="utf-8", errors="ignore")))
    return filings


def download_house_ptr_pdfs(
    year: int,
    *,
    filings: list[HouseFiling] | None = None,
    data_root: str | Path | None = None,
    retries: int = 3,
    backoff_seconds: float = 0.25,
    rate_limit_seconds: float = 0.5,
    read_bytes_fn=None,
    sleep_fn=None,
) -> dict[str, object]:
    """Fetch House PTR PDFs and write a filing-level manifest."""
    root = ensure_politicians_data_dirs(data_root)
    raw_dir = root / "raw" / "house" / str(year)
    manifest_dir = root / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"house_ptr_{year}.json"

    rows = filings if filings is not None else load_house_filings_from_archives(year, data_root=root)
    ptr_rows = [row for row in rows if _is_ptr_document(row.document_type)]
    pdf_entries: list[dict[str, object]] = []
    errors: list[str] = []

    for filing in ptr_rows:
        source_url = build_house_pdf_url(filing)
        target = raw_dir / f"{filing.report_id}.pdf"
        entry: dict[str, object] = {
            **filing.to_dict(),
            "source_url": source_url,
            "path": str(target),
            "downloaded_at": None,
            "content_type": None,
            "content_length": None,
            "sha256": None,
            "status": "pending",
            "reused_existing": False,
            "remote_changed": False,
        }
        try:
            content, content_type = _read_url_bytes_with_retries(
                source_url,
                retries=retries,
                backoff_seconds=backoff_seconds,
                rate_limit_seconds=rate_limit_seconds,
                read_bytes_fn=read_bytes_fn,
                sleep_fn=sleep_fn,
            )
            remote_hash = _sha256_bytes(content)
            if target.exists():
                local_hash = _sha256_file(target)
                if local_hash == remote_hash:
                    entry.update({
                        "status": "ok",
                        "reused_existing": True,
                        "downloaded_at": utc_now_iso(),
                        "content_type": content_type,
                        "content_length": len(content),
                        "sha256": local_hash,
                    })
                else:
                    target.write_bytes(content)
                    entry.update({
                        "status": "ok",
                        "remote_changed": True,
                        "downloaded_at": utc_now_iso(),
                        "content_type": content_type,
                        "content_length": len(content),
                        "sha256": remote_hash,
                    })
            else:
                target.write_bytes(content)
                entry.update({
                    "status": "ok",
                    "downloaded_at": utc_now_iso(),
                    "content_type": content_type,
                    "content_length": len(content),
                    "sha256": remote_hash,
                })
        except Exception as exc:
            message = f"{filing.report_id}: {type(exc).__name__}: {exc}"
            errors.append(message)
            entry.update({
                "status": "missing_artifact",
                "error": message,
            })
        pdf_entries.append(entry)

    status = "ok" if pdf_entries and not errors else "degraded"
    manifest = {
        "source": "house",
        "year": year,
        "generated_at": utc_now_iso(),
        "status": status,
        "filing_count": len(ptr_rows),
        "pdf_count": sum(1 for entry in pdf_entries if entry["status"] == "ok"),
        "missing_artifact_count": sum(1 for entry in pdf_entries if entry["status"] == "missing_artifact"),
        "filings": pdf_entries,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_source_health(
        "house",
        status=status,
        message="House PTR PDF fetch complete." if status == "ok" else "House PTR PDF fetch completed with missing artifacts.",
        data_root=root,
        errors=errors,
        extra={
            "year": year,
            "filing_count": len(ptr_rows),
            "pdf_count": manifest["pdf_count"],
            "missing_artifact_count": manifest["missing_artifact_count"],
            "manifest_path": str(manifest_path),
        },
    )
    return manifest


def _write_manifest(
    year: int,
    manifest_path: Path,
    artifacts: list[DownloadedArtifact],
    errors: list[str],
    *,
    offline: bool,
) -> dict[str, object]:
    now = utc_now_iso()
    manifest = {
        "source": "house",
        "year": year,
        "generated_at": now,
        "offline_mode": offline,
        "status": "ok" if artifacts and not errors else "degraded",
        "artifact_count": len(artifacts),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "errors": list(errors),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _artifact_from_existing(
    year: int,
    path: Path,
    *,
    url: str,
    reused_existing: bool = True,
) -> DownloadedArtifact:
    return DownloadedArtifact(
        source="house",
        year=year,
        url=url,
        filename=path.name,
        path=str(path),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        downloaded_at=utc_now_iso(),
        reused_existing=reused_existing,
        status="ok",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _filename_from_url(url: str, *, default: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    return name or default


def _read_url_bytes(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _read_url_bytes_with_retries(
    url: str,
    *,
    retries: int,
    backoff_seconds: float,
    rate_limit_seconds: float,
    read_bytes_fn=None,
    sleep_fn=None,
) -> tuple[bytes, str]:
    reader = read_bytes_fn or _read_url_bytes_with_content_type
    sleeper = sleep_fn or time.sleep
    attempts = max(1, retries)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            content, content_type = reader(url)
            if rate_limit_seconds > 0:
                sleeper(rate_limit_seconds)
            return content, content_type
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                sleeper(backoff_seconds * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def _read_url_bytes_with_content_type(url: str) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes(), "application/pdf"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "application/pdf")
        return response.read(), content_type


def _is_ptr_document(document_type: str) -> bool:
    lowered = document_type.strip().lower()
    return "ptr" in lowered or "periodic" in lowered


def _load_house_filings_from_zip(year: int, path: Path) -> list[HouseFiling]:
    filings: list[HouseFiling] = []
    with ZipFile(path) as archive:
        for name in archive.namelist():
            lowered = name.lower()
            if lowered.endswith((".txt", ".csv")):
                content = archive.read(name).decode("utf-8", errors="ignore")
                filings.extend(_parse_house_index_text(year, content))
    return filings


def _parse_house_index_text(year: int, content: str) -> list[HouseFiling]:
    filings: list[HouseFiling] = []
    headers: list[str] = []
    for line in content.splitlines():
        raw = line.strip()
        if not raw:
            continue
        delimiter = "\t" if "\t" in raw else "|" if "|" in raw else ","
        parts = [part.strip().strip('"') for part in raw.split(delimiter)]
        if raw.lower().startswith("prefix"):
            headers = [_normalize_house_header(part) for part in parts]
            continue
        if len(parts) < 4:
            continue
        row = _house_index_row(headers, parts)
        report_id = _first_numeric_token(parts)
        if not report_id:
            continue
        filing_type_code = row.get("filingtype")
        document_type = (
            _document_type_from_house_code(filing_type_code)
            or _first_matching_part(parts, ("ptr", "periodic"))
            or _first_matching_part(parts, ("financial", "annual"))
            or "unknown"
        )
        filer_name = _house_filer_name_from_row(row) or _best_effort_filer_name(parts, report_id, document_type)
        filings.append(HouseFiling(
            report_id=report_id,
            filing_year=year,
            document_type=document_type,
            filer_name=filer_name,
            filed_date=_normalize_house_date(row.get("filingdate")),
            state_district=row.get("statedst"),
            filing_type_code=filing_type_code,
        ))
    return filings


def _first_numeric_token(parts: list[str]) -> str | None:
    for part in reversed(parts):
        token = part.strip()
        if token.isdigit() and len(token) >= 4:
            return token
    return None


def _first_matching_part(parts: list[str], needles: tuple[str, ...]) -> str | None:
    for part in parts:
        lowered = part.lower()
        if any(needle in lowered for needle in needles):
            return part
    return None


def _best_effort_filer_name(parts: list[str], report_id: str, document_type: str) -> str:
    candidates = [
        part
        for part in parts
        if part and part != report_id and part != document_type and not part.isdigit()
    ]
    return " ".join(candidates[:2]).strip() or "Unknown"


def _normalize_house_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _house_index_row(headers: list[str], parts: list[str]) -> dict[str, str]:
    if not headers:
        return {}
    return {
        headers[idx]: parts[idx]
        for idx in range(min(len(headers), len(parts)))
        if headers[idx]
    }


def _document_type_from_house_code(code: str | None) -> str | None:
    normalized = (code or "").strip().upper()
    if normalized == "P":
        return "PTR"
    if normalized == "A":
        return "Annual"
    if normalized == "C":
        return "Candidate"
    if normalized == "W":
        return "Termination"
    if normalized == "X":
        return "Extension"
    if normalized == "D":
        return "Amendment"
    return None


def _house_filer_name_from_row(row: dict[str, str]) -> str | None:
    if not row:
        return None
    parts = [
        row.get("prefix"),
        row.get("first"),
        row.get("last"),
        row.get("suffix"),
    ]
    name = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", name).strip() or None


def _normalize_house_date(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", value)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", value)
    if match:
        m, d, y = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None
