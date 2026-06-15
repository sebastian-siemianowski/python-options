"""Senate eFD public search adapter."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from html import unescape
from pathlib import Path
from typing import Any

from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso, write_source_health


SENATE_EFD_SEARCH_URL = "https://efdsearch.senate.gov/search/"
SENATE_EFD_HOME_URL = "https://efdsearch.senate.gov/search/home/"
SENATE_EFD_REPORT_DATA_URL = "https://efdsearch.senate.gov/search/report/data/"
SENATE_PTR_REPORT_TYPE_ID = "11"
SENATE_IMPERSONATE_BROWSER = "chrome124"
USER_AGENT = "python-options-politician-monitor/0.1 (+official-source-research)"


@dataclass(frozen=True)
class SenateSearchResult:
    """Normalized Senate public-search result row."""

    report_id: str
    filing_year: int
    filer_name: str
    report_type: str
    filed_date: str | None
    document_url: str
    raw_artifact_path: str | None = None
    document_status: str = "pending"
    document_content_type: str | None = None
    document_fetched_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "filing_year": self.filing_year,
            "filer_name": self.filer_name,
            "report_type": self.report_type,
            "filed_date": self.filed_date,
            "document_url": self.document_url,
            "raw_artifact_path": self.raw_artifact_path,
            "document_status": self.document_status,
            "document_content_type": self.document_content_type,
            "document_fetched_at": self.document_fetched_at,
        }


def search_senate_ptr_filings(
    *,
    date_from: str,
    date_to: str,
    data_root: str | Path | None = None,
    search_url: str = SENATE_EFD_SEARCH_URL,
    read_response_fn=None,
) -> dict[str, object]:
    """Search Senate eFD public pages for latest PTR filings in a date window."""
    root = ensure_politicians_data_dirs(data_root)
    year = _year_from_date(date_from)
    raw_dir = root / "raw" / "senate" / str(year)
    manifest_dir = root / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "report_type": "Periodic Transaction Report",
        "date_from": date_from,
        "date_to": date_to,
    }
    request_url = f"{search_url}?{urllib.parse.urlencode(params)}"
    raw_name = f"search_{date_from}_{date_to}_{_short_hash(request_url)}.html"
    raw_path = raw_dir / raw_name
    manifest_path = manifest_dir / f"senate_{year}.json"
    errors: list[str] = []
    results: list[SenateSearchResult] = []
    response_meta: dict[str, object]

    if read_response_fn is None and search_url == SENATE_EFD_SEARCH_URL:
        live_result = _search_live_senate_ptr_filings(
            date_from=date_from,
            date_to=date_to,
            fallback_year=year,
            raw_dir=raw_dir,
            request_url=request_url,
        )
        results = live_result["results"]
        response_meta = live_result["response_meta"]
        raw_path = live_result["raw_path"]
        errors.extend(live_result["errors"])
        status = live_result["status"]
    else:
        try:
            content, response_meta = _read_senate_response(request_url, read_response_fn=read_response_fn)
            raw_path.write_bytes(content)
            html = content.decode("utf-8", errors="ignore")
            access_reason = _detect_access_control(html)
            if access_reason:
                errors.append(access_reason)
                status = "degraded"
            else:
                results = _parse_senate_ptr_results(html, base_url=request_url, fallback_year=year)
                if not results:
                    errors.append(
                        "No Senate PTR document links found. The public search layout may have changed; "
                        "inspect the raw artifact and update the parser selectors."
                    )
                    status = "degraded"
                else:
                    status = "ok"
        except Exception as exc:
            response_meta = {
                "status_code": None,
                "content_type": None,
                "content_length": 0,
                "fetched_at": utc_now_iso(),
                "source_url": request_url,
            }
            errors.append(f"senate_search_failed: {type(exc).__name__}: {exc}")
            status = "degraded"

    manifest = {
        "source": "senate",
        "year": year,
        "generated_at": utc_now_iso(),
        "status": status,
        "request": {
            "url": search_url,
            "params": params,
            "resolved_url": request_url,
        },
        "response": {
            **response_meta,
            "raw_artifact_path": str(raw_path) if raw_path.exists() else None,
        },
        "filing_count": len(results),
        "filings": [result.to_dict() for result in results],
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_source_health(
        "senate",
        status=status,
        message="Senate eFD search complete." if status == "ok" else "Senate eFD search degraded; inspect manifest errors.",
        data_root=root,
        errors=errors,
        extra={
            "year": year,
            "filing_count": len(results),
            "manifest_path": str(manifest_path),
            "raw_artifact_path": str(raw_path) if raw_path.exists() else None,
        },
    )
    return manifest


def _read_senate_response(url: str, *, read_response_fn=None) -> tuple[bytes, dict[str, object]]:
    if read_response_fn is not None:
        content, meta = read_response_fn(url)
        return content, {
            "status_code": meta.get("status_code", 200),
            "content_type": meta.get("content_type", "text/html"),
            "content_length": len(content),
            "fetched_at": utc_now_iso(),
            "source_url": url,
        }
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
        return content, {
            "status_code": getattr(response, "status", 200),
            "content_type": response.headers.get("Content-Type", "text/html"),
            "content_length": len(content),
            "fetched_at": utc_now_iso(),
            "source_url": url,
        }


def _search_live_senate_ptr_filings(
    *,
    date_from: str,
    date_to: str,
    fallback_year: int,
    raw_dir: Path,
    request_url: str,
) -> dict[str, Any]:
    """Search the live Senate eFD flow using its public agreement and JSON data endpoint."""
    errors: list[str] = []
    try:
        session = _new_browser_like_session()
        home = session.get(SENATE_EFD_HOME_URL, timeout=30)
        home.raise_for_status()
        home_html = home.text
        csrf = _extract_csrf(home_html)
        if not csrf:
            raise RuntimeError("Senate eFD acknowledgement page did not include a CSRF token.")

        search_page = session.post(
            SENATE_EFD_HOME_URL,
            data={"csrfmiddlewaretoken": csrf, "prohibition_agreement": "1"},
            headers={"Referer": SENATE_EFD_HOME_URL},
            timeout=30,
        )
        search_page.raise_for_status()
        search_html = search_page.text
        access_reason = _detect_blocking_access_control(search_html)
        if access_reason:
            raise RuntimeError(access_reason)

        page_csrf = _extract_csrf(search_html) or session.cookies.get("csrftoken") or csrf
        search_form = session.post(
            SENATE_EFD_SEARCH_URL,
            data={
                "csrfmiddlewaretoken": page_csrf,
                "report_type": SENATE_PTR_REPORT_TYPE_ID,
                "submitted_start_date": _iso_to_senate_date(date_from),
                "submitted_end_date": _iso_to_senate_date(date_to),
            },
            headers={"Referer": SENATE_EFD_SEARCH_URL, "X-CSRFToken": session.cookies.get("csrftoken", page_csrf)},
            timeout=30,
        )
        search_form.raise_for_status()
        raw_search_path = raw_dir / f"search_{date_from}_{date_to}_{_short_hash(request_url)}.html"
        raw_search_path.write_text(search_form.text, encoding="utf-8")

        report_rows: list[list[Any]] = []
        records_total = None
        start = 0
        page_size = 100
        latest_response_meta: dict[str, object] = {
            "status_code": search_form.status_code,
            "content_type": search_form.headers.get("content-type", "text/html"),
            "content_length": len(search_form.content),
            "fetched_at": utc_now_iso(),
            "source_url": SENATE_EFD_SEARCH_URL,
            "raw_search_artifact_path": str(raw_search_path),
        }
        while True:
            payload = _senate_report_data_payload(
                date_from=date_from,
                date_to=date_to,
                start=start,
                length=page_size,
                draw=(start // page_size) + 1,
            )
            response = session.post(
                SENATE_EFD_REPORT_DATA_URL,
                data=payload,
                headers={"Referer": SENATE_EFD_SEARCH_URL, "X-CSRFToken": session.cookies.get("csrftoken", page_csrf)},
                timeout=30,
            )
            response.raise_for_status()
            latest_response_meta = {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "application/json"),
                "content_length": len(response.content),
                "fetched_at": utc_now_iso(),
                "source_url": SENATE_EFD_REPORT_DATA_URL,
                "raw_search_artifact_path": str(raw_search_path),
            }
            data = response.json()
            rows = data.get("data") or []
            if not isinstance(rows, list):
                rows = []
            report_rows.extend(rows)
            records_total = int(data.get("recordsFiltered") or data.get("recordsTotal") or len(report_rows))
            if len(report_rows) >= records_total or not rows:
                raw_data_path = raw_dir / f"report_data_{date_from}_{date_to}_{_short_hash(request_url)}.json"
                raw_data_path.write_text(json.dumps({
                    "recordsTotal": data.get("recordsTotal", records_total),
                    "recordsFiltered": data.get("recordsFiltered", records_total),
                    "data": report_rows,
                    "result": data.get("result", "ok"),
                }, indent=2, sort_keys=True), encoding="utf-8")
                latest_response_meta["raw_artifact_path"] = str(raw_data_path)
                break
            start += page_size

        results = _parse_senate_report_data_rows(report_rows, base_url=SENATE_EFD_SEARCH_URL, fallback_year=fallback_year)
        fetched_results: list[SenateSearchResult] = []
        for result in results:
            fetched, document_error = _fetch_senate_document(session, result, raw_dir=raw_dir)
            fetched_results.append(fetched)
            if document_error:
                errors.append(document_error)

        if not fetched_results:
            errors.append(
                "No Senate PTR document links found in the public report-data response; "
                "inspect the raw artifact and update the parser selectors."
            )
            status = "degraded"
        elif all(result.document_status != "ok" for result in fetched_results):
            errors.append("Senate PTR search succeeded, but no report documents could be downloaded.")
            status = "degraded"
        else:
            status = "ok"
        return {
            "status": status,
            "results": fetched_results,
            "errors": errors,
            "response_meta": latest_response_meta,
            "raw_path": Path(str(latest_response_meta.get("raw_artifact_path") or raw_search_path)),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "results": [],
            "errors": [f"senate_search_failed: {type(exc).__name__}: {exc}"],
            "response_meta": {
                "status_code": None,
                "content_type": None,
                "content_length": 0,
                "fetched_at": utc_now_iso(),
                "source_url": request_url,
            },
            "raw_path": raw_dir / f"search_{date_from}_{date_to}_{_short_hash(request_url)}.html",
        }


def _new_browser_like_session():
    try:
        from curl_cffi import requests as curl_requests
    except Exception as exc:
        raise RuntimeError("curl_cffi is required for Senate eFD's browser-gated public search.") from exc
    return curl_requests.Session(impersonate=SENATE_IMPERSONATE_BROWSER)


def _senate_report_data_payload(*, date_from: str, date_to: str, start: int, length: int, draw: int) -> dict[str, str]:
    return {
        "draw": str(draw),
        "start": str(start),
        "length": str(length),
        "search[value]": "",
        "search[regex]": "false",
        "order[0][column]": "1",
        "order[0][dir]": "asc",
        "report_types": f"[{SENATE_PTR_REPORT_TYPE_ID}]",
        "filer_types": "[]",
        "submitted_start_date": f"{_iso_to_senate_date(date_from)} 00:00:00",
        "submitted_end_date": f"{_iso_to_senate_date(date_to)} 23:59:59",
        "candidate_state": "",
        "senator_state": "",
        "office_id": "",
        "first_name": "",
        "last_name": "",
    }


def _parse_senate_report_data_rows(
    rows: list[Any],
    *,
    base_url: str,
    fallback_year: int,
) -> list[SenateSearchResult]:
    results: list[SenateSearchResult] = []
    for raw_row in rows:
        if isinstance(raw_row, dict):
            cells = [
                raw_row.get("first_name") or raw_row.get("first") or "",
                raw_row.get("last_name") or raw_row.get("last") or "",
                raw_row.get("office") or "",
                raw_row.get("report_type") or raw_row.get("report") or "",
                raw_row.get("filed_date") or raw_row.get("date") or "",
            ]
        elif isinstance(raw_row, list):
            cells = raw_row
        else:
            continue
        if len(cells) < 5:
            continue
        first_name = _strip_html(str(cells[0]))
        last_name = _strip_html(str(cells[1]))
        office = _strip_html(str(cells[2]))
        report_html = str(cells[3])
        filed_text = _strip_html(str(cells[4]))
        href = _extract_href(report_html)
        if not href:
            continue
        document_url = urllib.parse.urljoin(base_url, href)
        report_label = _strip_html(report_html)
        report_id = _extract_report_id(document_url) or _short_hash(document_url)
        filed_date = _extract_date(filed_text) or _extract_date(report_label)
        filing_year = _year_from_date(filed_date) if filed_date else fallback_year
        filer_name = _format_filer_name(first_name=first_name, last_name=last_name, office=office)
        results.append(SenateSearchResult(
            report_id=report_id,
            filing_year=filing_year,
            filer_name=filer_name,
            report_type=report_label or "Periodic Transaction Report",
            filed_date=filed_date,
            document_url=document_url,
        ))
    deduped: dict[str, SenateSearchResult] = {}
    for result in results:
        deduped[result.document_url] = result
    return list(deduped.values())


def _fetch_senate_document(session, result: SenateSearchResult, *, raw_dir: Path) -> tuple[SenateSearchResult, str | None]:
    fetched_at = utc_now_iso()
    extension = ".html"
    if ".pdf" in result.document_url.lower():
        extension = ".pdf"
    artifact_path = raw_dir / f"{_safe_report_id(result.report_id)}{extension}"
    try:
        response = session.get(result.document_url, headers={"Referer": SENATE_EFD_SEARCH_URL}, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "application/octet-stream")
        content = response.content
        if "pdf" in content_type.lower() and extension != ".pdf":
            artifact_path = raw_dir / f"{_safe_report_id(result.report_id)}.pdf"
        artifact_path.write_bytes(content)
        return SenateSearchResult(
            report_id=result.report_id,
            filing_year=result.filing_year,
            filer_name=result.filer_name,
            report_type=result.report_type,
            filed_date=result.filed_date,
            document_url=result.document_url,
            raw_artifact_path=str(artifact_path),
            document_status="ok",
            document_content_type=content_type,
            document_fetched_at=fetched_at,
        ), None
    except Exception as exc:
        return SenateSearchResult(
            report_id=result.report_id,
            filing_year=result.filing_year,
            filer_name=result.filer_name,
            report_type=result.report_type,
            filed_date=result.filed_date,
            document_url=result.document_url,
            raw_artifact_path=None,
            document_status="degraded",
            document_content_type=None,
            document_fetched_at=fetched_at,
        ), f"senate_document_fetch_failed:{result.report_id}:{type(exc).__name__}: {exc}"


def _parse_senate_ptr_results(html: str, *, base_url: str, fallback_year: int) -> list[SenateSearchResult]:
    results: list[SenateSearchResult] = []
    for match in re.finditer(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL):
        href, label_html = match.groups()
        context = _nearby_context(html, match.start(), match.end())
        text = _strip_html(f"{label_html} {context}")
        if "periodic" not in text.lower() and "ptr" not in text.lower() and "ptr" not in href.lower():
            continue
        document_url = urllib.parse.urljoin(base_url, href)
        report_id = _extract_report_id(document_url) or _short_hash(document_url)
        filed_date = _extract_date(text)
        filing_year = _year_from_date(filed_date) if filed_date else fallback_year
        filer_name = _extract_filer_name(text)
        results.append(SenateSearchResult(
            report_id=report_id,
            filing_year=filing_year,
            filer_name=filer_name,
            report_type="Periodic Transaction Report",
            filed_date=filed_date,
            document_url=document_url,
        ))
    deduped: dict[str, SenateSearchResult] = {}
    for result in results:
        deduped[result.document_url] = result
    return list(deduped.values())


def _detect_access_control(html: str) -> str | None:
    lowered = html.lower()
    blocked_markers = ("captcha", "access denied", "log in", "authentication required")
    acknowledgement_markers = ("i agree", "acknowledge", "public disclosure agreement")
    if any(marker in lowered for marker in blocked_markers):
        return "Senate eFD access control detected; refusing to bypass CAPTCHA, login, or access restrictions."
    if any(marker in lowered for marker in acknowledgement_markers):
        return "Senate eFD acknowledgement gate detected; manual public-use acknowledgement is required."
    return None


def _detect_blocking_access_control(html: str) -> str | None:
    lowered = html.lower()
    blocked_markers = ("captcha", "access denied", "log in", "authentication required")
    if any(marker in lowered for marker in blocked_markers):
        return "Senate eFD access control detected; refusing to bypass CAPTCHA, login, or access restrictions."
    return None


def _nearby_context(html: str, start: int, end: int, radius: int = 500) -> str:
    return html[max(0, start - radius): min(len(html), end + radius)]


def _strip_html(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", unescape(text))
    return re.sub(r"\s+", " ", without_tags).strip()


def _extract_href(html: str) -> str | None:
    match = re.search(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    return unescape(match.group(1)) if match else None


def _extract_report_id(text: str) -> str | None:
    match = re.search(r"/(?:ptr|paper)/([^/?#]+)/?", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    matches = re.findall(r"(\d{5,})", text)
    return matches[-1] if matches else None


def _extract_date(text: str) -> str | None:
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text)
    if match:
        m, d, y = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None


def _extract_filer_name(text: str) -> str:
    cleaned = text.replace("Periodic Transaction Report", " ").replace("PTR", " ")
    cleaned = re.sub(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}/20\d{2})\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:|")
    return cleaned[:120] or "Unknown"


def _year_from_date(value: str | None) -> int:
    if not value:
        return date.today().year
    return int(value[:4])


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _extract_csrf(html: str) -> str | None:
    match = re.search(r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'value=["\']([^"\']+)["\']\s+name=["\']csrfmiddlewaretoken["\']', html, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _iso_to_senate_date(value: str) -> str:
    year, month, day = value.split("-")
    return f"{int(month):02d}/{int(day):02d}/{int(year):04d}"


def _format_filer_name(*, first_name: str, last_name: str, office: str) -> str:
    name = " ".join(part for part in (first_name.strip(), last_name.strip()) if part).strip()
    if name:
        return re.sub(r"\s+", " ", name)
    office_name = re.sub(r"\([^)]*\)", " ", office)
    office_name = re.sub(r"\s+", " ", office_name).strip(" ,")
    return office_name or "Unknown"


def _safe_report_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or _short_hash(value)
