"""Tests for Senate eFD public search adapter."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.cli import main as politicians_cli_main
from ingestion.politicians.source_health import read_source_health
from ingestion.politicians.sources.senate import search_senate_ptr_filings


def _html_response(html: str, status_code: int = 200):
    def reader(url):
        return html.encode("utf-8"), {"status_code": status_code, "content_type": "text/html"}

    return reader


def test_senate_search_date_window_stores_raw_manifest_and_document_url(tmp_path):
    html = """
    <html>
      <body>
        <table>
          <tr>
            <td>Jane Senator</td>
            <td>Periodic Transaction Report</td>
            <td>05/12/2026</td>
            <td><a href="/search/view/ptr/123456/">View PTR</a></td>
          </tr>
        </table>
      </body>
    </html>
    """

    manifest = search_senate_ptr_filings(
        date_from="2026-05-01",
        date_to="2026-05-29",
        data_root=tmp_path / "politicians",
        read_response_fn=_html_response(html),
    )

    row = manifest["filings"][0]
    raw_path = manifest["response"]["raw_artifact_path"]

    assert manifest["status"] == "ok"
    assert manifest["request"]["params"]["date_from"] == "2026-05-01"
    assert manifest["request"]["params"]["date_to"] == "2026-05-29"
    assert manifest["response"]["status_code"] == 200
    assert manifest["response"]["content_type"] == "text/html"
    assert manifest["response"]["content_length"] == len(html.encode("utf-8"))
    assert raw_path.endswith(".html")
    assert os.path.exists(raw_path)
    assert row["report_id"] == "123456"
    assert row["filing_year"] == 2026
    assert row["report_type"] == "Periodic Transaction Report"
    assert row["filed_date"] == "2026-05-12"
    assert row["document_url"] == "https://efdsearch.senate.gov/search/view/ptr/123456/"


def test_senate_search_refuses_access_control_or_acknowledgement_gate(tmp_path):
    html = "<html><body>Public disclosure agreement <button>I Agree</button></body></html>"

    manifest = search_senate_ptr_filings(
        date_from="2026-05-01",
        date_to="2026-05-29",
        data_root=tmp_path / "politicians",
        read_response_fn=_html_response(html),
    )
    health = read_source_health(tmp_path / "politicians")

    assert manifest["status"] == "degraded"
    assert manifest["filing_count"] == 0
    assert "acknowledgement" in manifest["errors"][0].lower()
    assert health["sources"]["senate"]["status"] == "degraded"


def test_senate_search_degrades_when_public_layout_changes(tmp_path):
    html = "<html><body><main>No recognizable result rows here.</main></body></html>"

    manifest = search_senate_ptr_filings(
        date_from="2026-05-01",
        date_to="2026-05-29",
        data_root=tmp_path / "politicians",
        read_response_fn=_html_response(html),
    )

    assert manifest["status"] == "degraded"
    assert manifest["filing_count"] == 0
    assert "layout may have changed" in manifest["errors"][0]


def test_senate_search_records_exact_request_and_response_metadata(tmp_path):
    html = """
    <a href="https://efdsearch.senate.gov/search/view/paper/999999/">
      John Senator Periodic Transaction Report 2026-05-15
    </a>
    """
    seen_urls = []

    def reader(url):
        seen_urls.append(url)
        return html.encode("utf-8"), {"status_code": 203, "content_type": "text/html; charset=utf-8"}

    manifest = search_senate_ptr_filings(
        date_from="2026-05-01",
        date_to="2026-05-29",
        data_root=tmp_path / "politicians",
        search_url="https://efdsearch.senate.gov/search/custom",
        read_response_fn=reader,
    )

    assert seen_urls == [manifest["request"]["resolved_url"]]
    assert manifest["request"]["url"] == "https://efdsearch.senate.gov/search/custom"
    assert manifest["request"]["params"]["report_type"] == "Periodic Transaction Report"
    assert manifest["response"]["status_code"] == 203
    assert manifest["response"]["source_url"] == manifest["request"]["resolved_url"]
    assert manifest["filings"][0]["document_url"] == "https://efdsearch.senate.gov/search/view/paper/999999/"


def test_senate_search_cli_shape(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(tmp_path / "politicians"))
    monkeypatch.setattr(
        "ingestion.politicians.cli.search_senate_ptr_filings",
        lambda date_from, date_to: {
            "source": "senate",
            "year": 2026,
            "status": "ok",
            "filing_count": 1,
            "errors": [],
        },
    )

    exit_code = politicians_cli_main(["senate", "search", "--from", "2026-05-01", "--to", "2026-05-29"])
    out = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert out["source"] == "senate"
    assert out["year"] == 2026
    assert out["status"] == "ok"
    assert out["filing_count"] == 1
