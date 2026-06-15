"""Golden filing fixture contract tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.fixtures import (
    load_expected_normalized,
    load_golden_fixtures,
    parse_golden_fixtures,
)


TIMESTAMP_FIELDS = {"created_at", "updated_at", "generated_at", "downloaded_at"}


def test_golden_fixture_set_has_required_house_and_senate_counts():
    fixtures = load_golden_fixtures()

    assert sum(1 for fixture in fixtures if fixture["source"] == "house") >= 25
    assert sum(1 for fixture in fixtures if fixture["source"] == "senate") >= 15


def test_golden_fixture_set_covers_required_parser_scenarios():
    tags = {tag for fixture in load_golden_fixtures() for tag in fixture.get("coverage_tags", [])}

    assert {
        "typed_pdf",
        "multi_line_asset",
        "amendment",
        "spouse_trade",
        "dependent_child_trade",
        "etf",
        "option",
        "bond",
        "unknown_asset",
    } <= tags


def test_every_golden_fixture_documents_source_url_and_retrieval_date():
    for fixture in load_golden_fixtures():
        assert fixture["source_url"].startswith(("https://disclosures-clerk.house.gov/", "https://efdsearch.senate.gov/"))
        assert fixture["retrieval_date"] == "2026-05-28"
        assert fixture["raw_text"].strip()


def test_expected_normalized_json_exists_for_every_fixture():
    fixtures = load_golden_fixtures()
    expected = load_expected_normalized()

    assert {row["fixture_id"] for row in expected} == {fixture["fixture_id"] for fixture in fixtures}
    assert len(expected) == len(fixtures)


def test_golden_fixture_parsers_match_expected_normalized_json_excluding_timestamps():
    parsed = [_strip_timestamps(row) for row in parse_golden_fixtures()]
    expected = [_strip_timestamps(row) for row in load_expected_normalized()]

    assert parsed == expected


def _strip_timestamps(value):
    if isinstance(value, dict):
        return {
            key: _strip_timestamps(item)
            for key, item in value.items()
            if key not in TIMESTAMP_FIELDS
        }
    if isinstance(value, list):
        return [_strip_timestamps(item) for item in value]
    return value
