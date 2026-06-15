"""Tests for politician disclosure date normalization."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.normalize import get_event_study_anchor_date, normalize_date_fields


def test_date_fields_are_stored_independently():
    result = normalize_date_fields(
        transaction_date="04/12/2026",
        notification_date="04/18/2026",
        filed_date="2026-04-27",
        disclosure_date="2026-04-28",
    )

    assert result["transaction_date"] == "2026-04-12"
    assert result["notification_date"] == "2026-04-18"
    assert result["filed_date"] == "2026-04-27"
    assert result["disclosure_date"] == "2026-04-28"


def test_unknown_dates_are_null_not_inferred():
    result = normalize_date_fields(transaction_date="unknown", filed_date="2026-04-27")

    assert result["transaction_date"] is None
    assert result["notification_date"] is None
    assert result["filed_date"] == "2026-04-27"
    assert result["disclosure_date"] is None
    assert result["delay_days"] is None


def test_delay_days_uses_disclosure_minus_transaction_when_both_present():
    result = normalize_date_fields(transaction_date="2026-04-12", disclosure_date="2026-04-28")

    assert result["delay_days"] == 16
    assert result["validation_status"] == "valid"


def test_impossible_disclosure_before_transaction_warns():
    result = normalize_date_fields(transaction_date="2026-04-12", disclosure_date="2026-04-01")

    assert result["delay_days"] == -11
    assert result["validation_status"] == "valid_with_warnings"
    assert "disclosure_before_transaction" in result["date_warnings"]


def test_event_study_helper_defaults_to_disclosure_date():
    record = {"transaction_date": "2026-04-12", "disclosure_date": "2026-04-28"}

    assert get_event_study_anchor_date(record) == "2026-04-28"

    with pytest.raises(ValueError, match="disclosure_date"):
        get_event_study_anchor_date(record, anchor="transaction_date")
