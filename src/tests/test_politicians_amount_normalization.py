"""Tests for politician disclosure amount bucket normalization."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.normalize import AMOUNT_ESTIMATE_LABEL, normalize_amount_bucket


OFFICIAL_BUCKETS = [
    ("$1 - $1,000", 1, 1000),
    ("$1,001 - $15,000", 1001, 15000),
    ("$15,001 - $50,000", 15001, 50000),
    ("$50,001 - $100,000", 50001, 100000),
    ("$100,001 - $250,000", 100001, 250000),
    ("$250,001 - $500,000", 250001, 500000),
    ("$500,001 - $1,000,000", 500001, 1000000),
    ("$1,000,001 - $5,000,000", 1000001, 5000000),
    ("$5,000,001 - $25,000,000", 5000001, 25000000),
    ("$25,000,001 - $50,000,000", 25000001, 50000000),
]


def test_amount_bucket_standard_range():
    result = normalize_amount_bucket("$1,001 - $15,000")

    assert result["amount_min_usd"] == 1001
    assert result["amount_max_usd"] == 15000
    assert result["amount_mid_usd"] == 8000.5
    assert result["amount_bucket_parse_status"] == "range"


def test_amount_bucket_over_max_is_open_ended():
    result = normalize_amount_bucket("Over $50,000,000")

    assert result["amount_min_usd"] == 50000001
    assert result["amount_max_usd"] is None
    assert result["amount_mid_usd"] is None
    assert result["amount_bucket_parse_status"] == "open_ended"


def test_spouse_and_dependent_child_special_buckets_preserve_raw_label():
    spouse = normalize_amount_bucket("Spouse/DC: $15,001 - $50,000")
    dependent = normalize_amount_bucket("Dependent Child: $1,001 - $15,000")

    assert spouse["amount_bucket_raw"] == "Spouse/DC: $15,001 - $50,000"
    assert spouse["amount_min_usd"] == 15001
    assert spouse["amount_max_usd"] == 50000
    assert dependent["amount_bucket_raw"] == "Dependent Child: $1,001 - $15,000"


def test_midpoint_values_are_labelled_as_estimates_for_api_and_ui():
    result = normalize_amount_bucket("$15,001 - $50,000")

    assert result["amount_mid_usd"] == 32500.5
    assert result["amount_mid_usd_is_estimate"] is True
    assert result["amount_estimate_label"] == AMOUNT_ESTIMATE_LABEL


def test_all_observed_official_range_buckets_parse():
    for raw, minimum, maximum in OFFICIAL_BUCKETS:
        result = normalize_amount_bucket(raw)
        assert result["amount_min_usd"] == minimum
        assert result["amount_max_usd"] == maximum
        assert result["amount_mid_usd"] == (minimum + maximum) / 2
        assert result["amount_bucket_parse_status"] == "range"


def test_amount_bucket_or_less_parses_as_zero_to_max():
    result = normalize_amount_bucket("$1,000 or less")

    assert result["amount_min_usd"] == 0
    assert result["amount_max_usd"] == 1000
    assert result["amount_mid_usd"] == 500
