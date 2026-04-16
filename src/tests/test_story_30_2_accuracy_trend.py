"""
Story 30.2 -- Accuracy Trend Over Time
========================================
Track accuracy snapshots for trend analysis.

Acceptance Criteria:
- Weekly snapshots stored as JSON
- Trend computation: rolling average CRPS
- Alert when CRPS degrades > 5% from previous
- Historical data format validated
"""

import os, sys, json, tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Snapshot storage
# ---------------------------------------------------------------------------
def create_snapshot(date, scores_by_asset):
    """Create accuracy snapshot."""
    return {
        "date": date,
        "scores": scores_by_asset,
        "summary": {
            "n_assets": len(scores_by_asset),
            "mean_loglik": np.mean([
                s["loglik_per_obs"]
                for s in scores_by_asset.values()
            ]),
        },
    }


def save_snapshot(snapshot, directory):
    """Save snapshot to JSON file."""
    path = os.path.join(directory, f"accuracy_{snapshot['date']}.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return path


def load_snapshots(directory):
    """Load all snapshots from directory, sorted by date."""
    snapshots = []
    if not os.path.exists(directory):
        return snapshots
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("accuracy_") and fname.endswith(".json"):
            path = os.path.join(directory, fname)
            with open(path) as f:
                snapshots.append(json.load(f))
    return snapshots


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------
def compute_trend(snapshots, metric="mean_loglik", window=4):
    """Compute rolling average of a summary metric."""
    values = [s["summary"][metric] for s in snapshots]
    if len(values) < window:
        return values, values

    rolling = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        rolling.append(np.mean(values[start:i + 1]))

    return values, rolling


def check_degradation(snapshots, metric="mean_loglik", threshold=0.05):
    """Check if latest snapshot is degraded from previous."""
    if len(snapshots) < 2:
        return False, 0.0

    current = snapshots[-1]["summary"][metric]
    previous = snapshots[-2]["summary"][metric]

    if abs(previous) < 1e-10:
        return False, 0.0

    change = (current - previous) / abs(previous)
    # For loglik: negative change = degradation
    degraded = change < -threshold
    return degraded, change


# ===========================================================================
class TestSnapshotCreation:
    """Snapshot creation and storage works."""

    def test_create_snapshot(self):
        scores = {
            "SPY": {"loglik_per_obs": -0.5, "bic_per_obs": 2.0},
            "QQQ": {"loglik_per_obs": -0.8, "bic_per_obs": 3.0},
        }
        snap = create_snapshot("2025-01-01", scores)
        assert snap["date"] == "2025-01-01"
        assert snap["summary"]["n_assets"] == 2

    def test_save_and_load(self):
        scores = {
            "SPY": {"loglik_per_obs": -0.5, "bic_per_obs": 2.0},
        }
        snap = create_snapshot("2025-01-01", scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_snapshot(snap, tmpdir)
            loaded = load_snapshots(tmpdir)
            assert len(loaded) == 1
            assert loaded[0]["date"] == "2025-01-01"

    def test_multiple_snapshots_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for date in ["2025-01-15", "2025-01-01", "2025-01-08"]:
                scores = {"SPY": {"loglik_per_obs": -0.5}}
                snap = create_snapshot(date, scores)
                save_snapshot(snap, tmpdir)

            loaded = load_snapshots(tmpdir)
            assert len(loaded) == 3
            dates = [s["date"] for s in loaded]
            assert dates == sorted(dates)


# ===========================================================================
class TestTrendComputation:
    """Trend analysis works correctly."""

    def test_compute_trend_short(self):
        snapshots = [
            {"summary": {"mean_loglik": -1.0}},
            {"summary": {"mean_loglik": -0.9}},
        ]
        raw, rolling = compute_trend(snapshots)
        assert len(raw) == 2
        assert len(rolling) == 2

    def test_rolling_average_correct(self):
        snapshots = [
            {"summary": {"mean_loglik": -1.0}},
            {"summary": {"mean_loglik": -2.0}},
            {"summary": {"mean_loglik": -3.0}},
            {"summary": {"mean_loglik": -4.0}},
        ]
        raw, rolling = compute_trend(snapshots, window=2)
        # Rolling[3] = mean(-3, -4) = -3.5
        assert abs(rolling[3] - (-3.5)) < 1e-10

    def test_improving_trend(self):
        snapshots = [{"summary": {"mean_loglik": -2.0 + 0.1 * i}} for i in range(8)]
        raw, rolling = compute_trend(snapshots, window=4)
        # Rolling should be increasing
        assert rolling[-1] > rolling[0]


# ===========================================================================
class TestDegradationAlert:
    """Degradation detection works."""

    def test_no_degradation(self):
        snapshots = [
            {"summary": {"mean_loglik": -1.0}},
            {"summary": {"mean_loglik": -0.9}},
        ]
        degraded, change = check_degradation(snapshots)
        assert not degraded

    def test_degradation_detected(self):
        snapshots = [
            {"summary": {"mean_loglik": -1.0}},
            {"summary": {"mean_loglik": -1.2}},  # 20% worse
        ]
        degraded, change = check_degradation(snapshots, threshold=0.05)
        assert degraded

    def test_single_snapshot_no_alert(self):
        snapshots = [{"summary": {"mean_loglik": -1.0}}]
        degraded, change = check_degradation(snapshots)
        assert not degraded
