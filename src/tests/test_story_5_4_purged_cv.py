"""
Story 5.4: Walk-Forward Cross-Validation with Purged Gaps
===========================================================
Purged gaps prevent information leakage between train and test folds.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.gaussian import make_purged_cv_folds


class TestPurgedCVFolds:
    """Acceptance criteria for Story 5.4."""

    def test_purge_gap_respected(self):
        """AC1: Gap of g=5 days between train and test fold."""
        folds = make_purged_cv_folds(n=500, n_folds=5, purge_gap=5)
        assert len(folds) >= 1
        for train_start, train_end, test_start, test_end in folds:
            gap = test_start - train_end - 1
            assert gap >= 5, f"Gap={gap} < 5 between train_end={train_end} and test_start={test_start}"

    def test_at_least_5_folds(self):
        """AC2: At least 5 folds with sufficient data."""
        folds = make_purged_cv_folds(n=1000, n_folds=5, purge_gap=5)
        assert len(folds) >= 5, f"Only {len(folds)} folds, expected >= 5"

    def test_expanding_window(self):
        """AC3: Expanding window (each fold has more training data)."""
        folds = make_purged_cv_folds(n=1000, n_folds=5, purge_gap=5)
        train_sizes = [te - ts for ts, te, _, _ in folds]
        # Each subsequent fold should have same or more training data
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], \
                f"Fold {i} train size {train_sizes[i]} < fold {i-1} size {train_sizes[i-1]}"

    def test_min_train_50_percent(self):
        """AC4: Minimum training size = 50% of data."""
        folds = make_purged_cv_folds(n=500, n_folds=5, min_train_frac=0.5, purge_gap=5)
        for train_start, train_end, _, _ in folds:
            train_size = train_end - train_start
            assert train_size >= 250, f"Train size {train_size} < 250 (50% of 500)"

    def test_no_overlap(self):
        """AC5: No overlap between train and test within each fold."""
        folds = make_purged_cv_folds(n=500, n_folds=5, purge_gap=5)
        for train_start, train_end, test_start, test_end in folds:
            assert test_start > train_end, \
                f"Overlap: train_end={train_end}, test_start={test_start}"

    def test_test_within_bounds(self):
        """AC6: Test indices don't exceed n."""
        n = 500
        folds = make_purged_cv_folds(n=n, n_folds=5, purge_gap=5)
        for _, _, test_start, test_end in folds:
            assert test_end <= n, f"test_end={test_end} > n={n}"
            assert test_start >= 0

    def test_small_dataset_graceful(self):
        """AC7: Small dataset produces valid folds or empty list."""
        folds = make_purged_cv_folds(n=50, n_folds=5, purge_gap=5)
        # Should still produce at least 1 fold or gracefully return empty
        for _, train_end, test_start, test_end in folds:
            assert test_start > train_end
            assert test_end > test_start

    def test_gap_0_vs_gap_5(self):
        """AC8: Gap=0 and gap=5 produce different fold boundaries."""
        folds_0 = make_purged_cv_folds(n=500, n_folds=5, purge_gap=0)
        folds_5 = make_purged_cv_folds(n=500, n_folds=5, purge_gap=5)
        if folds_0 and folds_5:
            # Test starts should differ
            test_starts_0 = [ts for _, _, ts, _ in folds_0]
            test_starts_5 = [ts for _, _, ts, _ in folds_5]
            assert test_starts_0 != test_starts_5
