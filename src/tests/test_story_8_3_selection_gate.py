"""
Story 8.3: Mixture vs Two-Piece Model Selection Gate
=====================================================
Data-driven heuristic selection beyond pure BIC.
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

from models.phi_student_t import compute_model_selection_gate


class TestModelSelectionGate:
    """Acceptance criteria for Story 8.3."""

    def test_high_vov_prefers_mixture(self):
        """AC1: Mixture preferred when vol-of-vol > 0.5."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.01, n)
        # High vol-of-vol: variable volatility
        vol = np.abs(np.random.normal(0.02, 0.015, n))
        # Give mixture a slightly better BIC
        selected, diag = compute_model_selection_gate(
            returns, vol, bic_mixture=1000.0, bic_two_piece=1010.0
        )
        assert diag['vov_favors_mixture'] is True
        assert selected == 'mixture'

    def test_high_skew_prefers_two_piece(self):
        """AC2: Two-piece preferred when |skewness| > 0.3."""
        np.random.seed(42)
        n = 500
        # Generate skewed returns
        returns = np.concatenate([
            np.random.normal(-0.02, 0.02, 200),
            np.random.normal(0.005, 0.008, 300),
        ])
        vol = np.full(n, 0.01)
        selected, diag = compute_model_selection_gate(
            returns, vol, bic_mixture=1000.0, bic_two_piece=1005.0
        )
        assert diag['skew_favors_two_piece'] is True
        assert selected == 'two_piece'

    def test_similar_bic_prefers_simpler(self):
        """AC3: When BIC within 3 nats, prefer simpler (two-piece)."""
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        # BIC very close (< 6 in BIC scale = 3 nats)
        selected, diag = compute_model_selection_gate(
            returns, vol, bic_mixture=1000.0, bic_two_piece=1004.0
        )
        assert selected == 'two_piece'

    def test_clear_mixture_winner(self):
        """AC4: When mixture clearly wins BIC, select mixture."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.01, n)
        vol = np.abs(np.random.normal(0.02, 0.015, n))  # High VoV
        selected, diag = compute_model_selection_gate(
            returns, vol, bic_mixture=900.0, bic_two_piece=1020.0
        )
        assert selected == 'mixture'

    def test_diagnostics_complete(self):
        """AC5: All diagnostic fields present."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        _, diag = compute_model_selection_gate(returns, vol, 1000.0, 1010.0)
        required = {'empirical_skewness', 'abs_skewness', 'vol_of_vol',
                     'bic_mixture', 'bic_two_piece', 'bic_diff_nats',
                     'selected', 'skew_favors_two_piece', 'vov_favors_mixture'}
        assert required.issubset(diag.keys())

    def test_no_both_high_weight(self):
        """AC6: Selection produces exactly one winner."""
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        selected, _ = compute_model_selection_gate(returns, vol, 1000.0, 1000.0)
        assert selected in ('mixture', 'two_piece')
