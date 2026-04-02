"""
Test Story 6.10: Executive Summary.

Validates:
  1. Strong metrics -> DEPLOY
  2. Weak metrics -> REJECT
  3. Marginal -> REVIEW
  4. Significance test
  5. Save/load round-trip
  6. Reasons populated
  7. Structure complete
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from calibration.executive_summary import (
    generate_executive_summary,
    save_executive_summary,
    load_executive_summary,
    ExecutiveSummary,
)


class TestExecutiveSummary(unittest.TestCase):
    """Tests for executive summary."""

    def test_strong_metrics_deploy(self):
        """Strong metrics -> DEPLOY."""
        s = generate_executive_summary(
            sharpe=0.8, sharpe_ci_lower=0.2, sharpe_ci_upper=1.4,
            hit_rate=0.60, max_drawdown=-0.10, cagr_pct=12.0,
        )
        self.assertEqual(s.recommendation, "DEPLOY")

    def test_weak_metrics_reject(self):
        """Weak metrics -> REJECT."""
        s = generate_executive_summary(
            sharpe=0.1, hit_rate=0.45, max_drawdown=-0.30,
        )
        self.assertEqual(s.recommendation, "REJECT")

    def test_marginal_review(self):
        """Marginal metrics -> REVIEW."""
        s = generate_executive_summary(
            sharpe=0.4, sharpe_ci_lower=0.1, sharpe_ci_upper=0.8,
            hit_rate=0.56, max_drawdown=-0.12,
        )
        self.assertEqual(s.recommendation, "REVIEW")

    def test_significance(self):
        """CI lower > 0 -> significant."""
        s = generate_executive_summary(
            sharpe=0.8, sharpe_ci_lower=0.2, sharpe_ci_upper=1.4,
            hit_rate=0.60, max_drawdown=-0.10,
        )
        self.assertTrue(s.skill_significant)
        
        s2 = generate_executive_summary(
            sharpe=0.3, sharpe_ci_lower=-0.1, sharpe_ci_upper=0.7,
            hit_rate=0.56, max_drawdown=-0.12,
        )
        self.assertFalse(s2.skill_significant)

    def test_save_load_roundtrip(self):
        """Save and load produce identical summary."""
        s = generate_executive_summary(
            sharpe=0.7, sharpe_ci_lower=0.1, sharpe_ci_upper=1.3,
            hit_rate=0.58, max_drawdown=-0.12,
            best_asset="NVDA", worst_asset="BA",
        )
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "summary.json")
            save_executive_summary(s, path)
            loaded = load_executive_summary(path)
            
            self.assertEqual(loaded.sharpe, s.sharpe)
            self.assertEqual(loaded.recommendation, s.recommendation)
            self.assertEqual(loaded.best_asset, "NVDA")

    def test_reasons_populated(self):
        """Rejected verdict has reasons."""
        s = generate_executive_summary(
            sharpe=0.1, hit_rate=0.45, max_drawdown=-0.30,
        )
        self.assertGreater(len(s.reasons), 0)

    def test_structure(self):
        """Summary has all required fields."""
        s = generate_executive_summary(sharpe=0.5, hit_rate=0.55, max_drawdown=-0.10)
        self.assertIsInstance(s, ExecutiveSummary)
        self.assertIsInstance(s.reasons, list)
        self.assertIsInstance(s.recommendation, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
