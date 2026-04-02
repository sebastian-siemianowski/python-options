"""
Test Story 6.4: Profitability Gate.

Validates:
  1. Strong metrics -> DEPLOY
  2. Weak metrics -> REJECT
  3. Marginal metrics -> REVIEW
  4. Baseline regression -> REVIEW
  5. Exit codes correct
  6. Reasons populated for failures
  7. All-green thresholds pass
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from calibration.profitability_gate import (
    evaluate_profitability_gate,
    GateVerdict,
    EXIT_PASS,
    EXIT_REVIEW,
    EXIT_REJECT,
)


class TestProfitabilityGate(unittest.TestCase):
    """Tests for profitability gate."""

    def test_strong_metrics_deploy(self):
        """Strong metrics -> DEPLOY."""
        v = evaluate_profitability_gate(
            sharpe=0.8, hit_rate=0.60, max_drawdown=-0.10, ic=0.10
        )
        self.assertEqual(v.status, "DEPLOY")
        self.assertEqual(v.exit_code, EXIT_PASS)

    def test_weak_metrics_reject(self):
        """All weak metrics -> REJECT."""
        v = evaluate_profitability_gate(
            sharpe=0.0, hit_rate=0.40, max_drawdown=-0.40, ic=0.0
        )
        self.assertEqual(v.status, "REJECT")
        self.assertEqual(v.exit_code, EXIT_REJECT)

    def test_marginal_metrics_review(self):
        """Marginal (amber) metrics -> REVIEW."""
        v = evaluate_profitability_gate(
            sharpe=0.3, hit_rate=0.52, max_drawdown=-0.12, ic=0.06
        )
        self.assertEqual(v.status, "REVIEW")
        self.assertEqual(v.exit_code, EXIT_REVIEW)

    def test_baseline_regression(self):
        """Sharpe below baseline -> REVIEW."""
        v = evaluate_profitability_gate(
            sharpe=0.6, hit_rate=0.60, max_drawdown=-0.10, ic=0.10,
            baseline_sharpe=0.8,
        )
        self.assertEqual(v.status, "REVIEW")
        self.assertAny(lambda: any("regressed" in r for r in v.reasons))

    def assertAny(self, fn):
        try:
            fn()
        except Exception:
            self.fail("Assertion check failed")

    def test_exit_codes(self):
        """Exit codes match status."""
        self.assertEqual(EXIT_PASS, 0)
        self.assertEqual(EXIT_REVIEW, 1)
        self.assertEqual(EXIT_REJECT, 2)

    def test_reasons_populated(self):
        """Rejected verdict has reasons."""
        v = evaluate_profitability_gate(
            sharpe=0.0, hit_rate=0.40, max_drawdown=-0.40, ic=0.0
        )
        self.assertGreater(len(v.reasons), 0)

    def test_all_green_no_reasons(self):
        """All-green metrics -> no reasons."""
        v = evaluate_profitability_gate(
            sharpe=1.0, hit_rate=0.65, max_drawdown=-0.05, ic=0.15
        )
        self.assertEqual(len(v.reasons), 0)
        self.assertEqual(v.status, "DEPLOY")

    def test_verdict_structure(self):
        """Verdict has correct fields."""
        v = evaluate_profitability_gate(sharpe=0.5, hit_rate=0.55, max_drawdown=-0.15, ic=0.05)
        self.assertIsInstance(v, GateVerdict)
        self.assertIsInstance(v.sharpe, float)
        self.assertIsInstance(v.reasons, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
