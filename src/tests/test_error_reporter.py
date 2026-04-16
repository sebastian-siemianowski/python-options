"""
Test Story 4.8: Unified Error Reporting Pipeline.

Validates:
  1. Error recorded and retrievable
  2. Severity filtering
  3. Source filtering
  4. Buffer cleared
  5. Prune old errors
  6. Error structure correct
"""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from decision.error_reporter import (
    report_error,
    get_recent_errors,
    prune_old_errors,
    clear_buffer,
    get_buffer,
    Severity,
    ErrorRecord,
    ERRORS_DIR,
)


class TestErrorReporter(unittest.TestCase):
    """Tests for unified error reporting."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        import decision.error_reporter as er
        self._orig_dir = er.ERRORS_DIR
        er.ERRORS_DIR = self.test_dir
        clear_buffer()

    def tearDown(self):
        import decision.error_reporter as er
        er.ERRORS_DIR = self._orig_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)
        clear_buffer()

    def test_report_and_retrieve(self):
        """Error recorded and retrievable."""
        report_error(Severity.ERROR, "tuning", "AAPL", "Convergence failed")
        errors = get_recent_errors(days=1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["asset"], "AAPL")
        self.assertEqual(errors[0]["severity"], "ERROR")

    def test_severity_filter(self):
        """Filter by severity."""
        report_error(Severity.INFO, "tuning", "SPY", "Started")
        report_error(Severity.ERROR, "tuning", "AAPL", "Failed")
        errors = get_recent_errors(days=1, severity_filter="ERROR")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["asset"], "AAPL")

    def test_source_filter(self):
        """Filter by source."""
        report_error(Severity.ERROR, "tuning", "AAPL", "Tune fail")
        report_error(Severity.ERROR, "signals", "NVDA", "Signal fail")
        errors = get_recent_errors(days=1, source_filter="signals")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["asset"], "NVDA")

    def test_buffer_cleared(self):
        """Buffer clears properly."""
        report_error(Severity.INFO, "test", "X", "msg")
        self.assertEqual(len(get_buffer()), 1)
        clear_buffer()
        self.assertEqual(len(get_buffer()), 0)

    def test_error_structure(self):
        """ErrorRecord has correct fields."""
        record = report_error(Severity.CRITICAL, "data", "GLD", "API timeout", "traceback...")
        self.assertEqual(record.source, "data")
        self.assertEqual(record.severity, "CRITICAL")
        self.assertEqual(record.stack_trace, "traceback...")
        d = record.to_dict()
        self.assertIn("timestamp", d)

    def test_prune_old(self):
        """Prune removes old error files."""
        # Create old file manually
        old_path = os.path.join(self.test_dir, "errors_2020-01-01.json")
        with open(old_path, "w") as f:
            json.dump([], f)
        
        report_error(Severity.INFO, "test", "X", "recent")
        removed = prune_old_errors(retention_days=7)
        self.assertEqual(removed, 1)

    def test_multiple_errors_ordered(self):
        """Multiple errors returned newest first."""
        report_error(Severity.INFO, "test", "A", "first")
        report_error(Severity.INFO, "test", "B", "second")
        errors = get_recent_errors(days=1)
        self.assertEqual(len(errors), 2)
        # B is newer, should come first
        self.assertEqual(errors[0]["asset"], "B")

    def test_empty_when_no_dir(self):
        """No errors dir -> empty list."""
        import decision.error_reporter as er
        er.ERRORS_DIR = "/tmp/nonexistent_errors_xyz"
        errors = get_recent_errors(days=1)
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
