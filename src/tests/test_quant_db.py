"""
Test Story 7.4: SQLite Database Layer.

Validates:
  1. Insert and query forecasts
  2. Batch insert
  3. Query with filters
  4. Backtest results CRUD
  5. Error records CRUD
  6. WAL mode active
  7. Count records
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from models.quant_db import QuantDB


class TestQuantDB(unittest.TestCase):
    """Tests for SQLite database layer."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db = QuantDB(os.path.join(self.td, "test.db"))

    def test_insert_query_forecast(self):
        """Insert and query a single forecast."""
        self.db.insert_forecast("SPY", "2026-03-01", 7, 1.5, confidence=0.65)
        results = self.db.query_forecasts(symbol="SPY", horizon=7)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["symbol"], "SPY")
        self.assertAlmostEqual(results[0]["forecast_pct"], 1.5)

    def test_batch_insert(self):
        """Batch insert multiple forecasts."""
        rows = [
            ("SPY", "2026-03-01", 1, 0.5, 0.6),
            ("SPY", "2026-03-01", 7, 1.2, 0.5),
            ("AAPL", "2026-03-01", 7, 2.0, 0.7),
        ]
        count = self.db.insert_forecasts_batch(rows)
        self.assertEqual(count, 3)
        
        results = self.db.query_forecasts(symbol="SPY")
        self.assertEqual(len(results), 2)

    def test_query_filters(self):
        """Query with date range filter."""
        self.db.insert_forecast("SPY", "2026-01-01", 7, 1.0)
        self.db.insert_forecast("SPY", "2026-02-01", 7, 1.5)
        self.db.insert_forecast("SPY", "2026-03-01", 7, 2.0)
        
        results = self.db.query_forecasts(
            symbol="SPY", date_from="2026-02-01", date_to="2026-02-28"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["date"], "2026-02-01")

    def test_backtest_results(self):
        """Insert and query backtest results."""
        self.db.insert_backtest_result(
            "run_001", symbol="SPY", sharpe=0.8, hit_rate=0.58
        )
        results = self.db.query_backtest_results("run_001")
        
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["sharpe"], 0.8)

    def test_error_records(self):
        """Insert and query errors."""
        self.db.insert_error("2026-03-01T10:00:00", 2, "tune", "TSLA", "Failed convergence")
        errors = self.db.query_errors(severity_min=1)
        
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["asset"], "TSLA")

    def test_wal_mode(self):
        """WAL mode is active."""
        import sqlite3
        conn = sqlite3.connect(self.db.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode, "wal")

    def test_count_records(self):
        """Count records across tables."""
        self.db.insert_forecast("SPY", "2026-03-01", 7, 1.0)
        self.db.insert_forecast("AAPL", "2026-03-01", 7, 2.0)
        self.db.insert_backtest_result("run_001")
        
        counts = self.db.count_records()
        self.assertEqual(counts["forecasts"], 2)
        self.assertEqual(counts["backtest_results"], 1)
        self.assertEqual(counts["errors"], 0)

    def test_upsert_forecast(self):
        """Insert same (symbol, date, horizon) updates the record."""
        self.db.insert_forecast("SPY", "2026-03-01", 7, 1.0)
        self.db.insert_forecast("SPY", "2026-03-01", 7, 2.0)
        
        results = self.db.query_forecasts(symbol="SPY", horizon=7)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["forecast_pct"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
