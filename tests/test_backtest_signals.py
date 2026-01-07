#!/usr/bin/env python3
"""
test_backtest_signals.py

Test suite for the backtesting framework.
"""

import unittest
import os
import sys
import json
import pandas as pd
import numpy as np

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from backtest_signals import run_backtest, calculate_performance_metrics, save_report

class TestBacktestSignals(unittest.TestCase):
    def setUp(self):
        """Set up a dummy price and signal series for testing."""
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
        prices = 100 + np.random.randn(100).cumsum()
        self.price_series = pd.Series(prices, index=dates)

        signals = ['BUY'] * 25 + ['SELL'] * 25 + ['BUY'] * 25 + ['HOLD'] * 25
        pos_strength = np.linspace(0.5, 1.0, 100)
        self.signal_series = pd.DataFrame({
            'date': dates,
            'signal_label': signals,
            'pos_strength': pos_strength
        })
        self.t_costs = 0.0005
        self.reports_dir = 'tests/temp_reports'
        os.makedirs(self.reports_dir, exist_ok=True)

    def tearDown(self):
        """Clean up any created files."""
        if os.path.exists(self.reports_dir):
            for f in os.listdir(self.reports_dir):
                os.remove(os.path.join(self.reports_dir, f))
            os.rmdir(self.reports_dir)

    def test_run_backtest(self):
        """Test the core backtesting engine."""
        equity_curve = run_backtest(self.price_series, self.signal_series, self.t_costs)

        self.assertIsInstance(equity_curve, pd.DataFrame)
        self.assertIn('equity_curve', equity_curve.columns)
        self.assertFalse(equity_curve['equity_curve'].isnull().any())

    def test_calculate_performance_metrics(self):
        """Test the performance metrics calculation."""
        equity_curve_df = run_backtest(self.price_series, self.signal_series, self.t_costs)
        metrics = calculate_performance_metrics(equity_curve_df)

        self.assertIsInstance(metrics, dict)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertGreaterEqual(metrics['max_drawdown'], -1.0)
        self.assertLessEqual(metrics['max_drawdown'], 0.0)

    def test_save_report(self):
        """Test the JSON report saving functionality."""
        dummy_metrics = {'cagr': 0.1, 'sharpe_ratio': 1.5}
        asset = 'DUMMY_ASSET'

        save_report(dummy_metrics, asset, self.reports_dir)

        report_path = os.path.join(self.reports_dir, f"{asset}_performance_report.json")
        self.assertTrue(os.path.exists(report_path))

        with open(report_path, 'r') as f:
            report_data = json.load(f)
        self.assertEqual(report_data['cagr'], 0.1)

if __name__ == '__main__':
    unittest.main()
