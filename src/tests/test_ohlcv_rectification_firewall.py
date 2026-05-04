import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rectification.ohlcv_firewall import (
    CostConfig,
    SplitConfig,
    apply_purged_splits,
    audit_ohlcv_contract,
    build_causal_panel,
    compute_ohlcv_features,
    dataframe_hash,
    feature_catalog_report,
    hypothesis_registry_report,
    load_ohlcv_frame,
    null_control_report,
    score_edge_map,
)


def _synthetic_ohlcv(symbol="AAA", n=260, drift=0.0007):
    dates = pd.bdate_range("2020-01-01", periods=n)
    wave = np.sin(np.arange(n) / 7.0)
    close = 100.0 * np.exp(np.cumsum(drift + 0.004 * wave + 0.002 * np.cos(np.arange(n) / 3.0)))
    open_ = close * (1.0 - 0.001 * np.sign(wave))
    high = np.maximum(open_, close) * (1.0 + 0.004 + 0.001 * np.abs(wave))
    low = np.minimum(open_, close) * (1.0 - 0.004 - 0.001 * np.abs(wave))
    volume = 1_000_000.0 * (1.0 + 0.15 * np.cos(np.arange(n) / 5.0))
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": symbol,
    })


class OHLCVRectificationFirewallTest(unittest.TestCase):
    def test_loader_consumes_only_ohlcv_and_ignores_adjusted_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AAA.csv"
            frame = _synthetic_ohlcv("AAA", 40)
            frame.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}).assign(
                **{"Adj Close": frame["close"] * 0.5, "Ticker": "AAA"}
            ).to_csv(path, index=False)
            loaded = load_ohlcv_frame("AAA", Path(tmp))
            self.assertEqual(set(["open", "high", "low", "close", "volume"]).issubset(loaded.columns), True)
            self.assertNotIn("adj_close", loaded.columns)
            audit = audit_ohlcv_contract({"AAA": loaded}, Path(tmp))
            self.assertIn("Adj Close", audit["per_symbol"]["AAA"]["ignored_columns"])
            self.assertFalse(audit["unresolved_leakage_or_contract_failures"])

    def test_features_and_labels_are_causal_to_decision_timestamp(self):
        base = _synthetic_ohlcv("AAA", 180)
        mutated = base.copy()
        mutated.loc[80, "close"] = mutated.loc[80, "close"] * 3.0
        original_features = compute_ohlcv_features(base)
        mutated_features = compute_ohlcv_features(mutated)
        feature_cols = ["ret_1", "ret_5", "range_pct", "volume_z_20"]
        pd.testing.assert_frame_equal(
            original_features.loc[:40, feature_cols],
            mutated_features.loc[:40, feature_cols],
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        panel = build_causal_panel({"AAA": base}, horizons=(5,), cost_config=CostConfig())
        self.assertTrue((pd.to_datetime(panel["label_timestamp"]) > pd.to_datetime(panel["decision_timestamp"])).all())

    def test_purged_splits_do_not_overlap_and_include_embargo_gap(self):
        frames = {symbol: _synthetic_ohlcv(symbol, 240, drift=0.0004 + i * 0.0001) for i, symbol in enumerate(["AAA", "BBB", "CCC"])}
        panel = build_causal_panel(frames, horizons=(1, 5))
        split_panel, manifest = apply_purged_splits(panel, SplitConfig(purge_bars=5, embargo_bars=20))
        self.assertIn("split_membership_hash", manifest)
        date_sets = {
            split: set(pd.to_datetime(split_panel.loc[split_panel["split"] == split, "decision_timestamp"]).unique())
            for split in ["discovery_train", "validation", "final_test"]
        }
        self.assertTrue(date_sets["discovery_train"].isdisjoint(date_sets["validation"]))
        self.assertTrue(date_sets["validation"].isdisjoint(date_sets["final_test"]))
        self.assertGreater(manifest["validation_start_index_after_embargo"], manifest["train_end_index_after_purge"])
        self.assertGreater(manifest["final_start_index_after_embargo"], manifest["validation_end_index_after_purge"])

    def test_catalog_registry_and_edge_maps_are_diagnostic(self):
        frames = {symbol: _synthetic_ohlcv(symbol, 320, drift=0.0004 + i * 0.0002) for i, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"])}
        panel = build_causal_panel(frames, horizons=(1, 5))
        panel, _manifest = apply_purged_splits(panel, SplitConfig(purge_bars=5, embargo_bars=30))
        catalog = feature_catalog_report(panel)
        registry = hypothesis_registry_report(catalog, horizons=(1, 5))
        self.assertGreater(catalog["feature_count"], 10)
        self.assertTrue(registry["created_before_validation"])
        edge = score_edge_map(panel, fit_split="discovery_train", eval_split="discovery_train", min_samples=50)
        nulls = null_control_report(edge)
        self.assertIn("claim_count", edge)
        self.assertEqual(nulls["claim_count"], edge["claim_count"])
        self.assertEqual(dataframe_hash(panel, ["symbol", "decision_timestamp", "horizon"]), dataframe_hash(panel, ["symbol", "decision_timestamp", "horizon"]))


if __name__ == "__main__":
    unittest.main()
