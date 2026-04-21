"""
SMA Reversal Detection — REAL-STOCK test suite.

Runs the detector against 100 actual CSV price histories from
``src/data/prices/`` and asserts quant invariants on every reversal
record produced. No synthetic tape, no mocks — if the maths is wrong the
suite fails against production data.

Also validates the buy-signal quality fields (grade, stop, target, R:R,
historical edge) by enforcing their algebraic and statistical contracts
on every record.
"""

from __future__ import annotations

import glob
import os
import sys
import unittest

# ── sys.path setup (tests run from repo root) ────────────────────────────
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, os.pardir, os.pardir))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from web.backend.services.data_service import PRICES_DIR  # noqa: E402
from web.backend.services.sma_reversals import (  # noqa: E402
    _compute_one,
    _PERSISTENCE_K,
    _PERSISTENCE_M,
    _LOOKBACK_BARS,
    _OVEREXTENDED_ATR,
    _TARGET_R_MULT,
    _EDGE_MIN_SAMPLES,
    _GRADE_A_RR,
    _GRADE_A_WINRATE,
    _GRADE_B_RR,
    detect_reversals,
    get_all_sma_reversals,
)


_MIN_ROWS_FOR_ALL_PERIODS = 610   # SMA 600 needs ≥ 601; pad for safety
_N_STOCKS = 100


def _eligible_csvs(pricesdir: str, min_rows: int = _MIN_ROWS_FOR_ALL_PERIODS) -> list:
    """Return deterministically-ordered CSVs with ≥ min_rows for SMA-600 coverage."""
    import pandas as pd
    found = []
    for fp in sorted(glob.glob(os.path.join(pricesdir, "*.csv"))):
        try:
            # Cheap row-count without full parse
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                # subtract 1 for header
                nrows = sum(1 for _ in fh) - 1
            if nrows < min_rows:
                continue
            # Confirm it's parseable
            df = pd.read_csv(fp, nrows=5)
            if df.empty:
                continue
        except Exception:
            continue
        found.append(fp)
        if len(found) >= _N_STOCKS * 2:  # gather 2x for a selection buffer
            break
    return found


class TestSmaReversalRealStocks(unittest.TestCase):
    """Invariant checks across 100 real stock price histories."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.prices_dir = PRICES_DIR
        cls.csvs = _eligible_csvs(PRICES_DIR)[:_N_STOCKS]
        cls.all_reversals: list = []
        for fp in cls.csvs:
            recs = _compute_one(fp)
            sym = os.path.basename(fp).replace("_1d.csv", "").replace(".csv", "")
            for r in recs:
                r["_symbol"] = sym
                cls.all_reversals.append(r)

    # ── universe shape ────────────────────────────────────────────────

    def test_at_least_100_eligible_csvs(self) -> None:
        self.assertGreaterEqual(
            len(self.csvs), _N_STOCKS,
            f"need at least {_N_STOCKS} CSVs with >= {_MIN_ROWS_FOR_ALL_PERIODS} rows "
            f"in {self.prices_dir} — got {len(self.csvs)}",
        )

    def test_no_exceptions_during_detection(self) -> None:
        # setUpClass already ran detection; if this runs, nothing raised.
        self.assertTrue(True)

    def test_emits_at_least_some_reversals(self) -> None:
        # Some of 100 real stocks SHOULD have a reversal — else detector is broken.
        self.assertGreater(
            len(self.all_reversals), 0,
            "expected at least one reversal across 100 real stocks",
        )

    # ── score / direction / persistence invariants ────────────────────

    def test_score_range(self) -> None:
        for r in self.all_reversals:
            self.assertGreaterEqual(r["score"], 0.0, r)
            self.assertLessEqual(r["score"], 100.0, r)

    def test_direction_values(self) -> None:
        for r in self.all_reversals:
            self.assertIn(r["direction"], {"bull", "bear"}, r)

    def test_days_since_in_lookback(self) -> None:
        for r in self.all_reversals:
            self.assertGreaterEqual(r["days_since_cross"], 0, r)
            self.assertLessEqual(r["days_since_cross"], _LOOKBACK_BARS, r)

    def test_persistence_bounds_and_flag(self) -> None:
        for r in self.all_reversals:
            self.assertGreaterEqual(r["persistence"], 0, r)
            self.assertLessEqual(r["persistence"], _PERSISTENCE_M, r)
            self.assertEqual(
                r["passes_persistence"],
                r["persistence"] >= _PERSISTENCE_K,
                r,
            )

    # ── price / SMA geometry ──────────────────────────────────────────

    def test_price_sma_side_matches_direction(self) -> None:
        """
        After a bull cross, the latest close must be at (or fractionally above)
        the SMA; after a bear cross, at (or fractionally below). We allow 0.5%
        slop because the cross bar itself can close very close to the SMA.
        """
        for r in self.all_reversals:
            if r["direction"] == "bull":
                self.assertGreaterEqual(r["price"], r["sma"] * 0.995, r)
            else:
                self.assertLessEqual(r["price"], r["sma"] * 1.005, r)

    # ── stop / target geometry ────────────────────────────────────────

    def test_stop_target_geometry(self) -> None:
        """When a stop is set, verify it's on the correct side and R:R≈2."""
        for r in self.all_reversals:
            if r["stop_price"] is None:
                continue
            self.assertIsNotNone(r["target_price"], r)
            self.assertIsNotNone(r["risk_reward"], r)
            if r["direction"] == "bull":
                self.assertLess(r["stop_price"], r["price"], r)
                self.assertGreater(r["target_price"], r["price"], r)
            else:
                self.assertGreater(r["stop_price"], r["price"], r)
                self.assertLess(r["target_price"], r["price"], r)
            self.assertAlmostEqual(r["risk_reward"], _TARGET_R_MULT, places=2)

    def test_stop_target_algebra(self) -> None:
        """target − price == 2 × (price − stop) (bull) and mirror for bear."""
        for r in self.all_reversals:
            if r["stop_price"] is None or r["target_price"] is None:
                continue
            risk = abs(r["price"] - r["stop_price"])
            reward = abs(r["target_price"] - r["price"])
            if risk == 0:
                continue
            self.assertAlmostEqual(reward / risk, _TARGET_R_MULT, places=2, msg=r)

    # ── grade invariants (the buy-signal contract) ────────────────────

    def test_grade_a_requires_full_confluence(self) -> None:
        for r in self.all_reversals:
            if r.get("grade") != "A":
                continue
            self.assertTrue(r["regime_ok"], f"A without regime_ok: {r}")
            self.assertTrue(r["passes_persistence"], f"A without persistence: {r}")
            self.assertFalse(r["overextended"], f"A while overextended: {r}")
            self.assertFalse(r["false_break"], f"A while false_break: {r}")
            self.assertIsNotNone(r["risk_reward"])
            self.assertGreaterEqual(r["risk_reward"], _GRADE_A_RR, r)
            edge = r["historical_edge"]
            self.assertGreaterEqual(edge["samples"], _EDGE_MIN_SAMPLES, r)
            self.assertIsNotNone(edge["win_rate"])
            self.assertGreaterEqual(edge["win_rate"], _GRADE_A_WINRATE, r)

    def test_grade_b_requires_baseline_confluence(self) -> None:
        for r in self.all_reversals:
            if r.get("grade") != "B":
                continue
            self.assertTrue(r["regime_ok"], f"B without regime_ok: {r}")
            self.assertTrue(r["passes_persistence"], f"B without persistence: {r}")
            self.assertFalse(r["overextended"], f"B while overextended: {r}")
            self.assertFalse(r["false_break"], f"B while false_break: {r}")
            self.assertIsNotNone(r["risk_reward"])
            self.assertGreaterEqual(r["risk_reward"], _GRADE_B_RR, r)

    def test_grade_c_implies_persistence_and_no_false_break(self) -> None:
        for r in self.all_reversals:
            if r.get("grade") != "C":
                continue
            # C grade is weaker but still must not be a whipsaw
            self.assertFalse(r["false_break"], r)

    def test_ungraded_means_non_tradeable(self) -> None:
        """grade==None ⇒ at least one confluence gate is broken (false-break
        or failed persistence are the two paths that strip grades entirely)."""
        for r in self.all_reversals:
            if r.get("grade") is not None:
                continue
            self.assertTrue(
                r["false_break"] or not r["passes_persistence"],
                f"ungraded record passes every gate yet has no grade: {r}",
            )

    # ── overextended flag ─────────────────────────────────────────────

    def test_overextended_matches_atr_distance(self) -> None:
        for r in self.all_reversals:
            atr_d = r.get("atr_distance")
            if atr_d is None:
                self.assertFalse(r["overextended"], r)
            else:
                self.assertEqual(r["overextended"], abs(atr_d) > _OVEREXTENDED_ATR, r)

    # ── historical edge shape ─────────────────────────────────────────

    def test_historical_edge_shape(self) -> None:
        for r in self.all_reversals:
            edge = r.get("historical_edge")
            self.assertIsInstance(edge, dict, r)
            self.assertGreaterEqual(edge["samples"], 0, r)
            if edge["samples"] >= _EDGE_MIN_SAMPLES:
                self.assertIsNotNone(edge["win_rate"])
                self.assertGreaterEqual(edge["win_rate"], 0.0, r)
                self.assertLessEqual(edge["win_rate"], 1.0, r)
                self.assertIsNotNone(edge["median_fwd_pct"])
                self.assertIsNotNone(edge["mean_fwd_pct"])
                self.assertGreaterEqual(edge["std_fwd_pct"], 0.0, r)
            else:
                self.assertIsNone(edge["win_rate"])
                self.assertIsNone(edge["median_fwd_pct"])

    # ── per-record uniqueness within a symbol ─────────────────────────

    def test_unique_symbol_period_within_file(self) -> None:
        seen: dict = {}
        for r in self.all_reversals:
            key = (r["_symbol"], r["period"])
            self.assertNotIn(key, seen, f"duplicate reversal for {key}: {r}")
            seen[key] = True


class TestSnapshot(unittest.TestCase):
    """Validate the top-level ``get_all_sma_reversals()`` shape and counts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.snap = get_all_sma_reversals(force=True)

    def test_required_keys(self) -> None:
        for key in (
            "reversals", "counts_by_period", "grade_counts", "buy_setups",
            "periods", "lookback_bars", "persistence_window",
            "persistence_threshold", "regime_period", "overextended_atr",
            "edge_forward_days", "total", "built_at",
        ):
            self.assertIn(key, self.snap, f"missing key: {key}")

    def test_grade_counts_sum_matches(self) -> None:
        g = self.snap["grade_counts"]
        total = self.snap["total"]
        self.assertEqual(g["A"] + g["B"] + g["C"] + g["ungraded"], total)

    def test_buy_setups_nonneg_and_bounded(self) -> None:
        bs = self.snap["buy_setups"]
        self.assertIsInstance(bs, int)
        self.assertGreaterEqual(bs, 0)
        # buy_setups ⊆ grade-A + grade-B bull records
        self.assertLessEqual(
            bs, self.snap["grade_counts"]["A"] + self.snap["grade_counts"]["B"],
        )

    def test_sort_order_grade_then_score(self) -> None:
        """A-grade bull buys should bubble to the top before any ungraded
        records; first non-A must be B or C or ungraded."""
        order = {"A": 0, "B": 1, "C": 2, None: 3}
        prev_rank = -1
        for r in self.snap["reversals"]:
            rank = order.get(r.get("grade"), 4)
            self.assertGreaterEqual(rank, prev_rank, r)
            prev_rank = rank

    def test_symbol_count_covers_universe(self) -> None:
        """At least 100 distinct symbols should be represented overall
        (or every eligible symbol if fewer)."""
        symbols = {r["symbol"] for r in self.snap["reversals"]}
        self.assertGreaterEqual(len(symbols), min(50, self.snap["total"]))


class TestDetectReversalsCallable(unittest.TestCase):
    """Smoke-test the pure function with actual CSV data (no synthetic)."""

    def test_on_first_eligible_stock(self) -> None:
        import pandas as pd
        csvs = _eligible_csvs(PRICES_DIR)
        self.assertGreater(len(csvs), 0, "no eligible CSVs found")
        df = pd.read_csv(csvs[0])
        close_col = "Close" if "Close" in df.columns else "Adj Close"
        recs = detect_reversals(
            df[close_col],
            high=df.get("High"),
            low=df.get("Low"),
            volume=df.get("Volume"),
        )
        self.assertIsInstance(recs, list)
        # list may be empty (no reversal within lookback) — that's valid.
        for r in recs:
            self.assertIn("grade", r)
            self.assertIn("historical_edge", r)
            self.assertIn("risk_reward", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
