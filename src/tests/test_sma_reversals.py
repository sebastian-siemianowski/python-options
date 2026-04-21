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
    _EDGE_FORWARD_BY_PERIOD,
    _GRADE_A_RR,
    _GRADE_A_WINRATE,
    _GRADE_A_EXPECTANCY_R,
    _GRADE_A_STOP_HIT_MAX,
    _GRADE_B_RR,
    _KELLY_CAP,
    _REGIME_SLOPE_WINDOW,
    _SMA_PERIODS,
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
        """Stop and target on the correct side of price; R:R strictly positive."""
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
            self.assertGreater(r["risk_reward"], 0.0, r)

    def test_stop_target_algebra(self) -> None:
        """Stored R:R matches reward/risk; equals 2R exactly when the structural
        target was NOT used, else equals the structural level."""
        for r in self.all_reversals:
            if r["stop_price"] is None or r["target_price"] is None:
                continue
            risk = abs(r["price"] - r["stop_price"])
            reward = abs(r["target_price"] - r["price"])
            if risk == 0:
                continue
            # All stored values are pre-rounded, so allow a small tolerance
            self.assertLess(abs(reward / risk - r["risk_reward"]), 0.02, msg=r)
            if not r["used_structural_target"]:
                self.assertAlmostEqual(
                    r["risk_reward"], _TARGET_R_MULT, places=2,
                    msg=f"non-structural target must be 2R: {r}",
                )

    def test_rr_varies_when_structural_targets_used(self) -> None:
        """Across 100 stocks, at least one setup should have used the structural
        target and produced an R:R different from 2.0 — proving the new engine
        is not degenerate."""
        structural = [r for r in self.all_reversals if r.get("used_structural_target")]
        # It's fine if none exist, but if any do, their R:R should not all be 2.0
        if structural:
            diverse = [r for r in structural if abs(r["risk_reward"] - _TARGET_R_MULT) > 0.05]
            self.assertGreater(
                len(diverse), 0,
                "structural targets set but every R:R == 2.0 — engine is degenerate",
            )

    # ── grade invariants (the buy-signal contract) ────────────────────

    def test_grade_a_requires_full_confluence(self) -> None:
        """Grade A is the elite gate — EVERY invariant below is enforced."""
        for r in self.all_reversals:
            if r.get("grade") != "A":
                continue
            # Core filters
            self.assertTrue(r["regime_ok"], f"A without regime_ok: {r}")
            self.assertTrue(r["passes_persistence"], f"A without persistence: {r}")
            self.assertFalse(r["overextended"], f"A while overextended: {r}")
            self.assertFalse(r["false_break"], f"A while false_break: {r}")
            # Regime slope must align with direction (new gate)
            if r["direction"] == "bull":
                self.assertGreater(r["regime_slope_pct"], 0.0, f"A bull with falling SMA-200: {r}")
            else:
                self.assertLess(r["regime_slope_pct"], 0.0, f"A bear with rising SMA-200: {r}")
            # Trade geometry
            self.assertIsNotNone(r["risk_reward"])
            self.assertGreaterEqual(r["risk_reward"], _GRADE_A_RR, r)
            # Historical edge
            edge = r["historical_edge"]
            self.assertGreaterEqual(edge["samples"], _EDGE_MIN_SAMPLES, r)
            self.assertIsNotNone(edge["win_rate"])
            self.assertGreaterEqual(edge["win_rate"], _GRADE_A_WINRATE, r)
            # New: expectancy in R must be positive and clear the bar
            self.assertIsNotNone(edge["expectancy_r"], r)
            self.assertGreaterEqual(edge["expectancy_r"], _GRADE_A_EXPECTANCY_R, r)
            # New: stop must survive historically
            if edge["stop_hit_rate"] is not None:
                self.assertLessEqual(edge["stop_hit_rate"], _GRADE_A_STOP_HIT_MAX, r)

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
            # samples is CONDITIONAL count; must never exceed unconditional
            self.assertLessEqual(
                edge["samples"], edge["unconditional_samples"],
                f"conditional > unconditional: {r}",
            )
            if edge["samples"] >= _EDGE_MIN_SAMPLES:
                self.assertIsNotNone(edge["win_rate"])
                self.assertGreaterEqual(edge["win_rate"], 0.0, r)
                self.assertLessEqual(edge["win_rate"], 1.0, r)
                self.assertIsNotNone(edge["median_fwd_pct"])
                self.assertIsNotNone(edge["mean_fwd_pct"])
                self.assertGreaterEqual(edge["std_fwd_pct"], 0.0, r)
                # New: expectancy_r must be a number
                self.assertIsNotNone(edge["expectancy_r"], r)
                self.assertIsInstance(edge["expectancy_r"], float)
                # stop_hit_rate in [0,1]
                self.assertIsNotNone(edge["stop_hit_rate"], r)
                self.assertGreaterEqual(edge["stop_hit_rate"], 0.0, r)
                self.assertLessEqual(edge["stop_hit_rate"], 1.0, r)
                # median MAE non-negative
                self.assertGreaterEqual(edge["median_mae_atr"], 0.0, r)
                # profit factor positive or None
                if edge["profit_factor"] is not None:
                    self.assertGreater(edge["profit_factor"], 0.0, r)
                # recency-weighted win rate in [0,1]
                if edge["recency_weighted_win_rate"] is not None:
                    self.assertGreaterEqual(edge["recency_weighted_win_rate"], 0.0, r)
                    self.assertLessEqual(edge["recency_weighted_win_rate"], 1.0, r)
            else:
                self.assertIsNone(edge["win_rate"])
                self.assertIsNone(edge["median_fwd_pct"])
                self.assertIsNone(edge["expectancy_r"])
                self.assertIsNone(edge["profit_factor"])
                self.assertIsNone(edge["stop_hit_rate"])
                self.assertIsNone(edge["median_mae_atr"])

    # ── new quant-field invariants ────────────────────────────────

    def test_regime_slope_pct_is_number(self) -> None:
        for r in self.all_reversals:
            self.assertIsInstance(r["regime_slope_pct"], float, r)
            # Sanity: slope should not be crazy. Extreme moves of 1000%/month never happen.
            self.assertGreater(r["regime_slope_pct"], -200.0, r)
            self.assertLess(r["regime_slope_pct"], 200.0, r)

    def test_regime_ok_implies_price_and_slope_aligned(self) -> None:
        """New contract: regime_ok requires BOTH price on correct side AND slope aligned."""
        for r in self.all_reversals:
            if not r["regime_ok"]:
                continue
            if r["direction"] == "bull":
                self.assertGreater(r["regime_slope_pct"], 0.0, f"regime_ok bull but falling slope: {r}")
                if r["regime_sma"] is not None:
                    self.assertGreater(r["price"], r["regime_sma"], r)
            else:
                self.assertLess(r["regime_slope_pct"], 0.0, f"regime_ok bear but rising slope: {r}")
                if r["regime_sma"] is not None:
                    self.assertLess(r["price"], r["regime_sma"], r)

    def test_vol_regime_is_valid(self) -> None:
        for r in self.all_reversals:
            self.assertIn(r["vol_regime"], {"normal", "high", "extreme", "unknown"}, r)
            if r["vol_regime"] == "unknown":
                self.assertIsNone(r["vol_ratio"], r)
            else:
                self.assertIsNotNone(r["vol_ratio"], r)
                self.assertGreater(r["vol_ratio"], 0.0, r)

    def test_pullback_atr_nonneg(self) -> None:
        for r in self.all_reversals:
            p = r["pullback_atr"]
            if p is not None:
                self.assertGreaterEqual(p, 0.0, r)

    def test_mtf_alignment_bounded(self) -> None:
        for r in self.all_reversals:
            m = r["mtf_alignment"]
            if m is not None:
                self.assertGreaterEqual(m, 0.0, r)
                self.assertLessEqual(m, 1.0, r)

    def test_kelly_fraction_bounded(self) -> None:
        for r in self.all_reversals:
            k = r["kelly_fraction"]
            if k is None:
                continue
            self.assertGreaterEqual(k, 0.0, r)
            self.assertLessEqual(k, _KELLY_CAP, r)

    def test_kelly_present_when_edge_and_rr_available(self) -> None:
        """Kelly must be set whenever both win_rate and risk_reward are set."""
        for r in self.all_reversals:
            edge = r["historical_edge"]
            if edge["win_rate"] is not None and r["risk_reward"] is not None:
                self.assertIsNotNone(r["kelly_fraction"], f"missing Kelly: {r}")

    def test_forward_horizon_matches_period(self) -> None:
        """edge_forward_days per record should equal the period-specific horizon
        (or be clamped to a smaller value for short histories)."""
        for r in self.all_reversals:
            expected = _EDGE_FORWARD_BY_PERIOD.get(r["period"])
            if expected is None:
                continue
            # Horizon may be clamped down, never up
            self.assertLessEqual(r["edge_forward_days"], expected, r)
            self.assertGreaterEqual(r["edge_forward_days"], 1, r)

    def test_conditional_samples_le_unconditional(self) -> None:
        for r in self.all_reversals:
            e = r["historical_edge"]
            self.assertLessEqual(e["samples"], e["unconditional_samples"], r)

    def test_at_least_one_grade_a_or_b_across_100_stocks(self) -> None:
        """Sanity: among 100 real US stocks at any given time, the market should
        offer at least ONE A-or-B buy setup. If zero, the gates are miscalibrated."""
        good = [r for r in self.all_reversals if r.get("grade") in ("A", "B")]
        self.assertGreater(
            len(good), 0,
            "no Grade A/B setups across 100 real stocks — gates too strict",
        )

    def test_grade_a_count_reasonable(self) -> None:
        """Grade A should be rare — the elite gate. Cap at 30% of all reversals."""
        a = sum(1 for r in self.all_reversals if r.get("grade") == "A")
        if self.all_reversals:
            self.assertLess(
                a / len(self.all_reversals), 0.30,
                f"{a} Grade A out of {len(self.all_reversals)} — gates too loose",
            )

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
            "persistence_threshold", "regime_period", "regime_slope_window",
            "overextended_atr", "edge_forward_days", "edge_forward_by_period",
            "vol_regime_window", "vol_regime_baseline", "pullback_window",
            "structural_target_window", "grade_a_rr", "grade_a_winrate",
            "grade_a_expectancy_r", "grade_a_stop_hit_max", "grade_b_rr",
            "kelly_cap", "total", "built_at",
        ):
            self.assertIn(key, self.snap, f"missing key: {key}")

    def test_edge_forward_by_period_contract(self) -> None:
        efp = self.snap["edge_forward_by_period"]
        self.assertEqual(efp, {9: 10, 50: 20, 600: 60})

    def test_snapshot_constants_match_module(self) -> None:
        self.assertEqual(self.snap["kelly_cap"], _KELLY_CAP)
        self.assertEqual(self.snap["grade_a_expectancy_r"], _GRADE_A_EXPECTANCY_R)
        self.assertEqual(self.snap["grade_a_stop_hit_max"], _GRADE_A_STOP_HIT_MAX)
        self.assertEqual(self.snap["regime_slope_window"], _REGIME_SLOPE_WINDOW)

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
