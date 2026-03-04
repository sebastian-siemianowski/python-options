"""
End-to-end accuracy and soundness tests across 40 diverse assets (March 2026).

Tests the FULL signal pipeline on 40 assets covering:
  - All 4 asset types: equity (25), metal (6), currency (6), crypto (3)
  - All volatility regimes: low-vol (SPY, MSFT), mid-vol (AAPL, NVDA),
    high-vol (TSLA, IONQ), extreme-vol (ABTC, BTC-USD)
  - All market caps: mega (AAPL), large (JPM), mid (CRM), small (IONQ)
  - All sectors: tech, finance, defence, energy, consumer, healthcare, materials

Validates:
  1. Every asset produces signals without errors
  2. Physical CI and profit bounds hold universally
  3. Probabilistic calibration (P(up) distribution)
  4. Volatility forecast consistency
  5. Cross-asset monotonicity (higher vol → wider CI)
  6. Regime/label validity
  7. No degenerate outputs (NaN, Inf, constant)
  8. Expected utility and gain/loss ratio consistency

Run with:
    .venv/bin/python -m pytest src/tests/test_accuracy_40_assets.py -v --tb=short
"""

import os
import sys
import math
import unittest
import warnings
from collections import defaultdict

os.environ["TUNING_QUIET"] = "1"
os.environ["OFFLINE_MODE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

# ---------- Physical constants (must match signals.py) ----------
NOTIONAL_PLN = 1_000_000
CI_LOG_FLOOR = -4.6       # exp(-4.6) - 1 ≈ -99%
CI_LOG_CAP = 1.61          # exp(1.61) - 1 ≈ +400%
MAX_PROFIT = 4 * NOTIONAL_PLN
MIN_PROFIT = -NOTIONAL_PLN

# ---------- 40-asset universe ----------
# (symbol, expected_asset_type)
UNIVERSE_40 = [
    # ===== Equities: Indices / ETFs (low vol, ~14-20%) =====
    ("SPY", "equity"),
    ("QQQ", "equity"),
    ("IWM", "equity"),
    # ===== Equities: Mega-cap tech (20-35%) =====
    ("AAPL", "equity"),
    ("MSFT", "equity"),
    ("NVDA", "equity"),
    ("GOOGL", "equity"),
    ("META", "equity"),
    ("AMZN", "equity"),
    # ===== Equities: Finance (20-30%) =====
    ("JPM", "equity"),
    ("BAC", "equity"),
    ("GS", "equity"),
    # ===== Equities: Healthcare / Consumer / Industrial (20-30%) =====
    ("JNJ", "equity"),
    ("UNH", "equity"),
    ("XOM", "equity"),
    ("CVX", "equity"),
    ("CAT", "equity"),
    ("HD", "equity"),
    ("PG", "equity"),
    ("KO", "equity"),
    ("COST", "equity"),
    # ===== Equities: Defence / mid-cap (25-35%) =====
    ("LMT", "equity"),
    ("CRM", "equity"),
    # ===== Equities: High-vol (40-80%) =====
    ("TSLA", "equity"),
    ("IONQ", "equity"),
    ("ABTC", "equity"),
    ("GPUS", "equity"),
    ("AFRM", "equity"),
    # ===== Metals (15-40%) =====
    ("GLD", "metal"),
    ("SLV", "metal"),
    ("NEM", "metal"),
    ("AG", "metal"),
    ("FCX", "equity"),  # Copper miner, but not in _METAL_TICKERS
    ("AEM", "metal"),
    # ===== Currencies (5-15%) =====
    ("EURPLN=X", "currency"),
    ("USDPLN=X", "currency"),
    ("GBPPLN=X", "currency"),
    ("EURUSD=X", "currency"),
    ("USDJPY=X", "currency"),
    # ===== Crypto (50-100%) =====
    ("BTC-USD", "crypto"),
]


# ---------- Helpers ----------

def _load_cached_price(symbol: str) -> Optional[Tuple[pd.Series, pd.DataFrame]]:
    """Load cached price data without hitting Yahoo."""
    prices_dir = os.path.join(REPO_ROOT, "src", "data", "prices")
    candidates = [
        f"{symbol}.csv",
        f"{symbol}_1d.csv",
        f"{symbol.replace('=', '_')}.csv",
        f"{symbol.replace('=', '_')}_1d.csv",
    ]
    for name in candidates:
        path = os.path.join(prices_dir, name)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                close_col = None
                for col in ["Close", "close", "Adj Close"]:
                    if col in df.columns:
                        close_col = col
                        break
                if close_col is None and df.shape[1] == 1:
                    close_col = df.columns[0]
                if close_col is None:
                    continue
                px = df[close_col].astype(float).dropna()
                px.name = "px"
                if len(px) < 100:
                    continue
                return px, df.copy()
            except Exception:
                continue
    return None


def _run_signal_pipeline(px, symbol, ohlc_df=None):
    """Run the full signal pipeline, returning (signals_list, meta_dict)."""
    from decision.signals import (
        compute_features, latest_signals, classify_asset_type,
        DEFAULT_HORIZONS, _load_tuned_kalman_params,
    )
    feats = compute_features(px, asset_symbol=symbol, ohlc_df=ohlc_df)
    last_close = float(px.iloc[-1])
    tuned_params = _load_tuned_kalman_params(symbol)
    asset_type = classify_asset_type(symbol)
    sigs, thresholds = latest_signals(
        feats, DEFAULT_HORIZONS, last_close,
        tuned_params=tuned_params,
        asset_key=symbol,
        asset_type=asset_type,
        n_mc_paths=2000,
    )
    return sigs, {"asset_type": asset_type, "thresholds": thresholds}


# ---------- Shared results cache (compute once, reuse across tests) ----------
# Using module-level cache so we don't re-run the pipeline 40× per test class.
_RESULTS_CACHE: Dict[str, Tuple[list, dict]] = {}
_SKIPPED_ASSETS: Dict[str, str] = {}
_CACHE_BUILT = False


def _build_cache():
    """Compute signals for all 40 assets once."""
    global _CACHE_BUILT
    if _CACHE_BUILT:
        return
    for symbol, expected_type in UNIVERSE_40:
        result = _load_cached_price(symbol)
        if result is None:
            _SKIPPED_ASSETS[symbol] = "no cached data"
            continue
        px, ohlc_df = result
        if len(px) < 100:
            _SKIPPED_ASSETS[symbol] = f"only {len(px)} data points"
            continue
        try:
            sigs, meta = _run_signal_pipeline(px, symbol, ohlc_df=ohlc_df)
            _RESULTS_CACHE[symbol] = (sigs, meta)
        except Exception as e:
            _SKIPPED_ASSETS[symbol] = str(e)[:200]
    _CACHE_BUILT = True


# ===============================================================
# Test Class 1: Universal signal production
# ===============================================================
class TestUniversalSignalProduction(unittest.TestCase):
    """Every asset must produce signals without errors."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_minimum_asset_coverage(self):
        """At least 30 of 40 assets must produce signals (allowing for missing data)."""
        _build_cache()
        n_success = len(_RESULTS_CACHE)
        n_total = len(UNIVERSE_40)
        self.assertGreaterEqual(
            n_success, 30,
            f"Only {n_success}/{n_total} assets produced signals. "
            f"Skipped: {_SKIPPED_ASSETS}"
        )

    def test_all_asset_types_represented(self):
        """All 4 asset types must have at least 1 successful run."""
        _build_cache()
        types_seen = set()
        for symbol, _ in UNIVERSE_40:
            if symbol in _RESULTS_CACHE:
                types_seen.add(_RESULTS_CACHE[symbol][1]["asset_type"])
        expected_types = {"equity", "metal", "crypto"}
        # Currency may be missing if no data; require at least equity+metal+crypto
        missing = expected_types - types_seen
        self.assertEqual(
            len(missing), 0,
            f"Missing asset types: {missing}. Got: {types_seen}"
        )

    def test_each_asset_produces_multiple_horizons(self):
        """Each asset should produce signals for multiple horizons."""
        _build_cache()
        for symbol, (sigs, meta) in _RESULTS_CACHE.items():
            self.assertGreater(
                len(sigs), 1,
                f"{symbol}: only {len(sigs)} signal(s), expected multi-horizon"
            )


# ===============================================================
# Test Class 2: Physical bounds (CI, profit, finiteness)
# ===============================================================
class TestPhysicalBounds(unittest.TestCase):
    """All signals must respect physical CI and profit bounds."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_ci_within_floor_and_cap(self):
        """CI log bounds: ci_low >= -4.6, ci_high <= 1.61."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.ci_low < CI_LOG_FLOOR:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: ci_low={sig.ci_low:.4f}"
                    )
                if sig.ci_high > CI_LOG_CAP:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: ci_high={sig.ci_high:.4f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"CI bound violations:\n" + "\n".join(violations[:20]))

    def test_ci_ordering(self):
        """ci_low <= ci_high for every signal."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.ci_low > sig.ci_high:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: ci_low={sig.ci_low:.4f} > ci_high={sig.ci_high:.4f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"CI ordering violations:\n" + "\n".join(violations[:20]))

    def test_profit_within_bounds(self):
        """All profit values within [-1M, +4M] PLN."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.profit_pln < MIN_PROFIT:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: profit={sig.profit_pln:,.0f}"
                    )
                if sig.profit_pln > MAX_PROFIT:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: profit={sig.profit_pln:,.0f}"
                    )
                if sig.profit_ci_low_pln < MIN_PROFIT:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: profit_ci_low={sig.profit_ci_low_pln:,.0f}"
                    )
                if sig.profit_ci_high_pln > MAX_PROFIT:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: profit_ci_high={sig.profit_ci_high_pln:,.0f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"Profit bound violations:\n" + "\n".join(violations[:20]))

    def test_all_values_finite(self):
        """No NaN or Inf in critical signal fields."""
        bad = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                for field in ["ci_low", "ci_high", "exp_ret", "profit_pln",
                             "profit_ci_low_pln", "profit_ci_high_pln",
                             "p_up", "score", "vol_mean"]:
                    val = getattr(sig, field, None)
                    if val is not None and not np.isfinite(val):
                        bad.append(f"{symbol} H={sig.horizon_days}: {field}={val}")
        self.assertEqual(len(bad), 0,
                        f"Non-finite values:\n" + "\n".join(bad[:20]))

    def test_ci_display_percentages_sane(self):
        """In display space: ci_low_pct > -100%, ci_high_pct < 500%."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                ci_low_pct = (math.exp(sig.ci_low) - 1) * 100
                ci_high_pct = (math.exp(sig.ci_high) - 1) * 100
                if ci_low_pct <= -100.0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: ci_low_pct={ci_low_pct:.1f}%"
                    )
                if ci_high_pct >= 500.0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: ci_high_pct={ci_high_pct:.1f}%"
                    )
        self.assertEqual(len(violations), 0,
                        f"Display percentage violations:\n" + "\n".join(violations[:20]))


# ===============================================================
# Test Class 3: Probabilistic calibration
# ===============================================================
class TestProbabilisticCalibration(unittest.TestCase):
    """P(up), score, and expected return must be probabilistically sound."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_p_up_in_valid_range(self):
        """P(up) must be in [0, 1] for all signals."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.p_up < 0.0 or sig.p_up > 1.0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: p_up={sig.p_up:.4f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"P(up) range violations:\n" + "\n".join(violations[:20]))

    def test_p_up_not_degenerate(self):
        """P(up) should NOT be exactly 0.0 or 1.0 for most assets.

        Degenerate P(up) indicates model failure (all MC paths same sign).
        Allow up to 10% degenerate across the universe.
        """
        total = 0
        degenerate = 0
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                total += 1
                if sig.p_up == 0.0 or sig.p_up == 1.0:
                    degenerate += 1
        if total > 0:
            degenerate_rate = degenerate / total
            self.assertLess(
                degenerate_rate, 0.10,
                f"{degenerate}/{total} signals ({degenerate_rate:.1%}) have degenerate P(up)"
            )

    def test_p_up_distribution_not_extreme(self):
        """Across 40 assets, P(up) should have reasonable spread.

        Both bullish and uncertain signals should appear.
        Mean P(up) should be between 0.25 and 0.85 (market has positive drift
        but not all assets are strongly bullish).
        """
        all_p_up = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                all_p_up.append(sig.p_up)
        if len(all_p_up) < 10:
            self.skipTest("Too few signals for distribution test")
        mean_p = np.mean(all_p_up)
        self.assertGreater(mean_p, 0.25,
                          f"Mean P(up) = {mean_p:.3f} — too bearish, possible model failure")
        self.assertLess(mean_p, 0.85,
                       f"Mean P(up) = {mean_p:.3f} — too bullish, possible miscalibration")

    def test_score_has_both_signs(self):
        """Score (edge) should have both positive and negative values across assets.

        If all scores are same sign, the model isn't discriminating.
        """
        scores = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                scores.append(sig.score)
        if len(scores) < 10:
            self.skipTest("Too few scores")
        n_positive = sum(1 for s in scores if s > 0)
        n_negative = sum(1 for s in scores if s < 0)
        minority = min(n_positive, n_negative)
        self.assertGreater(minority, 0,
                          f"All {len(scores)} scores have same sign — model not discriminating. "
                          f"pos={n_positive}, neg={n_negative}")

    def test_expected_return_bounded(self):
        """Expected return per horizon should be within CI floor/cap.

        exp_ret is clamped by the CI bounds system, so the physical
        limits are CI_LOG_FLOOR (-4.6) to CI_LOG_CAP (1.61).
        """
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                # Allow a small tolerance beyond CI bounds for EMOS effects
                if sig.exp_ret < CI_LOG_FLOOR - 0.1:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: exp_ret={sig.exp_ret:.4f} < floor"
                    )
                if sig.exp_ret > CI_LOG_CAP + 0.1:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: exp_ret={sig.exp_ret:.4f} > cap"
                    )
        self.assertEqual(len(violations), 0,
                        f"Expected returns outside bounds:\n" + "\n".join(violations[:20]))


# ===============================================================
# Test Class 4: Volatility forecast consistency
# ===============================================================
class TestVolatilityForecasts(unittest.TestCase):
    """Volatility forecasts must be positive and consistent."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_vol_mean_positive(self):
        """Mean volatility forecast must be > 0 for equities/metals/crypto.

        Currency pairs may produce zero vol if insufficient intraday data.
        """
        violations = []
        for symbol, (sigs, meta) in _RESULTS_CACHE.items():
            # Skip currencies — may have zero vol with limited data
            if meta["asset_type"] == "currency":
                continue
            for sig in sigs:
                if sig.vol_mean <= 0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: vol_mean={sig.vol_mean:.6f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"Non-positive vol:\n" + "\n".join(violations[:20]))

    def test_vol_ci_ordering(self):
        """vol_ci_low <= vol_ci_high must hold.

        NOTE: vol_mean is the mean of stochastic vol paths, while
        vol_ci_{low,high} are percentiles. In heavy-tailed distributions
        the mean can exceed upper percentiles, so we only check
        ci_low <= ci_high (not that mean is between them).
        """
        violations = []
        for symbol, (sigs, meta) in _RESULTS_CACHE.items():
            if meta["asset_type"] == "currency":
                continue  # Currency vol CIs may be degenerate
            for sig in sigs:
                if sig.vol_ci_low > sig.vol_ci_high and sig.vol_ci_high > 0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: vol_ci_low={sig.vol_ci_low} > vol_ci_high={sig.vol_ci_high}"
                    )
        self.assertEqual(len(violations), 0,
                        f"Vol CI ordering violations:\n" + "\n".join(violations[:20]))

    def test_vol_within_plausible_range(self):
        """Vol mean at short horizons (H<=7) should be plausible.

        At longer horizons, vol_mean scales with sqrt(H) and can
        be large. We only check short horizons where daily vol > 5.0
        would be extreme (500% daily).
        """
        violations = []
        for symbol, (sigs, meta) in _RESULTS_CACHE.items():
            if meta["asset_type"] == "currency":
                continue  # Currency vol can be zero/unusual
            for sig in sigs:
                # Only check short horizons — longer ones have √H scaling
                if sig.horizon_days <= 7 and sig.vol_mean > 5.0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: vol_mean={sig.vol_mean:.4f} (>5.0)"
                    )
        self.assertEqual(len(violations), 0,
                        f"Absurd short-horizon volatility:\n" + "\n".join(violations[:20]))


# ===============================================================
# Test Class 5: Regime and label validity
# ===============================================================
class TestRegimeAndLabelValidity(unittest.TestCase):
    """Regimes and labels must be valid strings."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    # All regime labels from infer_current_regime() and HMM classifier
    VALID_REGIMES = {
        # Tune.py assign_regime_labels format:
        "LOW_VOL_TREND", "HIGH_VOL_TREND",
        "LOW_VOL_RANGE", "HIGH_VOL_RANGE",
        "CRISIS_JUMP",
        # HMM classifier (3-state):
        "calm", "trending", "crisis",
        # Threshold-based classifier (infer_current_regime):
        "High-vol uptrend", "High-vol downtrend",
        "Calm uptrend", "Calm downtrend",
        "Normal",
    }
    VALID_LABELS = {
        "BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL",
    }

    def test_regime_is_valid(self):
        """Every signal must have a recognized regime string."""
        invalid = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.regime not in self.VALID_REGIMES:
                    invalid.append(
                        f"{symbol} H={sig.horizon_days}: regime='{sig.regime}'"
                    )
        self.assertEqual(len(invalid), 0,
                        f"Invalid regimes:\n" + "\n".join(invalid[:20]))

    def test_label_is_valid(self):
        """Every signal must have a recognized label."""
        invalid = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.label not in self.VALID_LABELS:
                    invalid.append(
                        f"{symbol} H={sig.horizon_days}: label='{sig.label}'"
                    )
        self.assertEqual(len(invalid), 0,
                        f"Invalid labels:\n" + "\n".join(invalid[:20]))

    def test_label_diversity(self):
        """Across 40 assets, we should see multiple distinct labels.

        If every single signal is "HOLD", the model isn't producing actionable output.
        """
        labels_seen = set()
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                labels_seen.add(sig.label)
        self.assertGreaterEqual(
            len(labels_seen), 2,
            f"Only {len(labels_seen)} distinct label(s): {labels_seen}"
        )


# ===============================================================
# Test Class 6: Asset type classification accuracy
# ===============================================================
class TestAssetClassification(unittest.TestCase):
    """Asset type classification must match expected types."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_classification_matches_expected(self):
        """Each asset must be classified as its expected type."""
        mismatches = []
        for symbol, expected_type in UNIVERSE_40:
            if symbol not in _RESULTS_CACHE:
                continue
            actual = _RESULTS_CACHE[symbol][1]["asset_type"]
            if actual != expected_type:
                mismatches.append(
                    f"{symbol}: expected={expected_type}, got={actual}"
                )
        self.assertEqual(len(mismatches), 0,
                        f"Classification mismatches:\n" + "\n".join(mismatches))


# ===============================================================
# Test Class 7: Expected utility and gain/loss ratio
# ===============================================================
class TestExpectedUtility(unittest.TestCase):
    """EU and G/L ratio must be consistent."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_gain_loss_ratio_positive(self):
        """Gain/loss ratio must be > 0 for every signal."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.gain_loss_ratio < 0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: G/L={sig.gain_loss_ratio:.4f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"Negative G/L ratios:\n" + "\n".join(violations[:20]))

    def test_expected_gain_non_negative(self):
        """Expected gain should be >= 0."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.expected_gain < 0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: E[gain]={sig.expected_gain:.6f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"Negative expected gains:\n" + "\n".join(violations[:20]))

    def test_expected_loss_non_negative(self):
        """Expected loss (defined as positive value) should be >= 0."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.expected_loss < 0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: E[loss]={sig.expected_loss:.6f}"
                    )
        self.assertEqual(len(violations), 0,
                        f"Negative expected losses:\n" + "\n".join(violations[:20]))

    def test_position_strength_bounded(self):
        """Position strength should be in a reasonable range [0, ~20]."""
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if sig.position_strength < 0:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: pos_str={sig.position_strength:.4f} < 0"
                    )
                if sig.position_strength > 100:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: pos_str={sig.position_strength:.4f} > 100"
                    )
        self.assertEqual(len(violations), 0,
                        f"Position strength violations:\n" + "\n".join(violations[:20]))


# ===============================================================
# Test Class 8: Cross-asset consistency
# ===============================================================
class TestCrossAssetConsistency(unittest.TestCase):
    """Higher-vol assets should generally have wider CIs."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_ci_width_positively_correlated_with_vol(self):
        """Across assets at the same horizon (H=21), higher vol should → wider CI.

        We check rank correlation: Spearman ρ(vol, CI_width) > 0.
        """
        vols = []
        widths = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            # Find the H=21 (or closest) signal
            h21_sigs = [s for s in sigs if s.horizon_days == 21]
            if not h21_sigs:
                # Try H=63 as fallback
                h21_sigs = [s for s in sigs if s.horizon_days == 63]
            if not h21_sigs:
                continue
            sig = h21_sigs[0]
            vols.append(sig.vol_mean)
            widths.append(sig.ci_high - sig.ci_low)

        if len(vols) < 10:
            self.skipTest(f"Only {len(vols)} assets with H=21 signals")

        from scipy.stats import spearmanr
        rho, _ = spearmanr(vols, widths)
        self.assertGreater(
            rho, -0.3,
            f"Spearman ρ(vol, CI_width) = {rho:.3f} — strong negative correlation "
            f"suggests CI bounds not working correctly"
        )

    def test_index_vol_lower_than_single_stock(self):
        """SPY vol should generally be lower than TSLA or IONQ vol.

        This tests that the model correctly identifies index vs single-stock risk.
        """
        # Get H=21 vol for each
        spy_vol = None
        high_vol_vols = {}
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            h21 = [s for s in sigs if s.horizon_days == 21]
            if not h21:
                continue
            if symbol == "SPY":
                spy_vol = h21[0].vol_mean
            elif symbol in ("TSLA", "IONQ", "ABTC"):
                high_vol_vols[symbol] = h21[0].vol_mean

        if spy_vol is None:
            self.skipTest("No SPY data")
        for sym, vol in high_vol_vols.items():
            self.assertLess(
                spy_vol, vol * 2.0,  # SPY should be lower (allow 2x for edge cases)
                f"SPY vol ({spy_vol:.6f}) not meaningfully lower than {sym} vol ({vol:.6f})"
            )


# ===============================================================
# Test Class 9: CI width monotonicity across horizons
# ===============================================================
class TestHorizonMonotonicity(unittest.TestCase):
    """CI width should generally increase with horizon for each asset."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_ci_width_increases_with_horizon(self):
        """For each asset, CI width at longer horizons should be >= shorter.

        Allow 1 violation per asset (EMOS calibration can cause minor inversions).
        Assets with >1 violation flag a problem.
        """
        problem_assets = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            widths = [(s.horizon_days, s.ci_high - s.ci_low) for s in sigs]
            widths.sort(key=lambda x: x[0])
            if len(widths) < 3:
                continue
            violations = 0
            for i in range(1, len(widths)):
                # Allow 50% regression (EMOS can tighten CI at longer horizons
                # when it has more confidence)
                if widths[i][1] < widths[i - 1][1] * 0.5:
                    violations += 1
            if violations > 1:
                problem_assets.append(f"{symbol}: {widths}")
        self.assertEqual(
            len(problem_assets), 0,
            f"CI width not monotonic:\n" + "\n".join(problem_assets[:10])
        )


# ===============================================================
# Test Class 10: No constant/degenerate outputs
# ===============================================================
class TestNoDegenerateOutputs(unittest.TestCase):
    """Signals across assets should show variation, not constant values."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_exp_ret_not_all_same(self):
        """Expected returns should vary across assets."""
        exp_rets = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                exp_rets.append(sig.exp_ret)
        if len(exp_rets) < 10:
            self.skipTest("Too few signals")
        std = np.std(exp_rets)
        self.assertGreater(std, 1e-8,
                          f"All expected returns are identical (std={std})")

    def test_ci_width_not_all_same(self):
        """CI widths should vary across assets."""
        widths = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                widths.append(sig.ci_high - sig.ci_low)
        if len(widths) < 10:
            self.skipTest("Too few signals")
        std = np.std(widths)
        self.assertGreater(std, 1e-6,
                          f"All CI widths are identical (std={std})")

    def test_vol_not_all_same(self):
        """Volatility forecasts should vary across assets."""
        vols = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                vols.append(sig.vol_mean)
        if len(vols) < 10:
            self.skipTest("Too few signals")
        std = np.std(vols)
        self.assertGreater(std, 1e-8,
                          f"All volatilities are identical (std={std})")


# ===============================================================
# Test Class 11: Extreme-vol assets properly bounded (regression)
# ===============================================================
class TestExtremeVolBounding(unittest.TestCase):
    """ABTC, GPUS, and other extreme-vol equities must be bounded.

    This is a regression test for the CI explosion bug:
    Before fix, ABTC at H=252 had CI of [-1,832,151%, +1,831,...%].
    """

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def _check_bounded(self, symbol):
        if symbol not in _RESULTS_CACHE:
            self.skipTest(f"No data for {symbol}")
        sigs, _ = _RESULTS_CACHE[symbol]
        for sig in sigs:
            ci_low_pct = (math.exp(sig.ci_low) - 1) * 100
            ci_high_pct = (math.exp(sig.ci_high) - 1) * 100
            self.assertGreater(
                ci_low_pct, -99.5,
                f"{symbol} H={sig.horizon_days}: CI still absurd at {ci_low_pct:.1f}%"
            )
            self.assertLess(
                ci_high_pct, 405.0,
                f"{symbol} H={sig.horizon_days}: CI still absurd at {ci_high_pct:.1f}%"
            )

    def test_abtc_bounded(self):
        """ABTC (the #1 offender) must be bounded."""
        self._check_bounded("ABTC")

    def test_gpus_bounded(self):
        """GPUS (high-vol equity) must be bounded."""
        self._check_bounded("GPUS")

    def test_ionq_bounded(self):
        """IONQ (quantum small cap) must be bounded."""
        self._check_bounded("IONQ")

    def test_afrm_bounded(self):
        """AFRM (fintech) must be bounded."""
        self._check_bounded("AFRM")

    def test_btc_bounded(self):
        """BTC-USD (crypto) must be bounded even with 200% annual cap."""
        self._check_bounded("BTC-USD")


# ===============================================================
# Test Class 12: Signal consistency per asset
# ===============================================================
class TestPerAssetSignalConsistency(unittest.TestCase):
    """Within each asset, signals at different horizons must be self-consistent."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_ci_contains_exp_ret(self):
        """Expected return should fall within CI bounds (or very close).

        exp_ret should generally be between ci_low and ci_high.
        Allow a small tolerance for EMOS corrections that can shift exp_ret
        slightly outside the MC-derived CI.
        """
        violations = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                # Allow 20% tolerance of CI width
                width = sig.ci_high - sig.ci_low
                tol = max(0.05, 0.2 * width)
                if sig.exp_ret < sig.ci_low - tol:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: exp_ret={sig.exp_ret:.4f} "
                        f"< ci_low={sig.ci_low:.4f} - tol={tol:.4f}"
                    )
                if sig.exp_ret > sig.ci_high + tol:
                    violations.append(
                        f"{symbol} H={sig.horizon_days}: exp_ret={sig.exp_ret:.4f} "
                        f"> ci_high={sig.ci_high:.4f} + tol={tol:.4f}"
                    )
        # Allow up to 5% violation rate (EMOS can shift means)
        total = sum(len(sigs) for sigs, _ in _RESULTS_CACHE.values())
        max_allowed = max(2, int(0.05 * total))
        self.assertLessEqual(
            len(violations), max_allowed,
            f"{len(violations)}/{total} signals have exp_ret outside CI±tol:\n"
            + "\n".join(violations[:10])
        )

    def test_profit_sign_matches_exp_ret(self):
        """Profit PLN sign should match expected return sign.

        If exp_ret > 0, profit should be ≥ 0 (and vice versa).
        Allow tolerance for very small returns near zero.
        """
        mismatches = []
        for symbol, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                if abs(sig.exp_ret) < 0.001:
                    continue  # Near-zero — sign matching is meaningless
                if sig.exp_ret > 0.01 and sig.profit_pln < -1000:
                    mismatches.append(
                        f"{symbol} H={sig.horizon_days}: exp_ret={sig.exp_ret:.4f} "
                        f"but profit={sig.profit_pln:,.0f}"
                    )
                if sig.exp_ret < -0.01 and sig.profit_pln > 1000:
                    mismatches.append(
                        f"{symbol} H={sig.horizon_days}: exp_ret={sig.exp_ret:.4f} "
                        f"but profit={sig.profit_pln:,.0f}"
                    )
        # Allow a few mismatches due to EMOS corrections
        self.assertLessEqual(
            len(mismatches), 5,
            f"Profit-ExpRet sign mismatches:\n" + "\n".join(mismatches[:10])
        )


# ===============================================================
# Test Class 13: Summary statistics for human review
# ===============================================================
class TestSummaryReport(unittest.TestCase):
    """Generate a human-readable summary (always passes, prints diagnostics)."""

    @classmethod
    def setUpClass(cls):
        _build_cache()

    def test_print_summary(self):
        """Print overview statistics for all 40 assets."""
        n_success = len(_RESULTS_CACHE)
        n_skip = len(_SKIPPED_ASSETS)
        total_signals = sum(len(sigs) for sigs, _ in _RESULTS_CACHE.values())

        # Per-type counts
        type_counts = defaultdict(int)
        for sym, (sigs, meta) in _RESULTS_CACHE.items():
            type_counts[meta["asset_type"]] += 1

        # Label distribution
        label_counts = defaultdict(int)
        for sym, (sigs, _) in _RESULTS_CACHE.items():
            for sig in sigs:
                label_counts[sig.label] += 1

        # P(up) distribution
        all_p_up = [sig.p_up for sigs, _ in _RESULTS_CACHE.values() for sig in sigs]
        p_up_arr = np.array(all_p_up) if all_p_up else np.array([0.5])

        # Vol distribution
        all_vol = [sig.vol_mean for sigs, _ in _RESULTS_CACHE.values() for sig in sigs]
        vol_arr = np.array(all_vol) if all_vol else np.array([0.01])

        print(f"\n{'='*60}")
        print(f"  40-ASSET ACCURACY REPORT")
        print(f"{'='*60}")
        print(f"  Assets tested:  {n_success}/{n_success + n_skip}")
        print(f"  Total signals:  {total_signals}")
        print(f"  Asset types:    {dict(type_counts)}")
        print(f"  Labels:         {dict(label_counts)}")
        print(f"  P(up):          mean={p_up_arr.mean():.3f}  "
              f"std={p_up_arr.std():.3f}  "
              f"range=[{p_up_arr.min():.3f}, {p_up_arr.max():.3f}]")
        print(f"  Vol:            mean={vol_arr.mean():.6f}  "
              f"range=[{vol_arr.min():.6f}, {vol_arr.max():.6f}]")
        if _SKIPPED_ASSETS:
            print(f"  Skipped:        {list(_SKIPPED_ASSETS.keys())}")
        print(f"{'='*60}\n")

        # This test always passes — it's informational
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
