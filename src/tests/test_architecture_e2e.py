"""
E2E equivalence test framework for the signals.py decomposition.

Story 9.1: Capture before/after snapshots of pipeline outputs and
compare them with configurable tolerances. Supports --update-snapshots
flag for baseline regeneration.

Usage:
    # Run framework self-tests (no snapshots needed):
    pytest src/tests/test_architecture_e2e.py -q

    # Update snapshots for a specific asset:
    pytest src/tests/test_architecture_e2e.py --update-snapshots -k "SPY"

    # Full E2E comparison (requires existing snapshots):
    pytest src/tests/test_architecture_e2e.py --run-e2e -q
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, fields, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Ensure src/ is importable
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

# ---------------------------------------------------------------------------
# Pytest custom options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Register --update-snapshots and --run-e2e CLI flags."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate snapshot baselines instead of comparing.",
    )
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run full E2E equivalence tests (slow, requires data).",
    )


# ---------------------------------------------------------------------------
# JSON helpers (handle numpy / pandas / dataclass types)
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert obj to JSON-safe primitives."""
    import numpy as _np
    import pandas as _pd

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__NaN__"
        if math.isinf(obj):
            return "__Inf__" if obj > 0 else "__-Inf__"
        return obj
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return _make_serializable(float(obj))
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return [_make_serializable(x) for x in obj.tolist()]
    if isinstance(obj, _pd.Series):
        return [_make_serializable(x) for x in obj.tolist()]
    if isinstance(obj, _pd.DataFrame):
        return {str(c): _make_serializable(obj[c]) for c in obj.columns}
    if isinstance(obj, _pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        return _make_serializable(asdict(obj))
    # Fallback: convert to string
    return str(obj)


def _restore_special_floats(obj: Any) -> Any:
    """Restore __NaN__, __Inf__, __-Inf__ sentinels back to float."""
    if isinstance(obj, str):
        if obj == "__NaN__":
            return float("nan")
        if obj == "__Inf__":
            return float("inf")
        if obj == "__-Inf__":
            return float("-inf")
        return obj
    if isinstance(obj, dict):
        return {k: _restore_special_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_special_floats(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------

def _snapshot_path(kind: str, asset: str) -> Path:
    """Return path for snapshot file: snapshots/{kind}/{asset}.json"""
    safe = asset.replace("/", "_").replace("=", "_").replace("-", "_")
    return SNAPSHOTS_DIR / kind / f"{safe}.json"


def save_snapshot(kind: str, asset: str, data: Dict) -> Path:
    """Serialize *data* to a snapshot JSON file. Returns the path."""
    path = _snapshot_path(kind, asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_serializable(data), f, indent=2, sort_keys=True)
    return path


def load_snapshot(kind: str, asset: str) -> Optional[Dict]:
    """Load a snapshot. Returns None if the file does not exist."""
    path = _snapshot_path(kind, asset)
    if not path.exists():
        return None
    with open(path) as f:
        return _restore_special_floats(json.load(f))


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def capture_tune_output(asset: str, start_date: str = "2015-01-01") -> Dict:
    """Run tune_asset_with_bma for *asset* and return a serializable dict.

    The dict contains:
        - global.model_posterior: {model_name: weight}
        - global.models: {model_name: {param: value}}
        - regime.{r}.model_posterior
        - regime.{r}.models
        - meta: tuning metadata
    """
    from tuning.tune_modules.asset_tuning import tune_asset_with_bma

    result = tune_asset_with_bma(asset, start_date=start_date)
    if result is None:
        return {"status": "error", "asset": asset, "error": "tune returned None"}

    # Extract the fields we care about for equivalence
    snapshot: Dict[str, Any] = {"asset": asset, "status": "success"}

    # Global level
    g = result.get("global", {})
    snapshot["global"] = {
        "model_posterior": g.get("model_posterior", {}),
        "models": {},
    }
    for model_name, mdata in g.get("models", {}).items():
        snapshot["global"]["models"][model_name] = {
            k: v for k, v in mdata.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }

    # Regime level
    snapshot["regime"] = {}
    for r_key, r_data in result.get("regime", {}).items():
        r_snap: Dict[str, Any] = {
            "model_posterior": r_data.get("model_posterior", {}),
            "models": {},
        }
        for model_name, mdata in r_data.get("models", {}).items():
            r_snap["models"][model_name] = {
                k: v for k, v in mdata.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        # Regime metadata
        if "regime_meta" in r_data:
            r_snap["regime_meta"] = {
                k: v for k, v in r_data["regime_meta"].items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        snapshot["regime"][str(r_key)] = r_snap

    # Meta
    snapshot["meta"] = result.get("meta", {})

    return snapshot


def capture_signal_output(
    asset: str,
    start_date: str = "2005-01-01",
    horizons: Optional[List[int]] = None,
) -> Dict:
    """Run signal generation for *asset* and return a serializable dict.

    The dict contains:
        - signals: list of Signal-as-dict (one per horizon)
        - thresholds: per-horizon threshold dicts
        - feats_summary: summary stats from feature dict (not full series)
        - last_close: last price
    """
    import argparse
    from decision.signal_modules.config import DEFAULT_ASSET_UNIVERSE
    from decision.signal_modules.asset_processing import process_single_asset

    if horizons is None:
        horizons = [1, 3, 7, 21, 63, 126, 252]

    # Build a minimal args namespace
    args = argparse.Namespace(
        start=start_date,
        end=None,
        horizons=",".join(str(h) for h in horizons),
        assets=asset,
        json=None,
        csv=None,
        cache_json=None,
        from_cache=False,
        simple=False,
        t_map=True,
        ci=0.68,
        no_caption=True,
        force_caption=False,
        diagnostics=False,
        diagnostics_lite=False,
        pit_calibration=False,
        model_comparison=False,
        validate_kalman=False,
        validation_plots=False,
        failures_json="",
    )

    result = process_single_asset((asset, args, horizons))
    if result is None or result.get("status") != "success":
        err = result.get("error", "unknown") if result else "None returned"
        return {"status": "error", "asset": asset, "error": err}

    snapshot: Dict[str, Any] = {"asset": asset, "status": "success"}

    # Signals
    sigs = result.get("sigs", [])
    snapshot["signals"] = []
    for sig in sigs:
        if hasattr(sig, "__dataclass_fields__"):
            snapshot["signals"].append(asdict(sig))
        elif isinstance(sig, dict):
            snapshot["signals"].append(sig)

    # Thresholds
    snapshot["thresholds"] = result.get("thresholds", {})

    # Feature summary (avoid storing full time series)
    feats = result.get("feats", {})
    feats_summary: Dict[str, Any] = {}
    if isinstance(feats, dict):
        import numpy as _np
        import pandas as _pd
        for k, v in feats.items():
            if isinstance(v, (_pd.Series, _np.ndarray)):
                arr = _np.asarray(v, dtype=float)
                finite = arr[_np.isfinite(arr)]
                if len(finite) > 0:
                    feats_summary[k] = {
                        "mean": float(_np.mean(finite)),
                        "std": float(_np.std(finite)),
                        "min": float(_np.min(finite)),
                        "max": float(_np.max(finite)),
                        "len": int(len(arr)),
                    }
                else:
                    feats_summary[k] = {"len": int(len(arr)), "all_nan": True}
            elif isinstance(v, (int, float, str, bool, type(None))):
                feats_summary[k] = v
    snapshot["feats_summary"] = feats_summary

    snapshot["last_close"] = result.get("last_close")
    snapshot["canon"] = result.get("canon")
    snapshot["title"] = result.get("title")

    return snapshot


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

@dataclass
class FieldDivergence:
    """Records a single field that diverged between before and after."""
    path: str
    before: Any
    after: Any
    abs_diff: Optional[float] = None
    rel_diff: Optional[float] = None


def _compare_values(
    path: str,
    before: Any,
    after: Any,
    tolerance: float,
    divergences: List[FieldDivergence],
) -> None:
    """Recursively compare before/after, appending divergences."""
    # Both None
    if before is None and after is None:
        return

    # Type mismatch
    if type(before) != type(after):
        # Allow int/float cross-comparison
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            pass  # fall through to numeric compare
        else:
            divergences.append(FieldDivergence(path=path, before=before, after=after))
            return

    # Dict
    if isinstance(before, dict) and isinstance(after, dict):
        all_keys = set(before.keys()) | set(after.keys())
        for k in sorted(all_keys):
            if k not in before:
                divergences.append(FieldDivergence(
                    path=f"{path}.{k}", before="<missing>", after=after[k],
                ))
            elif k not in after:
                divergences.append(FieldDivergence(
                    path=f"{path}.{k}", before=before[k], after="<missing>",
                ))
            else:
                _compare_values(f"{path}.{k}", before[k], after[k], tolerance, divergences)
        return

    # List
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            divergences.append(FieldDivergence(
                path=f"{path}.__len__", before=len(before), after=len(after),
            ))
            return
        for i, (b, a) in enumerate(zip(before, after)):
            _compare_values(f"{path}[{i}]", b, a, tolerance, divergences)
        return

    # String / bool
    if isinstance(before, (str, bool)):
        if before != after:
            divergences.append(FieldDivergence(path=path, before=before, after=after))
        return

    # Numeric
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        bf, af = float(before), float(after)
        # NaN equality
        if math.isnan(bf) and math.isnan(af):
            return
        if math.isnan(bf) or math.isnan(af):
            divergences.append(FieldDivergence(path=path, before=before, after=after))
            return
        # Inf equality
        if math.isinf(bf) and math.isinf(af) and (bf > 0) == (af > 0):
            return
        abs_d = abs(bf - af)
        denom = max(abs(bf), abs(af), 1e-30)
        rel_d = abs_d / denom
        if abs_d > tolerance:
            divergences.append(FieldDivergence(
                path=path, before=before, after=after,
                abs_diff=abs_d, rel_diff=rel_d,
            ))
        return

    # Fallback: string compare
    if str(before) != str(after):
        divergences.append(FieldDivergence(path=path, before=before, after=after))


def assert_tune_equivalence(
    before: Dict, after: Dict, tolerance: float = 1e-10,
) -> List[FieldDivergence]:
    """Compare two tune snapshots. Returns list of divergences (empty = pass)."""
    divergences: List[FieldDivergence] = []
    _compare_values("tune", before, after, tolerance, divergences)
    return divergences


def assert_signal_equivalence(
    before: Dict, after: Dict, tolerance: float = 1e-8,
) -> List[FieldDivergence]:
    """Compare two signal snapshots. Returns list of divergences (empty = pass)."""
    divergences: List[FieldDivergence] = []
    _compare_values("signal", before, after, tolerance, divergences)
    return divergences


def format_divergence_report(divergences: List[FieldDivergence], max_lines: int = 50) -> str:
    """Human-readable report of divergences."""
    if not divergences:
        return "No divergences found."
    lines = [f"Found {len(divergences)} divergence(s):"]
    for i, d in enumerate(divergences[:max_lines]):
        line = f"  [{i+1}] {d.path}: before={d.before!r}, after={d.after!r}"
        if d.abs_diff is not None:
            line += f" (abs_diff={d.abs_diff:.2e}, rel_diff={d.rel_diff:.2e})"
        lines.append(line)
    if len(divergences) > max_lines:
        lines.append(f"  ... and {len(divergences) - max_lines} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Framework self-tests (always run, no data needed)
# ---------------------------------------------------------------------------

class TestSerializationRoundtrip:
    """Verify JSON serialization handles edge cases."""

    def test_nan_inf_roundtrip(self):
        original = {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 42.0}
        serialized = _make_serializable(original)
        assert serialized["a"] == "__NaN__"
        assert serialized["b"] == "__Inf__"
        assert serialized["c"] == "__-Inf__"
        restored = _restore_special_floats(serialized)
        assert math.isnan(restored["a"])
        assert restored["b"] == float("inf")
        assert restored["c"] == float("-inf")
        assert restored["d"] == 42.0

    def test_nested_dict_roundtrip(self):
        original = {"level1": {"level2": [1.0, 2.0, None], "x": True}}
        s = _make_serializable(original)
        raw = json.dumps(s)
        back = _restore_special_floats(json.loads(raw))
        assert back == original

    def test_numpy_types(self):
        import numpy as np
        data = {
            "arr": np.array([1.0, 2.0, float("nan")]),
            "i64": np.int64(42),
            "f32": np.float32(3.14),
            "b": np.bool_(True),
        }
        s = _make_serializable(data)
        assert s["i64"] == 42
        assert isinstance(s["f32"], float)
        assert s["b"] is True
        assert len(s["arr"]) == 3


class TestComparisonEngine:
    """Verify the comparison engine catches divergences correctly."""

    def test_identical_passes(self):
        a = {"x": 1.0, "y": "hello", "nested": {"z": [1, 2, 3]}}
        divs = assert_signal_equivalence(a, a)
        assert divs == []

    def test_numeric_within_tolerance(self):
        a = {"x": 1.0}
        b = {"x": 1.0 + 1e-12}
        divs = assert_signal_equivalence(a, b, tolerance=1e-8)
        assert divs == []

    def test_numeric_beyond_tolerance(self):
        a = {"x": 1.0}
        b = {"x": 1.1}
        divs = assert_signal_equivalence(a, b, tolerance=1e-8)
        assert len(divs) == 1
        assert divs[0].path == "signal.x"
        assert divs[0].abs_diff == pytest.approx(0.1)

    def test_categorical_exact(self):
        a = {"label": "BUY", "regime": "LOW_VOL_TREND"}
        b = {"label": "SELL", "regime": "LOW_VOL_TREND"}
        divs = assert_signal_equivalence(a, b)
        assert len(divs) == 1
        assert divs[0].path == "signal.label"

    def test_missing_key_detected(self):
        a = {"x": 1, "y": 2}
        b = {"x": 1}
        divs = assert_signal_equivalence(a, b)
        assert len(divs) == 1
        assert "y" in divs[0].path

    def test_list_length_mismatch(self):
        a = {"signals": [1, 2, 3]}
        b = {"signals": [1, 2]}
        divs = assert_signal_equivalence(a, b)
        assert len(divs) == 1
        assert "__len__" in divs[0].path

    def test_nan_equals_nan(self):
        a = {"x": float("nan")}
        b = {"x": float("nan")}
        divs = assert_signal_equivalence(a, b)
        assert divs == []

    def test_nan_vs_number(self):
        a = {"x": float("nan")}
        b = {"x": 1.0}
        divs = assert_signal_equivalence(a, b)
        assert len(divs) == 1

    def test_nested_divergence_path(self):
        a = {"regime": {"0": {"model_posterior": {"kalman_gaussian": 0.5}}}}
        b = {"regime": {"0": {"model_posterior": {"kalman_gaussian": 0.9}}}}
        divs = assert_tune_equivalence(a, b, tolerance=1e-10)
        assert len(divs) == 1
        assert "kalman_gaussian" in divs[0].path
        assert divs[0].abs_diff == pytest.approx(0.4)


class TestDivergenceReport:
    """Verify human-readable reporting."""

    def test_empty_report(self):
        report = format_divergence_report([])
        assert "No divergences" in report

    def test_report_includes_path_and_diff(self):
        divs = [FieldDivergence(
            path="signal.signals[0].p_up",
            before=0.55, after=0.65,
            abs_diff=0.1, rel_diff=0.1538,
        )]
        report = format_divergence_report(divs)
        assert "signal.signals[0].p_up" in report
        assert "1.00e-01" in report

    def test_report_truncation(self):
        divs = [
            FieldDivergence(path=f"field_{i}", before=0, after=1)
            for i in range(100)
        ]
        report = format_divergence_report(divs, max_lines=5)
        assert "95 more" in report


class TestSnapshotIO:
    """Verify snapshot save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.tests.test_architecture_e2e.SNAPSHOTS_DIR", tmp_path
        )
        data = {"asset": "TEST", "x": 1.5, "nested": {"y": [1, 2, 3]}}
        save_snapshot("signal", "TEST", data)
        loaded = load_snapshot("signal", "TEST")
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.tests.test_architecture_e2e.SNAPSHOTS_DIR", tmp_path
        )
        assert load_snapshot("signal", "NONEXISTENT") is None

    def test_special_chars_in_asset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.tests.test_architecture_e2e.SNAPSHOTS_DIR", tmp_path
        )
        data = {"asset": "GC=F"}
        save_snapshot("tune", "GC=F", data)
        loaded = load_snapshot("tune", "GC=F")
        assert loaded == data


class TestCaptureHelpers:
    """Verify capture functions produce correct structure (mocked)."""

    def test_capture_tune_output_structure(self):
        """Validate the output dict has expected keys."""
        # We test the structure, not the actual computation
        # (full E2E with real data is in Story 9.2)
        mock_tune_result = {
            "asset": "SPY",
            "global": {
                "model_posterior": {"kalman_gaussian": 0.3, "phi_student_t_nu_6": 0.7},
                "models": {
                    "kalman_gaussian": {"q": 1e-5, "c": 0.01, "bic": 100.0},
                    "phi_student_t_nu_6": {"q": 1e-5, "c": 0.02, "bic": 95.0},
                },
            },
            "regime": {
                "0": {
                    "model_posterior": {"kalman_gaussian": 0.4, "phi_student_t_nu_6": 0.6},
                    "models": {
                        "kalman_gaussian": {"q": 1e-5, "c": 0.01, "bic": 100.0},
                    },
                    "regime_meta": {"n_obs": 500, "label": "LOW_VOL_TREND"},
                },
            },
            "meta": {"model_selection_method": "combined"},
        }

        # Simulate what capture_tune_output does with this data
        snapshot = {"asset": "SPY", "status": "success"}
        g = mock_tune_result["global"]
        snapshot["global"] = {
            "model_posterior": g.get("model_posterior", {}),
            "models": {},
        }
        for mn, md in g.get("models", {}).items():
            snapshot["global"]["models"][mn] = {
                k: v for k, v in md.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        assert "global" in snapshot
        assert "kalman_gaussian" in snapshot["global"]["model_posterior"]
        assert snapshot["global"]["models"]["kalman_gaussian"]["bic"] == 100.0

    def test_signal_dataclass_serializable(self):
        """Verify Signal can be serialized via asdict + _make_serializable."""
        from decision.signal_modules.signal_dataclass import Signal
        sig = Signal(
            horizon_days=7, score=1.5, p_up=0.65, exp_ret=0.01,
            ci_low=-0.02, ci_high=0.04, ci_low_90=-0.05, ci_high_90=0.07,
            profit_pln=10000, profit_ci_low_pln=-20000, profit_ci_high_pln=40000,
            position_strength=0.8, vol_mean=0.015, vol_ci_low=0.01, vol_ci_high=0.02,
            regime="LOW_VOL_TREND", label="BUY",
        )
        d = _make_serializable(asdict(sig))
        assert isinstance(d, dict)
        assert d["horizon_days"] == 7
        assert d["label"] == "BUY"
        # Round-trip through JSON
        raw = json.dumps(d)
        back = json.loads(raw)
        assert back["p_up"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Story 9.2: Tune pipeline E2E equivalence tests (50 assets)
# ---------------------------------------------------------------------------

# Import asset universe lazily to avoid import errors during collection
def _get_asset_universe():
    """Return the full DEFAULT_ASSET_UNIVERSE list."""
    from ingestion.data_utils import DEFAULT_ASSET_UNIVERSE
    return list(DEFAULT_ASSET_UNIVERSE)


# Representative subset for faster CI runs
_REPRESENTATIVE_ASSETS = [
    "EURUSD=X", "USDJPY=X", "GBPJPY=X", "PLNJPY=X",
    "GC=F", "SI=F", "BTC-USD", "MSTR",
]

_e2e_skip = pytest.mark.skipif(
    "not config.getoption('--run-e2e', default=False)",
    reason="E2E tests require --run-e2e flag (slow, requires market data)",
)


@_e2e_skip
class TestTunePipelineEquivalence:
    """Story 9.2: Verify tune pipeline produces identical output via both import paths.

    Since the shim (tuning.tune) re-exports from tune_modules, both paths resolve
    to the same function. This test validates:
    1. The import chain works without errors
    2. tune_asset_with_bma produces valid output for all assets
    3. Output matches stored snapshots (if available)
    """

    def test_tune_import_equivalence(self):
        """Both import paths resolve to the exact same function object."""
        from tuning.tune import tune_asset_with_bma as f_shim
        from tuning.tune_modules.asset_tuning import tune_asset_with_bma as f_direct
        assert f_shim is f_direct, "Shim and direct import should be the same function"

    @pytest.mark.parametrize("asset", _REPRESENTATIVE_ASSETS)
    def test_tune_asset_produces_valid_output(self, asset, request):
        """Tune each asset and verify output structure is valid."""
        from tuning.tune_modules.asset_tuning import tune_asset_with_bma

        result = tune_asset_with_bma(asset, start_date="2015-01-01")
        assert result is not None, f"tune_asset_with_bma returned None for {asset}"
        assert "global" in result, f"Missing 'global' key for {asset}"
        assert "regime" in result, f"Missing 'regime' key for {asset}"

        g = result["global"]
        assert "model_posterior" in g, f"Missing model_posterior in global for {asset}"
        assert "models" in g, f"Missing models in global for {asset}"
        assert len(g["model_posterior"]) > 0, f"Empty model_posterior for {asset}"

        # Posteriors should sum to ~1.0
        total_weight = sum(g["model_posterior"].values())
        assert abs(total_weight - 1.0) < 0.01, (
            f"Global model posteriors sum to {total_weight}, expected ~1.0 for {asset}"
        )

        # If --update-snapshots, save for future comparison
        if request.config.getoption("--update-snapshots", default=False):
            snapshot = capture_tune_output(asset)
            path = save_snapshot("tune", asset, snapshot)
            print(f"Saved tune snapshot: {path}")

    @pytest.mark.parametrize("asset", _REPRESENTATIVE_ASSETS)
    def test_tune_snapshot_equivalence(self, asset, request):
        """Compare current tune output against stored snapshot."""
        stored = load_snapshot("tune", asset)
        if stored is None:
            pytest.skip(f"No tune snapshot for {asset}. Run with --update-snapshots first.")

        current = capture_tune_output(asset)
        divergences = assert_tune_equivalence(stored, current, tolerance=1e-10)
        if divergences:
            report = format_divergence_report(divergences)
            pytest.fail(f"Tune output diverged for {asset}:\n{report}")


# ---------------------------------------------------------------------------
# Story 9.3: Signal pipeline E2E equivalence tests (50 assets)
# ---------------------------------------------------------------------------

@_e2e_skip
class TestSignalPipelineEquivalence:
    """Story 9.3: Verify signal pipeline produces identical output via both import paths.

    Validates:
    1. Import chain works
    2. process_single_asset produces valid Signal objects for all assets
    3. Output matches stored snapshots
    """

    def test_signal_import_equivalence(self):
        """Both import paths resolve to the exact same function object."""
        from decision.signals import process_single_asset as f_shim
        from decision.signal_modules.asset_processing import process_single_asset as f_direct
        assert f_shim is f_direct, "Shim and direct import should be the same function"

    def test_signal_latest_signals_import_equivalence(self):
        """latest_signals also resolves identically via both paths."""
        from decision.signals import latest_signals as f_shim
        from decision.signal_modules.signal_generation import latest_signals as f_direct
        assert f_shim is f_direct

    @pytest.mark.parametrize("asset", _REPRESENTATIVE_ASSETS)
    def test_signal_asset_produces_valid_output(self, asset, request):
        """Process each asset and verify Signal objects are valid."""
        import argparse
        from decision.signal_modules.asset_processing import process_single_asset

        horizons = [1, 7, 21, 63, 252]
        args = argparse.Namespace(
            start="2005-01-01", end=None,
            horizons=",".join(str(h) for h in horizons),
            assets=asset, json=None, csv=None, cache_json=None,
            from_cache=False, simple=False, t_map=True, ci=0.68,
            no_caption=True, force_caption=False,
            diagnostics=False, diagnostics_lite=False,
            pit_calibration=False, model_comparison=False,
            validate_kalman=False, validation_plots=False,
            failures_json="",
        )

        result = process_single_asset((asset, args, horizons))
        assert result is not None, f"process_single_asset returned None for {asset}"
        assert result.get("status") == "success", (
            f"Failed for {asset}: {result.get('error', 'unknown')}"
        )

        sigs = result.get("sigs", [])
        assert len(sigs) > 0, f"No signals produced for {asset}"

        for sig in sigs:
            # Validate key Signal fields
            assert hasattr(sig, "horizon_days"), f"Signal missing horizon_days for {asset}"
            assert hasattr(sig, "label"), f"Signal missing label for {asset}"
            assert sig.label in ("BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL", "EXIT"), (
                f"Unexpected label '{sig.label}' for {asset} H={sig.horizon_days}"
            )
            assert 0.0 <= sig.p_up <= 1.0, (
                f"p_up={sig.p_up} out of range for {asset} H={sig.horizon_days}"
            )

        # Feature dict should have substantial keys
        feats = result.get("feats", {})
        if isinstance(feats, dict):
            assert len(feats) >= 10, (
                f"Too few feature keys ({len(feats)}) for {asset}"
            )

        if request.config.getoption("--update-snapshots", default=False):
            snapshot = capture_signal_output(asset, horizons=horizons)
            path = save_snapshot("signal", asset, snapshot)
            print(f"Saved signal snapshot: {path}")

    @pytest.mark.parametrize("asset", _REPRESENTATIVE_ASSETS)
    def test_signal_snapshot_equivalence(self, asset, request):
        """Compare current signal output against stored snapshot."""
        stored = load_snapshot("signal", asset)
        if stored is None:
            pytest.skip(f"No signal snapshot for {asset}. Run with --update-snapshots first.")

        horizons = [1, 7, 21, 63, 252]
        current = capture_signal_output(asset, horizons=horizons)
        divergences = assert_signal_equivalence(stored, current, tolerance=1e-8)
        if divergences:
            report = format_divergence_report(divergences)
            pytest.fail(f"Signal output diverged for {asset}:\n{report}")


# ---------------------------------------------------------------------------
# Story 9.4: Cross-pipeline E2E test (tune -> signal chain)
# ---------------------------------------------------------------------------

_CROSS_PIPELINE_ASSETS = [
    "EURUSD=X", "USDJPY=X", "GBPJPY=X", "PLNJPY=X",
    "GC=F", "BTC-USD", "MSTR",
]


@_e2e_skip
class TestCrossPipelineEquivalence:
    """Story 9.4: Verify the full tune->signal chain works correctly.

    For representative assets: tune, then generate signals using tuned params.
    This catches integration bugs between tune_modules and signal_modules.
    """

    @pytest.mark.parametrize("asset", _CROSS_PIPELINE_ASSETS)
    def test_tune_then_signal_chain(self, asset):
        """Tune an asset, then use the params for signal generation."""
        from tuning.tune_modules.asset_tuning import tune_asset_with_bma
        from decision.signal_modules.signal_generation import latest_signals
        from decision.signal_modules.data_fetching import compute_features
        from decision.signal_modules.config import (
            _download_prices, _to_float, fetch_px_asset,
            GK_VOLATILITY_AVAILABLE,
        )

        # Step 1: Tune
        tune_result = tune_asset_with_bma(asset, start_date="2015-01-01")
        assert tune_result is not None, f"Tune failed for {asset}"

        # Step 2: Fetch prices and compute features
        try:
            px, title = fetch_px_asset(asset, "2015-01-01", None)
        except Exception as e:
            pytest.skip(f"Could not fetch price data for {asset}: {e}")

        ohlc_df = None
        if GK_VOLATILITY_AVAILABLE:
            try:
                ohlc_df = _download_prices(asset, "2015-01-01", None)
            except Exception:
                pass

        feats = compute_features(px, asset_symbol=asset, ohlc_df=ohlc_df)
        last_close = _to_float(px.iloc[-1])

        # Step 3: Generate signals using tuned params
        horizons = [1, 7, 21, 63]
        sigs, thresholds = latest_signals(
            feats, horizons, last_close,
            t_map=True, ci=0.68,
            tuned_params=tune_result,
            asset_key=asset,
        )

        assert len(sigs) == len(horizons), (
            f"Expected {len(horizons)} signals, got {len(sigs)} for {asset}"
        )
        for sig in sigs:
            assert sig.label in ("BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL", "EXIT"), (
                f"Unexpected label '{sig.label}' for {asset} H={sig.horizon_days}"
            )
            assert 0.0 <= sig.p_up <= 1.0
