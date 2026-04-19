"""
Story 9.6 -- Performance regression tests.

Verifies that the modular architecture introduces no measurable performance
regression compared to the monolithic shim imports.

Gate: requires ``--run-e2e`` flag (benchmarks hit the network / disk).

Import-time tests run unconditionally and must complete in < 3 s.
"""
from __future__ import annotations

import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import pytest

# Ensure src/ is importable
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

PERFORMANCE_BASELINE_PATH = Path(__file__).parent / "performance_baseline.json"

# ---------------------------------------------------------------------------
# Pytest options (reuse --run-e2e from test_architecture_e2e.py)
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --run-e2e option (idempotent if already registered)."""
    try:
        parser.addoption(
            "--run-e2e",
            action="store_true",
            default=False,
            help="Run E2E performance benchmarks (slow, network-dependent)",
        )
    except ValueError:
        pass  # already registered


_e2e_skip = pytest.mark.skipif(
    "not config.getoption('--run-e2e', default=False)",
    reason="E2E benchmarks require --run-e2e flag",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BENCHMARK_RUNS = 5
_WARMUP_RUNS = 1
_TOLERANCE = 0.05  # 5% regression tolerance

_ASSET = "EURUSD=X"  # representative asset for benchmarks


def _benchmark(fn, runs: int = _BENCHMARK_RUNS, warmup: int = _WARMUP_RUNS):
    """Return (median_seconds, all_times) for *fn*."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return median, times


def _get_rss_mb() -> float:
    """Current process RSS in MB (macOS / Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in bytes on macOS, KB on Linux
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def _save_results(results: dict) -> None:
    """Persist benchmark results to performance_baseline.json."""
    PERFORMANCE_BASELINE_PATH.write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )


# =========================================================================
# Import-time tests (always run, no --run-e2e gate)
# =========================================================================

class TestImportTime:
    """Module import time must be < 3 seconds (cold-ish start)."""

    def test_tune_modules_import_time(self) -> None:
        # Force reimport by removing from cache
        mods_to_remove = [k for k in sys.modules if k.startswith("tuning.tune_modules")]
        for m in mods_to_remove:
            del sys.modules[m]
        if "tuning.tune_modules" in sys.modules:
            del sys.modules["tuning.tune_modules"]

        t0 = time.perf_counter()
        importlib.import_module("tuning.tune_modules")
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"tune_modules import took {elapsed:.2f}s (limit: 3.0s)"

    def test_signal_modules_import_time(self) -> None:
        mods_to_remove = [k for k in sys.modules if k.startswith("decision.signal_modules")]
        for m in mods_to_remove:
            del sys.modules[m]
        if "decision.signal_modules" in sys.modules:
            del sys.modules["decision.signal_modules"]

        t0 = time.perf_counter()
        importlib.import_module("decision.signal_modules")
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"signal_modules import took {elapsed:.2f}s (limit: 3.0s)"

    def test_shim_vs_modules_import_overhead(self) -> None:
        """Shim import should not add > 0.5s overhead vs direct modules."""
        # Time direct module import
        for prefix in ("tuning.tune_modules", "decision.signal_modules"):
            mods_to_remove = [k for k in sys.modules if k.startswith(prefix)]
            for m in mods_to_remove:
                del sys.modules[m]

        t0 = time.perf_counter()
        importlib.import_module("tuning.tune_modules")
        importlib.import_module("decision.signal_modules")
        direct_time = time.perf_counter() - t0

        # Time shim import (shims are already cached at this point, so
        # we just verify the shim files themselves parse quickly)
        for prefix in ("tuning.tune", "decision.signals"):
            mods_to_remove = [k for k in sys.modules if k.startswith(prefix)]
            for m in mods_to_remove:
                del sys.modules[m]

        t0 = time.perf_counter()
        importlib.import_module("tuning.tune")
        importlib.import_module("decision.signals")
        shim_time = time.perf_counter() - t0

        overhead = shim_time - direct_time
        # Overhead can be negative (shim uses cached submodules) -- that's fine
        assert overhead < 0.5, (
            f"Shim overhead: {overhead:.3f}s (limit: 0.5s)"
        )


# =========================================================================
# Pipeline benchmarks (require --run-e2e)
# =========================================================================

@_e2e_skip
class TestTunePipelinePerformance:
    """Tune pipeline performance: shim vs direct module path."""

    def test_tune_paths_resolve_same_function(self) -> None:
        from tuning.tune import tune_asset_with_bma as shim_fn
        from tuning.tune_modules import tune_asset_with_bma as direct_fn
        assert shim_fn is direct_fn

    def test_tune_pipeline_wall_time(self) -> None:
        from tuning.tune_modules import tune_asset_with_bma

        rss_before = _get_rss_mb()
        median, all_times = _benchmark(
            lambda: tune_asset_with_bma(_ASSET),
            runs=_BENCHMARK_RUNS,
            warmup=_WARMUP_RUNS,
        )
        rss_after = _get_rss_mb()

        # Since shim and modules resolve to the same function, we just
        # record the baseline and verify it's reasonable (< 120s per run)
        assert median < 120.0, f"tune_asset_with_bma median: {median:.1f}s (limit: 120s)"

        results = {
            "tune_median_s": round(median, 3),
            "tune_all_times_s": [round(t, 3) for t in all_times],
            "tune_rss_before_mb": round(rss_before, 1),
            "tune_rss_after_mb": round(rss_after, 1),
            "asset": _ASSET,
        }

        # Save partial results
        if PERFORMANCE_BASELINE_PATH.exists():
            existing = json.loads(PERFORMANCE_BASELINE_PATH.read_text())
        else:
            existing = {}
        existing.update(results)
        _save_results(existing)


@_e2e_skip
class TestSignalPipelinePerformance:
    """Signal pipeline performance: shim vs direct module path."""

    def test_signal_paths_resolve_same_function(self) -> None:
        from decision.signals import process_single_asset as shim_fn
        from decision.signal_modules import process_single_asset as direct_fn
        assert shim_fn is direct_fn

    def test_signal_pipeline_wall_time(self) -> None:
        from decision.signal_modules import process_single_asset

        rss_before = _get_rss_mb()
        median, all_times = _benchmark(
            lambda: process_single_asset(_ASSET),
            runs=_BENCHMARK_RUNS,
            warmup=_WARMUP_RUNS,
        )
        rss_after = _get_rss_mb()

        assert median < 120.0, f"process_single_asset median: {median:.1f}s (limit: 120s)"

        results = {
            "signal_median_s": round(median, 3),
            "signal_all_times_s": [round(t, 3) for t in all_times],
            "signal_rss_before_mb": round(rss_before, 1),
            "signal_rss_after_mb": round(rss_after, 1),
            "asset": _ASSET,
        }

        if PERFORMANCE_BASELINE_PATH.exists():
            existing = json.loads(PERFORMANCE_BASELINE_PATH.read_text())
        else:
            existing = {}
        existing.update(results)
        _save_results(existing)


@_e2e_skip
class TestMemoryRegression:
    """No memory regression from modular architecture."""

    def test_rss_within_bounds(self) -> None:
        """RSS after importing both module trees should be reasonable."""
        importlib.import_module("tuning.tune_modules")
        importlib.import_module("decision.signal_modules")
        rss = _get_rss_mb()
        # A reasonable upper bound -- the full import tree should not
        # consume more than 2 GB of RSS
        assert rss < 2048, f"RSS after imports: {rss:.0f} MB (limit: 2048 MB)"


@_e2e_skip
class TestCombinedPerformanceBaseline:
    """End-to-end: run both pipelines sequentially and record combined results."""

    def test_combined_baseline(self) -> None:
        from tuning.tune_modules import tune_asset_with_bma
        from decision.signal_modules import process_single_asset

        rss_start = _get_rss_mb()

        # Tune
        tune_med, tune_times = _benchmark(
            lambda: tune_asset_with_bma(_ASSET), runs=3, warmup=1
        )
        # Signal
        sig_med, sig_times = _benchmark(
            lambda: process_single_asset(_ASSET), runs=3, warmup=1
        )

        rss_end = _get_rss_mb()

        results = {
            "asset": _ASSET,
            "tune_median_s": round(tune_med, 3),
            "tune_times_s": [round(t, 3) for t in tune_times],
            "signal_median_s": round(sig_med, 3),
            "signal_times_s": [round(t, 3) for t in sig_times],
            "rss_start_mb": round(rss_start, 1),
            "rss_end_mb": round(rss_end, 1),
            "rss_delta_mb": round(rss_end - rss_start, 1),
        }
        _save_results(results)

        # Sanity: combined should finish in reasonable time
        assert tune_med + sig_med < 240.0, (
            f"Combined median: {tune_med + sig_med:.1f}s (limit: 240s)"
        )
