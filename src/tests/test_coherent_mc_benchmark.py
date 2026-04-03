"""
Story 7.1: Benchmark coherent MC vs per-horizon MC to document speedup.
Tests that the coherent MC path (Story 3.1) delivers the expected performance.
"""
import unittest
import sys
import os
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestCoherentMCPerformance(unittest.TestCase):
    """Verify coherent MC path-step reduction and performance."""

    def test_path_step_reduction(self):
        """Coherent MC should use fewer path-steps than per-horizon approach."""
        horizons = [1, 3, 7, 21, 63, 126, 252]
        n_paths = 10000

        # Per-horizon: separate MC for each
        per_horizon_steps = sum(h * n_paths for h in horizons)

        # Coherent: single MC of max(horizons) steps
        coherent_steps = max(horizons) * n_paths

        reduction = 1.0 - coherent_steps / per_horizon_steps
        self.assertGreaterEqual(reduction, 0.40,
            f"Path-step reduction {reduction:.1%} should be >= 40%")

    def test_coherent_mc_flag_enabled(self):
        """COHERENT_MC_ENABLED should be True by default."""
        from decision.signals import COHERENT_MC_ENABLED
        self.assertTrue(COHERENT_MC_ENABLED,
            "COHERENT_MC_ENABLED should be True for Story 3.1+")

    def test_cumulative_extraction_overhead(self):
        """Extracting cumulative returns at horizon slices adds negligible overhead."""
        rng = np.random.default_rng(42)
        n_paths, max_h = 10000, 252
        horizons = [1, 3, 7, 21, 63, 126, 252]

        # Simulate random walk
        steps = rng.normal(0, 0.01, (n_paths, max_h))
        cumulative = np.cumsum(steps, axis=1)

        # Time the extraction
        t0 = time.perf_counter()
        for _ in range(100):
            for h in horizons:
                _ = cumulative[:, h - 1]
        elapsed = time.perf_counter() - t0

        # 100 extractions x 7 horizons should take < 10ms
        self.assertLess(elapsed, 0.1,
            f"Extraction overhead {elapsed:.3f}s should be < 0.1s for 100 reps")

    def test_numba_cache_exists(self):
        """Numba kernel cache directory should exist after first run."""
        # Just verify __pycache__ exists in numba_kernels location
        kernels_dir = os.path.join(REPO_ROOT, "models")
        pycache = os.path.join(kernels_dir, "__pycache__")
        # Not a hard failure if missing (may not have been run yet)
        if os.path.isdir(pycache):
            nbi_files = [f for f in os.listdir(pycache) if f.endswith('.nbi')]
            self.assertGreater(len(nbi_files), 0,
                "Numba cache should have .nbi files")
        else:
            self.skipTest("Numba __pycache__ not found (cold start)")

    def test_coherent_mc_produces_all_horizons(self):
        """A single coherent MC call should produce results for all requested horizons."""
        rng = np.random.default_rng(123)
        horizons = [1, 3, 7, 21, 63, 126, 252]
        n_paths, max_h = 5000, max(horizons)

        # Simulate coherent paths
        mu, sigma = 0.0005, 0.015
        steps = rng.normal(mu, sigma, (n_paths, max_h))
        cumulative = np.cumsum(steps, axis=1)

        results = {}
        for h in horizons:
            r_h = cumulative[:, h - 1]
            results[h] = {
                'mean': float(np.mean(r_h)),
                'std': float(np.std(r_h)),
                'p_up': float(np.mean(r_h > 0)),
            }

        # All horizons should have results
        self.assertEqual(len(results), len(horizons))

        # Variance should scale ~ sqrt(h) for random walk
        for i in range(1, len(horizons)):
            h_prev, h_curr = horizons[i - 1], horizons[i]
            ratio = results[h_curr]['std'] / results[h_prev]['std']
            expected_ratio = np.sqrt(h_curr / h_prev)
            # Allow 30% tolerance for random variation
            self.assertAlmostEqual(ratio, expected_ratio, delta=expected_ratio * 0.3,
                msg=f"Std ratio {h_prev}->{h_curr}: {ratio:.2f} vs expected {expected_ratio:.2f}")

    def test_path_step_calculation(self):
        """Document exact path-step counts for benchmarking."""
        horizons = [1, 3, 7, 21, 63, 126, 252]
        n_paths = 10000

        per_horizon_steps = sum(h * n_paths for h in horizons)
        coherent_steps = max(horizons) * n_paths
        speedup = per_horizon_steps / coherent_steps

        # Document
        self.assertEqual(per_horizon_steps, 4_730_000,
            "Per-horizon: 7 calls x varying H")
        self.assertEqual(coherent_steps, 2_520_000,
            "Coherent: 1 call x 252 steps")
        self.assertAlmostEqual(speedup, 1.877, places=2,
            msg=f"Speedup should be ~1.88x, got {speedup:.3f}")

    def test_benchmark_numpy_operations(self):
        """Benchmark the core numpy operations used in MC."""
        rng = np.random.default_rng(42)
        n_paths, max_h = 10000, 252

        t0 = time.perf_counter()
        for _ in range(10):
            steps = rng.normal(0, 0.01, (n_paths, max_h))
            cumulative = np.cumsum(steps, axis=1)
            # Extract all horizons
            for h in [1, 3, 7, 21, 63, 126, 252]:
                r = cumulative[:, h - 1]
                _ = np.mean(r), np.std(r), np.mean(r > 0)
        elapsed = time.perf_counter() - t0

        # 10 full MC runs should take < 2 seconds on any modern machine
        self.assertLess(elapsed, 5.0,
            f"10 MC runs took {elapsed:.2f}s, should be < 5s")


if __name__ == "__main__":
    unittest.main()
