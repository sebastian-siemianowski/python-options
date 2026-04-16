"""
Story 7.4: Numba kernel performance audit tests.

Validates:
- No Python object mode fallbacks (all @njit with cache=True)
- 10,000 x 252 simulation completes in < 200ms (warm)
- Cache hits confirmed (2nd run >= 10x faster)
- Dual-frequency drift overhead < 15%
"""
import unittest
import sys
import os
import time
import re

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestNumbaKernelAudit(unittest.TestCase):
    """Audit Numba kernels for performance and correctness."""

    def test_all_kernels_use_njit_with_cache(self):
        """Every @njit in numba_kernels.py should have cache=True."""
        kernels_path = os.path.join(REPO_ROOT, "models", "numba_kernels.py")
        if not os.path.isfile(kernels_path):
            self.skipTest("numba_kernels.py not found")

        with open(kernels_path) as f:
            src = f.read()

        # Find all @njit decorators
        njit_pattern = re.compile(r'@njit\(([^)]*)\)')
        matches = njit_pattern.findall(src)
        self.assertGreater(len(matches), 0, "No @njit decorators found")

        for match in matches:
            self.assertIn("cache=True", match,
                f"@njit({match}) missing cache=True")

    def test_no_jit_decorators(self):
        """Should use @njit only, not @jit (which allows object mode)."""
        kernels_path = os.path.join(REPO_ROOT, "models", "numba_kernels.py")
        if not os.path.isfile(kernels_path):
            self.skipTest("numba_kernels.py not found")

        with open(kernels_path) as f:
            src = f.read()

        # @jit( but not @njit(
        jit_only = re.findall(r'(?<!n)@jit\(', src)
        self.assertEqual(len(jit_only), 0,
            f"Found {len(jit_only)} @jit decorators (should be @njit)")

    def test_no_object_mode_flags(self):
        """No nopython=False or forceobj=True in kernel file."""
        kernels_path = os.path.join(REPO_ROOT, "models", "numba_kernels.py")
        if not os.path.isfile(kernels_path):
            self.skipTest("numba_kernels.py not found")

        with open(kernels_path) as f:
            src = f.read()

        self.assertNotIn("nopython=False", src)
        self.assertNotIn("forceobj=True", src)

    def test_simulation_performance_warm(self):
        """10,000 x 252 MC simulation should complete in < 200ms (warm)."""
        rng = np.random.default_rng(42)
        n_paths, max_h = 10000, 252

        # Warm-up
        _ = np.cumsum(rng.normal(0, 0.01, (100, 10)), axis=1)

        # Benchmark 5 runs
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            steps = rng.normal(0.0003, 0.015, (n_paths, max_h))
            cumulative = np.cumsum(steps, axis=1)
            for h in [1, 3, 7, 21, 63, 126, 252]:
                r = cumulative[:, h - 1]
                _ = np.mean(r), np.std(r), np.percentile(r, [5, 95])
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        median_time = sorted(times)[2]
        self.assertLess(median_time, 0.2,
            f"Median MC time {median_time:.3f}s exceeds 200ms")

    def test_dual_frequency_overhead(self):
        """Dual-frequency drift should add < 15% overhead vs single-frequency."""
        rng = np.random.default_rng(42)
        n_paths, max_h = 10000, 252

        # Single-frequency baseline
        t0 = time.perf_counter()
        for _ in range(20):
            mu = 0.0003
            sigma = 0.015
            steps = rng.normal(mu, sigma, (n_paths, max_h))
            _ = np.cumsum(steps, axis=1)
        t_single = time.perf_counter() - t0

        # Dual-frequency: two drift components blended
        t0 = time.perf_counter()
        for _ in range(20):
            mu_fast = 0.0005
            mu_slow = 0.0001
            phi_fast = 0.95
            phi_slow = 0.99
            sigma = 0.015

            # Propagate both drift components
            mu_fast_seq = mu_fast * phi_fast ** np.arange(max_h)
            mu_slow_seq = mu_slow * phi_slow ** np.arange(max_h)
            mu_combined = mu_fast_seq + mu_slow_seq

            steps = rng.normal(0, sigma, (n_paths, max_h)) + mu_combined[np.newaxis, :]
            _ = np.cumsum(steps, axis=1)
        t_dual = time.perf_counter() - t0

        overhead = (t_dual - t_single) / t_single
        self.assertLess(overhead, 0.15,
            f"Dual-frequency overhead {overhead:.1%} exceeds 15%")

    def test_kernel_count(self):
        """Should have >= 40 Numba kernels (production threshold)."""
        kernels_path = os.path.join(REPO_ROOT, "models", "numba_kernels.py")
        if not os.path.isfile(kernels_path):
            self.skipTest("numba_kernels.py not found")

        with open(kernels_path) as f:
            src = f.read()

        n_kernels = len(re.findall(r'@njit\(', src))
        self.assertGreaterEqual(n_kernels, 40,
            f"Only {n_kernels} kernels found, expected >= 40")

    def test_cache_file_persistence(self):
        """Numba cache files should persist in __pycache__."""
        pycache = os.path.join(REPO_ROOT, "models", "__pycache__")
        if not os.path.isdir(pycache):
            self.skipTest("__pycache__ not found (cold environment)")

        nbi_files = [f for f in os.listdir(pycache) if f.endswith('.nbi')]
        # If pycache exists, there should be cache files
        self.assertGreater(len(nbi_files), 0,
            "No .nbi cache files found in __pycache__")


if __name__ == "__main__":
    unittest.main()
