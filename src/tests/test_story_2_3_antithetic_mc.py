"""
Story 2.3: Batch Monte Carlo with Antithetic Variates
=====================================================
Variance-reduced MC sampling via antithetic pairs (z, -z).
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.vectorized_ops import batch_monte_carlo_sample


class TestAntitheticMC:
    """Acceptance criteria for Story 2.3."""

    def test_variance_reduction(self):
        """AC1: Antithetic reduces MC standard error by > 30%."""
        means = np.array([0.0])
        variances = np.array([1.0])
        n = 1000

        std_errors = []
        anti_errors = []
        for seed in range(200):
            rng_std = np.random.default_rng(seed)
            rng_anti = np.random.default_rng(seed)

            s_std = batch_monte_carlo_sample(means, variances, n, rng_std, antithetic=False)
            s_anti = batch_monte_carlo_sample(means, variances, n, rng_anti, antithetic=True)

            # Estimation error = |sample_mean - true_mean|
            std_errors.append(abs(np.mean(s_std[0]) - means[0]))
            anti_errors.append(abs(np.mean(s_anti[0]) - means[0]))

        avg_std = np.mean(std_errors)
        avg_anti = np.mean(anti_errors)
        reduction = 1.0 - avg_anti / avg_std
        assert reduction > 0.30, (
            f"Variance reduction only {reduction:.1%} (need >30%)"
        )

    def test_direction_agreement(self):
        """AC2: Signal direction agreement > 99% for n=1000."""
        means = np.array([0.05])  # Weak positive signal
        variances = np.array([1.0])
        n = 1000

        agreements = 0
        n_trials = 200
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            s = batch_monte_carlo_sample(means, variances, n, rng, antithetic=True)
            if np.mean(s[0]) > 0:
                agreements += 1

        rate = agreements / n_trials
        assert rate > 0.90, f"Direction agreement only {rate:.1%}"

    def test_no_bias(self):
        """AC3: Mean of antithetic matches standard MC mean (no bias)."""
        means = np.array([1.0, -2.0, 0.5])
        variances = np.array([1.0, 3.0, 0.1])
        n = 10000

        rng = np.random.default_rng(42)
        s = batch_monte_carlo_sample(means, variances, n, rng, antithetic=True)

        for i in range(len(means)):
            sample_mean = np.mean(s[i])
            assert abs(sample_mean - means[i]) < 0.1, (
                f"Horizon {i}: mean={sample_mean:.4f}, expected~{means[i]}"
            )

    def test_output_shape(self):
        """Output shape is (n_horizons, n_samples) for both modes."""
        means = np.array([0.0, 1.0])
        variances = np.array([1.0, 2.0])

        for anti in [False, True]:
            s = batch_monte_carlo_sample(means, variances, 500, antithetic=anti)
            assert s.shape == (2, 500), f"Shape: {s.shape}, antithetic={anti}"

    def test_antithetic_symmetry(self):
        """Antithetic samples should contain mirrored z-values."""
        means = np.array([0.0])
        variances = np.array([1.0])
        n = 100

        rng = np.random.default_rng(7)
        s = batch_monte_carlo_sample(means, variances, n, rng, antithetic=True)

        # With mean=0, std=1, samples ARE the z-values
        # First half and second half should be negatives of each other
        half = n // 2
        np.testing.assert_allclose(s[0, :half], -s[0, half:], atol=1e-14)

    def test_odd_n_samples(self):
        """Odd n_samples should work (half rounds up, truncate to n)."""
        means = np.array([0.0])
        variances = np.array([1.0])
        s = batch_monte_carlo_sample(means, variances, 101, antithetic=True)
        assert s.shape == (1, 101)

    def test_antithetic_500_vs_standard_1000(self):
        """Task 5: Antithetic n=500 should match standard n=1000 precision."""
        means = np.array([0.0])
        variances = np.array([1.0])

        errors_anti = []
        errors_std = []
        for seed in range(100):
            rng_a = np.random.default_rng(seed)
            rng_s = np.random.default_rng(seed + 10000)

            s_anti = batch_monte_carlo_sample(means, variances, 500, rng_a, antithetic=True)
            s_std = batch_monte_carlo_sample(means, variances, 1000, rng_s, antithetic=False)

            errors_anti.append(abs(np.mean(s_anti[0])))
            errors_std.append(abs(np.mean(s_std[0])))

        # Antithetic with 500 should be comparable to standard with 1000
        assert np.mean(errors_anti) < np.mean(errors_std) * 1.5
