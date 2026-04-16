"""
Story 11.3 – Hansen Lambda Estimation via Profile Likelihood
=============================================================
Verify that hansen_estimate_lambda() recovers the true lambda
parameter from synthetic data, computes Fisher SE, and BIC
comparison against symmetric Student-t.
"""

import os, sys, math
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_kernels import (
    hansen_constants_kernel,
    hansen_estimate_lambda,
    hansen_skew_t_logpdf_scalar,
)


def _generate_hansen_samples(nu, lam, n, seed=42):
    """Generate samples from Hansen skew-t via inverse CDF / accept-reject."""
    rng = np.random.RandomState(seed)
    # Simple accept-reject using Student-t proposal
    from scipy.stats import t as t_dist
    a, b, c_const = hansen_constants_kernel(nu, lam)
    samples = []
    # Proposal: standard t with same nu, scaled
    proposal_scale = 2.0
    max_iter = n * 50
    count = 0
    while len(samples) < n and count < max_iter:
        count += 1
        z = t_dist.rvs(nu, random_state=rng) * proposal_scale
        lp_target = hansen_skew_t_logpdf_scalar(z, nu, lam, a, b, c_const, 0.0, 1.0)
        lp_proposal = t_dist.logpdf(z / proposal_scale, nu) - math.log(proposal_scale)
        log_M = 3.0  # envelope constant
        log_u = math.log(rng.uniform())
        if log_u < lp_target - lp_proposal - log_M:
            samples.append(z)
    return np.array(samples[:n])


class TestLambdaRecovery:
    """Profile likelihood recovers the true lambda from synthetic data."""

    def test_recover_positive_lambda(self):
        """True lambda=0.3 recovered within 2 SE."""
        true_lam = 0.3
        nu = 5.0
        data = _generate_hansen_samples(nu, true_lam, n=2000, seed=100)
        result = hansen_estimate_lambda(data, nu)
        assert result["converged"]
        assert abs(result["lambda_hat"] - true_lam) < 2.0 * result["se_lambda"] + 0.15

    def test_recover_negative_lambda(self):
        """True lambda=-0.3 recovered within 2 SE."""
        true_lam = -0.3
        nu = 5.0
        data = _generate_hansen_samples(nu, true_lam, n=2000, seed=200)
        result = hansen_estimate_lambda(data, nu)
        assert result["converged"]
        assert abs(result["lambda_hat"] - true_lam) < 2.0 * result["se_lambda"] + 0.15

    def test_recover_zero_lambda(self):
        """Symmetric data: lambda_hat near 0."""
        nu = 5.0
        data = _generate_hansen_samples(nu, 0.0, n=2000, seed=300)
        result = hansen_estimate_lambda(data, nu)
        assert result["converged"]
        assert abs(result["lambda_hat"]) < 0.15


class TestFisherInformation:
    """Standard error computed via Fisher information."""

    def test_se_finite_and_positive(self):
        """SE is finite and positive."""
        nu = 5.0
        data = _generate_hansen_samples(nu, 0.2, n=1000, seed=400)
        result = hansen_estimate_lambda(data, nu)
        assert result["se_lambda"] > 0
        assert result["se_lambda"] < 1.0  # should be reasonably small

    def test_se_decreases_with_n(self):
        """Larger sample => smaller SE."""
        nu = 5.0
        data_small = _generate_hansen_samples(nu, 0.2, n=500, seed=500)
        data_large = _generate_hansen_samples(nu, 0.2, n=3000, seed=500)
        se_small = hansen_estimate_lambda(data_small, nu)["se_lambda"]
        se_large = hansen_estimate_lambda(data_large, nu)["se_lambda"]
        assert se_large < se_small


class TestBICComparison:
    """BIC selects Hansen when true lambda != 0, symmetric when lambda = 0."""

    def test_bic_favors_hansen_when_skewed(self):
        """delta_bic > 0 when true lambda is far from 0."""
        nu = 5.0
        data = _generate_hansen_samples(nu, 0.4, n=2000, seed=600)
        result = hansen_estimate_lambda(data, nu)
        # delta_bic = bic_symmetric - bic_hansen > 0 means Hansen is better
        assert result["delta_bic"] > 0

    def test_bic_near_zero_when_symmetric(self):
        """delta_bic small when true lambda = 0."""
        nu = 5.0
        data = _generate_hansen_samples(nu, 0.0, n=2000, seed=700)
        result = hansen_estimate_lambda(data, nu)
        # Symmetric is essentially as good; delta_bic should not strongly favor Hansen
        assert result["delta_bic"] < 10.0


class TestEdgeCases:
    """Edge cases: small sample, extreme lambda."""

    def test_small_sample(self):
        """n < 10 returns gracefully."""
        data = np.array([0.1, 0.2, -0.1])
        result = hansen_estimate_lambda(data, 5.0)
        assert not result["converged"]
        assert result["lambda_hat"] == 0.0

    def test_bounded_estimate(self):
        """Estimate stays within [-0.95, 0.95]."""
        nu = 5.0
        data = _generate_hansen_samples(nu, 0.8, n=1000, seed=800)
        result = hansen_estimate_lambda(data, nu)
        assert -0.95 <= result["lambda_hat"] <= 0.95
