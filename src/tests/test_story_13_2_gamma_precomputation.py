"""
Story 13.2 – Gamma Precomputation Consistency
===============================================
Verify precompute_gamma_values() LRU cache correctness,
scipy agreement, and consistency across repeated calls.
"""

import os, sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.special import gammaln
from models.numba_wrappers import precompute_gamma_values, validate_gamma_precomputation


class TestGammaSciPyAgreement:
    """precompute_gamma_values matches scipy to machine precision."""

    @pytest.mark.parametrize("nu", [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 20.0, 50.0])
    def test_matches_scipy(self, nu):
        lg_half, lg_half_plus = precompute_gamma_values(nu)
        ref_half = float(gammaln(nu / 2.0))
        ref_half_plus = float(gammaln((nu + 1.0) / 2.0))
        assert abs(lg_half - ref_half) < 1e-14
        assert abs(lg_half_plus - ref_half_plus) < 1e-14


class TestCacheConsistency:
    """Repeated calls return identical (cached) values."""

    def test_cache_idempotent(self):
        """Same nu returns same objects."""
        a1, b1 = precompute_gamma_values(6.0)
        a2, b2 = precompute_gamma_values(6.0)
        assert a1 == a2
        assert b1 == b2

    def test_cache_info_hits(self):
        """Repeated calls increment hits."""
        precompute_gamma_values.cache_clear()
        precompute_gamma_values(5.0)
        precompute_gamma_values(5.0)
        precompute_gamma_values(5.0)
        info = precompute_gamma_values.cache_info()
        assert info.hits >= 2


class TestNonIntegerNu:
    """Non-integer nu values handled."""

    @pytest.mark.parametrize("nu", [3.5, 4.7, 7.1, 15.3])
    def test_fractional_nu(self, nu):
        lg_half, lg_half_plus = precompute_gamma_values(nu)
        ref_half = float(gammaln(nu / 2.0))
        ref_half_plus = float(gammaln((nu + 1.0) / 2.0))
        assert abs(lg_half - ref_half) < 1e-14
        assert abs(lg_half_plus - ref_half_plus) < 1e-14


class TestMathProperties:
    """Gamma function mathematical properties."""

    def test_monotonic_increase(self):
        """gammaln(x) is increasing for x > 1.46."""
        nus = [4.0, 6.0, 8.0, 10.0, 20.0]
        vals = [precompute_gamma_values(nu)[0] for nu in nus]
        for i in range(1, len(vals)):
            assert vals[i] > vals[i - 1]

    def test_recurrence_relation(self):
        """gammaln(x+1) = ln(x) + gammaln(x)."""
        for nu in [4.0, 6.0, 10.0]:
            x = nu / 2.0
            lg_x = float(gammaln(x))
            lg_x1 = float(gammaln(x + 1.0))
            assert abs(lg_x1 - (np.log(x) + lg_x)) < 1e-13


class TestValidateDiagnostic:
    """validate_gamma_precomputation() works."""

    def test_default_run(self):
        diag = validate_gamma_precomputation()
        assert diag["all_consistent"]
        assert diag["max_error"] < 1e-14
        assert diag["cache_info"]["maxsize"] == 64
