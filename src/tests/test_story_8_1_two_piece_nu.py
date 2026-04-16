"""
Story 8.1: Two-Piece Student-t with Continuous Nu_L and Nu_R
=============================================================
Profile likelihood refinement of asymmetric tail parameters.
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

from models.phi_student_t import refine_two_piece_nu


class TestTwoPieceContinuousNu:
    """Acceptance criteria for Story 8.1."""

    def _generate_two_piece_data(self, nu_L, nu_R, n=2000, seed=42):
        """Generate two-piece Student-t data."""
        rng = np.random.RandomState(seed)
        left = rng.standard_t(nu_L, n // 2)
        right = rng.standard_t(nu_R, n // 2)
        left = left[left < 0]
        right = right[right >= 0]
        return np.concatenate([left, right])

    def test_equities_nu_L_less_nu_R(self):
        """AC1: Equities should have nu_L < nu_R (heavier crash tails)."""
        data = self._generate_two_piece_data(nu_L=4.0, nu_R=12.0, n=3000)
        nu_L, nu_R, ll = refine_two_piece_nu(4.0, 12.0, data)
        assert nu_L < nu_R, f"Expected nu_L < nu_R, got {nu_L:.2f} vs {nu_R:.2f}"

    def test_gold_symmetric(self):
        """AC2: Gold-like symmetric data should have nu_L ~ nu_R."""
        rng = np.random.RandomState(77)
        data = rng.standard_t(8, 3000)  # Symmetric Student-t
        nu_L, nu_R, ll = refine_two_piece_nu(8.0, 8.0, data)
        # Should be approximately symmetric
        assert abs(nu_L - nu_R) < 4.0, f"Expected symmetric, got nu_L={nu_L:.2f}, nu_R={nu_R:.2f}"

    def test_bic_improvement_over_grid(self):
        """AC3: Refined nu should have better (or equal) LL than grid values."""
        data = self._generate_two_piece_data(nu_L=3.5, nu_R=15.0, n=3000)
        nu_L_ref, nu_R_ref, ll_ref = refine_two_piece_nu(3.0, 12.0, data)

        # Compute grid LL for comparison
        from scipy.special import gammaln as scipy_gammaln
        z = data
        left_mask = z < 0.0
        right_mask = ~left_mask

        def _ll(nu_L, nu_R):
            ll = 0.0
            for mask, nu in [(left_mask, nu_L), (right_mask, nu_R)]:
                x = z[mask]
                ll += np.sum(
                    scipy_gammaln(0.5 * (nu + 1)) - scipy_gammaln(0.5 * nu)
                    - 0.5 * np.log(nu * np.pi)
                    - 0.5 * (nu + 1) * np.log(1.0 + x ** 2 / nu)
                )
            return ll

        ll_grid = _ll(3.0, 12.0)
        assert ll_ref >= ll_grid - 0.1, \
            f"Refined LL {ll_ref:.2f} worse than grid LL {ll_grid:.2f}"

    def test_returns_valid_nu(self):
        """AC4: Refined nu values within valid bounds."""
        rng = np.random.RandomState(42)
        data = rng.standard_t(5, 2000)
        nu_L, nu_R, ll = refine_two_piece_nu(5.0, 8.0, data)
        assert 2.1 <= nu_L <= 30.0, f"nu_L={nu_L} out of bounds"
        assert 2.1 <= nu_R <= 30.0, f"nu_R={nu_R} out of bounds"

    def test_no_degradation_symmetric(self):
        """AC5: When two-piece loses (symmetric data), no degradation."""
        rng = np.random.RandomState(99)
        data = rng.standard_t(6, 2000)
        nu_L, nu_R, ll_tp = refine_two_piece_nu(6.0, 6.0, data)

        # Symmetric LL
        from scipy.special import gammaln as scipy_gammaln
        ll_sym = np.sum(
            scipy_gammaln(0.5 * 7) - scipy_gammaln(0.5 * 6)
            - 0.5 * np.log(6 * np.pi)
            - 0.5 * 7 * np.log(1.0 + data ** 2 / 6)
        )
        # Two-piece should be at least as good as symmetric (it can be symmetric)
        assert ll_tp >= ll_sym - 1.0

    def test_heavy_left_tail(self):
        """AC6: With heavy left tail data, nu_L should be small."""
        rng = np.random.RandomState(42)
        left = rng.standard_t(3.0, 1500)
        right = rng.standard_t(20.0, 1500)
        left = left[left < 0]
        right = right[right >= 0]
        data = np.concatenate([left, right])
        nu_L, nu_R, ll = refine_two_piece_nu(3.0, 20.0, data)
        assert nu_L < nu_R, f"Expected nu_L < nu_R for heavy left tail"
        assert nu_L < 8.0, f"Expected nu_L < 8 for heavy crash tails, got {nu_L:.2f}"
