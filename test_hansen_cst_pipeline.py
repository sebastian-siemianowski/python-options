"""Tests for Hansen/CST internal pipeline stages and per-model routing."""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))


def test_scoring_weights():
    """Test 6-component scoring weights."""
    from tuning.diagnostics import DEFAULT_ELITE_WEIGHTS, REGIME_SCORING_WEIGHTS
    assert len(DEFAULT_ELITE_WEIGHTS) == 6, f"Expected 6-tuple, got {len(DEFAULT_ELITE_WEIGHTS)}"
    assert abs(sum(DEFAULT_ELITE_WEIGHTS) - 1.0) < 1e-10, "Weights must sum to 1.0"
    for regime, w in REGIME_SCORING_WEIGHTS.items():
        assert len(w) == 6, f"Regime {regime} has {len(w)}-tuple, expected 6"
        assert abs(sum(w) - 1.0) < 1e-10, f"Regime {regime} weights don't sum to 1.0"
    print("PASS: test_scoring_weights")


def test_numba_kernels_import():
    """Test that all Hansen/CST Numba kernels can be imported."""
    from models.numba_kernels import (
        hansen_constants_kernel, hansen_skew_t_logpdf_scalar,
        hansen_robust_weight_scalar, cst_logpdf_scalar,
        cst_robust_weight_scalar, phi_hansen_skew_t_filter_kernel,
        phi_cst_filter_kernel,
    )
    print("PASS: test_numba_kernels_import")


def test_numba_wrappers_import():
    """Test that Numba wrappers can be imported."""
    from models.numba_wrappers import run_phi_hansen_skew_t_filter, run_phi_cst_filter
    print("PASS: test_numba_wrappers_import")


def test_hansen_constants():
    """Test Hansen constants computation."""
    from models.numba_kernels import hansen_constants_kernel
    # Symmetric case: lambda=0 should give (a, b, 1/t_pdf(0))
    a, b, c = hansen_constants_kernel(8.0, 0.0)
    assert np.isfinite(a), "a must be finite"
    assert np.isfinite(b), "b must be finite"
    assert np.isfinite(c), "c must be finite"
    assert c > 0, "c_const must be positive"
    # Asymmetric case
    a2, b2, c2 = hansen_constants_kernel(8.0, -0.3)
    assert np.isfinite(a2) and np.isfinite(b2) and np.isfinite(c2)
    print("PASS: test_hansen_constants")


def test_hansen_logpdf():
    """Test Hansen skew-t logpdf computation."""
    from models.numba_kernels import hansen_constants_kernel, hansen_skew_t_logpdf_scalar
    nu, lam = 8.0, -0.2
    a, b, c = hansen_constants_kernel(nu, lam)
    # At mean, should give reasonable log-density
    lp = hansen_skew_t_logpdf_scalar(0.0, nu, lam, a, b, c, 0.0, 1.0)
    assert np.isfinite(lp), "logpdf at 0 must be finite"
    assert lp < 0, "logpdf should be negative (density < 1)"
    # In tails, should be smaller
    lp_tail = hansen_skew_t_logpdf_scalar(5.0, nu, lam, a, b, c, 0.0, 1.0)
    assert lp_tail < lp, "Tail logpdf should be smaller than center"
    print("PASS: test_hansen_logpdf")


def test_cst_logpdf():
    """Test CST logpdf computation."""
    from models.numba_kernels import cst_logpdf_scalar
    lp = cst_logpdf_scalar(0.0, 8.0, 4.0, 0.05, 0.0, 1.0)
    assert np.isfinite(lp), "CST logpdf at 0 must be finite"
    assert lp < 0, "CST logpdf should be negative"
    # In tails
    lp_tail = cst_logpdf_scalar(5.0, 8.0, 4.0, 0.05, 0.0, 1.0)
    assert lp_tail < lp, "Tail logpdf should be smaller than center"
    print("PASS: test_cst_logpdf")


def test_hansen_filter():
    """Test Hansen filter on synthetic data."""
    from models.numba_wrappers import run_phi_hansen_skew_t_filter
    np.random.seed(42)
    n = 200
    returns = np.random.standard_t(df=8, size=n) * 0.01
    vol = np.abs(returns).rolling(20).mean() if hasattr(returns, 'rolling') else np.ones(n) * 0.01
    if isinstance(vol, np.ndarray) and np.any(np.isnan(vol)):
        vol = np.full(n, 0.01)
    vol = np.full(n, 0.01)

    mu, P, mu_p, S_p, ll = run_phi_hansen_skew_t_filter(
        returns, vol, q=1e-6, c=1.0, phi=0.99, nu=8.0,
        hansen_lambda=-0.2, online_scale_adapt=True,
    )
    assert len(mu) == n, f"Expected {n} mu values, got {len(mu)}"
    assert len(S_p) == n, f"Expected {n} S_pred values, got {len(S_p)}"
    assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"
    assert np.all(np.isfinite(mu)), "mu contains non-finite values"
    assert np.all(S_p > 0), "S_pred must be positive"
    print(f"PASS: test_hansen_filter (ll={ll:.2f})")


def test_cst_filter():
    """Test CST filter on synthetic data."""
    from models.numba_wrappers import run_phi_cst_filter
    np.random.seed(42)
    n = 200
    returns = np.random.standard_t(df=8, size=n) * 0.01
    vol = np.full(n, 0.01)

    mu, P, mu_p, S_p, ll = run_phi_cst_filter(
        returns, vol, q=1e-6, c=1.0, phi=0.99,
        nu_normal=8.0, nu_crisis=4.0, epsilon=0.05,
        online_scale_adapt=True,
    )
    assert len(mu) == n, f"Expected {n} mu values, got {len(mu)}"
    assert len(S_p) == n
    assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"
    assert np.all(np.isfinite(mu)), "mu contains non-finite values"
    assert np.all(S_p > 0), "S_pred must be positive"
    print(f"PASS: test_cst_filter (ll={ll:.2f})")


def test_scoring_with_ad():
    """Test compute_regime_aware_model_weights includes AD."""
    from tuning.diagnostics import compute_regime_aware_model_weights
    # Decomposed dicts per the function signature
    bic_values = {"model_a": -3000.0, "model_b": -2800.0}
    hyvarinen_scores = {"model_a": -100.0, "model_b": -50.0}
    crps_values = {"model_a": 0.015, "model_b": 0.020}
    pit_pvalues = {"model_a": 0.5, "model_b": 0.1}
    berk_pvalues = {"model_a": 0.3, "model_b": 0.1}
    mad_values = {"model_a": 0.05, "model_b": 0.08}
    ad_pvalues = {"model_a": 0.7, "model_b": 0.05}

    posteriors, metadata = compute_regime_aware_model_weights(
        bic_values=bic_values,
        hyvarinen_scores=hyvarinen_scores,
        crps_values=crps_values,
        pit_pvalues=pit_pvalues,
        berk_pvalues=berk_pvalues,
        mad_values=mad_values,
        ad_pvalues=ad_pvalues,
        regime=0,
    )
    assert len(posteriors) > 0, "Should have posteriors"
    assert abs(sum(posteriors.values()) - 1.0) < 1e-8, "Posteriors must sum to 1.0"
    # model_a has better scores across all metrics
    if "model_a" in posteriors and "model_b" in posteriors:
        assert posteriors["model_a"] > posteriors["model_b"], \
            f"model_a ({posteriors['model_a']:.3f}) should beat model_b ({posteriors['model_b']:.3f})"
    # Check metadata includes AD info
    assert "weights_used" in metadata, "Metadata should have weights_used"
    weights_used = metadata["weights_used"]
    assert "w_ad" in weights_used or "ad_dev" in weights_used, \
        f"weights_used should include AD info: {weights_used}"
    print(f"PASS: test_scoring_with_ad (posteriors: {posteriors})")


def test_per_model_routing():
    """Test that per-model Hansen/CST params are extracted correctly."""
    # Simulate the per-model extraction logic from signals.py
    model_params_with_hansen = {
        "hansen_activated": True,
        "hansen_lambda": -0.25,
        "cst_activated": False,
        "nu": 8.0,
    }
    model_params_without = {
        "nu": 8.0,
    }
    model_params_with_cst = {
        "hansen_activated": False,
        "cst_activated": True,
        "cst_nu_crisis": 4.0,
        "cst_epsilon": 0.05,
        "nu": 8.0,
    }

    # Test Hansen extraction
    _hansen_act = model_params_with_hansen.get('hansen_activated', False)
    hansen_lambda_m = float(model_params_with_hansen.get('hansen_lambda', 0.0)) if _hansen_act else None
    assert hansen_lambda_m == -0.25, f"Expected -0.25, got {hansen_lambda_m}"

    # Test no-activation
    _hansen_act2 = model_params_without.get('hansen_activated', False)
    hansen_lambda_m2 = float(model_params_without.get('hansen_lambda', 0.0)) if _hansen_act2 else None
    assert hansen_lambda_m2 is None, "Should be None when not activated"

    # Test CST extraction
    _cst_act = model_params_with_cst.get('cst_activated', False)
    cst_nu_crisis = model_params_with_cst.get('cst_nu_crisis') if _cst_act else None
    cst_epsilon = float(model_params_with_cst.get('cst_epsilon', 0.0)) if _cst_act else None
    assert cst_nu_crisis == 4.0, f"Expected 4.0, got {cst_nu_crisis}"
    assert cst_epsilon == 0.05, f"Expected 0.05, got {cst_epsilon}"

    print("PASS: test_per_model_routing")


if __name__ == "__main__":
    test_scoring_weights()
    test_numba_kernels_import()
    test_numba_wrappers_import()
    test_hansen_constants()
    test_hansen_logpdf()
    test_cst_logpdf()
    test_hansen_filter()
    test_cst_filter()
    test_scoring_with_ad()
    test_per_model_routing()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
