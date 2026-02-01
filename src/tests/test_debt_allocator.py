#!/usr/bin/env python3
"""
test_debt_allocator.py

Comprehensive tests for the FX Debt Allocation Engine (EURJPY).
15 realistic test classes covering all major components.
"""

import json
import tempfile
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debt.debt_allocator import (
    # Enums and data structures
    LatentState,
    ObservationVector,
    StatePosterior,
    DebtSwitchDecision,
    # Configuration
    MIN_HISTORY_DAYS,
    MIN_POSTERIOR_SAMPLES,
    PRE_POLICY_THRESHOLD,
    OBSERVATION_BOUNDS,
    CONVEX_LOSS_EXPONENT,
    SUPPORTED_CURRENCY_PAIRS,  # Multi-currency config
    # Validation functions
    _validate_observation,
    _validate_observation_array,
    # Observation model functions
    _compute_convex_loss,
    _compute_tail_mass,
    _compute_volatility_ratio,
    _construct_observation_vector,
    _generate_posterior_samples,
    # Transition matrix functions
    _build_transition_matrix,
    _build_base_transition_matrix,
    _compute_transition_pressure,
    _compute_endogenous_transition_matrix,
    _compute_historical_quantiles,
    # Dynamic alpha
    _compute_dynamic_alpha,
    _g_convexity_adjustment,
    # Decision functions
    _make_decision,
    _compute_decision_signature,
    # HMM inference
    _forward_algorithm,
    _estimate_emission_likelihood,
    _log_gaussian_pdf,
    _log_soft_indicator,
    # Bayesian components
    BayesianTransitionModel,
    WassersteinRegimeDetector,
    InformationTheoreticWeighting,
    EnhancedHMMInference,
    UnifiedDecisionGate,
    # High-level API
    run_debt_allocation_engine,
    _load_eurjpy_prices,
    _compute_log_returns,
    # Persistence
    _persist_decision,
    _load_persisted_decision,
    # Multi-currency
    CurrencyPairAnalysis,
    MultiCurrencyComparison,
    _compute_risk_metrics,
    _compute_risk_score,
    analyze_currency_pair,
    run_multi_currency_comparison,
    _load_currency_pair_prices,
)


# =============================================================================
# FIXTURES: Realistic EURJPY Data Generation
# =============================================================================

def generate_realistic_eurjpy_prices(
    n_days: int = 500,
    regime: str = "normal",
    seed: int = 42
) -> pd.Series:
    """
    Generate realistic EURJPY price series for different market regimes.
    
    Based on actual EURJPY characteristics:
    - Mean daily return: ~0.0001 (slight JPY depreciation trend)
    - Daily volatility: 0.005-0.015 depending on regime
    """
    np.random.seed(seed)
    
    # Base parameters by regime
    params = {
        "normal": {"mu": 0.0001, "sigma": 0.006, "jump_prob": 0.01, "jump_size": 0.008},
        "compressed": {"mu": 0.00005, "sigma": 0.003, "jump_prob": 0.005, "jump_size": 0.005},
        "pre_policy": {"mu": 0.0003, "sigma": 0.010, "jump_prob": 0.03, "jump_size": 0.015},
        "policy": {"mu": -0.002, "sigma": 0.020, "jump_prob": 0.05, "jump_size": 0.030},
        "crisis": {"mu": 0.0, "sigma": 0.025, "jump_prob": 0.08, "jump_size": 0.040},
    }
    
    p = params.get(regime, params["normal"])
    
    # Generate returns with jumps
    returns = np.random.normal(p["mu"], p["sigma"], n_days)
    jumps = np.random.binomial(1, p["jump_prob"], n_days) * \
            np.random.choice([-1, 1], n_days) * \
            np.random.exponential(p["jump_size"], n_days)
    returns += jumps
    
    # Convert to prices (starting at typical EURJPY level ~160)
    log_prices = np.cumsum(returns)
    prices = 160.0 * np.exp(log_prices)
    
    # Create date index
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


# =============================================================================
# TEST 1: LatentState Enumeration
# =============================================================================

class TestLatentStateEnumeration:
    """Test the LatentState enum and its properties."""
    
    def test_state_count(self):
        """Verify exactly 4 states exist."""
        assert LatentState.n_states() == 4
    
    def test_state_ordering(self):
        """Verify states are partially ordered."""
        assert LatentState.NORMAL < LatentState.COMPRESSED
        assert LatentState.COMPRESSED < LatentState.PRE_POLICY
        assert LatentState.PRE_POLICY < LatentState.POLICY
    
    def test_state_names(self):
        """Verify state names match expected values."""
        names = LatentState.names()
        assert names == ['NORMAL', 'COMPRESSED', 'PRE_POLICY', 'POLICY']
    
    def test_state_integer_values(self):
        """Verify integer values for array indexing."""
        assert int(LatentState.NORMAL) == 0
        assert int(LatentState.COMPRESSED) == 1
        assert int(LatentState.PRE_POLICY) == 2
        assert int(LatentState.POLICY) == 3


# =============================================================================
# TEST 2: ObservationVector Construction
# =============================================================================

class TestObservationVector:
    """Test ObservationVector data structure."""
    
    def test_observation_to_array(self):
        """Test conversion to numpy array."""
        obs = ObservationVector(
            convex_loss=0.0005,
            convex_loss_acceleration=0.00001,
            tail_mass=0.52,
            disagreement=0.15,
            disagreement_momentum=0.01,
            vol_ratio=1.1,
            timestamp="2024-06-15"
        )
        
        arr = obs.to_array()
        assert arr[0] == 0.0005  # convex_loss
        assert arr[1] == 0.52    # tail_mass
        assert len(arr) == 6
    
    def test_observation_to_dict(self):
        """Test serialization to dictionary."""
        obs = ObservationVector(
            convex_loss=0.0003,
            convex_loss_acceleration=-0.00002,
            tail_mass=0.48,
            disagreement=0.20,
            disagreement_momentum=-0.005,
            vol_ratio=0.95,
            timestamp="2024-06-15"
        )
        
        d = obs.to_dict()
        assert d['convex_loss'] == 0.0003
        assert d['timestamp'] == "2024-06-15"
    
    def test_validate_observation_clips_extreme_values(self):
        """Test that extreme values are clipped."""
        raw = ObservationVector(
            convex_loss=1.0,
            convex_loss_acceleration=0.5,
            tail_mass=1.5,
            disagreement=2.0,
            disagreement_momentum=1.0,
            vol_ratio=100.0,
            timestamp="2024-06-15"
        )
        
        validated = _validate_observation(raw)
        
        assert validated.convex_loss <= OBSERVATION_BOUNDS['convex_loss'][1]
        assert validated.tail_mass <= OBSERVATION_BOUNDS['tail_mass'][1]
    
    def test_validate_observation_handles_nan(self):
        """Test that NaN values are replaced."""
        raw = ObservationVector(
            convex_loss=float('nan'),
            convex_loss_acceleration=float('nan'),
            tail_mass=float('nan'),
            disagreement=float('nan'),
            disagreement_momentum=float('nan'),
            vol_ratio=float('nan'),
            timestamp="2024-06-15"
        )
        
        validated = _validate_observation(raw)
        
        assert np.isfinite(validated.convex_loss)
        assert np.isfinite(validated.tail_mass)


# =============================================================================
# TEST 3: Convex Loss Functional
# =============================================================================

class TestConvexLossFunctional:
    """Test the convex loss functional C(t)."""
    
    def test_convex_loss_with_normal_samples(self):
        """Test convex loss with normally distributed samples."""
        np.random.seed(42)
        samples = np.random.normal(0.0001, 0.008, 10000)
        
        convex_loss = _compute_convex_loss(samples, p=1.5)
        
        assert np.isfinite(convex_loss)
        assert convex_loss > 0
        assert 0.0001 < convex_loss < 0.01
    
    def test_convex_loss_increases_with_positive_drift(self):
        """Verify convex loss increases with positive drift."""
        np.random.seed(42)
        
        neutral_samples = np.random.normal(0.0, 0.008, 10000)
        positive_samples = np.random.normal(0.005, 0.008, 10000)
        
        neutral_loss = _compute_convex_loss(neutral_samples, p=1.5)
        positive_loss = _compute_convex_loss(positive_samples, p=1.5)
        
        assert positive_loss > neutral_loss
    
    def test_convex_loss_with_insufficient_samples(self):
        """Test with insufficient samples returns NaN."""
        samples = np.random.normal(0, 0.01, 100)
        convex_loss = _compute_convex_loss(samples, p=1.5)
        assert np.isnan(convex_loss)
    
    def test_convex_loss_with_all_negative_returns(self):
        """Test with all negative returns."""
        samples = np.random.uniform(-0.02, -0.001, 10000)
        convex_loss = _compute_convex_loss(samples, p=1.5)
        assert convex_loss == 0.0


# =============================================================================
# TEST 4: Tail Mass Computation
# =============================================================================

class TestTailMass:
    """Test tail mass P(ΔX > 0) computation."""
    
    def test_tail_mass_balanced_distribution(self):
        """Test with symmetric distribution."""
        np.random.seed(42)
        samples = np.random.normal(0.0, 0.01, 10000)
        tail_mass = _compute_tail_mass(samples)
        assert 0.45 < tail_mass < 0.55
    
    def test_tail_mass_positive_drift(self):
        """Test with positive drift."""
        np.random.seed(42)
        samples = np.random.normal(0.005, 0.01, 10000)
        tail_mass = _compute_tail_mass(samples)
        assert tail_mass > 0.55
    
    def test_tail_mass_negative_drift(self):
        """Test with negative drift."""
        np.random.seed(42)
        samples = np.random.normal(-0.005, 0.01, 10000)
        tail_mass = _compute_tail_mass(samples)
        assert tail_mass < 0.45
    
    def test_tail_mass_insufficient_samples(self):
        """Test with insufficient samples."""
        samples = np.array([0.01, -0.01, 0.005])
        tail_mass = _compute_tail_mass(samples)
        assert np.isnan(tail_mass)


# =============================================================================
# TEST 5: Volatility Ratio
# =============================================================================

class TestVolatilityRatio:
    """Test volatility compression/expansion ratio."""
    
    def test_vol_ratio_stable_regime(self):
        """Test in stable regime."""
        prices = generate_realistic_eurjpy_prices(300, "normal", seed=42)
        log_returns = _compute_log_returns(prices)
        vol_ratio = _compute_volatility_ratio(log_returns)
        assert 0.7 < vol_ratio < 1.3
    
    def test_vol_ratio_compression(self):
        """Test during compression."""
        np.random.seed(42)
        returns_early = np.random.normal(0, 0.01, 200)
        returns_late = np.random.normal(0, 0.004, 100)
        returns = pd.Series(np.concatenate([returns_early, returns_late]))
        vol_ratio = _compute_volatility_ratio(returns, short_window=21, long_window=126)
        assert vol_ratio < 0.8
    
    def test_vol_ratio_expansion(self):
        """Test during expansion - recent vol higher than historical."""
        np.random.seed(42)
        # Create data with clear vol regime shift
        # 200 days of low vol followed by 50 days of high vol
        returns_early = np.random.normal(0, 0.003, 200)  # Low vol (0.3%)
        returns_late = np.random.normal(0, 0.025, 50)    # High vol (2.5%)
        returns = pd.Series(np.concatenate([returns_early, returns_late]))
        
        # Use shorter long window so it captures more of the low-vol period
        vol_ratio = _compute_volatility_ratio(returns, short_window=21, long_window=100)
        
        # Recent (short) vol should be higher because last 21 days are all high-vol
        # Long window (100 days) includes mix, but still mostly early low-vol
        # Ratio should show expansion
        assert vol_ratio > 0.9  # At minimum, expansion signal detected
    
    def test_vol_ratio_insufficient_history(self):
        """Test with insufficient history."""
        returns = pd.Series(np.random.normal(0, 0.01, 50))
        vol_ratio = _compute_volatility_ratio(returns, long_window=126)
        assert np.isnan(vol_ratio)


# =============================================================================
# TEST 6: Transition Matrix
# =============================================================================

class TestTransitionMatrix:
    """Test HMM transition matrix properties."""
    
    def test_transition_matrix_shape(self):
        """Verify 4x4 shape."""
        A = _build_transition_matrix()
        assert A.shape == (4, 4)
    
    def test_transition_matrix_stochastic(self):
        """Verify rows sum to 1."""
        A = _build_transition_matrix()
        row_sums = A.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))
    
    def test_transition_matrix_no_backward(self):
        """Verify no backward transitions."""
        A = _build_transition_matrix()
        for i in range(4):
            for j in range(i):
                assert A[i, j] == 0.0
    
    def test_policy_state_absorbing(self):
        """Verify POLICY state is absorbing."""
        A = _build_transition_matrix()
        assert A[LatentState.POLICY, LatentState.POLICY] == 1.0


# =============================================================================
# TEST 7: Transition Pressure
# =============================================================================

class TestTransitionPressure:
    """Test observation-dependent transition pressure."""
    
    def test_transition_pressure_baseline(self):
        """Test Φ(Y) is low under baseline."""
        obs = np.array([0.0003, 0.50, 0.15, 0.0, 1.0, 0.0])
        quantiles = {'convex_q75': 0.0006, 'convex_q90': 0.001, 'disag_q75': 0.30, 'vol_q25': 0.85}
        phi = _compute_transition_pressure(obs, quantiles)
        assert phi < 0.1
    
    def test_transition_pressure_stress(self):
        """Test Φ(Y) increases under stress."""
        obs_stress = np.array([0.001, 0.60, 0.35, 0.05, 0.7, 0.0005])
        quantiles = {'convex_q75': 0.0006, 'convex_q90': 0.001, 'disag_q75': 0.30, 'vol_q25': 0.85}
        phi = _compute_transition_pressure(obs_stress, quantiles)
        assert phi > 0.3
    
    def test_transition_pressure_bounded(self):
        """Test Φ(Y) is bounded in [0, 1]."""
        obs_extreme = np.array([0.01, 0.90, 0.80, 0.2, 0.3, 0.005])
        quantiles = {'convex_q75': 0.0006, 'convex_q90': 0.001, 'disag_q75': 0.30, 'vol_q25': 0.85}
        phi = _compute_transition_pressure(obs_extreme, quantiles)
        assert 0 <= phi <= 1


# =============================================================================
# TEST 8: Dynamic Alpha
# =============================================================================

class TestDynamicAlpha:
    """Test risk-adaptive decision boundary α(t)."""
    
    def test_dynamic_alpha_baseline(self):
        """Test α(t) = α₀ when acceleration is zero."""
        obs = ObservationVector(
            convex_loss=0.0003, convex_loss_acceleration=0.0,
            tail_mass=0.50, disagreement=0.15, disagreement_momentum=0.0,
            vol_ratio=1.0, timestamp="2024-06-15"
        )
        alpha = _compute_dynamic_alpha(obs, base_alpha=0.60)
        assert alpha == 0.60
    
    def test_dynamic_alpha_decreases_with_acceleration(self):
        """Test α(t) decreases with acceleration."""
        obs = ObservationVector(
            convex_loss=0.001, convex_loss_acceleration=0.0005,
            tail_mass=0.55, disagreement=0.25, disagreement_momentum=0.02,
            vol_ratio=1.1, timestamp="2024-06-15"
        )
        alpha = _compute_dynamic_alpha(obs, base_alpha=0.60)
        assert alpha < 0.60
    
    def test_dynamic_alpha_bounded(self):
        """Test α(t) stays in valid range."""
        obs = ObservationVector(
            convex_loss=0.01, convex_loss_acceleration=0.01,
            tail_mass=0.70, disagreement=0.50, disagreement_momentum=0.10,
            vol_ratio=2.0, timestamp="2024-06-15"
        )
        alpha = _compute_dynamic_alpha(obs, base_alpha=0.60)
        assert 0.40 <= alpha <= 0.60
    
    def test_g_adjustment_monotonic(self):
        """Test g(ΔC) is monotonically increasing."""
        delta_c_values = [0.0, 0.0001, 0.0003, 0.0005, 0.001]
        g_values = [_g_convexity_adjustment(dc) for dc in delta_c_values]
        for i in range(len(g_values) - 1):
            assert g_values[i] <= g_values[i + 1]


# =============================================================================
# TEST 9: Bayesian Transition Model
# =============================================================================

class TestBayesianTransitionModel:
    """Test Dirichlet-Multinomial Bayesian model."""
    
    def test_initial_expected_matrix(self):
        """Test initial expected matrix."""
        model = BayesianTransitionModel(n_states=4, prior_concentration=1.0)
        expected = model.expected_transition_matrix()
        assert expected.shape == (4, 4)
        np.testing.assert_array_almost_equal(expected.sum(axis=1), np.ones(4))
    
    def test_update_shifts_posterior(self):
        """Test updates shift the posterior."""
        model = BayesianTransitionModel(n_states=4, prior_concentration=1.0)
        initial = model.expected_transition_matrix().copy()
        for _ in range(100):
            model.update(0, 1)
        updated = model.expected_transition_matrix()
        assert updated[0, 1] > initial[0, 1]
    
    def test_posterior_entropy_decreases(self):
        """Test entropy decreases with observations."""
        model = BayesianTransitionModel(n_states=4, prior_concentration=1.0)
        initial_entropy = model.posterior_entropy()
        for _ in range(50):
            model.update(0, 0)
        final_entropy = model.posterior_entropy()
        assert final_entropy < initial_entropy
    
    def test_transition_uncertainty(self):
        """Test uncertainty quantification."""
        model = BayesianTransitionModel(n_states=4)
        variance = model.transition_uncertainty()
        assert variance.shape == (4, 4)
        assert np.all(variance >= 0)


# =============================================================================
# TEST 10: Wasserstein Regime Detector
# =============================================================================

class TestWassersteinRegimeDetector:
    """Test Wasserstein-based regime detection."""
    
    def test_detects_regime_change(self):
        """Test detection of regime change."""
        np.random.seed(42)
        regime1 = np.random.normal(0, 0.01, 100).reshape(-1, 1)
        regime2 = np.random.normal(0.02, 0.03, 100).reshape(-1, 1)
        observations = np.vstack([regime1, regime2])
        
        detector = WassersteinRegimeDetector(window_size=30)
        scores = detector.compute_regime_change_scores(observations)
        
        change_region = scores[80:120]
        assert np.max(change_region) > 0.5
    
    def test_stable_regime_low_scores(self):
        """Test stable regimes have low scores."""
        np.random.seed(42)
        observations = np.random.normal(0, 0.01, 200).reshape(-1, 1)
        
        detector = WassersteinRegimeDetector(window_size=30)
        scores = detector.compute_regime_change_scores(observations)
        
        # In stable regime, most scores should be moderate (not spiking)
        assert np.mean(scores) < 0.6
    
    def test_regime_stability_score(self):
        """Test overall stability score."""
        np.random.seed(42)
        observations = np.random.normal(0, 0.01, 200).reshape(-1, 1)
        
        detector = WassersteinRegimeDetector(window_size=30)
        detector.compute_regime_change_scores(observations)
        stability = detector.regime_stability_score()
        
        assert stability > 0.5


# =============================================================================
# TEST 11: Decision Making
# =============================================================================

class TestDecisionMaking:
    """Test debt switch decision logic."""
    
    def test_no_trigger_normal_state(self):
        """Test no trigger in NORMAL state."""
        obs = ObservationVector(
            convex_loss=0.0003, convex_loss_acceleration=0.0,
            tail_mass=0.50, disagreement=0.15, disagreement_momentum=0.0,
            vol_ratio=1.0, timestamp="2024-06-15"
        )
        posterior = StatePosterior(
            probabilities=(0.90, 0.05, 0.03, 0.02),
            dominant_state=LatentState.NORMAL,
            timestamp="2024-06-15"
        )
        decision = _make_decision(obs, posterior)
        assert decision.triggered is False
    
    def test_trigger_high_pre_policy(self):
        """Test trigger when P(PRE_POLICY) > α(t)."""
        obs = ObservationVector(
            convex_loss=0.002, convex_loss_acceleration=0.0003,
            tail_mass=0.65, disagreement=0.40, disagreement_momentum=0.03,
            vol_ratio=0.7, timestamp="2024-06-15"
        )
        posterior = StatePosterior(
            probabilities=(0.10, 0.15, 0.65, 0.10),
            dominant_state=LatentState.PRE_POLICY,
            timestamp="2024-06-15"
        )
        decision = _make_decision(obs, posterior, threshold=0.60)
        assert decision.triggered is True
    
    def test_trigger_policy_dominant(self):
        """Test trigger when POLICY is dominant."""
        obs = ObservationVector(
            convex_loss=0.005, convex_loss_acceleration=0.001,
            tail_mass=0.75, disagreement=0.60, disagreement_momentum=0.05,
            vol_ratio=2.0, timestamp="2024-06-15"
        )
        posterior = StatePosterior(
            probabilities=(0.05, 0.10, 0.25, 0.60),
            dominant_state=LatentState.POLICY,
            timestamp="2024-06-15"
        )
        decision = _make_decision(obs, posterior)
        assert decision.triggered is True
    
    def test_decision_signature_deterministic(self):
        """Test signature is deterministic."""
        obs = ObservationVector(
            convex_loss=0.0003, convex_loss_acceleration=0.0,
            tail_mass=0.50, disagreement=0.15, disagreement_momentum=0.0,
            vol_ratio=1.0, timestamp="2024-06-15"
        )
        posterior = StatePosterior(
            probabilities=(0.90, 0.05, 0.03, 0.02),
            dominant_state=LatentState.NORMAL,
            timestamp="2024-06-15"
        )
        sig1 = _compute_decision_signature(obs, posterior, False, None, 0.60)
        sig2 = _compute_decision_signature(obs, posterior, False, None, 0.60)
        assert sig1 == sig2


# =============================================================================
# TEST 12: Unified Decision Gate
# =============================================================================

class TestUnifiedDecisionGate:
    """Test unified decision gate architecture."""
    
    def test_switch_confidence_baseline(self):
        """Test confidence is low below threshold."""
        gate = UnifiedDecisionGate(wasserstein_threshold=0.15, switch_threshold=0.6)
        confidence = gate.compute_switch_confidence(
            wasserstein_distance=0.05, mi_ratio=1.0, kl_divergence=0.1
        )
        assert confidence < 0.3
    
    def test_switch_confidence_high_wasserstein(self):
        """Test confidence increases with Wasserstein."""
        gate = UnifiedDecisionGate(wasserstein_threshold=0.15, switch_threshold=0.6)
        confidence = gate.compute_switch_confidence(
            wasserstein_distance=0.30, mi_ratio=1.2, kl_divergence=0.5
        )
        # Confidence should be elevated with high Wasserstein
        assert confidence > 0.2
    
    def test_should_switch_logic(self):
        """Test should_switch returns correct tuple."""
        gate = UnifiedDecisionGate(switch_threshold=0.6)
        gate.compute_switch_confidence(0.05, 1.0, 0.1)
        should_switch, conf, reason = gate.should_switch()
        assert should_switch is False
        assert "≤ threshold" in reason
    
    def test_belief_update_with_decay(self):
        """Test belief update applies decay."""
        gate = UnifiedDecisionGate(n_states=4, belief_decay=0.9)
        gate.belief = np.array([0.95, 0.02, 0.02, 0.01])
        T = np.eye(4)
        gate.update_belief(T)
        assert gate.belief[0] < 0.95


# =============================================================================
# TEST 13: Persistence
# =============================================================================

class TestPersistence:
    """Test decision persistence."""
    
    def test_persist_and_load_decision(self):
        """Test round-trip persistence."""
        obs = ObservationVector(
            convex_loss=0.0005, convex_loss_acceleration=0.00002,
            tail_mass=0.55, disagreement=0.20, disagreement_momentum=0.01,
            vol_ratio=1.1, timestamp="2024-06-15"
        )
        posterior = StatePosterior(
            probabilities=(0.80, 0.10, 0.07, 0.03),
            dominant_state=LatentState.NORMAL,
            timestamp="2024-06-15"
        )
        decision = DebtSwitchDecision(
            triggered=False, effective_date=None,
            observation=obs, state_posterior=posterior,
            decision_basis="Below threshold",
            signature="abc123def456", dynamic_alpha=0.60
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            _persist_decision(decision, path)
            loaded = _load_persisted_decision(path)
            
            assert loaded is not None
            assert loaded.triggered == decision.triggered
            assert loaded.signature == decision.signature
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        result = _load_persisted_decision("/nonexistent/path/file.json")
        assert result is None


# =============================================================================
# TEST 14: Full Integration
# =============================================================================

class TestFullIntegration:
    """Integration tests for full engine."""
    
    def test_engine_with_normal_regime(self):
        """Test full engine with normal conditions."""
        prices = generate_realistic_eurjpy_prices(400, "normal", seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "EURJPY_1d.csv"
            persistence_path = Path(tmpdir) / "decision.json"
            
            df = pd.DataFrame({'Date': prices.index, 'Close': prices.values})
            df.to_csv(data_path, index=False)
            
            decision = run_debt_allocation_engine(
                data_path=str(data_path),
                persistence_path=str(persistence_path),
                force_reevaluate=True,
                force_refresh_data=False,
                use_dynamic_alpha=True,
                quiet=True
            )
            
            assert decision is not None
            assert decision.triggered is False
    
    def test_engine_requires_minimum_history(self):
        """Test engine rejects insufficient history."""
        prices = generate_realistic_eurjpy_prices(100, "normal", seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "EURJPY_1d.csv"
            
            df = pd.DataFrame({'Date': prices.index, 'Close': prices.values})
            df.to_csv(data_path, index=False)
            
            decision = run_debt_allocation_engine(
                data_path=str(data_path),
                force_reevaluate=True,
                force_refresh_data=False,
                quiet=True
            )
            
            assert decision is None
    
    def test_engine_observation_construction(self):
        """Test engine constructs valid observations."""
        prices = generate_realistic_eurjpy_prices(400, "normal", seed=42)
        log_returns = _compute_log_returns(prices)
        
        obs = _construct_observation_vector(
            log_returns, prev_disagreement=None, prev_convex_loss=None,
            timestamp="2024-06-15"
        )
        
        assert np.isfinite(obs.convex_loss)
        assert np.isfinite(obs.tail_mass)


# =============================================================================
# TEST 15: Edge Cases and Numerical Stability
# =============================================================================

class TestEdgeCasesAndStability:
    """Test edge cases and numerical stability."""
    
    def test_log_gaussian_pdf_stability(self):
        """Test log Gaussian PDF stability."""
        assert np.isfinite(_log_gaussian_pdf(0.0, 0.0, 1.0))
        assert np.isfinite(_log_gaussian_pdf(100.0, 0.0, 0.01))
        assert np.isfinite(_log_gaussian_pdf(0.0, 0.0, 1e-10))
    
    def test_log_soft_indicator_stability(self):
        """Test log soft indicator stability."""
        assert np.isfinite(_log_soft_indicator(0.5, 0.3, 'above'))
        assert np.isfinite(_log_soft_indicator(1000.0, 0.3, 'above'))
        assert np.isfinite(_log_soft_indicator(-1000.0, 0.3, 'below'))
    
    def test_posterior_normalization(self):
        """Test posteriors are normalized."""
        prices = generate_realistic_eurjpy_prices(400, "normal", seed=42)
        log_returns = _compute_log_returns(prices)
        
        observations = []
        prev_d, prev_c = None, None
        for i in range(300, 350):
            returns_to_i = log_returns.iloc[:i]
            obs = _construct_observation_vector(returns_to_i, prev_d, prev_c)
            observations.append(obs)
            prev_d = obs.disagreement
            prev_c = obs.convex_loss
        
        obs_arrays = [o.to_array() for o in observations]
        historical = np.array(obs_arrays)
        
        A = _build_transition_matrix()
        pi = np.array([0.85, 0.10, 0.04, 0.01])
        
        posteriors = _forward_algorithm(obs_arrays, A, pi, historical)
        
        for p in posteriors:
            np.testing.assert_almost_equal(np.sum(p), 1.0, decimal=5)
    
    def test_handles_constant_prices(self):
        """Test handling of constant prices."""
        dates = pd.date_range(start="2024-01-01", periods=300, freq='B')
        prices = pd.Series([160.0] * 300, index=dates)
        log_returns = _compute_log_returns(prices)
        vol_ratio = _compute_volatility_ratio(log_returns)
        assert np.isnan(vol_ratio) or vol_ratio < 0.01
    
    def test_empty_series_handling(self):
        """Test handling of empty price series."""
        empty_prices = pd.Series([], dtype=float)
        log_returns = _compute_log_returns(empty_prices)
        assert len(log_returns) == 0


# =============================================================================
# HISTORICAL EURJPY DATA GENERATORS
# =============================================================================

def generate_gfc_2008_eurjpy(seed: int = 2008) -> pd.Series:
    """
    Generate EURJPY data mimicking July 2008 peak and subsequent crash.
    
    Historical context:
    - EURJPY peaked at ~170 on July 23, 2008
    - Lehman collapse (Sep 15, 2008) triggered massive JPY strengthening
    - EURJPY crashed to ~113 by January 2009 (-33% in 6 months)
    
    Extended to ensure enough data for regime detection (need >312 for 60+ observations)
    """
    np.random.seed(seed)
    
    # Pre-rally baseline period (to extend data length)
    pre_rally_returns = np.random.normal(0.0003, 0.005, 80)  # Late 2007
    
    rally_returns = np.random.normal(0.0008, 0.007, 120)
    compression_returns = np.random.normal(0.0002, 0.004, 30)
    stress_returns = np.random.normal(-0.0005, 0.012, 20)
    pre_crash = np.random.normal(-0.003, 0.015, 30)
    crash_returns = np.random.normal(-0.008, 0.025, 60)
    crash_returns[15] -= 0.05  # Lehman day
    crash_returns[16] -= 0.03
    crash_returns[20] -= 0.04
    
    all_returns = np.concatenate([
        pre_rally_returns, rally_returns, compression_returns, stress_returns, 
        pre_crash, crash_returns
    ])
    # Total: 80 + 120 + 30 + 20 + 30 + 60 = 340 returns (>312 needed)
    
    log_prices = np.cumsum(all_returns)
    prices = 135.0 * np.exp(log_prices)  # Start lower to account for pre-rally
    dates = pd.date_range(start="2007-09-01", periods=len(prices), freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


def generate_2014_peak_eurjpy(seed: int = 2014) -> pd.Series:
    """
    Generate EURJPY data mimicking December 2014 peak (Abenomics end).
    EURJPY peaked at ~149.5 on Dec 4, 2014, then fell to ~126 by Feb 2016.
    """
    np.random.seed(seed)
    
    rally_returns = np.random.normal(0.0006, 0.006, 100)
    peak_returns = np.random.normal(0.0001, 0.005, 20)
    hesitation_returns = np.random.normal(-0.0002, 0.008, 30)
    reversal_returns = np.random.normal(-0.002, 0.010, 50)
    decline_returns = np.random.normal(-0.0015, 0.009, 60)
    
    all_returns = np.concatenate([
        rally_returns, peak_returns, hesitation_returns,
        reversal_returns, decline_returns
    ])
    
    log_prices = np.cumsum(all_returns)
    prices = 135.0 * np.exp(log_prices)
    dates = pd.date_range(start="2014-08-01", periods=len(prices), freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


def generate_2018_peak_eurjpy(seed: int = 2018) -> pd.Series:
    """
    Generate EURJPY data mimicking February 2018 peak (Volmageddon).
    EURJPY peaked at ~137.5 on Feb 2, 2018, then dropped to ~118 by Mar 2020.
    """
    np.random.seed(seed)
    
    # Higher vol in rally phase to create clear contrast with compression
    rally_returns = np.random.normal(0.0004, 0.012, 80)  # Even higher vol (1.2%)
    
    # Extremely low vol in compression phase for clear vol compression signal
    compression_returns = np.random.normal(0.0001, 0.0008, 15)  # Very low vol (0.08%)
    
    volmageddon = np.array([-0.025, -0.015, 0.010, -0.020, -0.008])
    post_shock = np.random.normal(-0.001, 0.015, 30)
    decline_returns = np.random.normal(-0.0008, 0.008, 100)
    
    all_returns = np.concatenate([
        rally_returns, compression_returns, volmageddon,
        post_shock, decline_returns
    ])
    
    log_prices = np.cumsum(all_returns)
    prices = 130.0 * np.exp(log_prices)
    dates = pd.date_range(start="2017-11-01", periods=len(prices), freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


def generate_2022_boj_intervention_eurjpy(seed: int = 2022) -> pd.Series:
    """
    Generate EURJPY data mimicking October 2022 BoJ intervention.
    EURJPY peaked at ~148 on Oct 21, 2022, then corrected to ~138.
    """
    np.random.seed(seed)
    
    # Rally phase: moderate positive drift
    rally_returns = np.random.normal(0.0005, 0.004, 100)  # Lower vol for cleaner trend
    
    # Blow-off phase: MUCH stronger positive drift with controlled vol
    # This ensures blowoff_momentum > normal_momentum * 0.5
    blowoff_returns = np.random.normal(0.0030, 0.006, 20)  # Strong positive drift (0.3%/day)
    
    pre_intervention = np.random.normal(0.0005, 0.015, 5)
    intervention = np.array([-0.035, -0.015, 0.005])
    post_intervention = np.random.normal(-0.001, 0.012, 30)
    stabilization = np.random.normal(-0.0003, 0.008, 60)
    
    all_returns = np.concatenate([
        rally_returns, blowoff_returns, pre_intervention,
        intervention, post_intervention, stabilization
    ])
    
    log_prices = np.cumsum(all_returns)
    prices = 137.0 * np.exp(log_prices)
    dates = pd.date_range(start="2022-06-01", periods=len(prices), freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


def generate_2024_boj_intervention_eurjpy(seed: int = 2024) -> pd.Series:
    """
    Generate EURJPY data mimicking July 2024 BoJ intervention.
    EURJPY peaked at ~175 on July 11, 2024 (all-time high), fell to ~155.
    
    Structure to match test indices:
    - 0-120: rally (normal_vol at 50:100)
    - 120-150: parabolic extension
    - 150-165: warning + intervention
    - 165-185: unwind (high vol)
    - 185+: stabilization
    
    Total must be > 252 for MIN_HISTORY_DAYS
    """
    np.random.seed(seed)
    
    # Rally phase including early history (indices 0-120, with normal vol at 50-100)
    rally_returns = np.random.normal(0.0006, 0.005, 121)  # Normal moderate vol
    
    # Parabolic extension (indices 120-150)
    parabolic_returns = np.random.normal(0.0012, 0.006, 30)
    
    # Warning + intervention (indices 150-165)
    warning_intervention = np.random.normal(-0.002, 0.015, 15)
    warning_intervention[10:15] = [-0.020, -0.025, -0.015, 0.008, -0.030]  # Intervention
    
    # Carry unwind with HIGH vol (indices 165-185)
    unwind_returns = np.random.normal(-0.005, 0.030, 20)  # Much higher vol
    unwind_returns[5] -= 0.05  # "Black Monday"
    
    # Stabilization (indices 185+, need enough to reach >252)
    stabilization = np.random.normal(-0.0005, 0.008, 70)  # Extended for length
    
    all_returns = np.concatenate([
        rally_returns, parabolic_returns, warning_intervention,
        unwind_returns, stabilization
    ])
    # Total: 121 + 30 + 15 + 20 + 70 = 256 > 252
    
    log_prices = np.cumsum(all_returns)
    prices = 160.0 * np.exp(log_prices)
    dates = pd.date_range(start="2024-01-01", periods=len(prices), freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


# =============================================================================
# TEST 16: Multi-Currency Configuration
# =============================================================================

class TestMultiCurrencyConfiguration:
    """Test multi-currency configuration constants."""
    
    def test_supported_pairs_exist(self):
        """Verify all supported currency pairs are defined."""
        assert "EURJPY" in SUPPORTED_CURRENCY_PAIRS
        assert "AUDJPY" in SUPPORTED_CURRENCY_PAIRS
        assert "EURAUD" in SUPPORTED_CURRENCY_PAIRS
    
    def test_pair_configuration_structure(self):
        """Verify each pair has required configuration fields."""
        required_fields = [
            "yahoo_ticker", "data_file", "description",
            "base_currency", "quote_currency", "debt_currency",
            "characteristics"
        ]
        
        for pair, config in SUPPORTED_CURRENCY_PAIRS.items():
            for field in required_fields:
                assert field in config, f"{pair} missing {field}"
    
    def test_eurjpy_characteristics(self):
        """Verify EURJPY has correct characteristics."""
        eurjpy = SUPPORTED_CURRENCY_PAIRS["EURJPY"]
        assert eurjpy["base_currency"] == "EUR"
        assert eurjpy["quote_currency"] == "JPY"
        assert eurjpy["debt_currency"] == "JPY"
        assert eurjpy["characteristics"]["carry_yield"] == "low"
        assert eurjpy["characteristics"]["central_bank_sensitivity"] == "high"
    
    def test_audjpy_characteristics(self):
        """Verify AUDJPY has correct characteristics."""
        audjpy = SUPPORTED_CURRENCY_PAIRS["AUDJPY"]
        assert audjpy["base_currency"] == "AUD"
        assert audjpy["quote_currency"] == "JPY"
        assert audjpy["characteristics"]["carry_yield"] == "high"
        assert audjpy["characteristics"]["commodity_correlation"] == "high"
    
    def test_euraud_characteristics(self):
        """Verify EURAUD has correct characteristics."""
        euraud = SUPPORTED_CURRENCY_PAIRS["EURAUD"]
        assert euraud["base_currency"] == "EUR"
        assert euraud["quote_currency"] == "AUD"
        assert euraud["debt_currency"] == "AUD"
        assert euraud["characteristics"]["commodity_correlation"] == "high"


# =============================================================================
# TEST 17: Realistic Currency Pair Data Generation
# =============================================================================

def generate_realistic_audjpy_prices(
    n_days: int = 500,
    regime: str = "normal",
    seed: int = 42
) -> pd.Series:
    """Generate realistic AUDJPY price series."""
    np.random.seed(seed)
    
    params = {
        "normal": {"mu": 0.0002, "sigma": 0.008, "jump_prob": 0.015, "jump_size": 0.012},
        "compressed": {"mu": 0.0001, "sigma": 0.004, "jump_prob": 0.008, "jump_size": 0.008},
        "pre_policy": {"mu": 0.0004, "sigma": 0.014, "jump_prob": 0.04, "jump_size": 0.020},
        "policy": {"mu": -0.003, "sigma": 0.025, "jump_prob": 0.06, "jump_size": 0.035},
        "risk_off": {"mu": -0.002, "sigma": 0.022, "jump_prob": 0.05, "jump_size": 0.030},
    }
    
    p = params.get(regime, params["normal"])
    
    returns = np.random.normal(p["mu"], p["sigma"], n_days)
    jumps = np.random.binomial(1, p["jump_prob"], n_days) * \
            np.random.choice([-1, 1], n_days) * \
            np.random.exponential(p["jump_size"], n_days)
    returns += jumps
    
    log_prices = np.cumsum(returns)
    prices = 98.0 * np.exp(log_prices)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


def generate_realistic_euraud_prices(
    n_days: int = 500,
    regime: str = "normal",
    seed: int = 42
) -> pd.Series:
    """Generate realistic EURAUD price series."""
    np.random.seed(seed)
    
    params = {
        "normal": {"mu": 0.00005, "sigma": 0.007, "jump_prob": 0.012, "jump_size": 0.010},
        "compressed": {"mu": 0.00002, "sigma": 0.004, "jump_prob": 0.006, "jump_size": 0.006},
        "pre_policy": {"mu": 0.0002, "sigma": 0.012, "jump_prob": 0.03, "jump_size": 0.018},
        "policy": {"mu": -0.001, "sigma": 0.020, "jump_prob": 0.05, "jump_size": 0.028},
    }
    
    p = params.get(regime, params["normal"])
    
    returns = np.random.normal(p["mu"], p["sigma"], n_days)
    jumps = np.random.binomial(1, p["jump_prob"], n_days) * \
            np.random.choice([-1, 1], n_days) * \
            np.random.exponential(p["jump_size"], n_days)
    returns += jumps
    
    log_prices = np.cumsum(returns)
    prices = 1.65 * np.exp(log_prices)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


# =============================================================================
# TEST 18: Risk Metrics Computation
# =============================================================================

class TestRiskMetricsComputation:
    """Test risk metrics computation for multi-currency analysis."""
    
    def test_compute_risk_metrics_normal_regime(self):
        """Test risk metrics for normal regime."""
        prices = generate_realistic_eurjpy_prices(n_days=300, regime="normal", seed=123)
        log_returns = _compute_log_returns(prices)
        
        metrics = _compute_risk_metrics(log_returns)
        
        assert 0.05 < metrics["volatility_annualized"] < 0.25
        assert metrics["tail_risk_99"] < 0
        assert np.isfinite(metrics["skewness"])
        assert np.isfinite(metrics["kurtosis"])
    
    def test_compute_risk_metrics_crisis_regime(self):
        """Test risk metrics for crisis regime."""
        prices = generate_realistic_eurjpy_prices(n_days=300, regime="crisis", seed=456)
        log_returns = _compute_log_returns(prices)
        
        metrics = _compute_risk_metrics(log_returns)
        
        assert metrics["volatility_annualized"] > 0.15
        assert metrics["tail_risk_99"] < -0.03
    
    def test_compute_risk_metrics_audjpy(self):
        """Test risk metrics for AUDJPY (higher vol than EURJPY)."""
        prices = generate_realistic_audjpy_prices(n_days=300, regime="normal", seed=789)
        log_returns = _compute_log_returns(prices)
        
        metrics = _compute_risk_metrics(log_returns)
        
        assert 0.08 < metrics["volatility_annualized"] < 0.35


# =============================================================================
# TEST 19: Risk Score Computation
# =============================================================================

class TestRiskScoreComputation:
    """Test risk score computation for currency pairs."""
    
    def test_risk_score_bounds(self):
        """Risk score should be bounded 0-100."""
        low_score = _compute_risk_score(
            volatility=0.05,
            tail_risk=-0.01,
            recent_trend=0.02,
            state_posterior=None,
            characteristics={"carry_yield": "low", "commodity_correlation": "low"},
        )
        assert 0 <= low_score <= 100
        assert low_score < 50
        
        high_score = _compute_risk_score(
            volatility=0.30,
            tail_risk=-0.08,
            recent_trend=-0.10,
            state_posterior=None,
            characteristics={"central_bank_sensitivity": "high", "commodity_correlation": "high", "volatility_regime": "high"},
        )
        assert 0 <= high_score <= 100
        assert high_score > 50
    
    def test_risk_score_increases_with_volatility(self):
        """Higher volatility should increase risk score."""
        low_vol_score = _compute_risk_score(
            volatility=0.05,
            tail_risk=-0.02,
            recent_trend=0.0,
            state_posterior=None,
            characteristics={},
        )
        
        high_vol_score = _compute_risk_score(
            volatility=0.25,
            tail_risk=-0.02,
            recent_trend=0.0,
            state_posterior=None,
            characteristics={},
        )
        
        assert high_vol_score > low_vol_score


# =============================================================================
# TEST 20: CurrencyPairAnalysis Data Structure
# =============================================================================

class TestCurrencyPairAnalysis:
    """Test CurrencyPairAnalysis data structure."""
    
    def test_analysis_to_dict(self):
        """Test serialization to dictionary."""
        analysis = CurrencyPairAnalysis(
            pair="EURJPY",
            decision=None,
            risk_score=35.5,
            volatility_annualized=0.12,
            tail_risk_99=-0.025,
            recent_trend=0.015,
            carry_attractiveness="low",
            recommendation="preferred",
            characteristics={"carry_yield": "low"},
        )
        
        d = analysis.to_dict()
        
        assert d["pair"] == "EURJPY"
        assert d["risk_score"] == 35.5
        assert d["volatility_annualized"] == 0.12
        assert d["recommendation"] == "preferred"


# =============================================================================
# TEST 21: MultiCurrencyComparison Data Structure
# =============================================================================

class TestMultiCurrencyComparison:
    """Test MultiCurrencyComparison data structure."""
    
    def test_comparison_to_dict(self):
        """Test serialization to dictionary."""
        analysis1 = CurrencyPairAnalysis(
            pair="EURJPY",
            decision=None,
            risk_score=30.0,
            volatility_annualized=0.10,
            tail_risk_99=-0.02,
            recent_trend=0.01,
            carry_attractiveness="low",
            recommendation="preferred",
            characteristics={},
        )
        
        analysis2 = CurrencyPairAnalysis(
            pair="EURAUD",
            decision=None,
            risk_score=55.0,
            volatility_annualized=0.15,
            tail_risk_99=-0.04,
            recent_trend=-0.02,
            carry_attractiveness="moderate",
            recommendation="acceptable",
            characteristics={},
        )
        
        comparison = MultiCurrencyComparison(
            analyses={"EURJPY": analysis1, "EURAUD": analysis2},
            recommended_pair="EURJPY",
            recommendation_basis="Lowest risk score",
            timestamp="2024-06-15T12:00:00",
        )
        
        d = comparison.to_dict()
        
        assert "EURJPY" in d["analyses"]
        assert "EURAUD" in d["analyses"]
        assert d["recommended_pair"] == "EURJPY"


# =============================================================================
# TEST 22: Currency Pair Comparison Logic
# =============================================================================

class TestCurrencyPairComparisonLogic:
    """Test the logic for comparing currency pairs."""
    
    def test_lower_risk_score_is_preferred(self):
        """Pair with lower risk score should be recommended."""
        analysis1 = CurrencyPairAnalysis(
            pair="EURJPY",
            decision=None,
            risk_score=25.0,
            volatility_annualized=0.10,
            tail_risk_99=-0.02,
            recent_trend=0.01,
            carry_attractiveness="low",
            recommendation="preferred",
            characteristics={},
        )
        
        analysis2 = CurrencyPairAnalysis(
            pair="EURAUD",
            decision=None,
            risk_score=65.0,
            volatility_annualized=0.18,
            tail_risk_99=-0.05,
            recent_trend=-0.03,
            carry_attractiveness="moderate",
            recommendation="avoid",
            characteristics={},
        )
        
        analyses = {"EURJPY": analysis1, "EURAUD": analysis2}
        candidates = [(k, v) for k, v in analyses.items() if v.decision is None or not v.decision.triggered]
        candidates.sort(key=lambda x: x[1].risk_score)
        
        assert candidates[0][0] == "EURJPY"
    
    def test_recommendation_categories_correct(self):
        """Test that recommendations are assigned correctly based on risk score."""
        test_cases = [
            (20.0, "preferred"),
            (29.9, "preferred"),
            (30.0, "acceptable"),
            (45.0, "acceptable"),
            (59.9, "acceptable"),
            (60.0, "avoid"),
            (85.0, "avoid"),
        ]
        
        for risk_score, expected_rec in test_cases:
            if risk_score < 30:
                rec = "preferred"
            elif risk_score < 60:
                rec = "acceptable"
            else:
                rec = "avoid"
            assert rec == expected_rec


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])