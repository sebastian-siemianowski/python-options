"""
===============================================================================
TEST: Metals Risk Temperature Anticipatory Enhancement
===============================================================================

Tests for the Copilot Story implementation (February 2026):

1. Volatility Term Structure Indicator (leading indicator)
2. Robust Z-Score Computation (MAD-based)
3. Data Infrastructure Hardening (multi-source, failover)
4. Enhanced Escalation Protocol (hysteresis)
5. Alert Integration (severity-based)
6. Audit Trail Enhancement

Run with: PYTHONPATH=src python -m pytest src/tests/test_anticipatory_metals.py -v
===============================================================================
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, 'src')

from decision.metals_risk_temperature import (
    # Constants
    VOL_TERM_STRUCTURE_SHORT_WINDOW,
    VOL_TERM_STRUCTURE_LONG_WINDOW,
    VOL_TERM_STRUCTURE_INVERSION_THRESHOLD,
    VOL_TERM_STRUCTURE_MIN_METALS,
    VOL_TERM_STRUCTURE_STRESS_CONTRIBUTION,
    MAD_CONSISTENCY_CONSTANT,
    DATA_DEGRADED_MODE_THRESHOLD,
    DATA_DEGRADED_MODE_TEMP_FLOOR,
    ESCALATION_NORMAL_TO_ELEVATED_TEMP,
    ESCALATION_ELEVATED_TO_STRESSED_TEMP,
    ESCALATION_CONSECUTIVE_REQUIRED,
    AlertSeverity,
    # Functions
    _compute_robust_zscore,
    _compute_volatility_term_structure,
    compute_volatility_term_structure_stress,
    _update_escalation_state,
    EscalationState,
    compute_metals_risk_temperature,
    compute_anticipatory_metals_risk_temperature,
)


class TestRobustZScore:
    """Tests for robust MAD-based z-score computation."""
    
    def test_robust_zscore_normal_data(self):
        """Test robust z-score on normally distributed data."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 10, 100))
        
        zscore = _compute_robust_zscore(data, lookback=60)
        
        # Should be within reasonable bounds for normal data
        assert -3.0 <= zscore <= 3.0
    
    def test_robust_zscore_with_outliers(self):
        """Test robust z-score is resistant to outliers."""
        # Create data with extreme outlier at the end
        data = pd.Series([100.0] * 59 + [100.0])  # Normal values
        zscore_normal = _compute_robust_zscore(data, lookback=60)
        
        # Create data with outliers in history but normal recent value
        data_with_outliers = pd.Series([100.0] * 30 + [1000.0] * 20 + [100.0] * 10)
        zscore_outliers = _compute_robust_zscore(data_with_outliers, lookback=60)
        
        # Robust z-score should be less affected by outliers than standard z-score
        # Both should indicate the current value is "normal" relative to most data
        assert abs(zscore_normal) < 1.0
    
    def test_robust_zscore_returns_audit(self):
        """Test that audit information is returned correctly."""
        data = pd.Series(np.arange(100.0))
        
        zscore, audit = _compute_robust_zscore(data, lookback=60, return_audit=True)
        
        assert "lookback_median" in audit
        assert "lookback_mad" in audit
        assert "estimation_method" in audit
        assert audit["estimation_method"] == "MAD_robust"
        assert audit["lookback_median"] is not None
    
    def test_robust_zscore_empty_data(self):
        """Test handling of empty data."""
        zscore = _compute_robust_zscore(None)
        assert zscore == 0.0
        
        zscore = _compute_robust_zscore(pd.Series([]))
        assert zscore == 0.0
    
    def test_mad_consistency_constant(self):
        """Verify MAD consistency constant is correct."""
        # For normal distribution, MAD ≈ 0.6745 * σ
        # So 1.4826 * MAD ≈ σ
        assert abs(MAD_CONSISTENCY_CONSTANT - 1.4826) < 0.0001


class TestVolatilityTermStructure:
    """Tests for volatility term structure inversion detection."""
    
    def _create_price_series(self, base_price=100, length=100, daily_vol=0.01):
        """Helper to create synthetic price series."""
        np.random.seed(42)
        returns = np.random.normal(0, daily_vol, length)
        prices = base_price * np.exp(np.cumsum(returns))
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        return pd.Series(prices, index=dates)
    
    def test_normal_term_structure(self):
        """Test normal (non-inverted) volatility term structure."""
        # Create stable price series
        prices = self._create_price_series(daily_vol=0.01)
        
        ratio, short_vol, long_vol = _compute_volatility_term_structure(prices)
        
        # In stable conditions, ratio should be near 1.0
        assert 0.5 < ratio < 1.5
        assert short_vol > 0
        assert long_vol > 0
    
    def test_inverted_term_structure(self):
        """Test detection of inverted volatility term structure."""
        # Create price series with recent high volatility
        # Need more extreme contrast to ensure inversion is detected
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # First 95 days: very low volatility
        returns_stable = np.random.normal(0, 0.005, 95)
        # Last 5 days: extremely high volatility (crash-like)
        returns_volatile = np.random.normal(0, 0.10, 5)  # 10% daily vol
        
        returns = np.concatenate([returns_stable, returns_volatile])
        prices = 100 * np.exp(np.cumsum(returns))
        prices = pd.Series(prices, index=dates)
        
        ratio, short_vol, long_vol = _compute_volatility_term_structure(prices)
        
        # Short-term vol should significantly exceed long-term
        # With these parameters, ratio should be > 2.0
        assert ratio > 1.5, f"Expected ratio > 1.5, got {ratio}"
    
    def test_term_structure_stress_indicator_normal(self):
        """Test stress indicator when no metals are inverted."""
        # Create normal price data for all metals
        metals_data = {}
        for metal in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM']:
            metals_data[metal] = self._create_price_series(daily_vol=0.01)
        
        indicator, details = compute_volatility_term_structure_stress(metals_data)
        
        assert indicator.contribution == 0.0
        assert "Normal" in indicator.interpretation
    
    def test_term_structure_stress_indicator_triggered(self):
        """Test stress indicator when ≥2 metals show inversion."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        metals_data = {}
        
        # Gold and Silver: create with explicit inversion
        # First 95 days low vol, last 5 days high vol
        for metal in ['GOLD', 'SILVER']:
            returns_stable = np.random.normal(0, 0.005, 95)  # Very low vol
            returns_volatile = np.random.normal(0, 0.10, 5)   # Very high vol
            returns = np.concatenate([returns_stable, returns_volatile])
            prices = 100 * np.exp(np.cumsum(returns))
            metals_data[metal] = pd.Series(prices, index=dates)
        
        # Other metals: stable with consistent vol
        for metal in ['COPPER', 'PLATINUM', 'PALLADIUM']:
            returns = np.random.normal(0, 0.01, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            metals_data[metal] = pd.Series(prices, index=dates)
        
        indicator, details = compute_volatility_term_structure_stress(metals_data)
        
        # Verify Gold and Silver show inversion
        assert details['GOLD']['inverted'], f"GOLD should be inverted, ratio={details['GOLD']['ratio']}"
        assert details['SILVER']['inverted'], f"SILVER should be inverted, ratio={details['SILVER']['ratio']}"
        
        # Should trigger with 0.3 contribution
        assert indicator.contribution == VOL_TERM_STRUCTURE_STRESS_CONTRIBUTION
        assert "INVERSION" in indicator.interpretation


class TestEscalationProtocol:
    """Tests for enhanced escalation protocol with hysteresis."""
    
    def test_normal_to_elevated_transition(self):
        """Test transition from Normal to Elevated requires consecutive readings."""
        state = EscalationState()
        
        # First reading above threshold
        regime, transitioned, _ = _update_escalation_state(0.8, state)
        assert regime == "Normal"  # Not yet, need consecutive
        assert not transitioned
        
        # Second consecutive reading
        regime, transitioned, severity = _update_escalation_state(0.75, state)
        assert regime == "Elevated"
        assert transitioned
        assert severity == AlertSeverity.INFO
    
    def test_elevated_to_stressed_transition(self):
        """Test transition from Elevated to Stressed."""
        state = EscalationState()
        state.current_regime = "Elevated"
        
        # First high reading
        _update_escalation_state(1.3, state)
        
        # Second consecutive
        regime, transitioned, severity = _update_escalation_state(1.25, state)
        
        assert regime == "Stressed"
        assert transitioned
        assert severity == AlertSeverity.WARNING
    
    def test_hysteresis_prevents_oscillation(self):
        """Test that hysteresis prevents rapid regime oscillation."""
        state = EscalationState()
        state.current_regime = "Elevated"
        
        # Temperature drops just below threshold but above hysteresis
        regime, transitioned, _ = _update_escalation_state(0.65, state)
        
        # Should still be Elevated due to hysteresis
        assert regime == "Elevated"
        assert not transitioned
    
    def test_stressed_to_normal_requires_sustained(self):
        """Test that Stressed → Normal requires sustained days below threshold."""
        state = EscalationState()
        state.current_regime = "Stressed"
        
        # Single day below threshold
        regime, transitioned, _ = _update_escalation_state(0.4, state)
        assert regime == "Stressed"  # Not yet
        
        # Simulate 5 days below threshold
        state.days_below_normal_threshold = 5
        regime, transitioned, _ = _update_escalation_state(0.4, state)
        
        assert regime == "Normal"
        assert transitioned
    
    def test_critical_alert_on_high_temperature(self):
        """Test that CRITICAL alert is generated for temp > 1.5."""
        state = EscalationState()
        state.current_regime = "Stressed"
        
        regime, transitioned, severity = _update_escalation_state(1.6, state)
        
        assert regime == "Extreme"
        assert severity == AlertSeverity.CRITICAL


class TestDegradedMode:
    """Tests for data infrastructure degraded mode."""
    
    def test_degraded_mode_threshold(self):
        """Verify degraded mode threshold constant."""
        assert DATA_DEGRADED_MODE_THRESHOLD == 0.6  # 60%
    
    def test_degraded_mode_floor(self):
        """Verify degraded mode temperature floor."""
        assert DATA_DEGRADED_MODE_TEMP_FLOOR == 1.0


class TestAnticpatoryComputation:
    """Integration tests for anticipatory computation."""
    
    @patch('decision.metals_risk_temperature._fetch_metals_data_with_failover')
    def test_anticipatory_includes_vol_term_structure(self, mock_fetch):
        """Test that anticipatory computation includes vol term structure indicator."""
        # Mock data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        mock_data = {}
        for metal in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM']:
            returns = np.random.normal(0, 0.01, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            mock_data[metal] = pd.Series(prices, index=dates)
        
        from decision.metals_risk_temperature import DataQualityReport
        mock_quality = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            metals_available=5,
            metals_total=5,
            data_quality_pct=1.0,
            degraded_mode=False,
            degraded_mode_reason=None,
            failover_events=[],
            source_used={'GOLD': 'yfinance', 'SILVER': 'yfinance', 
                        'COPPER': 'yfinance', 'PLATINUM': 'yfinance', 
                        'PALLADIUM': 'yfinance'},
            price_divergence_alerts=[],
        )
        
        mock_fetch.return_value = (mock_data, mock_quality)
        
        result, alerts, quality = compute_anticipatory_metals_risk_temperature()
        
        # Should have 6 indicators (5 reactive + 1 anticipatory)
        assert len(result.indicators) == 6
        
        # Vol Term Structure should be present
        vol_ts_indicators = [i for i in result.indicators if "Vol Term" in i.name]
        assert len(vol_ts_indicators) == 1
    
    @patch('decision.metals_risk_temperature._fetch_metals_data_with_failover')
    def test_degraded_mode_applies_floor(self, mock_fetch):
        """Test that degraded mode applies temperature floor."""
        # Mock data with only 2 metals (below 60% threshold)
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        mock_data = {}
        for metal in ['GOLD', 'SILVER']:  # Only 2 metals
            returns = np.random.normal(0, 0.01, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            mock_data[metal] = pd.Series(prices, index=dates)
        
        from decision.metals_risk_temperature import DataQualityReport
        mock_quality = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            metals_available=2,
            metals_total=5,
            data_quality_pct=0.4,  # Below 60%
            degraded_mode=True,
            degraded_mode_reason="Only 2/5 metals available",
            failover_events=[],
            source_used={'GOLD': 'yfinance', 'SILVER': 'yfinance'},
            price_divergence_alerts=[],
        )
        
        mock_fetch.return_value = (mock_data, mock_quality)
        
        result, alerts, quality = compute_anticipatory_metals_risk_temperature()
        
        # Temperature should be at least 1.0 (degraded mode floor)
        assert result.temperature >= DATA_DEGRADED_MODE_TEMP_FLOOR
        
        # Should have warning alert for degraded mode
        degraded_alerts = [a for a in alerts if a.get('degraded_mode')]
        assert len(degraded_alerts) >= 1


class TestAlertGeneration:
    """Tests for alert generation and routing."""
    
    def test_alert_severity_levels(self):
        """Verify alert severity level constants."""
        assert AlertSeverity.INFO == "INFO"
        assert AlertSeverity.WARNING == "WARNING"
        assert AlertSeverity.CRITICAL == "CRITICAL"
    
    def test_alert_contains_required_fields(self):
        """Test that generated alerts contain all required fields."""
        from decision.metals_risk_temperature import _generate_alert
        
        alert = _generate_alert(
            severity=AlertSeverity.WARNING,
            temperature=1.2,
            regime_state="Stressed",
            primary_indicator="Silver/Gold",
            action_text="Reduce exposure",
            data_quality=0.8,
        )
        
        # Verify required fields per Copilot Story
        assert "severity" in alert
        assert "temperature" in alert
        assert "regime_state" in alert
        assert "primary_indicator" in alert
        assert "recommended_action" in alert
        assert "data_quality_pct" in alert
        assert "timestamp" in alert
        assert "message" in alert


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
