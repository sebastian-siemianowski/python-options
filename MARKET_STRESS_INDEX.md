# Market Stress Index - Custom Indicator for 2022-Style Market Protection

## Overview

The Market Stress Index is a custom mathematical indicator designed to detect challenging market regimes like 2022 and prevent poor trading performance during such periods. This indicator combines multiple mathematical factors to identify high-stress market conditions and automatically filter out trade entries when conditions become unfavorable.

## Problem Statement

In 2022, market performance was significantly challenged due to:
- Elevated and persistent volatility
- Prolonged drawdowns and bear market conditions  
- Conflicting momentum signals across timeframes
- Correlation breakdowns between traditional indicators
- Liquidity stress during market sell-offs

The existing trend and volatility filters in the system, while helpful, were not comprehensive enough to fully protect against these complex market conditions.

## Solution: Market Stress Index

### Mathematical Components

The Market Stress Index combines five key stress factors:

#### 1. Volatility Regime Stress (Weight: 30%)
- **Absolute Volatility Levels**: Detects when 30-day realized volatility exceeds 20-25% (typical of stressed markets)
- **Volatility Regime Structure**: Identifies when short-term volatility significantly exceeds medium and long-term volatility
- **Enhanced Sensitivity**: Lowered thresholds compared to basic volatility filters for earlier detection

```python
# Absolute volatility stress
vol_level_stress = np.where(rv30 > 0.25, 0.7, np.where(rv30 > 0.20, 0.4, 0.1))

# Regime-based volatility stress  
vol_regime_stress = np.where(
    (rv10 > rv30 * 1.3) & (rv30 > rv60 * 1.15),  # More sensitive thresholds
    0.8,  # High stress
    0.6 if medium_conditions else 0.2  # Medium/low stress
)
```

#### 2. Correlation Breakdown Stress (Weight: 15%)
- Detects when traditional price-volume correlations break down
- Indicates market structure deterioration
- 20-period rolling correlation analysis

#### 3. Drawdown Stress (Weight: 25%)
- **Critical Component for 2022-style conditions**
- Measures persistent drawdowns from recent highs
- Triggers high stress when drawdowns exceed 10-20%
- Uses 60-day (3-month) rolling maximum as reference

```python
rolling_max = px.rolling(60).max()
drawdown = (px - rolling_max) / rolling_max
drawdown_stress = np.where(
    drawdown < -0.20,  # >20% drawdown = high stress
    0.8,
    0.5 if drawdown < -0.10 else 0.1  # >10% = medium stress
)
```

#### 4. Momentum Stress (Weight: 20%)
- Detects persistent negative momentum across timeframes
- Identifies conflicting momentum signals
- Enhanced to focus on sustained declining trends

#### 5. Liquidity Stress (Weight: 10%)
- Detects volume spikes coinciding with negative returns
- Indicates panic selling or liquidity crises
- Based on volume exceeding 20-day moving average

### Combined Index Calculation

```python
market_stress = (
    0.30 * vol_stress +      # Volatility regime
    0.15 * corr_stress +     # Correlation breakdown  
    0.25 * drawdown_stress + # Drawdown (key for 2022)
    0.20 * momentum_stress + # Momentum divergence
    0.10 * liquidity_stress  # Liquidity stress
)
```

The index is smoothed using a 5-day rolling average to reduce noise and provide stable signals.

## Implementation Details

### Integration Points

1. **Function**: `calculate_market_stress_index(hist)`
   - Input: Historical price data DataFrame
   - Output: Pandas Series with stress scores (0.0 to 1.0)

2. **Backtest Integration**: Added to `backtest_breakout_option_strategy()`
   - Calculates stress index if filter is enabled
   - Applies filter logic during trade entry evaluation

3. **Filter Logic**: Applied after existing trend/volatility filters
```python
if market_stress_filter and 'market_stress' in df.columns:
    stress_score = df['market_stress'].iloc[i]
    if not np.isnan(stress_score) and stress_score > market_stress_threshold:
        # Skip trade entry - market stress too high
        continue
```

### Command Line Parameters

- `--bt_market_stress_filter`: Enable/disable the filter (default: false)
- `--bt_market_stress_threshold`: Stress threshold (default: 0.4)

### Usage Examples

```bash
# Enable market stress filter with default threshold (0.4)
python options.py --bt_market_stress_filter true

# Enable with custom threshold (more restrictive)
python options.py --bt_market_stress_filter true --bt_market_stress_threshold 0.3

# Enable with less restrictive threshold
python options.py --bt_market_stress_filter true --bt_market_stress_threshold 0.5
```

## Performance Analysis

### 2022 Market Conditions
Based on historical analysis of SPY:
- **2022 Average Stress**: 0.295 (elevated)
- **2022 Maximum Stress**: 0.545 (significant stress periods)
- **Comparison**: Higher than 2021 (0.165) and 2023 (0.176)

### Threshold Recommendations
- **Conservative (0.3)**: Filters more aggressively, fewer trades but higher protection
- **Balanced (0.4)**: Default setting, good balance of protection vs. opportunity  
- **Moderate (0.5)**: Allows more trades while still filtering extreme conditions

### Expected Benefits
1. **Reduced Drawdowns**: Avoids entries during prolonged market declines
2. **Improved Risk Management**: Complements existing filters for comprehensive protection
3. **Adaptive**: Responds to multiple market stress factors simultaneously
4. **Configurable**: Threshold can be adjusted based on risk tolerance

## Technical Notes

### Dependencies
- Requires pandas, numpy (standard dependencies)
- Integrates with existing yfinance data loading
- Compatible with current backtesting framework

### Performance Impact
- Minimal computational overhead
- Calculated once per backtest run if enabled
- Smoothed with simple rolling average for efficiency

### Limitations
1. **Lookback Requirement**: Needs ~60 days of data for full effectiveness
2. **Parameter Sensitivity**: Threshold may need adjustment for different assets/markets
3. **Not Predictive**: Reactive indicator based on recent conditions

## Future Enhancements

Potential improvements for future versions:
1. **Asset-Specific Calibration**: Adjust thresholds by asset class
2. **Regime Detection**: Add bull/bear market regime awareness
3. **Volatility Surface Integration**: Incorporate options-based stress measures
4. **Machine Learning**: Use historical data to optimize component weights

## Testing and Validation

The implementation includes a comprehensive test suite (`test_market_stress.py`) that:
- Validates mathematical calculations
- Confirms integration with backtesting system
- Analyzes historical performance across different market periods
- Ensures proper threshold behavior

Run tests with:
```bash
python test_market_stress.py
```

## Conclusion

The Market Stress Index provides a sophisticated, multi-factor approach to detecting challenging market conditions like those experienced in 2022. By combining volatility analysis, drawdown detection, momentum assessment, and liquidity stress measures, it offers comprehensive protection against complex market regimes that simple filters might miss.

The indicator is designed to be:
- **Mathematically Sound**: Based on proven market stress concepts
- **Practical**: Easy to configure and integrate
- **Effective**: Specifically tuned to detect 2022-style conditions
- **Flexible**: Adjustable thresholds for different risk tolerances

This implementation directly addresses the issue description's request for a custom indicator to prevent years like 2022 from causing poor performance, providing traders with an advanced tool for navigating challenging market environments.