# PIT Calibration Verification Guide

## Overview

**PIT (Probability Integral Transform) calibration** is the single most important test for probabilistic forecasts in quantitative trading systems. It verifies that your predicted probabilities are **truth-telling machines** — a Level-7 requirement for world-class quant systems.

## Why PIT Calibration Matters

### The Problem

When your model predicts **P↑ = 0.62** (62% chance of positive return), does that mean:
- ✅ **Well-calibrated**: 62% of similar predictions actually result in positive outcomes
- ❌ **Overconfident**: Only 55% actually positive (predictions too high → over-bet → ruin risk)
- ❌ **Underconfident**: 70% actually positive (predictions too low → under-bet → missed opportunities)

### Why This Is Critical

**Kelly sizing assumes calibrated probabilities.** If your probabilities are wrong:
- **Overconfident** → bet too large → blow up risk
- **Underconfident** → bet too small → opportunity cost

PIT calibration turns your model from a black box into a **falsifiable, truth-telling machine**.

---

## Usage

### Command-Line Interface

Enable PIT calibration verification with the `--pit-calibration` flag:

```bash
# Run calibration test on PLNJPY
python scripts/signals.py --assets="PLNJPY=X" --pit-calibration

# Or use make target with calibration
make fx-plnjpy EXTRA_ARGS="--pit-calibration"
```

**Note**: PIT calibration is **expensive** (requires walk-forward out-of-sample testing with 500+ predictions). Expect runtime of 5-10 minutes per asset.

### Programmatic Usage

```python
from pit_calibration import run_pit_calibration_test
import pandas as pd

# Load price series
px = pd.Series(...)  # Your price data

# Run calibration test
calibration_results = run_pit_calibration_test(
    px=px,
    horizons=[1, 21, 63],      # Test 1-day, 1-month, 3-month forecasts
    n_bins=10,                  # 10 bins for reliability diagram
    train_days=504,             # ~2 years training window
    test_days=21,               # ~1 month test window
    max_predictions=500         # Collect up to 500 out-of-sample predictions
)

# Check results
for horizon, metrics in calibration_results.items():
    print(f"Horizon {horizon}:")
    print(f"  ECE: {metrics.expected_calibration_error:.4f}")
    print(f"  Calibrated: {metrics.calibrated}")
    print(f"  Diagnosis: {metrics.calibration_diagnosis}")
```

---

## Interpreting Results

### Calibration Report Format

```
═══════════════════════════════════════════════════════
  PIT CALIBRATION REPORT: PLNJPY=X
═══════════════════════════════════════════════════════

📊 Horizon: 1 month (21 days)
   Predictions: 450

   Status: ✅ WELL-CALIBRATED

   Expected Calibration Error (ECE): 0.0342
   Maximum Calibration Error (MCE): 0.0821
   Brier Score: 0.2156

   Uniformity Test: ✅ PASS (p=0.234)

   Reliability Diagram:
   ┌─────────────────────────────────────────┐
   │ Predicted →   Actual Frequency         │
   ├─────────────────────────────────────────┤
   │  0.15  →  0.14  (n= 42) ✓ │
   │  0.25  →  0.27  (n= 38) ✓ │
   │  0.35  →  0.32  (n= 45) ✓ │
   │  0.45  →  0.48  (n= 51) ✓ │
   │  0.55  →  0.53  (n= 48) ✓ │
   │  0.65  →  0.67  (n= 44) ✓ │
   │  0.75  →  0.72  (n= 39) ✓ │
   │  0.85  →  0.88  (n= 35) ✓ │
   └─────────────────────────────────────────┘

   Legend: ✓ = good fit, ↑ = overconfident, ↓ = underconfident

🎯 OVERALL: All horizons are well-calibrated
   → Probabilities are truth-telling
   → Kelly sizing is safe to use
```

### Key Metrics Explained

#### 1. Expected Calibration Error (ECE)

**Definition**: Weighted average deviation between predicted probabilities and actual frequencies.

```
ECE = Σ (n_i / n) × |predicted_i - actual_i|
```

**Interpretation**:
- **ECE < 0.05**: ✅ Well-calibrated (acceptable deviation)
- **ECE 0.05-0.10**: ⚠️ Slightly miscalibrated (use caution)
- **ECE > 0.10**: ❌ Miscalibrated (probabilities unreliable)

**Example**: ECE = 0.034 means predictions deviate by 3.4% on average from actual frequencies.

#### 2. Maximum Calibration Error (MCE)

**Definition**: Worst-case deviation in any probability bin.

**Interpretation**:
- **MCE < 0.10**: Good worst-case accuracy
- **MCE > 0.15**: Some predictions are very wrong

**Example**: MCE = 0.082 means the worst bin deviates by 8.2%.

#### 3. Brier Score

**Definition**: Mean squared error of probabilistic predictions (proper scoring rule).

```
BS = (1/n) × Σ (predicted_i - actual_binary_i)²
```

**Interpretation**:
- **BS = 0.00**: Perfect predictions
- **BS = 0.25**: Random predictions (50/50 guess)
- **BS < 0.20**: Good probabilistic accuracy

**Example**: BS = 0.216 means predictions are better than random but not perfect.

#### 4. Uniformity Test

**Definition**: Chi-squared test for uniform distribution of predicted probabilities.

**Why it matters**: Under well-calibrated forecasts, predicted probabilities should be uniformly distributed (all bins equally populated).

**Interpretation**:
- **p-value ≥ 0.05**: ✅ PASS — probabilities are uniformly distributed (good sign)
- **p-value < 0.05**: ⚠️ FAIL — non-uniform distribution (possible mis-specification)

#### 5. Reliability Diagram

**Definition**: Bin predicted probabilities and compare with actual outcome frequencies.

**Perfect calibration**: Points lie on the diagonal (predicted = actual).

**Symbols**:
- `✓` = deviation < 3% (good fit)
- `↑` = predicted > actual (overconfident)
- `↓` = predicted < actual (underconfident)

---

## Diagnosis and Remediation

### Well-Calibrated (✅)

**Characteristics**:
- ECE < 0.05
- Uniformity test passes (p ≥ 0.05)
- Reliability diagram near-diagonal

**Action**: ✅ **No action needed.** Probabilities are truth-telling. Kelly sizing is safe.

---

### Overconfident (↑)

**Characteristics**:
- ECE > 0.05
- Average deviation: predicted > actual
- Reliability diagram: points above diagonal

**Example**:
```
Predicted: 0.70  →  Actual: 0.55  (n=45) ↑
```

**What this means**: When you predict 70% chance of positive return, only 55% actually occur. You're **overconfident**.

**Risks**:
- Over-bet via Kelly criterion
- Higher than expected losses
- Ruin risk if severe

**Remediation**:
1. **Isotonic regression**: Fit monotonic calibration curve to map predicted → calibrated
2. **Platt scaling**: Logistic regression to rescale probabilities
3. **Temperature scaling**: Scale logits by factor T > 1 (softens confidence)
4. **Reduce Kelly fraction**: Use 0.25× Kelly instead of 0.5× until recalibrated

---

### Underconfident (↓)

**Characteristics**:
- ECE > 0.05
- Average deviation: predicted < actual
- Reliability diagram: points below diagonal

**Example**:
```
Predicted: 0.50  →  Actual: 0.68  (n=52) ↓
```

**What this means**: When you predict 50% chance, 68% actually occur. You're **underconfident**.

**Risks**:
- Under-bet via Kelly criterion
- Missed opportunities
- Lower than expected returns

**Remediation**:
1. **Isotonic regression**: Fit monotonic calibration curve
2. **Platt scaling**: Logistic regression to rescale
3. **Temperature scaling**: Scale logits by factor T < 1 (sharpens confidence)
4. **Increase Kelly fraction**: Use 0.75× Kelly (but verify via backtesting)

---

## Technical Details

### Walk-Forward Methodology

PIT calibration uses **strict out-of-sample testing** to avoid look-ahead bias:

1. **Split data** into non-overlapping train/test windows
2. **Fit model** on training window (e.g., 504 days = ~2 years)
3. **Predict probabilities** at end of training window
4. **Observe actual outcomes** H days forward in test window
5. **Repeat** for all windows (walk-forward)

This ensures every prediction uses **only past data**, making results unbiased.

### Binning Strategy

Predicted probabilities are binned into equal-width intervals (default 10 bins):
- Bin 1: [0.0, 0.1)
- Bin 2: [0.1, 0.2)
- ...
- Bin 10: [0.9, 1.0]

Within each bin:
- **Predicted**: Mean predicted probability
- **Actual**: Fraction of positive outcomes
- **Deviation**: |Predicted - Actual|

### Statistical Tests

#### Expected Calibration Error (ECE)

```python
ECE = Σ_i (n_i / n) × |pred_i - actual_i|
```

Weighted by bin size to avoid empty-bin bias.

#### Uniformity (Chi-Squared Test)

```
H0: Predicted probabilities are uniformly distributed
H1: Non-uniform distribution

χ² = Σ_i (observed_i - expected_i)² / expected_i

p-value from χ²(df = n_bins - 1)
```

#### Brier Score

```
BS = (1/n) × Σ_i (p_i - y_i)²

where:
  p_i = predicted probability
  y_i = actual outcome (0 or 1)
```

Proper scoring rule: incentivizes truthful reporting of beliefs.

---

## Example: Real-World Interpretation

### Case 1: Good Calibration

```
ECE: 0.0342
MCE: 0.0821
Brier: 0.2156
Uniformity: p=0.234 (PASS)
Diagnosis: well_calibrated
```

**Interpretation**: Your probabilities are accurate. When you say 62%, about 62% actually happen. Safe to use Kelly sizing.

**Action**: Continue monitoring. No changes needed.

---

### Case 2: Overconfident

```
ECE: 0.1523
MCE: 0.2341
Brier: 0.2789
Uniformity: p=0.012 (FAIL)
Diagnosis: overconfident
```

**Reliability Diagram**:
```
0.30 → 0.18  ↑  (predicted 30%, actual 18%)
0.50 → 0.38  ↑  (predicted 50%, actual 38%)
0.70 → 0.55  ↑  (predicted 70%, actual 55%)
0.90 → 0.78  ↑  (predicted 90%, actual 78%)
```

**Interpretation**: You're systematically overestimating probabilities by ~12%. This leads to over-betting.

**Action**:
1. **Immediate**: Reduce Kelly fraction from 0.5× to 0.25×
2. **Short-term**: Apply isotonic regression calibration
3. **Long-term**: Investigate model: are you overfitting? Using too-optimistic priors?

---

### Case 3: Underconfident

```
ECE: 0.0981
MCE: 0.1523
Brier: 0.2234
Uniformity: p=0.089 (PASS)
Diagnosis: underconfident
```

**Reliability Diagram**:
```
0.30 → 0.42  ↓  (predicted 30%, actual 42%)
0.50 → 0.61  ↓  (predicted 50%, actual 61%)
0.70 → 0.79  ↓  (predicted 70%, actual 79%)
```

**Interpretation**: You're systematically underestimating by ~10%. Missing opportunities.

**Action**:
1. **Verify**: Run out-of-sample backtest to confirm underconfidence
2. **Calibrate**: Apply isotonic regression or Platt scaling
3. **Consider**: Increase Kelly fraction to 0.6× after recalibration

---

## Integration with Existing System

### How It Fits Into Level-7 Architecture

```
Signal Generation Pipeline:
1. Kalman Filter → Drift μ_t (with uncertainty)
2. GARCH → Volatility σ_t (with uncertainty)
3. HMM → Regime detection
4. Monte Carlo → Forward paths
5. Probability Mapping → P(return > 0) via Student-t CDF + Edgeworth
6. **PIT Calibration** → Verify P(return > 0) matches reality ← YOU ARE HERE
7. Kelly Criterion → Position sizing (safe if #6 passes)
```

### When to Run

**Frequency**:
- **Development**: Every model change
- **Production**: Weekly or monthly
- **After market stress**: After large regime shifts

**Cost**: ~5-10 minutes per asset (500 out-of-sample predictions via walk-forward)

---

## References

### Academic

1. **Probability Integral Transform**:
   - Dawid (1984) "Statistical Theory: The Prequential Approach"
   - Diebold et al. (1998) "Evaluating Density Forecasts"

2. **Calibration Metrics**:
   - Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
   - Guo et al. (2017) "On Calibration of Modern Neural Networks"

3. **Kelly Criterion**:
   - Kelly (1956) "A New Interpretation of Information Rate"
   - Thorp (2008) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"

### Industry

- RiskMetrics methodology (J.P. Morgan, 1996)
- Brier Score (1950) — proper scoring rule for probabilistic forecasts
- Expected Calibration Error — standard in ML calibration (Guo et al., 2017)

---

## FAQ

### Q: Why is ECE threshold 0.05?

**A**: Industry standard from ML calibration literature. Represents 5% average deviation, which is:
- Tight enough to catch serious miscalibration
- Loose enough to accommodate sampling noise with ~500 predictions

### Q: What if I don't have 500+ predictions?

**A**: PIT calibration requires **sufficient sample size** (typically 200-500 per horizon). With fewer:
- Increase ECE threshold to 0.08 or 0.10
- Use bootstrap confidence intervals
- Collect more data before trusting calibration

### Q: Should I recalibrate after every test?

**A**: **No.** Recalibration on the test set introduces look-ahead bias. Instead:
1. Run PIT test on holdout data
2. If miscalibrated, fit recalibration on **separate validation set**
3. Apply calibration to future predictions
4. Re-test on new holdout after deployment

### Q: What about multi-horizon calibration?

**A**: Test each horizon separately (1d, 21d, 63d, etc.). Calibration may vary by horizon:
- Short horizons (1d): Harder to calibrate (more noise)
- Medium horizons (21d): Usually best calibrated
- Long horizons (252d): Fewer predictions, less reliable test

### Q: How does this relate to backtesting?

**A**: Complementary:
- **Backtest**: Tests strategy profitability (Sharpe, drawdown, etc.)
- **PIT Calibration**: Tests probability accuracy (are your forecasts honest?)

You can have great backtests with miscalibrated probabilities (lucky) or bad backtests with calibrated probabilities (unlucky). Both are needed.

---

## Summary

**PIT calibration is the single most important test for probabilistic systems** because:

1. ✅ **Kelly sizing requires calibration** — miscalibration leads to ruin or missed opportunities
2. ✅ **Turns model into truth-telling machine** — predictions match reality
3. ✅ **Falsifiable** — clear pass/fail criterion (ECE < 0.05)
4. ✅ **Level-7 hallmark** — separates quant research from production-grade systems

**When calibrated**:
- Your P↑=0.62 means 62% of outcomes are actually positive
- Kelly sizing is safe
- You have confidence in your edge

**When miscalibrated**:
- Probabilities lie
- Kelly sizing is dangerous
- Recalibrate before deploying

---

**Run PIT calibration on your system today. Make your probabilities truth-telling.**
