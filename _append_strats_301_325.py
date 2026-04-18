#!/usr/bin/env python3
"""Append strategies 301-325 to Indicators.md"""

content = r"""
### 301 | Fama-French Five-Factor Alpha Capture
**School:** Academic/Chicago (Eugene Fama, Kenneth French) | **Class:** Multi-Factor
**Timeframe:** Monthly Rebalance | **Assets:** Equities

**Mathematics:**
```
Five-Factor Model:
  R_i - R_f = alpha_i + beta_MKT*(R_m - R_f) + beta_SMB*SMB + beta_HML*HML
              + beta_RMW*RMW + beta_CMA*CMA + epsilon_i

  MKT = Market excess return
  SMB = Small Minus Big (size premium)
  HML = High Minus Low (value premium: high B/M minus low B/M)
  RMW = Robust Minus Weak (profitability premium)
  CMA = Conservative Minus Aggressive (investment premium)

Alpha Capture Strategy:
  For each stock in universe (500+):
    1. Regress 60 months of returns on 5 factors
    2. Estimate alpha_i (risk-adjusted return after factor exposure)
    3. Rank stocks by alpha_i
    
    Long: top decile by alpha (highest unexplained positive returns)
    Short: bottom decile (most negative unexplained returns)

Factor Timing Layer:
  Momentum_factor = 12M-1M factor return for each factor
  If SMB momentum positive: overweight small cap allocation
  If HML momentum positive: overweight value allocation
  Dynamic factor tilts based on factor momentum
```

**Signal:**
- **Long:** Top decile by 5-factor alpha + positive alpha_t-stat > 2 (statistically significant)
- **Short:** Bottom decile by 5-factor alpha with significant negative alpha
- **Factor tilt:** Overweight factors with positive 12M-1M momentum
- **Rebalance:** Monthly; transition slowly to minimize impact

**Risk:** Market-neutral or beta-controlled; sector limits; Risk per position 1%
**Edge:** Five-factor alpha isolates returns that cannot be explained by market, size, value, profitability, or investment factors. Persistent positive alpha indicates genuine mispricing or an unmeasured factor. By trading the extremes (top/bottom decile) with statistical significance requirement, you isolate stocks with the highest probability of genuine alpha. Factor timing adds 1-3% annually by overweighting factors in their momentum phase.

---

### 302 | Hidden Markov Model Regime Switching
**School:** Academic/Quantitative | **Class:** Regime Detection
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
HMM for Market Regimes:
  Hidden States: S = {Bull, Bear, Sideways}
  Observable: Daily returns r_t
  
  Emission Distributions:
    Bull: r_t ~ N(mu_bull, sigma_bull)      mu_bull > 0, sigma_bull small
    Bear: r_t ~ N(mu_bear, sigma_bear)      mu_bear < 0, sigma_bear large
    Sideways: r_t ~ N(mu_side, sigma_side)  mu_side ~ 0, sigma_side medium
  
  Transition Matrix A:
    A[i,j] = P(S_{t+1} = j | S_t = i)
    Typical:
      Bull->Bull: 0.95, Bull->Bear: 0.02, Bull->Sideways: 0.03
      Bear->Bear: 0.90, Bear->Bull: 0.03, Bear->Sideways: 0.07
      Sideways->Sideways: 0.85, Sideways->Bull: 0.10, Sideways->Bear: 0.05

  Estimation: Baum-Welch (EM) algorithm on rolling 500-day window
  State Inference: Viterbi algorithm or forward-backward smoothing
  
  Current State Probability:
    P(Bull | r_{1:t}) via forward algorithm
    If P(Bull) > 0.7: regime = Bull
    If P(Bear) > 0.7: regime = Bear
    Else: regime = Sideways/Uncertain
```

**Signal:**
- **Long:** P(Bull) > 0.7 AND P(Bull) rising (entering bull regime)
- **Short/Cash:** P(Bear) > 0.7 (entering bear regime)
- **Reduce exposure:** P(Sideways) dominant or no regime > 0.6 (uncertain)
- **Regime transition:** P(Bull) drops from > 0.7 to < 0.5 = early exit signal

**Risk:** Position size proportional to regime confidence; Full in bull, cash in bear
**Edge:** HMM provides a principled probabilistic framework for regime detection, superior to ad-hoc indicators because it jointly estimates regime parameters AND transition probabilities from data. The transition matrix captures an essential market feature: regimes are persistent (bull stays bull ~95% of days) but transitions are asymmetric (bear-to-bull is rare and sudden). The probabilistic output allows gradual position adjustment rather than binary switching.

---

### 303 | Quality-Momentum Composite Factor
**School:** Academic/AQR (Cliff Asness) | **Class:** Multi-Factor
**Timeframe:** Monthly Rebalance | **Assets:** Equities

**Mathematics:**
```
Quality Score:
  ROE_zscore = zscore(ROE, cross_section)
  Accruals_zscore = -zscore(Accruals/Assets)  (lower accruals = higher quality)
  Leverage_zscore = -zscore(D/E)  (lower leverage = higher quality)
  Earnings_Stability = -zscore(std(EPS_growth, 5yr))  (stable = quality)
  
  Quality = equal_weight(ROE_z, Accruals_z, Leverage_z, Stability_z)

Momentum Score:
  MOM_12_1 = (Price / Price_12m_ago) - 1  (skip last month: reversal avoidance)
  MOM_z = zscore(MOM_12_1, cross_section)

Composite Factor:
  QM = 0.5 * Quality_z + 0.5 * MOM_z
  
  Long: top quintile by QM (high quality + high momentum)
  Short: bottom quintile (low quality + low momentum)

Interaction Effect:
  Quality AND Momentum together outperform either alone because:
    Momentum without quality = trend in junk stocks (crash risk)
    Quality without momentum = dead money (value traps)
    Quality + Momentum = trending high-quality stocks (persistent winners)
```

**Signal:**
- **Long:** Top quintile by QM composite (quality + momentum = best of both worlds)
- **Short:** Bottom quintile (low quality + low momentum = worst of both worlds)
- **Rebalance:** Monthly with 20% turnover cap (avoid excessive trading)
- **Crash protection:** Momentum crash filter: if MOM factor < -10% in prior month, reduce exposure

**Risk:** Market-neutral or long-biased (130/30); Sector diversification; Risk 1% per name
**Edge:** The Quality-Momentum combination has produced the highest Sharpe ratio of any published two-factor strategy over the past 50 years (~0.8-1.0 net of costs). Quality selects stocks with sustainable businesses; momentum ensures those businesses are currently being recognized by the market. The interaction effect is crucial: momentum in quality stocks is MORE persistent than momentum in junk, and quality stocks recover faster from momentum crashes.

---

### 304 | Carry Factor Cross-Asset Strategy
**School:** London/AQR | **Class:** Carry Factor
**Timeframe:** Monthly Rebalance | **Assets:** Multi-Asset (FX, Bonds, Commodities, Equities)

**Mathematics:**
```
Carry Definition (per asset class):
  FX: Carry = short-rate_foreign - short-rate_domestic
  Bonds: Carry = (yield * duration - funding_cost) / duration  (roll-adjusted)
  Commodities: Carry = -(futures_price - spot_price) / spot_price  (backwardation = positive carry)
  Equities: Carry = dividend_yield - risk_free_rate

Carry Rank:
  Within each asset class, rank instruments by carry
  Long: top third (highest carry)
  Short: bottom third (lowest carry)

Cross-Asset Carry Portfolio:
  FX_carry_portfolio (10 currencies)
  Bond_carry_portfolio (10 countries)
  Commodity_carry_portfolio (20 commodities)
  Equity_carry_portfolio (20 markets/sectors)
  
  Equal risk allocation across asset classes:
  w_class = target_vol / realized_vol_class

Carry Risk Premium:
  E[carry_return] = carry_yield - E[price_change]
  Under UIP (uncovered interest parity): E[price_change] offsets carry
  Empirical: UIP FAILS -> carry earns ~3-5% annually across all asset classes
  Sharpe ratio of diversified carry: ~0.7-0.9
```

**Signal:**
- **Long:** Highest carry instruments in each asset class
- **Short:** Lowest carry instruments in each asset class
- **Size:** Risk-parity across asset classes (equal volatility contribution)
- **Rebalance:** Monthly; adjust for changes in carry rankings

**Risk:** Carry strategies have tail risk (crash when carry unwinds); Limit leverage; Risk budget per class
**Edge:** The carry premium is one of the most persistent and diversified risk premiums in finance. It exists across FX, bonds, commodities, and equities because it compensates holders for bearing crash risk (carry trades lose money during liquidity crises). By diversifying carry across four asset classes, the strategy reduces the tail risk of any single carry trade. Cross-asset carry has a Sharpe of ~0.8, higher than any single-asset carry strategy.

---

### 305 | Low Volatility Anomaly Factor
**School:** Academic/Robeco (Pim van Vliet) | **Class:** Defensive Factor
**Timeframe:** Quarterly Rebalance | **Assets:** Equities

**Mathematics:**
```
Low Volatility Factor:
  For each stock: sigma = std(daily_returns, 252) * sqrt(252)  (annualized vol)
  
  Rank stocks by sigma (ascending)
  Long: bottom quintile (lowest volatility stocks)
  Short: top quintile (highest volatility stocks) [optional]

Low-Vol Anomaly:
  CAPM predicts: E[R_i] = R_f + beta_i * (E[R_m] - R_f)
  Higher beta -> higher expected return
  
  REALITY: Low-beta stocks earn HIGHER risk-adjusted returns than high-beta
  Sharpe(low_vol) >> Sharpe(high_vol) empirically
  
  This is the "low volatility anomaly" -- one of the most robust anomalies in finance
  Documented across 30+ countries, 50+ years of data

Low-Vol with Quality Filter:
  Pure low-vol can load on "boring" sectors (utilities, staples)
  Add quality screen: ROE > median AND Debt/Equity < median
  
  Low_Vol_Quality = rank_by_vol * 0.6 + quality_rank * 0.4
  Long: top quintile by Low_Vol_Quality

Betting Against Beta (BAB):
  Long: low-beta stocks (leveraged to beta = 1)
  Short: high-beta stocks (deleveraged to beta = 1)
  This creates a zero-beta portfolio that earns ~5-7% annually
```

**Signal:**
- **Long:** Bottom quintile by volatility with quality filter (low vol + high quality)
- **Short (optional):** Top quintile by volatility (highest vol = worst risk-adjusted returns)
- **BAB variant:** Lever low-beta longs, delever high-beta shorts to equalize beta
- **Rebalance:** Quarterly (low-vol is slow-moving)

**Risk:** Low-vol strategies can underperform in speculative rallies; Sector concentration risk
**Edge:** The low volatility anomaly is one of the most robust, persistent, and unexplained anomalies in finance. It exists because: (1) leverage-constrained investors chase high-beta stocks, overvaluing them; (2) lottery preference makes investors overpay for volatile stocks; (3) benchmarking creates incentives against low-vol strategies. By systematically buying low-vol stocks, you harvest a premium that most institutional investors are structurally prevented from capturing.

---

### 306 | Adaptive Asset Allocation (Butler/Philbrick)
**School:** Canadian (ReSolve Asset Management) | **Class:** Regime-Adaptive
**Timeframe:** Monthly Rebalance | **Assets:** Multi-Asset

**Mathematics:**
```
Adaptive Asset Allocation (AAA):
  Universe: Equities, Bonds, REITs, Gold, Commodities (ETFs)
  
  Step 1: Momentum Score
    For each asset: MOM = mean(return_1m, return_3m, return_6m, return_12m)
    Select top 50% by momentum (filter out downtrending assets)
  
  Step 2: Minimum Variance Optimization
    For selected assets, compute covariance matrix (60-day rolling)
    Optimize for minimum variance portfolio with constraints:
      w_i >= 0 (no shorting)
      sum(w_i) = 1
      w_i <= 0.25 (max 25% in any single asset)
  
  Step 3: Apply weights to selected assets
    Assets NOT in top 50% momentum: weight = 0
    Assets in top 50%: weights from min-variance optimization

Key Innovation:
  Traditional 60/40: fixed allocation regardless of conditions
  Risk parity: fixed risk allocation regardless of trends
  AAA: BOTH momentum filtering AND risk optimization
  
  In crisis: momentum filter removes falling assets
  In calm: min-variance optimizes within rising assets
```

**Signal:**
- **Asset selection:** Top 50% of multi-asset universe by composite momentum
- **Weighting:** Minimum variance optimization among selected assets
- **Monthly rebalance:** Full re-optimization each month
- **Defensive:** If < 2 assets have positive momentum, move remainder to short-term bonds

**Risk:** Max 25% per asset; Min-variance limits concentration; Cash when no momentum
**Edge:** AAA combines the two most powerful portfolio techniques: momentum filtering (removes losing assets) and minimum variance optimization (minimizes portfolio risk among winners). The interaction is powerful: momentum ensures you only hold assets in uptrends, while min-variance ensures those uptrend allocations are diversified optimally. Backtesting shows AAA delivering equity-like returns with bond-like volatility (Sharpe ~1.0) across multiple decades.

---

### 307 | Sector Rotation via Relative Strength Matrix
**School:** New York/Dorsey Wright | **Class:** Relative Strength
**Timeframe:** Monthly | **Assets:** US Sectors (11 GICS sectors)

**Mathematics:**
```
Relative Strength Matrix:
  For 11 sectors, compute pairwise relative strength:
  RS(i,j) = Sector_i / Sector_j (price ratio)
  
  For each sector:
    Wins = count(RS(i,j) > RS(i,j)_200d_avg for all j != i)
    RS_Score = Wins / 10  (percentage of sectors outperformed)
  
  Rank sectors by RS_Score

Relative Strength Ranking:
  Top 3 sectors (RS_Score > 70%): OVERWEIGHT (leading sectors)
  Bottom 3 sectors (RS_Score < 30%): UNDERWEIGHT (lagging sectors)
  Middle 5 sectors: MARKET WEIGHT

Sector Momentum:
  For ranked sectors:
    RS_Score rising from < 50% to > 70%: NEW LEADER emerging (buy)
    RS_Score falling from > 70% to < 50%: LEADER fading (sell)
    
  Persistence: sector leadership persists ~4-8 months on average

Macro Overlay:
  Rate-sensitive sectors (Utilities, REITs): weight by yield curve signal
  Cyclical sectors (Materials, Industrials): weight by ISM signal
  Growth sectors (Tech, Comm Services): weight by earnings momentum
```

**Signal:**
- **Overweight:** Top 3 sectors by RS matrix score (outperforming most peers)
- **Underweight:** Bottom 3 sectors (underperforming most peers)
- **New leader:** RS_Score crossing above 70% from below (sector rotation entry)
- **Rebalance:** Monthly; verify with macro overlay for cyclical sectors

**Risk:** Diversify within sectors; max 25% in any sector; equal-risk contribution
**Edge:** The pairwise relative strength matrix is the most comprehensive sector ranking method because it compares each sector against ALL others simultaneously, not just against the benchmark. A sector ranking #1 relative to one benchmark might be #5 relative to another sector. The matrix approach produces a TRUE rank that is benchmark-independent. Leadership persistence (4-8 months) provides enough time for profitable rotation.

---

### 308 | Volatility Regime Conditional Strategy Selection
**School:** Quantitative | **Class:** Regime-Conditional
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Volatility Regime Classification:
  sigma = EWMA_vol(returns, 60)
  sigma_percentile = percentile_rank(sigma, sigma_history_252d)
  
  Regime 1: sigma < 25th percentile -> LOW VOL
  Regime 2: 25th < sigma < 75th -> NORMAL VOL
  Regime 3: sigma > 75th percentile -> HIGH VOL
  Regime 4: sigma > 95th AND sigma rising -> CRISIS
  
  Transition Detection:
    regime_change = current_regime != prior_day_regime
    Use HMM or simple threshold crossing with hysteresis band

Strategy Selection per Regime:
  LOW VOL (< 25th pctile):
    Primary: Mean reversion (RSI, Bollinger Band)
    Size: Larger (low vol = smaller moves, need size for returns)
    Stop: Tight (2x ATR still small in absolute terms)
  
  NORMAL VOL (25th-75th):
    Primary: Trend following (MACD, Moving Average)
    Size: Standard
    Stop: Medium (3x ATR)
  
  HIGH VOL (75th-95th):
    Primary: Momentum (breakout, Donchian)
    Size: Reduced (high vol = large moves, reduce to control risk)
    Stop: Wide (4x ATR)
  
  CRISIS (> 95th, rising):
    Primary: Risk-off (cash, short, hedges)
    Size: Minimal or zero
    Stop: Very wide or time-based
```

**Signal:**
- **Low vol:** Activate mean-reversion strategy suite, larger position sizes
- **Normal vol:** Activate trend-following suite, standard sizing
- **High vol:** Activate momentum suite, reduced sizing
- **Crisis:** Risk-off, minimal exposure, protective positions

**Risk:** Position size inversely proportional to vol regime; Strategy-specific stops
**Edge:** Most strategy failures come from applying the wrong strategy to the wrong volatility regime. Mean reversion works in low vol (markets oscillate around fair value), trend following in normal vol (trends develop cleanly), and momentum/risk-off in high vol (moves are large and directional). By conditioning strategy selection on the current volatility regime, you avoid the 60-70% of losses that come from strategy-regime mismatch.

---

### 309 | Momentum Crash Protection via Factor Crowding
**School:** Academic/AQR | **Class:** Risk Management
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
Factor Crowding Measure:
  For momentum factor:
    Short_interest_ratio = median(short_interest / float) for momentum shorts
    Valuation_spread = mean(P/E of momentum longs) - mean(P/E of momentum shorts)
    Correlation_concentration = max_eigenvalue / sum_eigenvalues of momentum portfolio
  
  Crowding_Score = z(short_interest) + z(valuation_spread) + z(correlation_concentration)
  Range: typically [-3, +3]
  
  Crowding > +2: Momentum trade extremely crowded (crash risk elevated)
  Crowding < 0: Momentum trade uncrowded (safe to run full momentum)

Momentum Crash Dynamics:
  Momentum crashes occur when:
    1. Bear market reversal (losers suddenly outperform winners)
    2. High crowding (everyone in the same trade)
    3. Leverage unwind (forced selling of winners)
  
  Crash probability model:
    P(crash) = logistic(a * VIX_change + b * Crowding + c * Bear_signal)
    If P(crash) > 0.3: reduce momentum exposure by 50%
    If P(crash) > 0.5: hedge or exit momentum entirely

Crash Protection Rules:
  If market drops > 10% in 20 days: Pause new momentum entries for 5 days
  If momentum factor returns < -5% in 5 days: Cut momentum exposure 50%
  If crowding > 2: Run momentum at 50% of normal allocation
```

**Signal:**
- **Full momentum:** Crowding < 0 AND no crash signals = run full momentum allocation
- **Reduced momentum:** Crowding > 1.5 OR market down > 5% = reduce by 50%
- **Pause momentum:** P(crash) > 0.3 = pause new entries, maintain existing
- **Exit momentum:** P(crash) > 0.5 OR momentum factor < -5% in 5 days = exit

**Risk:** This is a META-strategy that adjusts momentum portfolio sizing; not standalone
**Edge:** Momentum crashes are the primary risk of momentum strategies: they occur suddenly, lose 20-40% in weeks, and have fat-left-tail distributions. The factor crowding measure identifies when momentum is most vulnerable because crowding amplifies crash severity (more people exit simultaneously). By reducing exposure when crowding is high, you give up ~1% of annual return in normal times but avoid 50-70% of crash drawdowns. This dramatically improves the tail risk profile of momentum.

---

### 310 | Cross-Sectional Momentum with Industry Adjustment
**School:** Academic (Moskowitz & Grinblatt) | **Class:** Long-Short Equity
**Timeframe:** Monthly | **Assets:** US Equities (500+)

**Mathematics:**
```
Industry-Adjusted Momentum:
  Raw_MOM = (Price_{t} / Price_{t-12m}) - 1  (12-month return, skip recent month)
  
  Industry_MOM = mean(Raw_MOM for all stocks in same industry)
  
  Stock_specific_MOM = Raw_MOM - Industry_MOM
  
  This isolates STOCK-SPECIFIC momentum from INDUSTRY momentum

Two Portfolios:
  Portfolio 1 (Industry Momentum):
    Long: top 3 industries by Industry_MOM
    Short: bottom 3 industries by Industry_MOM
  
  Portfolio 2 (Stock-Specific Momentum):
    Within each industry:
      Long: top quintile by Stock_specific_MOM
      Short: bottom quintile by Stock_specific_MOM

Combined Signal:
  W = 0.4 * Industry_MOM_signal + 0.6 * Stock_specific_MOM_signal
  
  Research shows:
    Industry momentum: explains ~55% of total momentum profits
    Stock-specific: explains ~45% but has LOWER crash risk
    Optimal blend: 40/60 for best risk-adjusted returns
```

**Signal:**
- **Long:** Stocks in top industries with top stock-specific momentum (double winner)
- **Short:** Stocks in bottom industries with bottom stock-specific momentum (double loser)
- **Industry tilt:** Overweight leading industries, underweight lagging
- **Rebalance:** Monthly; 20% turnover cap per month

**Risk:** Market-neutral; Industry-neutral within stock-specific portfolio; Risk 1% per position
**Edge:** Decomposing momentum into industry and stock-specific components is crucial because they have different drivers and different risk profiles. Industry momentum captures sector rotation (macro-driven) while stock-specific momentum captures company-level information diffusion. The combination diversifies across both sources and reduces the correlation that causes momentum crashes (which are primarily driven by industry-level reversals, not stock-specific).

---

### 311 | Dynamic Factor Timing via Macroeconomic Variables
**School:** Academic/Swiss (Robeco, SSGA) | **Class:** Factor Timing
**Timeframe:** Monthly | **Assets:** Factor Portfolios

**Mathematics:**
```
Factor Timing Model:
  Factors: MKT, SMB, HML, MOM, QMJ, BAB (6 standard factors)
  
  Macro Predictors:
    1. Yield curve slope (10Y - 2Y)
    2. Credit spread (BAA - AAA)
    3. VIX level
    4. ISM Manufacturing
    5. Inflation surprise (CPI actual - CPI expected)
    6. Money supply growth (M2 YoY)
  
  For each factor:
    Expected_return_factor = alpha + sum(beta_k * Macro_k)
    Estimated via rolling 120-month OLS
  
  Factor Weight:
    If Expected_return > 0: w_factor = Expected_return / sum(|Expected_returns|)
    If Expected_return < 0: w_factor = 0 (don't short factors with negative expected return)

Macro-Factor Relationships (empirical):
  Rising yields: favors VALUE (HML), hurts GROWTH (anti-HML)
  Rising credit spreads: favors QUALITY (QMJ), hurts JUNK
  Rising VIX: favors LOW-VOL (BAB), hurts HIGH-BETA
  Rising ISM: favors MOMENTUM, SIZE (SMB)
  Rising inflation: favors VALUE, COMMODITIES
```

**Signal:**
- **Overweight factor:** Expected macro environment favors factor (positive expected return)
- **Zero weight:** Expected environment unfavorable (negative expected return)
- **Rebalance:** Monthly based on updated macro data releases
- **Conviction scaling:** Higher R-squared in macro-factor regression = higher allocation

**Risk:** Diversify across 3-4 factors minimum; Never all-in on single factor; Risk budget per factor
**Edge:** Factor timing via macro variables adds ~1-2% annually to static factor allocation because macro conditions predictably affect factor returns. The yield curve is the single best predictor: steepening favors value/size, flattening favors quality/low-vol. Credit spreads predict momentum (tightening = good for momentum). These relationships are structural (not spurious) because they reflect the economic mechanisms that drive factor premiums.

---

### 312 | Principal Component Analysis Residual Alpha
**School:** Academic/Quantitative | **Class:** Statistical Factor Model
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
PCA Factor Model:
  R = n x T matrix of stock returns (n stocks, T months)
  
  Step 1: Extract K principal components from R
    SVD: R = U * S * V'
    First K components explain > 80% of variance
    Typically K = 5-10 for 500 stocks
  
  Step 2: Factor loadings and residuals
    F = K x T factor returns (principal components)
    B = n x K factor loadings
    
    R_hat = B * F  (explained by factors)
    Epsilon = R - R_hat  (RESIDUAL returns, factor-orthogonal)
  
  Step 3: Residual Momentum
    For each stock: Res_MOM = sum(epsilon_{t-12:t-1})  (12M residual return)
    
    Long: top quintile by Res_MOM (positive factor-orthogonal momentum)
    Short: bottom quintile by Res_MOM

Key Insight:
  Standard momentum = factor momentum + idiosyncratic momentum
  PCA removes factor momentum, leaving PURE stock-specific alpha
  
  Residual momentum has:
    Lower crash risk (factor momentum causes crashes)
    Higher Sharpe (~0.6 vs ~0.45 for standard momentum)
    Less capacity constrained
```

**Signal:**
- **Long:** Top quintile by 12-month residual momentum (stock-specific alpha)
- **Short:** Bottom quintile (stock-specific underperformance)
- **Factor-neutral:** By construction, portfolio has near-zero loading on all PCA factors
- **Rebalance:** Monthly; turnover ~30%

**Risk:** Market-neutral by construction; Sector limits; Risk 1% per position
**Edge:** PCA residual momentum isolates the PURE stock-specific component of momentum, removing all systematic factor exposures. This produces a momentum strategy with dramatically lower crash risk because momentum crashes are driven by FACTOR reversals, not stock-specific reversals. The Sharpe ratio improvement (~0.6 vs 0.45) comes from removing the noisy factor component and retaining only the informative stock-specific signal.

---

### 313 | Bayesian Dynamic Linear Model for Returns
**School:** Academic (Harrison & West, 1997) | **Class:** Bayesian Time Series
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Dynamic Linear Model (DLM):
  Observation equation: y_t = F_t' * theta_t + v_t, v_t ~ N(0, V_t)
  State equation: theta_t = G_t * theta_{t-1} + w_t, w_t ~ N(0, W_t)
  
  theta_t = [level_t, trend_t, seasonal_t, regression_t]' (state vector)
  F_t = observation matrix (maps states to observed returns)
  G_t = state transition matrix (how states evolve)
  V_t = observation variance
  W_t = state variance (process noise)

Kalman Filter for Sequential Estimation:
  Prior: theta_t | D_{t-1} ~ N(a_t, R_t)
  Forecast: y_t | D_{t-1} ~ N(f_t, Q_t)
    where f_t = F_t' * a_t, Q_t = F_t' * R_t * F_t + V_t
  
  Update: theta_t | D_t ~ N(m_t, C_t)
    A_t = R_t * F_t / Q_t  (Kalman gain)
    e_t = y_t - f_t  (forecast error)
    m_t = a_t + A_t * e_t
    C_t = R_t - A_t * Q_t * A_t'

Trading Signal:
  Predictive distribution: y_{t+1} | D_t ~ N(f_{t+1}, Q_{t+1})
  
  Long: if f_{t+1} > 0 AND f_{t+1} / sqrt(Q_{t+1}) > 1 (positive mean, high Sharpe)
  Short: if f_{t+1} < 0 AND |f_{t+1}| / sqrt(Q_{t+1}) > 1
  Size: proportional to f_{t+1} / sqrt(Q_{t+1}) (signal-to-noise ratio)
```

**Signal:**
- **Long:** Predictive mean > 0 with SNR > 1 (expected positive return with conviction)
- **Short:** Predictive mean < 0 with SNR > 1
- **Size:** Linear in SNR (higher conviction = larger position)
- **Exit:** SNR drops below 0.5 (conviction insufficient)

**Risk:** Size by predictive variance (wider = smaller position); Max 2% risk
**Edge:** The Bayesian DLM is the theoretically optimal framework for sequential prediction under uncertainty because it exactly solves the problem of updating beliefs as new data arrives. Unlike rolling regressions (which treat each window independently), the DLM builds on all prior information while continuously adjusting for structural changes via the state evolution. The predictive distribution provides not just a forecast but a calibrated uncertainty estimate, enabling optimal Kelly-like position sizing.

---

### 314 | Disposition Effect Contrarian Strategy
**School:** Behavioral/Academic (Shefrin & Statman, 1985) | **Class:** Behavioral Exploitation
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Disposition Effect:
  Investors tend to:
    Sell winners too early (realize gains prematurely)
    Hold losers too long (refuse to realize losses)
  
  This creates UNDERREACTION to positive information (winners sold early)
  and OVERREACTION via delayed selling (losers held too long, eventually dumped)

Capital Gains Overhang (CGO):
  For each stock at time t:
    Reference_Price = volume-weighted average cost basis (estimated)
    
    Simplified: RP = sum(Price_k * Volume_k) / sum(Volume_k) over past 260 days
    
    CGO = (Price - RP) / RP
    
    CGO > 0: Most holders are sitting on GAINS (will sell = headwind)
    CGO < 0: Most holders are sitting on LOSSES (won't sell = potential bottoming)

Contrarian Strategy:
  Long: Stocks with CGO > +30% AND positive momentum
    (despite disposition selling pressure, momentum persists = strong signal)
  
  Short: Stocks with CGO < -30% AND negative momentum
    (holders refusing to sell but will eventually capitulate)
  
  The INTERACTION of CGO and momentum is key:
    High CGO + high momentum: winners keep winning despite selling pressure
    Low CGO + low momentum: losers keep losing as capitulation continues
```

**Signal:**
- **Long:** CGO > +30% AND 12M-1M momentum positive (persistent winner despite selling pressure)
- **Short:** CGO < -30% AND 12M-1M momentum negative (persistent loser before capitulation)
- **Avoid:** CGO near zero (no behavioral bias at work)
- **Rebalance:** Monthly

**Risk:** Market-neutral; Sector-neutral; Risk 1% per position
**Edge:** The disposition effect is one of the most robust behavioral biases: investors systematically sell winners too early, creating a headwind that makes positive momentum more informative. A stock that maintains positive momentum DESPITE heavy selling by profitable holders must have exceptionally strong underlying demand. By conditioning momentum on CGO, you isolate the most informative momentum signals -- those that survive behavioral selling pressure.

---

### 315 | Trend-Following Multi-Asset with Volatility Targeting
**School:** London/Systematic (Man AHL, Winton) | **Class:** CTA/Managed Futures
**Timeframe:** Daily | **Assets:** Multi-Asset (50+ futures markets)

**Mathematics:**
```
Trend Signal per Market:
  EWMA_fast = EWMA(price, 16 days)
  EWMA_slow = EWMA(price, 64 days)
  
  Raw_Signal = (EWMA_fast - EWMA_slow) / sigma_price
  where sigma_price = EWMA_vol(daily_price_change, 60) * sqrt(252)
  
  Capped_Signal = clip(Raw_Signal, -2, +2) / 2
  Range: [-1, +1] (-1 = max short, +1 = max long)

Volatility Targeting (per market):
  Contract_risk = contract_value * sigma_return * position_size
  target_risk_per_market = portfolio_vol_target / N_markets
  
  Position_size = target_risk_per_market / (contract_value * sigma_return)
  
  Scaled_position = Capped_Signal * Position_size

Portfolio Volatility Targeting:
  portfolio_vol = realized_vol(portfolio_returns, 20)
  
  If portfolio_vol > 1.5 * target_vol:
    Scale all positions by target_vol / portfolio_vol (deleverage)
  If portfolio_vol < 0.5 * target_vol:
    Scale up (unless in crisis regime)

  target_vol = 10-15% annually (typical for CTA)
```

**Signal:**
- **Per market:** EWMA crossover signal scaled by inverse volatility and capped
- **Long bias:** Capped_Signal > +0.1 = long; < -0.1 = short; between = flat
- **Portfolio:** 50+ markets, each contributing target_risk/N to total risk
- **Deleverage:** If portfolio vol > 1.5x target, proportionally reduce all positions

**Risk:** Target vol 10-15%; Per-market risk capped; Correlation limits across positions
**Edge:** This is the canonical CTA/managed futures strategy implemented by firms managing $50B+ collectively. The edge comes from: (1) 50+ diversified markets provide ~5 independent bets at any time, (2) volatility targeting ensures consistent risk regardless of market conditions, (3) EWMA crossover captures the persistent serial correlation in futures prices, and (4) the strategy has persistent negative correlation to equities during crises (crisis alpha). The crisis alpha property makes this strategy a genuine diversifier in traditional portfolios.

---

### 316 | Earnings Surprise Drift Strategy (PEAD)
**School:** Academic (Ball & Brown, 1968; Bernard & Thomas, 1989) | **Class:** Event Alpha
**Timeframe:** Quarterly | **Assets:** Equities

**Mathematics:**
```
Post-Earnings Announcement Drift (PEAD):
  SUE = (EPS_actual - EPS_estimate) / std(EPS_surprise_history)
  (Standardized Unexpected Earnings)
  
  PEAD Effect:
    Stocks with SUE > +2: drift +2-3% over 60 days post-announcement
    Stocks with SUE < -2: drift -2-3% over 60 days post-announcement
    
    This drift should NOT exist in efficient markets
    But it persists because:
      1. Analyst underreaction to earnings surprises
      2. Institutional investor delayed response
      3. Information processing costs for small-cap stocks

Strategy:
  On earnings announcement day:
    If SUE > +2: Buy next open, hold 60 trading days
    If SUE < -2: Short next open, hold 60 trading days
    
  Enhanced with:
    Volume filter: surprise with high volume > low volume (3x more drift)
    Revenue surprise: confirmation if revenue also beats (reduces false signals by 40%)
    Guidance: upward guidance revision + earnings beat = strongest signal

PEAD Decay:
  60% of drift occurs in first 20 days
  30% in days 20-40
  10% in days 40-60
  Optimal: enter day 1, aggressive exit at day 20, trailing to day 60
```

**Signal:**
- **Buy:** SUE > +2 AND volume spike AND revenue beat (triple confirmation)
- **Short:** SUE < -2 AND volume spike AND revenue miss
- **Size:** Proportional to SUE magnitude (larger surprise = larger position)
- **Exit:** 60-day time stop; accelerated exit if 20-day return > 80% of expected drift

**Risk:** Risk 1% per position; diversify across 20+ earnings events per quarter
**Edge:** PEAD is the oldest and most well-documented anomaly in finance (Ball & Brown, 1968). Despite 55+ years of academic publication, it persists because the information processing mechanism that causes it (analyst underreaction to earnings surprises) is structural, not arbitrageable away. The triple-confirmation variant (SUE + volume + revenue) increases the signal quality by ~40% over pure SUE, capturing only the highest-conviction earnings surprises.

---

### 317 | Intraday Momentum Pattern (Open-to-Close)
**School:** Academic/Quantitative | **Class:** Intraday Pattern
**Timeframe:** Daily (uses intraday data) | **Assets:** US Equities / ETFs

**Mathematics:**
```
Intraday Momentum Pattern:
  First_Half_Return = Price_1pm / Price_Open - 1
  Last_Half_Return = Price_Close / Price_1pm - 1
  
  Intraday Momentum Effect:
    If First_Half_Return > +0.5%: Last_Half_Return tends to be positive (+62% probability)
    If First_Half_Return < -0.5%: Last_Half_Return tends to be negative (+58% probability)
    
    = Intraday momentum (morning direction continues into afternoon)

Open-to-Close Predictability:
  Overnight_Return = Open / Close[1] - 1
  
  If Overnight_Return > +1%: Day return tends to be positive (+55%)
  If Overnight_Return < -1%: Day return tends to be negative (+57%)
  
  = Gap continuation effect

Implementable Strategy:
  At 1pm each day:
    If First_Half_Return > +0.5%: Buy, hold to close (afternoon momentum)
    If First_Half_Return < -0.5%: Short, hold to close
    
  Average return per trade: +0.15% (before costs)
  Frequency: ~120 signals per year per market
  
  Execution: Must trade at 1pm (or equivalent for your market)
  Close: Always close by end of day (no overnight risk)
```

**Signal:**
- **Buy (1pm):** First-half return > +0.5% (morning momentum predicts afternoon)
- **Short (1pm):** First-half return < -0.5%
- **Volume filter:** Only when morning volume > average (higher conviction)
- **Close:** Always by market close (intraday only, no overnight risk)

**Risk:** Max loss: daily stop at 1% adverse from entry; Pure intraday, no overnight risk
**Edge:** Intraday momentum exists because information flow during the morning creates positioning that continues into the afternoon. Institutional orders that begin in the morning are often not fully executed by midday, continuing their price impact in the afternoon. The 62% hit rate for afternoon continuation of >0.5% morning moves is statistically robust across 20+ years of data. The strategy has ZERO overnight risk, making it particularly attractive for risk-conscious traders.

---

### 318 | Event-Driven Catalyst Strategy
**School:** New York (Hedge Fund) | **Class:** Event-Driven
**Timeframe:** Event-based (days to weeks) | **Assets:** Equities

**Mathematics:**
```
Catalyst Scoring Model:
  For each upcoming corporate event:
    Event_Type_Score:
      Earnings: base = 3
      FDA decision: base = 5
      M&A: base = 4
      Activist campaign: base = 4
      Restructuring: base = 3
      Share buyback: base = 2
    
    Probability_Adjustment:
      P(positive_outcome) estimated from analyst consensus, historical patterns
      Adj_Score = Event_Type_Score * P(positive) - Event_Type_Score * P(negative) * loss_ratio

    Timing_Premium:
      Days_to_event: closer = higher premium
      T_premium = max(0, 1 - days_to_event / 30)

    Vol_Premium:
      IV_rank > 50 AND putting on directional trade: vol premium positive
      (options market pricing more movement than historical suggests)

    Catalyst_Score = Adj_Score * T_premium * (1 + Vol_Premium)

Portfolio:
  Select top 10 catalysts by Catalyst_Score
  Long: events where P(positive) > 0.6
  Short/Hedge: events where P(positive) < 0.4
  Size: proportional to Catalyst_Score
```

**Signal:**
- **Entry:** 5-10 days before catalyst event (capture pre-event drift)
- **Direction:** Based on P(positive_outcome) from analyst consensus and historical patterns
- **Size:** Proportional to Catalyst_Score (combines event magnitude, probability, timing)
- **Exit:** 1-3 days post-event (capture announcement effect, avoid noise)

**Risk:** Risk 1-2% per event; diversify across 10+ events; always size for worst-case outcome
**Edge:** Event-driven strategies exploit the systematic mispricing that occurs around corporate events. Markets tend to underreact pre-event (information asymmetry) and overreact post-event (emotional response). By scoring catalysts quantitatively and entering before the event, you capture both the pre-event drift (3-5 days before) and the announcement effect. The timing premium and probability adjustment ensure you take higher-conviction trades when events are imminent and outcomes are clearer.

---

### 319 | Machine Learning Feature Importance Factor
**School:** Quantitative/Modern | **Class:** ML-Driven Factor
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Feature Engineering (200+ raw features):
  Price-based: Returns (1M, 3M, 6M, 12M), Volatility, Skewness, Kurtosis
  Fundamental: P/E, P/B, ROE, Debt/Equity, Earnings Growth, Revenue Growth
  Technical: RSI, MACD, Bollinger %B, Volume Trend, Breakout Strength
  Market Structure: Short Interest, Options Skew, Institutional Ownership

Gradient Boosting Model (XGBoost / LightGBM):
  Target: next_month_return > median (binary classification)
  
  Model trained on rolling 60-month window
  Features selected by permutation importance
  
  Hyperparameters: optimized via time-series cross-validation (expanding window)
  
  Feature Importance Ranking:
    Top 5 features typically include:
      1. 12M-1M momentum (~15% importance)
      2. Earnings revision ratio (~12%)
      3. Short interest change (~10%)
      4. Volatility change (~8%)
      5. Price-to-book z-score (~7%)

Meta-Strategy:
  Instead of using ML predictions directly:
  Extract FEATURE IMPORTANCE and build INTERPRETABLE factor from top features
  
  Composite_ML_Factor = sum(importance_i * z_score_feature_i) for top 10 features
  Long: top quintile by Composite_ML_Factor
  Short: bottom quintile
```

**Signal:**
- **Long:** Top quintile by ML-derived composite factor (highest predicted probability)
- **Short:** Bottom quintile
- **Feature monitoring:** Track feature importance stability (sudden change = regime shift)
- **Rebalance:** Monthly with retraining every 6 months

**Risk:** Market-neutral; Sector-neutral; Risk 1% per position; Model decay monitoring
**Edge:** Using ML to identify the most important features (rather than to make direct predictions) combines the pattern-finding power of ML with the interpretability and robustness of factor models. Direct ML predictions overfit; ML feature importance is more stable because it identifies WHICH factors matter, not what the exact prediction is. This approach has produced Sharpe ratios of 0.8-1.2 in academic studies while remaining fully interpretable and auditable.

---

### 320 | Bayesian Model Averaging Signal Ensemble
**School:** Academic/Bayesian | **Class:** Model Ensemble
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Bayesian Model Averaging:
  Models: M_1, M_2, ..., M_K (K competing models)
  
  For each model M_k:
    p(y_{t+1} | M_k, D_t) = predictive distribution under model k
    p(M_k | D_t) = posterior model probability (how likely model k is correct)
  
  BMA Prediction:
    p(y_{t+1} | D_t) = sum_k p(y_{t+1} | M_k, D_t) * p(M_k | D_t)
    = weighted average of predictions, weighted by model probability
  
  Model Probabilities (via BIC approximation):
    BIC_k = -2 * loglik_k + p_k * log(n)
    p(M_k | D_t) proportional to exp(-BIC_k / 2) / sum_j exp(-BIC_j / 2)

Model Suite (example):
  M_1: Kalman filter (Gaussian state-space)
  M_2: Student-t Kalman filter (heavy tails)
  M_3: GARCH(1,1)
  M_4: Regime-switching (2-state)
  M_5: Random walk (null model)
  
  Each model produces a predictive distribution
  BMA weights determine which models are most consistent with recent data
  
  If market is trending: Kalman filter gets high weight
  If volatile: Student-t gets high weight
  If regime-changing: Regime-switching gets high weight
  If unpredictable: Random walk gets high weight (stay out)

Signal:
  BMA_mean = sum(mean_k * w_k)
  BMA_var = sum((var_k + mean_k^2) * w_k) - BMA_mean^2
  
  Trade when: |BMA_mean| / sqrt(BMA_var) > threshold (high signal-to-noise)
```

**Signal:**
- **Long:** BMA mean > 0 AND BMA Sharpe (mean/sqrt(var)) > 1 (confident positive forecast)
- **Short:** BMA mean < 0 AND |BMA Sharpe| > 1
- **No trade:** Random walk model has highest weight (market unpredictable)
- **Size:** Proportional to BMA Sharpe

**Risk:** Size by BMA variance; higher uncertainty = smaller position; Risk 1-2%
**Edge:** BMA is the gold standard for prediction under model uncertainty because it accounts for the fact that we DON'T KNOW which model is correct. Instead of betting on one model, BMA weights all models by their posterior probability. When the random walk model receives the highest weight, BMA tells you the market is unpredictable -- a critical signal to stay out that single-model approaches can never provide. The BMA variance also captures model disagreement, enabling optimal position sizing.

---

### 321 | Seasonal Factor with Anomaly Calendar
**School:** Academic/Various | **Class:** Calendar Effect
**Timeframe:** Daily / Monthly | **Assets:** All markets

**Mathematics:**
```
Calendar Anomaly Composite:
  1. January Effect (small caps):
     Small_Jan = mean(small_cap_return, Jan) - mean(small_cap_return, Feb-Dec)
     If positive: overweight small cap in December-January
  
  2. Sell-in-May (Halloween Effect):
     Summer_Return = mean(return, May-Oct)
     Winter_Return = mean(return, Nov-Apr)
     Spread = Winter - Summer  (typically +5-10% annually for equities)
  
  3. Turn-of-Month:
     TOM_Return = mean(return, day -1 to day +3 of month)
     Rest_Return = mean(return, day +4 to day -2)
     TOM accounts for ~70% of monthly returns historically
  
  4. Day-of-Week:
     Monday = weakest day (historically negative)
     Friday = strongest day
  
  5. Pre-Holiday:
     Day before holiday: +0.2% average (3x normal daily return)

Composite Seasonal Score:
  S = w_1*Jan_effect + w_2*May_effect + w_3*TOM + w_4*DOW + w_5*PreHoliday
  
  Each component scored -1 to +1 based on current calendar position
  S > +0.5: Favorable seasonal period (overweight)
  S < -0.5: Unfavorable seasonal period (underweight)
```

**Signal:**
- **Overweight:** November-April (winter), Turn-of-Month days, Pre-holiday, Fridays
- **Underweight:** May-October (summer), Mid-month, Mondays
- **January small-cap tilt:** Overweight small caps in late Dec through January
- **Combine with other signals:** Seasonal as confirmation (not standalone entry)

**Risk:** Seasonal effects are AVERAGES, not guarantees; use as tilt, not binary signal
**Edge:** Calendar anomalies are among the most persistent patterns in finance because they're driven by structural mechanisms: mutual fund flows (turn of month), tax-loss harvesting (January), vacation-reduced trading (summer), pre-holiday optimism. The composite approach is essential because no single calendar effect is reliable enough to trade alone. The combined seasonal score, used as a TILT on other strategies, adds 1-2% annually with minimal additional risk.

---

### 322 | Fractal Adaptive Moving Average (FRAMA) System
**School:** Academic/Australian (John Ehlers) | **Class:** Fractal-Adaptive
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
FRAMA (Fractal Adaptive Moving Average):
  Step 1: Compute fractal dimension
    N = period (default 16, must be even)
    N1_High = highest(High, N/2, first half)
    N1_Low = lowest(Low, N/2, first half)
    N2_High = highest(High, N/2, second half)
    N2_Low = lowest(Low, N/2, second half)
    N3_High = highest(High, N)
    N3_Low = lowest(Low, N)
    
    D1 = (N1_High - N1_Low) / (N/2)
    D2 = (N2_High - N2_Low) / (N/2)
    D3 = (N3_High - N3_Low) / N
    
    D = (log(D1 + D2) - log(D3)) / log(2)  (fractal dimension)
  
  Step 2: Convert fractal dimension to alpha
    alpha = exp(-4.6 * (D - 1))
    Clip alpha between 0.01 and 1.0
    
    D near 1.0 (trending): alpha near 1.0 (fast MA, close to price)
    D near 2.0 (random): alpha near 0.01 (very slow MA, flat)
  
  Step 3: FRAMA
    FRAMA_t = alpha * Close + (1 - alpha) * FRAMA_{t-1}
```

**Signal:**
- **Buy:** Close crosses above FRAMA (trend change confirmed by fractal-adaptive MA)
- **Sell:** Close crosses below FRAMA
- **Trend quality:** Lower D (closer to 1.0) = higher quality trend = larger position
- **Exit:** FRAMA crossover in opposite direction

**Risk:** Stop at FRAMA; Trail with FRAMA; Risk 1%
**Edge:** FRAMA adapts its smoothing based on the FRACTAL DIMENSION of the price series. When D is near 1 (smooth, trending), FRAMA becomes fast and tracks price closely. When D is near 2 (noisy, random), FRAMA becomes extremely slow, essentially ignoring the noise. This is mathematically optimal because it uses the actual complexity of the price series to determine how much smoothing is needed, rather than relying on fixed periods or momentum-based adaptation.

---

### 323 | Risk Parity with Trend Overlay
**School:** Bridgewater/AQR | **Class:** Risk-Balanced Portfolio
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
Risk Parity Allocation:
  Assets: Equities, Bonds, Commodities, Gold, TIPS (5 asset classes)
  
  For equal risk contribution:
    w_i * sigma_i * corr(R_i, R_portfolio) = constant for all i
    
    Simplified (assuming low correlations):
    w_i proportional to 1 / sigma_i
    
    sigma_i = EWMA_vol(asset_i, 60) * sqrt(252)

Trend Overlay:
  For each asset:
    Trend_signal = sign(EMA(price, 10 months) - EMA(price, 24 months))
    
    If Trend_signal = +1: keep risk parity weight (or increase slightly)
    If Trend_signal = -1: ZERO weight (remove from portfolio)
  
  Redistribute zero-weight allocation to remaining positive-trend assets
  
  This creates "Risk Parity + Trend" (RPT):
    Risk parity provides base diversification
    Trend overlay removes assets in downtrends
    Combination avoids the #1 risk parity problem: all assets falling simultaneously

Performance Comparison:
  Risk Parity: Sharpe ~0.7, MaxDD ~15%
  Trend Only: Sharpe ~0.6, MaxDD ~12%
  Risk Parity + Trend: Sharpe ~0.9, MaxDD ~8% (synergy exceeds either alone)
```

**Signal:**
- **Base allocation:** Risk parity (inverse vol weighting across 5 asset classes)
- **Trend filter:** Remove any asset class with negative trend (10M EMA < 24M EMA)
- **Rebalance:** Monthly; redistribute from negative-trend to positive-trend assets
- **Leverage:** Scale portfolio to target 10% vol (typical for risk parity)

**Risk:** Max leverage 2x; Monthly rebalancing; Crisis protection via trend filter
**Edge:** Risk parity alone is vulnerable to correlated drawdowns (2008: stocks AND commodities fell). The trend overlay solves this by removing downtrending assets, preventing the portfolio from holding assets in bear markets. The synergy between risk parity (diversification) and trend following (drawdown avoidance) produces risk-adjusted returns superior to either approach alone. This is the foundation of many institutional multi-asset strategies managing $100B+.

---

### 324 | Short Interest Ratio Squeeze Detection
**School:** New York (Short Selling Analysis) | **Class:** Sentiment
**Timeframe:** Bi-Weekly (short interest data) | **Assets:** Equities

**Mathematics:**
```
Short Interest Ratio (SIR):
  SIR = Short_Interest / Average_Daily_Volume  (days to cover)
  
  SIR > 10 days: Heavily shorted (squeeze potential)
  SIR > 20 days: Extremely shorted (high squeeze probability)

Short Squeeze Conditions:
  1. SIR > 10 (heavy short positioning)
  2. Price breaks above recent resistance (shorts start covering)
  3. Volume > 2x average on breakout day (urgency)
  4. Cost_to_borrow > 5% annually (shorts under financing pressure)
  
  Squeeze_Score = SIR/20 + Break_confirmation + Volume_signal + CTB_signal
  If Squeeze_Score > 3: HIGH probability squeeze

Short Interest Change:
  Delta_SI = (SI_current - SI_prior) / SI_prior * 100
  
  Delta_SI < -10%: Shorts covering (squeeze beginning)
  Delta_SI > +10%: New shorts entering (squeeze risk reducing)
  
  Delta_SI < -10% WITH SIR > 10 = ACTIVE SQUEEZE (strongest signal)

Expected Squeeze Magnitude:
  Avg return during short squeeze: +15-30% over 10-20 days
  Duration: typically 5-15 trading days
  Exhaustion: when SIR drops below 5 days (shorts mostly covered)
```

**Signal:**
- **Buy (squeeze):** SIR > 10 AND price breaks resistance AND volume > 2x average
- **Confirm:** Delta_SI < -10% (shorts actively covering = squeeze in progress)
- **Exit:** SIR drops below 5 (squeeze exhaustion) OR price reverses from parabolic
- **Avoid:** SIR > 10 but no price breakout (heavy shorts CAN be right)

**Risk:** Squeezes are volatile; use tight stops; Risk 1-2%; time stop 20 days max
**Edge:** Short squeezes create forced buying that produces returns uncorrelated with fundamentals, market direction, or any other factor. The SIR > 10 threshold identifies stocks where forced buying is physically large relative to daily volume, guaranteeing significant price impact when covering begins. The breakout + volume confirmation ensures you enter after the covering has started, not before (where you'd be hoping for a catalyst). Short squeeze returns are among the highest-magnitude short-term signals in equity markets.

---

### 325 | Macro Regime Indicator (MRI) Dashboard
**School:** Swiss/Institutional | **Class:** Macro Regime
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
Macro Regime Indicator (4 regimes):
  Variables:
    GDP_growth_surprise = GDP_actual - GDP_consensus
    Inflation_surprise = CPI_actual - CPI_consensus
  
  Regime Classification:
    Goldilocks: Growth UP + Inflation DOWN (best for equities)
    Reflation: Growth UP + Inflation UP (good for commodities)
    Stagflation: Growth DOWN + Inflation UP (worst for most assets)
    Deflation: Growth DOWN + Inflation DOWN (good for bonds)
  
  Growth Score:
    G = z(ISM_mfg) + z(initial_claims_inverted) + z(retail_sales_3m_change)
    G > 0: Growth above trend
    G < 0: Growth below trend
  
  Inflation Score:
    I = z(CPI_surprise) + z(breakeven_inflation_change) + z(commodity_index_3m)
    I > 0: Inflation rising
    I < 0: Inflation falling

Asset Allocation per Regime:
  Goldilocks (G>0, I<0):  Equities 50%, Bonds 30%, Gold 10%, Cash 10%
  Reflation (G>0, I>0):   Equities 30%, Commodities 30%, TIPS 20%, Gold 20%
  Stagflation (G<0, I>0): Gold 30%, Commodities 20%, TIPS 30%, Cash 20%
  Deflation (G<0, I<0):   Bonds 50%, Equities 20%, Gold 20%, Cash 10%
```

**Signal:**
- **Goldilocks:** Overweight equities, growth assets (best risk-reward)
- **Reflation:** Overweight commodities, TIPS, real assets
- **Stagflation:** Defensive: gold, TIPS, cash (worst for traditional 60/40)
- **Deflation:** Overweight long-duration bonds (flight to safety)

**Risk:** Regime changes can be sudden; transition gradually (25% per month toward new allocation)
**Edge:** The macro regime framework captures the two most important economic variables for asset prices: growth and inflation. Each combination creates a distinct environment where specific assets perform best. By identifying the current regime and tilting allocation accordingly, you avoid the catastrophic losses of holding the wrong assets in the wrong regime (e.g., equities in stagflation, commodities in deflation). The gradual transition prevents whipsaw from noisy regime signals.

---

"""

with open("Indicators.md", "a") as f:
    f.write(content)

print("Appended strategies 301-325 to Indicators.md")
