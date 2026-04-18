#!/usr/bin/env python3
"""Append strategies 326-350 to Indicators.md"""

content = r"""
### 326 | Value-Momentum Barbell Strategy
**School:** Academic (Asness, Moskowitz, Pedersen, 2013) | **Class:** Factor Combo
**Timeframe:** Monthly | **Assets:** Equities, FX, Bonds, Commodities

**Mathematics:**
```
Value Score (per asset class):
  Equities: Value = -zscore(P/E)  (low P/E = high value)
  FX: Value = zscore(PPP deviation)  (more undervalued vs PPP = higher value)
  Bonds: Value = zscore(real_yield)  (higher real yield = higher value)
  Commodities: Value = zscore(-5yr_return)  (more below 5yr average = higher value)

Momentum Score:
  All classes: MOM = zscore(12M-1M return)

Barbell Composite:
  VM = 0.5 * Value + 0.5 * Momentum
  
  Key Insight: Value and Momentum are NEGATIVELY CORRELATED
    Correlation ~ -0.5 across all asset classes
    When value works, momentum struggles (and vice versa)
    
    50/50 blend HEDGES the crash risk of each:
      Momentum crash (2009): Value rallied, offsetting loss
      Value drawdown (2020): Momentum rallied, offsetting loss
    
    Combined Sharpe: ~1.0 (vs ~0.5 for each individually)

Cross-Asset Diversification:
  VM_equities * 0.25 + VM_fx * 0.25 + VM_bonds * 0.25 + VM_commodities * 0.25
  Diversification across asset classes further reduces drawdowns
```

**Signal:**
- **Long:** Top quintile by VM composite across all asset classes
- **Short:** Bottom quintile by VM composite
- **Equal weight:** 50/50 value-momentum within each asset class
- **Cross-asset:** Equal risk across 4 asset classes

**Risk:** The negative correlation between V and M is the KEY hedge; market-neutral; Risk 1%
**Edge:** The value-momentum barbell exploits the most important empirical finding in factor investing: value and momentum are negatively correlated (-0.5). This means a 50/50 blend has dramatically lower variance than either factor alone, producing a Sharpe ratio approximately DOUBLE that of either factor individually. This is genuine diversification, not repackaging -- the two factors are driven by different behavioral biases (anchoring for value, herding for momentum) that naturally offset.

---

### 327 | Turbulence Index Portfolio Protection
**School:** Academic (Mark Kritzman, 2010) | **Class:** Risk Management
**Timeframe:** Daily | **Assets:** Multi-Asset

**Mathematics:**
```
Mahalanobis Distance (Financial Turbulence):
  d_t = (r_t - mu)' * Sigma^{-1} * (r_t - mu)
  
  r_t = vector of asset returns at time t
  mu = long-run mean return vector
  Sigma = long-run covariance matrix
  
  d_t measures how "unusual" today's return vector is relative to history
  Accounts for BOTH magnitude AND correlation structure
  
  d_t ~ chi-squared with K degrees of freedom (K = number of assets)

Turbulence Index:
  TI_t = d_t  (raw Mahalanobis distance)
  TI_threshold = chi2.ppf(0.95, K)  (95th percentile of chi-squared)
  
  If TI_t > TI_threshold: TURBULENT (unusual joint behavior)
  If TI_t < TI_threshold: CALM (normal behavior)

Portfolio Protection:
  Normal (TI < threshold):
    Run standard portfolio (equities, factor strategies, etc.)
  
  Turbulent (TI > threshold):
    Reduce equity allocation by 50%
    Increase bonds/gold/cash
    Tighten all stop losses
  
  Extreme Turbulence (TI > 3 * threshold):
    Move to 70% cash + 30% treasuries
    Cancel all new entries
    Wait for TI to return below threshold

Historical:
  TI correctly flagged: Oct 1987, Aug 1998, Sep 2008, Mar 2020
  Average advance warning: 1-3 days before major drawdown
```

**Signal:**
- **Normal:** TI < threshold = full risk-on portfolio
- **Elevated:** TI > 1x threshold = reduce risk 50%, tighten stops
- **Crisis:** TI > 3x threshold = max defensive, 70% cash
- **Recovery:** TI drops below threshold for 5+ consecutive days = return to normal

**Risk:** This is a meta-strategy; overlay on any portfolio; reduces returns in calm markets
**Edge:** The Turbulence Index uses Mahalanobis distance to detect when the multivariate return distribution is behaving abnormally -- accounting for BOTH unusual returns AND unusual correlations. Standard risk measures (VaR, vol) miss correlation breaks; the TI catches them because unusual correlation structure increases the Mahalanobis distance even when individual asset returns look normal. This detected the 2008 crisis, COVID crash, and every major event because crises always involve correlation breakdown.

---

### 328 | Absorption Ratio Market Fragility
**School:** Academic (Mark Kritzman, 2010) | **Class:** Systemic Risk
**Timeframe:** Daily | **Assets:** Market-Level

**Mathematics:**
```
Absorption Ratio:
  Given N assets, compute correlation matrix and extract eigenvalues
  
  AR = sum(variance_explained by top K eigenvectors) / total_variance
  
  K = N/5 (top 20% of eigenvectors)
  
  AR near 1: Markets tightly coupled (few factors explain most variance)
    = FRAGILE (shock to one asset affects all)
  
  AR near 0.3: Markets loosely coupled (many independent factors)
    = RESILIENT (shock to one asset stays contained)

AR Dynamics:
  Delta_AR = AR - AR_15day_average
  
  Delta_AR > +1 std: Markets BECOMING more fragile (warning)
  Delta_AR < -1 std: Markets BECOMING more resilient (all clear)

Typical Values:
  S&P 500 sectors (11 sectors):
    Normal: AR ~ 0.65-0.75
    Pre-crisis: AR > 0.85 (2007: AR rose to 0.88 before crash)
    Crisis: AR > 0.90 (all sectors moving together)
    Post-crisis: AR drops back to 0.70

Trading Application:
  AR > 0.85: Reduce net exposure, buy protection (puts, VIX calls)
  AR < 0.70: Markets diversified, increase risk appetite
  Rising AR: Tighten stops on all positions
  Falling AR: Widen stops, add to positions
```

**Signal:**
- **Risk-on:** AR < 0.70 AND falling (markets well-diversified and improving)
- **Cautious:** AR > 0.80 AND rising (markets becoming fragile)
- **Defensive:** AR > 0.85 (high systemic fragility, reduce exposure)
- **Size adjustment:** Position size inversely proportional to AR

**Risk:** AR is a meta-indicator; size and hedge decisions, not entry/exit
**Edge:** The Absorption Ratio measures the degree to which markets are "coupled" -- when AR is high, a shock to any single sector affects all sectors because they're driven by the same few factors. This is the mathematical definition of systemic fragility. AR rose above 0.85 before every major market dislocation in the past 30 years, typically 2-4 weeks before the crash. By reducing exposure when AR is elevated, you avoid the concentrated losses that come from markets all falling simultaneously.

---

### 329 | Cross-Sectional Volatility Factor
**School:** Academic | **Class:** Dispersion Factor
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Cross-Sectional Volatility (CSV):
  CSV_t = std(return_i for all stocks in universe at time t)
  
  CSV_t high: stocks are dispersing (some up a lot, some down a lot)
    = stock picking environment (idiosyncratic factors dominate)
  
  CSV_t low: stocks are tightly correlated (all moving together)
    = market-driven environment (beta dominates)

CSV-Based Strategy Selection:
  When CSV > median:
    Stock-specific factors (alpha strategies) work best
    Run: momentum, value, quality, earnings surprise
    Expect: wide dispersion of returns -> alpha strategies capture spread
  
  When CSV < median:
    Market factors dominate
    Run: market timing, index-level strategies
    Reduce: stock-specific bets (all stocks moving together, no spread)

Dispersion Trading:
  Sell index volatility + Buy single-stock volatility
  Profit when CSV > implied dispersion (i.e., realized > implied)
  
  Long_straddle(stocks) - Short_straddle(index)
  = pure bet on dispersion exceeding what options market expects
```

**Signal:**
- **High CSV:** Activate alpha strategies (momentum, value, quality); stocks are differentiating
- **Low CSV:** Reduce alpha, increase beta management; stocks are correlated
- **Dispersion trade:** If realized CSV > implied CSV, enter dispersion trade (long stock vol, short index vol)
- **Size:** Alpha strategy allocation proportional to CSV level

**Risk:** CSV changes slowly; adjust over weeks, not days; Risk per strategy unchanged
**Edge:** Cross-sectional volatility tells you WHETHER stock picking can work in the current environment. When CSV is low, all stocks move together and stock-specific analysis is useless (macro dominates). When CSV is high, stocks are differentiating and alpha strategies have wide opportunity sets. By adjusting the allocation to stock-picking vs. market-timing based on CSV, you avoid the 40-50% of periods where alpha strategies cannot work regardless of their quality.

---

### 330 | Tactical Tilting via Yield Curve Signals
**School:** Fed/Institutional | **Class:** Macro Signal
**Timeframe:** Monthly | **Assets:** Equities + Bonds

**Mathematics:**
```
Yield Curve Signals:
  1. Slope (10Y - 2Y):
     Slope > 100bp: Steep curve (expansion, growth ahead)
     Slope < 0bp: Inverted curve (recession warning)
  
  2. Level (10Y yield):
     Level rising: tightening financial conditions
     Level falling: easing financial conditions
  
  3. Curvature (2*(5Y) - 2Y - 10Y):
     Positive: belly rich (normal)
     Negative: belly cheap (unusual, often pre-recession)

Yield Curve Regime:
  Bull Steepener: Rates falling + slope rising = BEST for equities
  Bear Steepener: Rates rising + slope rising = GOOD for equities/cyclicals
  Bear Flattener: Rates rising + slope falling = CAUTION for equities
  Bull Flattener: Rates falling + slope falling = RISK-OFF

Tactical Allocation:
  Bull Steepener: Equities 70%, Bonds 20%, Cash 10%
  Bear Steepener: Equities 60%, Commodities 20%, Bonds 10%, Cash 10%
  Bear Flattener: Equities 40%, Bonds 30%, Cash 20%, Gold 10%
  Bull Flattener: Bonds 50%, Gold 20%, Cash 20%, Equities 10%

Inversion Recession Timer:
  After first inversion (10Y-2Y < 0):
    Recession typically follows in 12-18 months
    Equities often RALLY for 6-12 months after inversion
    Then decline into recession
    
    Tactical: Stay long equities for 6 months post-inversion
    Then reduce to defensive allocation
```

**Signal:**
- **Bull steepener:** Maximum equity allocation (rates falling = easy money, slope rising = growth)
- **Bear flattener:** Reduce equities, increase bonds and cash
- **Inversion:** Start 12-month countdown to recession; stay invested 6 months, then defensive
- **Rebalance:** Monthly based on yield curve regime changes

**Risk:** Yield curve changes slowly; avoid reacting to daily noise; Monthly assessment only
**Edge:** The yield curve is the single most reliable recession predictor because it reflects the collective expectation of the entire bond market (the largest, most liquid market in the world). Every US recession since 1960 was preceded by an inverted yield curve. The 4-regime framework (steepener/flattener x bull/bear) provides actionable allocation guidance because each regime has distinct implications for growth and financial conditions.

---

### 331 | Sentiment Composite (AAII + Put/Call + VIX)
**School:** Contrarian | **Class:** Multi-Source Sentiment
**Timeframe:** Weekly | **Assets:** US Equities

**Mathematics:**
```
Sentiment Components:
  1. AAII Bull-Bear Spread:
     AAII_spread = %Bulls - %Bears (weekly survey)
     AAII_z = zscore(AAII_spread, 104 weeks)  (2-year z-score)
  
  2. Put/Call Ratio:
     PCR = total_put_volume / total_call_volume (equity options)
     PCR_z = zscore(PCR, 52 weeks)  (1-year z-score)
     Note: HIGH PCR = bearish sentiment = contrarian bullish
     So: PCR_contrarian_z = -PCR_z (invert for consistency)
  
  3. VIX Term Structure:
     VIX_ratio = VIX / VIX3M (spot vs 3-month VIX)
     VIX_z = zscore(VIX_ratio, 52 weeks)
     VIX_ratio > 1: backwardation (fear) = contrarian bullish
     VIX_contrarian_z = -VIX_z (invert)

Sentiment Composite:
  SC = (AAII_z + PCR_contrarian_z + VIX_contrarian_z) / 3
  
  SC > +1.5: Extreme OPTIMISM (contrarian bearish -> reduce exposure)
  SC < -1.5: Extreme PESSIMISM (contrarian bullish -> increase exposure)
  |SC| < 0.5: Neutral (no sentiment edge)

Historical Performance:
  Buying when SC < -1.5: 12-month forward return +18% average
  Buying when SC > +1.5: 12-month forward return +4% average
  = 14% annual difference based on sentiment extremes
```

**Signal:**
- **Contrarian buy:** SC < -1.5 (extreme pessimism across all three measures)
- **Contrarian sell/reduce:** SC > +1.5 (extreme optimism)
- **Neutral:** |SC| < 0.5 (no actionable sentiment extreme)
- **Use as tilt:** Adjust equity allocation +/-20% based on SC

**Risk:** Sentiment is a TILT indicator, not timing; adjust allocation, don't go all-in or all-out
**Edge:** Combining three independent sentiment measures (survey-based AAII, options-based put/call, volatility-based VIX structure) produces a more reliable contrarian signal than any single measure. When all three independently confirm extreme pessimism, the probability of a significant rally over the following 6-12 months exceeds 80%. The three measures capture different investor populations (retail, options traders, institutional hedgers), ensuring genuine consensus in sentiment.

---

### 332 | Gaussian Mixture Model Clustering Regime
**School:** Academic/ML | **Class:** Unsupervised Regime Detection
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
GMM for Regime Detection:
  Feature vector: X_t = [return_t, volatility_t, volume_t, correlation_t]
  
  Assume K = 3 regimes (clusters):
    p(X_t) = sum_k pi_k * N(X_t | mu_k, Sigma_k)
    
    pi_k = mixing proportions (probability of each regime)
    mu_k = mean of feature vector in regime k
    Sigma_k = covariance matrix in regime k
  
  Estimation: EM algorithm on rolling 500-day window
  
  Regime Assignment:
    gamma_k(t) = P(regime = k | X_t)  (posterior probability)
    Regime_t = argmax_k gamma_k(t)

Typical Regime Discovery:
  Regime 1: mu_return > 0, sigma_vol low, volume normal
    = QUIET BULL (steady uptrend, low vol)
  
  Regime 2: mu_return ~ 0, sigma_vol medium, volume low
    = RANGE/CHOP (sideways, normal vol)
  
  Regime 3: mu_return < 0, sigma_vol high, volume high
    = CRISIS (selloff, high vol, high volume)

Strategy per Regime:
  Quiet Bull: Trend following, buy dips, leverage OK
  Range/Chop: Mean reversion, sell volatility, reduce size
  Crisis: Risk-off, buy protection, short or cash
```

**Signal:**
- **Regime 1 (Quiet Bull):** Full risk-on, trend following, buy dips
- **Regime 2 (Range):** Mean reversion, neutral exposure, sell options
- **Regime 3 (Crisis):** Risk-off, protective positions, cash
- **Transition alert:** gamma_k changing rapidly = regime transition underway

**Risk:** GMM can produce spurious regime changes; require gamma > 0.7 for regime assignment
**Edge:** GMM discovers regimes from the data itself, without imposing predefined categories. This is superior to threshold-based regime detection (e.g., "VIX > 20 = crisis") because GMM considers the JOINT distribution of multiple features simultaneously. The Gaussian mixture captures the multi-modal nature of market behavior: returns are not normally distributed but are well-modeled as a MIXTURE of normals corresponding to different regimes.

---

### 333 | Lottery Demand Factor (MAX Effect)
**School:** Academic (Bali, Cakici, Whitelaw, 2011) | **Class:** Behavioral Anomaly
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
MAX Factor:
  MAX = average of the 5 highest daily returns in the prior month
  
  MAX measures "lottery-like" characteristics:
    High MAX = stock had extreme positive days = lottery-like payoff profile
    Low MAX = stock had moderate daily returns = boring/stable
  
  The MAX anomaly:
    High-MAX stocks (top quintile): UNDERPERFORM by 1-2% per month
    Low-MAX stocks (bottom quintile): OUTPERFORM by 0.5-1% per month
    
    Spread: ~2% per month, ~15-20% annualized
    
    This persists because investors OVERPAY for lottery-like stocks
    (same psychology as lottery tickets: overweigh small probability of huge gain)

Strategy:
  Long: Bottom quintile by MAX (boring, non-lottery stocks)
  Short: Top quintile by MAX (lottery-like, overvalued)
  
  Enhanced with idiosyncratic volatility:
    IVOL = residual volatility after removing market factor
    Low IVOL + Low MAX = MOST boring = highest future returns
    High IVOL + High MAX = MOST lottery-like = lowest future returns

MAX Factor Properties:
  Sharpe: ~0.7-0.9 (one of the highest single-factor Sharpes)
  Correlation with momentum: -0.3 (slight negative = diversifying)
  Correlation with value: +0.2 (slight positive)
  Survives after controlling for size, value, momentum, profitability
```

**Signal:**
- **Long:** Bottom quintile by MAX (non-lottery stocks)
- **Short:** Top quintile by MAX (lottery stocks)
- **Enhanced:** Long low-MAX + low-IVOL; Short high-MAX + high-IVOL
- **Rebalance:** Monthly (MAX changes rapidly)

**Risk:** Market-neutral; Sector-neutral; Risk 1% per position
**Edge:** The MAX effect exploits one of the strongest and most persistent behavioral anomalies: investors systematically overpay for lottery-like stocks (high MAX) and underpay for boring stocks (low MAX). This is driven by prospect theory -- humans overweight small probabilities of large gains. The Sharpe ratio of 0.7-0.9 is among the highest for any published single factor, and the effect has survived in out-of-sample data across 40+ countries.

---

### 334 | Overnight vs Intraday Return Decomposition
**School:** Academic/Microstructure | **Class:** Return Decomposition
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
Return Decomposition:
  Total_Return = Overnight_Return + Intraday_Return
  
  Overnight = Open_t / Close_{t-1} - 1  (close to open)
  Intraday = Close_t / Open_t - 1  (open to close)

Empirical Facts:
  For US equities (1993-2023):
    Average overnight return: +0.04% per day (+10% annualized)
    Average intraday return: -0.01% per day (-2.5% annualized)
    
    ALL of the equity premium comes from overnight returns!
    Intraday returns are approximately ZERO on average

Strategy Implications:
  1. Buy-at-Close, Sell-at-Open (overnight capture):
     Expected: +0.04% daily, +10% annual (before costs)
     Sharpe: ~0.5
  
  2. Short-at-Open, Cover-at-Close (intraday short):
     Expected: +0.01% daily (approximately zero)
     Not profitable after costs
  
  3. Factor-Enhanced Overnight:
     Overnight premium is LARGER for:
       Small caps: +0.06% vs +0.03% for large caps
       High short interest stocks: +0.08%
       Pre-earnings: +0.15% (5 days before earnings)

Implementation:
  At market close: Buy portfolio of small-cap, high-SI stocks
  At market open: Sell entire portfolio
  Net exposure: overnight only (no intraday market risk)
```

**Signal:**
- **Buy at close:** Small-cap stocks with high short interest (largest overnight premium)
- **Sell at open:** Liquidate entire position at open (capture overnight return)
- **Pre-earnings:** Buy 5 days before earnings at close, sell at open each day
- **No intraday exposure:** Strategy runs ONLY overnight

**Risk:** Overnight gap risk exists; diversify across 20+ stocks; Risk 2% per night
**Edge:** The overnight return premium is one of the most robust anomalies in equity markets, driven by the structural feature that informed trading occurs during market hours while uncertainty (which commands a premium) accumulates overnight. By holding stocks ONLY overnight, you capture 100% of the equity premium while being exposed to the market for only 16 out of 24 hours. The enhancement with short interest captures the additional overnight premium created by short-covering pressure at the open.

---

### 335 | Idiosyncratic Volatility Puzzle Factor
**School:** Academic (Ang, Hodrick, Xing, Zhang, 2006) | **Class:** Anomaly
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Idiosyncratic Volatility (IVOL):
  Step 1: Regress stock returns on Fama-French 3 factors (daily, rolling 30 days)
    R_i = alpha + beta_MKT*MKT + beta_SMB*SMB + beta_HML*HML + epsilon
  
  Step 2: IVOL = std(epsilon) * sqrt(252)  (annualized residual volatility)
  
  IVOL Puzzle:
    High IVOL stocks: UNDERPERFORM by 1.1% per month (Ang et al., 2006)
    Low IVOL stocks: OUTPERFORM
    
    This CONTRADICTS theory (higher risk should = higher return)
    But it PERSISTS across 23 developed markets over 40+ years

Strategy:
  Long: Bottom quintile by IVOL (low idiosyncratic risk)
  Short: Top quintile by IVOL (high idiosyncratic risk)
  
  Spread: ~1% per month, ~10-12% annualized
  Sharpe: ~0.6

Explanations (all partial):
  1. Lottery preference (high IVOL = lottery-like -> overpriced)
  2. Short-sale constraints (overpriced high-IVOL stocks can't be shorted easily)
  3. Attention bias (high-IVOL stocks attract retail attention -> overpriced)
  4. Information asymmetry (high IVOL = high info uncertainty -> overpriced by optimists)

IVOL + MAX Combination:
  Long: Low IVOL + Low MAX (doubly boring)
  Short: High IVOL + High MAX (doubly lottery-like)
  Combined Sharpe: ~1.0 (IVOL and MAX capture overlapping but distinct effects)
```

**Signal:**
- **Long:** Bottom quintile by IVOL (low-risk stocks outperform)
- **Short:** Top quintile by IVOL (high-risk stocks underperform)
- **Enhanced:** Combine with MAX factor for double selection
- **Rebalance:** Monthly

**Risk:** Market-neutral; sector limits; IVOL factor can underperform for 6-12 months
**Edge:** The IVOL puzzle is one of the most robust anomalies in asset pricing -- stocks with higher idiosyncratic risk earn LOWER returns, contradicting standard risk-return theory. The anomaly persists because the forces creating it (lottery preference, short-sale constraints, attention bias) are structural features of how humans invest. By systematically buying low-IVOL and shorting high-IVOL, you earn a premium that exists because most investors do the opposite.

---

### 336 | Equity Duration Factor
**School:** Academic/Swiss (Lettau & Wachter, 2007) | **Class:** Duration Factor
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Equity Duration:
  Cash flows of a stock: dividends D_t and terminal value
  
  Equity Duration = sum(t * PV(D_t)) / sum(PV(D_t))
  
  Where PV(D_t) = D_t * exp(-r*t) is the present value of cash flow at time t
  
  Simplified proxy (Weber, 2018):
    Duration_proxy = (1 + 1/r) - (1 + 1/r - D/P) * (D * (1+g) / (P * (r-g)))
    
    Where: r = discount rate, D/P = dividend yield, g = growth rate
  
  Practical approximation:
    Short-duration stocks: high dividend yield, low growth (value stocks)
    Long-duration stocks: low/no dividend yield, high growth (growth stocks)

Interest Rate Sensitivity:
  Long-duration stocks: highly sensitive to discount rate changes
    When rates rise: long-duration stocks fall more
    When rates fall: long-duration stocks rise more
  
  Short-duration stocks: less sensitive
    More stable across rate environments

Strategy:
  In rising rate environment: Long short-duration, Short long-duration
  In falling rate environment: Long long-duration, Short short-duration
  
  Signal: 3-month change in 10-year yield
    Delta_10Y > +50bp: Favor short-duration stocks
    Delta_10Y < -50bp: Favor long-duration stocks
```

**Signal:**
- **Rising rates (Delta_10Y > +50bp):** Long short-duration (high dividend, value), short long-duration (growth)
- **Falling rates (Delta_10Y < -50bp):** Long long-duration (growth), short short-duration (value)
- **Neutral rates:** Equal weight or standard factor exposure
- **Rebalance:** Monthly; transition based on rate regime

**Risk:** Sector concentration risk (duration correlates with sector); Sector-neutral implementation
**Edge:** Equity duration explains WHY growth stocks and value stocks respond differently to interest rates -- it's not about "style" but about the mathematical duration of their cash flow streams. Growth stocks are long-duration assets (most cash flows are far in the future) while value stocks are short-duration (high current dividends). By conditioning equity factor allocation on the rate environment, you avoid the catastrophic mismatch of holding long-duration stocks in rising rate environments.

---

### 337 | Network Centrality Systemic Risk
**School:** Academic/Network Theory | **Class:** Network Risk
**Timeframe:** Monthly | **Assets:** Equities / Financials

**Mathematics:**
```
Stock Return Network:
  Build correlation network: nodes = stocks, edges = correlations
  Edge exists if |corr(R_i, R_j)| > threshold (e.g., 0.5)
  
Centrality Measures:
  1. Degree Centrality:
     DC_i = (number of edges for stock i) / (N-1)
     High DC = stock connected to many others (hub)
  
  2. Betweenness Centrality:
     BC_i = sum(shortest_paths_through_i / total_shortest_paths) for all pairs
     High BC = stock is a bridge/conduit for information/contagion
  
  3. Eigenvector Centrality:
     EC_i = principal eigenvector component for stock i
     High EC = connected to other highly-connected stocks (systemically important)

Systemic Risk Application:
  Portfolio_centrality = weighted_avg(EC_i, w_i)
  
  If Portfolio_centrality high: Portfolio exposed to systemic risk
    = will fall hard in market crises (correlated with everything)
  
  If Portfolio_centrality low: Portfolio has independent risk
    = less affected by market-wide events

Strategy:
  Long: Low centrality stocks (independent, less systemic)
  Short: High centrality stocks (overexposed to systemic events)
  
  Or: Use centrality as RISK FILTER
    Remove high-centrality stocks from any long-only portfolio
    Remaining portfolio has better crisis-period performance
```

**Signal:**
- **Select:** Stocks with low eigenvector centrality (independent of system)
- **Avoid:** Stocks with high eigenvector centrality (systemic risk hubs)
- **Hedge:** If portfolio centrality rises, add hedges or reduce exposure
- **Rebalance:** Monthly (network structure changes slowly)

**Risk:** Low-centrality stocks may have lower liquidity; size constraints; Risk 1%
**Edge:** Network centrality measures provide information about systemic risk that standard factor models miss. A stock can have low beta (market sensitivity) but high centrality (connected to many other stocks through non-market channels). During crises, these high-centrality stocks transmit and amplify shocks. By avoiding systemically central stocks, you reduce portfolio exposure to network contagion -- the primary mechanism by which crises spread across seemingly unrelated assets.

---

### 338 | Implied Volatility Surface Momentum
**School:** Derivatives/Quantitative | **Class:** Vol Surface Signal
**Timeframe:** Weekly | **Assets:** Equities with Listed Options

**Mathematics:**
```
IV Surface:
  IV(K, T): Implied volatility as function of Strike (K) and Expiry (T)
  
  Key Surface Features:
    ATM_IV: At-the-money implied volatility (overall vol level)
    Skew: IV(25D put) - IV(25D call) (crash premium)
    Term_Structure: IV(3M) - IV(1M) (vol expectations)
    Kurtosis: IV(10D put) + IV(10D call) - 2*ATM_IV (tail premium)

Surface Momentum:
  For each feature, compute 5-day change:
    Delta_ATM_IV, Delta_Skew, Delta_Term, Delta_Kurtosis
  
  Surface Momentum Signal:
    SM = -w1*Delta_ATM_IV - w2*Delta_Skew + w3*Delta_Term - w4*Delta_Kurtosis
    
    Negative ATM_IV change: vol declining (bullish for stock)
    Negative Skew change: less crash premium (bullish)
    Positive Term change: vol curve normalizing (bullish)
    Negative Kurtosis change: less tail premium (bullish)
    
    w = [0.3, 0.3, 0.2, 0.2]

Trading:
  SM > +1 sigma: Options market becoming bullish (buy stock)
  SM < -1 sigma: Options market becoming bearish (sell/short stock)
  
  Lead time: IV surface changes lead stock price changes by 1-3 days
```

**Signal:**
- **Buy:** SM > +1 sigma (options market signaling bullish shift)
- **Sell:** SM < -1 sigma (options market signaling bearish shift)
- **Cross-sectional:** Rank stocks by SM; long top decile, short bottom decile
- **Exit:** SM returns to neutral

**Risk:** Stop at 2x ATR; Signal has 1-3 day lead time; Risk 1%
**Edge:** The implied volatility surface embeds the option market's collective forecast of future price dynamics. Changes in the surface (momentum) reveal shifting expectations before they manifest in stock prices because options traders are typically better informed or faster than equity traders. The 1-3 day lead time of IV surface momentum over stock price changes is documented in academic literature and persists because options and stock markets are partially segmented.

---

### 339 | Herding Indicator Cross-Sectional
**School:** Behavioral Finance | **Class:** Crowd Behavior
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Cross-Sectional Return Dispersion:
  CSAD_t = (1/N) * sum(|R_{i,t} - R_{m,t}|)
  (Cross-Sectional Absolute Deviation of returns from market return)

Herding Detection (Chang, Cheng, Khorana, 2000):
  Under rational pricing: CSAD increases linearly with |R_m|
  Under herding: CSAD increases less than linearly (or decreases) with |R_m|^2
  
  Regression:
    CSAD_t = alpha + beta_1*|R_{m,t}| + beta_2*R_{m,t}^2 + epsilon
  
  If beta_2 < 0: HERDING present (stocks converging during large market moves)
  If beta_2 > 0: No herding (stocks dispersing during large moves)

Herding Intensity Index:
  HII = -beta_2 (from rolling 120-day regression)
  
  HII > 0: Herding active (stocks moving together excessively)
    = contrarian opportunities exist (individual stocks mispriced)
  
  HII < 0: Anti-herding (excessive dispersion)
    = momentum works well (stocks differentiating on fundamentals)

Strategy:
  HII > 1 std above mean: Activate contrarian/mean-reversion strategies
    Stocks have been pushed too far together -> individual reversals likely
  
  HII < 1 std below mean: Activate momentum strategies
    Stocks are differentiating -> trends in individual stocks are genuine
```

**Signal:**
- **High herding (HII high):** Use contrarian strategies; stocks mispriced together will differentiate
- **Low herding (HII low):** Use momentum strategies; stock-level trends are information-driven
- **Meta-signal:** HII determines WHICH strategy class to deploy
- **Rebalance:** Monthly based on rolling HII

**Risk:** HII is a regime indicator; affects strategy selection, not individual position sizing
**Edge:** Herding creates systematic mispricing because individual stocks are dragged along with the crowd regardless of their fundamentals. When herding is detected (beta_2 significantly negative), it means stocks that should have differentiated have instead converged, creating mean-reversion opportunities. Conversely, when herding is absent, stock-level momentum reflects genuine information flow. This meta-signal for strategy selection is more valuable than any single entry/exit rule.

---

### 340 | Credit Impulse Leading Indicator
**School:** Institutional/Macro (Michael Biggs, 2009) | **Class:** Macro Leading Indicator
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
Credit Impulse:
  Credit_Flow = Change in total credit outstanding (monthly)
  CI = Change(Credit_Flow) / GDP  (acceleration of credit relative to GDP)
  
  CI is the SECOND DERIVATIVE of credit:
    Credit level rising + Credit flow rising + CI positive:
      = credit accelerating (strongest stimulus)
    Credit level rising + Credit flow rising + CI negative:
      = credit decelerating (stimulus fading)
    Credit level falling + Credit flow negative + CI negative:
      = credit contracting and accelerating downward (crisis)

Historical Lead Time:
  CI leads GDP growth by 3-6 months
  CI leads equity markets by 2-4 months
  CI leads corporate bond spreads by 1-3 months

Trading Application:
  CI turning positive from negative: MAJOR bullish signal
    Equities rally 3-6 months later
    Corporate bonds tighten
    Commodities often rise
  
  CI turning negative from positive: MAJOR bearish signal
    Equities weaken 3-6 months later
    Corporate bonds widen
    Commodities may fall

Country-Level CI:
  Compute CI for US, China, Eurozone, Japan separately
  Global_CI = GDP_weighted_average(country_CIs)
  
  Global CI positive: risk-on across all asset classes
  Global CI negative: risk-off
```

**Signal:**
- **Bullish:** CI turns positive (credit acceleration beginning = growth ahead)
- **Bearish:** CI turns negative (credit deceleration = slowdown ahead)
- **Global risk-on:** Global CI positive (all major economies accelerating)
- **Implement:** 2-4 month lead; adjust allocation BEFORE markets move

**Risk:** CI is a LEADING indicator; can be early; Transition allocation gradually over 2-3 months
**Edge:** Credit impulse is the single best leading indicator for global growth because credit creation IS the mechanism through which monetary policy affects the real economy. By measuring the ACCELERATION of credit (not the level), you detect the inflection points where monetary stimulus begins to bite or where tightening begins to hurt. The 2-4 month lead over equity markets provides ample time for position adjustment. CI correctly signaled every major equity market turn since 2000.

---

### 341 | Relative Value Multi-Country Equity
**School:** London/Institutional | **Class:** Global Macro Equity
**Timeframe:** Monthly | **Assets:** Country Equity Indices

**Mathematics:**
```
Country Valuation:
  CAPE_ratio = country_CAPE (Shiller P/E)
  CAPE_z = (CAPE - mean(CAPE, 20yr)) / std(CAPE, 20yr)
  
  Value_rank = rank(countries, by CAPE_z ascending)
  (lowest CAPE_z = most undervalued = highest value rank)

Country Momentum:
  MOM = 12M-1M return for each country index
  MOM_rank = rank(countries, by MOM descending)
  (highest momentum = highest momentum rank)

Country Quality (Economic):
  Quality = z(GDP_growth) + z(current_account/GDP) + z(-government_debt/GDP)
  Quality_rank = rank(countries, by Quality descending)

Composite Country Score:
  CS = 0.4*Value_rank + 0.3*MOM_rank + 0.3*Quality_rank
  
  Long: Top 5 countries by CS (undervalued, positive momentum, good fundamentals)
  Short: Bottom 5 countries by CS

Currency Hedge Decision:
  If carry positive (foreign rate > domestic): unhedged (earn carry)
  If carry negative: 50% hedged (partial protection)
  If volatility regime = crisis: 100% hedged
```

**Signal:**
- **Long:** Top 5 countries by composite score (value + momentum + quality)
- **Short:** Bottom 5 countries
- **Currency:** Hedge based on carry and volatility regime
- **Rebalance:** Monthly

**Risk:** Country-level diversification; FX risk managed by hedge overlay; Risk 2% per country
**Edge:** Country-level value, momentum, and quality factors have HIGHER Sharpe ratios than stock-level factors because country returns are driven by macro factors that are more persistent and less arbitraged. CAPE mean-reversion at the country level has worked for 100+ years of data. Adding momentum ensures you don't buy value traps (cheap countries getting cheaper) and quality ensures fundamental support.

---

### 342 | Implied Correlation Trading Strategy
**School:** Derivatives/Quantitative | **Class:** Correlation Trade
**Timeframe:** Monthly | **Assets:** Index + Component Options

**Mathematics:**
```
Implied Correlation:
  Index_IV^2 = sum_i sum_j w_i * w_j * IV_i * IV_j * rho_implied(i,j)
  
  Average implied correlation:
    rho_implied = (Index_IV^2 - sum(w_i^2 * IV_i^2)) / (sum_{i!=j} w_i*w_j*IV_i*IV_j)
  
  This is the correlation the options market implies between index components

Correlation Risk Premium:
  Historically: Implied_correlation > Realized_correlation (on average)
  The difference (CRP) averages 5-10 correlation points
  
  This premium exists because:
    1. Portfolio managers buy index puts (pushes up index vol relative to stock vol)
    2. Correlation increases in crashes (implied prices this in permanently)
    3. Demand for index hedges exceeds demand for stock hedges

Dispersion Trade:
  Short correlation (sell implied correlation):
    Sell index straddle + Buy component straddles
    
    Position sizing:
      Sell 1 index straddle notional = $1M
      Buy component straddles totaling = $1M (weighted by index weight)
    
    Profit when realized correlation < implied correlation (the typical case)
    Loss when realized correlation > implied (crash scenario)

  Risk management:
    Max loss = index straddle max loss - component straddle min gain
    Stop: if implied-realized gap narrows to < 2 points
```

**Signal:**
- **Sell correlation (dispersion):** When implied > realized + 8 points (premium elevated)
- **Avoid:** When implied close to realized (no premium to capture)
- **Unwind:** At expiry or when implied-realized gap < 2 points
- **Hedge tail risk:** If VIX > 35, reduce or close position (correlation spikes in crashes)

**Risk:** Tail risk in market crashes (correlation goes to 1); Limit notional; Risk 3-5% of portfolio
**Edge:** The correlation risk premium is one of the most persistent and well-documented risk premiums in derivatives markets. Selling implied correlation via dispersion trades captures a 5-10 point premium that exists because of structural demand imbalances in index vs. component options. The strategy is profitable ~70% of months but has tail risk during market crashes when correlations spike. The key is position sizing: sizing the trade to survive a correlation-1 event while capturing the premium in normal times.

---

### 343 | Attention-Based Trading (Google Trends)
**School:** Behavioral/Quantitative (Da, Engelberg, Gao, 2011) | **Class:** Attention Anomaly
**Timeframe:** Weekly | **Assets:** Equities

**Mathematics:**
```
Google Search Volume Index (SVI):
  SVI_t = Google Trends score for stock ticker (weekly, 0-100)
  
  Abnormal Attention:
    ASVI = log(SVI_t) - log(median(SVI, 8 weeks))
    
    ASVI > 0: More attention than normal (increased retail interest)
    ASVI < 0: Less attention than normal (retail disinterest)

Attention Effect (Da et al., 2011):
  Stocks with high ASVI:
    Week 1: positive return (+0.3%) driven by retail buying
    Weeks 2-8: negative return (-0.5%) as attention fades
    Net: NEGATIVE (attention spike = temporary overpricing)
  
  Trading Strategy:
    Short stocks with ASVI > +1 std (attention spike)
    Expected return: +0.5% over next 4 weeks (capturing reversal)
  
  Long stocks with ASVI < -1 std (attention drought)
    Expected return: +0.2% over next 4 weeks (neglect premium)

Enhanced with Sentiment:
  High ASVI + positive news sentiment: larger initial overreaction
    = BIGGER short opportunity after 1 week
  High ASVI + negative news sentiment: panic selling
    = contrarian buy opportunity (reversal from oversold)
```

**Signal:**
- **Short (after 1 week delay):** ASVI > +1 std + positive sentiment (retail euphoria)
- **Long:** ASVI < -1 std (neglected stocks with neglect premium)
- **Contrarian buy:** ASVI spike + negative sentiment (panic selling creates opportunity)
- **Exit:** 4-week time stop (attention effects decay within 4 weeks)

**Risk:** Short positions carry unlimited risk; strict stop at +5% loss; Risk 0.5%
**Edge:** Retail attention, measured by Google search volume, predicts short-term overpricing because retail investors tend to be net buyers of stocks they search for (Da et al., 2011). This buying pressure creates temporary overpricing that reverses over 2-8 weeks. The academic evidence is strong: abnormal search volume predicts next-month negative returns with a t-statistic > 3 across 15+ years of data. The 1-week delay before shorting avoids the initial positive momentum of the attention spike.

---

### 344 | Supply Chain Network Alpha
**School:** Quantitative/Academic | **Class:** Network Signal
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
Supply Chain Network:
  Build directed graph: Supplier -> Customer relationships
  Data source: SEC filings (10-K, 10-Q) which disclose major customers
  
  Node: Company
  Edge: A -> B means A supplies to B (with revenue weight)

Customer Momentum Signal:
  For stock i:
    Customer_MOM_i = weighted_avg(MOM_j for all customers j of stock i)
    Weight = revenue_share of customer j in stock i's total revenue
  
  Intuition: If my customers are doing well (high momentum),
  my future revenue will be strong -> I should outperform

  Time delay: Customer momentum leads supplier returns by 1-3 months
  (news flows from customers to suppliers with a lag)

Supplier Distress Signal:
  For stock i:
    Supplier_Distress_i = weighted_avg(default_prob_j for all suppliers j of stock i)
    
  If key supplier in distress -> supply chain disruption risk
  = sell signal for stock i

Combined Network Score:
  NS = Customer_MOM_z * 0.6 - Supplier_Distress_z * 0.4
  
  Long: top quintile by NS (strong customer momentum, healthy suppliers)
  Short: bottom quintile (weak customers, distressed suppliers)
```

**Signal:**
- **Long:** Top quintile by Network Score (customer momentum strong + suppliers healthy)
- **Short:** Bottom quintile (customer weakness + supplier distress)
- **Leading signal:** Customer momentum change leads supplier return by 1-3 months
- **Rebalance:** Monthly

**Risk:** Market-neutral; Sector limits; Risk 1% per position; Data quality monitoring
**Edge:** Information flows along supply chain networks with predictable delays: when a customer firm announces strong results, its suppliers' stock prices react slowly (1-3 months). This delay exists because investors analyze companies in isolation rather than tracing supply chain connections. The customer momentum signal provides a genuine information advantage that is too complex for most investors to process manually but straightforward to implement systematically.

---

### 345 | Entropy-Based Market Regime Detection
**School:** Information Theory/Quantitative | **Class:** Information Regime
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Shannon Entropy of Returns:
  Bin daily returns into K quantiles (K = 10)
  p_k = fraction of returns in bin k (over rolling N-day window)
  
  H = -sum(p_k * log2(p_k))  (Shannon entropy)
  
  N = 60 days, K = 10

Entropy Properties:
  Maximum entropy (H = log2(K) = 3.32 bits): returns uniformly distributed
    = MOST RANDOM, LEAST PREDICTABLE market
  
  Low entropy (H << log2(K)): returns concentrated in few bins
    = LESS RANDOM, MORE PREDICTABLE (trending or stuck)

Regime Detection:
  H_z = zscore(H, 252 days)
  
  H_z > +1: HIGH entropy regime (very random, avoid directional bets)
  H_z < -1: LOW entropy regime (predictable, directional strategies work)

Entropy Change (Information Gain):
  Delta_H = H_t - H_{t-5}
  
  Rapidly falling entropy: market BECOMING more predictable
    = trend forming (momentum strategies will work)
  
  Rapidly rising entropy: market BECOMING more random
    = trend ending (switch to mean reversion or reduce exposure)

Transfer Entropy (Cross-Asset):
  TE(X->Y) = information flowing from asset X to asset Y
  If TE(SPY->AAPL) high: SPY returns predict AAPL returns
  If TE(VIX->SPY) high: VIX changes predict SPY returns
  Use TE to identify which assets are leading and which are following
```

**Signal:**
- **Low entropy regime:** Activate trend strategies (market is predictable)
- **High entropy regime:** Reduce exposure or use mean-reversion (market is random)
- **Falling entropy:** Trend forming; enter momentum positions
- **Transfer entropy:** Trade following assets with leading asset as signal

**Risk:** Entropy changes can be abrupt; reduce position size during high-entropy periods
**Edge:** Entropy measures the fundamental INFORMATION CONTENT of the return distribution, which is more rigorous than volatility or any indicator for determining whether a market is tradeable. A market can have high volatility but low entropy (large moves in one direction = trending and predictable). A market can have low volatility but high entropy (small random moves = untradeable). Entropy correctly distinguishes these cases, telling you WHEN to trade, not just WHAT to trade.

---

### 346 | Anchoring Bias Factor (52-Week High Proximity)
**School:** Behavioral (George & Hwang, 2004) | **Class:** Behavioral Momentum
**Timeframe:** Monthly | **Assets:** Equities

**Mathematics:**
```
52-Week High Proximity:
  P_52 = Price / 52_week_high
  Range: [0, 1] (1 = at the 52-week high, 0 = far below)

Anchoring Effect:
  Investors anchor to the 52-week high as a reference point:
    When P_52 near 1.0: Investors reluctant to buy "at the high"
      -> UNDERREACTION to positive news (stock should be even higher)
    
    When P_52 near 0.5: Investors see "bargain" relative to 52-week high
      -> OVERREACTION to reference point (may not be a bargain)

Strategy (George & Hwang):
  Long: Top quintile by P_52 (closest to 52-week high)
  Short: Bottom quintile by P_52 (furthest from 52-week high)
  
  Monthly return: ~1.0% (vs ~0.8% for standard 12-1 momentum)
  
  Key advantage over standard momentum:
    52-week high strategy is LESS prone to momentum crashes
    Because it captures CONTINUOUS nearness to high (not past returns)
    and crashes involve PAST return reversal, not high-proximity reversal

52-Week High + Standard Momentum:
  Intersection stocks (high P_52 AND high momentum):
    Strongest signal: stocks near highs WITH momentum = genuine strength
  
  Divergence stocks (high P_52 BUT low recent momentum):
    Consolidating near high (may break out or break down)
```

**Signal:**
- **Long:** Top quintile by P_52 (near 52-week high)
- **Short:** Bottom quintile by P_52 (far from 52-week high)
- **Highest conviction:** Near high AND positive momentum (double confirmation)
- **Rebalance:** Monthly

**Risk:** Market-neutral; lower crash risk than standard momentum; Risk 1%
**Edge:** The 52-week high proximity strategy outperforms standard momentum AND has lower crash risk because it captures a behavioral bias (anchoring) rather than past returns. Investors systematically underreact to stocks near their 52-week high, creating predictable drift. The strategy avoids momentum crashes because proximity-to-high is a LEVEL variable (0-1) rather than a RETURN variable, and crashes affect past returns but not the high-proximity measure.

---

### 347 | Volatility-of-Volatility (VoV) Premium
**School:** Derivatives/Academic | **Class:** Higher-Order Vol
**Timeframe:** Monthly | **Assets:** Equities, Indices

**Mathematics:**
```
Volatility-of-Volatility:
  vol = realized_volatility(daily_returns, 20)
  VoV = std(vol, 60)  (standard deviation of volatility)
  
  Or from options: VVIX / VIX (ratio = implied VoV)

VoV Risk Premium:
  Investors dislike uncertainty about uncertainty (VoV)
  They pay a premium to hedge VoV (buy options on VIX)
  
  Implied_VoV > Realized_VoV (on average)
  Premium: ~2-5% monthly
  
  Selling VoV:
    Sell VIX options (straddles or strangles)
    Profit when realized VoV < implied VoV (typical)
    Loss when VoV spikes unexpectedly (rare but severe)

Stock-Level VoV Factor:
  For each stock: VoV_i = std(realized_vol_i, 60)
  
  High VoV stocks: volatile volatility (uncertainty about uncertainty)
    = investors demand higher returns (VoV premium)
    REALITY: high VoV stocks UNDERPERFORM (overpriced)
  
  Low VoV stocks: stable volatility (predictable risk)
    = investors accept lower returns
    REALITY: low VoV stocks OUTPERFORM (underpriced)
  
  Strategy: Long low-VoV, Short high-VoV
  Spread: ~0.5-0.8% per month
```

**Signal:**
- **Long:** Bottom quintile by VoV (stable-vol stocks outperform)
- **Short:** Top quintile by VoV (volatile-vol stocks underperform)
- **VIX options:** Sell VIX straddles when VVIX/VIX ratio > historical median (elevated premium)
- **Rebalance:** Monthly

**Risk:** VoV strategies have EXTREME tail risk; strict position limits; Risk 0.5-1%
**Edge:** VoV premium exists because investors have a strong aversion to uncertainty about uncertainty -- they will pay a premium to avoid it. At the stock level, this creates the same pattern as IVOL: high-VoV stocks are overpriced (investors irrationally accept negative expected returns for the "excitement"). At the index level, the VVIX premium provides a persistent return from selling insurance against volatility spikes. The key is sizing for tail risk.

---

### 348 | Geopolitical Risk Premium Factor
**School:** Institutional/Macro | **Class:** Geopolitical Risk
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
Geopolitical Risk Index (GPR) - Caldara & Iacoviello:
  Text-based index from news articles mentioning:
    wars, terrorism, military tensions, nuclear threats
  
  GPR_t = count(geopolitical_words in newspapers) / total_words * scaling
  
  Higher GPR = more geopolitical risk in news
  
  GPR is available monthly since 1900
  Average: GPR ~ 100 (normalized)
  Spikes: GPR > 200 during wars, major geopolitical events

GPR Impact on Assets:
  GPR spike (+1 std):
    Oil: +5% (supply disruption premium)
    Gold: +3% (safe haven demand)
    Equities: -2% (risk aversion)
    Bonds: +1% (flight to quality)
    Defense stocks: +4% (spending expectations)
  
  GPR decline (-1 std):
    Reversal of above effects (but slower, asymmetric)

Strategy:
  GPR Regime:
    Low GPR (< 80): Risk-on; overweight equities, underweight gold/defense
    Normal GPR (80-150): Neutral allocation
    Elevated GPR (150-250): Tilt toward gold, defense, energy; reduce equities
    Crisis GPR (> 250): Full defensive; treasuries, gold, cash
  
  GPR Momentum:
    Rising GPR (3-month trend up): gradually shift defensive
    Falling GPR: gradually shift risk-on
```

**Signal:**
- **Risk-on:** GPR < 80 AND falling (geopolitical calm)
- **Defensive tilt:** GPR > 150 AND rising (tensions escalating)
- **Crisis mode:** GPR > 250 (active geopolitical crisis)
- **Sector tilt:** High GPR: overweight defense, energy, gold miners

**Risk:** GPR changes can be sudden (events are unpredictable); maintain permanent hedges
**Edge:** Geopolitical risk is systematically underpriced because it's non-quantifiable by traditional financial models (CAPM, factor models don't include GPR). The GPR index provides a quantitative measure that predicts asset class returns, particularly for gold, oil, and defense stocks. During geopolitical events, the assets that benefit (gold, defense, energy) move FASTER than the assets that suffer (broad equities), creating asymmetric profit opportunities for prepared portfolios.

---

### 349 | Profitability Factor (Gross Profitability Premium)
**School:** Academic (Robert Novy-Marx, 2013) | **Class:** Quality Factor
**Timeframe:** Quarterly/Monthly | **Assets:** Equities

**Mathematics:**
```
Gross Profitability:
  GP = (Revenue - COGS) / Total_Assets
  
  Why GROSS (not net) profitability?
    Net profitability (ROE, ROA) is manipulable via:
      Depreciation policy, tax strategy, financial engineering
    Gross profitability uses only Revenue and COGS:
      Hardest to manipulate, most persistent, most predictive
  
  GP Factor:
    GP_z = zscore(GP, cross_section)
    Long: top quintile by GP (highest gross profitability)
    Short: bottom quintile by GP (lowest gross profitability)
    
    Spread: ~0.5% per month (~6% annual)
    Sharpe: ~0.6

GP + Value Combination (Novy-Marx):
  Key Finding: Gross profitability and value (B/M) are NEGATIVELY correlated
    High GP stocks tend to have LOW book-to-market (look like growth)
    Low GP stocks tend to have HIGH book-to-market (look like value)
  
  But BOTH factors are independently profitable!
  
  Combined strategy: Long (high GP AND high B/M), Short (low GP AND low B/M)
  Combined Sharpe: ~0.9 (vs ~0.5 for either alone)
  
  This is the "profitable value" strategy:
    Cheap stocks WITH good businesses (not value traps)
```

**Signal:**
- **Long:** Top quintile by GP (highest gross profitability) + value filter (top 50% B/M)
- **Short:** Bottom quintile by GP + anti-value (bottom 50% B/M)
- **Rebalance:** Quarterly (GP changes slowly)
- **Sector-neutral:** Equal GP within each sector

**Risk:** Market-neutral; long-short; Risk 1% per position
**Edge:** Gross profitability is the cleanest measure of business quality because it uses only the most reliable accounting line items (revenue and COGS) that are hardest to manipulate. The negative correlation with value (B/M) creates a natural combination: profitable-value stocks are cheap stocks with genuinely good businesses, not value traps. This combination has one of the highest Sharpe ratios of any two-factor strategy in the academic literature.

---

### 350 | Dynamic Hedge Ratio with Regime Conditioning
**School:** Institutional/Risk Management | **Class:** Adaptive Hedging
**Timeframe:** Daily | **Assets:** Any hedgeable position

**Mathematics:**
```
Regime-Conditional Hedge Ratio:
  Standard hedge ratio: h = -cov(R_asset, R_hedge) / var(R_hedge)
    (OLS regression beta)

Problem: Hedge ratio CHANGES across regimes
  Bull market: beta(stock, index) might be 0.8
  Bear market: beta(stock, index) might be 1.3
  Crisis: beta(stock, index) might be 1.8
  
  Using a single hedge ratio under-hedges in crises (when you need it most)

Regime-Conditional Estimation:
  Step 1: Detect regime (HMM, vol threshold, or VIX level)
  Step 2: Estimate hedge ratio within each regime separately
    h_bull = regression(R_asset, R_hedge) for bull regime days
    h_normal = regression for normal days
    h_crisis = regression for crisis days
  
  Step 3: Current hedge = h_regime_current
    If current regime = bull: hedge = h_bull (light hedge)
    If current regime = crisis: hedge = h_crisis (heavy hedge)

DCC-GARCH Approach:
  Model time-varying correlation:
    rho_t = DCC-GARCH(R_asset, R_hedge)
    sigma_asset_t = GARCH(R_asset)
    sigma_hedge_t = GARCH(R_hedge)
    
    h_t = rho_t * sigma_asset_t / sigma_hedge_t
  
  This gives a DAILY-UPDATED hedge ratio that adapts to current conditions

Implementation:
  Target: constant portfolio variance
  Rebalance hedge when: |h_t - h_current| > threshold (e.g., 0.1)
  Cost: include transaction costs in rebalance decision
```

**Signal:**
- **Increase hedge:** Regime shifts to crisis OR h_t increases significantly
- **Decrease hedge:** Regime shifts to bull OR h_t decreases
- **Rebalance trigger:** |change in h_t| > 0.1 from current hedge ratio
- **Cost-adjusted:** Only rebalance if benefit > transaction cost

**Risk:** Hedging reduces both downside AND upside; accept reduced returns for controlled risk
**Edge:** Static hedge ratios fail when you need them most -- during crises when correlations and betas increase. By conditioning the hedge ratio on the current regime (or using DCC-GARCH for continuous updating), you automatically increase hedge notional when the market becomes more dangerous. This eliminates the gap between theoretical hedge protection and actual hedge performance. The regime-conditional approach captures 85-90% of hedge effectiveness vs. 60-70% for static hedges during crisis events.

---

# SECTION VIII: DERIVATIVES-INFORMED STRATEGIES (351-400)

---

"""

with open("Indicators.md", "a") as f:
    f.write(content)

print("Appended strategies 326-350 to Indicators.md")
