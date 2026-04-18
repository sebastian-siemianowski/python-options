#!/usr/bin/env python3
"""Append strategies 401-425 to Indicators.md"""

content = r"""
### 401 | Global Macro Regime Allocation Framework
**School:** Bridgewater/Institutional | **Class:** Macro Regime
**Timeframe:** Monthly | **Assets:** Multi-Asset Global

**Mathematics:**
```
Four Macro Regimes (2x2 matrix):
  
  Growth\Inflation |  Rising Inflation    |  Falling Inflation
  ────────────────────────────────────────────────────────────
  Rising Growth    |  REFLATION           |  GOLDILOCKS
                   |  Commodities, TIPS   |  Equities, Credit
                   |  EM equities         |  Growth stocks
  ────────────────────────────────────────────────────────────
  Falling Growth   |  STAGFLATION         |  DEFLATION
                   |  Gold, Cash          |  Treasuries, USD
                   |  Energy              |  Defensive equities
  ────────────────────────────────────────────────────────────

Regime Detection:
  Growth Signal:
    G = zscore(ISM_Manufacturing) + zscore(NFP_3m_avg) + zscore(Retail_Sales_YoY)
    G > 0: Rising growth
    G < 0: Falling growth
  
  Inflation Signal:
    I = zscore(CPI_YoY) + zscore(PPI_YoY) + zscore(Breakeven_5Y)
    I > 0: Rising inflation
    I < 0: Falling inflation

Allocation per Regime:
  GOLDILOCKS (G>0, I<0):
    Equities 50%, Credit 20%, Bonds 15%, Alternatives 15%
    
  REFLATION (G>0, I>0):
    Commodities 30%, TIPS 20%, EM Equities 25%, Equities 15%, Cash 10%
    
  STAGFLATION (G<0, I>0):
    Gold 25%, Cash 25%, Energy 20%, TIPS 15%, Commodities 15%
    
  DEFLATION (G<0, I<0):
    Treasuries 40%, USD 20%, Defensive Equity 20%, Cash 20%

Transition Probabilities (Markov):
  From\To        GOLD  REFL  STAG  DEFL
  GOLDILOCKS     0.70  0.15  0.05  0.10
  REFLATION      0.20  0.55  0.20  0.05
  STAGFLATION    0.05  0.15  0.60  0.20
  DEFLATION      0.15  0.05  0.10  0.70
```

**Signal:**
- **Regime identification:** Monthly based on G and I scores
- **Allocation shift:** Gradual (25% per month toward target, avoid whipsaw)
- **Transition alert:** When either G or I crosses zero (regime boundary)
- **Confirmation:** Require 2 consecutive months in new regime before full shift

**Risk:** Regime changes can be abrupt; gradual transition avoids whipsaw; max 25% shift per month
**Edge:** The 2x2 growth-inflation framework is the simplest complete model of macro regimes. Every asset class has a clear regime preference: equities prefer Goldilocks, commodities prefer Reflation, treasuries prefer Deflation, gold prefers Stagflation. By detecting the current regime and allocating accordingly, you systematically avoid the worst environments for each asset class. Historical backtest shows ~200bp annual improvement over static allocation with significantly lower drawdowns during regime changes.

---

### 402 | Dollar Smile Framework (FX Macro)
**School:** FX/London (Stephen Jen, 2001) | **Class:** FX Regime
**Timeframe:** Monthly | **Assets:** USD + Major FX Pairs

**Mathematics:**
```
The Dollar Smile:
  USD strengthens in TWO very different environments:
  
  1. LEFT SIDE (Risk-Off): Global crisis/fear
     USD rises as safe haven (flight to quality)
     Mechanism: USD repatriation, FX reserve demand
     Coincides with: VIX spike, EM weakness, risk-off
  
  2. MIDDLE (Normal): USD weak
     Normal growth environment
     Carry flows to higher-yielding currencies
     USD declines as capital leaves for EM, AUD, NZD
  
  3. RIGHT SIDE (US Exceptionalism): US growth outperformance
     USD rises on strong US growth relative to world
     Mechanism: capital inflows to US assets, rate differentials
     Coincides with: US equities outperforming, US rates rising

Regime Detection:
  Risk Appetite Index: RAI = zscore(-VIX) + zscore(HY_spread) + zscore(EM_spread)
  
  US Growth Differential: GD = US_ISM - Global_PMI
  
  LEFT (Risk-Off): RAI < -1.0
    -> USD long (safe haven), Short EM FX
  
  MIDDLE (Normal): RAI > -0.5 AND GD < 0.5
    -> USD short, Long carry currencies (AUD, BRL, MXN)
  
  RIGHT (US Exceptionalism): RAI > -0.5 AND GD > 0.5
    -> USD long (growth differential), Short EUR, JPY

Performance:
  Dollar smile regime-based FX: Sharpe ~0.7
  Buy-and-hold EURUSD: Sharpe ~0.1
  Improvement from regime conditioning: massive
```

**Signal:**
- **USD long (risk-off):** RAI < -1.0 (crisis regime, left side of smile)
- **USD short (normal):** RAI > -0.5 AND GD < 0.5 (middle of smile, carry works)
- **USD long (exceptionalism):** RAI > -0.5 AND GD > 0.5 (right side)
- **FX pair selection:** Match currency exposure to regime

**Risk:** Regime misidentification; USD can be strong for different reasons; max 3% FX risk
**Edge:** The Dollar Smile captures the non-linear relationship between USD and global conditions that confuses most FX traders. USD can strengthen in BOTH risk-off (left side) and strong-US-growth (right side) environments, but the TRADING EXPRESSION is different in each case. On the left side, you want USD vs EM (safe haven); on the right side, you want USD vs EUR/JPY (growth differential). Understanding which side of the smile is active determines the correct FX trades.

---

### 403 | Carry Factor Universal (Multi-Asset)
**School:** AQR/Institutional | **Class:** Universal Carry
**Timeframe:** Monthly | **Assets:** FX, Rates, Equities, Commodities

**Mathematics:**
```
Carry Definition (per asset class):

FX Carry:
  Carry_fx = interest_rate_foreign - interest_rate_domestic
  Long high-carry currencies, Short low-carry
  
Bond Carry:
  Carry_bond = bond_yield - repo_rate (term premium)
  + Roll_down = yield change from 1 month of aging
  Total_Carry = Yield + Roll_down - Financing
  
Equity Carry:
  Carry_equity = earnings_yield - risk_free_rate (equity risk premium)
  = E/P - r_f
  
Commodity Carry:
  Carry_commodity = -futures_roll = (F_near - F_far) / F_near
  (backwardation = positive carry, contango = negative carry)

Universal Carry Portfolio:
  Within each asset class:
    Long: top tertile by carry
    Short: bottom tertile by carry
    Weight: equal risk across positions
  
  Across asset classes:
    Equal risk allocation to each carry strategy
    4 asset classes * equal risk = 25% risk per class

Cross-Asset Diversification:
  Correlation between carry strategies:
    FX-Bond: +0.15 (slightly positive)
    FX-Equity: +0.10
    FX-Commodity: +0.05
    Bond-Equity: +0.20
    Average pair: +0.12
  
  Low average correlation -> strong diversification benefit
  
  Single-asset carry Sharpe: ~0.4-0.6
  Universal carry Sharpe: ~0.8-1.1 (diversification multiplier)
```

**Signal:**
- **Within asset class:** Long top tertile by carry, Short bottom tertile
- **Across asset classes:** Equal risk to FX, Bond, Equity, Commodity carry
- **Rebalance:** Monthly
- **Risk management:** If any single carry strategy drawdown > 5%, reduce to half size

**Risk:** Carry strategies vulnerable to sudden unwinds; diversification is the primary defense
**Edge:** Universal carry exploits the most fundamental concept in finance: compensation for bearing risk. Every asset class has a version of "carry" (the return from holding the position, ignoring price changes), and in every asset class, high-carry assets outperform low-carry assets on average. The universal portfolio achieves a Sharpe near 1.0 through cross-asset diversification: carry crashes (like JPY carry unwinds) are typically asset-class-specific, so other carry strategies provide offsetting returns.

---

### 404 | Global Liquidity Cycle Trading
**School:** Cross-Market/Macro | **Class:** Liquidity Regime
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
Global Liquidity Index:
  L = w1*Fed_Balance_Sheet_YoY + w2*ECB_Balance_Sheet_YoY 
    + w3*BOJ_Balance_Sheet_YoY + w4*PBOC_Credit_Impulse
  
  Weights: w = [0.35, 0.25, 0.20, 0.20] (GDP-weighted)

Liquidity Cycle Phases:
  Phase 1 - TIGHTENING: L falling from peak
    Central banks reducing balance sheets
    = Risk assets peak, bonds rally
    Allocation: Bonds 40%, Cash 30%, Gold 20%, Equities 10%
  
  Phase 2 - CONTRACTION: L at trough
    Liquidity at minimum, growth weakening
    = Risk assets bottoming, high-quality bonds peak
    Allocation: Bonds 30%, Equities 30%, Gold 20%, Cash 20%
  
  Phase 3 - EASING: L rising from trough
    Central banks expanding balance sheets
    = Risk assets rally, bonds sell off gradually
    Allocation: Equities 50%, EM 20%, Commodities 15%, Bonds 15%
  
  Phase 4 - EXPANSION: L at peak
    Liquidity at maximum, growth strong
    = Risk assets peak, inflation risk rising
    Allocation: Equities 30%, Commodities 25%, TIPS 20%, Cash 15%, Bonds 10%

Liquidity Lead Times:
  L leads equities by 6-9 months
  L leads commodities by 3-6 months
  L leads credit spreads by 3-6 months
  L leads economic growth by 9-12 months
```

**Signal:**
- **Phase detection:** Monthly based on L level and direction
- **Easing (L rising from trough):** Maximum risk-on (equities, EM, commodities)
- **Tightening (L falling from peak):** Maximum risk-off (bonds, gold, cash)
- **Transition:** Gradual allocation shift over 2-3 months

**Risk:** Central bank policy changes can be sudden; maintain 20% cash minimum; Risk 10% target vol
**Edge:** Global liquidity (central bank balance sheets) is the single most important driver of cross-asset returns because it determines the supply of investable capital. When liquidity is expanding, capital flows into risk assets mechanically (central banks create reserves, which banks invest). When liquidity is contracting, capital is withdrawn. The 6-9 month lead time over equity markets provides ample time for allocation adjustment. Every major equity bear market since 2000 was preceded by a liquidity contraction.

---

### 405 | Copper/Gold Ratio Growth Signal
**School:** Commodity/Macro | **Class:** Growth Indicator
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
Copper/Gold Ratio:
  CG = Copper_Price / Gold_Price
  
  Copper: industrial metal, demand driven by manufacturing and construction
    = proxy for GROWTH expectations
  
  Gold: monetary metal, demand driven by uncertainty and inflation
    = proxy for FEAR expectations
  
  CG ratio = Growth Expectations / Fear Expectations

Interpretation:
  CG rising: Growth expectations > Fear (risk-on environment)
  CG falling: Fear > Growth expectations (risk-off environment)
  
  CG correlation with:
    ISM Manufacturing: +0.65 (strong growth proxy)
    10Y yield: +0.60 (higher CG -> higher rates expected)
    S&P 500: +0.50 (equity-bullish)
    VIX: -0.45 (lower fear)

Trading Application:
  CG_z = zscore(CG, 252 days)
  CG_momentum = CG / CG_60day_avg - 1
  
  CG_z > +1 AND CG_momentum > 0:
    Strong growth signal: Overweight equities, cyclicals, EM
    Underweight: bonds, gold, defensives
  
  CG_z < -1 AND CG_momentum < 0:
    Growth concern: Overweight bonds, gold, defensives
    Underweight: equities, cyclicals, EM
  
  CG as bond yield predictor:
    CG leads 10Y yield by 2-3 months
    When CG rises: expect rates to follow (reduce duration)
    When CG falls: expect rates to fall (increase duration)
```

**Signal:**
- **Risk-on:** CG rising AND above 252-day average (growth expectations improving)
- **Risk-off:** CG falling AND below 252-day average (growth concerns)
- **Rate signal:** CG leads 10Y yield by 2-3 months; adjust duration accordingly
- **Allocation:** Shift between cyclical (CG bullish) and defensive (CG bearish)

**Risk:** Commodity-specific factors can distort CG ratio; use with other macro signals
**Edge:** The copper/gold ratio is the market's most real-time growth barometer because both metals trade 24 hours and reflect genuine physical demand (copper) and fear demand (gold) globally. Unlike survey-based indicators (ISM, PMI) which are reported monthly with lags, CG updates continuously. The 2-3 month lead over bond yields and equity sector rotation provides an actionable timing advantage. CG correctly signaled every US recession and recovery since 1990.

---

### 406 | Emerging Market Contagion Early Warning
**School:** Academic/IMF | **Class:** Systemic EM Risk
**Timeframe:** Weekly | **Assets:** EM FX, EM Equities

**Mathematics:**
```
EM Contagion Indicators:

1. DXY (Dollar Index) Stress:
   DXY_z = zscore(DXY, 252 days)
   DXY > +1 std: Dollar strength -> EM stress (USD debt burden)
   
2. EM FX Correlation:
   Average pairwise correlation of EM currencies (20-day rolling)
   Normal: ~0.4
   Contagion: > 0.7 (all EM moving together)
   
3. EM Sovereign Spread:
   EMBI_spread_z = zscore(JPM_EMBI_spread, 252 days)
   EMBI_spread > +1.5 std: EM credit stress

4. EM Capital Flow Proxy:
   EM_ETF_flow_z = zscore(monthly_flows_to_EM_ETFs, 60 months)
   Large outflows (< -1.5 std): capital flight

Contagion Composite:
  CC = 0.3*DXY_z + 0.25*EM_corr_z + 0.25*EMBI_z - 0.2*EM_flow_z
  
  CC > +2.0: HIGH contagion risk (multiple indicators aligned)
  CC > +1.0: ELEVATED risk
  CC < +0.5: NORMAL

Historical Contagion Events:
  1997 Asian Crisis: CC reached 3.5
  1998 Russian Crisis: CC reached 3.2
  2013 Taper Tantrum: CC reached 2.8
  2018 EM Crisis: CC reached 2.5
  2020 COVID: CC reached 4.0
  
  Lead time: CC > 2.0 typically 2-4 weeks before peak EM drawdown
```

**Signal:**
- **Risk-off EM:** CC > 2.0 (high contagion risk, exit EM positions)
- **Reduce EM:** CC > 1.0 (elevated, reduce by 50%)
- **Risk-on EM:** CC < 0.5 AND falling (normal, resume EM allocation)
- **Selection:** Within EM, favor countries with current account surplus (less vulnerable)

**Risk:** EM crises can be severe (-30 to -50%); act decisively when CC elevated; Risk 3%
**Edge:** EM contagion is highly predictable because the same four factors drive every EM crisis: USD strength (increases external debt burden), EM correlation surge (herding/contagion), sovereign spread widening (credit stress), and capital outflows (portfolio reallocation). By monitoring these four factors in a composite, you get 2-4 weeks of advance warning before EM drawdowns peak. This is sufficient to reduce EM exposure and avoid the worst 70-80% of EM crisis losses.

---

### 407 | Global PMI Momentum Cross-Asset
**School:** Institutional/Macro | **Class:** PMI Signal
**Timeframe:** Monthly | **Assets:** Multi-Asset

**Mathematics:**
```
PMI Signal Construction:
  Global_PMI = GDP_weighted_average(country_PMIs)
  
  PMI Level Signal:
    PMI > 50: Expansion (risk-on)
    PMI < 50: Contraction (risk-off)
  
  PMI Momentum Signal (MORE IMPORTANT than level):
    PMI_mom = PMI - PMI_3month_ago
    
    PMI rising (PMI_mom > 0): Economy IMPROVING
      Even if PMI < 50 (contraction improving = early recovery)
    
    PMI falling (PMI_mom < 0): Economy DETERIORATING
      Even if PMI > 50 (expansion deteriorating = late cycle)

Four PMI Regimes:
  PMI > 50 AND rising: ACCELERATING EXPANSION
    Best for: Equities, Cyclicals, EM, Commodities
    Worst for: Bonds, Defensives, Gold
  
  PMI > 50 AND falling: DECELERATING EXPANSION
    Best for: Quality Equities, Bonds (beginning to rally)
    Worst for: Cyclicals, Commodities (fading)
  
  PMI < 50 AND falling: ACCELERATING CONTRACTION
    Best for: Bonds, Gold, Cash, Defensives
    Worst for: Equities, Cyclicals, EM
  
  PMI < 50 AND rising: DECELERATING CONTRACTION (EARLY RECOVERY)
    Best for: Equities (bottom signal), Cyclicals
    Worst for: Bonds (yields about to rise), Gold

Key Insight:
  The MOST PROFITABLE regime is PMI < 50 AND rising (early recovery)
  Because: markets anticipate recovery BEFORE economy confirms it
  Equities rally 20-30% from PMI trough before PMI crosses 50
```

**Signal:**
- **Max risk-on:** PMI < 50 AND rising (early recovery = best forward returns)
- **Sustained risk-on:** PMI > 50 AND rising (accelerating expansion)
- **Reduce risk:** PMI > 50 AND falling (deceleration warning)
- **Defensive:** PMI < 50 AND falling (contraction deepening)

**Risk:** PMI data is monthly with potential revisions; confirm with other indicators
**Edge:** PMI MOMENTUM is more important than PMI level for asset allocation because markets are forward-looking and PMI direction predicts the next 3-6 months of economic trajectory. The single most important signal is PMI below 50 but rising -- this "green shoots" signal has preceded every equity market recovery since 1990 by 3-6 months. Buying equities when PMI is contracting but improving generates the highest forward returns because most investors wait for the all-clear (PMI > 50), which comes too late.

---

### 408 | TED Spread Systemic Risk Signal
**School:** Banking/Macro | **Class:** Interbank Stress
**Timeframe:** Weekly | **Assets:** Multi-Asset

**Mathematics:**
```
TED Spread:
  TED = 3M_LIBOR - 3M_T-bill_yield
  (Now: SOFR-based equivalent or OIS-T-bill spread)
  
  T-bill: risk-free rate (government guarantee)
  LIBOR/SOFR: interbank lending rate (bank credit risk)
  
  TED = compensation banks demand for lending to each other
  = measure of INTERBANK TRUST

Normal: TED ~ 20-50bp (banks trust each other)
Elevated: TED 50-100bp (caution in interbank market)
Crisis: TED > 200bp (banks refusing to lend, systemic risk)

Historical Peaks:
  Oct 2008: TED hit 450bp (banks completely frozen)
  Aug 2007: TED spiked to 250bp (subprime warning)
  Sep 2001: TED hit 100bp (post-9/11)
  Mar 2020: TED hit 150bp (COVID liquidity crisis)

Trading Application:
  TED < 30bp: Full risk-on (no systemic stress)
  TED 30-75bp: Normal, no adjustment
  TED 75-150bp: Reduce equity by 25%, increase treasury allocation
  TED > 150bp: Crisis mode (50% cash, 30% treasuries, 20% gold)
  
  TED momentum matters more than level:
    Rising TED (even from low level): EARLY WARNING
    TED rising > 2bp per week from below 50bp: alert
    
  Lead time: TED typically rises 2-4 weeks before equity drawdown
```

**Signal:**
- **Risk-off:** TED > 75bp AND rising (interbank stress emerging)
- **Crisis mode:** TED > 150bp (systemic risk, maximum defensive)
- **Risk-on:** TED < 30bp AND falling (banking system healthy)
- **Early warning:** TED rising > 2bp/week from below 50bp

**Risk:** TED spikes can be very fast (days, not weeks); have contingency plan
**Edge:** The TED spread directly measures the willingness of banks to lend to each other, which is the most fundamental indicator of financial system health. Unlike equity volatility (VIX) which measures MARKET fear, TED measures SYSTEMIC fear -- the possibility that banks will stop functioning. Every systemic financial crisis in the past 40 years was preceded by TED spread widening. The 2-4 week lead time provides enough warning to shift portfolios to defensive mode before equity markets crash.

---

### 409 | Risk Appetite Indicator Composite
**School:** Institutional/Multi-Signal | **Class:** Risk Sentiment
**Timeframe:** Weekly | **Assets:** Multi-Asset

**Mathematics:**
```
Risk Appetite Index (RAI) Components:
  
  1. VIX Level:
     VIX_z = -zscore(VIX, 252)  (inverted: low VIX = high appetite)
  
  2. High-Yield Spread:
     HY_z = -zscore(HY_OAS, 252)  (inverted: tight spreads = appetite)
  
  3. EM Spread:
     EM_z = -zscore(EMBI_spread, 252)
  
  4. USD:
     USD_z = -zscore(DXY, 252)  (inverted: weak USD = risk appetite)
  
  5. Equity-Bond Correlation:
     EBC_z = zscore(rolling_60d_corr(SPY, TLT), 252)
     Positive corr: risk-off (both driven by growth fears)
     Negative corr: risk-on (normal regime)
  
  6. Small vs Large Cap:
     SL_z = zscore(IWM/SPY, 60)  (small outperformance = risk appetite)

RAI Composite:
  RAI = (VIX_z + HY_z + EM_z + USD_z + EBC_z + SL_z) / 6
  
  RAI > +1.5: EXTREME risk appetite (complacency -> contrarian bearish)
  RAI +0.5 to +1.5: Healthy risk appetite (momentum favorable)
  RAI -0.5 to +0.5: Neutral
  RAI -1.5 to -0.5: Low risk appetite (caution)
  RAI < -1.5: EXTREME risk aversion (panic -> contrarian bullish)

Dual Use:
  Trend-following: go WITH RAI between -1.5 and +1.5
  Contrarian: go AGAINST RAI when beyond +/- 1.5
```

**Signal:**
- **Risk-on:** RAI between +0.5 and +1.5 (healthy risk appetite, trend-follow)
- **Risk-off:** RAI between -1.5 and -0.5 (low appetite, reduce exposure)
- **Contrarian buy:** RAI < -1.5 (extreme panic, historically good entry)
- **Contrarian sell:** RAI > +1.5 (extreme complacency, reduce risk)

**Risk:** RAI is a composite; individual components can diverge; require 4/6 agreement for extremes
**Edge:** The six-component RAI captures risk appetite from credit (HY, EM), equity (VIX, small/large), FX (USD), and cross-asset correlation (equity-bond). This breadth makes the composite far more reliable than any single indicator. The dual use (trend-following in normal ranges, contrarian at extremes) captures the empirical fact that risk appetite is persistent within ranges but mean-reverting at extremes. Historical 12-month returns after RAI < -1.5 average +22% (vs +8% unconditional).

---

### 410 | Fiscal Impulse Trading Signal
**School:** Macro/Policy | **Class:** Fiscal Policy
**Timeframe:** Quarterly | **Assets:** Equities, Bonds, FX

**Mathematics:**
```
Fiscal Impulse:
  FI = Change(Government_Deficit / GDP)
  
  FI > 0: Government spending ACCELERATING relative to GDP
    = fiscal stimulus (positive for growth, negative for bonds)
  
  FI < 0: Government spending DECELERATING relative to GDP
    = fiscal drag (negative for growth, positive for bonds)

Country-Level Fiscal Impulse:
  Compute FI for major economies: US, China, Eurozone, Japan, UK
  
  Global_FI = GDP_weighted(country_FIs)

Fiscal Impulse Impact:
  FI leads GDP growth by 2-4 quarters
  FI leads corporate earnings by 3-6 quarters
  
  FI > +1% GDP: Strong fiscal stimulus
    Equities: +15% forward 12-month return (historical average)
    Bonds: -3% (rates rise on growth + deficit fears)
    USD: depends on relative FI vs other countries
  
  FI < -1% GDP: Fiscal drag
    Equities: +5% forward 12-month return (below average)
    Bonds: +5% (rates fall on growth weakness)

Cross-Country Relative FI:
  US_FI - Global_FI = relative fiscal impulse
  
  If US relative FI > 0: US outperforms (overweight US equities)
  If US relative FI < 0: Non-US outperforms (overweight international)
```

**Signal:**
- **Equity bullish:** FI > +1% GDP (strong fiscal stimulus ahead)
- **Bond bearish:** FI > +1% GDP (deficit spending -> rates rise)
- **Country rotation:** Overweight countries with highest relative FI
- **Equity bearish:** FI < -1% GDP (fiscal drag on growth)

**Risk:** Fiscal data is released with lags (1-2 quarters); use budget estimates for real-time
**Edge:** Fiscal impulse is the most underappreciated macro signal because most investors focus on monetary policy (central banks) while ignoring fiscal policy (government spending). Since 2008, fiscal policy has been the LARGER driver of GDP growth in most developed economies. The 2-4 quarter lead time of FI over GDP provides ample time for portfolio adjustment. COVID-era fiscal spending (FI > 10% GDP in many countries) demonstrated that fiscal impulse overwhelms monetary policy as a growth driver.

---

### 411 | ISM New Orders/Inventories Spread
**School:** US Manufacturing/Macro | **Class:** Cycle Timing
**Timeframe:** Monthly | **Assets:** US Equities, Cyclicals

**Mathematics:**
```
ISM Spread:
  ISM_Spread = ISM_New_Orders - ISM_Inventories
  
  New_Orders: demand for manufactured goods (forward-looking)
  Inventories: stock of unsold goods (backward-looking)
  
  Spread > 0: Orders > Inventories (demand exceeding supply = expansion)
  Spread < 0: Inventories > Orders (supply exceeding demand = contraction)

ISM Spread Cycle:
  The spread traces a cycle:
    Rising from trough: EARLY RECOVERY (strongest equity signal)
    At peak: MID-CYCLE EXPANSION
    Falling from peak: LATE CYCLE (caution)
    At trough: RECESSION/BOTTOMING

Lead Properties:
  ISM_Spread leads S&P 500 by 3-6 months
  ISM_Spread leads ISM headline by 2-3 months
  ISM_Spread leads GDP by 4-6 months

Historical Performance:
  Buy S&P when ISM_Spread > 0 AND rising:
    Forward 6M return: +9% average
    Hit rate: 72%
  
  Buy S&P when ISM_Spread crosses zero from below:
    Forward 12M return: +18% average
    Hit rate: 85% (one of the highest)
  
  Sell/reduce when ISM_Spread < 0 AND falling:
    Forward 6M return: +1% average (barely positive)
```

**Signal:**
- **Strong buy:** ISM_Spread crosses zero from below (early recovery)
- **Buy:** ISM_Spread > 0 AND rising (expansion)
- **Reduce:** ISM_Spread peaks and starts falling (late cycle)
- **Defensive:** ISM_Spread < 0 AND falling (contraction)

**Risk:** ISM data is monthly and can be revised; use alongside PMI for confirmation
**Edge:** The ISM New Orders minus Inventories spread is the single best timing signal for US equities because it captures the inventory cycle -- the most powerful short-term economic cycle. When orders exceed inventories, companies must produce more (boosting GDP, earnings, employment). When inventories exceed orders, production must be cut. The zero-cross from below has generated average 18% forward 12-month returns with 85% hit rate over 60+ years of data.

---

### 412 | Baltic Dry Index Shipping Signal
**School:** Commodity/Macro | **Class:** Global Trade
**Timeframe:** Weekly | **Assets:** Commodities, Equities, EM

**Mathematics:**
```
Baltic Dry Index (BDI):
  Composite of shipping rates for dry bulk cargo:
    Capesize (iron ore, coal): largest ships
    Panamax (grain, coal): medium ships
    Supramax (minor bulk): smaller ships
  
  BDI reflects REAL demand for physical shipping
  = pure supply/demand signal (no financial speculation)

BDI as Growth Indicator:
  BDI leads industrial production by 1-3 months
  BDI leads commodity prices by 1-2 months
  BDI leads EM equity markets by 1-2 months
  
  Because: you need to SHIP raw materials BEFORE you can manufacture goods
  Shipping demand = earliest indicator of manufacturing demand

BDI Signal:
  BDI_z = zscore(BDI, 252 days)
  BDI_mom = (BDI / BDI_60day) - 1
  
  BDI_z > +1 AND BDI_mom > 0: Strong global trade (risk-on)
  BDI_z < -1 AND BDI_mom < 0: Weak global trade (risk-off)

Trading Application:
  BDI strong (z > +1):
    Overweight: Commodities, EM equities, Shipping stocks
    Underweight: Bonds, USD
  
  BDI weak (z < -1):
    Overweight: Bonds, USD, Defensives
    Underweight: Commodities, EM, Cyclicals

Caveats:
  BDI is influenced by shipping fleet supply (new ship deliveries)
  A falling BDI could be demand weakness OR supply glut
  Cross-reference with: commodity prices AND trade data for confirmation
```

**Signal:**
- **Risk-on (global trade booming):** BDI > +1 std AND rising momentum
- **Risk-off (trade weakening):** BDI < -1 std AND falling
- **Commodity overweight:** BDI breakout (shipping demand surging)
- **Confirmation:** Cross-reference with commodity prices and PMI

**Risk:** BDI is volatile; can move 50%+ in a month; use as DIRECTIONAL signal, not timing
**Edge:** The Baltic Dry Index is the purest real-economy leading indicator because it measures physical demand for shipping that cannot be speculated on (you can't "buy" BDI futures for speculation -- it reflects actual shipping contracts). When BDI rises sharply, it means real physical demand for raw materials is increasing, which leads to higher commodity prices, stronger manufacturing, and ultimately higher corporate earnings. The 1-3 month lead over industrial production provides actionable timing for commodity and EM allocation.

---

### 413 | TIPS Breakeven Inflation Trading
**School:** Fixed Income/Macro | **Class:** Inflation Expectations
**Timeframe:** Monthly | **Assets:** Bonds, TIPS, Commodities

**Mathematics:**
```
TIPS Breakeven:
  BE = Nominal_Treasury_Yield - TIPS_Real_Yield
  
  BE represents the market's expectation of average inflation
  over the bond's life
  
  5Y BE: inflation expectations for next 5 years
  10Y BE: inflation expectations for next 10 years
  5Y5Y Forward BE: inflation expected from year 5 to year 10

Breakeven Signal:
  BE_z = zscore(10Y_BE, 252 days)
  
  BE_z > +1.5: Inflation fears ELEVATED
    Market pricing high inflation
    If you believe inflation will be LOWER than market expects:
      Long nominal Treasuries, Short TIPS
      = bet on lower-than-expected inflation
  
  BE_z < -1.5: Inflation fears DEPRESSED
    Market pricing low inflation
    If you believe inflation will be HIGHER:
      Long TIPS, Short nominal Treasuries
      = bet on higher-than-expected inflation

Mean Reversion of Breakevens:
  BE mean-reverts with half-life of ~6 months
  Because: extreme inflation expectations are usually wrong
  
  When BE > 3%: historically, realized inflation averaged 2.5% (overpriced)
  When BE < 1.5%: historically, realized inflation averaged 2.0% (underpriced)

Trading Application:
  Macro:
    Rising BE: Overweight commodities, TIPS, real assets
    Falling BE: Overweight nominal bonds, growth equities
  
  Relative value:
    5Y5Y forward BE < 10Y BE: inflation expectations front-loaded (fade)
    5Y5Y forward BE > 10Y BE: inflation expectations back-loaded (agree)
```

**Signal:**
- **Long TIPS (inflation trade):** BE < -1.5 std (inflation expectations too low)
- **Long nominals (deflation trade):** BE > +1.5 std (inflation expectations too high)
- **Commodity overweight:** Rising BE momentum (inflation trending higher)
- **Mean reversion:** Trade toward long-run average BE when at extremes

**Risk:** Inflation surprises can persist; use half-position at 1.5 std, full at 2.0 std
**Edge:** TIPS breakevens are the MARKET-PRICED inflation expectation that aggregates all available information from the largest, most liquid market in the world (US Treasuries). When breakevens reach extremes (>+1.5 std from mean), they have historically mean-reverted because inflation expectations overshoot in both directions. This is one of the most reliable mean-reversion signals in fixed income, with a ~70% hit rate and well-defined risk (the trade has natural bounds).

---

### 414 | Term Premium Decomposition Signal
**School:** Fed/Academic (Adrian, Crump, Moench) | **Class:** Rate Decomposition
**Timeframe:** Monthly | **Assets:** Treasuries, Equities

**Mathematics:**
```
10Y Yield Decomposition:
  10Y_Yield = Expected_Short_Rate_Path + Term_Premium
  
  Expected_Short_Rate = market's expectation of average Fed funds over 10 years
  Term_Premium = compensation for HOLDING 10Y vs rolling 3M
  
  ACM (Adrian, Crump, Moench) model estimates both components
  Published daily by NY Fed

Term Premium Signal:
  TP = ACM_Term_Premium(10Y)
  
  TP > +100bp: Investors demanding HIGH compensation for duration
    = risk-off for bonds (term premium may rise further)
    = usually coincides with Fed tightening or inflation fears
    
  TP near 0 or negative: Investors PAYING for duration (safe haven demand)
    = risk-on signal paradoxically (market complacent about rates)
    = bonds may be overpriced

TP Dynamics:
  Rising TP: Bonds selling off due to RISK PREMIUM (not rate expectations)
    = different from rate-expectation-driven selloff
    = typically associated with fiscal deficits, inflation uncertainty
  
  Falling TP: Bonds rallying due to SAFE HAVEN demand
    = flight to quality, risk-off
    = typically before or during equity weakness

Trading Application:
  TP regime determines bond allocation:
    TP < 0 AND falling: Long bonds (safe haven bid)
    TP > 50bp AND rising: Short bonds (term premium expanding)
    
  TP + Equity cross-signal:
    TP rising + Equities rising: LATE CYCLE (both can't continue)
    TP falling + Equities falling: CRISIS (flight to bonds)
    TP falling + Equities rising: GOLDILOCKS (best of both)
```

**Signal:**
- **Long bonds:** TP falling (safe haven demand = flight to quality)
- **Short bonds:** TP rising above 100bp (investors demanding more compensation)
- **Equity caution:** TP rising + equities rising (late cycle, unsustainable)
- **Goldilocks:** TP falling + equities rising (best environment for both)

**Risk:** ACM model estimates are model-dependent; use as DIRECTIONAL signal with other confirmation
**Edge:** The term premium decomposition separates INTEREST RATE EXPECTATIONS from RISK PREMIUM in bond yields. This distinction is critical because bonds sell off for two very different reasons: (1) expected rate hikes (short-lived, policy-driven) or (2) rising term premium (persistent, sentiment-driven). The ACM model allows you to identify which force is driving yield changes. Rising term premium is more persistent and predicts continued bond weakness, while rising rate expectations often reverse when the economy slows.

---

### 415 | Real Rate Differential FX Strategy
**School:** FX/Institutional | **Class:** Rate-Based FX
**Timeframe:** Monthly | **Assets:** G10 FX Pairs

**Mathematics:**
```
Real Rate Differential:
  RRD(A/B) = Real_Rate(A) - Real_Rate(B)
  
  Where Real_Rate = Nominal_Rate - Inflation_Expectations
  (Use TIPS yield or nominal - breakeven)

FX-Real Rate Relationship:
  Currencies tend to strengthen when their real rates rise
  (capital flows to highest real return)
  
  RRD(EURUSD) = Real_Rate(US) - Real_Rate(EUR)
  RRD > 0: USD has higher real rates -> USD should strengthen
  RRD < 0: EUR has higher real rates -> EUR should strengthen

RRD Signal Construction:
  For each G10 pair:
    RRD_z = zscore(RRD, 252 days)
    RRD_mom = change(RRD, 60 days)
  
  Signal = alpha * RRD_z + beta * RRD_mom
  (combine level and momentum of real rate differential)
  
  alpha = 0.5, beta = 0.5

Trading:
  For each G10 pair:
    Signal > +1: Long base currency (higher real rates)
    Signal < -1: Short base currency (lower real rates)
  
  Portfolio: trade 10 G10 crosses, volatility-weighted
  Expected Sharpe: ~0.6-0.8

Superior to Nominal Carry:
  Nominal carry: Long high nominal rate vs low nominal rate
  Real rate diff: Long high real rate vs low real rate
  
  Real rate strategy has: higher Sharpe (0.7 vs 0.4)
  Because: nominal carry is contaminated by inflation differential
  When inflation is high: nominal rates are high but currency WEAKENS
  Real rates adjust for this distortion
```

**Signal:**
- **Long currency:** RRD > +1 std AND rising (real rate advantage increasing)
- **Short currency:** RRD < -1 std AND falling (real rate disadvantage)
- **Pair selection:** Trade G10 crosses with widest RRD (most conviction)
- **Rebalance:** Monthly

**Risk:** Real rates change slowly; positions are medium-term; Risk 2% per pair
**Edge:** Real rate differentials dominate FX movements because they represent the true inflation-adjusted return on capital in each currency. Unlike nominal carry (which is distorted by inflation), real rate differentials correctly predict FX movements even during inflationary periods. When a country has rising real rates (tightening monetary policy faster than inflation), capital flows in and the currency strengthens. This is the most fundamental FX valuation framework used by institutional macro funds.

---

### 416 | Fed Funds Futures Implied Policy Path
**School:** Fed Watching/Institutional | **Class:** Policy Signal
**Timeframe:** Daily to Weekly | **Assets:** Rates, Equities, FX

**Mathematics:**
```
Fed Funds Futures:
  Each contract prices the AVERAGE Fed funds rate for a specific month
  
  Implied rate = 100 - futures_price
  
  Implied Probability of Hike/Cut:
    P(hike) = (Implied_Rate - Current_Rate) / 25bp
    (probability of a 25bp hike at the next meeting)

Policy Path Extraction:
  Chain of futures contracts gives the full policy path:
    Meeting 1: implied rate = X bp
    Meeting 2: implied rate = Y bp
    Meeting 6: implied rate = Z bp
  
  Path = sequence of implied rates at each meeting
  Number of hikes priced: (far_rate - current_rate) / 25bp

Surprise Component:
  Fed_Surprise = Actual_Rate_Change - Market_Expected_Change
  
  Positive surprise: Fed TIGHTER than expected (hawkish shock)
    Equities: -0.5% per 25bp surprise (immediate)
    USD: +0.5% per 25bp surprise
    Bonds: -0.5% per 25bp surprise
  
  Negative surprise: Fed EASIER than expected (dovish shock)
    Reverse of above

Trading Application:
  1. Pre-FOMC positioning:
     If market pricing > 90% probability of cut: position for the cut
     If market pricing < 10%: position for no cut (contrarian)
     
  2. Post-FOMC drift:
     After dovish surprise: equities tend to drift higher for 2-3 weeks
     After hawkish surprise: equities drift lower for 2-3 weeks
     
  3. Path vs dots:
     If implied path differs significantly from Fed dots:
       Trade toward convergence (market adjusts to dots over 1-3 months)
```

**Signal:**
- **Dovish positioning:** Market pricing > 80% probability of cut + equities weak
- **Post-FOMC drift:** Enter in direction of surprise for 2-3 week holding
- **Path convergence:** If market and dots diverge, trade toward dots
- **Risk-off:** If implied path shows rapid tightening (multiple hikes priced)

**Risk:** Fed can surprise dramatically; position before meetings is risky; Risk 1% pre-FOMC
**Edge:** Fed funds futures provide the MARKET'S expectation of monetary policy with precision (to the basis point). When market pricing diverges significantly from Fed communication (dots, speeches), the convergence trade has a high win rate because the market gradually adjusts. The post-FOMC drift effect (continued movement in the direction of the surprise for 2-3 weeks) is well-documented and provides a systematic trading opportunity after every Fed meeting.

---

### 417 | Cross-Market Volatility Transmission
**School:** Academic/Multi-Asset | **Class:** Vol Spillover
**Timeframe:** Weekly | **Assets:** Multi-Market

**Mathematics:**
```
Volatility Spillover Framework (Diebold & Yilmaz, 2012):
  Estimate VAR model of volatilities:
    [vol_equity, vol_bonds, vol_fx, vol_commodities]_t
    = A * [vol_equity, vol_bonds, vol_fx, vol_commodities]_{t-1} + epsilon_t
  
  Spillover Index:
    S = (sum of off-diagonal elements of forecast error variance decomposition)
        / (total forecast error variance) * 100
  
  S measures: what fraction of vol in each market comes from OTHER markets
  
  S low (~30%): Markets driven by own factors (diversification works)
  S high (~70%): Markets driven by spillovers (contagion, diversification fails)

Directional Spillovers:
  S(equity -> bonds): how much equity vol affects bond vol
  S(bonds -> equity): how much bond vol affects equity vol
  
  Net spillover: S(i -> j) - S(j -> i)
  
  If equity net spillover is high: equities are TRANSMITTING vol
    = equity market is the source of risk
    = reduce equity exposure
  
  If bonds net spillover is high: bonds are transmitting vol
    = rate market is the source of risk
    = reduce duration, hedge rates

Trading Application:
  Total Spillover Index > 60%:
    Markets highly interconnected
    Diversification ineffective
    REDUCE overall portfolio risk
  
  Total Spillover Index < 40%:
    Markets independent
    Diversification works well
    Can take full risk
  
  Directional: underweight the asset class that is NET TRANSMITTING vol
```

**Signal:**
- **Reduce risk:** Spillover index > 60% (markets interconnected, diversification fails)
- **Full risk:** Spillover index < 40% (markets independent)
- **Underweight:** Asset class with highest net vol transmission (source of risk)
- **Rebalance:** Weekly (spillovers can shift quickly during crises)

**Risk:** Spillover model requires estimation; rolling 200-day window; model risk exists
**Edge:** Cross-market volatility spillovers reveal when diversification will FAIL -- the periods when you need it most. When the spillover index is high, correlations are rising and a shock to any market will transmit to all others. By reducing overall risk during high-spillover regimes, you avoid the largest cross-asset drawdowns. The directional spillover further identifies WHICH market is causing the contagion, allowing targeted risk reduction rather than blanket deleveraging.

---

### 418 | Eurodollar/SOFR Spread Arbitrage
**School:** Rates/Institutional | **Class:** Funding Basis
**Timeframe:** Weekly | **Assets:** Short-Term Rates

**Mathematics:**
```
SOFR-Fed Funds Spread:
  SOFR is collateralized (Treasury repo rate)
  Fed Funds is unsecured (interbank lending)
  
  Normal: SOFR slightly below Fed Funds (collateral makes it safer)
  Stressed: SOFR can spike ABOVE Fed Funds (Treasury repo squeeze)

Quarter-End Dynamics:
  At quarter-end (Mar, Jun, Sep, Dec):
    SOFR spikes: banks reduce repo lending (balance sheet constraints)
    SOFR-FF spread can reach +20 to +50bp
    Predictable and tradeable
  
  Trade:
    2 weeks before quarter-end: Position for SOFR spike
    Receive SOFR in swap, Pay Fixed (lock in high SOFR)
    After quarter-end: SOFR normalizes, close position
    
    Expected profit: 5-20bp (annualized: ~2-4% with leverage)

Month-End Dynamics:
  Similar but smaller effect at month-end
  SOFR-FF spread widens 5-10bp
  
  Combined calendar trade:
    Quarter-ends: larger position (bigger move)
    Month-ends: smaller position (smaller move)
    ~12 trades per year with 70%+ win rate

SOFR Futures Basis:
  SOFR_futures_rate - SOFR_spot_rate
  
  When basis > +5bp: futures expensive (sell SOFR futures)
  When basis < -5bp: futures cheap (buy SOFR futures)
  Convergence at settlement is guaranteed
```

**Signal:**
- **Quarter-end trade:** 2 weeks before quarter-end, position for SOFR spike
- **Month-end trade:** Similar but smaller
- **Basis convergence:** Sell SOFR futures when basis > +5bp
- **Close:** After quarter/month-end normalization (1-3 days)

**Risk:** Balance sheet regulation changes can affect magnitude; low risk per trade; 0.5-1%
**Edge:** Quarter-end SOFR spikes are driven by bank balance sheet regulations (Basel III leverage ratios) that force banks to reduce repo lending at reporting dates. This regulatory constraint creates a STRUCTURAL and PREDICTABLE pattern that occurs every quarter. The spike magnitude (10-50bp) depends on overall market conditions but the direction (SOFR rises) is nearly guaranteed. This is one of the few trades in fixed income with true alpha because it exploits regulatory mechanics, not market views.

---

### 419 | Cross-Country Equity Earnings Yield Spread
**School:** Global Equity/Zurich | **Class:** Relative Valuation
**Timeframe:** Monthly | **Assets:** Country Equity Indices

**Mathematics:**
```
Earnings Yield Spread:
  For country pair (A, B):
    EY_spread = E/P(country_A) - E/P(country_B)
    
    Where E/P = earnings yield = inverse of P/E ratio
    
  Positive spread: Country A cheaper (higher earnings yield)
  Negative spread: Country B cheaper

Cross-Country Signal:
  For each pair of major countries (US, EU, UK, Japan, China, etc.):
    EY_spread_z = zscore(EY_spread, 10 years)
    
    EY_spread_z > +1.5: Country A extremely cheap vs B
      = Long A, Short B (valuation mean reversion)
    
    EY_spread_z < -1.5: Country B extremely cheap vs A
      = Long B, Short A

Enhanced with Real Rate Differential:
  Pure valuation mean reversion is SLOW (years)
  Add timing trigger:
    Enter valuation trade WHEN supported by rate differential change
    
    If EY_spread favors country A
    AND real_rate(A) is rising relative to B
    = STRONGEST signal (value + rate momentum aligned)
  
  This combination has: Sharpe ~0.7 (vs 0.3 for pure value)

Historical Performance:
  US vs Europe EY spread was +3% in 2008 (Europe cheap):
    Forward 3Y return: Europe outperformed US by 15%
  
  Japan EY spread was +5% vs US in 2012:
    Forward 3Y return: Japan outperformed by 40% (Abenomics)
```

**Signal:**
- **Long cheap country:** EY_spread > +1.5 std (extreme relative cheapness)
- **Timing trigger:** Enter when real rate differential confirms (momentum)
- **Pairs:** Trade major country pairs (US/EU, US/Japan, US/China, EU/UK)
- **Rebalance:** Monthly; hold for 6-24 months (valuation reversion is slow)

**Risk:** Country valuation can diverge for years; use rate differential for timing; Risk 3% per pair
**Edge:** Cross-country earnings yield spreads mean-revert over 3-5 year horizons because extreme relative valuations reflect temporary dislocations (currency moves, sentiment) rather than permanent differences. By combining valuation (which tells you WHAT to trade) with real rate differentials (which tell you WHEN to trade), you avoid the value trap problem of entering too early. The enhanced strategy generates Sharpe ~0.7 vs ~0.3 for pure valuation.

---

### 420 | Oil-Equity Correlation Regime
**School:** Cross-Asset/Energy | **Class:** Correlation Regime
**Timeframe:** Monthly | **Assets:** Oil, Equities

**Mathematics:**
```
Oil-Equity Correlation Regimes:
  The correlation between oil and equities CHANGES sign depending on the driver:
  
  Regime 1: DEMAND-DRIVEN oil moves
    Oil up because of strong demand -> equities also up
    Correlation: POSITIVE (+0.3 to +0.6)
    Indicators: PMI rising, industrial production strong
    
  Regime 2: SUPPLY-DRIVEN oil moves
    Oil up because of supply disruption -> equities down (cost shock)
    Correlation: NEGATIVE (-0.2 to -0.5)
    Indicators: geopolitical events, OPEC cuts, inventory draws
    
  Regime 3: FINANCIAL/RISK oil moves
    Oil and equities both driven by risk appetite
    Correlation: POSITIVE (+0.4 to +0.7)
    Indicators: high VIX, EM stress, liquidity conditions

Regime Detection:
  Features: [oil_return, equity_return, PMI, VIX, oil_inventory]
  Use logistic regression or HMM to classify regime
  
  Simplified:
    If PMI > 52 AND VIX < 20: DEMAND regime (positive correlation)
    If PMI < 48 AND Oil_inventory falling: SUPPLY regime (negative correlation)
    If VIX > 25: FINANCIAL regime (positive correlation)

Trading Implication:
  In DEMAND regime:
    Oil and equities move together -> no hedging benefit
    Long both for maximum exposure to growth
  
  In SUPPLY regime:
    Oil and equities are negatively correlated -> HEDGE
    Long oil + long equities = diversified (oil hedge against equity weakness)
    OR: use oil as hedge for equity portfolio
  
  In FINANCIAL regime:
    Both move together on risk appetite
    Reduce overall exposure (no diversification benefit)
```

**Signal:**
- **Demand regime:** Long both oil and equities (aligned, maximise growth exposure)
- **Supply regime:** Use oil as equity hedge (negative correlation)
- **Financial regime:** Reduce both positions (correlated, no diversification)
- **Regime identification:** Monthly based on PMI, VIX, inventory data

**Risk:** Regime classification error; confirm with multiple indicators; Risk 2% per asset
**Edge:** Most investors assume oil and equities are always positively correlated. In reality, the correlation changes sign depending on whether oil is moving for demand, supply, or financial reasons. In supply shock regimes, oil becomes a NATURAL HEDGE for equity portfolios (they move in opposite directions). By detecting the oil-equity correlation regime, you can use oil as a hedge during supply shocks and as an additional growth bet during demand-driven moves, dramatically improving portfolio efficiency.

---

### 421 | Japanese Yen Risk-Off Signal
**School:** FX/Tokyo-London | **Class:** Safe Haven FX
**Timeframe:** Daily to Weekly | **Assets:** JPY, Global Equities

**Mathematics:**
```
JPY as Risk Barometer:
  JPY strengthens during global risk-off events because:
    1. Carry trade unwind: JPY-funded carry trades close (buy back JPY)
    2. Repatriation: Japanese investors bring foreign capital home
    3. Current account surplus: structural JPY demand
    4. Negative correlation with equities: -0.6 during crises

JPY Risk Signal:
  USDJPY_z = zscore(USDJPY, 60 days)
  (note: lower USDJPY = stronger JPY = more risk-off)
  
  USDJPY_change_5d = 5-day change in USDJPY
  
  JPY Strengthening > 2% in 5 days:
    = Carry trade unwinding
    = RISK-OFF signal for equities
    = Historically: equities fall 3-5% in following 2 weeks

Cross-Asset JPY Signal:
  JPY_signal = -(USDJPY_5d_change + EURJPY_5d_change + AUDJPY_5d_change) / 3
  (average JPY strengthening against multiple currencies)
  
  JPY_signal > +1.5%: STRONG risk-off (JPY strengthening broadly)
    = Reduce equity exposure, increase cash and bonds
  
  JPY_signal < -1.5%: Risk-on (JPY weakening = carry trade flowing)
    = Increase equity/EM exposure

JPY Carry Unwind Cascade:
  Stage 1: JPY strengthens 1-2% (initial carry unwind)
  Stage 2: Margin calls force more unwinds (JPY strengthens 2-4%)
  Stage 3: Panic liquidation (JPY strengthens 5%+)
  Stage 4: BOJ intervention or stabilization
  
  At Stage 1: WARNING (reduce carry and EM positions)
  At Stage 2: ACTION (go defensive)
  At Stage 3: CONTRARIAN (prepare to buy dips)
```

**Signal:**
- **Risk-off warning:** JPY strengthening > 1.5% in 5 days (carry unwind beginning)
- **Defensive:** JPY strengthening > 3% (carry cascade in progress)
- **Contrarian:** After JPY strengthens > 5% (overextended, prepare for reversal)
- **Risk-on:** JPY weakening > 1.5% (carry trade flowing, risk appetite returning)

**Risk:** JPY moves can be violent (BOJ intervention); position sizes must accommodate 5%+ moves
**Edge:** The Japanese yen is the most reliable real-time risk barometer because it is the funding currency for the largest carry trade in the world. When global risk appetite declines, the carry trade unwinds by BUYING JPY, creating a predictable and measurable signal. JPY strengthening leads equity market weakness by 1-3 days because carry positions are leveraged and liquidated faster than equity positions. This lead time provides the window for defensive portfolio adjustment.

---

### 422 | Gold/Silver Ratio Macro Signal
**School:** Precious Metals/Macro | **Class:** Metals Ratio
**Timeframe:** Monthly | **Assets:** Precious Metals, Risk Assets

**Mathematics:**
```
Gold/Silver Ratio (GSR):
  GSR = Gold_Price / Silver_Price
  
  Gold: monetary/safe-haven metal (demand driven by fear/uncertainty)
  Silver: industrial metal (50% industrial use) + monetary metal
  
  GSR interpretation:
    GSR high (>80): Extreme risk aversion (gold outperforming silver)
      Silver's industrial component weak (growth concerns)
      Historical extremes: 2020 (127), 2008 (84), 1991 (100)
    
    GSR low (<60): Risk appetite strong (silver outperforming gold)
      Silver's industrial demand strong (growth environment)
      Historical lows: 2011 (32), 2006 (45)

GSR Mean Reversion:
  Long-run average: ~65 (1990-2024)
  
  GSR > 85: EXTREME fear (contrarian bullish for risk assets)
    Buy silver, sell gold (ratio should compress)
    Also: increase equity allocation (fear is overdone)
    Forward 12M: GSR compresses to ~70 (average)
    Silver outperforms gold by ~25% (average)
  
  GSR < 50: EXTREME optimism (contrarian bearish)
    Buy gold, sell silver (ratio should expand)
    Reduce equity allocation (complacency risk)

GSR Momentum:
  Rising GSR (fear increasing):
    Silver weakening vs gold = industrial demand slowing
    = early warning for equity weakness (1-3 month lead)
  
  Falling GSR (risk appetite increasing):
    Silver strengthening = industrial demand improving
    = confirmation of equity strength
```

**Signal:**
- **Long silver/short gold:** GSR > 85 (extreme fear, ratio compresses)
- **Long gold/short silver:** GSR < 50 (extreme optimism, ratio expands)
- **Equity overlay:** Rising GSR = reduce equities; Falling GSR = add equities
- **Timing:** GSR momentum change = leading indicator for equity turning points

**Risk:** GSR can remain extreme during prolonged crises; use with stop at further extreme; Risk 2%
**Edge:** The gold/silver ratio captures the industrial vs. safe-haven balance in precious metals markets. Since silver has 50% industrial demand, its relative performance vs gold directly measures industrial growth expectations. GSR extremes (>85 or <50) have been reliable contrarian indicators for risk assets because they represent extreme positioning in precious metals that typically mean-reverts. The ratio's 1-3 month lead over equity turning points provides a genuine information advantage.

---

### 423 | Credit Spread Momentum Cross-Asset
**School:** Credit/Multi-Asset | **Class:** Credit Signal
**Timeframe:** Monthly | **Assets:** Credit, Equities, FX

**Mathematics:**
```
Credit Spread Hierarchy:
  Investment Grade (IG): lowest spread, most liquid
  High Yield (HY): medium spread, medium liquidity
  Leveraged Loans: highest spread, least liquid
  
  Spreads widen: IG first (most liquid), then HY, then Loans
  Spreads tighten: reverse order (Loans tighten last)

Credit Momentum Signal:
  For each credit segment:
    Spread_mom = change(spread, 30 days) / sigma_spread
  
  Credit_Signal = -0.4*IG_mom - 0.35*HY_mom - 0.25*Loan_mom
  (negative because widening = bearish)

Credit-Equity Lead-Lag:
  Credit spreads LEAD equity returns by 1-3 weeks
  Because: credit markets are more sensitive to default risk
  and institutional credit investors are better informed than retail equity
  
  HY spread widening > 50bp in 1 month:
    Forward equity return (next month): -3% average
  
  HY spread tightening > 50bp in 1 month:
    Forward equity return (next month): +2% average

Cross-Asset Application:
  Credit_Signal > +1: Credit improving (tightening)
    = Overweight equities, EM, high-beta
    = Underweight: treasuries, USD
  
  Credit_Signal < -1: Credit deteriorating (widening)
    = Underweight equities, increase bonds/cash
    = Overweight: treasuries, USD, gold

Credit Curve Signal:
  HY-IG spread (credit curve):
    Steepening (HY spreads widening faster): RISK-OFF (low quality weakening)
    Flattening: RISK-ON (low quality improving)
```

**Signal:**
- **Risk-on:** Credit momentum positive (spreads tightening across all segments)
- **Risk-off:** Credit momentum negative (spreads widening)
- **Early warning:** IG widening before HY (first mover in credit)
- **Strongest signal:** HY and Loans both widening >50bp/month (crisis signal)

**Risk:** Credit markets can be illiquid during stress; use CDS indices for real-time signal
**Edge:** Credit spreads are the most reliable leading indicator for equity markets because credit investors are predominantly institutional (better informed), credit has contractual cash flows (more analyzable), and default risk manifests in credit markets BEFORE equity markets. The 1-3 week lead of credit over equities has been consistent across every major market dislocation. By using credit momentum as an equity allocation signal, you systematically reduce equity exposure before drawdowns.

---

### 424 | VIX-Equity Correlation Regime Switch
**School:** Cross-Asset/Quantitative | **Class:** Correlation Regime
**Timeframe:** Weekly | **Assets:** Equities, VIX

**Mathematics:**
```
VIX-Equity Relationship:
  Normal: VIX and SPX negatively correlated (~-0.75)
    SPX up -> VIX down (less fear)
    SPX down -> VIX up (more fear)
  
  Abnormal (rare but important):
    SPX up AND VIX up: "Wall of worry" (market rising despite fear)
      = STRONG bullish signal (skepticism means room to rally)
    
    SPX down AND VIX down: "Complacent decline" (market falling but no fear)
      = DANGEROUS (market declining without proper hedging)

Regime Detection:
  21-day rolling correlation: corr(SPX_return, VIX_change)
  
  Normal: corr < -0.5 (standard inverse relationship)
  Divergence: corr > -0.3 (relationship breaking down)
  Inversion: corr > 0 (both moving same direction)

VIX-Equity Divergence Signal:
  SPX_up AND VIX_up (positive correlation during rally):
    Frequency: ~15% of months
    Forward 3M equity return: +6% average (vs +3% unconditional)
    = Skeptical rally, strong continuation expected
    
  SPX_down AND VIX_down (positive correlation during selloff):
    Frequency: ~5% of months
    Forward 3M equity return: -4% average
    = Complacent decline, WARNING of larger selloff ahead
    
  SPX_down AND VIX_spike (normal relationship, extreme magnitude):
    VIX single-day spike > 20%: EXTREME fear
    Forward 3M return: +8% average (contrarian buy)
```

**Signal:**
- **Strong buy:** SPX rising AND VIX rising (wall of worry = bullish)
- **Strong warning:** SPX falling AND VIX falling (complacent decline = dangerous)
- **Contrarian buy:** VIX spike > 20% in one day (extreme fear = opportunity)
- **Normal:** Negative correlation maintained (no special signal)

**Risk:** VIX-equity divergences are infrequent; when they occur, signal is strong; Risk 2%
**Edge:** The VIX-equity correlation contains information beyond either VIX or equity prices alone. When both rise simultaneously, it indicates a market climbing a "wall of worry" -- the strongest type of rally because skepticism means there are still buyers on the sideline. When both fall simultaneously, it indicates a market declining without proper hedging -- the most dangerous type of decline because the crash protection hasn't been purchased. These divergences are rare but highly predictive.

---

### 425 | Intermarket Analysis Quad-Screen
**School:** Cross-Asset/Technical (John Murphy) | **Class:** Intermarket
**Timeframe:** Weekly | **Assets:** Equities, Bonds, Commodities, USD

**Mathematics:**
```
Murphy's Intermarket Relationships:
  The four asset classes are linked in a cycle:
  
  INFLATIONARY CYCLE:
    1. Dollar weakens (first mover)
    2. Commodities rise (dollar inverse)
    3. Interest rates rise (inflation -> bond sells)
    4. Equities fall eventually (rate pressure)
  
  DEFLATIONARY CYCLE:
    1. Dollar strengthens
    2. Commodities fall
    3. Interest rates fall (bonds rally)
    4. Equities rise initially (lower rates support PE)

Quad-Screen Signal:
  For each asset class, compute 60-day momentum:
    USD_mom, Commodity_mom, Bond_mom, Equity_mom
  
  Consistent intermarket confirmation:
    All four aligned with cycle: HIGH CONVICTION signal
    Example (inflation): USD_mom < 0, Commodity_mom > 0,
      Bond_mom < 0 (yields rising), Equity_mom varies
    
  Intermarket divergence:
    One or more asset classes diverge from cycle: WARNING
    Example: commodities rising but USD also rising
      = supply-driven commodity move (not demand)
      = less bullish for equities

Scoring:
  For inflationary cycle:
    Score = (-USD_mom_z + Commodity_mom_z - Bond_mom_z) / 3
    Score > +1: Strong inflationary confirmation
    Score < -1: Strong deflationary confirmation
  
  For equity overlay:
    Equity_signal = function(cycle_position):
      Early inflation: equity neutral to positive
      Late inflation: equity negative (rates biting)
      Early deflation: equity negative (slowdown)
      Late deflation: equity positive (rates falling)
```

**Signal:**
- **Inflationary confirmation:** USD down + Commodities up + Bonds down (yields up)
- **Deflationary confirmation:** USD up + Commodities down + Bonds up
- **Equity allocation:** Based on cycle position (early deflation = buy, late inflation = sell)
- **Divergence alert:** When any asset class breaks the cycle relationship

**Risk:** Intermarket relationships can shift (e.g., stocks-bonds correlation regime); Risk 2%
**Edge:** Murphy's intermarket analysis provides a framework for understanding HOW asset classes interact that goes beyond simple correlations. The cycle relationship (USD -> Commodities -> Bonds -> Equities) reflects real economic causation: dollar weakness makes commodities cheaper, rising commodities cause inflation, inflation causes rate hikes, and rate hikes eventually hurt equities. By monitoring where we are in this cycle, you can anticipate the NEXT asset class to move rather than reacting after it moves.

---

"""

with open("Indicators.md", "a") as f:
    f.write(content)

print("Appended strategies 401-425 to Indicators.md")
