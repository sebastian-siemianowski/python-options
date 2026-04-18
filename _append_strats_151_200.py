#!/usr/bin/env python3
"""Append strategies 151-200 to Indicators.md"""

content = r"""
### 151 | On-Balance Volume Divergence
**School:** New York (Joe Granville, 1963) | **Class:** Volume Confirmation
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
OBV_t = OBV_{t-1} + sign(Close_t - Close_{t-1}) * Volume_t

  If Close_t > Close_{t-1}: OBV += Volume (accumulation)
  If Close_t < Close_{t-1}: OBV -= Volume (distribution)
  If Close_t == Close_{t-1}: OBV unchanged

OBV Slope:
  OBV_slope = linreg_slope(OBV, 20)  (20-day trend in OBV)
  Price_slope = linreg_slope(Close, 20)

Divergence Detection:
  Bullish_Divergence = Price_slope < 0 AND OBV_slope > 0
    (price declining but volume accumulating = smart money buying)
  Bearish_Divergence = Price_slope > 0 AND OBV_slope < 0
    (price rising but volume distributing = smart money selling)

Divergence Strength:
  Div_Strength = |OBV_slope - Price_slope| / stdev(Price_slope, 120)
  Strong divergence: Div_Strength > 2.0
```

**Signal:**
- **Long:** Bullish divergence with Div_Strength > 2.0 AND price at support
- **Short:** Bearish divergence with Div_Strength > 2.0 AND price at resistance
- **Exit:** Divergence resolves (slopes align) OR price stops moving

**Risk:** Stop below support (long) or above resistance (short); Target 2x ATR; Risk 1%
**Edge:** OBV divergence detects institutional accumulation/distribution before price reflects it. Large funds accumulate over days/weeks, and their volume footprint shows in OBV even when price is flat or declining. Granville's insight -- volume precedes price -- remains one of the most reliable leading indicators in technical analysis.

---

### 152 | Volume-Weighted Average Price (VWAP) Institutional Flow
**School:** New York (Institutional Execution) | **Class:** Benchmark Trading
**Timeframe:** Intraday | **Assets:** US Equities, Futures

**Mathematics:**
```
VWAP = cumsum(Price_i * Volume_i) / cumsum(Volume_i)
  (running average price weighted by volume from session start)

VWAP Bands:
  Upper_1 = VWAP + 1 * StdDev(VWAP_deviation)
  Upper_2 = VWAP + 2 * StdDev(VWAP_deviation)
  Lower_1 = VWAP - 1 * StdDev(VWAP_deviation)
  Lower_2 = VWAP - 2 * StdDev(VWAP_deviation)

VWAP Deviation:
  Dev = (Price - VWAP) / VWAP * 100  (percent deviation)

Institutional Flow Detection:
  If Price consistently above VWAP with volume: institutional buying
  If Price consistently below VWAP with volume: institutional selling

Multi-Day Anchored VWAP:
  Anchor from: earnings date, IPO date, 52-week high/low, gap day
  AVWAP from anchor = cumulative VWAP from anchor date
```

**Signal:**
- **Long:** Price crosses above VWAP from below AND volume increasing (institutional demand)
- **Short:** Price crosses below VWAP from above AND volume increasing (institutional supply)
- **Mean reversion:** Price at 2-sigma VWAP band -> fade back toward VWAP
- **Anchored VWAP support:** Price touches anchored VWAP from significant date = major level

**Risk:** Stop at opposite VWAP band; Intraday risk only; Close before EOD
**Edge:** VWAP is THE benchmark for institutional execution. When price is above VWAP, institutional buyers are competing (pushing price up to fill orders). When below, sellers are aggressive. Understanding VWAP dynamics lets you align with or trade against institutional flow. Anchored VWAP from significant dates creates powerful support/resistance levels because institutional cost bases cluster there.

---

### 153 | Cumulative Volume Delta (CVD) Strategy
**School:** Chicago (Order Flow Analysis) | **Class:** Market Microstructure
**Timeframe:** Intraday (1-min to 15-min) | **Assets:** Futures (ES, NQ, CL)

**Mathematics:**
```
Volume Delta:
  For each trade:
    If trade at ask: Buy_Volume (aggressive buyer)
    If trade at bid: Sell_Volume (aggressive seller)
  
  Delta = Buy_Volume - Sell_Volume (per bar)
  CVD = cumsum(Delta)  (cumulative from session start)

CVD Divergence:
  Price making new high BUT CVD declining = bearish divergence
    (price rising on aggressive selling = exhaustion)
  Price making new low BUT CVD rising = bullish divergence
    (price falling on aggressive buying = absorption)

CVD Slope:
  CVD_slope = linreg_slope(CVD, 20_bars)
  If CVD_slope > 0: net aggressive buying (bullish flow)
  If CVD_slope < 0: net aggressive selling (bearish flow)

Delta Spike:
  Delta_Z = Delta / StdDev(Delta, 50_bars)
  |Delta_Z| > 3.0: extreme order flow event (absorption or capitulation)
```

**Signal:**
- **Long:** CVD rising AND price at support AND no bearish divergence
- **Short:** CVD declining AND price at resistance AND no bullish divergence
- **Divergence reversal:** CVD divergence confirmed with Delta_Z spike > 3 (capitulation)
- **Exit:** CVD slope flattens or reverses

**Risk:** Tight stops (intraday); Stop at prior bar extreme; Risk 0.3% per trade
**Edge:** CVD separates passive from aggressive order flow. Aggressive buyers (hitting the ask) drive price up; aggressive sellers (hitting the bid) drive price down. When CVD diverges from price, it reveals that the aggressive side is LOSING -- price moves despite opposing aggressive flow, which indicates exhaustion. This is the most direct measure of institutional vs. retail flow in futures markets.

---

### 154 | Volume Profile Point of Control (POC) Trading
**School:** Chicago (Market Profile, Peter Steidlmayer) | **Class:** Auction Theory
**Timeframe:** Daily / Intraday | **Assets:** Futures, Equities

**Mathematics:**
```
Volume Profile:
  Aggregate volume at each price level over a period (day, week, or custom)
  
  POC (Point of Control) = price level with highest volume
  Value Area = price range containing 70% of total volume
  VA_High = upper bound of value area
  VA_Low = lower bound of value area

Trading Rules:
  1. Price gravitates toward POC (volume-weighted equilibrium)
  2. VA_High and VA_Low act as support/resistance
  3. Acceptance (trading within VA) vs. Rejection (spike outside VA)

Prior Day Profile:
  If today opens above prior VA_High: bullish acceptance of higher prices
  If today opens below prior VA_Low: bearish acceptance of lower prices
  If today opens inside prior VA: range day expected, trade POC mean-reversion

Volume at Price Distribution:
  P-shape: early selling, then buying (bullish day)
  b-shape: early buying, then selling (bearish day)
  D-shape: balanced day (narrow range, high volume at center)
  B-shape: double distribution (two POCs, directional day)
```

**Signal:**
- **Long:** Open above prior VA_High AND volume confirms (value area migration upward)
- **Short:** Open below prior VA_Low AND volume confirms
- **Mean reversion:** Open inside prior VA -> fade to POC
- **Breakout:** Price breaks out of current VA with volume surge = directional move

**Risk:** Stop outside value area (above VA_High for shorts, below VA_Low for longs); Risk 1%
**Edge:** Volume Profile reveals WHERE the most trading occurred, identifying the fair value price (POC) and acceptance zones (VA). Unlike time-based charts that weight all bars equally, Volume Profile weights by actual participation. Price outside the value area is "unfair" and tends to return. Price migrating to a new value area is a genuine trend. This auction-theory framework is the foundation of institutional futures trading.

---

### 155 | Market Profile TPO Distribution
**School:** Chicago (CBOT, Peter Steidlmayer, 1984) | **Class:** Time-Price Opportunity
**Timeframe:** 30-min periods within daily | **Assets:** Futures

**Mathematics:**
```
TPO (Time Price Opportunity):
  Divide session into 30-minute brackets (A, B, C, ... N)
  At each price level, record which brackets traded there
  TPO_count(price) = number of 30-min brackets that traded at that price

  POC = price with most TPO counts (highest time exposure)
  Initial Balance (IB) = range of first 2 brackets (A + B period, 9:30-10:30 ET)
  IB_Range = IB_High - IB_Low

Day Type Classification:
  Normal Day: Range < 1.2 * IB_Range (contained, balanced)
  Normal Variation: Range = 1.2-2.0 * IB_Range (moderate extension)
  Trend Day: Range > 2.0 * IB_Range (strong directional move)
  Neutral Day: Wide IB, no extension (range-bound)

IB Extension Probability:
  P(extend_up) = historical frequency of IB_High break AND close above IB_High
  P(extend_down) = historical frequency of IB_Low break AND close below IB_Low
  Historical: ~60% of IB breaks lead to continuation in that direction
```

**Signal:**
- **Trend day entry:** IB break with volume AND single-print TPOs (gaps in profile) = strong trend
- **Range day fade:** Normal day classification -> fade IB extremes to POC
- **Opening type:** If open outside prior value area -> potential trend day opportunity

**Risk:** Stop at opposite IB extreme; Trend day: trail with developing POC; Range day: target POC
**Edge:** Market Profile classifies the day TYPE early in the session (using IB range), allowing you to select the appropriate strategy (trend-follow or fade) before committing capital. The 60% hit rate on IB extensions is one of the most reliable intraday patterns because it reflects the initial-balance-setting by institutional traders. Single-print TPOs (price levels visited only once) mark genuine directional intent.

---

### 156 | Wyckoff Volume Spread Analysis (VSA)
**School:** New York (Richard Wyckoff, 1930s) | **Class:** Supply/Demand Analysis
**Timeframe:** Daily | **Assets:** Equities, Indices

**Mathematics:**
```
Wyckoff VSA analyzes the relationship between:
  1. Spread = High - Low (bar range)
  2. Volume = total volume of the bar
  3. Close position = (Close - Low) / (High - Low)  (IBS)

Key VSA Patterns:

Stopping Volume (Selling Climax):
  Criteria: spread > 2*ATR AND volume > 2.5*avg AND IBS > 0.5
  Meaning: Massive selling absorbed by professional demand (market bottom)

No Demand:
  Criteria: spread < 0.5*ATR AND volume < 0.5*avg AND IBS > 0.5 AND upbar
  Meaning: Weak rise on low volume, no professional interest in higher prices

Upthrust:
  Criteria: New high penetration AND IBS < 0.3 AND volume > 1.5*avg
  Meaning: Price pushed above resistance to trap longs, then reversed

Spring:
  Criteria: New low penetration AND IBS > 0.7 AND volume > 1.5*avg
  Meaning: Price pushed below support to trap shorts, then reversed (bullish)
```

**Signal:**
- **Long:** Spring pattern OR Stopping Volume (with confirmation next bar up)
- **Short:** Upthrust pattern OR No Demand after extended rally
- **Confirmation:** Next bar must confirm (close in direction of signal with normal+ volume)
- **Exit:** Opposite VSA signal appears

**Risk:** Stop below spring low (long) or above upthrust high (short); Target 3x ATR; Risk 1%
**Edge:** Wyckoff's VSA reads the footprint of professional (institutional) money through the three-way relationship of spread, volume, and close position. No other method so directly reveals supply/demand dynamics. A spring (shakeout below support on volume, closing near high) is the professional entry pattern -- they push price below support to trigger retail stops, absorb the selling, then drive price higher. Understanding this manipulation is the core edge.

---

### 157 | Time-Weighted Average Price (TWAP) Deviation
**School:** Institutional (Algorithmic Execution) | **Class:** Execution Benchmark
**Timeframe:** Intraday | **Assets:** All liquid markets

**Mathematics:**
```
TWAP = sum(Price_i) / N  (simple average of all prices, equal time weight)
  (vs. VWAP which is volume-weighted)

TWAP Deviation:
  TWAP_Dev = (Price - TWAP) / TWAP * 100

TWAP vs VWAP Spread:
  Spread = VWAP - TWAP
  If Spread > 0: Volume concentrated at higher prices (buying pressure)
  If Spread < 0: Volume concentrated at lower prices (selling pressure)

Institutional Algo Detection:
  TWAP_algo_signature = Price clustering at regular time intervals
    (TWAP execution algorithms buy/sell at fixed intervals regardless of price)
  
  Detect: volume spikes at regular intervals (every 30s, 1min, 5min)
  If TWAP_algo detected AND direction is buy: price supported at TWAP
    (algo will absorb dips to TWAP level)
```

**Signal:**
- **VWAP-TWAP spread positive + increasing:** Aggressive buying above average prices -- bullish
- **VWAP-TWAP spread negative + increasing:** Aggressive selling below average prices -- bearish
- **TWAP support:** If TWAP algo detected buying, price dips to TWAP are buy opportunities
- **Exit:** Spread reverses or flattens

**Risk:** Intraday only; Tight stops at session extremes; Risk 0.3%
**Edge:** The VWAP-TWAP spread reveals whether volume is concentrated at higher or lower prices -- a direct measure of buyer/seller aggression that simple volume analysis misses. TWAP algorithm detection identifies when institutional orders are being executed mechanically, creating predictable price support/resistance levels for the session. Retail traders can exploit this by buying dips to the TWAP floor when a large TWAP buy program is active.

---

### 158 | Volume Climax Exhaustion
**School:** London (Volume Traders) | **Class:** Capitulation Detection
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Volume Climax:
  Vol_Z = (Volume - SMA(Volume, 50)) / StdDev(Volume, 50)
  Climax = Vol_Z > 3.0 (volume 3+ sigma above average)

Directional Climax:
  Climax_Up = Climax AND Close > Open AND Close in top 25% of range
    (massive buying on extreme volume = potential buying exhaustion at top)
  Climax_Down = Climax AND Close < Open AND Close in bottom 25% of range
    (massive selling on extreme volume = potential selling exhaustion at bottom)

Reversal Confirmation:
  After Climax_Down:
    Next 3 bars: Look for higher close on declining volume
    = selling exhausted, buyers stepping in on lower volume (healthy recovery)
  After Climax_Up:
    Next 3 bars: Look for lower close on declining volume
    = buying exhausted, sellers stepping in

Climax Follow-Through:
  FT = return(5_days_after_climax)
  Historical: After Climax_Down: FT = +1.8% avg (reversal up)
  After Climax_Up: FT = -0.3% avg (less reliable)
```

**Signal:**
- **Buy after Climax_Down:** Wait for 1-3 bars of confirmation (higher close, lower volume)
- **Sell after Climax_Up:** Wait for 1-3 bars of confirmation (lower close, lower volume)
- **Volume spike without climax (Vol_Z 2-3):** Warning but not actionable; need 3+ sigma for significance

**Risk:** Stop below climax bar low (long); Target 2x ATR or prior resistance; Risk 1.5%
**Edge:** Volume climaxes mark capitulation -- the point where the last marginal seller (or buyer) has acted. At extremes, there is literally no one left to sell, so price must reverse. The 3-sigma threshold ensures you only trade genuine capitulation events (not just high-volume days). Climax_Down reversals are more reliable than Climax_Up because fear creates sharper, more definitive capitulation than greed.

---

### 159 | Accumulation/Distribution Line (AD) Institutional Flow
**School:** New York (Marc Chaikin) | **Class:** Money Flow
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
Accumulation/Distribution:
  CLV = ((Close - Low) - (High - Close)) / (High - Low)
    (Close Location Value: where close is within the bar, range [-1, +1])
  AD_t = AD_{t-1} + CLV * Volume

  CLV > 0: Close above midpoint (accumulation)
  CLV < 0: Close below midpoint (distribution)

AD Divergence:
  Price new high AND AD NOT new high = distribution divergence (bearish)
  Price new low AND AD NOT new low = accumulation divergence (bullish)

AD Rate of Change:
  AD_ROC = (AD - AD_{n_periods_ago}) / |AD_{n_periods_ago}| * 100
  AD_ROC_Z = normalize(AD_ROC, 60)

Chaikin Oscillator (acceleration):
  CO = EMA(AD, 3) - EMA(AD, 10)  (MACD of AD line)
  CO_Z = normalize(CO, 60)
```

**Signal:**
- **Long:** Bullish AD divergence AND CO turns positive (accumulation divergence with acceleration)
- **Short:** Bearish AD divergence AND CO turns negative
- **Confirmation:** AD_ROC_Z > +1.5 with price breakout (volume-confirmed breakout)
- **Exit:** CO crosses zero in opposite direction

**Risk:** Stop at divergence price extreme; Target at prior resistance/support; Risk 1%
**Edge:** The AD line combines price position within the bar AND volume to measure whether money is flowing INTO (accumulation) or OUT OF (distribution) an asset. Unlike OBV which only uses close direction, AD uses the close POSITION within the bar, capturing the intra-bar battle between buyers and sellers. The Chaikin Oscillator adds timing by measuring the acceleration of money flow.

---

### 160 | Limit Order Book (LOB) Imbalance Strategy
**School:** London/New York (HFT/Market Making) | **Class:** Microstructure
**Timeframe:** Tick/Second | **Assets:** Large-cap equities, futures

**Mathematics:**
```
LOB Imbalance:
  Bid_Volume = sum(volume at top N bid levels)
  Ask_Volume = sum(volume at top N ask levels)
  
  Imbalance = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
  Range: [-1, +1]
  
  Imbalance > +0.3: More bid depth (buying pressure, bullish)
  Imbalance < -0.3: More ask depth (selling pressure, bearish)

Imbalance Predictive Power:
  E[ret_{t+1}] = alpha + beta * Imbalance_t + epsilon
  Historical: beta > 0 (positive imbalance predicts positive returns, ~1-5 sec horizon)
  Decays: predictive power halves every ~2 seconds

Depth-Weighted Imbalance:
  DWI = sum(V_bid_i / (mid - P_bid_i)) - sum(V_ask_i / (P_ask_i - mid))
  (volume weighted by distance from midpoint: closer levels count more)

Cancel-to-Trade Ratio:
  CTR = cancellations / executions
  CTR > 10: spoofing-like activity (fake liquidity, ignore those levels)
  CTR < 2: genuine liquidity
```

**Signal:**
- **Long (1-10 sec horizon):** DWI > +0.5 AND CTR < 5 (genuine bid-heavy imbalance)
- **Short:** DWI < -0.5 AND CTR < 5 (genuine ask-heavy imbalance)
- **No trade:** CTR > 10 at any level (potential spoofing, imbalance unreliable)
- **Exit:** DWI crosses zero or after 10 seconds (alpha decays rapidly)

**Risk:** Ultra-tight stops (1-2 ticks); Position sized for tick-level moves; Latency < 100ms required
**Edge:** LOB imbalance is the most granular predictive signal in market microstructure. The excess bid/ask volume at the top of book reflects the immediate supply/demand balance and predicts short-term price direction. However, the signal is extremely short-lived (~5 seconds) and requires low-latency infrastructure. The CTR filter removes spoofed levels that would otherwise corrupt the signal.

---

### 161 | Volume-Synchronized Probability of Informed Trading (VPIN)
**School:** Academic (Easley, Lopez de Prado, O'Hara, 2012) | **Class:** Information Asymmetry
**Timeframe:** Volume-based (volume bars) | **Assets:** Futures, large-cap equities

**Mathematics:**
```
VPIN = Volume-Synchronized Probability of Informed Trading

Step 1: Create volume bars (bars of equal volume V_bar)
Step 2: For each bar, classify buy/sell volume:
  V_buy_i = V_bar * CDF_normal((Close - Open) / sigma)
  V_sell_i = V_bar - V_buy_i

Step 3: Compute order imbalance in buckets of n bars:
  OI = |sum(V_buy) - sum(V_sell)| / (n * V_bar)

Step 4: VPIN = EMA(OI, window)
  Range: [0, 1]
  VPIN > 0.5: High probability of informed trading (information asymmetry)
  VPIN < 0.2: Low informed trading (normal market-making)

Flash Crash Prediction:
  VPIN exceeded 0.80 two hours before the May 2010 Flash Crash
  (informed traders accumulated short positions, creating massive imbalance)
```

**Signal:**
- **Risk reduction:** VPIN > 0.5 -- reduce position by 30%, widen stops (informed traders active)
- **Flash crash alert:** VPIN > 0.7 -- flatten or add tail hedges (extreme information asymmetry)
- **Normal trading:** VPIN < 0.3 -- full position sizes (symmetric market, fair pricing)
- **Edge opportunity:** VPIN > 0.5 AND you can identify the direction of informed flow -> join them

**Risk:** VPIN-based position sizing; Automatic deleveraging when VPIN > 0.5; Stop-loss tightening
**Edge:** VPIN measures the probability that the person on the other side of your trade has better information than you. When VPIN is high, market makers are being adversely selected (they lose to informed traders), spreads widen, and liquidity evaporates. This is the earliest warning of information-driven market disruption. It predicted the Flash Crash, multiple EM crises, and individual stock collapses days before they occurred.

---

### 162 | Shanghai Opening Auction Volume Signal
**School:** Shanghai (SSE A-share Trading) | **Class:** Opening Auction
**Timeframe:** Opening 30 min | **Assets:** SSE A-shares, CSI 300

**Mathematics:**
```
Shanghai Opening Auction (09:15-09:25 CST):
  Auction_Volume = volume matched in opening call auction
  Auction_Price = equilibrium price from order matching

Auction Volume Signal:
  AV_ratio = Auction_Volume / SMA(Auction_Volume, 20)
  
  AV_ratio > 2.0: Abnormal opening interest (institutional activity or news)
  AV_ratio < 0.5: Thin opening (no conviction, likely range day)

Auction Price vs. Prior Close:
  AP_gap = (Auction_Price - Prior_Close) / Prior_Close * 100
  
  Large gap up (>1.5%) + high AV: Bullish momentum (follow gap)
  Large gap down (<-1.5%) + high AV: Bearish momentum (follow gap)
  Small gap (<0.5%) + low AV: Range day (fade extremes)

Post-Auction Pattern (09:25-10:00):
  If gap fills within first 30 min: fade signal (gap was retail noise)
  If gap extends in first 30 min: trend signal (institutional follow-through)
```

**Signal:**
- **Trend follow:** AV_ratio > 2.0 AND |AP_gap| > 1.5% AND gap extends in first 30 min
- **Fade gap:** AV_ratio < 1.0 AND |AP_gap| > 1% (retail-driven gap with no volume support)
- **No trade:** AV_ratio 0.8-1.5 AND |AP_gap| < 0.5% (boring open, no edge)

**Risk:** Intraday stops at gap extreme; Target 2x gap size (for follow) or gap fill (for fade); Risk 0.5%
**Edge:** Shanghai's opening auction is a unique mechanism where orders accumulate for 10 minutes before matching. Abnormal auction volume reveals institutional pre-positioning before the continuous session. In China's retail-dominated market, auction volume > 2x average almost always signals institutional activity (state funds, Northbound Connect flows), providing a reliable early-morning directional signal.

---

### 163 | Money Flow Index (MFI) Divergence System
**School:** TradingView Standard | **Class:** Volume-Weighted Oscillator
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Money Flow Index (Gene Quong & Avrum Soudack):
  Typical_Price = (High + Low + Close) / 3
  Money_Flow = Typical_Price * Volume
  
  If Typical_Price > Typical_Price_{t-1}:
    Positive_MF += Money_Flow
  Else:
    Negative_MF += Money_Flow

  MFR = Positive_MF(n) / Negative_MF(n)  (n = 14)
  MFI = 100 - 100/(1 + MFR)

  MFI > 80: Overbought (high positive money flow)
  MFI < 20: Oversold (high negative money flow)

MFI Divergence:
  Bullish: Price new low BUT MFI higher low (money flowing in despite price drop)
  Bearish: Price new high BUT MFI lower high (money flowing out despite price rise)

MFI vs RSI:
  MFI incorporates volume; RSI does not
  MFI divergence is MORE significant than RSI divergence because it represents
  actual dollar flow, not just price momentum
```

**Signal:**
- **Buy:** MFI < 20 (oversold) AND bullish divergence forming
- **Sell:** MFI > 80 (overbought) AND bearish divergence forming
- **Confirmation:** Wait for MFI to cross back above 20 (buy) or below 80 (sell)
- **Exit:** MFI reaches opposite extreme or divergence resolves

**Risk:** Stop below oversold price low; Target at prior swing high; Risk 1%
**Edge:** MFI is the volume-weighted RSI -- it captures both price momentum AND volume conviction. When MFI diverges from price, it means the MONEY FLOW doesn't support the price movement. This is more reliable than pure price divergence because it requires actual volume participation to confirm. In low-float or mid-cap stocks where volume reveals institutional interest, MFI divergence has particular predictive power.

---

### 164 | Tick Volume Momentum Oscillator
**School:** Forex/CME (Tick-based Analysis) | **Class:** Tick Microstructure
**Timeframe:** Intraday (1-min to 5-min) | **Assets:** Forex, Futures

**Mathematics:**
```
Tick Volume = number of price changes per bar (in forex where real volume unavailable)

Tick Volume Relative:
  TV_rel = TickVolume / EMA(TickVolume, 50)
  TV_rel > 1.5: Active period (more tick changes than normal)
  TV_rel < 0.7: Quiet period (fewer changes)

Tick Momentum Oscillator:
  TMO = EMA(sign(Close - Open) * TickVolume, fast) - EMA(sign(Close-Open) * TickVolume, slow)
  fast = 5, slow = 13

  TMO > 0: Bullish tick momentum (more up-ticks with volume)
  TMO < 0: Bearish tick momentum
  TMO crossing zero: momentum shift

Uptick/Downtick Ratio:
  UT = count(price_changes > 0) / total_ticks
  DT = count(price_changes < 0) / total_ticks
  Tick_Ratio = UT / DT
  
  Tick_Ratio > 1.5: Strong buying pressure at tick level
  Tick_Ratio < 0.67: Strong selling pressure
```

**Signal:**
- **Long:** TMO crosses above zero AND TV_rel > 1.0 (momentum shift with activity)
- **Short:** TMO crosses below zero AND TV_rel > 1.0
- **Exit:** TMO crosses back to zero or TV_rel < 0.5 (low activity, noise)
- **Avoid:** TV_rel < 0.7 (quiet periods produce false TMO signals)

**Risk:** 2x ATR stop; Reduce position when TV_rel < 0.8; Risk 0.5%
**Edge:** Tick volume is the only volume proxy available in decentralized forex markets. The TMO combines directional tick analysis with volume momentum to detect when aggressive market orders are driving price in one direction. The activity filter (TV_rel) ensures you only trade during periods when the tick data is statistically meaningful.

---

### 165 | Force Index (Elder) Exhaustion Pattern
**School:** New York (Alexander Elder) | **Class:** Volume-Force
**Timeframe:** Daily / Weekly | **Assets:** Equities

**Mathematics:**
```
Force Index:
  FI = (Close_t - Close_{t-1}) * Volume_t
  FI_smoothed = EMA(FI, 2) (short-term) or EMA(FI, 13) (medium-term)

  FI > 0: Buying force (price up with volume)
  FI < 0: Selling force (price down with volume)

  |FI| magnitude: strength of the force (large move * large volume = big force)

Exhaustion Pattern:
  FI_Z = FI / StdDev(FI, 60)
  
  Buying_Exhaustion = FI_Z > +3 (extreme buying force, unsustainable)
  Selling_Exhaustion = FI_Z < -3 (extreme selling force, capitulation)

Elder Triple Screen Integration:
  Screen 1 (weekly): EMA(FI, 13) for trend direction
  Screen 2 (daily): EMA(FI, 2) for pullback detection
  Screen 3 (intraday): Entry timing

  Buy: Weekly FI > 0 (uptrend) AND Daily FI < 0 (pullback) -> enter long
  Sell: Weekly FI < 0 (downtrend) AND Daily FI > 0 (bounce) -> enter short
```

**Signal:**
- **Long pullback:** Weekly FI > 0 AND Daily FI_Z < -1.5 (pullback in uptrend)
- **Short bounce:** Weekly FI < 0 AND Daily FI_Z > +1.5 (bounce in downtrend)
- **Exhaustion counter-trade:** FI_Z > +3 -> prepare for reversal (selling climax of buying)
- **Exit:** Daily FI returns to zero or weekly FI changes sign

**Risk:** Stop at pullback extreme; Target at prior swing; Risk 1%
**Edge:** Force Index multiplies price change by volume, creating a direct measure of the "power" behind a move. Large price moves on small volume have weak force (unsustainable). Small price moves on massive volume have strong force (institutional accumulation). The triple screen framework uses multiple timeframes to ensure you only trade pullbacks within established trends, dramatically improving win rate.

---

### 166 | Volume Spread Divergence (Market Maker Absorption)
**School:** London (Tom Williams, VSA Extended) | **Class:** Smart Money Detection
**Timeframe:** Daily | **Assets:** Equities, Indices

**Mathematics:**
```
Volume-Spread Relationship:
  Expected: Wide spread (range) should come with high volume
            Narrow spread should come with low volume

  Anomaly_Wide_Low: Spread > 1.5*ATR BUT Volume < 0.8*avg
    (wide range on low volume = manipulation, not genuine move)
  
  Anomaly_Narrow_High: Spread < 0.5*ATR BUT Volume > 1.5*avg
    (narrow range on high volume = absorption, professionals buying/selling into move)

Absorption Detection:
  1. Price approaching resistance
  2. Volume increasing
  3. Spread NARROWING (close-open distance shrinking)
  = Professional sellers absorbing all buying pressure at resistance
  
  Absorption_Score = Volume_Z * (1 / Spread_Z)  (high volume, low spread = high score)

Test of Supply/Demand:
  After absorption:
    If price attempts breakout and volume DROPS = no supply left, breakout succeeds
    If price attempts breakout and volume SURGES = supply still present, breakout fails
```

**Signal:**
- **Sell (supply absorption):** Absorption at resistance + breakout attempt with low volume = fake breakout
- **Buy (demand absorption):** Absorption at support + breakdown attempt with low volume = spring
- **Confirmation:** Post-absorption test with declining volume = genuine signal

**Risk:** Stop above resistance (short) or below support (long); Tight 1.5x ATR stops; Risk 1%
**Edge:** Professional market makers and institutional traders leave their fingerprint in the volume-spread relationship. When they absorb supply (buying) at support, volume rises but range narrows -- they match every sell order without letting price fall. The subsequent "test" bar reveals whether all supply has been absorbed. This is the purest form of smart money detection available in public market data.

---

### 167 | Relative Volume at Time of Day (RVOL)
**School:** US Day Trading (Modern Intraday) | **Class:** Volume Context
**Timeframe:** Intraday | **Assets:** US Equities

**Mathematics:**
```
RVOL = Volume_so_far_today / Average_Volume_at_this_time_of_day(n_days)

  At 10:00 AM: Current_Volume = 500K shares
  Average at 10:00 AM over last 20 days = 300K shares
  RVOL = 500/300 = 1.67 (67% above average for this time of day)

Time-of-Day Volume Profile (typical US stock):
  09:30-10:00: 15-20% of daily volume (opening rush)
  10:00-11:30: 20-25% (institutional execution)
  11:30-14:00: 15-20% (lunch lull)
  14:00-15:30: 20-25% (afternoon session)
  15:30-16:00: 15-20% (closing cross)

RVOL Significance:
  RVOL > 2.0: Significant unusual volume (news, earnings, sector catalyst)
  RVOL > 5.0: Extreme (likely news-driven, be cautious of one-time events)
  RVOL < 0.5: Unusually quiet (low conviction, avoid new positions)

Sector RVOL:
  If single stock RVOL > 3.0 but sector RVOL < 1.0: stock-specific catalyst
  If sector RVOL > 2.0: sector-wide event (more sustainable move)
```

**Signal:**
- **High conviction entry:** RVOL > 2.0 AND price breaking key level (volume-confirmed breakout)
- **Avoid:** RVOL < 0.7 (below average activity, no institutional interest)
- **Size by RVOL:** position_size = base * min(RVOL / 2.0, 1.0) (scale with conviction)
- **Sector vs. stock:** Stock RVOL > Sector RVOL by 2x+ = stock-specific (more edge)

**Risk:** Standard ATR stops; Reduce targets when RVOL > 5 (one-time event, may not sustain)
**Edge:** Raw volume is meaningless without time-of-day context. 100K shares at 9:30 AM is normal; 100K at 12:30 PM is extraordinary. RVOL normalizes volume by the time-of-day pattern, providing the TRUE measure of unusual activity. This is the single most important filter for intraday breakout traders -- high RVOL breakouts succeed ~65% of the time vs. ~45% for low RVOL.

---

### 168 | Klinger Volume Oscillator (KVO) Momentum
**School:** New York (Stephen Klinger) | **Class:** Volume Oscillator
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
Volume Force (VF):
  dm = High + Low + Close (daily mean)
  trend = 1 if dm > dm_{t-1} else -1
  
  VF = Volume * |2 * (dm - dm_{t-1}) / dm_{t-1}| * trend * 100

KVO = EMA(VF, 34) - EMA(VF, 55)
Signal_Line = EMA(KVO, 13)

Interpretation:
  KVO > 0: Positive volume force (accumulation stronger than distribution)
  KVO < 0: Negative volume force (distribution stronger)
  
  KVO crossing Signal_Line from below: Buy signal
  KVO crossing Signal_Line from above: Sell signal

KVO Divergence:
  Price new high + KVO lower high = bearish (distribution divergence)
  Price new low + KVO higher low = bullish (accumulation divergence)

KVO Trend Strength:
  KVO_Z = normalize(KVO, 60)
  |KVO_Z| > 2: Strong volume momentum in that direction
```

**Signal:**
- **Buy:** KVO crosses above signal line AND KVO_Z > 0 (positive momentum with crossover)
- **Sell:** KVO crosses below signal line AND KVO_Z < 0
- **Divergence trade:** Price new extreme but KVO diverges = prepare for reversal
- **Exit:** KVO crosses signal line in opposite direction

**Risk:** Stop at prior swing; Trail with signal line; Risk 1%
**Edge:** KVO is specifically designed to detect the flow of money into and out of a security by measuring both the direction and the magnitude of money flow relative to the price trend. Unlike simpler volume indicators (OBV, AD), KVO uses dual EMA crossover for timing, making it self-contained as a trading system. The 34/55 period combination aligns with the Fibonacci sequence, capturing natural market cycles.

---

### 169 | Dark Pool Volume Detection
**School:** New York (Modern Market Structure) | **Class:** Off-Exchange Flow
**Timeframe:** Daily / Intraday | **Assets:** US Equities

**Mathematics:**
```
US Market Structure:
  ~40% of equity volume trades on dark pools (ATS - Alternative Trading Systems)
  Dark pool trades reported to FINRA with delay
  
Dark Pool Indicators:
  1. Short Volume Ratio = Short_Volume / Total_Volume (from FINRA data)
     Normal: 0.40-0.50 (routine market making)
     Elevated: > 0.55 (bearish positioning)
     Suppressed: < 0.35 (bullish positioning)

  2. Dark Pool % = Off_Exchange_Volume / Total_Volume
     If Dark_Pool_% increasing AND price flat: large order being executed off-exchange
     (institutional hiding large trades from lit market)

  3. Block Trade Frequency = count(trades > 10,000 shares or $200,000) / day
     Unusual block activity = institutional repositioning

Short Volume Z-score:
  SVR_Z = (SVR - SMA(SVR, 30)) / StdDev(SVR, 30)
  SVR_Z > +2.0: Unusually high short volume (bearish signal)
  SVR_Z < -2.0: Unusually low short volume (bullish signal)
```

**Signal:**
- **Bullish:** SVR_Z < -1.5 AND Dark_Pool_% increasing AND price stable (stealth accumulation)
- **Bearish:** SVR_Z > +2.0 AND Dark_Pool_% increasing (stealth distribution or short building)
- **Breakout anticipation:** Block_Trade_Frequency > 3x normal AND price consolidating

**Risk:** Confirmatory signal (use with price action); Standard stops; Risk 1%
**Edge:** 40% of US equity volume is invisible to lit-market participants. Dark pool data, available with a delay from FINRA, reveals institutional positioning that price-only analysis cannot detect. Short volume ratio is the most actionable dark pool metric: when short selling surges (SVR > 55%), institutions are actively positioning for a decline. When it collapses (< 35%), shorts are covering. This information asymmetry is the modern equivalent of insider knowledge.

---

### 170 | Chaikin Money Flow (CMF) Persistence
**School:** New York (Marc Chaikin) | **Class:** Money Flow Persistence
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
CLV = ((Close - Low) - (High - Close)) / (High - Low)
  Range: [-1, +1]

CMF(n) = sum(CLV * Volume, n) / sum(Volume, n)
  n = 21 (standard)

  CMF > 0: Money flowing IN (accumulation)
  CMF < 0: Money flowing OUT (distribution)

CMF Persistence:
  CMF_Streak = consecutive days CMF has been same sign
  Long_Persistence = CMF > 0 for 15+ days (sustained accumulation)
  Short_Persistence = CMF < 0 for 15+ days (sustained distribution)

  Historical:
    After 15+ day positive CMF streak: 65% probability next 20 days are positive
    After 15+ day negative CMF streak: 60% probability next 20 days are negative

CMF Thrust:
  CMF_thrust = CMF crosses from < -0.10 to > +0.10 within 5 days
  = rapid shift from distribution to accumulation (institutional repositioning)
  Thrust has 72% reliability for upward continuation over next 20 days
```

**Signal:**
- **Long:** CMF_thrust (rapid shift to accumulation) OR CMF_Streak > 15 days positive
- **Short:** CMF_thrust negative OR CMF_Streak > 15 days negative
- **Exit:** CMF reverses sign and stays reversed for 5+ days

**Risk:** Stop at prior swing; Target 20-day horizon; Risk 1%
**Edge:** CMF Persistence captures sustained institutional activity that OBV and other cumulative indicators miss. A single day of high-volume accumulation could be a one-off event. But 15+ consecutive days of positive CMF means institutions are systematically accumulating, which has strong predictive power. The CMF Thrust pattern (rapid shift from distribution to accumulation) is the strongest variant, capturing the exact moment institutions change their positioning.

---

### 171 | Volume Rate of Change (VROC) Breakout Filter
**School:** TradingView Community | **Class:** Volume Momentum
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
VROC = (Volume - Volume_{n_ago}) / Volume_{n_ago} * 100
  n = 14 (14-day rate of change)

VROC_EMA = EMA(VROC, 5)  (smoothed for less noise)

Breakout Quality Filter:
  Good_Breakout = Price_breakout AND VROC > +50% (volume increased 50%+ vs 14 days ago)
  Weak_Breakout = Price_breakout AND VROC < +20% (insufficient volume confirmation)
  Failed_Breakout = Price_breakout AND VROC < 0% (volume DECLINING during breakout)

VROC Divergence:
  Price trending up AND VROC declining = weakening momentum (fewer participants)
  Price trending down AND VROC declining = selling exhaustion

Volume Accumulation Phase:
  VROC oscillating between -20% and +20% for 10+ days = volume base building
  First VROC spike > +80% after base = accumulation complete, breakout beginning
```

**Signal:**
- **Buy breakout:** Price breaks resistance AND VROC > +50% (volume-confirmed breakout)
- **Skip breakout:** Price breaks resistance BUT VROC < +20% (likely false breakout)
- **Accumulation breakout:** VROC base (oscillating) followed by spike > +80% = high-probability entry
- **Exit:** VROC turns negative for 3+ days in a trend (participation declining)

**Risk:** Standard ATR stops; Target open-ended if VROC stays elevated; Risk 1.5%
**Edge:** Volume Rate of Change directly measures whether trading interest is ACCELERATING (more participants joining) or DECELERATING (participants leaving). Most breakout strategies fail because they ignore volume acceleration. A breakout with VROC > 50% means significantly more traders are participating vs. 14 days ago -- this crowding drives the initial move and creates momentum. Breakouts with declining VROC are traps.

---

### 172 | Ease of Movement (EMV) Indicator
**School:** Arms/TradingView | **Class:** Volume-Price Relationship
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
Ease of Movement (Richard Arms):
  Distance_Moved = ((H + L)/2 - (H_{t-1} + L_{t-1})/2)
  Box_Ratio = (Volume / 10000) / (H - L)
  
  EMV = Distance_Moved / Box_Ratio
  EMV_smoothed = SMA(EMV, 14)

  EMV > 0: Price moving up easily (low volume required for upward movement)
  EMV < 0: Price moving down easily (low volume required for downward movement)
  EMV near 0: Price struggling to move (high volume needed for small moves)

Key Concept:
  "Ease" = how much price change you get per unit of volume
  Easy upward movement (high positive EMV) = bullish: buyers in control, little resistance
  Easy downward movement (high negative EMV) = bearish: sellers in control
  Difficult movement (EMV near 0 despite price change) = congestion, reversal ahead

EMV Divergence:
  Price rising BUT EMV declining = movement becoming HARDER (bearish)
  Price falling BUT EMV rising = downside becoming HARDER (bullish, support building)
```

**Signal:**
- **Long:** EMV_smoothed crosses above zero (easy upward movement beginning)
- **Short:** EMV_smoothed crosses below zero (easy downward movement beginning)
- **Exit:** EMV approaches zero from either direction (movement becoming difficult)
- **Warning:** EMV near zero for extended period = consolidation, breakout imminent

**Risk:** Stop at EMV zero-line; Target at EMV extreme (movement becoming too easy = exhaustion)
**Edge:** EMV uniquely measures the EFFICIENCY of price movement relative to volume. High EMV means price is moving freely with little resistance. When EMV diverges from price, it reveals that the move is becoming increasingly difficult -- more volume is required for the same price change, indicating growing resistance. This concept of "ease vs. effort" is fundamental to understanding market dynamics.

---

### 173 | Net Volume Pressure Oscillator
**School:** Quantitative | **Class:** Buy/Sell Pressure
**Timeframe:** Intraday (5-min) | **Assets:** Futures, ETFs

**Mathematics:**
```
Buy Pressure (BP):
  BP = Close - Low  (portion of bar captured by buyers)

Sell Pressure (SP):
  SP = High - Close  (portion of bar captured by sellers)

Net Volume Pressure (NVP):
  NVP = (BP - SP) / (High - Low) * Volume
  (directional volume, scaled by who controlled the bar)

NVP Oscillator:
  NVPO = EMA(NVP, 8) - EMA(NVP, 21)
  NVPO_signal = EMA(NVPO, 5)

Pressure Accumulation:
  Cum_NVP = cumsum(NVP)  (cumulative from session start)
  
  If Cum_NVP rising: sustained buying pressure throughout session
  If Cum_NVP falling: sustained selling pressure
  If Cum_NVP flat: balanced (no net pressure)

Pressure Intensity:
  PI = |NVP| / Volume  (what fraction of volume is directional)
  PI > 0.7: Highly directional bar (one-sided flow)
  PI < 0.3: Balanced bar (two-way flow)
```

**Signal:**
- **Long:** NVPO crosses above signal AND Cum_NVP rising AND PI > 0.5
- **Short:** NVPO crosses below signal AND Cum_NVP falling AND PI > 0.5
- **Exit:** NVPO crosses signal in opposite direction OR PI < 0.3 (flow becoming balanced)
- **Avoid:** PI < 0.3 (balanced flow, no directional edge)

**Risk:** Tight intraday stops; 1.5x ATR; Risk 0.5%
**Edge:** NVP measures who WON each bar by analyzing where price closed relative to the range, then scales by volume. This is superior to simple up/down volume classification because it captures the DEGREE of buyer/seller dominance within each bar. The pressure intensity filter ensures you only trade when there is genuinely one-sided flow, avoiding the noise of balanced two-way trading.

---

### 174 | Arms Index (TRIN) Extreme Reversal
**School:** New York (Richard Arms, 1967) | **Class:** Market Breadth + Volume
**Timeframe:** Daily / Intraday | **Assets:** NYSE, NASDAQ

**Mathematics:**
```
TRIN (Trading Index / Arms Index):
  TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

  TRIN < 1.0: Volume is concentrated in advancing stocks (bullish)
  TRIN > 1.0: Volume is concentrated in declining stocks (bearish)
  TRIN = 1.0: Balanced

Interpretation:
  TRIN < 0.5: Extreme bullish (buyers very aggressive) -> overbought
  TRIN > 2.0: Extreme bearish (sellers capitulating) -> oversold
  TRIN > 3.0: Panic selling -> capitulation bottom likely

10-day MA(TRIN):
  MA10_TRIN > 1.3: Market oversold on sustained basis -> buy signal
  MA10_TRIN < 0.8: Market overbought on sustained basis -> sell signal

TRIN Spike vs. Level:
  Single TRIN > 2.0: emotional event (may not sustain)
  MA(TRIN, 5) > 1.5: sustained panic -> higher probability reversal
```

**Signal:**
- **Buy (oversold):** MA10_TRIN > 1.3 OR single TRIN > 3.0 with follow-through bullish day
- **Sell (overbought):** MA10_TRIN < 0.75 for 5+ days (sustained euphoria)
- **Exit:** TRIN normalizes to 0.85-1.15 range

**Risk:** Market-level indicator; size full index positions (SPY/QQQ); Stop at further TRIN extreme; Risk 2%
**Edge:** TRIN combines breadth (advancing vs. declining issues) with volume (where the volume is going). This is superior to breadth or volume alone because it reveals the CONVICTION behind the breadth. When TRIN > 2, not only are more stocks declining, but the declining volume is concentrated -- this is genuine panic. Historical data shows that 10-day MA TRIN > 1.3 preceded every major market bottom in the last 50 years.

---

### 175 | Volume Zone Oscillator (VZO)
**School:** TradingView (Walid Khalil & David Steckler) | **Class:** Volume Zone
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
Volume Zone Classification:
  VP = Volume if Close > Close_{t-1} else -Volume
  TVP = EMA(VP, 14)  (trend volume pressure)
  TV = EMA(Volume, 14)  (trend volume)

  VZO = (TVP / TV) * 100

  Zones:
    VZO > +40: Overbought (heavy buying pressure)
    +5 < VZO < +40: Bullish zone (moderate buying)
    -5 < VZO < +5: Neutral zone
    -40 < VZO < -5: Bearish zone (moderate selling)
    VZO < -40: Oversold (heavy selling pressure)

Trading Rules:
  Buy: VZO crosses above +5 from below (entering bullish zone)
  Sell: VZO crosses below -5 from above (entering bearish zone)
  Take Profit: VZO enters overbought (>+40) or oversold (<-40)
  
  Trend Filter: Only buy when EMA(14) > EMA(60) (trend confirmation)

VZO Divergence:
  Price new high + VZO lower high = volume distribution (bearish)
  Price new low + VZO higher low = volume accumulation (bullish)
```

**Signal:**
- **Long:** VZO crosses above +5 AND price above EMA(60) (bullish zone entry in uptrend)
- **Short:** VZO crosses below -5 AND price below EMA(60)
- **Exit long:** VZO > +40 (overbought) OR VZO crosses below -5
- **Exit short:** VZO < -40 (oversold) OR VZO crosses above +5

**Risk:** Stop at prior swing; Trail with 2x ATR; Risk 1%
**Edge:** VZO simplifies volume analysis into clear zones, making it easier to identify when volume is genuinely supporting price movement. The zone transitions (+5 and -5 crossovers) create clean entry signals, while the extreme zones (+40, -40) provide profit-taking levels. Unlike OBV or CMF which are cumulative, VZO is a bounded oscillator making overbought/oversold analysis straightforward.

---

### 176 | Footprint Chart Delta Imbalance
**School:** Chicago (Sierra Chart / OrderFlow+) | **Class:** Bid-Ask Microstructure
**Timeframe:** Intraday (1-min to 5-min) | **Assets:** ES, NQ, CL futures

**Mathematics:**
```
Footprint Chart:
  At each price level within a bar, show:
    Bid_Volume(price) = volume transacted at bid (aggressive sellers)
    Ask_Volume(price) = volume transacted at ask (aggressive buyers)
    Delta(price) = Ask_Volume(price) - Bid_Volume(price)

Imbalance at Price Level:
  Imbalance_Up = Ask_Volume(P) / Bid_Volume(P)
  If Imbalance_Up > 3.0: 3:1 buyers vs sellers at this level = demand zone

  Imbalance_Down = Bid_Volume(P) / Ask_Volume(P)
  If Imbalance_Down > 3.0: 3:1 sellers vs buyers = supply zone

Stacked Imbalance:
  3+ consecutive price levels with Imbalance > 3.0 in same direction
  = very strong directional conviction at that price zone

Bar Delta:
  Bar_Delta = sum(Delta(price)) across all prices in bar
  If Bar_Delta >> 0: Strong buying bar
  If Bar_Delta << 0: Strong selling bar
  
Bar_Delta_Z = normalize(Bar_Delta, 100_bars)
```

**Signal:**
- **Long:** Stacked buy imbalance (3+ levels with 3:1 ask/bid) at support level
- **Short:** Stacked sell imbalance (3+ levels with 3:1 bid/ask) at resistance level
- **Entry:** On pullback to stacked imbalance zone (these levels act as support/resistance)
- **Exit:** Bar_Delta_Z reverses direction AND new stacked imbalance appears against position

**Risk:** Stop 2 ticks beyond imbalance zone; Target at next stacked imbalance in opposite direction
**Edge:** Footprint charts reveal the actual bid-ask volume at EACH price level, providing the most granular view of order flow available. Stacked imbalances (3+ consecutive levels with 3:1+ ratio) represent genuine institutional conviction at specific prices. These zones become powerful support/resistance because the institutional orders that created them may not be fully filled, creating resting limit orders that defend the level.

---

### 177 | Weis Wave Volume Analysis
**School:** Chicago (David Weis, Wyckoff Extension) | **Class:** Wave Volume
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Weis Wave:
  1. Identify price waves (swings between higher and lower pivots)
  2. For each wave, sum the volume:
     Up_Wave_Volume = sum(Volume) during price advance
     Down_Wave_Volume = sum(Volume) during price decline
  3. Plot cumulative volume PER WAVE (resets at each swing pivot)

Wave Volume Comparison:
  Current_Up_Wave_Vol vs Prior_Up_Wave_Vol
  If current > prior: Increasing buying interest (bullish)
  If current < prior: Decreasing buying interest (bearish)

  Current_Down_Wave_Vol vs Prior_Down_Wave_Vol
  If current > prior: Increasing selling pressure (bearish)
  If current < prior: Decreasing selling pressure (bullish)

Effort vs Result:
  Effort = Wave_Volume (volume during wave)
  Result = Wave_Price_Change (points moved during wave)
  
  High_Effort_Low_Result: Lots of volume, little price movement = absorption
  Low_Effort_High_Result: Little volume, big price movement = easy movement (continuation)
```

**Signal:**
- **Buy:** Down_wave volume decreasing + up_wave volume increasing (effort shifting to buyers)
- **Sell:** Up_wave volume decreasing + down_wave volume increasing (effort shifting to sellers)
- **Absorption alert:** High volume wave with small price result = reversal imminent
- **Continuation:** Low volume wave with large price result = trend intact

**Risk:** Stop at prior wave pivot; Target at projected wave extension; Risk 1.5%
**Edge:** Weis Wave Volume aggregates volume by PRICE WAVE rather than by time bar, revealing which side (buyers or sellers) is putting in more effort over the course of each swing. This is Wyckoff analysis quantified. The effort-vs-result metric directly measures whether volume is producing proportional price movement. When it's not (high effort, low result), institutions are absorbing the move -- the definition of a reversal setup.

---

### 178 | Volume Weighted Momentum (VWM)
**School:** Quantitative | **Class:** Volume-Enhanced Momentum
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Standard Momentum: M = Close - Close_n
  Problem: Treats all price changes equally regardless of volume

Volume-Weighted Momentum:
  VWM = sum((Close_i - Close_{i-1}) * Volume_i, n) / sum(Volume_i, n)
  = average price change weighted by volume over n periods
  n = 14

VWM emphasizes price changes that occurred on HIGH volume (institutional moves)
VWM diminishes price changes on LOW volume (noise)

VWM_Z = normalize(VWM, 60)

VWM vs. Standard Momentum Divergence:
  If VWM positive BUT Standard_Mom negative:
    Volume-confirmed moves are positive, but recent low-volume moves are negative
    = underlying accumulation despite surface weakness (bullish)
  If VWM negative BUT Standard_Mom positive:
    Volume-confirmed moves are negative, but recent low-volume moves are positive
    = underlying distribution despite surface strength (bearish)
```

**Signal:**
- **Long:** VWM_Z > +1.5 AND VWM-Std_Mom divergence bullish (volume-backed momentum)
- **Short:** VWM_Z < -1.5 AND VWM-Std_Mom divergence bearish
- **Filter:** Only trade when VWM and Standard_Mom agree (both positive or both negative)
- **Exit:** VWM crosses zero

**Risk:** Standard ATR stops; Trail with 2x ATR; Risk 1%
**Edge:** Volume-weighted momentum separates MEANINGFUL price moves (high volume = institutional) from NOISE (low volume = retail). This is the mathematical formalization of the Wyckoff principle that volume validates price movement. When VWM and standard momentum diverge, the market is showing you which moves had genuine institutional backing -- always follow the volume-weighted signal over the price-only signal.

---

### 179 | Negative Volume Index (NVI) Smart Money Tracker
**School:** New York (Paul Dysart, 1930s; Norman Fosback, 1976) | **Class:** Smart Money
**Timeframe:** Daily | **Assets:** Equities, Indices

**Mathematics:**
```
NVI (Negative Volume Index):
  If Volume_t < Volume_{t-1}:
    NVI_t = NVI_{t-1} * (1 + ROC_t)  (update NVI with return)
  Else:
    NVI_t = NVI_{t-1}  (no change)

  NVI only changes on DOWN-VOLUME days
  Theory: Smart money trades on low-volume days (avoiding crowd)

PVI (Positive Volume Index):
  If Volume_t > Volume_{t-1}:
    PVI_t = PVI_{t-1} * (1 + ROC_t)
  Else:
    PVI_t = PVI_{t-1}

  PVI only changes on UP-VOLUME days (crowd behavior)

Trading Rules (Fosback):
  Bull market: NVI > SMA(NVI, 255) = 96% probability (Fosback's research)
  Bear market: NVI < SMA(NVI, 255) = high probability

  NVI + PVI Combined:
    NVI rising + PVI rising = smart + crowd both bullish (strong)
    NVI rising + PVI falling = smart money bullish, crowd bearish (smart money wins)
    NVI falling + PVI rising = crowd bullish, smart money bearish (warning)
```

**Signal:**
- **Long:** NVI > SMA(NVI, 255) AND NVI rising (smart money in bull mode)
- **Extra confidence:** NVI rising AND PVI falling (smart money buying while crowd sells = strongest)
- **Bearish:** NVI < SMA(NVI, 255) (smart money in bear mode)
- **Warning:** NVI falling AND PVI rising (crowd chasing, smart money distributing)

**Risk:** Long-term indicator; position-level sizing; Stop at NVI crossing below SMA(255); Risk 2%
**Edge:** NVI captures price changes that occur on DECLINING volume days -- when professional traders are most active (they prefer to accumulate in quiet markets). Fosback's research showed NVI > its 1-year MA correctly identified bull markets 96% of the time. The NVI-PVI divergence (smart money vs. crowd) provides the most reliable long-term market direction signal, historically outperforming price-only trend indicators.

---

### 180 | Volume Oscillator Breakout Confirmation
**School:** TradingView Standard | **Class:** Volume Oscillator
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Volume Oscillator:
  VO = ((EMA(Volume, short) - EMA(Volume, long)) / EMA(Volume, long)) * 100
  short = 5, long = 20

  VO > 0: Short-term volume above long-term (increasing participation)
  VO < 0: Short-term volume below long-term (decreasing participation)

VO as Breakout Confirmation:
  Good breakout = Price breaks level AND VO > +30% (50% more volume on short EMA)
  Weak breakout = Price breaks level AND VO < +10%
  Fake breakout = Price breaks level AND VO < 0% (declining volume at breakout)

VO Trend:
  VO_slope = linreg_slope(VO, 10)
  Rising VO in price trend = healthy (participation growing)
  Falling VO in price trend = unhealthy (participation shrinking = exhaustion)

Volume Dry-Up:
  VO < -40%: Extreme volume contraction (pre-breakout coiling)
  After volume dry-up, first VO > +30%: breakout confirmation
```

**Signal:**
- **Breakout entry:** Price breaks resistance/support AND VO > +30% (volume-confirmed)
- **Avoid entry:** Price breaks level BUT VO < +10% (insufficient volume, likely failure)
- **Pre-breakout setup:** VO < -40% for 5+ days THEN VO > +30% = volume expansion from compression
- **Exit:** VO turns negative AND stays negative for 3 days (participation declining)

**Risk:** Stop just inside the broken level; Target at projected breakout range; Risk 1.5%
**Edge:** The volume oscillator provides a simple, quantified answer to "is volume confirming this breakout?" The +30% threshold is based on empirical research showing breakouts with 30%+ volume expansion succeed 65% of the time vs. 42% without. The volume dry-up pattern (VO < -40% before breakout) is particularly powerful because it identifies the energy buildup that precedes explosive moves.

---

### 181 | Tokyo Opening Gap Volume Filter
**School:** Tokyo (TSE Volume Analysis) | **Class:** Gap Trading
**Timeframe:** Daily | **Assets:** Nikkei 225, TOPIX stocks

**Mathematics:**
```
Opening Gap:
  Gap_Pct = (Open - Prior_Close) / Prior_Close * 100

Gap Volume Assessment:
  Pre_Open_Volume = estimated volume from pre-market auction
  Relative_Gap_Volume = Pre_Open_Volume / SMA(Pre_Open_Volume, 20)

Gap Classification:
  Professional Gap: |Gap_Pct| > 0.5% AND Relative_Gap_Volume > 2.0
    (institutional order flow causing gap with volume = likely continuation)
  
  Retail Gap: |Gap_Pct| > 0.5% AND Relative_Gap_Volume < 1.0
    (gap on thin volume = likely to be faded)
  
  News Gap: |Gap_Pct| > 1.5% regardless of volume
    (event-driven, different behavior)

First-Hour Volume Test:
  FH_Vol_Ratio = Volume(first_hour) / SMA(Volume_first_hour, 20)
  If Professional_Gap AND FH_Vol_Ratio > 1.5: gap-and-go (trade in gap direction)
  If Professional_Gap AND FH_Vol_Ratio < 0.8: gap-and-trap (fade the gap)
```

**Signal:**
- **Gap-and-go:** Professional gap + first-hour volume > 1.5x = trade in gap direction
- **Gap fade:** Retail gap (thin volume) -> fade gap toward prior close
- **News gap hold:** |Gap| > 1.5% -> wait for first-hour resolution before entry
- **Exit:** Gap-and-go: trail with 1x ATR. Gap fade: target prior close.

**Risk:** Stop beyond gap extreme; Reduce size for news gaps (unpredictable); Risk 0.5-1%
**Edge:** Japanese markets have unique opening characteristics: the call auction at 09:00 JST aggregates overnight global information, and the gap reflects this repricing. Distinguishing professional gaps (volume-supported) from retail gaps (thin) is the critical edge. Tokyo's market microstructure rewards early volume assessment because institutional orders clustered in the opening auction create sustained directional pressure for the first hour.

---

### 182 | Accumulation Swing Index (ASI)
**School:** New York (J. Welles Wilder) | **Class:** Cumulative Price-Range
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Swing Index (SI):
  SI = 50 * (C - C_{t-1} + 0.5*(C - O) + 0.25*(C_{t-1} - O_{t-1})) / R * (K/T)

  where:
    R = largest of: |H-C_{t-1}|, |L-C_{t-1}|, |H-L|
    K = larger of: |H-C_{t-1}|, |L-C_{t-1}|
    T = limit move (maximum daily range, set to ATR(14)*3)

ASI = cumsum(SI)  (Accumulation Swing Index)

Key Properties:
  ASI makes higher highs before price = bullish leading indicator
  ASI makes lower lows before price = bearish leading indicator
  ASI confirms price breakout = reliable breakout

ASI Breakout Confirmation:
  If price breaks above resistance:
    AND ASI also breaks above its corresponding level: CONFIRMED
    BUT ASI fails to break: UNCONFIRMED (likely false breakout)
```

**Signal:**
- **Long:** Price breaks resistance AND ASI confirms by breaking its own resistance
- **Short:** Price breaks support AND ASI confirms by breaking its own support
- **Leading signal:** ASI breaks out before price = prepare for price breakout
- **Exit:** ASI reverses direction (develops opposite swing pattern)

**Risk:** Stop at ASI swing low (for longs) or high (for shorts); Trail with ASI direction; Risk 1%
**Edge:** Wilder's ASI incorporates open, high, low, close, and the limit move parameter to create a cumulative index that leads price breakouts. By comparing ASI breakouts to price breakouts, you can filter genuine from false breakouts. ASI "sees" the underlying supply-demand balance that simple price charts don't show because it weighs the relationship between consecutive bars' opens and closes.

---

### 183 | Volume Price Trend (VPT) System
**School:** Quantitative/TradingView | **Class:** Volume-Price Hybrid
**Timeframe:** Daily | **Assets:** Equities

**Mathematics:**
```
VPT:
  VPT_t = VPT_{t-1} + Volume * (Close - Close_{t-1}) / Close_{t-1}

  VPT is like OBV but proportional to % price change
  (OBV adds full volume; VPT adds volume * % change)

  This means:
    Large % move on large volume -> big VPT change
    Small % move on large volume -> moderate VPT change
    Large % move on small volume -> small VPT change

VPT vs. OBV:
  OBV: binary (all volume up or all down)
  VPT: proportional (weights by price change magnitude)
  VPT is more nuanced and less susceptible to noise

VPT Signal Line:
  VPT_signal = EMA(VPT, 21)
  VPT above signal: bullish (accumulation)
  VPT below signal: bearish (distribution)

VPT Divergence:
  Price new high + VPT lower high = distribution divergence
  Price new low + VPT higher low = accumulation divergence
```

**Signal:**
- **Long:** VPT crosses above signal AND price above SMA(50) (volume-confirmed uptrend)
- **Short:** VPT crosses below signal AND price below SMA(50)
- **Divergence:** VPT divergence at price extremes = reversal warning
- **Exit:** VPT crosses signal in opposite direction

**Risk:** Stop at prior swing; Trail with 2x ATR; Risk 1%
**Edge:** VPT improves on OBV by making volume contribution proportional to price change. This means a +5% day on high volume contributes 5x more to VPT than a +1% day on the same volume. The result is a cleaner signal that more accurately reflects institutional conviction (they drive large price moves on high volume). VPT divergence from price is one of the most reliable volume-based leading indicators.

---

### 184 | Volume Weighted RSI
**School:** Quantitative | **Class:** Volume-Enhanced Oscillator
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Standard RSI: Based on average gains and losses (price only)

Volume-Weighted RSI:
  VW_Gain = sum(max(ret, 0) * Volume, n) / sum(Volume, n)  (volume-weighted average gain)
  VW_Loss = sum(max(-ret, 0) * Volume, n) / sum(Volume, n)  (volume-weighted average loss)
  
  VW_RS = VW_Gain / VW_Loss
  VW_RSI = 100 - 100 / (1 + VW_RS)

Key Difference from Standard RSI:
  Standard RSI: All days weighted equally
  VW_RSI: High-volume days weighted more (institutional moves count more)

  Divergence between VW_RSI and Standard RSI:
    VW_RSI > RSI: Volume is concentrated on UP days (institutional buying)
    VW_RSI < RSI: Volume is concentrated on DOWN days (institutional selling)

  VW_RSI_spread = VW_RSI - RSI
  VW_RSI_spread_Z = normalize(VW_RSI_spread, 60)
```

**Signal:**
- **Buy:** VW_RSI < 30 (oversold on volume-weighted basis = heavy-volume selling exhausted)
- **Sell:** VW_RSI > 70 (overbought on volume-weighted basis)
- **Institutional signal:** VW_RSI_spread_Z > +1.5 (volume on up-days = institutional buying)
- **Exit:** VW_RSI returns to 50

**Risk:** Standard RSI stops; Trail with VW_RSI direction; Risk 1%
**Edge:** Standard RSI treats a +1% gain on 100K shares the same as a +1% gain on 10M shares. VW_RSI correctly weights the institutional-volume day 100x more. This produces a more accurate picture of whether the asset is truly overbought or oversold based on WHERE the volume is going. The spread between VW_RSI and standard RSI directly reveals institutional positioning bias.

---

### 185 | Intraday Volume Profile Value Area Migration
**School:** Chicago (James Dalton, CBOT) | **Class:** Profile Continuation
**Timeframe:** Intraday + Multi-day | **Assets:** Futures

**Mathematics:**
```
Daily Value Area:
  VA = price range containing 70% of daily volume
  VA_High = upper boundary
  VA_Low = lower boundary
  POC = point of control (highest volume price)

Multi-Day Value Area Migration:
  Track VA boundaries day by day:
  VA_Migration = direction of POC over 5 trading days
  
  Overlapping VA: consecutive days with overlapping value areas
    = balanced market, mean-reversion strategies work
  Migrating VA: consecutive days with non-overlapping, trending value areas
    = trending market, momentum strategies work

Opening Principle:
  Open in Value (between VA_L and VA_H): 
    80% chance of staying in range = fade extremes to POC
  Open out of Value (below VA_L or above VA_H):
    60% chance of re-entering value = test the edge of prior VA
    40% chance of establishing new value = trend continuation
```

**Signal:**
- **Trend entry:** 3+ days of non-overlapping VA migration in same direction = strong trend
- **Range trade:** 3+ days of overlapping VA = fade VA edges to POC
- **Opening trade:** Open out of value -> watch for acceptance (continuation) or rejection (reversal)
- **Exit:** VA migration reverses direction OR overlapping VAs resume

**Risk:** Stop outside prior day's value area; Target at projected VA migration; Risk 1%
**Edge:** Value area migration is the institutional-grade method for identifying whether a market is trending or range-bound. When daily value areas stack higher (each day's POC higher than the last), institutions are continuously accepting higher prices. When VAs overlap, institutions are establishing a range. The opening principle (in-value vs. out-of-value) provides an immediate daily trading framework used by professional futures traders.

---

### 186 | Buy/Sell Imbalance Heatmap Strategy
**School:** Chicago (BookMap/Jigsaw) | **Class:** Visual Order Flow
**Timeframe:** Intraday (tick-based) | **Assets:** ES, NQ, CL futures

**Mathematics:**
```
At each price level, track in real-time:
  Resting_Bids = limit buy orders on the book
  Resting_Asks = limit sell orders on the book
  Executed_At_Bid = market sell orders hitting bids
  Executed_At_Ask = market buy orders hitting asks

Iceberg Detection:
  If executions at a price >> visible resting orders:
    Hidden liquidity (iceberg order) = institutional presence
  Iceberg_Score = Executed_Volume / Max_Visible_Volume at that level
  If Iceberg_Score > 5: likely iceberg (5x more traded than was visible)

Absorption:
  At support: Resting_Bids consumed but price NOT dropping
    Bid_Refresh_Rate = New_Bids_Added / Bids_Consumed
    If Bid_Refresh_Rate > 0.8: Active defense (institutional buyer absorbing selling)

Spoofing Detection:
  Large resting order that cancels when price approaches
  Cancel_Rate = Cancellations / (Cancellations + Executions) at a level
  If Cancel_Rate > 0.95: spoofing (fake order, ignore this level)
```

**Signal:**
- **Long:** Iceberg detected at support (Iceberg_Score > 5 on bid side) + Bid_Refresh_Rate > 0.8
- **Short:** Iceberg at resistance (Iceberg_Score > 5 on ask side) + Ask_Refresh_Rate > 0.8
- **Avoid:** Cancel_Rate > 0.90 at key level (spoofing, level is fake)
- **Exit:** Iceberg exhausts (no more refreshing) or price breaks through absorbed level

**Risk:** Tight stops (2-3 ticks); Scalp-style targets; Ultra-low latency required
**Edge:** The order book heatmap reveals the most granular level of market information: WHERE institutional limit orders are hiding (icebergs), where they are actively defending levels (absorption with refresh), and where fake orders are creating false signals (spoofing). This is the closest thing to "seeing" institutional intent in real-time. Requires specialized software (BookMap, Jigsaw) and significant screen time to develop pattern recognition.

---

### 187 | Intraday Cumulative Tick Index Strategy
**School:** New York (NYSE Tick) | **Class:** Market Internals
**Timeframe:** Intraday | **Assets:** SPY, QQQ, ES, NQ

**Mathematics:**
```
NYSE TICK:
  TICK = (# stocks upticking) - (# stocks downticking) at any instant
  Range: typically -1000 to +1000
  
  TICK > +800: Extreme buying (nearly all stocks upticking simultaneously)
  TICK < -800: Extreme selling (nearly all stocks downticking)

Cumulative TICK (from session open):
  Cum_TICK = cumsum(TICK readings at 1-second intervals)
  Cum_TICK_slope = linreg_slope(Cum_TICK, last 30 minutes)

  Rising Cum_TICK: sustained institutional buying program (bullish intraday)
  Falling Cum_TICK: sustained institutional selling program (bearish intraday)

TICK Extreme Counts:
  Extreme_Buy_Count = count(TICK > +800, last 30 min)
  Extreme_Sell_Count = count(TICK < -800, last 30 min)
  
  If Extreme_Buy_Count > 5 in 30 min: Institutional buying wave
  If Extreme_Sell_Count > 5 in 30 min: Institutional selling wave
```

**Signal:**
- **Long:** Cum_TICK_slope positive AND Extreme_Buy_Count > 3 AND SPY above VWAP
- **Short:** Cum_TICK_slope negative AND Extreme_Sell_Count > 3 AND SPY below VWAP
- **Exit:** Cum_TICK_slope reverses OR TICK normalizes (no extremes for 30+ min)
- **Avoid:** Cum_TICK flat AND no TICK extremes (directionless, no institutional program)

**Risk:** Intraday only; 1.5x ATR stops; Close by 15:45 ET; Risk 0.5%
**Edge:** NYSE TICK measures the SYNCHRONIZATION of buying/selling across thousands of stocks simultaneously. When TICK hits extremes (800+), it means nearly every stock is moving in the same direction at the same moment -- this only happens when institutional programs are executing across the market. Cumulative TICK reveals whether these programs are sustained (trending) or episodic (mean-reverting). It is the most reliable intraday institutional activity indicator.

---

### 188 | Volume-at-Price Support/Resistance
**School:** TradingView/Market Profile | **Class:** Volume Structure
**Timeframe:** Daily (multi-day aggregation) | **Assets:** All markets

**Mathematics:**
```
Volume at Price (VAP):
  Aggregate all volume by price level over N days (typically 20-60 days)
  
  High Volume Node (HVN): Price level with locally maximum volume
    = fair value, strong support/resistance (price attracted here)
  
  Low Volume Node (LVN): Price level with locally minimum volume
    = rejection zone (price moves quickly through these levels)

  Composite Profile = aggregation of daily profiles over N days

HVN/LVN Detection:
  For each price_level:
    Vol(level) = total volume at this 0.1% price bucket over N days
    Is_HVN = Vol(level) > 1.5 * median(Vol_all_levels) AND local_max
    Is_LVN = Vol(level) < 0.5 * median(Vol_all_levels) AND local_min

Trading from HVN/LVN:
  Price approaching HVN from above: expect support (deceleration, buyers defend)
  Price approaching HVN from below: expect resistance (sellers defend)
  Price entering LVN: expect acceleration (no volume = no resistance to movement)
```

**Signal:**
- **Buy at HVN support:** Price tests HVN from above with decelerating momentum -> buy
- **Sell at HVN resistance:** Price tests HVN from below with decelerating momentum -> sell
- **Breakout through LVN:** After HVN break, price enters LVN -> expect fast move to next HVN
- **Target:** Next HVN in trend direction

**Risk:** Stop just beyond HVN (if HVN breaks, thesis is wrong); Risk 1%
**Edge:** Volume at Price reveals the price levels where the most trading has occurred historically. These levels represent consensus on "fair value" and create genuine support/resistance because institutions who traded there have positions to defend. Low-volume nodes are "air pockets" where price accelerates because there is no historical volume to provide friction. This volume-structural approach to S/R is more objective and reliable than price-pattern-based S/R.

---

### 189 | Hong Kong Connect Flow Signal
**School:** Hong Kong (HKEX Connect) | **Class:** Cross-Border Flow
**Timeframe:** Daily | **Assets:** HK-listed stocks, A-shares

**Mathematics:**
```
Stock Connect (Shanghai/Shenzhen-Hong Kong):
  Northbound Flow = mainland investors buying HK stocks
  Southbound Flow = HK/global investors buying A-shares
  
  Net_Flow = Northbound - Southbound (daily, in HKD/RMB millions)

Flow Signal:
  NB_5d = SMA(Northbound_Flow, 5)
  NB_Z = normalize(Net_Flow, 60)

  NB_Z > +2.0: Extreme northbound (mainland buying HK aggressively)
  NB_Z < -2.0: Extreme southbound (global buying A-shares aggressively)

Individual Stock Flow:
  For each stock in Connect list:
    Stock_NB_pct = Northbound_Volume / Total_Volume * 100
    Stock_NB_change = Stock_NB_pct - SMA(Stock_NB_pct, 20)
    
  If Stock_NB_change > +5%: Mainland institutions accumulating this stock
  If Stock_NB_change < -5%: Mainland institutions distributing

Sector Flow Rotation:
  Track NB/SB flow by sector -> identify sector rotation by mainland/global investors
```

**Signal:**
- **Buy HK stock:** Individual stock northbound flow surging (NB_change > +5%) AND HSI trend up
- **Buy A-shares:** NB_Z < -2.0 (global investors buying A-shares = smart money bullish on China)
- **Sell signal:** Stock_NB_change < -5% for 5+ consecutive days (sustained distribution)
- **Sector rotation:** Follow NB flow into sectors = mainland policy trade (often leads by 2-4 weeks)

**Risk:** FX risk (RMB/HKD); Position size by flow magnitude; Stop at flow reversal; Risk 1.5%
**Edge:** Stock Connect flow is the most direct observable measure of cross-border institutional conviction. Mainland investors buying Hong Kong stocks (northbound) often reflects PBOC/state fund policy signals that precede market-wide moves. The flow data is available daily with a 1-day lag. Tracking individual stock Connect flow identifies institutional accumulation/distribution before it shows in price, providing 2-4 weeks of lead time.

---

### 190 | Bid-Ask Spread Regime Strategy
**School:** London (Market Microstructure) | **Class:** Spread Analytics
**Timeframe:** Intraday | **Assets:** All liquid markets

**Mathematics:**
```
Effective Spread:
  ES = 2 * |Trade_Price - Midpoint| / Midpoint * 10000  (in basis points)

Spread Regime:
  ES_EMA = EMA(ES, 100_trades)
  ES_Z = (ES - ES_EMA) / StdDev(ES, 500_trades)
  
  Tight Spread (ES_Z < -1.0): High liquidity, low risk of adverse selection
  Normal Spread: Typical liquidity conditions
  Wide Spread (ES_Z > +1.0): Low liquidity, high adverse selection risk
  Extreme Spread (ES_Z > +2.5): Liquidity crisis at this price

Spread-Volume Relationship:
  If ES widening AND Volume increasing: Information event (news, insider)
  If ES widening AND Volume decreasing: Liquidity withdrawal (market makers pulling quotes)
  If ES tightening AND Volume increasing: Good liquidity (market makers competing)

Trading Implication:
  Tight spread = cheaper to trade = use tighter stops, more frequent trading
  Wide spread = expensive to trade = widen stops, reduce frequency
  Extreme spread = do not trade (cost > expected alpha)
```

**Signal:**
- **Trade freely:** ES_Z < 0 (good liquidity, execution costs manageable)
- **Reduce activity:** ES_Z > +1.0 (liquidity deteriorating, widen stops)
- **Do not trade:** ES_Z > +2.5 (liquidity crisis, execution costs exceed expected returns)
- **Liquidity restoration:** ES_Z was > +2 and drops below +0.5 = liquidity returning, resume trading

**Risk:** Size inversely with spread (wider spread = smaller position); Add spread cost to stop distances
**Edge:** Most retail traders ignore execution costs entirely, but the bid-ask spread is a direct tax on every trade. In normal conditions it's negligible, but during stress events, spreads can widen 10-50x, making profitable strategies unprofitable. The spread regime framework dynamically adjusts trading activity and position sizing based on real-time liquidity conditions. Professional market makers use this exact framework to decide when to provide liquidity (tight spreads) and when to withdraw (wide spreads).

---

### 191 | Put/Call Open Interest Imbalance
**School:** Chicago (Options Market Analysis) | **Class:** Sentiment from OI
**Timeframe:** Daily | **Assets:** Individual stock options

**Mathematics:**
```
Open Interest Analysis:
  Put_OI = total put open interest across all strikes and expirations
  Call_OI = total call open interest
  
  PCR_OI = Put_OI / Call_OI  (Put-Call Ratio by Open Interest)
  PCR_OI_Z = normalize(PCR_OI, 120)

OI-Weighted Delta:
  For each strike and expiry:
    OI_Delta = OI * Delta_of_option * contract_multiplier
  
  Net_OI_Delta = sum(Call_OI_Delta) + sum(Put_OI_Delta)
  (market-wide directional positioning from options)

  Net_OI_Delta > 0: Market positioned for upside (net long delta from options)
  Net_OI_Delta < 0: Market positioned for downside (net short delta from options)

Gamma Exposure (GEX):
  GEX = sum(OI * Gamma * 100 * S^2 / 100)
  For each strike: dealer is long/short gamma depending on OI holder type
  
  Positive GEX: Dealers long gamma (will buy dips, sell rallies = stabilizing)
  Negative GEX: Dealers short gamma (will sell dips, buy rallies = destabilizing)
```

**Signal:**
- **Bullish:** PCR_OI_Z > +2.0 (extreme put buying = contrarian buy signal)
- **Bearish:** PCR_OI_Z < -1.5 (extreme call buying = contrarian sell signal)
- **Gamma pin:** High positive GEX at a specific strike = price gravitates to that strike at expiry
- **Vol breakout:** Negative GEX + market moving = destabilizing feedback loop, trade momentum

**Risk:** OI signals work on 5-20 day horizon; Stop at 3% adverse move; Risk 1%
**Edge:** Options open interest reveals how the market is POSITIONED, not just how it's priced. Extreme put OI relative to calls (high PCR_OI) is a contrarian indicator: when everyone has already bought puts, the downside is hedged and the market tends to rally. GEX determines whether market makers will STABILIZE or DESTABILIZE price moves -- the most important short-term structural factor in modern markets.

---

### 192 | Dubai Liquidity Concentration Index
**School:** Dubai (DGCX/DFSA Market Analysis) | **Class:** EM Liquidity
**Timeframe:** Daily / Weekly | **Assets:** GCC equities, DGCX metals

**Mathematics:**
```
Liquidity Concentration Index (LCI):
  For a market with N stocks:
    LCI = HHI of volume = sum((Volume_i / Total_Volume)^2)
    Range: [1/N, 1]
    
    LCI near 1/N: volume evenly distributed (healthy market)
    LCI near 1: volume concentrated in few stocks (unhealthy)

GCC/Dubai Specific:
  Normal LCI for Dubai DFM: 0.05-0.10 (moderately concentrated)
  Pre-rally LCI: < 0.05 (breadth expanding, more stocks participating)
  Pre-crash LCI: > 0.15 (volume concentrating in few names = fragility)

Top-5 Concentration:
  Top5_share = sum(top_5_stocks_volume) / total_volume
  If Top5_share > 70%: extreme concentration (market dependent on 5 stocks)
  If Top5_share < 40%: healthy breadth

Volume Rotation Signal:
  Track weekly changes in individual stock volume share
  If new stocks entering top-10 volume: rotation (bullish for market breadth)
  If same stocks dominating for 4+ weeks: stagnation (bearish)
```

**Signal:**
- **Bullish (breadth expanding):** LCI declining AND Top5_share declining (more stocks participating)
- **Bearish (fragility):** LCI > 0.15 AND Top5_share > 70% (volume too concentrated)
- **Rotation opportunity:** New sector stocks entering volume top-10 = follow the rotation
- **Exit:** LCI rising for 3+ weeks (breadth contracting)

**Risk:** EM liquidity risk; Wide stops (3x ATR); Smaller position sizes; Risk 1%
**Edge:** GCC markets are retail-dominated with low foreign institutional ownership, creating extreme volume concentration patterns. When LCI rises (concentration increasing), it signals that retail is chasing a few names -- classic bubble formation. When LCI falls (breadth expanding), it signals new institutional interest across the market. In Dubai specifically, LCI < 0.05 preceded every major DFM rally in the last 15 years.

---

### 193 | Market Maker Inventory Model
**School:** Academic (Amihud-Mendelson, Garman) | **Class:** Inventory Risk
**Timeframe:** Intraday | **Assets:** Large-cap equities, futures

**Mathematics:**
```
Market Maker Inventory:
  MM accumulates inventory through providing liquidity:
    If buying demand exceeds selling: MM accumulates long inventory
    If selling exceeds buying: MM accumulates short inventory

  Estimated MM Inventory:
    Inv_t = Inv_{t-1} + (Sell_to_MM - Buy_from_MM)
    (estimated from order flow imbalance)

  MM Optimal Behavior:
    When Inv > target: MM LOWERS ask price to attract selling (unload inventory)
      -> price likely to DECLINE short-term
    When Inv < target: MM RAISES bid price to attract buying (build inventory)
      -> price likely to RISE short-term

Observable Proxy for MM Inventory:
  Cum_Delta = cumulative(Ask_Volume - Bid_Volume) from session start
  If Cum_Delta >> 0: MM has absorbed buying -> now has short inventory -> will sell
  If Cum_Delta << 0: MM has absorbed selling -> now has long inventory -> will buy

Inventory Mean Reversion:
  MM_Reversion = -sign(Cum_Delta) * |Cum_Delta_Z|
  Positive: expect price increase (MM will bid up to buy)
  Negative: expect price decrease (MM will offer down to sell)
```

**Signal:**
- **Buy (MM needs to buy):** Cum_Delta_Z < -2.0 (MM heavily long inventory, will support price)
- **Sell (MM needs to sell):** Cum_Delta_Z > +2.0 (MM heavily short, will press price)
- **Exit:** Cum_Delta returns to neutral
- **Timeframe:** 15-60 minute horizon (MM inventory adjusts within hours)

**Risk:** Tight stops (MM-driven moves are short-lived); 1x ATR stop; Risk 0.3%
**Edge:** Market makers must manage inventory risk: when their inventory becomes imbalanced, they MUST adjust prices to attract offsetting flow. This creates predictable short-term price pressures that can be traded systematically. The key insight is that aggressive buying (which pushes Cum_Delta positive) actually creates future SELLING pressure as the MM works to offload their accumulated short inventory.

---

### 194 | Volume-Adjusted Bollinger Band Width
**School:** Quantitative | **Class:** Volume-Vol Hybrid
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Standard BB Width:
  BBW = (Upper_BB - Lower_BB) / SMA(Close, 20)
  = 4 * StdDev(Close, 20) / SMA(Close, 20) * 100  (percentage width)

Volume-Adjusted BB Width:
  VA_BBW = BBW * sqrt(RVOL)  where RVOL = Volume / SMA(Volume, 50)
  
  If RVOL > 1 (above avg volume): VA_BBW > BBW (wider adjusted bands)
  If RVOL < 1 (below avg volume): VA_BBW < BBW (narrower adjusted bands)

  Rationale: High-volume price moves are more meaningful than low-volume moves
  Adjusting BB Width by volume gives a truer picture of volatility COMMITMENT

VA_BBW Squeeze:
  VA_BBW_pctile = percentile_rank(VA_BBW, 252)
  Squeeze = VA_BBW_pctile < 10 (volume-adjusted bands at yearly tights)
  
  Volume-Adjusted Squeeze is RARER and MORE RELIABLE than standard BB Squeeze
  Because it requires BOTH low price volatility AND low volume commitment
```

**Signal:**
- **Pre-breakout setup:** VA_BBW_pctile < 10 (volume-adjusted squeeze)
- **Breakout confirmed:** VA_BBW expands from squeeze + RVOL > 1.5 (vol+volume expanding)
- **Direction:** First close outside VA_BBW after squeeze = breakout direction
- **Exit:** VA_BBW_pctile returns to 50th percentile (expansion complete)

**Risk:** Stop inside the squeeze range; Target at 2x squeeze range; Risk 1.5%
**Edge:** Standard BB Width Squeeze ignores volume context -- a narrowing BB during low volume is less significant than during normal volume. Volume-adjusted BB Width captures TRUE compression (both price AND participation contracting), which is a more reliable breakout precursor. The VA_BBW Squeeze occurs ~40% less frequently than standard BB Squeeze but has a ~75% breakout success rate (vs. ~55% for standard).

---

### 195 | Equity Put/Call Volume Ratio Extremes
**School:** Chicago (CBOE Sentiment) | **Class:** Options Sentiment
**Timeframe:** Daily | **Assets:** US market (SPX, individual equities)

**Mathematics:**
```
Equity Put/Call Volume Ratio:
  EPCR = equity_put_volume / equity_call_volume
  (excludes index options, which are used for hedging and distort the signal)

EPCR Levels:
  Historical mean: ~0.60 (more calls than puts, structural optimism)
  EPCR > 0.80: Elevated fear (retail buying puts aggressively)
  EPCR > 1.00: Extreme fear (more puts than calls = rare)
  EPCR < 0.40: Extreme complacency (all calls, no hedging)

EPCR Smoothing:
  EPCR_5d = SMA(EPCR, 5)  (reduce noise)
  EPCR_Z = normalize(EPCR_5d, 252)

Contrarian Framework:
  EPCR_Z > +2.0: Extreme fear -> contrarian BUY (too many puts, market oversold)
  EPCR_Z < -2.0: Extreme greed -> contrarian SELL (too many calls, market overbought)
  
Historical hit rate:
  EPCR_Z > +2: positive 20-day returns 75% of the time (avg +2.8%)
  EPCR_Z < -2: negative 20-day returns 60% of the time (avg -1.2%)
```

**Signal:**
- **Buy (contrarian):** EPCR_Z > +2.0 (extreme fear in equity options)
- **Sell (contrarian):** EPCR_Z < -2.0 (extreme complacency)
- **Hold existing:** -1.0 < EPCR_Z < +1.0 (neutral sentiment, no signal)
- **Exit:** EPCR_Z returns to 0 (sentiment normalizes)

**Risk:** Market-level indicator; SPY/QQQ positions; Stop at further extreme; Risk 2%
**Edge:** Equity PCR (excluding index) captures RETAIL sentiment because retail investors are the primary equity option buyers. Extreme retail put-buying historically marks market bottoms because: (1) retail tends to buy puts late (after the decline), (2) market makers who sold those puts hedge by buying stock (creates buying pressure), (3) the hedging demand provides a mechanical floor. The 75% hit rate at EPCR_Z > +2 is one of the best sentiment signals available.

---

### 196 | Singapore STI Turnover Velocity
**School:** Singapore (SGX Market Analysis) | **Class:** Turnover Momentum
**Timeframe:** Daily | **Assets:** STI component stocks

**Mathematics:**
```
Turnover Velocity:
  TV = Daily_Volume * Close / Free_Float_Market_Cap * 100  (daily turnover %)
  TV_EMA = EMA(TV, 20)

  For STI stocks:
    Normal TV: 0.1-0.3% daily
    Elevated: > 0.5% (significant institutional activity)
    Extreme: > 1.0% (repositioning event, proxy statement, M&A speculation)

Market-Wide Turnover:
  MW_TV = sum(all_volume * all_close) / Total_Market_Cap * 100
  MW_TV_Z = normalize(MW_TV, 252)

Turnover Acceleration:
  TV_accel = TV_EMA - TV_EMA_5d_ago
  If TV_accel > 0 AND Price_trend > 0: confirmed uptrend with rising participation
  If TV_accel > 0 AND Price_trend < 0: capitulation selling (bottom approaching)
  If TV_accel < 0 AND Price_trend > 0: rising on thin turnover (exhaustion)

REIT-Specific (Singapore's REIT market is largest in Asia ex-Japan):
  REIT_TV vs Market_TV: If REIT_TV rising while Market_TV flat = yield seeking
```

**Signal:**
- **Long:** TV_accel positive AND Price_trend positive (rising participation in uptrend)
- **Short/avoid:** TV_accel negative AND Price_trend positive (thin participation in rally)
- **Bottom signal:** TV_accel positive AND Price_trend negative AND TV > 2x normal (capitulation)
- **REIT rotation:** REIT_TV relative to Market_TV rising = defensive positioning

**Risk:** Singapore liquidity premium (wider stops); 2.5x ATR stops; Risk 1%
**Edge:** Turnover velocity normalizes volume by market cap, providing a true measure of how actively a stock is being traded relative to its size. In Singapore's concentrated market (top 10 stocks = 50% of STI), turnover velocity reveals institutional repositioning before price moves. The REIT turnover signal is unique to Singapore, where REITs are 40%+ of the market and their turnover velocity reflects global yield-seeking flows.

---

### 197 | Options Max Pain and Gamma Exposure
**School:** Chicago (Options Market Structure) | **Class:** Structural Price Magnet
**Timeframe:** Weekly (approaching OpEx) | **Assets:** SPX, single stock options

**Mathematics:**
```
Max Pain:
  For each possible closing price P:
    Total_Pain(P) = sum(Put_OI(K) * max(K-P, 0) + Call_OI(K) * max(P-K, 0)) for all K
  Max_Pain_Price = argmin(Total_Pain(P))
  = price where option holders lose the most money (and MM profits most)

Gamma Exposure (GEX) by Strike:
  GEX(K) = OI(K) * Gamma(K, S, T, sigma) * 100 * S
  
  Total_GEX = sum(call_GEX) - sum(put_GEX) (net dealer gamma)
  
  Positive GEX: Dealers are LONG gamma
    -> They buy when price drops, sell when price rises = price stabilizer
    -> Market tends to pin around high-GEX strikes
    
  Negative GEX: Dealers are SHORT gamma
    -> They sell when price drops, buy when price rises = price amplifier
    -> Explosive moves, breakouts, whipsaws

GEX Flip Level:
  Price where GEX transitions from positive to negative
  Above flip: stable (mean-reverting) market
  Below flip: unstable (trending/crashing) market
```

**Signal:**
- **Approaching OpEx:** Price gravitates toward max pain (trade toward max pain)
- **Positive GEX:** Mean-reversion strategies optimal (fade moves to high-GEX strikes)
- **Negative GEX:** Momentum strategies optimal (trade breakouts, trends amplify)
- **GEX flip break:** Price breaks below GEX flip -> expect acceleration (destabilizing regime)

**Risk:** OpEx dynamics strongest last 3 days; Reduce size 5+ days from OpEx; Stop beyond max pain range
**Edge:** Max pain and GEX are STRUCTURAL forces, not predictions. They arise from the mechanical hedging behavior of dealers who are short options: when GEX is positive, dealer hedging stabilizes price (they buy dips, sell rallies). When negative, they amplify moves. This framework explains WHY markets sometimes pin to levels (positive GEX) and sometimes crash (negative GEX). Understanding this structural dynamic provides an edge available only through options market analysis.

---

### 198 | Volume Spike Reversion Trading
**School:** Quantitative | **Class:** Volume Mean Reversion
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Volume Spike:
  Vol_Z = (Volume - SMA(Volume, 50)) / StdDev(Volume, 50)
  Spike = Vol_Z > 3.0 (volume 3+ sigma above average)

Spike Direction:
  Up_Spike = Spike AND Close > Open (high-volume buying day)
  Down_Spike = Spike AND Close < Open (high-volume selling day)

Reversion Statistics:
  After Up_Spike:
    Next 1 day: +0.1% avg (continuation, modest)
    Next 5 days: -0.4% avg (REVERSION, significant)
    Next 10 days: -0.2% avg (reversion fading)
    
  After Down_Spike:
    Next 1 day: +0.2% avg (bounce)
    Next 5 days: +0.8% avg (REVERSION, significant)
    Next 10 days: +0.5% avg (reversion sustained)

Compound Spike:
  Two volume spikes within 5 days = compound spike
  Down compound spike -> next 10 days: +1.5% avg (strongest reversion signal)
```

**Signal:**
- **Buy (5-day hold):** Down_Spike detected (sell-off on extreme volume = selling exhausted)
- **Sell (5-day hold):** Up_Spike detected (buying climax = buying exhausted)
- **Strongest buy:** Compound down spike (two capitulation events = deep exhaustion)
- **Exit:** 5-day time stop (reversion effect concentrated in 1-5 day window)

**Risk:** 5-day hold period; Stop at 2% adverse; Fixed time horizon exit; Risk 1%
**Edge:** Volume spikes represent capitulation events where the marginal participant has exhausted their capacity to trade. After a selling capitulation (down spike), there are literally no more sellers left, so price must recover. This is quantified by the 5-day reversion statistics. The compound spike (two events) is the strongest signal because it represents a double-test of selling capacity where sellers are confirmed to be exhausted.

---

### 199 | London FX Session Volume Dynamics
**School:** London (FX Desks, 3:00-12:00 ET) | **Class:** Session Volume
**Timeframe:** Intraday (FX sessions) | **Assets:** G10 FX pairs

**Mathematics:**
```
FX Session Volume Proxy (tick volume or EBS/Reuters volumes):
  Asian Session: 18:00-03:00 ET (typically lowest volume, ~15% of daily)
  London Session: 03:00-12:00 ET (highest volume, ~43% of daily)
  NY Session: 08:00-17:00 ET (~35% of daily)
  Overlap (London+NY): 08:00-12:00 ET (PEAK volume, ~30% of daily in 4 hours)

London Open Dynamics:
  First_Hour_Vol = tick_volume(03:00-04:00 ET)
  Avg_First_Hour = SMA(First_Hour_Vol, 20)
  London_RVOL = First_Hour_Vol / Avg_First_Hour
  
  London_RVOL > 2.0: Significant institutional repositioning at London open
  London_RVOL < 0.5: Quiet open (range day likely)

Session Range Projection:
  Typical_London_Range = average London session range over 20 days
  If 50% of London range achieved by 06:00 ET (3 hours):
    Likely trend day (extend range)
  If 80% of London range achieved by 06:00 ET:
    Likely exhaustion (mean-revert remainder of session)
```

**Signal:**
- **London trend:** London_RVOL > 2.0 AND 50% of avg range by 06:00 = continue in direction
- **London fade:** 80% of avg range by 06:00 = fade the overextension
- **Overlap power:** If direction established by London confirmed at NY open (08:00) with volume = strong continuation
- **Exit:** End of London session (12:00 ET) or NY reversal signal

**Risk:** Session-based (close by session end); 1x ATR stop; Risk 0.5%
**Edge:** London session commands 43% of daily FX volume and sets the directional tone for the day. The first-hour volume at London open reveals institutional conviction: high RVOL means banks are repositioning (likely in response to overnight Asian developments or early European data). The session range projection methodology provides early identification of trend vs. range days, allowing appropriate strategy selection by 06:00 ET.

---

### 200 | Order Flow Toxicity (VPIN) Regime Filter
**School:** Academic/Quantitative | **Class:** Flow Toxicity
**Timeframe:** Volume-based | **Assets:** All liquid markets

**Mathematics:**
```
Order Flow Toxicity combines multiple microstructure metrics:

  1. VPIN (Strategy 161): Probability of informed trading
  2. Kyle's Lambda: Price impact per unit of order flow
     Lambda = |delta_P| / |Order_Imbalance|
     High Lambda: large price impact per trade = illiquid/toxic
  
  3. Amihud Illiquidity:
     ILLIQ = |ret| / (Volume * Close)  (price impact per dollar traded)
     ILLIQ_Z = normalize(ILLIQ, 120)

Composite Toxicity Score:
  Toxicity = 0.40 * VPIN_pctile + 0.30 * Lambda_pctile + 0.30 * ILLIQ_pctile
  Range: [0, 100]
  
  Toxicity < 30: Safe (low information asymmetry, good liquidity)
  30-60: Normal
  60-80: Elevated (increase caution, widen stops)
  > 80: Toxic (reduce or eliminate positions)

Toxicity as Alpha Signal:
  Toxicity spikes often PRECEDE significant price moves (1-5 day lead)
  Direction: usually the direction of the informed flow (which caused toxicity)
```

**Signal:**
- **Risk management:** Toxicity > 60 -> reduce position by Toxicity/100 * position
- **Full risk-off:** Toxicity > 80 -> flatten all positions
- **Alpha signal:** Toxicity spike from <30 to >60 in 2 days = informed traders active -> follow their direction
- **Resume trading:** Toxicity returns below 40

**Risk:** Toxicity-scaled position sizing; No new trades when Toxicity > 70; Wider stops at elevated toxicity
**Edge:** Order flow toxicity combines the best microstructure metrics into a single risk management score. It answers the most important question in trading: "Is the person on the other side of my trade better informed than me?" When toxicity is high, the answer is probably yes, and you should reduce exposure. The alpha signal (toxicity spike) identifies the arrival of informed flow before it moves the market, providing a 1-5 day lead time on significant price movements.

---

# SECTION V: PATTERN RECOGNITION AND FRACTAL STRATEGIES (201-250)

---

"""

with open("Indicators.md", "a") as f:
    f.write(content)

print("Appended strategies 151-200 to Indicators.md")
