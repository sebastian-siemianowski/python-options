#!/usr/bin/env python3
"""Append strategies 201-250 to Indicators.md"""

content = r"""
### 201 | Fractal Dimension Index (FDI) Regime Filter
**School:** Academic (Mandelbrot, Fractal Geometry) | **Class:** Fractal Analysis
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Fractal Dimension:
  Using box-counting method on price series:
  D = -log(N(epsilon)) / log(epsilon)
  where N(epsilon) = number of boxes of size epsilon needed to cover the price path

  Practical: Variation method (Hurst exponent)
    H = Hurst exponent estimated via rescaled range
    D = 2 - H  (fractal dimension from Hurst)

  D = 1.0: Perfect trend (straight line)
  D = 1.5: Random walk (Brownian motion)
  D = 2.0: Perfectly mean-reverting (space-filling)

FDI Regime:
  Rolling D over 30-day window:
  D < 1.3: Trending regime -> use trend-following strategies
  1.3 < D < 1.7: Random/ambiguous -> reduce position size
  D > 1.7: Mean-reverting regime -> use mean-reversion strategies

Regime Transition:
  D crossing 1.5 from above (was MR, now trending): trend entry
  D crossing 1.5 from below (was trending, now MR): fade entry
```

**Signal:**
- **Trend mode:** D < 1.3 -> apply momentum, trend-following, breakout strategies
- **MR mode:** D > 1.7 -> apply mean-reversion, range-bound strategies
- **No trade:** 1.3 < D < 1.7 (market is random, no structural edge)
- **Regime change:** D crossing 1.5 = switch strategy type

**Risk:** Meta-strategy filter (adjusts which strategies to use); Reduce size when 1.3<D<1.7
**Edge:** The fractal dimension directly measures whether a market is trending or mean-reverting -- the most fundamental regime classification for strategy selection. D < 1.3 means price has long memory (trends persist), D > 1.7 means short memory (moves reverse). Unlike simple trend indicators that react to past prices, FDI measures the STRUCTURAL properties of the price process. Most strategy failures come from applying trend strategies in MR regimes and vice versa.

---

### 202 | Three-Drive Harmonic Pattern
**School:** International (Scott Carney, Harmonic Trading) | **Class:** Fibonacci Structure
**Timeframe:** 4H / Daily | **Assets:** FX, Equities, Futures

**Mathematics:**
```
Three-Drive Pattern:
  Drive 1: Price makes initial swing from A to B
  Correction 1: B to C retraces 0.618 of AB
  Drive 2: C to D extends 1.272 of AB
  Correction 2: D to E retraces 0.618 of CD
  Drive 3: E to F extends 1.272 of CD (or 1.618 of AB)

Fibonacci Requirements:
  Each correction = 0.618 retracement (+/- 5% tolerance)
  Each drive = 1.272 extension (+/- 5% tolerance)
  Drives should be approximately symmetrical in time

  AB = first drive amplitude
  CD = second drive, should = AB * 1.272
  EF = third drive, should = CD * 1.272

Completion Zone:
  Pattern completes when third drive reaches 1.272-1.618 of first drive
  This is the reversal entry point

Pattern Scoring:
  Score = average(|actual_retrace - 0.618| / 0.618, |actual_extend - 1.272| / 1.272)
  Lower score = more precise Fibonacci alignment = higher probability
```

**Signal:**
- **Counter-trend entry:** At third drive completion zone (1.272-1.618 extension)
- **Direction:** Opposite to the three-drive direction (pattern is exhaustion structure)
- **Confirmation:** Reversal candle pattern at completion zone + RSI divergence
- **Exit:** First Fibonacci retracement of entire three-drive structure (0.382 or 0.618)

**Risk:** Stop beyond 1.618 extension; Target at 0.618 retracement of whole pattern; Risk 1%
**Edge:** Three-drive patterns represent natural market rhythms of impulse-correction-impulse. The Fibonacci ratios (0.618/1.272) appear because of the fractal nature of financial markets -- institutional order flow clusters at these mathematical levels, creating self-fulfilling support/resistance zones. When all three drives align with Fibonacci ratios within tight tolerance, the exhaustion probability at the completion zone exceeds 65%.

---

### 203 | Head and Shoulders Quantified Detection
**School:** New York (Classic Technical Analysis, Quantified) | **Class:** Reversal Pattern
**Timeframe:** Daily / Weekly | **Assets:** All markets

**Mathematics:**
```
Algorithmic Head & Shoulders Detection:

Step 1: Identify pivots using zigzag filter (minimum swing = 3% or 2*ATR)
Step 2: Pattern template matching:
  Left Shoulder (LS): local high H1
  Head (H): higher high H2 > H1
  Right Shoulder (RS): lower high H3 < H2, approximately = H1
  Neckline (NL): line connecting troughs between shoulders and head

Symmetry Requirement:
  |LS_height - RS_height| / H_height < 0.3 (shoulders within 30% of each other)
  |LS_duration - RS_duration| / H_duration < 0.5 (time symmetry)

Neckline Break:
  Close below neckline AND Volume > 1.3 * average = confirmed H&S top
  
Price Target:
  Target = Neckline - (Head_High - Neckline)  (head height projected downward)
  
  Historical reliability:
    H&S pattern reaches target: ~55-60% of the time
    Average target achieved: 83% of projected distance

Volume Pattern:
  LS volume > H volume > RS volume (declining volume through pattern = bearish)
  Neckline break volume > average = confirmation
```

**Signal:**
- **Short:** Neckline break on volume after confirmed H&S pattern (symmetry + volume profile)
- **Target:** Head height projected below neckline (measured move)
- **Failure:** Price closes back above neckline by 3% = pattern failure, stop out
- **Inverse H&S:** Same logic inverted for bottoming pattern (buy on neckline break above)

**Risk:** Stop above right shoulder (or 3% above neckline); Target is measured move; Risk 1.5%
**Edge:** H&S is the most studied reversal pattern in technical analysis. When quantified with strict symmetry requirements, volume confirmation, and neckline break confirmation, the pattern has ~60% reliability with favorable risk/reward (target typically 2-3x stop distance). The declining volume profile (LS > H > RS) is the critical quality filter -- it shows buying power diminishing through each successive attempt to push higher.

---

### 204 | Gartley 222 Harmonic Pattern
**School:** International (H.M. Gartley, 1935; refined by Larry Pesavento) | **Class:** Harmonic
**Timeframe:** 4H / Daily | **Assets:** All markets

**Mathematics:**
```
Gartley 222 Pattern (Bullish):
  X to A: Initial impulse move up
  A to B: Retracement of XA, must retrace 0.618 (+/- 5%)
  B to C: Retracement of AB, 0.382-0.886 of AB
  C to D: Final leg, must complete at 0.786 retracement of XA

  Critical: D point = 0.786 XA retracement = entry zone
  BC must be between 0.382 and 0.886 of AB
  CD must extend 1.272-1.618 of BC

PRZ (Potential Reversal Zone):
  Cluster of Fibonacci levels at D:
    0.786 XA retrace
    1.272-1.618 BC extension
    Ideally these converge within 1-2% price range

Pattern Validity Score:
  Each ratio deviation from ideal: penalty = |actual - ideal| / ideal
  Total_Penalty = sum(penalties)
  If Total_Penalty < 0.15: High-quality Gartley (tight ratios)
  If Total_Penalty > 0.30: Low-quality (loose ratios, skip)
```

**Signal:**
- **Buy (bullish Gartley):** Price reaches D point (0.786 XA) with reversal candle confirmation
- **Sell (bearish Gartley):** Inverted pattern, short at D completion
- **Target 1:** 0.382 retracement of AD (conservative)
- **Target 2:** 0.618 retracement of AD (aggressive)
- **Stop:** Below X point (invalidates entire pattern)

**Risk:** Stop below X; R:R typically 2:1 to 3:1; Risk 1%
**Edge:** The Gartley 222 is the foundational harmonic pattern, exploiting the Fibonacci structure of market swings. The Potential Reversal Zone (PRZ) creates a high-probability reversal area where multiple Fibonacci levels converge. Institutional algorithms programmed with these levels create self-reinforcing support/resistance zones. The key edge is the strict ratio requirements -- only high-quality patterns (Total_Penalty < 0.15) should be traded.

---

### 205 | Elliott Wave Automated Counting
**School:** New York (Ralph Nelson Elliott, 1930s) | **Class:** Wave Theory
**Timeframe:** Multi-timeframe | **Assets:** All markets

**Mathematics:**
```
Elliott Wave Structure:
  Impulse: 5 waves (1-2-3-4-5) in trend direction
  Correction: 3 waves (A-B-C) against trend

Wave Rules (inviolable):
  1. Wave 2 never retraces more than 100% of Wave 1
  2. Wave 3 is never the shortest impulse wave
  3. Wave 4 never enters Wave 1 price territory

Wave Guidelines (typical but not required):
  Wave 2 retraces 0.618-0.786 of Wave 1
  Wave 3 extends 1.618-2.618 of Wave 1 (longest wave)
  Wave 4 retraces 0.236-0.382 of Wave 3
  Wave 5 = Wave 1 in distance (equality) or 0.618/1.618 of Wave 1-3

Automated Wave Counting:
  For each set of 5 swing points:
    Check all 3 rules
    Score guideline compliance (0-100%)
    Select highest-scoring wave count
    Assign confidence: rules_met * guideline_score

Fibonacci Time Projections:
  Wave_3_end = Wave_1_end + 1.618 * Wave_1_duration (time)
  Wave_5_end = Wave_3_end + Wave_1_duration (equality in time)
```

**Signal:**
- **Buy at Wave 2 completion:** After impulsive Wave 1, buy 0.618-0.786 retracement (Wave 2 end)
- **Buy at Wave 4 completion:** After Wave 3 impulse, buy 0.236-0.382 retracement (Wave 4 end)
- **Short at Wave 5 completion:** After full 5-wave count, sell expecting ABC correction
- **Target:** Wave 3 target = Wave_1_end + 1.618 * Wave_1_length

**Risk:** Stop below Wave 1 start (for Wave 2 buy) or Wave 2 end (for Wave 4 buy); Risk 1-2%
**Edge:** Elliott Wave theory maps the natural psychological cycle of markets (optimism-doubt-euphoria-concern-denial). While subjective in manual application, automated counting with strict rule enforcement and guideline scoring removes the bias. The highest-value trades are Wave 3 entries (longest, most powerful impulse) from Wave 2 corrections, and Wave 5 exhaustion shorts. The three inviolable rules provide strict stop placement.

---

### 206 | Heikin-Ashi Trend Persistence Filter
**School:** Japanese (Modified Candlestick) | **Class:** Trend Smoothing
**Timeframe:** Daily / 4H | **Assets:** All markets

**Mathematics:**
```
Heikin-Ashi Candles:
  HA_Close = (Open + High + Low + Close) / 4
  HA_Open = (HA_Open_{t-1} + HA_Close_{t-1}) / 2
  HA_High = max(High, HA_Open, HA_Close)
  HA_Low = min(Low, HA_Open, HA_Close)

Trend Classification:
  Strong_Up: HA_Close > HA_Open AND no lower wick (HA_Low == HA_Open)
  Weak_Up: HA_Close > HA_Open AND has lower wick
  Doji: |HA_Close - HA_Open| < 0.1 * (HA_High - HA_Low) (indecision)
  Weak_Down: HA_Close < HA_Open AND has upper wick
  Strong_Down: HA_Close < HA_Open AND no upper wick (HA_High == HA_Open)

Trend Persistence Score:
  Count consecutive Strong_Up or Strong_Down candles
  Persistence(n) = count of same-direction strong candles in last n bars / n
  
  Persistence > 0.7 (over 10 bars): Strong trend (stay in trade)
  Persistence < 0.3: Trend exhausting (prepare to exit)

HA Color Change:
  After 5+ consecutive green HA bars, first red bar = potential reversal
  After 5+ consecutive red HA bars, first green bar = potential reversal
```

**Signal:**
- **Enter trend:** First Strong_Up after 3+ Weak/Doji bars (trend initiation)
- **Stay in trend:** Persistence > 0.5 (mostly strong candles)
- **Exit warning:** First opposite-color bar after 5+ streak = tighten stops
- **Exit:** Second opposite-color bar confirms trend change

**Risk:** Trail with HA candle extremes; Exit on color change; Risk 1%
**Edge:** Heikin-Ashi smooths price action by averaging with the prior bar, removing the noise that whipsaws standard candlestick traders. The "no lower wick" characteristic of strong uptrend bars is particularly powerful -- it means price never traded below the bar's open during that period, indicating uninterrupted buying pressure. The persistence score quantifies trend quality in a way that raw candlesticks cannot.

---

### 207 | Renko Brick Pattern Recognition
**School:** Japanese (Renko Charts) | **Class:** Noise Reduction
**Timeframe:** ATR-based bricks | **Assets:** All markets

**Mathematics:**
```
Renko Construction:
  Brick_Size = ATR(14)  (adaptive) or fixed (e.g., 10 points)
  
  New up-brick: when price moves Brick_Size above prior brick top
  New down-brick: when price moves Brick_Size below prior brick bottom
  No time axis -- purely price-based

Renko Patterns:
  Trend: 5+ consecutive same-color bricks
    Trend_Strength = count of consecutive bricks
  
  Reversal: Color change (green to red or red to green)
    After long streak: higher probability of trend change
    After short streak (1-2 bricks): likely false reversal
  
  Double-brick reversal: Requires 2x Brick_Size to reverse
    (more conservative, fewer false signals)

Renko Momentum:
  RM = sum(brick_direction, last_n_bricks) / n
  RM = +1: all up-bricks (maximum bullish momentum)
  RM = -1: all down-bricks (maximum bearish momentum)
  RM near 0: alternating (range-bound)

Brick Count Distribution:
  Average consecutive same-color bricks = expected trend length
  If current streak > 1.5x average: exhaustion risk
```

**Signal:**
- **Trend entry:** After 3 consecutive same-color bricks (trend confirmed)
- **Trend continuation:** RM > 0.6 (strong momentum in bricks)
- **Reversal alert:** Streak > 1.5x average streak length (exhaustion)
- **Exit:** Double-brick reversal (requires 2x brick size reversal)

**Risk:** Stop at prior brick reversal level; Target open-ended for trends; Risk 1%
**Edge:** Renko removes time from the analysis, focusing purely on price movement. By requiring a full brick_size move to create a new brick, ALL noise below the brick threshold is eliminated. This makes trend identification trivial (consecutive same-color bricks) and reversal identification clean (color change). The ATR-adaptive brick size automatically adjusts to current volatility, preventing the common Renko pitfall of fixed-size bricks being too large or too small.

---

### 208 | Fibonacci Cluster Zone Strategy
**School:** International (Multi-Source Fibonacci) | **Class:** Confluence
**Timeframe:** Multi-timeframe | **Assets:** All markets

**Mathematics:**
```
Fibonacci Sources (from multiple swings):
  For each significant swing (high/low pair):
    Generate retracement levels: 0.236, 0.382, 0.500, 0.618, 0.786
    Generate extension levels: 1.272, 1.618, 2.618

Cluster Detection:
  1. Take last 5-8 significant swings
  2. Generate all Fibonacci levels from each swing
  3. Create histogram of Fibonacci levels across price axis
  4. Cluster = price zone where 3+ independent Fibonacci levels converge within 0.5% price range

Cluster Strength:
  CS = number of converging Fibonacci levels in the cluster
  CS = 3: Moderate cluster (3 levels)
  CS = 5+: Strong cluster (5+ levels, high-probability reversal zone)
  CS = 7+: Extreme cluster (rare, very high probability)

Time Cluster (optional enhancement):
  Apply Fibonacci time ratios from multiple swings
  If price cluster AND time cluster align: strongest setup
```

**Signal:**
- **Reversal trade:** Price enters strong cluster zone (CS >= 5) + reversal candle = enter counter-trend
- **Breakout trade:** Price breaks through strong cluster -> expect acceleration to next cluster
- **Target:** Next cluster in trade direction
- **Exit:** At next strong cluster (support becomes target for longs)

**Risk:** Stop beyond cluster zone (1 ATR outside); Target at next cluster; Risk 1.5%
**Edge:** Individual Fibonacci levels are unreliable (any single retracement has ~50% chance of holding). But when multiple INDEPENDENT Fibonacci levels from different swings converge at the same price, the probability increases dramatically (3 levels: ~60%, 5+ levels: ~70%, 7+ levels: ~80%). This is because each Fibonacci level represents a different group of traders with cost bases at that mathematical relationship, creating genuine order flow concentration at cluster zones.

---

### 209 | Japanese Candlestick Ensemble Classifier
**School:** Tokyo (Steve Nison, Gregory Morris, Thomas Bulkowski) | **Class:** Pattern Ensemble
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Candlestick Pattern Library (40+ patterns):
  Reversal patterns: Hammer, Shooting Star, Engulfing, Harami, Morning/Evening Star,
                     Three White Soldiers, Three Black Crows, Abandoned Baby, etc.
  Continuation: Rising Three Methods, Falling Three Methods, Tasuki Gap, etc.

Pattern Detection Algorithm:
  For each bar, scan for ALL patterns and classify:
    Pattern_Direction: bullish (+1) or bearish (-1)
    Pattern_Reliability: from Bulkowski's statistical research [0, 1]
    Pattern_Context: with_trend (continuation) or against_trend (reversal)

Ensemble Score:
  ES = sum(Direction_i * Reliability_i * Context_Weight_i) for all detected patterns
  
  Context_Weight:
    Pattern aligns with SMA(50) trend: 1.5x
    Pattern opposes SMA(50) trend: 0.7x
    Pattern at support/resistance: 1.3x
    Pattern in middle of range: 0.8x

  ES > +1.5: Strong bullish ensemble signal
  ES < -1.5: Strong bearish ensemble signal

Bulkowski Top Patterns by Reliability:
  Bullish: Three White Soldiers (82%), Bullish Engulfing (63%), Morning Star (78%)
  Bearish: Three Black Crows (78%), Evening Star (72%), Bearish Engulfing (79%)
```

**Signal:**
- **Buy:** ES > +1.5 AND in uptrend or at support (multiple bullish patterns converging)
- **Sell:** ES < -1.5 AND in downtrend or at resistance
- **Ignore:** |ES| < 0.5 (weak or conflicting patterns)
- **Highest conviction:** ES > +2.0 with 3+ independent bullish patterns simultaneously

**Risk:** Stop below lowest candle in pattern cluster; Target 2x ATR; Risk 1%
**Edge:** Individual candlestick patterns have modest reliability (50-65%). But when MULTIPLE independent patterns trigger simultaneously (ensemble), the combined probability exceeds any single pattern. The context weighting (trend alignment, support/resistance) further improves accuracy. Bulkowski's database of 100,000+ pattern instances provides the statistical foundation. The ensemble approach is how Japanese rice traders originally used candles -- never one pattern in isolation.

---

### 210 | Wolfe Wave Projection
**School:** International (Bill Wolfe) | **Class:** Channel Projection
**Timeframe:** 4H / Daily | **Assets:** All markets

**Mathematics:**
```
Wolfe Wave Structure (Bullish):
  Point 1: Initial trough
  Point 2: Initial peak
  Point 3: Lower trough (below point 1)
  Point 4: Higher peak (above point 1, but below point 2)
  Point 5: Lower trough (breaks line 1-3, enters "sweet zone")

  Line 1-3: Lower trendline (support breaks at point 5)
  Line 2-4: Upper trendline (projected as the target line)
  Line 1-4: The EPA line (Estimated Price at Arrival = target projection)

Target: Price reaches line 1-4 after reversal from point 5
EPA (Estimated Price at Arrival):
  Project line connecting points 1 and 4 forward in time
  Price should reach this line within time equal to pattern duration

Sweet Zone at Point 5:
  Ideal: Point 5 is between 1.272 and 1.618 extension of line 1-3
  If Point 5 exactly at 1.272: high probability reversal
  If beyond 1.618: pattern may be failing
```

**Signal:**
- **Buy (bullish Wolfe):** At point 5, in the sweet zone (1.272-1.618 below line 1-3)
- **Sell (bearish Wolfe):** Inverted, at point 5 (1.272-1.618 above line 2-4)
- **Target:** Line 1-4 (EPA line) projected to expected arrival time
- **Failure:** Price does not reverse from sweet zone within 3 bars

**Risk:** Stop beyond 1.618 extension of pattern; Target at EPA line; Typically 3:1+ R:R
**Edge:** Wolfe Waves exploit the natural oscillatory behavior of price within converging channels. The mathematical elegance is that the TARGET (line 1-4) is determined by the pattern structure before the reversal occurs. Point 5 overshoots the lower channel (the "spring" effect), creating an entry below fair value. The time projection (EPA) adds a temporal dimension that most patterns lack.

---

### 211 | Linear Regression Channel Breakout
**School:** Quantitative | **Class:** Statistical Channel
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Linear Regression (n periods):
  y = a + b*x  where x = time index, y = close price
  b = slope (trend direction and speed)
  a = intercept

  Regression Line: predicted value at each time point
  Standard Error: SE = sqrt(sum(residuals^2) / (n-2))

Channel:
  Upper = Regression_Line + k * SE
  Lower = Regression_Line - k * SE
  k = 2 (2-sigma channel, contains ~95% of observations)

Regression Statistics:
  R^2 = coefficient of determination (trend quality)
    R^2 > 0.85: Strong linear trend (channel reliable)
    R^2 < 0.50: Weak trend (channel unreliable, do not trade)
  
  Slope_Z = b / SE(b)  (statistical significance of slope)
    |Slope_Z| > 2: Trend is statistically significant
    |Slope_Z| < 1: Trend is not significant (random)

Breakout Signal:
  Price closes outside 2-sigma channel AND R^2 > 0.70
  = statistically significant breakout from established trend
```

**Signal:**
- **Mean reversion (within channel):** Buy at lower channel, sell at upper channel IF R^2 > 0.85
- **Breakout:** Close beyond 2-sigma channel with R^2 > 0.70 = genuine trend change
- **Trend strength:** R^2 as position sizing factor (stronger trend = larger position)
- **Exit:** Price returns to regression line (mean reversion) or channel breaks (breakout trade)

**Risk:** Stop at channel center (MR) or opposite channel (breakout); Risk 1%
**Edge:** Linear regression channels provide statistically rigorous support/resistance, unlike subjective trendlines. The R^2 metric directly measures trend quality -- high R^2 means the trend is real (not noise), making channel trades reliable. The 2-sigma boundaries contain 95% of observations, so a breakout beyond them is a statistically significant event. This is the quantitative approach to what technicians do subjectively with trendlines.

---

### 212 | Cup and Handle Quantified Detection
**School:** New York (William O'Neil, CANSLIM) | **Class:** Continuation Pattern
**Timeframe:** Daily / Weekly | **Assets:** Growth Equities

**Mathematics:**
```
Cup Detection:
  1. Left lip: prior high (H_left)
  2. Bottom: at least 12% below left lip, typically 15-30%
  3. Right lip: price recovers to within 5% of left lip
  4. Duration: 7-65 weeks (7 week minimum for proper base)
  5. Shape: U-shaped (not V-shaped, no sharp bottom)

  U-shape test: midpoint_depth vs average_depth
    Ratio = midpoint_depth / avg_depth
    U-shape: Ratio 0.8-1.2 (even depth)
    V-shape: Ratio > 1.5 (too sharp at bottom)

Handle Detection:
  1. Forms in upper 1/3 of cup (after right lip formation)
  2. Pullback: 8-12% from right lip (smaller than cup depth)
  3. Duration: 1-5 weeks (shorter than cup)
  4. Volume: declining during handle formation (base building)

Breakout:
  Price breaks above handle high (lip level) on volume > 1.5x average
  
  Target: cup_depth projected above breakout level
  Historical success: 65-70% when all criteria met (O'Neil research)
```

**Signal:**
- **Buy:** Breakout above handle high on volume (all cup-handle criteria met)
- **Quality filter:** Cup depth 15-30%, handle depth 8-12%, duration 7-65 weeks, U-shape confirmed
- **Target:** Breakout level + cup depth (measured move)
- **Failure:** Close below handle low = pattern failure

**Risk:** Stop at handle low; Target at measured move; Typically 2:1+ R:R; Risk 1.5%
**Edge:** Cup and Handle is the premier growth stock continuation pattern, identified by O'Neil from studying every super-performing stock from 1880-2000. The U-shape requirement ensures accumulation was gradual (institutional, not speculative). Declining volume in the handle confirms the pullback is a pause, not a reversal. The measured move target (cup depth) has historical reliability of ~70% when strict criteria are applied.

---

### 213 | Ichimoku Cloud Multi-Signal System
**School:** Tokyo (Goichi Hosoda, 1960s) | **Class:** Complete Trading System
**Timeframe:** Daily | **Assets:** Equities, FX (originally Nikkei)

**Mathematics:**
```
Ichimoku Components:
  Tenkan-sen (Conversion): (highest_high(9) + lowest_low(9)) / 2
  Kijun-sen (Base): (highest_high(26) + lowest_low(26)) / 2
  Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
  Senkou Span B (Leading Span B): (highest_high(52) + lowest_low(52)) / 2, plotted 26 ahead
  Chikou Span (Lagging): Close plotted 26 periods back

Cloud (Kumo): Area between Senkou A and Senkou B (26 periods ahead)
  Bullish Cloud: Senkou A > Senkou B (green cloud)
  Bearish Cloud: Senkou A < Senkou B (red cloud)

Five Signal Components:
  1. Tenkan/Kijun Cross: bullish if Tenkan crosses above Kijun
  2. Price vs. Cloud: bullish if price above cloud
  3. Chikou vs. Price: bullish if Chikou above price from 26 bars ago
  4. Cloud Color: bullish if Senkou A > Senkou B
  5. Cloud Twist: Senkou A crossing Senkou B = regime change (26 periods ahead)

Composite Signal:
  Signal_Score = sum(each component: +1 for bullish, -1 for bearish)
  Range: [-5, +5]
  
  Score = +5: Maximum bullish (all 5 signals aligned)
  Score = -5: Maximum bearish
  |Score| < 2: Conflicting signals, do not trade
```

**Signal:**
- **Strong buy:** Score >= +4 (4+ of 5 signals bullish)
- **Strong sell:** Score <= -4
- **Cloud bounce:** Price touches cloud from above + Score >= +3 = buy the bounce
- **Kijun bounce:** Price touches Kijun from above in uptrend = buy opportunity
- **No trade:** |Score| <= 2

**Risk:** Stop below cloud (for longs) or Kijun; Target at cloud projection; Risk 1.5%
**Edge:** Ichimoku is a complete trading system that provides trend direction, support/resistance, and momentum in one indicator. The cloud is unique because it projects support/resistance 26 periods INTO THE FUTURE, providing forward-looking context that no other indicator offers. When all 5 components align (Score +5 or -5), the probability of continuation exceeds 75%. The system was specifically designed for the Nikkei's rhythmic patterns and works best in trending markets.

---

### 214 | Butterfly Harmonic Pattern
**School:** International (Scott Carney, 1999) | **Class:** Harmonic Extension
**Timeframe:** 4H / Daily | **Assets:** All markets

**Mathematics:**
```
Butterfly Pattern (Bullish):
  X to A: Initial impulse up
  A to B: 0.786 retracement of XA (key difference from Gartley)
  B to C: 0.382-0.886 retracement of AB
  C to D: 1.618-2.618 extension of XA (D is BELOW X)
  
  KEY: D point extends BEYOND X (unlike Gartley where D stays within XA)
  This makes Butterfly an EXTENSION pattern, reaching extreme oversold

CD Leg:
  CD = 1.618-2.618 extension of BC
  CD completion should align with 1.272-1.618 XA extension

Potential Reversal Zone (PRZ):
  Where XA extension AND BC extension converge
  Tightest PRZ = highest probability reversal

  AB must be 0.786 XA (+/- 5%)
  Not 0.618 (that's Gartley) or 0.886 (that's Bat)
  0.786 is the specific Butterfly identifier
```

**Signal:**
- **Buy at D completion:** When price reaches PRZ (1.272-1.618 XA extension) with reversal candle
- **Target 1:** 0.382 retracement of AD
- **Target 2:** 0.618 retracement of AD
- **Stop:** Below 2.618 XA extension (if D extends this far, pattern fails)

**Risk:** Stop at 2.618 extension; Target at 0.382-0.618 AD retrace; Risk 1%
**Edge:** The Butterfly is the deepest reversal pattern in harmonic trading. Because D extends beyond X, it catches the extended overreaction where most traders have been stopped out. The PRZ at the XA extension represents a mathematically-defined exhaustion point. The 0.786 AB retracement (specific to Butterfly) creates a specific structural rhythm that, when combined with the 1.272-1.618 XA extension, produces a high-probability reversal zone.

---

### 215 | Market Structure Break (CHoCH/BOS)
**School:** London (ICT/Smart Money Concepts) | **Class:** Structure Analysis
**Timeframe:** 15-min to Daily | **Assets:** All markets

**Mathematics:**
```
Market Structure:
  Higher Highs (HH) + Higher Lows (HL) = Uptrend
  Lower Highs (LH) + Lower Lows (LL) = Downtrend

Change of Character (CHoCH):
  In uptrend: first Lower Low (price breaks below prior HL)
  = potential trend reversal (first sign of weakness)
  
  In downtrend: first Higher High (price breaks above prior LH)
  = potential trend reversal (first sign of strength)

Break of Structure (BOS):
  In uptrend: price breaks above prior HH (trend continuation confirmed)
  In downtrend: price breaks below prior LL (trend continuation confirmed)

Swing Point Detection:
  Use zigzag filter with minimum swing = 1.5 * ATR
  Label each swing as HH, HL, LH, or LL based on sequence

Order Block at CHoCH:
  The last bullish candle before a bearish CHoCH = supply order block
  The last bearish candle before a bullish CHoCH = demand order block
  These are institutional entry zones that may be retested
```

**Signal:**
- **Buy (bullish CHoCH):** CHoCH from downtrend to uptrend -> buy at demand order block
- **Sell (bearish CHoCH):** CHoCH from uptrend to downtrend -> sell at supply order block
- **Continue trend:** BOS confirms -> add to position in trend direction
- **Exit:** Opposite CHoCH signal

**Risk:** Stop below order block (long) or above (short); Target at next structure level; Risk 1%
**Edge:** Market structure analysis reduces complex price action to a simple state machine: HH/HL (up) or LH/LL (down). CHoCH is the earliest possible signal of a trend reversal -- it occurs before a full reversal is confirmed, providing early entry. The order block concept identifies the specific price zone where institutional traders initiated the move that caused the CHoCH, making it a high-probability support/resistance zone for retests.

---

### 216 | AB=CD Measured Move Pattern
**School:** International (H.M. Gartley / Fibonacci) | **Class:** Measured Move
**Timeframe:** 4H / Daily | **Assets:** All markets

**Mathematics:**
```
AB=CD Pattern:
  AB: First impulse leg
  BC: Retracement of AB (0.382-0.886)
  CD: Second impulse leg, where CD = AB in price AND/OR time

Perfect AB=CD:
  CD_length / AB_length = 1.0 (equality)
  CD_time / AB_time = 1.0 (time symmetry)

Extended Variations:
  1.272 AB=CD: CD = 1.272 * AB (common when BC is deep, 0.618+)
  1.618 AB=CD: CD = 1.618 * AB (common when BC is shallow, 0.382)
  
Reciprocal Relationship:
  If BC = 0.382 AB: CD tends to extend to 2.24-2.618 BC
  If BC = 0.618 AB: CD tends to extend to 1.272-1.618 BC
  If BC = 0.786 AB: CD tends to extend to 1.272 BC

Completion Score:
  Price_Match = 1 - |CD/AB - target_ratio| / target_ratio
  Time_Match = 1 - |CD_time/AB_time - target_ratio| / target_ratio
  Completion = (Price_Match + Time_Match) / 2
  If Completion > 0.90: High-quality pattern
```

**Signal:**
- **Counter-trend at D:** When CD completes at AB equality (or 1.272/1.618) = reversal entry
- **Direction:** Opposite to CD leg direction
- **Target:** 0.382 or 0.618 retracement of AD
- **Stop:** Beyond 1.618 AB extension

**Risk:** Stop at 1.618 extension; Tight zone for high-quality patterns; Risk 1%
**Edge:** AB=CD is the most common and reliable harmonic pattern because it exploits the market's tendency toward measured moves -- the second impulse leg tends to match the first in both price and time. This pattern appears because institutional order flow repeats: the same institutional demand that created AB tends to re-engage at the BC retracement and push CD to match. The reciprocal BC-CD relationship adds precision to the target.

---

### 217 | Island Reversal Gap Pattern
**School:** New York (Classic TA, Gap Analysis) | **Class:** Gap Pattern
**Timeframe:** Daily | **Assets:** Equities, Futures

**Mathematics:**
```
Island Reversal:
  1. Exhaustion Gap: Price gaps in trend direction (final push)
  2. Island: Price trades in a narrow range for 1-5 days (isolated)
  3. Breakaway Gap: Price gaps in OPPOSITE direction, leaving the island

Identification:
  Gap_Up_Exhaustion = Open > Prior_High (gap up in uptrend)
  Island = trading range with no overlap with pre-gap and post-gap range
  Gap_Down_Breakaway = Open < Prior_Low (gap down, reversing)
  
  For bearish island:
    Island_Top = max(Highs during island)
    Gap1_floor = min(Opens/Lows of exhaustion gap)
    Gap2_ceiling = max(Opens/Highs of breakaway gap)
    
    No price overlap: Gap2_ceiling < Gap1_floor (true island)

Volume Profile:
  Exhaustion gap: usually high volume (last buyers rushing in)
  Island: declining volume (conviction dying)
  Breakaway gap: high volume (reversal conviction)

Statistical Reliability:
  True island reversals (no gap fill for 20 days): ~75% reliable
  Key: the island must remain unfilled (gaps not closed)
```

**Signal:**
- **Short (bearish island top):** After breakaway gap down completes the island pattern
- **Long (bullish island bottom):** After breakaway gap up completes the island pattern
- **Target:** Distance equal to the move that led to the island (measured move)
- **Failure:** Either gap fills (price returns to island territory)

**Risk:** Stop at island midpoint (if gap fills, thesis wrong); Target at measured move; Risk 1.5%
**Edge:** Island reversals are among the most powerful chart patterns because they represent complete ABANDONMENT of a price level -- no one is willing to trade there again (hence the gaps on both sides). The exhaustion gap shows the last desperate participants, and the breakaway gap shows the smart money reversing. The ~75% reliability for unfilled islands is one of the highest for any single chart pattern.

---

### 218 | Pivot Point Fibonacci Confluence
**School:** Floor Trading (NYSE/CME) | **Class:** Support/Resistance
**Timeframe:** Intraday / Daily | **Assets:** All markets

**Mathematics:**
```
Standard Pivot Points:
  PP = (High + Low + Close) / 3
  R1 = 2*PP - Low;    S1 = 2*PP - High
  R2 = PP + (High-Low);  S2 = PP - (High-Low)
  R3 = High + 2*(PP-Low);  S3 = Low - 2*(High-PP)

Fibonacci Pivot Points:
  PP = (H + L + C) / 3
  R1 = PP + 0.382*(H-L);  S1 = PP - 0.382*(H-L)
  R2 = PP + 0.618*(H-L);  S2 = PP - 0.618*(H-L)
  R3 = PP + 1.000*(H-L);  S3 = PP - 1.000*(H-L)

Confluence Zone:
  When Standard R1 and Fibonacci R1 are within 0.3% = strong resistance
  When Standard S1 and Fibonacci S1 are within 0.3% = strong support
  
  Confluence_Score = count of pivot levels within 0.5% price band
  Score 3+: Very strong level (multiple independent calculations agree)

Multi-Day Pivot Confluence:
  Calculate pivots from daily, weekly, and monthly timeframes
  When daily PP = weekly S1 = monthly fib level = extreme confluence
```

**Signal:**
- **Buy at support confluence:** Price reaches zone where 3+ pivot/Fibonacci levels converge from below
- **Sell at resistance confluence:** Price reaches 3+ level convergence from above
- **Breakout:** Price breaks through confluence zone with volume = significant move
- **Target:** Next confluence zone in trade direction

**Risk:** Stop 0.5% beyond confluence zone; Target at next zone; Risk 1%
**Edge:** Floor traders developed pivot points as daily S/R levels, and they became self-fulfilling because millions of traders use them. When standard AND Fibonacci pivots converge at the same price, the level is reinforced by two independent mathematical frameworks, doubling the order flow concentration. Multi-timeframe pivot confluence (daily + weekly + monthly) creates the most powerful intraday levels.

---

### 219 | Kagi Chart Trend Reversal System
**School:** Japanese (19th Century, Telephone Line Charts) | **Class:** Trend Structure
**Timeframe:** Price-based (no time axis) | **Assets:** All markets

**Mathematics:**
```
Kagi Construction:
  Reversal_Amount = 4% (typical) or ATR(14)
  
  Rising line (yang): current close > prior close in current direction
  Falling line (yin): current close < prior close in current direction
  
  Reversal: when price moves Reversal_Amount in opposite direction from last extreme
  
  Thick line (yang): new line exceeds prior line's extreme (bull trend)
  Thin line (yin): new line falls below prior line's extreme (bear trend)

Trading Rules:
  Buy: Line changes from thin (yin) to thick (yang) = trend shifts bullish
  Sell: Line changes from thick (yang) to thin (yin) = trend shifts bearish

Kagi Statistics:
  Yang_Count = consecutive thick segments
  Yin_Count = consecutive thin segments
  
  Shoulder: A yang line that doesn't exceed prior yang high = weakness
  Waist: A yin line that doesn't break prior yin low = strength

Pattern: Multiple shoulders without new highs = topping pattern
Pattern: Multiple waists without new lows = bottoming pattern
```

**Signal:**
- **Buy:** Yin-to-Yang transition (thin to thick line change) = bullish reversal
- **Sell:** Yang-to-Yin transition (thick to thin line change) = bearish reversal
- **Confirmation:** New yang exceeds prior yang high = trend continuation confirmed
- **Warning:** Shoulder formation (yang fails to exceed prior yang) = weakening trend

**Risk:** Stop at prior waist level (for longs); Target open-ended for trends; Risk 1%
**Edge:** Kagi charts, like Renko, eliminate time and focus on price movement. The thick/thin (yang/yin) line transitions provide clear, unambiguous trend change signals. The shoulder/waist concepts (equivalent to lower highs/higher lows) provide early warning of trend exhaustion. The 4% reversal amount filters out all moves smaller than 4%, ensuring only significant swings are captured.

---

### 220 | Fractal Breakout (Bill Williams)
**School:** New York (Bill Williams, Trading Chaos) | **Class:** Fractal
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Williams Fractal:
  Up Fractal: Bar with highest high, where the 2 bars before AND 2 bars after have lower highs
    Fractal_Up = High[n] > max(High[n-2], High[n-1], High[n+1], High[n+2])
  
  Down Fractal: Bar with lowest low, where 2 bars before AND after have higher lows
    Fractal_Down = Low[n] < min(Low[n-2], Low[n-1], Low[n+1], Low[n+2])

Fractal as Support/Resistance:
  Each fractal marks a local extreme that acted as S/R
  Recent fractals are more relevant than old ones

Breakout Strategy:
  Buy: Price closes above the most recent up-fractal
  Sell: Price closes below the most recent down-fractal
  
  With Alligator Filter (Williams):
    Only buy above fractal if fractal is ABOVE the Alligator's teeth (SMA(13))
    Only sell below fractal if fractal is BELOW the teeth

Fractal Dimension Enhancement:
  Count fractals per 100 bars:
    High count (>20): Choppy market (many reversals) -> avoid breakout strategy
    Low count (<10): Trending market (fewer reversals) -> breakout strategy optimal
    
  Fractal_Density = count(fractals, 100) / 100
```

**Signal:**
- **Buy:** Close above recent up-fractal AND fractal above Alligator teeth AND Fractal_Density < 0.15
- **Sell:** Close below recent down-fractal AND fractal below teeth AND Fractal_Density < 0.15
- **Skip:** Fractal_Density > 0.20 (too many fractals = choppy, breakouts fail)
- **Exit:** Opposite fractal breakout OR price crosses back through Alligator

**Risk:** Stop at opposite fractal; Trail with Alligator teeth; Risk 1%
**Edge:** Williams Fractals provide a precise, mechanical definition of local extremes (support/resistance). Combined with the Alligator filter (trend direction) and fractal density (regime quality), this creates a complete breakout system. The fractal density filter is the key innovation -- high fractal density means the market is choppy (fractals forming and breaking constantly), while low density means genuine trends (fractals forming and holding). Only trade breakouts in low-density environments.

---

### 221 | Point and Figure Column Reversal
**School:** New York (Charles Dow, 1880s) | **Class:** P&F Charting
**Timeframe:** Price-based | **Assets:** Equities

**Mathematics:**
```
Point & Figure Construction:
  Box_Size = ATR(14) / 5  (adaptive) or fixed
  Reversal = 3 boxes (standard 3-box reversal)
  
  X column: Rising prices (each X = one box_size up)
  O column: Falling prices (each O = one box_size down)
  New column: price reverses by 3 * box_size

Count Target:
  Horizontal_Count = number of columns in a consolidation pattern
  Vertical_Count = length of initial column at breakout
  
  Price_Target_H = breakout_level + horizontal_count * box_size * reversal_amount
  Price_Target_V = breakout_level + vertical_count * box_size

P&F Patterns:
  Double Top: Two X columns reaching same level -> breakout above = buy
  Double Bottom: Two O columns reaching same level -> breakdown below = sell
  Triple Top: Three X columns at same resistance -> breakout = very bullish
  Catapult: Breakout after bullish signal = continuation (most powerful)

Bullish Percent Index:
  BPI = (# of stocks on P&F buy signals) / total_stocks * 100
  BPI > 70: Market overbought (most stocks bullish)
  BPI < 30: Market oversold (most stocks bearish)
```

**Signal:**
- **Buy (double top breakout):** X column exceeds two prior X column tops
- **Sell (double bottom breakdown):** O column breaks below two prior O column bottoms
- **Triple top breakout:** Strongest P&F buy signal (three failed attempts, then break)
- **Market breadth:** BPI < 30 = market oversold (contrarian buy for market)

**Risk:** Stop at last O column low (for longs); Target by horizontal or vertical count; Risk 1.5%
**Edge:** Point and Figure is the oldest form of charting (predates bar charts) and eliminates time completely. The 3-box reversal requirement filters out 67%+ of price noise. The horizontal and vertical count targets have surprising accuracy because they measure the extent of accumulation/distribution before the breakout. P&F buy/sell signals are binary and unambiguous, eliminating subjective interpretation.

---

### 222 | Fair Value Gap (FVG) Strategy
**School:** London (ICT/Smart Money Concepts) | **Class:** Imbalance Trading
**Timeframe:** 15-min to Daily | **Assets:** All markets

**Mathematics:**
```
Fair Value Gap (FVG):
  A 3-candle pattern where the middle candle's range creates a gap
  that the wicks of candles 1 and 3 don't overlap
  
  Bullish FVG:
    Candle_1 High < Candle_3 Low  (gap between wick 1 top and wick 3 bottom)
    FVG_zone = [Candle_1 High, Candle_3 Low]
    This zone represents unfilled buy orders (demand zone)
  
  Bearish FVG:
    Candle_1 Low > Candle_3 High (gap between wick 1 bottom and wick 3 top)
    FVG_zone = [Candle_3 High, Candle_1 Low]
    This zone represents unfilled sell orders (supply zone)

FVG Quality:
  FVG_size = |FVG_zone| / ATR  (relative to volatility)
  Large FVG (> 1.5x ATR): Strong institutional imbalance (higher quality)
  Small FVG (< 0.3x ATR): Minor imbalance (lower quality, skip)
  
  Volume_Confirmation: Middle candle volume > 1.5x average = institutional FVG

Mitigation:
  FVG is "mitigated" when price returns to fill the gap zone
  Entry: at the 50% level of FVG zone (equilibrium of the imbalance)
  Target: opposite side of FVG
```

**Signal:**
- **Buy at bullish FVG:** When price retraces to bullish FVG zone (demand = buying opportunity)
- **Sell at bearish FVG:** When price retraces to bearish FVG zone (supply = selling opportunity)
- **Quality filter:** Only trade FVGs > 0.5x ATR AND middle candle volume > 1.5x average
- **Exit:** Opposite FVG or next structural resistance/support

**Risk:** Stop beyond FVG zone by 0.5x ATR; Target at next FVG in trade direction; Risk 1%
**Edge:** Fair Value Gaps represent genuine order flow imbalances -- price moved so aggressively that it left unfilled orders behind. When price returns to these zones, the resting orders get filled, creating support/resistance. The ICT framework identifies these as institutional "footprints" -- the specific zones where banks and institutions entered positions. Large FVGs with volume are the highest quality because they represent the largest institutional order flow imbalances.

---

### 223 | Andrews Pitchfork Median Line
**School:** International (Alan Andrews) | **Class:** Geometric Channel
**Timeframe:** 4H / Daily | **Assets:** All markets

**Mathematics:**
```
Andrews Pitchfork:
  Select 3 significant pivot points: P1, P2, P3
    P1: Start of the move
    P2: First swing extreme
    P3: Swing reversal (retracement extreme)
  
  Median Line: Line from P1 through midpoint of P2-P3
  Upper Parallel: Line through P2 parallel to Median Line
  Lower Parallel: Line through P3 parallel to Median Line

Median Line Hypothesis (Andrews):
  "Price will return to the median line approximately 80% of the time"
  
  If price is above median: tends to pull back toward median
  If price is below median: tends to rally toward median
  
  Median Line acts as attractor (equilibrium)

Schiff Modification:
  Move P1 to midpoint of P1 and P2 (flatter pitchfork)
  Better for steep initial moves

Trading at Parallels:
  Price at upper parallel + reversal candle = sell toward median
  Price at lower parallel + reversal candle = buy toward median
  Price breaks above upper parallel = strong uptrend (buy breakout)
  Price breaks below lower parallel = strong downtrend (sell breakout)
```

**Signal:**
- **Buy at lower parallel:** Price touches lower parallel with reversal candle -> target median line
- **Sell at upper parallel:** Price touches upper parallel with reversal candle -> target median line
- **Breakout buy:** Close above upper parallel = strong bull, buy with target at extended parallel
- **Breakout sell:** Close below lower parallel

**Risk:** Stop beyond parallel by 1x ATR; Target at median line; Risk 1%
**Edge:** Andrews' Pitchfork is geometrically derived from THREE points, making it more structurally sound than arbitrary trendlines. The median line hypothesis (80% return rate) is well-tested and provides a clear target for every trade. The pitchfork's parallels create dynamic, forward-projecting support/resistance that adjusts to the market's rhythm. The Schiff modification handles steep moves gracefully.

---

### 224 | Darvas Box Breakout System
**School:** New York (Nicolas Darvas, 1958) | **Class:** Box Breakout
**Timeframe:** Daily | **Assets:** Growth Equities

**Mathematics:**
```
Darvas Box Construction:
  1. Identify a new 52-week high
  2. Box_Top = that 52-week high
  3. Wait until price fails to make new high for 3 consecutive days
  4. Box_Top is confirmed
  5. Box_Bottom = lowest low during the same period
  6. Wait until price fails to make new low for 3 consecutive days
  7. Box_Bottom is confirmed
  8. Box is complete: [Box_Bottom, Box_Top]

Trading Rules:
  Buy: Price closes above Box_Top (breakout confirmed)
  Stop: Box_Bottom (sell if price drops below bottom of box)
  New Box: After breakout, build new box at higher levels (trailing)

Volume Requirement:
  Breakout_Volume > 2x SMA(Volume, 50) = confirmed
  If volume < 1.5x average at breakout: weak signal (skip or reduce size)

Darvas Box Filter:
  Only trade stocks that:
    1. Made new 52-week high (already in strong uptrend)
    2. Box is narrow (< 15% range = tight consolidation)
    3. Breakout on high volume
```

**Signal:**
- **Buy:** Price breaks above Darvas Box top on 2x+ volume
- **Stop:** At box bottom (the box IS your risk management)
- **Trail:** Build new boxes at successively higher levels = trailing stop system
- **Exit:** Price breaks below current box bottom

**Risk:** Stop at box bottom; R:R defined by box height; Risk 1.5%
**Edge:** Darvas developed this system as a part-time dancer, making it deliberately simple. The genius is the requirement for a 52-week high FIRST, then a consolidation box. This ensures you only buy the strongest stocks in the market during their rest phase, not random breakouts. The box itself provides automatic stop-loss (box bottom) and profit management (build new boxes higher). Darvas turned $10,000 into $2M using this system in the late 1950s.

---

### 225 | Pennant/Flag Continuation Quantified
**School:** Classic TA (Quantified) | **Class:** Continuation Pattern
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Flag/Pennant Detection:

Flagpole:
  Strong move: return > 2 * ATR(20) in N days (N < 10)
  Pole_Length = price change during flagpole
  Pole_Duration = number of bars in flagpole

Flag (Rectangle):
  Parallel consolidation channel after flagpole
  Flag_Direction: opposite to flagpole (counter-trend pullback)
  Flag_Duration: 5-20 bars (< pole duration)
  Flag_Depth: < 50% of flagpole (shallow retracement)

Pennant (Converging):
  Converging trendlines (triangle) after flagpole
  Pennant_Duration: 5-15 bars
  Pennant_Depth: < 33% of flagpole (even shallower)

Quality Metrics:
  Volume: Declining during flag/pennant (base building)
  Breakout: Volume spike (> 1.5x average) on flag/pennant break
  
  Measured Move Target = breakout_level + flagpole_length
  Historical success: ~65% hit rate for full measured move

Flag_Score = pole_strength * volume_pattern * symmetry * depth_quality
```

**Signal:**
- **Buy (bull flag):** Breakout above flag/pennant upper boundary on volume
- **Sell (bear flag):** Breakdown below flag/pennant lower boundary on volume
- **Target:** Measured move = flagpole length projected from breakout
- **Failure:** Close back inside flag/pennant pattern

**Risk:** Stop at opposite side of flag/pennant; Target at measured move; Risk 1%
**Edge:** Flags and pennants are the most reliable continuation patterns because they represent brief pauses in strong moves (the flagpole). The shallow retracement (<50% for flags, <33% for pennants) combined with declining volume confirms the pause is rest, not reversal. The measured move target (flagpole length) has ~65% reliability because the same institutional demand that created the flagpole re-engages at the breakout.

---

### 226 | Fibonacci Speed Resistance Fan
**School:** International (Fibonacci/Gann Hybrid) | **Class:** Geometric Analysis
**Timeframe:** Daily / Weekly | **Assets:** All markets

**Mathematics:**
```
Speed Resistance Lines:
  From a significant low (L) to high (H):
  
  Speed Line 1/3: Line from L through (time_midpoint, L + 1/3*(H-L))
  Speed Line 2/3: Line from L through (time_midpoint, L + 2/3*(H-L))

Fibonacci Fan Lines:
  From significant low to significant high:
  Fan_38.2: Line from L through (time_end, L + 0.382*(H-L))
  Fan_50.0: Line from L through (time_end, L + 0.500*(H-L))
  Fan_61.8: Line from L through (time_end, L + 0.618*(H-L))

Dynamic Support/Resistance:
  These lines slope upward/downward over time
  As time progresses, the support/resistance levels change
  
  Trading Rule (Speed Lines):
    In uptrend: price pulling back to 2/3 speed line = support
    If 2/3 breaks: drop to 1/3 speed line (deeper support)
    If 1/3 breaks: trend is over (full reversal)

Combined with Fibonacci Time Zones:
  Time_Zones = Fibonacci sequence (1, 2, 3, 5, 8, 13, 21, 34, 55...)
  Major moves tend to end at Fibonacci time intervals from prior extreme
```

**Signal:**
- **Buy at fan support:** Price touches 61.8% fan line from above = strongest support
- **Sell at fan resistance:** Price touches 38.2% fan line from below = strongest resistance
- **Break:** 2/3 speed line break = first warning; 1/3 break = full reversal signal
- **Time:** Fibonacci time zones mark when moves may exhaust or reverse

**Risk:** Stop below fan line by 1 ATR; Target at next fan line; Risk 1%
**Edge:** Fibonacci fans create DYNAMIC support/resistance that changes with time, unlike horizontal Fibonacci levels that are static. This captures the reality that support/resistance weakens or strengthens as time passes. Speed resistance lines divide the move into thirds and provide the classic floor-trader rule: "A move retracing beyond the 2/3 line signals the move is over." The combination of price and time Fibonacci creates a geometric framework for projecting both WHERE and WHEN reversals occur.

---

### 227 | Momentum Divergence Multi-Indicator
**School:** Quantitative | **Class:** Divergence Ensemble
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Multi-Indicator Divergence Detection:
  Scan for divergence simultaneously across:
    1. RSI(14) vs. Price
    2. MACD Histogram vs. Price
    3. Stochastic %K vs. Price
    4. CCI(20) vs. Price
    5. ROC(12) vs. Price

Divergence at each indicator:
  Bullish: Price lower low BUT indicator higher low
  Bearish: Price higher high BUT indicator lower high

Divergence Confluence Score:
  DCS = sum(has_divergence_i) for i = 1 to 5
  DCS = 1: Single indicator divergence (weak, ignore)
  DCS = 2: Moderate (one out of context, one isn't)
  DCS = 3: Strong (3 independent indicators diverging)
  DCS = 4-5: Extreme (near-certain reversal setup)

Quality Weighting:
  Divergence at oversold/overbought extreme: 1.5x weight
  Divergence at support/resistance level: 1.3x weight
  Divergence on timeframe higher than entry: 1.5x weight
```

**Signal:**
- **Buy:** DCS >= 3 (3+ indicators show bullish divergence) at support or oversold
- **Sell:** DCS >= 3 bearish divergence at resistance or overbought
- **Ignore:** DCS < 2 (too few indicators agree)
- **Highest conviction:** DCS = 5 at key structural level

**Risk:** Stop below most recent low (bullish div) or above high (bearish div); Risk 1%
**Edge:** Single-indicator divergence has a ~45% success rate -- barely better than a coin flip. But when 3+ independent indicators all show divergence simultaneously, the probability jumps to ~65-70% because each indicator measures a DIFFERENT aspect of momentum (RSI = magnitude, MACD = acceleration, Stochastic = position, CCI = deviation, ROC = velocity). Multiple independent confirmations dramatically reduce false signals.

---

### 228 | London Session Liquidity Sweep
**School:** London (ICT/Smart Money) | **Class:** Liquidity Hunt
**Timeframe:** Intraday (15-min) | **Assets:** FX, Index Futures

**Mathematics:**
```
Liquidity Concept:
  Buy-side Liquidity: cluster of stop-losses above resistance/highs
    (if price sweeps above, these stops trigger buying = liquidity for institutions to sell)
  
  Sell-side Liquidity: cluster of stop-losses below support/lows
    (if price sweeps below, these stops trigger selling = liquidity for institutions to buy)

Liquidity Sweep:
  Price briefly exceeds a key level (prior high/low, round number, prior session extreme)
  Then REVERSES sharply within 1-3 candles
  
  Sweep_Size = max(price_beyond_level) - level  (how far price went past)
  Sweep_Duration = bars spent beyond level (shorter = sharper = more likely genuine sweep)
  
Optimal London Sweep Time:
  Asian session creates a range (Asia High, Asia Low)
  London session (03:00-05:00 ET) sweeps one side of Asian range THEN reverses
  
  London_Sweep_Up = price briefly exceeds Asia High, then reverses below
  London_Sweep_Down = price briefly breaks Asia Low, then reverses above

  Historical: 60%+ of London session moves start with a sweep of Asia range
```

**Signal:**
- **Buy (sweep-and-reverse):** Price sweeps below Asia Low then reverses above within 3 bars
- **Sell (sweep-and-reverse):** Price sweeps above Asia High then reverses below within 3 bars
- **Entry:** On reversal candle close back inside Asian range
- **Target:** Opposite end of Asian range (or further if London trend develops)

**Risk:** Stop beyond sweep extreme; Target at Asia opposite extreme; Risk 0.5-1%
**Edge:** Institutional traders need LIQUIDITY to fill large orders. Stop-loss clusters above/below obvious levels provide that liquidity. The London sweep pattern exploits this: institutions intentionally push price beyond the Asian range to trigger stops, absorb the resulting order flow, then reverse to trade in their intended direction. Understanding that the first move in London is often a FAKE (liquidity sweep) provides a high-probability counter-trade entry.

---

### 229 | Triple Screen Enhanced (Elder)
**School:** New York (Alexander Elder, Extended) | **Class:** Multi-Timeframe
**Timeframe:** Multi-timeframe | **Assets:** All markets

**Mathematics:**
```
Alexander Elder's Triple Screen (Enhanced):

Screen 1 (Tide, Weekly):
  Trend = direction of 13-week EMA slope
  Force = 13-period Force Index direction
  If both positive: weekly tide is bullish
  If both negative: weekly tide is bearish
  If conflicting: no trade

Screen 2 (Wave, Daily):
  Look for pullbacks against the weekly tide
  If weekly bullish: look for daily oscillator oversold (RSI < 30 or Stoch < 20)
  If weekly bearish: look for daily oscillator overbought (RSI > 70 or Stoch > 80)
  
  Enhancement: Use multiple oscillators
    Signal_Count = count(RSI_oversold, Stoch_oversold, Williams_oversold, CCI_oversold)
    If Signal_Count >= 2: strong pullback signal

Screen 3 (Ripple, Intraday or Breakout Entry):
  If weekly bullish AND daily pullback identified:
    Enter on intraday breakout (buy stop above prior day's high)
  If weekly bearish AND daily bounce identified:
    Enter on intraday breakdown (sell stop below prior day's low)

Triple Screen Score:
  Weekly_Score (0-2): trend + force alignment
  Daily_Score (0-4): number of oscillators in pullback
  Entry_Score (0-1): breakout confirmed
  Total = Weekly + Daily + Entry (max 7)
```

**Signal:**
- **Buy:** Total Score >= 5 (weekly bullish + 2+ oscillators oversold + entry trigger)
- **Sell:** Total Score >= 5 (weekly bearish + 2+ oscillators overbought + entry trigger)
- **No trade:** Weekly Screen conflicting OR Daily Score < 2
- **Exit:** Weekly tide changes direction OR daily oscillators reach opposite extreme

**Risk:** Stop at prior day's opposite extreme; Trail with daily oscillator; Risk 1%
**Edge:** Elder's Triple Screen forces alignment across THREE timeframes before entering a trade, dramatically reducing false signals. The enhancement of using multiple oscillators on Screen 2 (instead of just one) ensures the pullback is genuine. The breakout entry on Screen 3 prevents buying into a falling knife. The total score quantification removes subjectivity from what is traditionally a discretionary approach.

---

### 230 | Engulfing Pattern with Volume Confirmation
**School:** Japanese/Quantitative | **Class:** Reversal Candle
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Bullish Engulfing:
  Prior_Bar: Close < Open (bearish bar)
  Current_Bar: Open < Prior_Close AND Close > Prior_Open
    (current bar's body completely engulfs prior bar's body)
  
  Additional Requirements for HIGH-QUALITY engulfing:
    1. Engulfing_Ratio = Current_Body / Prior_Body > 1.5 (significantly larger)
    2. Volume_Ratio = Current_Volume / Prior_Volume > 1.3 (volume confirmation)
    3. Close in top 20% of current bar's range (strong close)
    4. Prior bar was in downtrend context (not random)

Quality Score:
  QS = Engulfing_Ratio * Volume_Ratio * Close_Position * Trend_Context
  
  Where:
    Close_Position = (Close - Low) / (High - Low)  (IBS, want > 0.8)
    Trend_Context = 1.5 if at support, 1.0 otherwise

Statistical Edge (Bulkowski):
  Bullish Engulfing at bottom: 63% reversal rate (decent)
  With Volume > 1.5x: 71% reversal rate (significant improvement)
  With Engulfing Ratio > 2.0: 68% reversal rate
  With all quality filters: ~75% reversal rate
```

**Signal:**
- **Buy:** Bullish engulfing with QS > 2.0 at support level
- **Sell:** Bearish engulfing with QS > 2.0 at resistance level
- **Skip:** QS < 1.5 (low-quality engulfing, likely noise)
- **Exit:** Next opposite engulfing or price target at 2x ATR

**Risk:** Stop below engulfing bar's low; Target at 2x ATR or prior resistance; Risk 1%
**Edge:** Engulfing patterns are the most common reversal candle, but unfiltered they have only ~55-60% accuracy. The quality score combining engulfing ratio, volume, close position, and trend context raises this to ~75%. The critical filter is volume confirmation (1.3x+): a bullish engulfing with strong volume means aggressive buyers overwhelmed sellers AND committed capital to do so.

---

### 231 | Donchian Channel Width Squeeze
**School:** New York (Richard Donchian, Turtle Trading) | **Class:** Channel Squeeze
**Timeframe:** Daily | **Assets:** Futures, Equities

**Mathematics:**
```
Donchian Channel:
  Upper = highest_high(n)
  Lower = lowest_low(n)
  Width = Upper - Lower
  
  n = 20 (standard)

Donchian Width Percentile:
  DW_pctile = percentile_rank(Width, 252)
  
  DW_pctile < 10: Extreme narrowing (historical low volatility = squeeze)
  DW_pctile > 90: Extreme widening (high volatility = extension)

Squeeze-and-Fire:
  Phase 1 (Squeeze): DW_pctile < 15 for 3+ days = compression
  Phase 2 (Fire): Price breaks above Upper or below Lower Channel
  Phase 3 (Run): Width expanding rapidly (DW_pctile rising)

Original Turtle Rules (simplified):
  Buy: Close > 20-day highest high (break upper Donchian)
  Sell: Close < 20-day lowest low (break lower Donchian)
  Stop: 2 * ATR from entry
  Add: every 0.5 * ATR in favorable direction (pyramiding)
```

**Signal:**
- **Squeeze setup:** DW_pctile < 15 (channel at narrow extreme, breakout imminent)
- **Breakout long:** Close > Upper Channel from squeeze condition
- **Breakout short:** Close < Lower Channel from squeeze condition
- **Exit:** Opposite Donchian break (10-period for exit vs 20-period for entry)

**Risk:** Stop at 2x ATR; Pyramid at 0.5 ATR increments; Max 4 units; Risk 1% per unit
**Edge:** Donchian Channels are the simplest trend-following system (used by the legendary Turtle traders). The Width Squeeze enhancement adds a pre-condition that dramatically improves signal quality: only take breakouts when the channel is at historical narrows. This filters out ~70% of Donchian breakout signals but keeps the highest-quality ones. The Turtles earned >80% annual returns for a decade using variations of this basic channel breakout system.

---

### 232 | Candlestick Pattern + S/R Confluence
**School:** Quantitative (Systematic TA) | **Class:** Pattern + Level
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Level Detection:
  Support/Resistance levels from multiple sources:
    1. Prior pivots (swing highs/lows)
    2. Round numbers (psychological levels)
    3. Volume POC (from volume profile)
    4. Moving averages (SMA 50, 100, 200)
    5. Fibonacci retracements

Level Strength:
  LS = count of independent sources at this price (+/- 0.5% tolerance)
  LS = 1: Weak level (single source)
  LS = 3+: Strong level (confluence)

Pattern at Level:
  When bullish reversal candle pattern appears within 0.5% of strong level:
    Confluence_Score = Pattern_Reliability * Level_Strength * Volume_Confirmation
    
  When bearish reversal candle appears at strong resistance:
    Same Confluence_Score calculation

Statistical Enhancement:
  Pattern ALONE: ~55% accuracy
  Strong Level ALONE: ~52% accuracy (price bounces 52% of time)
  Pattern + Strong Level: ~68-72% accuracy (synergy)
  Pattern + Strong Level + Volume: ~75% accuracy
```

**Signal:**
- **Buy:** Bullish candle pattern at strong support (LS >= 3) with volume confirmation
- **Sell:** Bearish candle pattern at strong resistance (LS >= 3) with volume confirmation
- **Ignore:** Pattern at weak level (LS < 2) or without volume
- **Exit:** At next strong level in trade direction

**Risk:** Stop below support level; Target at next strong S/R level; Risk 1%
**Edge:** The core insight is that candlestick patterns and support/resistance levels are INDEPENDENTLY derived but work synergistically. A bullish engulfing at a random price has ~55% accuracy. A bullish engulfing at a triple-confluence support (prior pivot + SMA 200 + Fibonacci 61.8%) has ~72% accuracy. The probability improvement from combining independent signals is the statistical foundation of all systematic trading.

---

### 233 | Gann Square of Nine Price/Time
**School:** New York (W.D. Gann, 1930s) | **Class:** Geometric/Esoteric
**Timeframe:** Daily / Weekly | **Assets:** All markets

**Mathematics:**
```
Square of Nine:
  Spiral arrangement of numbers starting from center:
  Center = 1, spiral outward: 2,3,4,5,...,9,10,...,25,...

  Key Price Levels from Square of Nine:
    For a base price P:
    Level_n = (sqrt(P) + n/8 * 2)^2  (adding n octaves of 45 degrees)
    
    90-degree resistance:  (sqrt(P) + 0.25*2)^2
    180-degree resistance: (sqrt(P) + 0.50*2)^2
    270-degree resistance: (sqrt(P) + 0.75*2)^2
    360-degree resistance: (sqrt(P) + 1.00*2)^2

Gann Angles (from pivot):
  1x1 line: 45 degrees (price moves 1 unit per time unit)
  2x1 line: 63.75 degrees (price moves 2 units per time)
  1x2 line: 26.25 degrees (price moves 1 unit per 2 time units)
  
  Trading above 1x1: bullish
  Trading below 1x1: bearish
  Each Gann angle provides support/resistance

Gann Time Cycles:
  Key cycle lengths: 30, 60, 90, 120, 180, 270, 360 calendar days
  Anniversaries of significant highs/lows (+/- 3 days) often produce reversals
```

**Signal:**
- **Support/Resistance:** Square of Nine levels (90/180/270/360 degree prices)
- **Trend:** Trading above 1x1 Gann angle = bullish; below = bearish
- **Time cycle:** 180/360 day anniversaries of major pivots = potential reversal dates
- **Entry:** Price at Gann level + reversal candle + time cycle alignment

**Risk:** Stop beyond Gann level; Target at next Gann level; Risk 1%
**Edge:** Gann's methods are controversial but have devoted followers because they occasionally produce precise price/time calls that seem impossible. The Square of Nine generates price levels based on geometric relationships (square root scaling), and these levels do appear as support/resistance in practice, likely because enough traders monitor them to create self-fulfilling prophecy. The time cycle component (180/360 day anniversaries) is independently validated by seasonal and behavioral research.

---

### 234 | Quantified Support/Resistance Strength
**School:** Quantitative | **Class:** Level Scoring
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
For each potential S/R level (from pivots, round numbers, etc.):

Touch Count:
  TC = number of times price tested this level (within 0.3% tolerance)
  More touches = stronger level (more confirmed)

Recency:
  R = 1 / sqrt(bars_since_last_touch)
  Recent touches = more relevant

Volume at Level:
  VAL = average volume during touches / overall average volume
  High volume at level = stronger (more committed orders)

Rejection Quality:
  RQ = average |reversal_move_after_touch| / ATR
  Strong rejections = more reactive level

Time Holding:
  TH = average bars price spent within 0.5% of level / average bar duration
  More time = more acceptance (less likely to hold as S/R next time)

Level Strength Score:
  LSS = (TC * 0.25 + R * 0.20 + VAL * 0.25 + RQ * 0.20 + (1-TH) * 0.10)
  Normalized to [0, 100]

  LSS > 80: Very strong S/R (high probability of holding)
  LSS 50-80: Moderate S/R
  LSS < 50: Weak S/R (likely to break)
```

**Signal:**
- **Buy at strong support:** LSS > 80 AND reversal candle at level
- **Sell at strong resistance:** LSS > 80 AND reversal candle at level
- **Breakout through weak level:** LSS < 50 AND price breaks level with volume = genuine breakout
- **Avoid:** Trading at moderate levels (50-80) without additional confirmation

**Risk:** Stop 1% beyond S/R level; Target at next strong level; Risk 1%
**Edge:** Most traders assess support/resistance subjectively ("it looks like a strong level"). This framework quantifies level strength using five independent metrics: how many times it was tested, how recently, with how much volume, how strongly price rejected, and how much time was spent there. The resulting Level Strength Score provides an objective basis for prioritizing which levels to trade and which to expect breakouts through.

---

### 235 | Spinning Top and Doji Indecision Cluster
**School:** Japanese/Quantitative | **Class:** Indecision Analysis
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Indecision Candles:
  Doji: |Close - Open| < 0.1 * (High - Low) AND range > 0.5*ATR
  Spinning Top: |Close - Open| < 0.3 * (High - Low) AND wicks on both sides
  High Wave: Spinning top with unusually long wicks (wicks > 2x body)

Indecision Cluster:
  IC = count(Doji + Spinning_Top + High_Wave, last 5 bars)
  
  IC = 0-1: Decisive market (trending, clear direction)
  IC = 2-3: Building indecision (market uncertain)
  IC >= 4: Extreme indecision (5 bars of indecision = MUST resolve soon)

Resolution Direction:
  After IC >= 3:
    First decisive bar (body > 0.7 * range) determines direction
    This decisive bar on high volume = breakout from indecision
    
  Historical:
    After IC >= 3: 70% chance of 2x ATR move within next 5 bars (breakout)
    Direction: 55% in direction of prior trend (continuation), 45% reversal

Volume During Cluster:
  If volume declining during cluster: healthy consolidation, breakout likely in trend direction
  If volume increasing during cluster: distribution/accumulation, reversal more likely
```

**Signal:**
- **Pre-breakout alert:** IC >= 3 (multiple indecision candles = energy building)
- **Breakout entry:** First decisive candle after IC >= 3 cluster -> trade in that direction
- **Trend bias:** If volume declining in cluster AND prior trend clear -> bias toward continuation
- **Exit:** After 2x ATR achieved (most of the breakout energy captured) or 5 bars

**Risk:** Stop at opposite end of cluster range; Target 2x ATR; Risk 1%
**Edge:** Indecision clusters are the visual expression of a market in equilibrium -- buyers and sellers are perfectly balanced. This equilibrium is unstable and must eventually resolve with a directional move. The longer the indecision (more candles), the more stored energy is released when the resolution occurs. By waiting for the first decisive candle to reveal direction, you enter at the moment the equilibrium breaks, which has a 70% probability of producing a significant move.

---

### 236 | Gap Analysis with Classification
**School:** New York (Gap Trading) | **Class:** Gap Pattern Taxonomy
**Timeframe:** Daily | **Assets:** Equities, Futures

**Mathematics:**
```
Gap Classification (Bulkowski/Edwards & Magee):

Common Gap:
  Occurs within trading range, no trend context
  Fill_Rate: ~90% (almost always fills within 3-5 days)
  Action: Fade (trade toward gap fill)

Breakaway Gap:
  Occurs at start of new trend, breaks out of consolidation
  Fill_Rate: ~50% (often doesn't fill for weeks/months)
  Confirmation: High volume + outside value area
  Action: Trade in gap direction (trend beginning)

Runaway (Measuring) Gap:
  Occurs in middle of established trend (continuation)
  Fill_Rate: ~65% (eventually fills but after continued move)
  Size: approximately half of the total expected move
  Action: Trade in gap direction with measured move target

Exhaustion Gap:
  Occurs at END of trend, often with island reversal
  Fill_Rate: ~95% (fills quickly, trend reversing)
  Confirmation: Volume spike + RSI divergence
  Action: Fade gap (reversal trade)

Gap Classification Algorithm:
  If in_range AND volume_normal: Common
  If breakout AND volume_high AND out_of_value: Breakaway
  If mid_trend AND volume_moderate: Runaway
  If extended_trend AND volume_spike AND divergence: Exhaustion
```

**Signal:**
- **Fade common gap:** Trade toward gap fill (90% fill rate)
- **Follow breakaway gap:** Trade in gap direction (50% fill = often runs)
- **Target from runaway gap:** Gap position = halfway target (project full move)
- **Fade exhaustion gap:** Counter-trend trade after island or reversal confirmation

**Risk:** Stop depends on gap type; Tighter for fade, wider for follow; Risk 1-1.5%
**Edge:** Gap classification transforms gap trading from a ~50% game (random direction) to a 60-70% game by identifying WHICH TYPE of gap occurred. The key is context: is the gap happening at the start of a trend (breakaway), in the middle (runaway), at the end (exhaustion), or randomly (common)? Each type has dramatically different fill rates and directional implications. The measurement concept from runaway gaps (gap = 50% of move) provides a precise price target.

---

### 237 | Double Top/Bottom Quantified
**School:** Classic TA (Quantified) | **Class:** Reversal Pattern
**Timeframe:** Daily / Weekly | **Assets:** All markets

**Mathematics:**
```
Double Top Detection:
  Peak_1: local high
  Trough: low between peaks (minimum 10% from peaks)
  Peak_2: second high within 3% of Peak_1 (approximate equality)
  
  Neckline = Trough price level

Requirements:
  |Peak_1 - Peak_2| / Peak_1 < 0.03 (peaks within 3%)
  Duration: 20-120 days between peaks
  Trough_Depth > 0.10 * Peak_1 (minimum 10% pullback between peaks)
  Volume: Peak_2 volume < Peak_1 volume (declining conviction)

Confirmation:
  Break below neckline on volume > 1.3x average
  
Target:
  Measured_Move = Neckline - (Peak_Avg - Neckline)
  Historical: 63% reach target (Bulkowski)

Adam & Eve Variants:
  Adam (V-shaped peak): Sharper, more volatile peak
  Eve (U-shaped peak): Rounder, more distributed peak
  Adam-Eve combo: most reliable variant (70% target hit rate)
```

**Signal:**
- **Short:** Break below neckline after confirmed double top (volume + pattern criteria)
- **Long:** Break above neckline after confirmed double bottom
- **Target:** Measured move (pattern height projected from neckline)
- **Failure:** Price closes 3%+ above Peak_2 = pattern failure

**Risk:** Stop 3% above Peak_2 (or Peak_2 itself); Target at measured move; Risk 1.5%
**Edge:** Double tops/bottoms are one of the most common and well-tested reversal patterns. Quantifying the requirements (3% peak tolerance, 10% minimum trough, volume decline, neckline break) raises the success rate from ~50% (unfiltered) to ~63-70% (filtered). The Adam-Eve variant (sharp first peak, round second peak) is particularly reliable because it represents the transition from aggressive (informed) selling to gradual (retail) exhaustion.

---

### 238 | Descending/Ascending Triangle Quantified
**School:** Classic TA (Quantified) | **Class:** Continuation/Reversal
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Ascending Triangle:
  Flat resistance: 3+ touches at approximately same level (within 0.5%)
  Rising support: higher lows (each trough higher than prior)
  
  Support Slope: linreg_slope of trough prices (positive for ascending)
  Resistance Level: average of resistance touches

  Volume: typically declining as triangle narrows (compression)
  Breakout: close above resistance on volume > 1.5x average
  
  Measured Move: height of triangle (resistance - first trough) projected upward

Descending Triangle:
  Flat support: 3+ touches at approximately same level
  Falling resistance: lower highs (each peak lower)
  
  Breakout: close below support on volume
  Measured Move: triangle height projected downward

Symmetrical Triangle:
  Both sides converging (lower highs AND higher lows)
  Can break either direction (50/50)
  
Triangle Quality:
  Touches = count of touches on both sides (more = better formed)
  Volume_Decline = volume trend during formation (declining = good)
  Duration = 10-60 bars ideal
  Width = initial height vs current (converging = proper)
```

**Signal:**
- **Buy (ascending):** Close above flat resistance on volume (continuation upward)
- **Sell (descending):** Close below flat support on volume (continuation downward)
- **Symmetrical:** Trade in direction of breakout with volume confirmation
- **Target:** Triangle height projected from breakout point

**Risk:** Stop inside triangle (below resistance for long, above support for short); Risk 1.5%
**Edge:** Ascending triangles are bullish continuation patterns (~75% break upward) because the flat resistance combined with rising support shows buyers becoming more aggressive (willing to pay higher) while sellers hold firm. Eventually, buying pressure overcomes the fixed supply. Descending triangles are the bearish mirror (~72% break downward). The volume decline during formation confirms energy building before the resolution.

---

### 239 | Wedge Pattern (Rising/Falling) Detection
**School:** Classic TA (Quantified) | **Class:** Reversal Pattern
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Rising Wedge (Bearish):
  Upper trendline: connecting higher highs (slope > 0)
  Lower trendline: connecting higher lows (slope > 0)
  Both slopes positive BUT lower slope STEEPER than upper slope
  = price converging upward (narrowing range while rising)

  Slope_Ratio = Lower_Slope / Upper_Slope
  If Slope_Ratio > 1.2: Valid rising wedge (lower line steeper)
  
  Breakout: Close below lower trendline (bearish breakdown)
  Target: Height of wedge at widest point, projected downward from breakout

Falling Wedge (Bullish):
  Both slopes negative BUT upper slope STEEPER than lower slope
  = price converging downward (narrowing while falling)
  
  Slope_Ratio = Upper_Slope / Lower_Slope  (both negative, ratio > 1)
  If Slope_Ratio > 1.2: Valid falling wedge
  
  Breakout: Close above upper trendline (bullish breakout)

Quality Metrics:
  Touches: >= 5 total (at least 2 on each line, ideally 3 each)
  Duration: 20-60 bars
  Volume: declining during wedge (compression)
  Convergence_Point: lines would intersect within 20 bars (tight convergence)
```

**Signal:**
- **Short:** Rising wedge breaks below lower trendline on volume (bearish reversal)
- **Long:** Falling wedge breaks above upper trendline on volume (bullish reversal)
- **Target:** Wedge height (widest point) projected from breakout
- **Failure:** Price re-enters wedge and continues in wedge direction

**Risk:** Stop inside wedge (last touch of broken trendline); Target at measured move; Risk 1.5%
**Edge:** Wedges are convergence patterns where the dominant side (buyers in rising wedge) is becoming WEAKER over time (narrowing range despite continuing in that direction). The converging slopes mathematically show that each successive push is generating less price movement -- diminishing returns. When the exhausted side finally gives way, the reversal is often sharp because the structural support was already weakening. Rising wedges break down ~67% of the time; falling wedges break up ~68%.

---

### 240 | Pivot Reversal with Confluence Scoring
**School:** Floor Trading (Enhanced) | **Class:** Reversal at Pivots
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Pivot High Detection:
  PH = High[n] > max(High[n-k:n-1]) AND High[n] > max(High[n+1:n+k])
  (highest bar with k bars on each side having lower highs)
  k = 3 (standard) or 5 (strong)

Pivot Low Detection:
  PL = Low[n] < min(Low[n-k:n-1]) AND Low[n] < min(Low[n+1:n+k])

Pivot Quality Score:
  1. Height: |Pivot - SMA(20)| / ATR (distance from mean)
  2. Symmetry: similarity of left side and right side moves to/from pivot
  3. Volume: volume at pivot bar vs surrounding bars
  4. Confluence: count of other S/R levels within 0.5% of pivot

  PQS = Height * 0.3 + Symmetry * 0.2 + Volume * 0.25 + Confluence * 0.25

Pivot Cluster:
  Multiple pivot highs or lows at similar prices (within 1%) across different time periods
  Cluster_Count >= 3: strong S/R zone (multiple independent tests)
```

**Signal:**
- **Buy at pivot low cluster:** Price tests zone with 3+ pivot lows -> buy with reversal candle
- **Sell at pivot high cluster:** Price tests zone with 3+ pivot highs -> sell with reversal candle
- **Breakout:** Price breaks through pivot cluster on volume = significant structural break
- **Quality filter:** Only trade pivots with PQS > 70 (top 30% quality)

**Risk:** Stop 1% beyond pivot cluster; Target at next pivot cluster; Risk 1%
**Edge:** Pivots are the most objective definition of local extremes -- they require bars on BOTH sides to be lower/higher, meaning the market definitively rejected that price. When multiple independent pivot points cluster at the same price (across different time periods), the level has been tested and defended multiple times, creating a robust S/R zone. The quality score adds volume and confluence analysis to filter for the most actionable pivots.

---

### 241 | Inside Bar Breakout Strategy
**School:** Price Action (Modern TA) | **Class:** Compression Breakout
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Inside Bar:
  High_t < High_{t-1} AND Low_t > Low_{t-1}
  (today's range entirely contained within yesterday's range)

  Mother Bar = prior bar (containing bar)
  Inside Bar = current bar (contained bar)

Multiple Inside Bars:
  2 consecutive inside bars: IB2 (stronger compression)
  3+ consecutive inside bars: IB3 (extreme compression, rare but powerful)

Inside Bar Quality:
  Relative_Size = (IB_Range) / (MB_Range)
  Smaller inside bar relative to mother = tighter compression = better
  Ideal: Relative_Size < 0.5 (inside bar < 50% of mother bar)

Breakout:
  Long: Close > Mother_Bar_High
  Short: Close < Mother_Bar_Low
  
  Mother_Bar_Midpoint = (MB_High + MB_Low) / 2
  If inside bar close > midpoint: slight bullish bias
  If inside bar close < midpoint: slight bearish bias

Context:
  Inside bar in uptrend: 60% break upward (continuation)
  Inside bar in downtrend: 58% break downward (continuation)
  Inside bar at S/R level: 65% break in reversal direction
```

**Signal:**
- **Buy breakout:** Close above mother bar high (prefer uptrend context)
- **Sell breakout:** Close below mother bar low (prefer downtrend context)
- **Size by compression:** IB2/IB3 = larger position (tighter compression = stronger breakout)
- **Exit:** Mother bar's opposite extreme (if long, stop at mother bar low)

**Risk:** Stop at opposite end of mother bar; Target 2x ATR; Risk 1%
**Edge:** Inside bars represent a period where NEITHER buyers NOR sellers could exceed the prior bar's range -- a perfect equilibrium that must resolve. The mother bar's range becomes the compression vessel, and the breakout direction reveals which side accumulated more orders during the pause. Multiple inside bars (IB2, IB3) create even tighter compression with even more explosive breakouts. The mother bar range provides a natural stop-loss level.

---

### 242 | Pin Bar (Hammer/Shooting Star) Quality Scoring
**School:** Price Action | **Class:** Rejection Candle
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Pin Bar (Bullish - Hammer):
  Long lower wick: (Open - Low) > 2 * |Close - Open| (or Close - Low if bullish)
  Small body: |Close - Open| < 0.33 * (High - Low)
  Wick_Ratio = lower_wick / total_range (should be > 0.60)

Pin Bar (Bearish - Shooting Star):
  Long upper wick: (High - max(Open,Close)) > 2 * |Close - Open|
  Small body: |Close - Open| < 0.33 * (High - Low)
  Wick_Ratio = upper_wick / total_range (should be > 0.60)

Pin Bar Quality Score:
  1. Wick_Ratio: longer wick = stronger rejection [0-1]
  2. Body_Position: body at extreme of range (best: body in top 20% for hammer) [0-1]
  3. Nose_Protrusion: wick extends beyond surrounding bars' range [0-1]
  4. Volume: volume > 1.2x average = conviction behind rejection [0-1]
  5. Context: at key S/R level = 1.5x multiplier
  
  PBQ = (Wick_Ratio + Body_Position + Nose_Protrusion + Volume) * Context
  Max ~6.0; trade when PBQ > 3.5

Bulkowski Statistics:
  Hammer at bottom: 60% reversal (moderate)
  With volume confirmation: 67%
  At support level: 72%
  Shooting star at top: 59% reversal
```

**Signal:**
- **Buy (hammer):** PBQ > 3.5 at support level (high-quality bullish rejection)
- **Sell (shooting star):** PBQ > 3.5 at resistance level (high-quality bearish rejection)
- **Skip:** PBQ < 2.5 (low-quality pin bar, likely noise)
- **Entry:** On close of pin bar OR on next bar's open

**Risk:** Stop beyond pin bar wick (the extreme of rejection); Target 2x wick length; Risk 1%
**Edge:** Pin bars represent the market's REJECTION of a price level -- price probed deep into a zone, found overwhelming opposing orders, and reversed sharply. The quality scoring ensures you only trade the most decisive rejections (long wick, extreme body position, volume, at a key level). A high-quality pin bar at a strong level is one of the most reliable single-bar reversal signals, with ~72% accuracy when all quality criteria are met.

---

### 243 | Three Line Break Chart Pattern
**School:** Japanese (Nison, adapted from Japanese charting) | **Class:** Trend Following
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Three Line Break Construction:
  Rules:
  1. Plot a new line if close exceeds the prior line's extreme
  2. For reversal: close must exceed the extreme of the LAST 3 lines
  
  Bullish line: Close > prior line high (extend upward)
  Bearish reversal: Close < lowest low of last 3 bullish lines (reversal)

  This creates a "three line break" that filters out moves smaller than 3 prior lines

Trading Signals:
  New bullish line after bearish sequence: BUY (new uptrend beginning)
  New bearish line after bullish sequence: SELL (new downtrend beginning)

Streak Analysis:
  Count consecutive same-direction lines
  Average streak = expected trend duration
  Current streak > 1.5x average = exhaustion risk
  
  After long bearish streak (6+ lines): bullish reversal line is very high probability

Line Size Analysis:
  Size = |Close - Prior_Extreme|
  If line sizes growing: trend accelerating (healthy)
  If line sizes shrinking: trend decelerating (weakening)
```

**Signal:**
- **Buy:** New bullish line after bearish sequence (trend change to uptrend)
- **Sell:** New bearish line after bullish sequence (trend change to downtrend)
- **Exhaustion:** Streak > 1.5x average AND line sizes shrinking = prepare for reversal
- **Exit:** Opposite reversal signal (three-line break in opposite direction)

**Risk:** Stop at the price that triggered the reversal signal; Trail with 3-line break method; Risk 1%
**Edge:** Three Line Break charts require a move to exceed THREE prior lines' worth of range to reverse, creating an extremely high threshold for noise. Only genuine trend changes survive this filter. The streak and line size analysis add quantitative metrics to what is traditionally a visual method. After extended bearish streaks (6+ lines), the market is deeply oversold by definition, making the eventual bullish reversal line a high-probability buy signal.

---

### 244 | Harmonic Bat Pattern
**School:** International (Scott Carney, 2001) | **Class:** Harmonic
**Timeframe:** 4H / Daily | **Assets:** All markets

**Mathematics:**
```
Bat Pattern (Bullish):
  X to A: Initial impulse
  A to B: 0.382-0.500 retracement of XA (KEY: NOT 0.618, that's Gartley)
  B to C: 0.382-0.886 retracement of AB
  C to D: 0.886 retracement of XA (KEY: the defining ratio)

  D Point (PRZ):
    0.886 XA retracement
    1.618-2.618 BC extension
    Convergence of these two = Potential Reversal Zone

Bat vs. Gartley:
  Gartley: B at 0.618 XA, D at 0.786 XA
  Bat: B at 0.382-0.500 XA, D at 0.886 XA (deeper completion)

  Bat pattern is more conservative (deeper retracement = bigger discount)
  Stop placement tighter (just beyond X, not far from D)

Pattern Validation:
  B ratio: 0.382-0.500 of XA (+/- 5% tolerance)
  D ratio: 0.886 of XA (+/- 3% tolerance, tighter for D)
  BC within AB: 0.382-0.886
```

**Signal:**
- **Buy at D:** When price reaches 0.886 XA retracement with reversal confirmation
- **Target 1:** 0.382 retracement of AD
- **Target 2:** 0.618 retracement of AD
- **Stop:** Below X point (tight stop, just below 1.0 XA retracement)

**Risk:** Stop below X; Very tight risk (D at 0.886, stop at 1.0 = only 0.114 XA risk); R:R often 3:1+
**Edge:** The Bat pattern provides the best risk/reward ratio of all harmonic patterns because D completes at 0.886 XA with a stop at X (just 11.4% further). This means the stop is extremely close relative to the targets (0.382 and 0.618 AD retracements). The shallow B retracement (0.382-0.500) distinguishes it from the Gartley and ensures you're entering at a genuine deep retracement, not a moderate one.

---

### 245 | Market Cipher-Style Multi-Indicator Confluence
**School:** TradingView (Cipher-Inspired) | **Class:** Visual Confluence
**Timeframe:** Daily / 4H | **Assets:** All markets

**Mathematics:**
```
Multi-Signal Overlay (inspired by Market Cipher B):

Component 1: Wave Trend (WT)
  WT = WaveTrend oscillator (enhanced CCI-based)
  WT1 = EMA(EMA((Typical_Price - SMA(TP, n1)), n1) / (0.015 * mean_dev), n2)
  WT2 = SMA(WT1, 4)
  Buy: WT1 crosses above WT2 from oversold (< -60)
  Sell: WT1 crosses below WT2 from overbought (> +60)

Component 2: Momentum Dots
  If WT1 > WT1[1] AND WT1 > WT1[2]: green dot (positive momentum)
  If WT1 < WT1[1] AND WT1 < WT1[2]: red dot (negative momentum)

Component 3: VWAP Deviation
  VWAP_Dev = (Close - VWAP) / VWAP * 100
  Above VWAP = bullish; Below = bearish

Component 4: MFI (Money Flow)
  MFI colored: green if accumulation, red if distribution

Combined Signal:
  Bullish Confluence = WT_buy AND green_dots AND above_VWAP AND MFI_green
  Bearish Confluence = WT_sell AND red_dots AND below_VWAP AND MFI_red
  
  Confluence_Count = count of aligned components (0-4)
```

**Signal:**
- **Strong buy:** Confluence_Count = 4 (all components bullish)
- **Moderate buy:** Confluence_Count = 3
- **No trade:** Confluence_Count <= 2 (insufficient agreement)
- **Strong sell:** All 4 bearish

**Risk:** Stop at WT oversold/overbought level reversal; Target at WT extreme; Risk 1%
**Edge:** Multi-indicator confluence (WaveTrend + momentum + VWAP + money flow) creates a high-probability signal by requiring agreement across independent analysis dimensions: oscillator timing (WT), momentum direction (dots), institutional positioning (VWAP), and money flow (MFI). When all four agree, the probability of a successful trade exceeds 70%, significantly better than any individual component. The visual overlay format (popular on TradingView) makes the confluence immediately apparent.

---

### 246 | Wyckoff Accumulation/Distribution Schematic
**School:** New York (Richard Wyckoff) | **Class:** Market Cycle
**Timeframe:** Daily / Weekly | **Assets:** Equities

**Mathematics:**
```
Wyckoff Accumulation Phases:
  Phase A: Stopping action
    PS (Preliminary Support): first significant buying in decline
    SC (Selling Climax): panic selling, volume spike, wide spread
    AR (Automatic Rally): rebound from SC (buying response)
    ST (Secondary Test): retest of SC area on lighter volume

  Phase B: Building the Cause
    Multiple tests of support and resistance within SC-AR range
    SOW (Signs of Weakness): brief breaks below support (testing supply)
    SOS (Signs of Strength): advances toward resistance with increasing volume

  Phase C: Test (Spring)
    Spring: price drops below Phase B support, triggers stop losses
    Then REVERSES sharply back above support on volume
    This is the definitive buy signal (supply exhausted)

  Phase D: Markup
    SOS with increasing volume (strong demand)
    Price breaks above resistance of accumulation range
    BU (Back-Up): last pullback to resistance (now support) = final buy opportunity

  Phase E: Uptrend
    Higher highs, higher lows, trending

Phase Detection Algorithm:
  Detect SC (volume spike + wide range + bounce)
  Detect AR (rebound)
  Detect ST (retest of SC on lower volume)
  Monitor for Spring (break below ST + sharp reversal)
  Confirm SOS (above AR on volume)
```

**Signal:**
- **Buy at Spring:** Price breaks below accumulation range then reverses sharply on volume
- **Buy at BU:** After SOS, price pulls back to broken resistance (Phase D)
- **Sell (distribution):** Mirror schematic inverted (UTAD = Upthrust After Distribution)
- **Exit:** Signs of distribution appear (volume divergence, narrowing range at top)

**Risk:** Stop below Spring low; Target at measured move (accumulation range projected up); Risk 1.5%
**Edge:** Wyckoff's accumulation/distribution schematics map the ENTIRE institutional lifecycle: stopping the decline (Phase A), building a position (Phase B), testing supply (Phase C Spring), marking up (Phase D), and trending (Phase E). Understanding which phase the market is in tells you WHO is active (institutions or retail) and WHAT is likely to happen next. The Spring is the highest-probability entry because it represents the final test where supply is definitively proven to be exhausted.

---

### 247 | Measured Move (Swing Projection)
**School:** Classic TA | **Class:** Symmetry Projection
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Measured Move (AB=CD):
  First Swing (AB): price moves from A to B
  Correction (BC): price retraces portion of AB
  Second Swing (CD): price moves from C, projected to equal AB

  Target D = C + (B - A)  (AB distance added to C)

Alternate Projections:
  Conservative: D = C + 0.618 * (B - A)  (partial measured move)
  Standard: D = C + 1.000 * (B - A)  (equality)
  Extended: D = C + 1.272 * (B - A) or 1.618 * (B - A)

  Use BC retracement to estimate CD extension:
    BC = 0.382 of AB -> CD tends to extend to 1.272 of AB from C
    BC = 0.500 of AB -> CD tends to equal AB from C
    BC = 0.618 of AB -> CD tends to reach 0.786 of AB from C

Time Symmetry:
  Time(CD) / Time(AB) should be between 0.618 and 1.618
  Time equality (1.0) is the most common

Measured Move Success Rate:
  Standard (D = C + AB): 62% reach target (Bulkowski)
  With volume confirmation: 68%
  With Fibonacci BC alignment: 70%
```

**Signal:**
- **Target projection:** After AB move and BC correction, project CD target = C + AB
- **Entry:** At start of CD leg (after BC correction complete, confirmed by reversal)
- **Exit:** At projected D level (take profit at measured move target)
- **Stop:** Below C (correction low for bullish) or above C (for bearish)

**Risk:** Stop at C; Target at D; Typically 1.5:1 to 2:1 R:R; Risk 1%
**Edge:** Measured moves exploit the market's tendency toward SYMMETRY -- the second leg of a move tends to equal the first in both price and time. This is not mystical; it reflects institutional execution behavior: the same order flow that created AB tends to re-engage after the correction, producing a similar-sized move. The BC retracement depth provides refinement on the CD extension target.

---

### 248 | Rising/Falling Window (Japanese Gap Trading)
**School:** Tokyo (Traditional Candlestick) | **Class:** Gap Continuation
**Timeframe:** Daily | **Assets:** All markets

**Mathematics:**
```
Rising Window (Bullish Gap):
  Low_t > High_{t-1}  (today's low is above yesterday's high = gap up)
  Window = gap area = [High_{t-1}, Low_t]

Falling Window (Bearish Gap):
  High_t < Low_{t-1}  (today's high is below yesterday's low = gap down)
  Window = gap area = [High_t, Low_{t-1}]

Japanese Interpretation:
  "Window" = support/resistance zone
  Rising window: the gap becomes SUPPORT (price should hold above)
  Falling window: the gap becomes RESISTANCE (price should stay below)

Trading Rules:
  1. In uptrend: Rising window = continuation buy signal
     Buy on first pullback to window area (gap support)
  2. "A window will be closed" - Japanese proverb
     Eventually price returns to fill the gap, but the FIRST test of the window
     is usually a bounce (70% probability of initial test holding)

Window Quality:
  Gap_Size = |Low_t - High_{t-1}| / ATR
  Large window (> 1.5 ATR): Strong institutional gap (more likely to hold)
  Small window (< 0.3 ATR): Weak gap (likely to fill quickly)
  
  Volume at gap: if gap day volume > 2x average = institutional conviction
```

**Signal:**
- **Buy at rising window:** First pullback to window zone in uptrend (gap support)
- **Sell at falling window:** First rally to window zone in downtrend (gap resistance)
- **Quality filter:** Gap_Size > 0.5 ATR AND gap day volume > 1.5x average
- **Exit:** Window fills (gap closes) = thesis invalidated OR target 2x ATR

**Risk:** Stop below window (for rising window trades); Target 2x window size; Risk 1%
**Edge:** Japanese gap analysis treats gaps as ZONES of support/resistance, not just price levels. The first test of a gap zone (window) holds ~70% of the time because the institutional order flow that created the gap leaves unfilled orders within the gap zone. Large gaps (> 1.5 ATR) on high volume are the highest quality because they represent genuine institutional repricing. The Japanese "window will close" proverb correctly identifies that gaps eventually fill, but the first test is a high-probability continuation trade.

---

### 249 | Broadening Formation (Megaphone) Strategy
**School:** Classic TA | **Class:** Expanding Volatility Pattern
**Timeframe:** Daily / Weekly | **Assets:** Equities, Indices

**Mathematics:**
```
Broadening Formation:
  Upper trendline: connecting higher highs (expanding upward)
  Lower trendline: connecting lower lows (expanding downward)
  = EXPANDING range (opposite of triangle)

Detection:
  At least 5 alternating touches (3 on one side, 2 on other minimum)
  Each touch creates a wider range than the prior
  
  Upper_Slope > 0 AND Lower_Slope < 0 (diverging trendlines)
  |Upper_Slope - Lower_Slope| increasing (acceleration of expansion)

Characteristics:
  Appears at market tops (increasing volatility and indecision)
  Rare pattern (appears in <5% of charts)
  Very unreliable directional prediction (50/50 breakout direction)
  BUT: excellent for range trading WITHIN the pattern

Trading the Pattern:
  Buy: at lower trendline (touch #3, #5, etc.)
  Sell: at upper trendline (touch #2, #4, etc.)
  Each swing within the pattern is LARGER than the prior = bigger targets

  R:R improves with each successive touch (wider range = bigger opportunity)
```

**Signal:**
- **Buy at lower bound:** Price touches expanding lower trendline (bounce into pattern)
- **Sell at upper bound:** Price touches expanding upper trendline (fade into pattern)
- **Breakout:** After 5+ touches, eventual breakout is POWERFUL (stored energy massive)
- **Avoid:** Predicting breakout direction (50/50 odds)

**Risk:** Stop 1% beyond trendline; Target at opposite trendline; Risk 1.5%
**Edge:** Broadening formations are dangerous because volatility is EXPANDING (increasing uncertainty). Most traders avoid them. But trading WITHIN the pattern (buying low boundary, selling high boundary) exploits the pattern's defining characteristic: each swing is bigger than the last, providing progressively larger profit targets. The key insight is that broadening patterns cannot expand forever -- eventual breakout produces a powerful move because the expanding volatility has built enormous stored energy.

---

### 250 | Multi-Timeframe Pattern Confluence
**School:** Quantitative | **Class:** MTF Pattern Analysis
**Timeframe:** Weekly + Daily + 4H | **Assets:** All markets

**Mathematics:**
```
Pattern Detection across 3 timeframes:
  Weekly: Large structural patterns (H&S, double tops, accumulation)
  Daily: Intermediate patterns (flags, wedges, triangles)
  4H: Entry patterns (pin bars, engulfing, inside bars)

Confluence Matrix:
  For each pattern detected, assign:
    Direction: +1 (bullish) or -1 (bearish)
    Reliability: from backtested statistics [0, 1]
    Timeframe_Weight: Weekly = 3x, Daily = 2x, 4H = 1x

  MTF_Score = sum(Direction_i * Reliability_i * TF_Weight_i) for all patterns

  MTF_Score > +4: Strong multi-timeframe bullish confluence
  MTF_Score < -4: Strong multi-timeframe bearish confluence
  |MTF_Score| < 2: Insufficient confluence

Ideal Setup:
  Weekly: Major support + bullish pattern (e.g., Wyckoff accumulation spring)
  Daily: Continuation pattern (e.g., bull flag) within weekly structure
  4H: Entry pattern (e.g., bullish engulfing at daily support)
  
  This triple alignment = highest probability trade setup

Success Rate by Alignment:
  3-timeframe alignment: ~72% win rate
  2-timeframe alignment: ~58% win rate
  1-timeframe only: ~50% (no edge)
```

**Signal:**
- **Enter:** MTF_Score > +4 (strong 3-TF alignment) with 4H entry pattern
- **Skip:** |MTF_Score| < 2 (insufficient multi-timeframe agreement)
- **Position size:** Proportional to |MTF_Score| (stronger confluence = larger position)
- **Exit:** Weekly pattern target reached OR daily pattern failure

**Risk:** Stop based on smallest timeframe pattern; Target based on largest; Risk 1-2%
**Edge:** The most powerful concept in technical analysis is multi-timeframe confluence: when the same directional bias appears across weekly, daily, and intraday timeframes, each timeframe represents a different GROUP of market participants (institutional for weekly, swing for daily, retail for 4H) all acting in the same direction. This convergence of participant groups creates overwhelming order flow in one direction. The 72% win rate for 3-TF alignment vs. 50% for single-TF illustrates the statistical power of this approach.

---

# SECTION VI: OSCILLATOR AND CYCLE STRATEGIES (251-300)

---

"""

with open("Indicators.md", "a") as f:
    f.write(content)

print("Appended strategies 201-250 to Indicators.md")
