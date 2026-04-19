"""
SECTION V: PATTERN & FRACTAL STRATEGIES (201-250)
Each function takes ind dict -> pd.Series[-100, +100].
"""

import numpy as np
import pandas as pd


def _safe(s, fill=0.0):
    return s.fillna(fill) if isinstance(s, pd.Series) else pd.Series(s).fillna(fill)


def _norm(s, span=60):
    lo = s.rolling(span, min_periods=10).min()
    hi = s.rolling(span, min_periods=10).max()
    rng = (hi - lo).replace(0, np.nan)
    return ((s - lo) / rng * 2 - 1).clip(-1, 1).fillna(0)


def _z(s, span=60):
    m = s.rolling(span, min_periods=10).mean()
    sd = s.rolling(span, min_periods=10).std().replace(0, np.nan)
    return ((s - m) / sd).clip(-4, 4).fillna(0)


def _clip_signal(raw, smooth=3):
    return raw.clip(-100, 100).ewm(span=smooth, adjust=False).mean()


# ═════════════════════════════════════════════════════════════════════════════

def s201_fractal_dimension(ind):
    """201 | Fractal Dimension Index (FDI) Regime Filter"""
    hurst = ind["hurst"]
    fdi = 2 - hurst  # FDI ~ 1.0 = trending, ~1.5 = random, ~2.0 = mean-revert
    trend_regime = (fdi < 1.4).astype(float)
    mr_regime = (fdi > 1.6).astype(float)
    close = ind["close"]
    z = _z(close, 40)
    raw = trend_regime * ind["trend_score"] * 50 - mr_regime * z * 40
    return _clip_signal(raw)


def s202_three_drive(ind):
    """202 | Three-Drive Harmonic Pattern"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Detect 3 descending highs (bearish) or 3 ascending lows (bullish)
    h5 = high.rolling(5).max()
    l5 = low.rolling(5).min()
    bull = ((l5 > l5.shift(10)) & (l5.shift(10) > l5.shift(20)) & (l5.shift(20) > l5.shift(30))).astype(float)
    bear = ((h5 < h5.shift(10)) & (h5.shift(10) < h5.shift(20)) & (h5.shift(20) < h5.shift(30))).astype(float)
    raw = bull * 50 - bear * 50 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s203_head_shoulders(ind):
    """203 | Head and Shoulders Quantified Detection"""
    close = ind["close"]
    high = ind["high"]
    # Simplified: middle peak higher than surrounding peaks
    h10 = high.rolling(10).max()
    left_shoulder = h10.shift(20)
    head = h10.shift(10)
    right_shoulder = h10
    hs_top = ((head > left_shoulder) & (head > right_shoulder) &
              (right_shoulder < left_shoulder * 1.02)).astype(float)
    hs_bottom = False  # inverse for bottoms
    low_10 = ind["low"].rolling(10).min()
    left_s_low = low_10.shift(20)
    head_low = low_10.shift(10)
    right_s_low = low_10
    hs_bottom = ((head_low < left_s_low) & (head_low < right_s_low) &
                 (right_s_low > left_s_low * 0.98)).astype(float)
    raw = hs_bottom * 50 - hs_top * 50
    return _clip_signal(raw, smooth=5)


def s204_gartley_222(ind):
    """204 | Gartley 222 Harmonic Pattern"""
    close = ind["close"]
    # Fibonacci retracement proxy: detect 0.618 retracement levels
    swing_high = ind["hh"]
    swing_low = ind["ll"]
    rng = swing_high - swing_low
    retrace = (swing_high - close) / rng.replace(0, np.nan)
    near_618 = ((retrace > 0.58) & (retrace < 0.68)).astype(float)
    near_382 = ((retrace > 0.35) & (retrace < 0.42)).astype(float)
    bullish = near_618 * (ind["rsi"] < 40).astype(float)
    bearish = near_382 * (ind["rsi"] > 60).astype(float)
    raw = bullish * 50 - bearish * 50
    return _clip_signal(raw)


def s205_elliott_wave(ind):
    """205 | Elliott Wave Automated Counting"""
    close = ind["close"]
    mom_5 = ind["mom_5"]
    mom_20 = ind["mom_20"]
    # Wave proxy: momentum direction changes
    trend_up = (mom_20 > 0).astype(float)
    impulse = (mom_5 > 0) & trend_up  # wave 3/5 proxy
    correction = (mom_5 < 0) & trend_up  # wave 2/4 proxy
    # Buy on wave 2/4 corrections in uptrend
    raw = correction.astype(float) * 40 - ((mom_5 > 0) & (~trend_up)).astype(float) * 40
    raw = raw + ind["trend_score"] * 25
    return _clip_signal(raw)


def s206_heikin_ashi(ind):
    """206 | Heikin-Ashi Trend Persistence Filter"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    ha_close = (open_p + high + low + close) / 4
    ha_open = close.copy().astype(float)
    vals = ha_close.values.astype(float)
    o = open_p.values.astype(float)
    ho = np.zeros(len(vals))
    ho[0] = o[0]
    for i in range(1, len(vals)):
        ho[i] = (ho[i-1] + vals[i-1]) / 2
    ha_open = pd.Series(ho, index=close.index)
    ha_bull = (ha_close > ha_open).astype(float)
    # Persistence: consecutive HA candles in same direction
    persist = ha_bull.rolling(5).mean()
    raw = (persist - 0.5) * 120
    return _clip_signal(raw)


def s207_renko_pattern(ind):
    """207 | Renko Brick Pattern Recognition"""
    close = ind["close"]
    atr = ind["atr14"]
    brick = atr.rolling(20).mean()
    # Renko proxy: count net direction of ATR-sized moves
    ret = close.diff()
    bricks = (ret / brick.replace(0, np.nan)).fillna(0)
    net_bricks = bricks.rolling(10).sum()
    z = _z(net_bricks, 40)
    raw = z * 55
    return _clip_signal(raw)


def s208_fib_cluster(ind):
    """208 | Fibonacci Cluster Zone Strategy"""
    close = ind["close"]
    high = ind["hh"]
    low = ind["ll"]
    rng = high - low
    # Multiple fib levels
    fib_382 = low + 0.382 * rng
    fib_500 = low + 0.500 * rng
    fib_618 = low + 0.618 * rng
    # Cluster: price near multiple fib levels
    near_fib = sum([(close - level).abs() < ind["atr14"] * 0.5
                    for level in [fib_382, fib_500, fib_618]])
    cluster = (near_fib >= 2).astype(float)
    direction = np.sign(close - fib_500)
    raw = cluster * (-direction) * 40 + (1 - cluster) * ind["trend_score"] * 30
    return _clip_signal(raw)


def s209_candle_ensemble(ind):
    """209 | Japanese Candlestick Ensemble Classifier"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    body = close - open_p
    upper = high - pd.concat([close, open_p], axis=1).max(axis=1)
    lower = pd.concat([close, open_p], axis=1).min(axis=1) - low
    atr = ind["atr14"]
    # Hammer
    hammer = ((lower > 2 * body.abs()) & (upper < body.abs() * 0.5) & (ind["rsi"] < 35)).astype(float)
    # Engulfing
    bull_eng = ((body > 0) & (body.shift(1) < 0) & (body.abs() > body.shift(1).abs() * 1.5)).astype(float)
    bear_eng = ((body < 0) & (body.shift(1) > 0) & (body.abs() > body.shift(1).abs() * 1.5)).astype(float)
    # Morning/Evening star
    small_body = (body.abs() < atr * 0.3).astype(float)
    morning = (small_body.shift(1) * (body > 0) * (body.shift(2) < 0)).astype(float)
    evening = (small_body.shift(1) * (body < 0) * (body.shift(2) > 0)).astype(float)
    raw = (hammer + bull_eng + morning) * 25 - (bear_eng + evening) * 25 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s210_wolfe_wave(ind):
    """210 | Wolfe Wave Projection"""
    close = ind["close"]
    # Wedge proxy: converging highs and lows
    h20 = ind["high"].rolling(20).max()
    l20 = ind["low"].rolling(20).min()
    width = h20 - l20
    width_change = width.diff(10)
    converging = (width_change < 0).astype(float)
    diverging = (width_change > 0).astype(float)
    # Converging wedge near completion = reversal
    raw = converging * (ind["rsi"] - 50) / 50 * -40 + diverging * ind["trend_score"] * 30
    return _clip_signal(raw)


def s211_linreg_channel(ind):
    """211 | Linear Regression Channel Breakout"""
    close = ind["close"]
    n = 20
    x = np.arange(n, dtype=float)
    slope = close.rolling(n).apply(lambda y: np.polyfit(x, y, 1)[0] if len(y) == n else 0, raw=True)
    intercept = close.rolling(n).apply(lambda y: np.polyfit(x, y, 1)[1] if len(y) == n else y.mean(), raw=True)
    pred = slope * (n - 1) + intercept
    resid = close - pred
    std = resid.rolling(n).std().replace(0, np.nan)
    z = resid / std
    # Breakout: beyond 2 std channel
    breakout = z.clip(-3, 3) / 3 * 50
    raw = breakout + slope / close * 1e4 * 20
    return _clip_signal(raw)


def s212_cup_handle(ind):
    """212 | Cup and Handle Quantified Detection"""
    close = ind["close"]
    high = ind["high"]
    # Cup: round bottom over 30+ bars
    h60 = high.rolling(60).max()
    l30 = ind["low"].rolling(30, center=True).min()
    # Handle: small pullback near high
    near_high = (close > h60 * 0.95).astype(float)
    pullback = (close < close.rolling(5).max() * 0.98).astype(float)
    handle = near_high * pullback
    breakout = (close > h60).astype(float)
    raw = handle * 30 + breakout * 40
    return _clip_signal(raw)


def s213_ichimoku_multi(ind):
    """213 | Ichimoku Cloud Multi-Signal System"""
    close = ind["close"]
    tk = ind["ichi_tk"]
    above_cloud = ind["ichi_above_cloud"]
    sma_50 = ind["sma_50"]
    # TK cross
    tenkan = close.ewm(span=9, adjust=False).mean()
    kijun = close.ewm(span=26, adjust=False).mean()
    tk_cross = (tenkan > kijun).astype(float) - (tenkan < kijun).astype(float)
    # Combine signals
    raw = above_cloud * 25 + tk_cross * 25 + (close > sma_50).astype(float) * 20
    return _clip_signal(raw)


def s214_butterfly_harmonic(ind):
    """214 | Butterfly Harmonic Pattern"""
    close = ind["close"]
    swing_high = ind["hh"]
    swing_low = ind["ll"]
    rng = swing_high - swing_low
    retrace = (close - swing_low) / rng.replace(0, np.nan)
    # Butterfly: 1.272 or 1.618 extension
    ext_127 = ((retrace > 1.20) & (retrace < 1.32)).astype(float)
    ext_162 = ((retrace > 1.55) & (retrace < 1.68)).astype(float)
    # Extensions = potential reversal zones
    raw = -(ext_127 + ext_162) * 35 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s215_market_structure_break(ind):
    """215 | Market Structure Break (CHoCH/BOS)"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Break of structure: higher high / lower low violations
    prev_high = high.rolling(10).max().shift(1)
    prev_low = low.rolling(10).min().shift(1)
    bos_up = (close > prev_high).astype(float)
    bos_dn = (close < prev_low).astype(float)
    # CHoCH: change of character (break opposite to trend)
    trend = ind["above_50"].fillna(0.5)
    choch_bull = bos_up * (1 - trend)  # break up while below 50MA
    choch_bear = bos_dn * trend  # break down while above 50MA
    raw = (bos_up - bos_dn) * 30 + (choch_bull - choch_bear) * 25
    return _clip_signal(raw)


def s216_abcd_pattern(ind):
    """216 | AB=CD Measured Move Pattern"""
    close = ind["close"]
    # Detect equal-length swings
    swing = close.diff(10)
    prev_swing = swing.shift(20)
    ratio = swing / prev_swing.replace(0, np.nan)
    abcd_buy = ((ratio > 0.85) & (ratio < 1.15) & (swing < 0)).astype(float)
    abcd_sell = ((ratio > 0.85) & (ratio < 1.15) & (swing > 0)).astype(float)
    raw = abcd_buy * 45 - abcd_sell * 45
    return _clip_signal(raw)


def s217_island_reversal(ind):
    """217 | Island Reversal Gap Pattern"""
    close = ind["close"]
    open_p = ind["open"]
    # Gap up followed by gap down (or vice versa)
    gap_up = (open_p > close.shift(1) * 1.005).astype(float)
    gap_dn = (open_p < close.shift(1) * 0.995).astype(float)
    island_top = gap_up.shift(3).rolling(5).max() * gap_dn
    island_bottom = gap_dn.shift(3).rolling(5).max() * gap_up
    raw = island_bottom * 55 - island_top * 55
    return _clip_signal(raw, smooth=5)


def s218_pivot_fib(ind):
    """218 | Pivot Point Fibonacci Confluence"""
    high = ind["high"].shift(1)
    low = ind["low"].shift(1)
    close = ind["close"].shift(1)
    pivot = (high + low + close) / 3
    rng = high - low
    r1 = pivot + 0.382 * rng
    s1 = pivot - 0.382 * rng
    current = ind["close"]
    dist_r = (current - r1) / ind["atr14"].replace(0, np.nan)
    dist_s = (current - s1) / ind["atr14"].replace(0, np.nan)
    near_support = (dist_s.abs() < 0.5).astype(float) * (ind["rsi"] < 40).astype(float)
    near_resist = (dist_r.abs() < 0.5).astype(float) * (ind["rsi"] > 60).astype(float)
    raw = near_support * 45 - near_resist * 45
    return _clip_signal(raw)


def s219_kagi_reversal(ind):
    """219 | Kagi Chart Trend Reversal System"""
    close = ind["close"]
    atr = ind["atr14"]
    threshold = atr.rolling(20).mean()
    # Kagi reversal: direction change of threshold-sized move
    ret_cum = close.diff().rolling(10).sum()
    reversal_up = ((ret_cum > threshold) & (ret_cum.shift(10) < -threshold)).astype(float)
    reversal_dn = ((ret_cum < -threshold) & (ret_cum.shift(10) > threshold)).astype(float)
    raw = reversal_up * 50 - reversal_dn * 50
    return _clip_signal(raw, smooth=5)


def s220_williams_fractal(ind):
    """220 | Fractal Breakout (Bill Williams)"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    # Up fractal: high[i] > high[i-1], high[i-2] AND high[i] > high[i+1], high[i+2]
    # Using shifted data
    up_frac = ((high.shift(2) > high.shift(4)) & (high.shift(2) > high.shift(3)) &
               (high.shift(2) > high.shift(1)) & (high.shift(2) > high)).astype(float)
    dn_frac = ((low.shift(2) < low.shift(4)) & (low.shift(2) < low.shift(3)) &
               (low.shift(2) < low.shift(1)) & (low.shift(2) < low)).astype(float)
    # Breakout above up fractal / below down fractal
    last_up = high.shift(2) * up_frac
    last_up = last_up.replace(0, np.nan).ffill()
    last_dn = low.shift(2) * dn_frac
    last_dn = last_dn.replace(0, np.nan).ffill()
    raw = (close > last_up).astype(float) * 45 - (close < last_dn).astype(float) * 45
    return _clip_signal(raw)


def s221_pnf_reversal(ind):
    """221 | Point and Figure Column Reversal"""
    close = ind["close"]
    atr = ind["atr14"]
    box = atr.rolling(20).mean()
    # PnF proxy: count direction reversals
    ret = close.diff()
    boxes = (ret / box.replace(0, np.nan)).fillna(0)
    # 3-box reversal signal
    up_run = (boxes > 0).astype(float).rolling(3).sum()
    dn_run = (boxes < 0).astype(float).rolling(3).sum()
    raw = (up_run >= 3).astype(float) * 45 - (dn_run >= 3).astype(float) * 45
    return _clip_signal(raw)


def s222_fvg(ind):
    """222 | Fair Value Gap (FVG) Strategy"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # FVG: gap between high of bar N-2 and low of bar N
    bullish_fvg = (low > high.shift(2)).astype(float)  # gap up
    bearish_fvg = (high < low.shift(2)).astype(float)  # gap down
    # Price returns to fill FVG
    fill_bull = ((close < low.shift(1)) & bullish_fvg.shift(1).astype(bool)).astype(float)
    fill_bear = ((close > high.shift(1)) & bearish_fvg.shift(1).astype(bool)).astype(float)
    raw = fill_bull * 40 - fill_bear * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s223_pitchfork(ind):
    """223 | Andrews Pitchfork Median Line"""
    close = ind["close"]
    # Median line proxy: midpoint of high/low channel
    median = (ind["hh"] + ind["ll"]) / 2
    dist = (close - median) / ind["atr14"].replace(0, np.nan)
    # Price reverts to median line
    raw = -dist.clip(-3, 3) / 3 * 50 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s224_darvas_box(ind):
    """224 | Darvas Box Breakout System"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Box top/bottom: 20-day high/low
    box_top = high.rolling(20).max()
    box_bot = low.rolling(20).min()
    # Breakout
    break_up = (close > box_top.shift(1)).astype(float)
    break_dn = (close < box_bot.shift(1)).astype(float)
    vol_conf = (ind["vol_rel"] > 1.3).astype(float)
    raw = break_up * vol_conf * 55 - break_dn * vol_conf * 55
    return _clip_signal(raw)


def s225_flag_pennant(ind):
    """225 | Pennant/Flag Continuation Quantified"""
    close = ind["close"]
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    bb_width = ind["bb_width"]
    # Flag: strong move followed by consolidation
    strong_move = (mom.abs() > mom.rolling(60).std() * 1.5).astype(float).shift(5)
    consolidation = (bb_width < bb_width.rolling(60).quantile(0.25)).astype(float)
    direction = np.sign(mom.shift(5))
    raw = strong_move * consolidation * direction * 55
    return _clip_signal(raw)


def s226_fib_fan(ind):
    """226 | Fibonacci Speed Resistance Fan"""
    close = ind["close"]
    swing_high = ind["hh"]
    swing_low = ind["ll"]
    rng = swing_high - swing_low
    # Fan levels at 38.2%, 50%, 61.8%
    levels = [0.382, 0.500, 0.618]
    support_count = sum([(close > swing_low + l * rng - ind["atr14"] * 0.3).astype(float)
                        for l in levels])
    raw = (support_count / 3 - 0.5) * 80
    return _clip_signal(raw)


def s227_mom_divergence_multi(ind):
    """227 | Momentum Divergence Multi-Indicator"""
    close = ind["close"]
    rsi = ind["rsi"]
    macd = ind["macd_n"]
    mom = ind["mom_10"]
    price_slope = close.diff(10)
    rsi_div = _z(rsi.diff(10), 40) - _z(price_slope, 40)
    macd_div = _z(macd.diff(10), 40) - _z(price_slope, 40)
    mom_div = _z(mom.diff(10), 40) - _z(price_slope, 40)
    # Multiple divergences = stronger signal
    raw = (rsi_div + macd_div + mom_div) / 3 * 45
    return _clip_signal(raw)


def s228_liquidity_sweep(ind):
    """228 | London Session Liquidity Sweep"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Sweep: price breaks prior high/low then reverses
    prev_high = high.rolling(5).max().shift(1)
    prev_low = low.rolling(5).min().shift(1)
    sweep_high = ((high > prev_high) & (close < prev_high)).astype(float)
    sweep_low = ((low < prev_low) & (close > prev_low)).astype(float)
    raw = sweep_low * 50 - sweep_high * 50
    return _clip_signal(raw)


def s229_triple_screen(ind):
    """229 | Triple Screen Enhanced (Elder)"""
    # Screen 1: weekly trend (using 50MA)
    trend = ind["above_50"].fillna(0.5)
    # Screen 2: daily oscillator (RSI)
    rsi = ind["rsi"]
    osc_buy = (rsi < 35) & (trend > 0.5)
    osc_sell = (rsi > 65) & (trend < 0.5)
    # Screen 3: entry (breakout confirmation)
    breakout_up = (ind["close"] > ind["high"].shift(1)).astype(float)
    breakout_dn = (ind["close"] < ind["low"].shift(1)).astype(float)
    raw = osc_buy.astype(float) * breakout_up * 60 - osc_sell.astype(float) * breakout_dn * 60
    return _clip_signal(raw)


def s230_engulfing_volume(ind):
    """230 | Engulfing Pattern with Volume Confirmation"""
    close = ind["close"]
    open_p = ind["open"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean()
    body = close - open_p
    bull_eng = ((body > 0) & (body.shift(1) < 0) & (body.abs() > body.shift(1).abs()) &
                (volume > 1.5 * vol_ma)).astype(float)
    bear_eng = ((body < 0) & (body.shift(1) > 0) & (body.abs() > body.shift(1).abs()) &
                (volume > 1.5 * vol_ma)).astype(float)
    raw = bull_eng * 55 - bear_eng * 55
    return _clip_signal(raw, smooth=5)


def s231_donch_squeeze(ind):
    """231 | Donchian Channel Width Squeeze"""
    donch_h = ind["donch_high"]
    donch_l = ind["donch_low"]
    close = ind["close"]
    width = (donch_h - donch_l) / close * 100
    width_z = _z(width, 60)
    squeeze = (width_z < -1.0).astype(float)
    breakout_dir = np.sign(close - (donch_h + donch_l) / 2)
    raw = squeeze * breakout_dir * 40 + ind["trend_score"] * ind["donch_pct"] * 30
    return _clip_signal(raw)


def s232_candle_sr(ind):
    """232 | Candlestick Pattern + S/R Confluence"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    open_p = ind["open"]
    body = close - open_p
    # S/R levels: recent 20-day high/low
    resistance = high.rolling(20).max()
    support = low.rolling(20).min()
    near_support = ((close - support) / ind["atr14"].replace(0, np.nan) < 1.0).astype(float)
    near_resist = ((resistance - close) / ind["atr14"].replace(0, np.nan) < 1.0).astype(float)
    bull_candle = (body > 0).astype(float)
    bear_candle = (body < 0).astype(float)
    raw = near_support * bull_candle * 50 - near_resist * bear_candle * 50
    return _clip_signal(raw)


def s233_gann_square(ind):
    """233 | Gann Square of Nine Price/Time"""
    close = ind["close"]
    # Gann angle proxy: price change per unit time
    angle = close.diff(10) / (10 * ind["atr14"].replace(0, np.nan))
    z = _z(angle, 60)
    raw = z * 50
    return _clip_signal(raw)


def s234_sr_strength(ind):
    """234 | Quantified Support/Resistance Strength"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    # S/R strength: volume at price levels
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    dist = (close - vwap) / ind["atr14"].replace(0, np.nan)
    touches = (dist.abs() < 0.5).astype(float).rolling(20).sum()
    strength = touches / 20
    raw = -dist * strength * 40 + ind["trend_score"] * (1 - strength) * 30
    return _clip_signal(raw)


def s235_doji_indecision(ind):
    """235 | Spinning Top and Doji Indecision Cluster"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    body = (close - open_p).abs()
    rng = (high - low).replace(0, np.nan)
    doji = (body / rng < 0.1).astype(float)
    spinning = ((body / rng < 0.3) & (body / rng >= 0.1)).astype(float)
    cluster = (doji + spinning).rolling(5).sum()
    # Indecision cluster -> expect breakout
    z = _z(cluster, 40)
    raw = z * ind["trend_score"] * 50
    return _clip_signal(raw)


def s236_gap_classification(ind):
    """236 | Gap Analysis with Classification"""
    close = ind["close"]
    open_p = ind["open"]
    gap = (open_p - close.shift(1)) / close.shift(1) * 100
    vol_rel = ind["vol_rel"]
    # Common gap (fade), breakaway gap (follow), exhaustion gap (fade)
    small_gap = (gap.abs() < 0.5).astype(float)
    large_gap = (gap.abs() > 1.5).astype(float)
    high_vol = (vol_rel > 1.5).astype(float)
    # Breakaway: large gap + high volume -> follow
    # Exhaustion: large gap + low volume -> fade
    breakaway = large_gap * high_vol * np.sign(gap)
    exhaustion = large_gap * (1 - high_vol) * (-np.sign(gap))
    raw = breakaway * 45 + exhaustion * 30 + small_gap * (-np.sign(gap)) * 20
    return _clip_signal(raw)


def s237_double_top_bottom(ind):
    """237 | Double Top/Bottom Quantified"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Double top: two highs within 2% over 20-60 bars
    h20 = high.rolling(20).max()
    h40 = high.shift(20).rolling(20).max()
    double_top = ((h20 / h40 - 1).abs() < 0.02).astype(float) * (close < h20 * 0.97).astype(float)
    l20 = low.rolling(20).min()
    l40 = low.shift(20).rolling(20).min()
    double_bottom = ((l20 / l40 - 1).abs() < 0.02).astype(float) * (close > l20 * 1.03).astype(float)
    raw = double_bottom * 50 - double_top * 50
    return _clip_signal(raw, smooth=5)


def s238_triangle(ind):
    """238 | Descending/Ascending Triangle Quantified"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    h_slope = high.rolling(20).max().diff(10)
    l_slope = low.rolling(20).min().diff(10)
    # Ascending: flat top, rising bottom
    ascending = ((h_slope.abs() < ind["atr14"] * 0.1) & (l_slope > 0)).astype(float)
    # Descending: flat bottom, falling top
    descending = ((l_slope.abs() < ind["atr14"] * 0.1) & (h_slope < 0)).astype(float)
    raw = ascending * 45 - descending * 45
    return _clip_signal(raw)


def s239_wedge_pattern(ind):
    """239 | Wedge Pattern (Rising/Falling) Detection"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    h_slope = high.rolling(20).max().diff(10)
    l_slope = low.rolling(20).min().diff(10)
    # Rising wedge (bearish): both slopes up, converging
    rising = ((h_slope > 0) & (l_slope > 0) & (h_slope < l_slope)).astype(float)
    # Falling wedge (bullish): both slopes down, converging
    falling = ((h_slope < 0) & (l_slope < 0) & (h_slope > l_slope)).astype(float)
    raw = falling * 50 - rising * 50
    return _clip_signal(raw)


def s240_pivot_reversal(ind):
    """240 | Pivot Reversal with Confluence Scoring"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    r1 = 2 * pivot - low.shift(1)
    s1 = 2 * pivot - high.shift(1)
    near_s1 = ((close - s1).abs() < ind["atr14"] * 0.5).astype(float)
    near_r1 = ((close - r1).abs() < ind["atr14"] * 0.5).astype(float)
    rsi_oversold = (ind["rsi"] < 35).astype(float)
    rsi_overbought = (ind["rsi"] > 65).astype(float)
    raw = near_s1 * rsi_oversold * 55 - near_r1 * rsi_overbought * 55
    return _clip_signal(raw)


def s241_inside_bar(ind):
    """241 | Inside Bar Breakout Strategy"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    inside = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)
    # Breakout direction
    break_up = inside.shift(1) * (close > high.shift(1)).astype(float)
    break_dn = inside.shift(1) * (close < low.shift(1)).astype(float)
    raw = break_up * 55 - break_dn * 55
    return _clip_signal(raw, smooth=5)


def s242_pin_bar(ind):
    """242 | Pin Bar (Hammer/Shooting Star) Quality Scoring"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    body = (close - open_p).abs()
    upper = high - pd.concat([close, open_p], axis=1).max(axis=1)
    lower = pd.concat([close, open_p], axis=1).min(axis=1) - low
    rng = (high - low).replace(0, np.nan)
    # Quality: tail > 2/3 of range, body < 1/3
    bull_pin = ((lower > 0.66 * rng) & (body < 0.33 * rng)).astype(float)
    bear_pin = ((upper > 0.66 * rng) & (body < 0.33 * rng)).astype(float)
    # Quality scoring: tail length
    bull_q = bull_pin * (lower / rng)
    bear_q = bear_pin * (upper / rng)
    raw = bull_q * 60 - bear_q * 60
    return _clip_signal(raw, smooth=5)


def s243_three_line_break(ind):
    """243 | Three Line Break Chart Pattern"""
    close = ind["close"]
    # Simplified: new high/low that breaks 3 prior closes
    c = close.values.astype(float)
    n = len(c)
    sig = np.zeros(n)
    for i in range(3, n):
        if c[i] > max(c[i-1], c[i-2], c[i-3]):
            sig[i] = 1
        elif c[i] < min(c[i-1], c[i-2], c[i-3]):
            sig[i] = -1
    raw = pd.Series(sig, index=close.index) * 50
    return _clip_signal(raw)


def s244_bat_harmonic(ind):
    """244 | Harmonic Bat Pattern"""
    close = ind["close"]
    swing_high = ind["hh"]
    swing_low = ind["ll"]
    rng = swing_high - swing_low
    retrace = (close - swing_low) / rng.replace(0, np.nan)
    # Bat: 0.886 retracement
    near_886 = ((retrace > 0.85) & (retrace < 0.92)).astype(float)
    near_50 = ((retrace > 0.45) & (retrace < 0.55)).astype(float)
    raw = -(near_886 * 40) + near_50 * ind["trend_score"] * 30
    return _clip_signal(raw)


def s245_market_cipher(ind):
    """245 | Market Cipher-Style Multi-Indicator Confluence"""
    rsi = ind["rsi"]
    macd_h = ind["macd_hist"]
    stoch = ind["stoch_k"]
    mom = ind["mom_10"]
    vol_damp = ind["vol_dampener"]
    # Combine oscillators with weighting
    rsi_norm = (rsi - 50) / 50
    stoch_norm = (stoch - 50) / 50
    macd_z = _z(macd_h, 40)
    mom_z = _z(mom, 40)
    raw = (rsi_norm * 20 + stoch_norm * 15 + macd_z * 20 + mom_z * 15) * vol_damp
    return _clip_signal(raw)


def s246_wyckoff_schematic(ind):
    """246 | Wyckoff Accumulation/Distribution Schematic"""
    close = ind["close"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    # Accumulation: decreasing volume at support, springs
    in_range = (ind["bb_pctb"] > 0.2) & (ind["bb_pctb"] < 0.8)
    low_vol = (volume < 0.7 * vol_ma).astype(float)
    near_support = (ind["bb_pctb"] < 0.2).astype(float)
    spring = near_support * (volume > 1.5 * vol_ma).astype(float) * (close > ind["open"]).astype(float)
    # Distribution: decreasing volume at resistance
    near_resist = (ind["bb_pctb"] > 0.8).astype(float)
    upthrust = near_resist * (volume > 1.5 * vol_ma).astype(float) * (close < ind["open"]).astype(float)
    raw = (spring * 55 - upthrust * 55 +
           in_range.astype(float) * low_vol * ind["trend_score"] * 15)
    return _clip_signal(raw)


def s247_measured_move(ind):
    """247 | Measured Move (Swing Projection)"""
    close = ind["close"]
    mom_20 = ind["mom_20"]
    mom_40 = ind["mom_40"]
    # Project: if first swing = second swing
    proj = mom_20 / mom_40.replace(0, np.nan)
    # Near 1.0 = measured move completing
    completing = ((proj > 0.8) & (proj < 1.2)).astype(float)
    direction = np.sign(mom_20)
    raw = completing * (-direction) * 35 + (1 - completing) * ind["trend_score"] * 30
    return _clip_signal(raw)


def s248_window_gap(ind):
    """248 | Rising/Falling Window (Japanese Gap Trading)"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    # Rising window: today's low > yesterday's high
    rising = (low > high.shift(1)).astype(float)
    # Falling window: today's high < yesterday's low
    falling = (high < low.shift(1)).astype(float)
    # Windows act as support/resistance
    raw = rising * 40 - falling * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s249_megaphone(ind):
    """249 | Broadening Formation (Megaphone) Strategy"""
    high = ind["high"]
    low = ind["low"]
    h20 = high.rolling(20).max()
    l20 = low.rolling(20).min()
    width = h20 - l20
    expanding = (width > width.shift(10)).astype(float)
    # Megaphone: trade the range, fade extremes
    close = ind["close"]
    mid = (h20 + l20) / 2
    dist = (close - mid) / (width / 2).replace(0, np.nan)
    raw = expanding * (-dist) * 40 + (1 - expanding) * ind["trend_score"] * 30
    return _clip_signal(raw)


def s250_mtf_pattern(ind):
    """250 | Multi-Timeframe Pattern Confluence"""
    close = ind["close"]
    # Short-term: 5-day pattern
    short_trend = np.sign(ind["mom_5"])
    # Medium-term: 20-day pattern
    med_trend = np.sign(ind["mom_20"])
    # Long-term: 50-day pattern
    long_trend = (ind["above_50"].fillna(0.5) * 2 - 1)
    # Confluence: all timeframes agree
    agreement = (short_trend + med_trend + long_trend) / 3
    strength = agreement.abs()
    vol_adj = ind["vol_dampener"]
    raw = agreement * strength * vol_adj * 70
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    201: ("Fractal Dimension Regime", s201_fractal_dimension),
    202: ("Three-Drive Harmonic", s202_three_drive),
    203: ("Head & Shoulders", s203_head_shoulders),
    204: ("Gartley 222", s204_gartley_222),
    205: ("Elliott Wave Auto", s205_elliott_wave),
    206: ("Heikin-Ashi Persistence", s206_heikin_ashi),
    207: ("Renko Pattern", s207_renko_pattern),
    208: ("Fibonacci Cluster", s208_fib_cluster),
    209: ("Candlestick Ensemble", s209_candle_ensemble),
    210: ("Wolfe Wave", s210_wolfe_wave),
    211: ("LinReg Channel", s211_linreg_channel),
    212: ("Cup and Handle", s212_cup_handle),
    213: ("Ichimoku Multi-Signal", s213_ichimoku_multi),
    214: ("Butterfly Harmonic", s214_butterfly_harmonic),
    215: ("Market Structure Break", s215_market_structure_break),
    216: ("AB=CD Pattern", s216_abcd_pattern),
    217: ("Island Reversal", s217_island_reversal),
    218: ("Pivot Fibonacci", s218_pivot_fib),
    219: ("Kagi Reversal", s219_kagi_reversal),
    220: ("Williams Fractal", s220_williams_fractal),
    221: ("Point & Figure", s221_pnf_reversal),
    222: ("Fair Value Gap", s222_fvg),
    223: ("Andrews Pitchfork", s223_pitchfork),
    224: ("Darvas Box", s224_darvas_box),
    225: ("Flag/Pennant", s225_flag_pennant),
    226: ("Fibonacci Fan", s226_fib_fan),
    227: ("Momentum Divergence Multi", s227_mom_divergence_multi),
    228: ("Liquidity Sweep", s228_liquidity_sweep),
    229: ("Triple Screen Elder", s229_triple_screen),
    230: ("Engulfing + Volume", s230_engulfing_volume),
    231: ("Donchian Squeeze", s231_donch_squeeze),
    232: ("Candlestick + SR", s232_candle_sr),
    233: ("Gann Square", s233_gann_square),
    234: ("SR Strength", s234_sr_strength),
    235: ("Doji Indecision", s235_doji_indecision),
    236: ("Gap Classification", s236_gap_classification),
    237: ("Double Top/Bottom", s237_double_top_bottom),
    238: ("Triangle Pattern", s238_triangle),
    239: ("Wedge Pattern", s239_wedge_pattern),
    240: ("Pivot Reversal", s240_pivot_reversal),
    241: ("Inside Bar Breakout", s241_inside_bar),
    242: ("Pin Bar Quality", s242_pin_bar),
    243: ("Three Line Break", s243_three_line_break),
    244: ("Bat Harmonic", s244_bat_harmonic),
    245: ("Market Cipher Multi", s245_market_cipher),
    246: ("Wyckoff Schematic", s246_wyckoff_schematic),
    247: ("Measured Move", s247_measured_move),
    248: ("Window Gap Trading", s248_window_gap),
    249: ("Megaphone Formation", s249_megaphone),
    250: ("MTF Pattern Confluence", s250_mtf_pattern),
}
