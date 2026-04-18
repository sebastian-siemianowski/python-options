"""Batch 12: Replace S056-S060 skeletal implementations with full versions."""
import sys

filepath = "src/indicators/families/mean_reversion.py"
with open(filepath, "r") as f:
    content = f.read()

# ─── S056 DeMark TD9 Exhaustion ─────────────────────────────────────────────
old_056 = '''def s056_demark_td9(ind):
    """056 | DeMark Sequential TD9 Exhaustion"""
    close = ind["close"]
    c = close.values.astype(float)
    n = len(c)
    setup = np.zeros(n)
    for i in range(4, n):
        if c[i] < c[i-4]:
            setup[i] = max(setup[i-1], 0) + 1 if setup[i-1] >= 0 else 1
        elif c[i] > c[i-4]:
            setup[i] = min(setup[i-1], 0) - 1 if setup[i-1] <= 0 else -1
        else:
            setup[i] = 0
    setup_s = pd.Series(setup, index=close.index)
    buy_ex = (setup_s >= 9).astype(float)
    sell_ex = (setup_s <= -9).astype(float)
    raw = -buy_ex * 60 + sell_ex * 60
    raw = raw + ind["mom_score"] * 20
    return _clip_signal(raw)'''

new_056 = '''def s056_demark_td9(ind):
    """056 | DeMark Sequential TD9 Exhaustion
    School: New York (Tom DeMark)
    
    TD Setup (9-count):
      Buy Setup:  9 consecutive bars where Close < Close_4_bars_ago
      Sell Setup: 9 consecutive bars where Close > Close_4_bars_ago
    
    TD Countdown (13-count):
      After setup, count bars where:
        Buy:  Close <= Low_2_bars_ago (13 such bars needed)
        Sell: Close >= High_2_bars_ago
    
    TD Perfection (qualifier):
      Buy:  Bar 8 or 9 low <= bar 6 or 7 low (ensures deep pullback)
      Sell: Bar 8 or 9 high >= bar 6 or 7 high
    
    Signal: Perfected Setup 9 = potential exhaustion reversal.
    Highest conviction: Setup 9 + Countdown 13 both complete.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    c = close.values.astype(float)
    h = high.values.astype(float)
    lo = low.values.astype(float)
    n = len(c)
    
    # --- TD Setup (9-count) ---
    setup = np.zeros(n)
    for i in range(4, n):
        if c[i] < c[i - 4]:
            prev = setup[i - 1]
            setup[i] = (prev + 1) if prev > 0 else 1
        elif c[i] > c[i - 4]:
            prev = setup[i - 1]
            setup[i] = (prev - 1) if prev < 0 else -1
        else:
            setup[i] = 0
    
    setup_s = pd.Series(setup, index=close.index)
    
    # --- TD Perfection ---
    # Buy perfection: at setup 8 or 9, the low <= low at setup 6 or 7
    buy_perfect = np.zeros(n)
    sell_perfect = np.zeros(n)
    for i in range(8, n):
        if setup[i] >= 8:
            # Check if bar 8 or 9 low <= bar 6 or 7 low
            setup_start = i - int(setup[i]) + 1
            if setup_start >= 0 and setup_start + 7 <= i:
                bar6_low = lo[setup_start + 5] if setup_start + 5 < n else lo[i]
                bar7_low = lo[setup_start + 6] if setup_start + 6 < n else lo[i]
                if lo[i] <= min(bar6_low, bar7_low):
                    buy_perfect[i] = 1.0
        if setup[i] <= -8:
            setup_start = i - int(abs(setup[i])) + 1
            if setup_start >= 0 and setup_start + 7 <= i:
                bar6_high = h[setup_start + 5] if setup_start + 5 < n else h[i]
                bar7_high = h[setup_start + 6] if setup_start + 6 < n else h[i]
                if h[i] >= max(bar6_high, bar7_high):
                    sell_perfect[i] = 1.0
    
    buy_perfect_s = pd.Series(buy_perfect, index=close.index)
    sell_perfect_s = pd.Series(sell_perfect, index=close.index)
    
    # --- TD Countdown (13-count) ---
    # After buy setup completes (>=9), count bars where close <= low_2_bars_ago
    countdown_buy = np.zeros(n)
    countdown_sell = np.zeros(n)
    in_buy_countdown = False
    in_sell_countdown = False
    buy_cd_count = 0
    sell_cd_count = 0
    
    for i in range(4, n):
        # Start countdown after setup completes
        if setup[i] >= 9 and not in_buy_countdown:
            in_buy_countdown = True
            buy_cd_count = 0
        if setup[i] <= -9 and not in_sell_countdown:
            in_sell_countdown = True
            sell_cd_count = 0
        
        # Cancel countdown if opposite setup starts
        if setup[i] <= -4 and in_buy_countdown:
            in_buy_countdown = False
            buy_cd_count = 0
        if setup[i] >= 4 and in_sell_countdown:
            in_sell_countdown = False
            sell_cd_count = 0
        
        # Count buy countdown bars
        if in_buy_countdown and i >= 2:
            if c[i] <= lo[i - 2]:
                buy_cd_count += 1
            countdown_buy[i] = buy_cd_count
            if buy_cd_count >= 13:
                in_buy_countdown = False
        
        # Count sell countdown bars
        if in_sell_countdown and i >= 2:
            if c[i] >= h[i - 2]:
                sell_cd_count += 1
            countdown_sell[i] = sell_cd_count
            if sell_cd_count >= 13:
                in_sell_countdown = False
    
    countdown_buy_s = pd.Series(countdown_buy, index=close.index)
    countdown_sell_s = pd.Series(countdown_sell, index=close.index)
    
    # --- Event signals ---
    # Buy setup complete (9 or more) -> bearish exhaustion -> expect reversal UP
    buy_setup_9 = (setup_s >= 9).astype(float)
    sell_setup_9 = (setup_s <= -9).astype(float)
    
    # Perfected setup = higher conviction
    buy_perfected = (buy_setup_9 * buy_perfect_s).clip(0, 1)
    sell_perfected = (sell_setup_9 * sell_perfect_s).clip(0, 1)
    
    # Countdown completion = highest conviction
    buy_cd_13 = (countdown_buy_s >= 13).astype(float)
    sell_cd_13 = (countdown_sell_s >= 13).astype(float)
    
    # --- Signal composition ---
    # Setup 9 alone: moderate signal (counter-trend)
    # Note: buy SETUP means 9 bars of decline -> bullish reversal signal
    long_event = buy_setup_9 * 30 + buy_perfected * 15 + buy_cd_13 * 20
    short_event = sell_setup_9 * 30 + sell_perfected * 15 + sell_cd_13 * 20
    
    # Persistence: hold exhaustion signal for 5 bars
    long_persist = long_event.rolling(5, min_periods=1).max()
    short_persist = short_event.rolling(5, min_periods=1).max()
    
    # Softer setup signals (approaching 9-count)
    approaching_buy = ((setup_s >= 7) & (setup_s < 9)).astype(float) * 15
    approaching_sell = ((setup_s <= -7) & (setup_s > -9)).astype(float) * 15
    
    # --- Continuous baseline ---
    # Setup count as directional pressure (negative setup = downtrend pressure building)
    setup_pressure = setup_s.clip(-13, 13) / 13  # -1 to +1
    # Invert: long run of decline (positive setup) = building reversal pressure
    continuous = setup_pressure * 10
    
    raw = long_persist - short_persist + approaching_buy - approaching_sell + continuous
    return _clip_signal(raw)'''

assert old_056 in content, "S056 old text not found!"
content = content.replace(old_056, new_056, 1)
print("S056 replaced OK")

# ─── S057 Candlestick Ensemble ──────────────────────────────────────────────
old_057 = '''def s057_candlestick_ensemble(ind):
    """057 | Japanese Candlestick Pattern Ensemble"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    body = close - open_p
    body_pct = body / close * 100
    upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
    atr = ind["atr14"]
    # Hammer (bullish): small body, long lower shadow at support
    hammer = ((lower_shadow > 2 * body.abs()) & (body_pct > -0.5) & (ind["rsi"] < 40)).astype(float)
    # Shooting star (bearish): small body, long upper shadow at resistance
    star = ((upper_shadow > 2 * body.abs()) & (body_pct < 0.5) & (ind["rsi"] > 60)).astype(float)
    # Engulfing
    bull_engulf = ((body > 0) & (body.shift(1) < 0) & (body.abs() > body.shift(1).abs())).astype(float)
    bear_engulf = ((body < 0) & (body.shift(1) > 0) & (body.abs() > body.shift(1).abs())).astype(float)
    raw = (hammer * 30 + bull_engulf * 30 - star * 30 - bear_engulf * 30)
    raw = raw + ind["trend_score"] * 20
    return _clip_signal(raw)'''

new_057 = '''def s057_candlestick_ensemble(ind):
    """057 | Japanese Candlestick Pattern Ensemble
    School: Tokyo (Munehisa Homma, 18th century)
    
    Core body/shadow calculations:
      Body = |Close - Open|
      Upper_Shadow = High - max(Open, Close)
      Lower_Shadow = min(Open, Close) - Low
      Body_Ratio = Body / (High - Low)
    
    Reversal Patterns (weighted ensemble):
      Hammer(0.15), Engulfing(0.25), Morning/Evening Star(0.20),
      Three White Soldiers / Three Black Crows(0.20),
      Doji(0.10), Marubozu(0.10)
    
    Signal: Pattern_Score > 0.6 in oversold zone (RSI < 35) with volume = Long.
    """
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    rsi = ind["rsi"]
    atr = ind["atr14"].replace(0, np.nan)
    
    body = close - open_p
    body_abs = body.abs()
    hl_range = (high - low).replace(0, np.nan)
    body_ratio = body_abs / hl_range
    upper_shadow = high - close.where(close >= open_p, open_p)
    lower_shadow = close.where(close <= open_p, open_p) - low
    
    # --- Pattern Detection ---
    
    # 1. Hammer (bullish): lower shadow > 2x body, upper shadow < 30% body, in downtrend
    hammer = ((lower_shadow > 2 * body_abs) &
              (upper_shadow < 0.3 * body_abs + 1e-8) &
              (close.diff(3) < 0)).astype(float)
    
    # 2. Shooting Star (bearish): upper shadow > 2x body, in uptrend
    shooting_star = ((upper_shadow > 2 * body_abs) &
                     (lower_shadow < 0.3 * body_abs + 1e-8) &
                     (close.diff(3) > 0)).astype(float)
    
    # 3. Bullish Engulfing: today bullish, yesterday bearish, today's body covers yesterday's
    bull_engulf = ((body > 0) &
                   (body.shift(1) < 0) &
                   (close > open_p.shift(1)) &
                   (open_p < close.shift(1))).astype(float)
    
    # 4. Bearish Engulfing
    bear_engulf = ((body < 0) &
                   (body.shift(1) > 0) &
                   (close < open_p.shift(1)) &
                   (open_p > close.shift(1))).astype(float)
    
    # 5. Morning Star (bullish 3-bar): big bear, small body (gap), big bull
    small_body_mid = (body_ratio.shift(1) < 0.3)
    morning_star = ((body.shift(2) < 0) &
                    (body_abs.shift(2) > atr.shift(2) * 0.5) &
                    small_body_mid &
                    (body > 0) &
                    (close > (open_p.shift(2) + close.shift(2)) / 2)).astype(float)
    
    # 6. Evening Star (bearish 3-bar)
    evening_star = ((body.shift(2) > 0) &
                    (body_abs.shift(2) > atr.shift(2) * 0.5) &
                    small_body_mid &
                    (body < 0) &
                    (close < (open_p.shift(2) + close.shift(2)) / 2)).astype(float)
    
    # 7. Three White Soldiers: 3 consecutive large bullish bodies closing near high
    tws = ((body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) &
           (body_ratio > 0.6) & (body_ratio.shift(1) > 0.6) & (body_ratio.shift(2) > 0.6) &
           (close > close.shift(1)) & (close.shift(1) > close.shift(2))).astype(float)
    
    # 8. Three Black Crows
    tbc = ((body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) &
           (body_ratio > 0.6) & (body_ratio.shift(1) > 0.6) & (body_ratio.shift(2) > 0.6) &
           (close < close.shift(1)) & (close.shift(1) < close.shift(2))).astype(float)
    
    # 9. Doji: very small body relative to range
    doji = (body_ratio < 0.1).astype(float)
    doji_reversal = doji * (close.diff(3).apply(np.sign) * -1)  # reversal direction
    
    # 10. Marubozu: body nearly equals range (strong conviction bar)
    marubozu_bull = ((body_ratio > 0.9) & (body > 0)).astype(float)
    marubozu_bear = ((body_ratio > 0.9) & (body < 0)).astype(float)
    
    # --- Weighted Ensemble Score ---
    bull_score = (hammer * 0.15 + bull_engulf * 0.25 + morning_star * 0.20 +
                  tws * 0.20 + doji_reversal.clip(0, 1) * 0.10 + marubozu_bull * 0.10)
    bear_score = (shooting_star * 0.15 + bear_engulf * 0.25 + evening_star * 0.20 +
                  tbc * 0.20 + (-doji_reversal).clip(0, 1) * 0.10 + marubozu_bear * 0.10)
    
    # --- Oscillator confirmation ---
    oversold_zone = (rsi < 35).astype(float)
    overbought_zone = (rsi > 65).astype(float)
    
    # --- Volume confirmation ---
    vol_ma = volume.rolling(20, min_periods=5).mean()
    vol_confirm = (volume > vol_ma).astype(float) * 0.3 + 0.7
    
    # --- Signal composition ---
    # Event: strong pattern in right zone
    long_event = (bull_score > 0.3) & (oversold_zone > 0)
    short_event = (bear_score > 0.3) & (overbought_zone > 0)
    
    # Score intensity
    long_intensity = bull_score * vol_confirm * (0.5 + oversold_zone * 0.5)
    short_intensity = bear_score * vol_confirm * (0.5 + overbought_zone * 0.5)
    
    # Persistence: patterns effective for 3 bars
    long_persist = long_intensity.rolling(3, min_periods=1).max()
    short_persist = short_intensity.rolling(3, min_periods=1).max()
    
    # --- Continuous baseline from pattern flow ---
    pattern_flow = bull_score.ewm(span=10).mean() - bear_score.ewm(span=10).mean()
    continuous = pattern_flow * 25
    
    # --- Compose ---
    event_sig = long_persist * 55 - short_persist * 55
    raw = event_sig + continuous + ind["trend_score"] * 10
    
    return _clip_signal(raw)'''

assert old_057 in content, "S057 old text not found!"
content = content.replace(old_057, new_057, 1)
print("S057 replaced OK")

# ─── S058 Gaussian Channel MR ───────────────────────────────────────────────
old_058 = '''def s058_gaussian_channel_mr(ind):
    """058 | Gaussian Channel Mean Reversion"""
    close = ind["close"]
    # Gaussian filter via cascaded EMA
    g1 = close.ewm(span=20, adjust=False).mean()
    g2 = g1.ewm(span=20, adjust=False).mean()
    g3 = g2.ewm(span=20, adjust=False).mean()
    gc = 3 * g1 - 3 * g2 + g3
    dev = (close - gc) / ind["vol_20"].replace(0, np.nan) / close
    z = _z(dev, 60)
    raw = -z * 50
    return _clip_signal(raw)'''

new_058 = '''def s058_gaussian_channel_mr(ind):
    """058 | Gaussian Channel Mean Reversion
    School: TradingView (DonovanWall)
    
    Gaussian Filter (4-pole, cascaded EMA):
      GF_1 = EMA(Close, n)
      GF_2 = EMA(GF_1, n)
      GF_3 = EMA(GF_2, n)
      GF_4 = EMA(GF_3, n)
      Mid = GF_4 (4-pole Gaussian approximation, -80dB/decade rolloff)
    
    Channel:
      Upper = Mid + mult * GF_4(TrueRange)
      Lower = Mid - mult * GF_4(TrueRange)
    
    Channel_Position = (Close - Lower) / (Upper - Lower)
    
    Long:  Position < 0.05 AND Mid slope > 0 (lower band in uptrend)
    Short: Position > 0.95 AND Mid slope < 0 (upper band in downtrend)
    Exit:  Position returns to 0.50
    """
    close = ind["close"]
    tr = ind["tr"].fillna(0)
    n = 20
    mult = 2.0
    
    # --- 4-pole Gaussian filter (cascaded EMA) ---
    gf1 = close.ewm(span=n, adjust=False).mean()
    gf2 = gf1.ewm(span=n, adjust=False).mean()
    gf3 = gf2.ewm(span=n, adjust=False).mean()
    gf4 = gf3.ewm(span=n, adjust=False).mean()
    
    gf_mid = gf4  # This is the 4-pole Gaussian approximation
    
    # --- Filtered True Range for channel width ---
    tr_f1 = tr.ewm(span=n, adjust=False).mean()
    tr_f2 = tr_f1.ewm(span=n, adjust=False).mean()
    tr_f3 = tr_f2.ewm(span=n, adjust=False).mean()
    tr_f4 = tr_f3.ewm(span=n, adjust=False).mean()
    
    gf_upper = gf_mid + mult * tr_f4
    gf_lower = gf_mid - mult * tr_f4
    
    # --- Channel Position ---
    chan_range = (gf_upper - gf_lower).replace(0, np.nan)
    chan_pos = ((close - gf_lower) / chan_range).clip(-0.5, 1.5).fillna(0.5)
    
    # --- Mid slope (uptrend / downtrend) ---
    mid_slope = gf_mid.diff(5) / gf_mid.shift(5).replace(0, np.nan)
    slope_up = (mid_slope > 0).astype(float)
    slope_dn = (mid_slope < 0).astype(float)
    
    # --- Event signals ---
    # Long: at or below lower band in uptrend
    long_event = ((chan_pos < 0.05) & (slope_up > 0)).astype(float)
    # Short: at or above upper band in downtrend
    short_event = ((chan_pos > 0.95) & (slope_dn > 0)).astype(float)
    
    # Also detect bounces from near-band levels
    near_lower = ((chan_pos < 0.15) & (chan_pos > chan_pos.shift(1))).astype(float)
    near_upper = ((chan_pos > 0.85) & (chan_pos < chan_pos.shift(1))).astype(float)
    
    # Persistence: hold for 6 bars
    long_persist = long_event.rolling(6, min_periods=1).max()
    short_persist = short_event.rolling(6, min_periods=1).max()
    
    # --- Continuous: distance from midline (0.5) ---
    mid_dist = 0.5 - chan_pos  # positive below mid, negative above
    continuous = mid_dist.clip(-0.5, 0.5) / 0.5 * 25
    
    # --- Compose ---
    event_sig = long_persist * 40 + near_lower * 15 - short_persist * 40 - near_upper * 15
    raw = event_sig + continuous
    
    # Trend alignment boost
    trend_align = slope_up * 10 - slope_dn * 10
    raw = raw + trend_align
    
    return _clip_signal(raw)'''

assert old_058 in content, "S058 old text not found!"
content = content.replace(old_058, new_058, 1)
print("S058 replaced OK")

# ─── S059 PCA Pairs ─────────────────────────────────────────────────────────
old_059 = '''def s059_pca_pairs(ind):
    """059 | PCA Pairs Trading (single-asset: principal residual from MAs)"""
    close = ind["close"]
    ma_blend = 0.5 * ind["sma_20"] + 0.3 * ind["sma_50"] + 0.2 * ind["sma_200"]
    residual = close - ma_blend
    z = _z(residual, 60)
    raw = -z * 50
    return _clip_signal(raw)'''

new_059 = '''def s059_pca_pairs(ind):
    """059 | PCA Pairs Trading via Principal Component Analysis
    School: London (AHL/Man Group)
    
    Single-asset proxy for PCA residual trading:
    Factor = weighted blend of multiple MA horizons (proxy for sector/market factor).
    Residual = Close - Factor_Blend (stock-specific deviation).
    
    residual_i(t) = actual_return - factor_explained_return
    cumulative_residual = cumsum(residual)
    Z_residual = (cumulative_residual - SMA(60)) / StdDev(60)
    
    Long:  Z_residual < -2.0 (stock lagging its factor exposure)
    Short: Z_residual > +2.0
    Exit:  Z_residual returns to 0
    """
    close = ind["close"]
    
    # --- Factor blend (proxy for PCA first k components) ---
    # Multiple MA horizons capture market/sector factor at different timescales
    # Weights sum to 1.0, heavier on shorter MAs (more responsive)
    factor = (0.35 * ind["sma_20"] +
              0.25 * ind["sma_50"] +
              0.20 * ind["ema_50"] +
              0.20 * ind["sma_200"])
    
    # --- Residual: price minus factor explained component ---
    ret = close.pct_change(1).fillna(0)
    factor_ret = factor.pct_change(1).fillna(0)
    residual_ret = ret - factor_ret
    
    # Cumulative residual (stock-specific drift)
    cum_residual = residual_ret.cumsum()
    
    # --- Z-score of cumulative residual ---
    res_ma = cum_residual.rolling(60, min_periods=15).mean()
    res_std = cum_residual.rolling(60, min_periods=15).std().replace(0, np.nan)
    z_res = ((cum_residual - res_ma) / res_std).clip(-5, 5).fillna(0)
    
    # --- Event signals ---
    long_event = (z_res < -2.0).astype(float)
    short_event = (z_res > 2.0).astype(float)
    
    # Depth scaling
    long_depth = (-z_res - 2.0).clip(0, 3) / 3  # 0 to 1
    short_depth = (z_res - 2.0).clip(0, 3) / 3
    
    # Persistence: hold for 10 bars
    long_persist = long_event.rolling(10, min_periods=1).max()
    short_persist = short_event.rolling(10, min_periods=1).max()
    long_depth_p = long_depth.rolling(10, min_periods=1).max()
    short_depth_p = short_depth.rolling(10, min_periods=1).max()
    
    # --- Continuous: proportional MR pressure ---
    continuous = -z_res.clip(-3, 3) / 3 * 20
    
    # --- Compose ---
    event_sig = long_persist * (35 + long_depth_p * 15) - short_persist * (35 + short_depth_p * 15)
    raw = event_sig + continuous
    
    # Emergency: |Z| > 4 = structural break
    emergency = (z_res.abs() > 4.0).astype(float)
    raw = raw * (1 - emergency * 0.8)
    
    return _clip_signal(raw)'''

assert old_059 in content, "S059 old text not found!"
content = content.replace(old_059, new_059, 1)
print("S059 replaced OK")

# ─── S060 OU Optimal Entry ──────────────────────────────────────────────────
old_060 = '''def s060_ou_optimal_entry(ind):
    """060 | Mean Reversion with OU Optimal Entry"""
    close = ind["close"]
    log_p = np.log(close)
    mu = log_p.rolling(120, min_periods=40).mean()
    dev = log_p - mu
    vol = dev.rolling(20).std().replace(0, np.nan)
    z = dev / vol
    # Optimal entry: further from mean = stronger signal
    raw = -z.clip(-3, 3) / 3 * 70
    half_life_adj = ind["efficiency"]
    raw = raw * (1 - half_life_adj * 0.3)
    return _clip_signal(raw)'''

new_060 = '''def s060_ou_optimal_entry(ind):
    """060 | Mean Reversion with OU Optimal Entry
    School: Zurich (ETH Quantitative Finance)
    
    Optimal entry for OU process with transaction costs c:
      optimal_entry_distance = sigma_eq * sqrt(2 * ln(1 / cost_ratio))
      where sigma_eq = sigma / sqrt(2 * kappa)
      cost_ratio = transaction_cost / sigma_eq
    
    For typical params (sigma_eq=2%, cost=0.1%):
      optimal_entry = theta +/- 2.15 * sigma_eq
    
    Long:  Price < theta - optimal_entry_distance
    Short: Price > theta + optimal_entry_distance
    Exit:  Price crosses theta (equilibrium)
    Skip:  kappa < 0 (no MR) or half-life > 40 days
    """
    close = ind["close"]
    log_p = np.log(close.clip(lower=1e-8))
    n = len(close)
    
    # --- AR(1) calibration for kappa, theta, sigma_eq ---
    window = 60
    kappa_arr = np.zeros(n)
    theta_arr = np.full(n, np.nan)
    sigma_eq_arr = np.full(n, np.nan)
    
    lp = log_p.values.astype(float)
    for i in range(window, n):
        y = lp[i - window + 1 : i + 1]
        x = lp[i - window : i]
        xm = x.mean()
        ym = y.mean()
        ssxx = ((x - xm) ** 2).sum()
        if ssxx < 1e-15:
            continue
        b = ((x - xm) * (y - ym)).sum() / ssxx
        a = ym - b * xm
        b_c = np.clip(b, 0.001, 0.999)
        kap = -np.log(b_c)
        th = a / (1 - b_c + 1e-12)
        eps = y - (a + b * x)
        sig = eps.std()
        sig_eq = sig / np.sqrt(2.0 * kap + 1e-12)
        kappa_arr[i] = kap
        theta_arr[i] = th
        sigma_eq_arr[i] = sig_eq
    
    kappa_s = pd.Series(kappa_arr, index=close.index)
    theta_s = pd.Series(theta_arr, index=close.index).ffill().fillna(log_p.rolling(120).mean())
    sigma_eq_s = pd.Series(sigma_eq_arr, index=close.index).ffill().fillna(0.02)
    
    # --- Half-life and validity ---
    half_life = (np.log(2.0) / kappa_s.replace(0, np.nan)).clip(1, 200).fillna(60)
    valid = ((kappa_s > 0.01) & (half_life < 40)).astype(float)
    
    # --- Optimal entry distance (cost-aware) ---
    cost = 0.001  # 10 bps transaction cost
    cost_ratio = cost / sigma_eq_s.replace(0, np.nan)
    # optimal_dist = sigma_eq * sqrt(2 * ln(1 / cost_ratio))
    # Clamp ln argument to avoid negatives
    ln_arg = (1.0 / cost_ratio.clip(lower=0.01)).clip(1, 1000)
    optimal_dist = sigma_eq_s * np.sqrt(2 * np.log(ln_arg))
    optimal_dist = optimal_dist.clip(lower=sigma_eq_s * 1.5)  # minimum 1.5 sigma
    
    # --- Deviation from equilibrium ---
    deviation = log_p - theta_s
    
    # --- Entry signals ---
    long_entry = ((deviation < -optimal_dist) & (valid > 0)).astype(float)
    short_entry = ((deviation > optimal_dist) & (valid > 0)).astype(float)
    
    # Intensity: how far past optimal entry (deeper = stronger)
    long_excess = ((-deviation - optimal_dist) / sigma_eq_s.replace(0, np.nan)).clip(0, 3).fillna(0)
    short_excess = ((deviation - optimal_dist) / sigma_eq_s.replace(0, np.nan)).clip(0, 3).fillna(0)
    
    # Persistence: hold for up to 10 bars
    long_persist = long_entry.rolling(10, min_periods=1).max()
    short_persist = short_entry.rolling(10, min_periods=1).max()
    
    # --- Exit detection: price crosses theta ---
    near_theta = (deviation.abs() < sigma_eq_s * 0.5).astype(float)
    # Dampen signal near equilibrium
    exit_factor = 1 - near_theta * 0.7
    
    # --- Continuous: MR pressure proportional to deviation ---
    z_dev = (deviation / sigma_eq_s.replace(0, np.nan)).clip(-4, 4).fillna(0)
    continuous = -z_dev * valid * 15
    
    # --- Speed bonus: faster MR = larger position ---
    speed = (1.0 + (20.0 - half_life.clip(5, 40)) / 30.0).clip(0.7, 1.5)
    
    # --- Compose ---
    event_sig = long_persist * (35 + long_excess * 10) - short_persist * (35 + short_excess * 10)
    raw = (event_sig + continuous) * speed * exit_factor
    
    return _clip_signal(raw)'''

assert old_060 in content, "S060 old text not found!"
content = content.replace(old_060, new_060, 1)
print("S060 replaced OK")

# ─── Write out ──────────────────────────────────────────────────────────────
with open(filepath, "w") as f:
    f.write(content)

print("All S056-S060 replacements written successfully.")
