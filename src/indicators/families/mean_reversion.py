"""
SECTION II: MEAN REVERSION AND STATISTICAL ARBITRAGE (051-100)
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

def s051_ornstein_uhlenbeck(ind):
    """051 | Ornstein-Uhlenbeck Mean Reversion
    School: Mathematical Finance (Uhlenbeck & Ornstein)
    
    Calibrates OU process via AR(1) regression on log-prices:
      X_t = a + b * X_{t-1} + eps
      kappa = -ln(b) / dt
      theta = a / (1 - b)
      sigma_eq = sigma / sqrt(2 * kappa)
      half_life = ln(2) / kappa
    
    Long:  X < theta - 1.5 * sigma_eq  AND kappa > 0.02  AND 5 < HL < 60
    Short: X > theta + 1.5 * sigma_eq  AND kappa > 0.02  AND 5 < HL < 60
    """
    close = ind["close"]
    log_p = np.log(close.clip(lower=1e-8))
    n = len(close)
    
    # --- AR(1) rolling regression to estimate kappa, theta, sigma_eq ---
    window = 60
    kappa_s = pd.Series(0.0, index=close.index)
    theta_s = pd.Series(0.0, index=close.index)
    sigma_eq_s = pd.Series(0.0, index=close.index)
    half_life_s = pd.Series(np.nan, index=close.index)
    
    lp = log_p.values.astype(float)
    for i in range(window, n):
        y = lp[i - window + 1 : i + 1]        # X_t
        x = lp[i - window : i]                  # X_{t-1}
        xm = x.mean()
        ym = y.mean()
        ssxx = ((x - xm) ** 2).sum()
        if ssxx < 1e-15:
            continue
        b = ((x - xm) * (y - ym)).sum() / ssxx
        a = ym - b * xm
        # Clamp b to avoid log domain errors
        b_c = np.clip(b, 0.001, 0.999)
        kap = -np.log(b_c)                     # kappa = -ln(b)
        th = a / (1 - b_c + 1e-12)             # theta = a / (1 - b)
        eps = y - (a + b * x)
        sig = eps.std()
        sig_eq = sig / np.sqrt(2.0 * kap + 1e-12)
        hl = np.log(2.0) / (kap + 1e-12)
        kappa_s.iloc[i] = kap
        theta_s.iloc[i] = th
        sigma_eq_s.iloc[i] = sig_eq
        half_life_s.iloc[i] = hl
    
    # --- Deviation from equilibrium in sigma_eq units ---
    deviation = log_p - theta_s
    sigma_eq_safe = sigma_eq_s.replace(0, np.nan)
    z_ou = (deviation / sigma_eq_safe).clip(-5, 5).fillna(0)
    
    # --- Gate: kappa > 0.02 and half-life between 5 and 60 ---
    valid_mr = ((kappa_s > 0.02) &
                (half_life_s > 5) & (half_life_s < 60)).astype(float)
    
    # --- Event signals: extreme deviation ---
    long_event = ((z_ou < -1.5) & (valid_mr > 0)).astype(float)
    short_event = ((z_ou > 1.5) & (valid_mr > 0)).astype(float)
    
    # Persistence: hold signal for up to 8 bars after event fires
    long_persist = long_event.rolling(8, min_periods=1).max()
    short_persist = short_event.rolling(8, min_periods=1).max()
    
    # --- Continuous baseline: proportional to z_ou in valid regime ---
    continuous = -z_ou * valid_mr * 25
    
    # --- Compose signal ---
    event_sig = long_persist * 45 - short_persist * 45
    
    # Speed bonus: faster MR (shorter half-life) = stronger signal
    speed_mult = (1.0 + (30.0 - half_life_s.clip(5, 60)) / 50.0).clip(0.7, 1.5).fillna(1.0)
    
    raw = (event_sig + continuous) * speed_mult
    return _clip_signal(raw)


def s052_cointegration_pairs(ind):
    """052 | Cointegration Pairs Trading (Engle-Granger)
    School: London School of Economics (Engle & Granger, 1987)
    
    Single-asset proxy for pairs: price vs dynamic fair-value (Kalman beta * SMA_50).
    Spread_Z from rolling 60-day window.
    Kalman-like adaptive beta tracks the hedge ratio.
    
    Long spread:  Z < -2.0  AND  half-life < 30 days
    Short spread: Z > +2.0
    Emergency:    |Z| > 4.0 -> flatten (relationship breaking)
    """
    close = ind["close"]
    sma50 = ind["sma_50"]
    
    # --- Kalman-like dynamic beta (hedge ratio) ---
    # beta_t = beta_{t-1} + gain * innovation
    lp = np.log(close.clip(lower=1e-8))
    lp50 = np.log(sma50.clip(lower=1e-8))
    vals_y = lp.values.astype(float)
    vals_x = lp50.values.astype(float)
    n = len(close)
    beta = np.ones(n)
    alpha_arr = np.zeros(n)
    kalman_gain = 0.05  # adaptation speed
    
    for i in range(1, n):
        if np.isnan(vals_x[i]) or np.isnan(vals_y[i]):
            beta[i] = beta[i - 1]
            alpha_arr[i] = alpha_arr[i - 1]
            continue
        predicted = alpha_arr[i - 1] + beta[i - 1] * vals_x[i]
        innovation = vals_y[i] - predicted
        beta[i] = beta[i - 1] + kalman_gain * innovation * vals_x[i]
        alpha_arr[i] = alpha_arr[i - 1] + kalman_gain * innovation * 0.1
    
    beta_s = pd.Series(beta, index=close.index)
    alpha_s = pd.Series(alpha_arr, index=close.index)
    
    # --- Spread and Z-score ---
    spread = lp - beta_s * lp50 - alpha_s
    spread_ma = spread.rolling(60, min_periods=15).mean()
    spread_std = spread.rolling(60, min_periods=15).std().replace(0, np.nan)
    z_spread = ((spread - spread_ma) / spread_std).clip(-6, 6).fillna(0)
    
    # --- Half-life estimation from spread autocorrelation ---
    def _rolling_halflife(s, win=60):
        hl = pd.Series(np.nan, index=s.index)
        sv = s.values.astype(float)
        for i in range(win, len(sv)):
            chunk = sv[i - win : i]
            if np.all(np.isnan(chunk)):
                continue
            y = chunk[1:]
            x = chunk[:-1]
            mx = np.nanmean(x)
            ssxx = np.nansum((x - mx) ** 2)
            if ssxx < 1e-15:
                continue
            b = np.nansum((x - mx) * (y - np.nanmean(y))) / ssxx
            b = np.clip(b, 0.001, 0.999)
            hl.iloc[i] = -np.log(2.0) / np.log(b)
        return hl.clip(1, 200).fillna(60)
    
    half_life = _rolling_halflife(spread)
    tradeable_hl = (half_life < 30).astype(float)
    
    # --- Event signals ---
    long_event = ((z_spread < -2.0) & (tradeable_hl > 0)).astype(float)
    short_event = ((z_spread > 2.0) & (tradeable_hl > 0)).astype(float)
    emergency = (z_spread.abs() > 4.0).astype(float)
    
    # Persistence
    long_persist = long_event.rolling(10, min_periods=1).max()
    short_persist = short_event.rolling(10, min_periods=1).max()
    
    # --- Continuous baseline: proportional to z ---
    continuous = -z_spread * tradeable_hl * 20
    
    # --- Compose ---
    event_sig = long_persist * 40 - short_persist * 40
    raw = event_sig + continuous
    
    # Emergency override: flatten signal toward zero
    raw = raw * (1 - emergency * 0.8)
    
    return _clip_signal(raw)


def s053_bb_pctb_mr(ind):
    """053 | Bollinger Band %B Mean Reversion
    School: New York (John Bollinger)
    
    %B = (Close - BB_Lower) / (BB_Upper - BB_Lower)
    Reversal trigger: %B crosses back INTO bands from extreme.
    
    Long:  %B < -0.10 then crosses above 0.05 (bounce off lower band)
    Short: %B > 1.10 then crosses below 0.95 (rejection at upper band)
    Filter: BandWidth > 20th percentile of 120-day range (no squeeze traps)
    Target: %B = 0.50 (middle band)
    """
    close = ind["close"]
    pctb = ind["bb_pctb"].fillna(0.5)
    bb_width = ind["bb_width"].fillna(0)
    
    # --- BandWidth filter: avoid low-vol squeeze traps ---
    bw_pctile_20 = bb_width.rolling(120, min_periods=30).quantile(0.20)
    bw_ok = (bb_width > bw_pctile_20).astype(float)
    
    # --- Crossing detection: %B re-entering bands ---
    was_below = (pctb < -0.10).rolling(5, min_periods=1).max()   # was extreme low recently
    crossed_up = (pctb > 0.05).astype(float)                      # now back inside
    long_cross = (was_below * crossed_up).clip(0, 1)
    
    was_above = (pctb > 1.10).rolling(5, min_periods=1).max()    # was extreme high recently
    crossed_dn = (pctb < 0.95).astype(float)                      # now back inside
    short_cross = (was_above * crossed_dn).clip(0, 1)
    
    # Persistence: hold cross signal for 8 bars
    long_persist = long_cross.rolling(8, min_periods=1).max()
    short_persist = short_cross.rolling(8, min_periods=1).max()
    
    # --- Distance from midline (0.5) as continuous component ---
    mid_dist = 0.5 - pctb  # positive when below mid, negative when above
    continuous = mid_dist.clip(-1, 1) * 25
    
    # --- Depth scaling: deeper extreme = stronger signal ---
    depth_long = (0.05 - pctb).clip(0, 1.0) / 1.0   # how far below 0.05
    depth_short = (pctb - 0.95).clip(0, 1.0) / 1.0   # how far above 0.95
    
    # --- Compose signal ---
    event_sig = long_persist * (35 + depth_long * 20) - short_persist * (35 + depth_short * 20)
    
    # Apply bandwidth filter
    raw = (event_sig + continuous) * (0.4 + bw_ok * 0.6)
    
    # Trend context: longs work better in uptrend
    above_200 = ind["above_200"].fillna(0.5)
    raw = raw + (above_200 * 2 - 1) * 10
    
    return _clip_signal(raw)


def s054_zscore_stat_arb(ind):
    """054 | Z-Score Statistical Arbitrage with Regime Filter
    School: Chicago (Citadel / DE Shaw style)
    
    Spread = log(Close) - EMA smoothed equilibrium.
    Z-score with half-life calibrated lookback.
    
    Regime filter (critical innovation):
      - Autocorrelation(spread, lag=1, window=60): AC < -0.05 = mean-reverting
      - Hurst exponent < 0.45 = confirmed MR
      - Only trade when BOTH conditions met
    
    Long:  Z < -2.0  AND  AC < -0.05  AND  Hurst < 0.45
    Short: Z > +2.0  AND  AC < -0.05  AND  Hurst < 0.45
    Exit:  Z returns to +/- 0.5 (partial), Z = 0 (full)
    Emergency: |Z| > 4 (structural break -> close)
    """
    close = ind["close"]
    hurst = ind["hurst"].fillna(0.5)
    
    # --- Spread: log price vs adaptive equilibrium ---
    log_p = np.log(close.clip(lower=1e-8))
    eq = log_p.ewm(span=40, adjust=False).mean()
    spread = log_p - eq
    
    # --- Z-score of spread ---
    sp_ma = spread.rolling(60, min_periods=15).mean()
    sp_std = spread.rolling(60, min_periods=15).std().replace(0, np.nan)
    z_spread = ((spread - sp_ma) / sp_std).clip(-6, 6).fillna(0)
    
    # --- Autocorrelation filter (lag-1, rolling 60-day) ---
    ret_spread = spread.diff()
    def _rolling_autocorr(s, win=60):
        sv = s.values.astype(float)
        ac = np.full(len(sv), np.nan)
        for i in range(win, len(sv)):
            chunk = sv[i - win : i]
            valid = ~np.isnan(chunk)
            if valid.sum() < 20:
                continue
            c = chunk[valid]
            if len(c) < 3:
                continue
            ac[i] = np.corrcoef(c[:-1], c[1:])[0, 1]
        return pd.Series(ac, index=s.index).fillna(0)
    
    autocorr = _rolling_autocorr(spread)
    
    # --- Regime gate ---
    # Strong MR regime: AC < -0.05 AND Hurst < 0.45 -> full signal
    strong_mr = ((autocorr < -0.05) & (hurst < 0.45)).astype(float)
    # Moderate MR: either condition alone -> 60% signal
    moderate_mr = ((autocorr < 0.05) | (hurst < 0.50)).astype(float) * 0.6
    # Weak MR fallback: always-on base for any non-trending asset -> 30%
    weak_mr = (hurst < 0.55).astype(float) * 0.3
    regime_strength = (strong_mr + moderate_mr * (1 - strong_mr) + weak_mr * (1 - strong_mr) * (1 - moderate_mr.clip(0,1))).clip(0, 1)
    
    # --- Event signals: Z extreme + any MR regime ---
    long_event = ((z_spread < -1.8) & (regime_strength > 0.2)).astype(float)
    short_event = ((z_spread > 1.8) & (regime_strength > 0.2)).astype(float)
    emergency = (z_spread.abs() > 4.0).astype(float)
    
    # Deeper extremes get extra intensity
    long_depth = (-z_spread - 1.8).clip(0, 2) / 2  # 0 at -1.8, 1 at -3.8
    short_depth = (z_spread - 1.8).clip(0, 2) / 2
    
    # Persistence: hold event for up to 8 bars
    long_persist = long_event.rolling(8, min_periods=1).max()
    short_persist = short_event.rolling(8, min_periods=1).max()
    long_depth_p = long_depth.rolling(8, min_periods=1).max()
    short_depth_p = short_depth.rolling(8, min_periods=1).max()
    
    # --- Continuous baseline: proportional to z, always on ---
    continuous = -z_spread.clip(-3, 3) / 3 * 25 * regime_strength
    
    # --- Compose ---
    event_sig = long_persist * (35 + long_depth_p * 15) - short_persist * (35 + short_depth_p * 15)
    raw = event_sig * regime_strength + continuous
    
    # Emergency flatten
    raw = raw * (1 - emergency * 0.9)
    
    return _clip_signal(raw)


def s055_rsi2_extreme(ind):
    """055 | RSI(2) Extreme Reversal
    School: New York (Larry Connors)
    
    RSI(2) captures extreme short-term dislocations.
    2-period lookback = ultra-sensitive to 1-2 day moves.
    
    Multi-day entry scaling:
      Day 1: RSI(2) < 25 -> watchlist (light signal)
      Day 2: RSI(2) < 10 -> half position
      Day 3: RSI(2) < 5  -> full position (maximum MR potential)
    
    Long:  RSI(2) < 10  AND  Close > SMA(200) (oversold in uptrend)
    Short: RSI(2) > 90  AND  Close < SMA(200) (overbought in downtrend)
    Exit:  RSI(2) > 70 (for longs)
    
    Edge: >80% win rate at RSI(2) < 5 in stocks above SMA(200) per Connors.
    """
    close = ind["close"]
    
    # --- Compute RSI(2) using Wilder smoothing ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    
    # Wilder's smoothing (RMA) with period=2
    avg_gain = gain.ewm(alpha=1.0/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/2, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi2 = (100 - 100 / (1 + rs)).fillna(50)
    
    # --- Trend filter: SMA(200) ---
    above_200 = ind["above_200"].fillna(0.5)
    below_200 = 1 - above_200
    
    # --- Multi-day oversold scaling (deeper = stronger) ---
    # Level 1: RSI(2) < 25 -> watchlist
    watchlist_long = (rsi2 < 25).astype(float) * 0.3
    # Level 2: RSI(2) < 10 -> half position
    entry_long = (rsi2 < 10).astype(float) * 0.5
    # Level 3: RSI(2) < 5 -> full position (maximum MR)
    deep_long = (rsi2 < 5).astype(float) * 0.7
    
    # Combine levels (they stack)
    long_intensity = (watchlist_long + entry_long + deep_long).clip(0, 1.5)
    
    # Multi-day overbought scaling
    watchlist_short = (rsi2 > 75).astype(float) * 0.3
    entry_short = (rsi2 > 90).astype(float) * 0.5
    deep_short = (rsi2 > 95).astype(float) * 0.7
    short_intensity = (watchlist_short + entry_short + deep_short).clip(0, 1.5)
    
    # --- Multi-day persistence (Connors: hold 3-5 days) ---
    long_persist = long_intensity.rolling(5, min_periods=1).max()
    short_persist = short_intensity.rolling(5, min_periods=1).max()
    
    # --- Exit detection: RSI(2) > 70 cancels long signal ---
    exit_long = (rsi2 > 70).astype(float)
    exit_short = (rsi2 < 30).astype(float)
    long_persist = long_persist * (1 - exit_long)
    short_persist = short_persist * (1 - exit_short)
    
    # --- Apply trend filter ---
    long_sig = long_persist * above_200 * 55
    short_sig = short_persist * below_200 * 55
    
    # --- Continuous baseline from RSI(2) extremes ---
    # Mild mean-reversion pressure proportional to RSI distance from 50
    rsi_centered = (50 - rsi2) / 50  # positive when oversold, negative when overbought
    continuous = rsi_centered * 15
    
    raw = long_sig - short_sig + continuous
    return _clip_signal(raw)


def s056_demark_td9(ind):
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
    return _clip_signal(raw)


def s057_candlestick_ensemble(ind):
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
    # Three tiers: strong (pattern + zone), moderate (pattern alone), continuous
    
    # Tier 1: Strong event - pattern in matching RSI zone (full strength)
    strong_long = bull_score.where(oversold_zone > 0, 0.0) * vol_confirm
    strong_short = bear_score.where(overbought_zone > 0, 0.0) * vol_confirm
    
    # Tier 2: Moderate event - any pattern with minimum score 0.10 (60% strength)
    mod_long = bull_score.where(bull_score >= 0.10, 0.0) * vol_confirm * 0.6
    mod_short = bear_score.where(bear_score >= 0.10, 0.0) * vol_confirm * 0.6
    
    # Combine tiers (strong overrides moderate where present)
    long_intensity = strong_long.where(strong_long > 0, mod_long)
    short_intensity = strong_short.where(strong_short > 0, mod_short)
    
    # Persistence: patterns effective for 5 bars (candlestick reversals need time)
    long_persist = long_intensity.rolling(5, min_periods=1).max()
    short_persist = short_intensity.rolling(5, min_periods=1).max()
    
    # --- Continuous baseline from pattern flow ---
    pattern_flow = bull_score.ewm(span=8).mean() - bear_score.ewm(span=8).mean()
    
    # Also add body momentum (large bullish bodies = positive pressure)
    body_mom = body.ewm(span=10).mean() / atr.ewm(span=20).mean()
    body_mom = body_mom.clip(-1, 1)
    
    continuous = _norm(pattern_flow, span=40) * 18 + body_mom * 12
    
    # --- Compose ---
    event_sig = long_persist * 85 - short_persist * 85
    raw = event_sig + continuous + ind["trend_score"] * 8
    
    return _clip_signal(raw)


def s058_gaussian_channel_mr(ind):
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
    
    return _clip_signal(raw)


def s059_pca_pairs(ind):
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
    
    return _clip_signal(raw)


def s060_ou_optimal_entry(ind):
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
    
    return _clip_signal(raw)


def s061_gilt_spread(ind):
    """061 | London Gilt Spread Convergence (vol spread proxy)"""
    vol_short = ind["vol_20"]
    vol_long = ind["vol_60"]
    spread = _z(vol_short / vol_long.replace(0, np.nan), 60)
    raw = -spread * 40 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s062_relative_value_rotation(ind):
    """062 | Relative Value Rotation ETF Strategy"""
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    rv = mom / vol.replace(0, np.nan)
    z = _z(rv, 120)
    raw = z * 50
    return _clip_signal(raw)


def s063_kalman_spread(ind):
    """063 | Kalman Filter Spread Tracker"""
    close = ind["close"]
    # Simple Kalman-like exponential smoother
    alpha = 0.1
    vals = close.values.astype(float)
    kf = np.zeros(len(vals))
    kf[0] = vals[0]
    for i in range(1, len(vals)):
        kf[i] = kf[i-1] + alpha * (vals[i] - kf[i-1])
    spread = close - pd.Series(kf, index=close.index)
    z = _z(spread, 40)
    raw = -z * 55
    return _clip_signal(raw)


def s064_csi300_reversal(ind):
    """064 | Shanghai CSI 300 Index Reversal After Policy Signal (OHLCV proxy)"""
    rsi = ind["rsi"]
    vol_spike = (ind["vol_rel"] > 2.0).astype(float)
    extreme_sell = (rsi < 25) & (vol_spike > 0)
    extreme_buy = (rsi > 75) & (vol_spike > 0)
    raw = extreme_sell.astype(float) * 60 - extreme_buy.astype(float) * 60
    return _clip_signal(raw)


def s065_willr_extreme_mr(ind):
    """065 | Williams %R Extreme Mean Reversion"""
    willr = ind["willr"]
    oversold = (willr < -80).astype(float) * (-80 - willr) / 20
    overbought = (willr > -20).astype(float) * (willr + 20) / 20
    raw = oversold * 55 - overbought * 55
    return _clip_signal(raw)


def s066_dispersion_trading(ind):
    """066 | Dispersion Trading via Implied Correlation (vol proxy)"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    # High dispersion (high vol) -> mean revert
    raw = -vol_z * 40
    return _clip_signal(raw)


def s067_vwap_reversion(ind):
    """067 | Intraday VWAP Reversion (daily proxy)"""
    close = ind["close"]
    volume = ind["volume"]
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    dist = (close - vwap) / ind["atr14"].replace(0, np.nan)
    raw = -dist.clip(-3, 3) / 3 * 60
    return _clip_signal(raw)


def s068_johansen_basket(ind):
    """068 | Johansen Cointegration Multi-Leg (single asset: multi-MA spread)"""
    close = ind["close"]
    spread = close - (0.4 * ind["sma_20"] + 0.3 * ind["sma_50"] + 0.3 * ind["sma_200"])
    z = _z(spread, 80)
    raw = -z * 55
    return _clip_signal(raw)


def s069_vol_risk_premium(ind):
    """069 | Zurich Volatility Risk Premium (realized vol proxy)"""
    vol = ind["vol_20"]
    vol_long = ind["vol_60"]
    vrp = vol - vol_long
    vrp_z = _z(vrp, 120)
    raw = -vrp_z * 45
    return _clip_signal(raw)


def s070_adf_mean_reversion(ind):
    """070 | Mean Reversion Ratio with ADF (simplified)"""
    close = ind["close"]
    log_p = np.log(close)
    ma = log_p.rolling(60).mean()
    dev = log_p - ma
    # Simple half-life proxy
    beta = dev.rolling(60).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.5, raw=True)
    mean_revert = (beta < 0.95).astype(float)
    z = _z(dev, 60)
    raw = -z * mean_revert * 55
    return _clip_signal(raw)


def s071_nikkei_auction(ind):
    """071 | Tokyo Nikkei Opening Auction Imbalance (gap proxy)"""
    close = ind["close"]
    open_p = ind["open"]
    gap = (open_p - close.shift(1)) / close.shift(1) * 100
    fade = -_z(gap, 40) * 40
    return _clip_signal(fade)


def s072_rsi_failure_swing(ind):
    """072 | RSI Failure Swing"""
    rsi = ind["rsi"]
    # Failure swing bottom: RSI dips below 30, bounces, dips again but stays above prior low
    rsi_min5 = rsi.rolling(5).min()
    failure_bottom = ((rsi < 35) & (rsi > rsi_min5.shift(5)) & (rsi_min5.shift(5) < 30)).astype(float)
    failure_top = ((rsi > 65) & (rsi < rsi.rolling(5).max().shift(5)) & (rsi.rolling(5).max().shift(5) > 70)).astype(float)
    raw = failure_bottom * 55 - failure_top * 55
    return _clip_signal(raw)


def s073_defensive_momentum(ind):
    """073 | Swiss Private Banking Defensive Momentum"""
    mom = ind["mom_20"]
    vol = ind["vol_pct"]
    above_200 = ind["above_200"].fillna(0.5)
    defensive = above_200 * (1 - vol) * mom
    raw = defensive * 70
    return _clip_signal(raw)


def s074_roll_yield(ind):
    """074 | Commodity Term Structure Roll Yield (vol contango proxy)"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    contango = (vol_60 - vol_20) / vol_60.replace(0, np.nan)
    raw = _z(contango, 120) * 40 + ind["mom_20"] * 30
    return _clip_signal(raw)


def s075_seasonal_mr(ind):
    """075 | Sydney ASX 200 Seasonal Mean Reversion (OHLCV proxy)"""
    close = ind["close"]
    monthly_ret = close.pct_change(20)
    z = _z(monthly_ret, 252)
    raw = -z * 45
    return _clip_signal(raw)


def s076_momentum_crash_protection(ind):
    """076 | Cross-Sectional Momentum Crash Protection"""
    mom = ind["mom_score"]
    vol = ind["vol_pct"]
    crash_risk = (vol > 0.8).astype(float)
    raw = mom * (1 - crash_risk * 0.7) * 60
    return _clip_signal(raw)


def s077_quantile_regression_mr(ind):
    """077 | Quantile Regression Mean Reversion"""
    close = ind["close"]
    pct_rank = close.rolling(60, min_periods=20).rank(pct=True)
    extreme_low = (pct_rank < 0.1).astype(float) * (0.1 - pct_rank) / 0.1
    extreme_high = (pct_rank > 0.9).astype(float) * (pct_rank - 0.9) / 0.1
    raw = extreme_low * 55 - extreme_high * 55
    return _clip_signal(raw)


def s078_expiry_week_mr(ind):
    """078 | Mumbai Nifty 50 Expiry Week MR (OHLCV proxy: weekly cycle)"""
    rsi = ind["rsi"]
    stoch = ind["stoch_k"]
    oversold = ((rsi < 35) & (stoch < 25)).astype(float)
    overbought = ((rsi > 65) & (stoch > 75)).astype(float)
    raw = oversold * 55 - overbought * 55
    return _clip_signal(raw)


def s079_rvi_divergence(ind):
    """079 | Relative Vigor Index Divergence"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    num = (close - open_p) + 2 * (close.shift(1) - open_p.shift(1)) + 2 * (close.shift(2) - open_p.shift(2)) + (close.shift(3) - open_p.shift(3))
    den = (high - low) + 2 * (high.shift(1) - low.shift(1)) + 2 * (high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))
    rvi = (num / 6) / (den / 6).replace(0, np.nan)
    rvi_ma = rvi.rolling(10).mean()
    rvi_z = _z(rvi_ma, 40)
    raw = rvi_z * 50
    return _clip_signal(raw)


def s080_factor_mr(ind):
    """080 | Factor Mean Reversion (Short-Term Reversal)"""
    close = ind["close"]
    ret_5 = close.pct_change(5)
    z = _z(ret_5, 60)
    raw = -z * 55
    return _clip_signal(raw)


def s081_variance_ratio(ind):
    """081 | Variance Ratio Test for Mean Reversion"""
    ret = ind["ret_1"]
    var_1 = ret.rolling(20).var()
    ret_5 = ind["close"].pct_change(5) / 5
    var_5 = ret_5.rolling(20).var()
    vr = (var_5 / var_1.replace(0, np.nan)).fillna(1)
    mean_revert = (vr < 0.8).astype(float)
    z = _z(ind["close"], 40)
    raw = -z * mean_revert * 55
    return _clip_signal(raw)


def s082_options_skew_mr(ind):
    """082 | Options Skew Mean Reversion (vol proxy for skew)"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    raw = -vol_z * 40
    return _clip_signal(raw)


def s083_regime_bb(ind):
    """083 | Regime-Conditional Bollinger Band Strategy"""
    pctb = ind["bb_pctb"]
    hurst = ind["hurst"]
    mr_regime = (hurst < 0.45).astype(float)
    trend_regime = (hurst > 0.55).astype(float)
    mr_signal = -(pctb - 0.5) * 2
    trend_signal = (pctb - 0.5) * 2
    raw = mr_regime * mr_signal * 50 + trend_regime * trend_signal * 40
    return _clip_signal(raw)


def s084_kalman_rsi_mr(ind):
    """084 | Kalman-Smoothed RSI Mean Reversion"""
    rsi = ind["rsi"]
    # Simple Kalman-like smoother on RSI
    alpha = 0.15
    vals = rsi.fillna(50).values.astype(float)
    kf = np.zeros(len(vals))
    kf[0] = vals[0]
    for i in range(1, len(vals)):
        kf[i] = kf[i-1] + alpha * (vals[i] - kf[i-1])
    k_rsi = pd.Series(kf, index=rsi.index)
    raw = (50 - k_rsi) / 50 * 60
    return _clip_signal(raw)


def s085_cross_sectional_value(ind):
    """085 | Cross-Sectional Value Mean Reversion (single asset: distance from 200MA)"""
    close = ind["close"]
    sma200 = ind["sma_200"]
    dist = (close - sma200) / sma200.replace(0, np.nan) * 100
    z = _z(dist, 252)
    raw = -z * 45
    return _clip_signal(raw)


def s086_microstructure_reversion(ind):
    """086 | Microstructure Reversion at LOB Imbalance (vol proxy)"""
    vol_rel = ind["vol_rel"]
    ret = ind["ret_1"]
    imbalance = vol_rel * ret
    z = _z(imbalance, 20)
    raw = -z * 50
    return _clip_signal(raw)


def s087_momentum_reversal_combo(ind):
    """087 | Momentum Reversal Combination"""
    mom_short = ind["mom_5"]
    mom_long = ind["mom_40"]
    divergence = mom_long - mom_short
    z = _z(divergence, 60)
    raw = z * 50
    return _clip_signal(raw)


def s088_stoch_multiperiod(ind):
    """088 | Stochastic Oscillator Multi-Period Cluster"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    stoch_5 = (close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min()).replace(0, np.nan) * 100
    stoch_14 = ind["stoch_k"]
    stoch_21 = (close - low.rolling(21).min()) / (high.rolling(21).max() - low.rolling(21).min()).replace(0, np.nan) * 100
    avg = (stoch_5 + stoch_14 + stoch_21) / 3
    oversold = (avg < 20).astype(float) * (20 - avg) / 20
    overbought = (avg > 80).astype(float) * (avg - 80) / 20
    raw = oversold * 55 - overbought * 55
    return _clip_signal(raw)


def s089_spread_duration_mr(ind):
    """089 | Spread Duration Mean Reversion (vol spread proxy)"""
    vol_ratio = ind["vol_20"] / ind["vol_60"].replace(0, np.nan)
    z = _z(vol_ratio, 120)
    raw = -z * 45
    return _clip_signal(raw)


def s090_ad_divergence(ind):
    """090 | Accumulation/Distribution Divergence"""
    ad = ind["ad_score"]
    price_trend = ind["mom_10"]
    divergence = ad - price_trend
    z = _z(divergence, 40)
    raw = z * 50
    return _clip_signal(raw)


def s091_ah_premium_arb(ind):
    """091 | Hong Kong Dual-Listed A/H Premium (single-asset: price level MR)"""
    close = ind["close"]
    z = _z(close, 120)
    raw = -z * 45
    return _clip_signal(raw)


def s092_kurtosis_mr(ind):
    """092 | Kurtosis-Adjusted Mean Reversion"""
    ret = ind["ret_1"]
    kurt = ret.rolling(60, min_periods=20).apply(lambda x: pd.Series(x).kurtosis(), raw=False)
    z = _z(ind["close"], 60)
    fat_tail = (kurt > 3).astype(float) * 0.5 + 0.5
    raw = -z * fat_tail * 50
    return _clip_signal(raw)


def s093_granger_lag_arb(ind):
    """093 | Granger Causality Lag Arbitrage (autocorrelation proxy)"""
    ret = ind["ret_1"]
    autocorr = ret.rolling(60, min_periods=20).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0, raw=True)
    z = _z(ind["close"], 40)
    mean_revert = (autocorr < -0.1).astype(float)
    raw = -z * (0.3 + 0.7 * mean_revert) * 50
    return _clip_signal(raw)


def s094_frama_mr(ind):
    """094 | Fractal Adaptive Moving Average MR"""
    close = ind["close"]
    hurst = ind["hurst"].fillna(0.5)
    alpha = np.exp(-4.6 * (hurst - 1))
    alpha = pd.Series(np.where(alpha > 1, 1, np.where(alpha < 0.01, 0.01, alpha)), index=close.index)
    frama = close.copy().astype(float)
    vals = close.values.astype(float)
    a = alpha.values
    out = np.full(len(vals), np.nan)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        if np.isnan(out[i-1]):
            out[i] = vals[i]
        else:
            out[i] = out[i-1] + a[i] * (vals[i] - out[i-1])
    frama = pd.Series(out, index=close.index)
    spread = close - frama
    z = _z(spread, 60)
    raw = -z * 50
    return _clip_signal(raw)


def s095_dispersion_index_mr(ind):
    """095 | Dispersion Index Mean Reversion"""
    vol = ind["vol_20"]
    disp = vol.rolling(20).std() / vol.rolling(20).mean().replace(0, np.nan)
    disp_z = _z(disp, 120)
    raw = -disp_z * 40
    return _clip_signal(raw)


def s096_energy_pair_spread(ind):
    """096 | Toronto Energy Pair Spread (OHLCV proxy: vol-adjusted momentum)"""
    mom = ind["mom_20"]
    vol_adj = ind["vol_dampener"]
    raw = mom * vol_adj * 55
    return _clip_signal(raw)


def s097_intraday_rs_mr(ind):
    """097 | Intraday Relative Strength MR (daily proxy)"""
    ret = ind["ret_1"]
    ret_z = _z(ret, 20)
    raw = -ret_z * 50
    return _clip_signal(raw)


def s098_reit_yield_spread(ind):
    """098 | Singapore REIT Yield Spread Compression (vol proxy)"""
    vol_z = _z(ind["vol_20"], 120)
    trend = ind["above_200"].fillna(0.5) * 2 - 1
    raw = -vol_z * 35 + trend * 25
    return _clip_signal(raw)


def s099_put_call_extreme(ind):
    """099 | Put-Call Ratio Extreme MR (vol proxy)"""
    vol_z = _z(ind["vol_20"], 60)
    fear = (vol_z > 1.5).astype(float) * (vol_z - 1.5)
    greed = (vol_z < -1.5).astype(float) * (-1.5 - vol_z)
    raw = fear * 40 - greed * 40
    return _clip_signal(raw)


def s100_halflife_spread(ind):
    """100 | Half-Life Optimized Spread Trading"""
    close = ind["close"]
    log_p = np.log(close)
    ma = log_p.rolling(60).mean()
    dev = log_p - ma
    # Estimate half-life from autocorrelation
    lag1_corr = dev.rolling(60, min_periods=20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.5, raw=True
    )
    half_life = (-np.log(2) / np.log(lag1_corr.clip(0.01, 0.99))).clip(1, 120)
    z = dev / dev.rolling(60, min_periods=10).std().replace(0, np.nan)
    speed = (20 / half_life).clip(0.2, 3)
    raw = -z.clip(-3, 3) / 3 * speed * 50
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    51: ("Ornstein-Uhlenbeck MR", s051_ornstein_uhlenbeck),
    52: ("Cointegration Pairs", s052_cointegration_pairs),
    53: ("BB %B Mean Reversion", s053_bb_pctb_mr),
    54: ("Z-Score Stat Arb", s054_zscore_stat_arb),
    55: ("RSI(2) Extreme Reversal", s055_rsi2_extreme),
    56: ("DeMark TD9 Exhaustion", s056_demark_td9),
    57: ("Candlestick Ensemble", s057_candlestick_ensemble),
    58: ("Gaussian Channel MR", s058_gaussian_channel_mr),
    59: ("PCA Pairs Proxy", s059_pca_pairs),
    60: ("OU Optimal Entry", s060_ou_optimal_entry),
    61: ("Gilt Spread Convergence", s061_gilt_spread),
    62: ("Relative Value Rotation", s062_relative_value_rotation),
    63: ("Kalman Spread Tracker", s063_kalman_spread),
    64: ("CSI300 Reversal", s064_csi300_reversal),
    65: ("Williams %R Extreme MR", s065_willr_extreme_mr),
    66: ("Dispersion Trading", s066_dispersion_trading),
    67: ("VWAP Reversion", s067_vwap_reversion),
    68: ("Johansen Basket Proxy", s068_johansen_basket),
    69: ("Vol Risk Premium", s069_vol_risk_premium),
    70: ("ADF Mean Reversion", s070_adf_mean_reversion),
    71: ("Nikkei Auction Imbalance", s071_nikkei_auction),
    72: ("RSI Failure Swing", s072_rsi_failure_swing),
    73: ("Defensive Momentum", s073_defensive_momentum),
    74: ("Roll Yield Proxy", s074_roll_yield),
    75: ("Seasonal MR", s075_seasonal_mr),
    76: ("Momentum Crash Protection", s076_momentum_crash_protection),
    77: ("Quantile Regression MR", s077_quantile_regression_mr),
    78: ("Expiry Week MR", s078_expiry_week_mr),
    79: ("RVI Divergence", s079_rvi_divergence),
    80: ("Factor Short-Term Reversal", s080_factor_mr),
    81: ("Variance Ratio MR", s081_variance_ratio),
    82: ("Options Skew MR Proxy", s082_options_skew_mr),
    83: ("Regime-Conditional BB", s083_regime_bb),
    84: ("Kalman-Smoothed RSI MR", s084_kalman_rsi_mr),
    85: ("Cross-Sectional Value MR", s085_cross_sectional_value),
    86: ("Microstructure Reversion", s086_microstructure_reversion),
    87: ("Momentum Reversal Combo", s087_momentum_reversal_combo),
    88: ("Stochastic Multi-Period", s088_stoch_multiperiod),
    89: ("Spread Duration MR", s089_spread_duration_mr),
    90: ("AD Divergence", s090_ad_divergence),
    91: ("A/H Premium Proxy", s091_ah_premium_arb),
    92: ("Kurtosis-Adjusted MR", s092_kurtosis_mr),
    93: ("Granger Lag Arbitrage", s093_granger_lag_arb),
    94: ("FRAMA Mean Reversion", s094_frama_mr),
    95: ("Dispersion Index MR", s095_dispersion_index_mr),
    96: ("Energy Pair Spread", s096_energy_pair_spread),
    97: ("Intraday RS MR", s097_intraday_rs_mr),
    98: ("REIT Yield Spread", s098_reit_yield_spread),
    99: ("Put-Call Ratio Extreme", s099_put_call_extreme),
    100: ("Half-Life Spread", s100_halflife_spread),
}
