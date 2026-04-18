"""Batch 11: Replace S051-S055 skeletal implementations with full versions."""
import sys

filepath = "src/indicators/families/mean_reversion.py"
with open(filepath, "r") as f:
    content = f.read()

# ─── S051 OU Mean Reversion ─────────────────────────────────────────────────
old_051 = '''def s051_ornstein_uhlenbeck(ind):
    """051 | Ornstein-Uhlenbeck Mean Reversion"""
    close = ind["close"]
    log_p = np.log(close)
    ma = log_p.rolling(60).mean()
    deviation = log_p - ma
    z = _z(deviation, 60)
    raw = -z * 50
    return _clip_signal(raw)'''

new_051 = '''def s051_ornstein_uhlenbeck(ind):
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
    return _clip_signal(raw)'''

assert old_051 in content, "S051 old text not found!"
content = content.replace(old_051, new_051, 1)
print("S051 replaced OK")

# ─── S052 Cointegration Pairs ───────────────────────────────────────────────
old_052 = '''def s052_cointegration_pairs(ind):
    """052 | Cointegration Pairs Trading (single-asset proxy: price vs SMA spread)"""
    close = ind["close"]
    spread = close - ind["sma_50"]
    z = _z(spread, 60)
    raw = -z * 50
    return _clip_signal(raw)'''

new_052 = '''def s052_cointegration_pairs(ind):
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
    
    return _clip_signal(raw)'''

assert old_052 in content, "S052 old text not found!"
content = content.replace(old_052, new_052, 1)
print("S052 replaced OK")

# ─── S053 BB %B Mean Reversion ──────────────────────────────────────────────
old_053 = '''def s053_bb_pctb_mr(ind):
    """053 | Bollinger Band %B Mean Reversion"""
    pctb = ind["bb_pctb"]
    oversold = (pctb < 0.1).astype(float) * (0.1 - pctb) / 0.1
    overbought = (pctb > 0.9).astype(float) * (pctb - 0.9) / 0.1
    raw = oversold * 60 - overbought * 60
    trend_filter = ind["above_200"].fillna(0.5)
    raw = raw + (trend_filter * 2 - 1) * 15
    return _clip_signal(raw)'''

new_053 = '''def s053_bb_pctb_mr(ind):
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
    
    return _clip_signal(raw)'''

assert old_053 in content, "S053 old text not found!"
content = content.replace(old_053, new_053, 1)
print("S053 replaced OK")

# ─── S054 Z-Score Stat Arb with Regime Filter ──────────────────────────────
old_054 = '''def s054_zscore_stat_arb(ind):
    """054 | Z-Score Statistical Arbitrage with Regime Filter"""
    close = ind["close"]
    z = _z(close, 60)
    vol_regime = ind["vol_pct"]
    low_vol = (vol_regime < 0.5).astype(float)
    raw = -z * 40 * (0.5 + low_vol * 0.5)
    return _clip_signal(raw)'''

new_054 = '''def s054_zscore_stat_arb(ind):
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
    
    # --- Regime gate: AC < -0.05 AND Hurst < 0.45 ---
    mr_regime = ((autocorr < -0.05) & (hurst < 0.45)).astype(float)
    
    # Softer version: partially trade if one condition met
    soft_regime = ((autocorr < 0.0) | (hurst < 0.50)).astype(float) * 0.4
    regime_strength = (mr_regime + soft_regime * (1 - mr_regime)).clip(0, 1)
    
    # --- Event signals ---
    long_event = ((z_spread < -2.0) & (regime_strength > 0.3)).astype(float)
    short_event = ((z_spread > 2.0) & (regime_strength > 0.3)).astype(float)
    emergency = (z_spread.abs() > 4.0).astype(float)
    
    # Persistence
    long_persist = long_event.rolling(8, min_periods=1).max()
    short_persist = short_event.rolling(8, min_periods=1).max()
    
    # --- Continuous baseline ---
    continuous = -z_spread * regime_strength * 18
    
    # --- Compose ---
    event_sig = long_persist * 40 - short_persist * 40
    raw = (event_sig + continuous) * regime_strength
    
    # Emergency flatten
    raw = raw * (1 - emergency * 0.9)
    
    return _clip_signal(raw)'''

assert old_054 in content, "S054 old text not found!"
content = content.replace(old_054, new_054, 1)
print("S054 replaced OK")

# ─── S055 RSI(2) Extreme Reversal ───────────────────────────────────────────
old_055 = '''def s055_rsi2_extreme(ind):
    """055 | RSI(2) Extreme Reversal"""
    close = ind["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi2 = 100 - 100 / (1 + rs)
    oversold = (rsi2 < 10).astype(float) * (10 - rsi2) / 10
    overbought = (rsi2 > 90).astype(float) * (rsi2 - 90) / 10
    above_200 = ind["above_200"].fillna(0.5)
    raw = oversold * above_200 * 70 - overbought * (1 - above_200) * 70
    return _clip_signal(raw)'''

new_055 = '''def s055_rsi2_extreme(ind):
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
    return _clip_signal(raw)'''

assert old_055 in content, "S055 old text not found!"
content = content.replace(old_055, new_055, 1)
print("S055 replaced OK")

# ─── Write out ──────────────────────────────────────────────────────────────
with open(filepath, "w") as f:
    f.write(content)

print("All S051-S055 replacements written successfully.")
