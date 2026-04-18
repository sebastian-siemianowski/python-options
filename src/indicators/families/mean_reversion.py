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
    """051 | Ornstein-Uhlenbeck Mean Reversion"""
    close = ind["close"]
    log_p = np.log(close)
    ma = log_p.rolling(60).mean()
    deviation = log_p - ma
    z = _z(deviation, 60)
    raw = -z * 50
    return _clip_signal(raw)


def s052_cointegration_pairs(ind):
    """052 | Cointegration Pairs Trading (single-asset proxy: price vs SMA spread)"""
    close = ind["close"]
    spread = close - ind["sma_50"]
    z = _z(spread, 60)
    raw = -z * 50
    return _clip_signal(raw)


def s053_bb_pctb_mr(ind):
    """053 | Bollinger Band %B Mean Reversion"""
    pctb = ind["bb_pctb"]
    oversold = (pctb < 0.1).astype(float) * (0.1 - pctb) / 0.1
    overbought = (pctb > 0.9).astype(float) * (pctb - 0.9) / 0.1
    raw = oversold * 60 - overbought * 60
    trend_filter = ind["above_200"].fillna(0.5)
    raw = raw + (trend_filter * 2 - 1) * 15
    return _clip_signal(raw)


def s054_zscore_stat_arb(ind):
    """054 | Z-Score Statistical Arbitrage with Regime Filter"""
    close = ind["close"]
    z = _z(close, 60)
    vol_regime = ind["vol_pct"]
    low_vol = (vol_regime < 0.5).astype(float)
    raw = -z * 40 * (0.5 + low_vol * 0.5)
    return _clip_signal(raw)


def s055_rsi2_extreme(ind):
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
    return _clip_signal(raw)


def s056_demark_td9(ind):
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
    return _clip_signal(raw)


def s057_candlestick_ensemble(ind):
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
    return _clip_signal(raw)


def s058_gaussian_channel_mr(ind):
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
    return _clip_signal(raw)


def s059_pca_pairs(ind):
    """059 | PCA Pairs Trading (single-asset: principal residual from MAs)"""
    close = ind["close"]
    ma_blend = 0.5 * ind["sma_20"] + 0.3 * ind["sma_50"] + 0.2 * ind["sma_200"]
    residual = close - ma_blend
    z = _z(residual, 60)
    raw = -z * 50
    return _clip_signal(raw)


def s060_ou_optimal_entry(ind):
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
