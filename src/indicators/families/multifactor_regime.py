"""
SECTION VII: MULTIFACTOR & REGIME STRATEGIES (301-350)
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

def s301_fama_french_alpha(ind):
    """301 | Fama-French Five-Factor Alpha Capture"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Factor proxies: size (inverse vol), value (mean reversion), momentum
    size_f = _z(-vol, 60)  # low vol = small-cap proxy
    value_f = _z(-close.pct_change(60), 60)  # 60-day reversal = value
    mom_f = _z(mom, 60)
    quality_f = _z(ind["efficiency"], 60)
    alpha = 0.2 * size_f + 0.3 * value_f + 0.3 * mom_f + 0.2 * quality_f
    raw = alpha * 55
    return _clip_signal(raw)


def s302_hmm_regime(ind):
    """302 | Hidden Markov Model Regime Switching"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # HMM proxy: vol/momentum regime detection
    vol_z = _z(vol, 120)
    mom_z = _z(mom, 120)
    # Low vol + positive mom = bull
    # High vol + negative mom = bear
    bull_prob = (1 / (1 + np.exp(-(2 * mom_z - 1.5 * vol_z))))
    regime_signal = (bull_prob - 0.5) * 2
    raw = regime_signal * 55
    return _clip_signal(raw)


def s303_quality_momentum(ind):
    """303 | Quality-Momentum Composite Factor"""
    mom = _z(ind["mom_20"], 60)
    eff = _z(ind["efficiency"], 60)
    trend = ind["trend_score"]
    vol_adj = ind["vol_dampener"]
    # Quality: high efficiency + strong trend
    quality = (eff + trend) / 2
    composite = (0.5 * mom + 0.5 * quality) * vol_adj
    raw = composite * 65
    return _clip_signal(raw)


def s304_carry_factor(ind):
    """304 | Carry Factor Cross-Asset Strategy"""
    close = ind["close"]
    # Carry proxy: roll yield approximated by contango/backwardation in momentum
    short_mom = ind["mom_5"]
    long_mom = ind["mom_40"]
    carry = _z(short_mom - long_mom, 60)
    trend = ind["trend_score"]
    raw = carry * 35 + trend * 25
    return _clip_signal(raw)


def s305_low_vol_anomaly(ind):
    """305 | Low Volatility Anomaly Factor"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    # Low vol stocks outperform: buy low vol, sell high vol
    low_vol_sig = -vol_z
    # But combine with positive momentum for timing
    raw = low_vol_sig * 35 + _z(mom, 60) * 25
    return _clip_signal(raw)


def s306_adaptive_allocation(ind):
    """306 | Adaptive Asset Allocation (Butler/Philbrick)"""
    close = ind["close"]
    # Multi-timeframe momentum + volatility weighting
    mom_short = ind["mom_10"]
    mom_med = ind["mom_20"]
    mom_long = ind["mom_40"]
    vol_inv = 1 / ind["vol_20"].replace(0, np.nan)
    vol_weight = vol_inv / vol_inv.rolling(60).mean().replace(0, np.nan)
    avg_mom = (mom_short + mom_med + mom_long) / 3
    raw = _z(avg_mom, 60) * vol_weight.clip(0.5, 2) * 50
    return _clip_signal(raw)


def s307_sector_rotation(ind):
    """307 | Sector Rotation via Relative Strength Matrix"""
    close = ind["close"]
    mom_20 = ind["mom_20"]
    mom_40 = ind["mom_40"]
    vol = ind["vol_20"]
    # Relative strength: momentum rank proxy
    rs = _z(mom_20 + mom_40, 60)
    # Rotate into strong momentum, out of weak
    vol_adj = ind["vol_dampener"]
    raw = rs * vol_adj * 55
    return _clip_signal(raw)


def s308_vol_regime_conditional(ind):
    """308 | Volatility Regime Conditional Strategy Selection"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    rsi = ind["rsi"]
    # Low vol regime: trend following
    low_vol = (vol_z < -0.5).astype(float)
    # High vol regime: mean reversion
    high_vol = (vol_z > 0.5).astype(float)
    trend_sig = ind["trend_score"]
    mr_sig = -(rsi - 50) / 50
    raw = low_vol * trend_sig * 50 + high_vol * mr_sig * 50
    return _clip_signal(raw)


def s309_mom_crash_protection(ind):
    """309 | Momentum Crash Protection via Factor Crowding"""
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    vol_change = vol.pct_change(5)
    # Crowding proxy: everyone in same momentum trade = crash risk
    mom_z = _z(mom, 60)
    vol_spike = (vol_change > vol_change.rolling(60).quantile(0.9)).astype(float)
    # Normal: follow momentum; Crowded: reduce
    raw = mom_z * (1 - 0.8 * vol_spike) * 55
    return _clip_signal(raw)


def s310_xs_momentum(ind):
    """310 | Cross-Sectional Momentum with Industry Adjustment"""
    mom_10 = ind["mom_10"]
    mom_20 = ind["mom_20"]
    mom_40 = ind["mom_40"]
    # Industry-adjusted momentum proxy: detrend by long-term
    adj_mom = _z(mom_10, 60) - 0.3 * _z(mom_40, 120)
    raw = adj_mom * 55
    return _clip_signal(raw)


def s311_factor_timing(ind):
    """311 | Dynamic Factor Timing via Macroeconomic Variables"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    vol_60 = ind["vol_60"]
    # Macro proxy: vol trend as economic uncertainty
    vol_trend = _z(vol_60, 120)
    # Low uncertainty: momentum works; high uncertainty: quality works
    low_unc = (vol_trend < 0).astype(float)
    high_unc = (vol_trend > 0).astype(float)
    mom_sig = _z(mom, 60)
    quality_sig = _z(ind["efficiency"], 60)
    raw = low_unc * mom_sig * 40 + high_unc * quality_sig * 40 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s312_pca_residual(ind):
    """312 | Principal Component Analysis Residual Alpha"""
    close = ind["close"]
    # PCA proxy: remove market factor (SMA trend) to get idiosyncratic alpha
    market = ind["sma_50"]
    beta_proxy = close.rolling(60).corr(market).fillna(0)
    market_return = market.pct_change(20)
    asset_return = close.pct_change(20)
    residual = asset_return - beta_proxy * market_return
    z = _z(residual, 60)
    raw = z * 55
    return _clip_signal(raw)


def s313_bayesian_dlm(ind):
    """313 | Bayesian Dynamic Linear Model for Returns"""
    close = ind["close"]
    ret = close.pct_change()
    # DLM proxy: Kalman-like adaptive mean estimation
    alpha = 0.05
    vals = ret.fillna(0).values.astype(float)
    n = len(vals)
    state = np.zeros(n)
    state[0] = 0
    for i in range(1, n):
        state[i] = state[i-1] + alpha * (vals[i] - state[i-1])
    mu = pd.Series(state, index=close.index)
    z = _z(mu, 60)
    raw = z * 55
    return _clip_signal(raw)


def s314_disposition_contrarian(ind):
    """314 | Disposition Effect Contrarian Strategy"""
    close = ind["close"]
    # Disposition: investors sell winners too early, hold losers
    # Contrarian: buy recent winners that are being sold
    mom_20 = ind["mom_20"]
    vol = ind["volume"]
    vol_ma = vol.rolling(20).mean().replace(0, np.nan)
    # High volume on positive returns = disposition selling = buy signal
    winning = (mom_20 > 0).astype(float)
    high_vol = (vol > 1.3 * vol_ma).astype(float)
    raw = winning * high_vol * 40 - (1 - winning) * high_vol * 20 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s315_trend_vol_target(ind):
    """315 | Trend-Following Multi-Asset with Volatility Targeting"""
    trend = ind["trend_score"]
    vol = ind["vol_20"]
    target_vol = 0.15 / np.sqrt(252)  # ~15% annualized target
    vol_scale = target_vol / vol.replace(0, np.nan)
    raw = trend * vol_scale.clip(0.3, 3) * 50
    return _clip_signal(raw)


def s316_pead(ind):
    """316 | Earnings Surprise Drift Strategy (PEAD)"""
    close = ind["close"]
    volume = ind["volume"]
    # PEAD proxy: large moves with volume = earnings surprise
    ret_1 = ind["ret_1"]
    vol_rel = ind["vol_rel"]
    surprise = (ret_1.abs() > ret_1.rolling(60).std() * 2).astype(float)
    surprise_dir = np.sign(ret_1) * surprise
    # Drift: continue in surprise direction
    drift = surprise_dir.rolling(20).sum().clip(-3, 3) / 3
    raw = drift * 55
    return _clip_signal(raw)


def s317_open_close_momentum(ind):
    """317 | Intraday Momentum Pattern (Open-to-Close)"""
    close = ind["close"]
    open_p = ind["open"]
    oc_ret = (close - open_p) / open_p * 100
    oc_z = _z(oc_ret, 40)
    # Persistent open-to-close patterns
    consistency = oc_ret.rolling(5).mean() / oc_ret.rolling(5).std().replace(0, np.nan)
    raw = oc_z * 30 + consistency.clip(-2, 2) / 2 * 30
    return _clip_signal(raw)


def s318_event_catalyst(ind):
    """318 | Event-Driven Catalyst Strategy"""
    close = ind["close"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    ret = ind["ret_1"]
    # Event proxy: unusual volume + large move
    unusual_vol = (volume > 2 * vol_ma).astype(float)
    large_move = (ret.abs() > ret.rolling(60).std() * 1.5).astype(float)
    event = unusual_vol * large_move
    event_dir = event * np.sign(ret)
    # Post-event drift
    drift = event_dir.rolling(10).sum().clip(-3, 3) / 3
    raw = drift * 55
    return _clip_signal(raw)


def s319_ml_feature_importance(ind):
    """319 | Machine Learning Feature Importance Factor"""
    # Top features by typical importance: momentum, vol, trend, RSI, efficiency
    mom_z = _z(ind["mom_20"], 60)
    vol_z = _z(ind["vol_20"], 120)
    trend = ind["trend_score"]
    rsi_n = (ind["rsi"] - 50) / 50
    eff_z = _z(ind["efficiency"], 60)
    # Weighted by typical ML feature importance
    raw = (0.30 * mom_z + 0.10 * (-vol_z) + 0.25 * trend + 0.15 * rsi_n + 0.20 * eff_z) * 70
    return _clip_signal(raw)


def s320_bma_ensemble(ind):
    """320 | Bayesian Model Averaging Signal Ensemble"""
    # BMA: weight signals by their recent accuracy
    close = ind["close"]
    ret = close.pct_change()
    trend = ind["trend_score"]
    rsi_sig = -(ind["rsi"] - 50) / 50
    mom_sig = _z(ind["mom_20"], 60)
    vol_sig = -_z(ind["vol_20"], 120)
    # Recent accuracy weighting
    sigs = [trend, rsi_sig, mom_sig, vol_sig]
    weights = []
    for s in sigs:
        accuracy = (np.sign(s.shift(1)) == np.sign(ret)).astype(float).rolling(60).mean()
        weights.append(accuracy.fillna(0.25))
    total_w = sum(weights)
    total_w = total_w.replace(0, np.nan)
    weighted = sum([s * w for s, w in zip(sigs, weights)]) / total_w
    raw = weighted * 65
    return _clip_signal(raw)


def s321_seasonal_anomaly(ind):
    """321 | Seasonal Factor with Anomaly Calendar"""
    close = ind["close"]
    # Monthly seasonality proxy using rolling returns
    ret_22 = close.pct_change(22)
    ret_44 = close.pct_change(44)
    # Seasonal = monthly returns above/below average
    seasonal = _z(ret_22 - ret_44, 252)
    trend = ind["trend_score"]
    raw = seasonal * 30 + trend * 30
    return _clip_signal(raw)


def s322_frama(ind):
    """322 | Fractal Adaptive Moving Average (FRAMA) System"""
    close = ind["close"]
    hurst = ind["hurst"].fillna(0.5)
    n = 20
    # FRAMA alpha from fractal dimension
    fdi = 2 - hurst
    alpha = np.exp(-4.6 * (fdi - 1))
    alpha = alpha.clip(0.01, 1.0)
    vals = close.values.astype(float)
    a = alpha.values.astype(float)
    frama = np.zeros(len(vals))
    frama[0] = vals[0]
    for i in range(1, len(vals)):
        frama[i] = frama[i-1] + a[i] * (vals[i] - frama[i-1])
    f = pd.Series(frama, index=close.index)
    dist = (close - f) / ind["atr14"].replace(0, np.nan)
    raw = dist.clip(-3, 3) / 3 * 55
    return _clip_signal(raw)


def s323_risk_parity_trend(ind):
    """323 | Risk Parity with Trend Overlay"""
    vol = ind["vol_20"]
    trend = ind["trend_score"]
    # Risk parity: inverse vol weighting
    vol_inv = 1 / vol.replace(0, np.nan)
    vol_inv_z = _z(vol_inv, 60)
    raw = vol_inv_z * 25 + trend * 35
    return _clip_signal(raw)


def s324_short_interest_squeeze(ind):
    """324 | Short Interest Ratio Squeeze Detection"""
    close = ind["close"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    # Short squeeze proxy: sharp up move + very high volume
    ret = ind["ret_1"]
    vol_spike = (volume > 3 * vol_ma).astype(float)
    sharp_up = (ret > ret.rolling(60).std() * 2).astype(float)
    squeeze = vol_spike * sharp_up
    # Post-squeeze: continue up or mean revert
    post = squeeze.rolling(5).sum()
    raw = post * 30 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s325_macro_regime(ind):
    """325 | Macro Regime Indicator (MRI) Dashboard"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    mom = ind["mom_20"]
    mom_40 = ind["mom_40"]
    # Macro regime from vol trend + momentum trend
    vol_regime = _z(vol_60, 120)
    mom_regime = _z(mom_40, 120)
    # Risk-on: low vol + positive mom; Risk-off: high vol + negative mom
    mri = -vol_regime * 0.5 + mom_regime * 0.5
    raw = mri * 60
    return _clip_signal(raw)


def s326_value_momentum_barbell(ind):
    """326 | Value-Momentum Barbell Strategy"""
    close = ind["close"]
    # Value: long-term reversal
    value = -_z(close.pct_change(120), 120)
    # Momentum: short-term continuation
    momentum = _z(ind["mom_20"], 60)
    # Barbell: equal weight
    raw = (0.5 * value + 0.5 * momentum) * 60
    return _clip_signal(raw)


def s327_turbulence_index(ind):
    """327 | Turbulence Index Portfolio Protection"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # Turbulence: squared standardized returns
    z_ret = _z(ret, 60)
    turb = z_ret ** 2
    turb_ma = turb.ewm(span=20, adjust=False).mean()
    turb_z = _z(turb_ma, 120)
    # High turbulence: reduce risk; low: increase
    risk_adj = (1 - turb_z.clip(0, 3) / 3) * 2 - 1
    raw = risk_adj * ind["trend_score"] * 55
    return _clip_signal(raw)


def s328_absorption_ratio(ind):
    """328 | Absorption Ratio Market Fragility"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Absorption ratio proxy: vol clustering (vol_20 / vol_60)
    ar = vol / vol_60.replace(0, np.nan)
    ar_z = _z(ar, 120)
    # High absorption = fragile market
    fragile = (ar_z > 1.0).astype(float)
    raw = (1 - 2 * fragile) * ind["trend_score"] * 50
    return _clip_signal(raw)


def s329_xs_vol_factor(ind):
    """329 | Cross-Sectional Volatility Factor"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    # Low vol = outperform (vol anomaly)
    low_vol = (-vol_z).clip(-2, 2)
    # Combine with momentum for timing
    raw = low_vol * 30 + _z(mom, 60) * 30
    return _clip_signal(raw)


def s330_yield_curve_tilt(ind):
    """330 | Tactical Tilting via Yield Curve Signals"""
    close = ind["close"]
    # Yield curve proxy: long-term vs short-term momentum
    long_mom = ind["mom_40"]
    short_mom = ind["mom_5"]
    spread = _z(long_mom - short_mom, 120)
    # Steepening (positive spread) = risk-on
    raw = spread * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s331_sentiment_composite(ind):
    """331 | Sentiment Composite (AAII + Put/Call + VIX)"""
    rsi = ind["rsi"]
    vol = ind["vol_20"]
    mom = ind["mom_10"]
    # Sentiment proxy: extreme RSI + vol + momentum
    fear = ((rsi < 30).astype(float) + _z(vol, 60).clip(0, 3) / 3).clip(0, 2)
    greed = ((rsi > 70).astype(float) + (-_z(vol, 60)).clip(0, 3) / 3).clip(0, 2)
    # Contrarian: buy fear, sell greed
    raw = (fear - greed) * 40
    return _clip_signal(raw)


def s332_gmm_cluster(ind):
    """332 | Gaussian Mixture Model Clustering Regime"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # GMM proxy: classify by return + vol space
    ret_z = _z(ret, 60)
    vol_z = _z(vol, 120)
    # Cluster 1: low vol, positive returns (bull)
    # Cluster 2: high vol, negative returns (bear)
    bull_prob = 1 / (1 + np.exp(-(ret_z - vol_z)))
    raw = (bull_prob - 0.5) * 120
    return _clip_signal(raw)


def s333_lottery_demand(ind):
    """333 | Lottery Demand Factor (MAX Effect)"""
    close = ind["close"]
    ret = ind["ret_1"]
    # MAX effect: maximum daily return over past month
    max_ret = ret.rolling(22).max()
    # High MAX = lottery stock = underperform
    max_z = _z(max_ret, 120)
    raw = -max_z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s334_overnight_intraday(ind):
    """334 | Overnight vs Intraday Return Decomposition"""
    close = ind["close"]
    open_p = ind["open"]
    # Overnight: open - prev close
    overnight = (open_p - close.shift(1)) / close.shift(1) * 100
    # Intraday: close - open
    intraday = (close - open_p) / open_p * 100
    # Overnight premium: institutions; Intraday: retail
    on_z = _z(overnight.rolling(10).sum(), 60)
    id_z = _z(intraday.rolling(10).sum(), 60)
    raw = on_z * 30 + id_z * 25
    return _clip_signal(raw)


def s335_idio_vol(ind):
    """335 | Idiosyncratic Volatility Puzzle Factor"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Idiosyncratic vol proxy: vol after removing market trend
    market_vol = vol.rolling(60).mean()
    idio = vol - market_vol
    idio_z = _z(idio, 120)
    # Low idio vol = outperform (puzzle)
    raw = -idio_z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s336_equity_duration(ind):
    """336 | Equity Duration Factor"""
    close = ind["close"]
    mom_40 = ind["mom_40"]
    vol = ind["vol_20"]
    # Duration proxy: sensitivity to discount rate (vol as proxy)
    duration = _z(close.pct_change(120), 120)
    rate_proxy = _z(vol, 120)
    # High duration + falling rates (vol) = outperform
    raw = duration * (-rate_proxy) * 35 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s337_network_centrality(ind):
    """337 | Network Centrality Systemic Risk"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Centrality proxy: correlation with own volatility persistence
    vol_acf = vol.rolling(20).corr(vol.shift(5)).fillna(0)
    centrality = vol_acf.abs()
    # High centrality = systemic; reduce exposure in stress
    vol_z = _z(vol, 120)
    raw = (1 - centrality * vol_z.clip(0, 3) / 3) * ind["trend_score"] * 55
    return _clip_signal(raw)


def s338_iv_surface_momentum(ind):
    """338 | Implied Volatility Surface Momentum"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # IV surface proxy: vol term structure
    term_spread = _z(vol_60 - vol, 120)
    term_mom = term_spread.diff(5)
    z = _z(term_mom, 40)
    raw = z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s339_herding(ind):
    """339 | Herding Indicator Cross-Sectional"""
    close = ind["close"]
    volume = ind["volume"]
    ret = ind["ret_1"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    # Herding proxy: volume concentration + directional consistency
    vol_conc = volume / vol_ma
    ret_cons = ret.rolling(10).apply(lambda x: (np.sign(x) == np.sign(x.iloc[-1])).mean(), raw=False)
    herding = vol_conc * ret_cons.fillna(0.5)
    h_z = _z(herding, 60)
    # Contrarian to extreme herding
    raw = -h_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s340_credit_impulse(ind):
    """340 | Credit Impulse Leading Indicator"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Credit impulse proxy: change in vol acceleration
    vol_accel = vol.diff().diff()
    impulse = -_z(vol_accel, 60)  # decreasing vol accel = positive credit impulse
    raw = impulse * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s341_relative_value(ind):
    """341 | Relative Value Multi-Country Equity"""
    close = ind["close"]
    # Relative value: deviation from long-term trend
    sma_200 = ind["sma_200"]
    dev = (close - sma_200) / sma_200.replace(0, np.nan) * 100
    dev_z = _z(dev, 252)
    # Mean revert from extremes, trend follow in middle
    extreme = (dev_z.abs() > 1.5).astype(float)
    raw = extreme * (-dev_z) * 30 + (1 - extreme) * ind["trend_score"] * 40
    return _clip_signal(raw)


def s342_implied_corr(ind):
    """342 | Implied Correlation Trading Strategy"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Implied correlation proxy: vol ratio changes
    ratio = vol / vol_60.replace(0, np.nan)
    ratio_z = _z(ratio, 120)
    # High correlation = systemic risk; diversification doesn't help
    raw = -ratio_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s343_attention_trading(ind):
    """343 | Attention-Based Trading (Google Trends)"""
    volume = ind["volume"]
    vol_ma = volume.rolling(60).mean().replace(0, np.nan)
    # Attention proxy: abnormal volume
    attention = _z(volume / vol_ma, 60)
    ret = ind["ret_1"]
    # High attention + positive ret = momentum; high attention + negative = fade
    raw = attention * np.sign(ret.rolling(5).sum()) * 30 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s344_supply_chain(ind):
    """344 | Supply Chain Network Alpha"""
    close = ind["close"]
    mom = ind["mom_20"]
    eff = ind["efficiency"]
    # Supply chain proxy: leading indicator from efficiency + momentum
    lead = _z(eff.shift(5), 60)
    lag = _z(mom, 60)
    alpha = lead - lag  # lead-lag spread
    raw = alpha * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s345_entropy_regime(ind):
    """345 | Entropy-Based Market Regime Detection"""
    ret = ind["ret_1"]
    # Shannon entropy of return distribution
    def rolling_entropy(x):
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))
    entropy = ret.rolling(60, min_periods=20).apply(rolling_entropy, raw=True)
    ent_z = _z(entropy, 120)
    # High entropy = uncertain; low entropy = structured
    structured = (ent_z < -0.5).astype(float)
    raw = structured * ind["trend_score"] * 45 + (1 - structured) * _z(ind["mom_20"], 60) * 25
    return _clip_signal(raw)


def s346_anchoring_52wk(ind):
    """346 | Anchoring Bias Factor (52-Week High Proximity)"""
    close = ind["close"]
    high_252 = ind["high"].rolling(252, min_periods=60).max()
    proximity = close / high_252.replace(0, np.nan)
    # Near 52-week high: anchoring bias, investors slow to react
    near_high = (proximity > 0.9).astype(float) * (proximity - 0.9) / 0.1
    far_high = (proximity < 0.7).astype(float) * (0.7 - proximity) / 0.3
    raw = near_high * 40 - far_high * 30 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s347_vol_of_vol(ind):
    """347 | Volatility-of-Volatility (VoV) Premium"""
    vol = ind["vol_20"]
    vov = vol.rolling(20).std() / vol.rolling(20).mean().replace(0, np.nan)
    vov_z = _z(vov, 120)
    # High VoV = uncertainty premium; reduce risk
    risk_adj = (1 - vov_z.clip(0, 3) / 3)
    raw = risk_adj * ind["trend_score"] * 55
    return _clip_signal(raw)


def s348_geopolitical_risk(ind):
    """348 | Geopolitical Risk Premium Factor"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Geopolitical proxy: large negative returns + vol spike
    shock = ((ret < -ret.rolling(60).std() * 2) &
             (vol > vol.rolling(60).quantile(0.8))).astype(float)
    post_shock = shock.rolling(20).sum()
    # Buy after geopolitical shocks (premium capture)
    raw = post_shock.clip(0, 3) / 3 * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s349_profitability_factor(ind):
    """349 | Profitability Factor (Gross Profitability Premium)"""
    close = ind["close"]
    eff = ind["efficiency"]
    mom = ind["mom_20"]
    # Profitability proxy: efficiency + momentum consistency
    profit = _z(eff, 60)
    consistency = mom.rolling(60).apply(lambda x: (x > 0).mean(), raw=True)
    raw = profit * 30 + _z(consistency, 60) * 30
    return _clip_signal(raw)


def s350_dynamic_hedge(ind):
    """350 | Dynamic Hedge Ratio with Regime Conditioning"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    vol_z = _z(vol, 120)
    # Hedge ratio: increase hedging in high vol, decrease in low vol
    hedge = vol_z.clip(0, 3) / 3
    # Net exposure = trend - hedge
    trend = ind["trend_score"]
    raw = trend * (1 - hedge * 0.6) * 55
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    301: ("Fama-French Alpha", s301_fama_french_alpha),
    302: ("HMM Regime Switch", s302_hmm_regime),
    303: ("Quality-Momentum", s303_quality_momentum),
    304: ("Carry Factor", s304_carry_factor),
    305: ("Low Vol Anomaly", s305_low_vol_anomaly),
    306: ("Adaptive Allocation", s306_adaptive_allocation),
    307: ("Sector Rotation RS", s307_sector_rotation),
    308: ("Vol Regime Cond", s308_vol_regime_conditional),
    309: ("Mom Crash Protect", s309_mom_crash_protection),
    310: ("XS Momentum", s310_xs_momentum),
    311: ("Factor Timing", s311_factor_timing),
    312: ("PCA Residual Alpha", s312_pca_residual),
    313: ("Bayesian DLM", s313_bayesian_dlm),
    314: ("Disposition Contrarian", s314_disposition_contrarian),
    315: ("Trend + Vol Target", s315_trend_vol_target),
    316: ("PEAD Drift", s316_pead),
    317: ("Open-Close Momentum", s317_open_close_momentum),
    318: ("Event Catalyst", s318_event_catalyst),
    319: ("ML Feature Importance", s319_ml_feature_importance),
    320: ("BMA Signal Ensemble", s320_bma_ensemble),
    321: ("Seasonal Anomaly", s321_seasonal_anomaly),
    322: ("FRAMA System", s322_frama),
    323: ("Risk Parity + Trend", s323_risk_parity_trend),
    324: ("Short Squeeze", s324_short_interest_squeeze),
    325: ("Macro Regime MRI", s325_macro_regime),
    326: ("Value-Mom Barbell", s326_value_momentum_barbell),
    327: ("Turbulence Index", s327_turbulence_index),
    328: ("Absorption Ratio", s328_absorption_ratio),
    329: ("XS Vol Factor", s329_xs_vol_factor),
    330: ("Yield Curve Tilt", s330_yield_curve_tilt),
    331: ("Sentiment Composite", s331_sentiment_composite),
    332: ("GMM Cluster Regime", s332_gmm_cluster),
    333: ("Lottery Demand MAX", s333_lottery_demand),
    334: ("Overnight/Intraday", s334_overnight_intraday),
    335: ("Idiosyncratic Vol", s335_idio_vol),
    336: ("Equity Duration", s336_equity_duration),
    337: ("Network Centrality", s337_network_centrality),
    338: ("IV Surface Momentum", s338_iv_surface_momentum),
    339: ("Herding Indicator", s339_herding),
    340: ("Credit Impulse", s340_credit_impulse),
    341: ("Relative Value", s341_relative_value),
    342: ("Implied Correlation", s342_implied_corr),
    343: ("Attention Trading", s343_attention_trading),
    344: ("Supply Chain Alpha", s344_supply_chain),
    345: ("Entropy Regime", s345_entropy_regime),
    346: ("52-Wk High Anchor", s346_anchoring_52wk),
    347: ("Vol-of-Vol Premium", s347_vol_of_vol),
    348: ("Geopolitical Risk", s348_geopolitical_risk),
    349: ("Profitability Factor", s349_profitability_factor),
    350: ("Dynamic Hedge Ratio", s350_dynamic_hedge),
}
