"""
SECTION X: HYBRID & ENSEMBLE STRATEGIES (451-500)
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

def s451_bma_signal_ensemble(ind):
    """451 | Bayesian Model Averaging Signal Ensemble"""
    close = ind["close"]
    ret = close.pct_change()
    # Multiple model signals
    trend = ind["trend_score"]
    rsi_sig = -(ind["rsi"] - 50) / 50
    mom_sig = _z(ind["mom_20"], 60)
    vol_sig = -_z(ind["vol_20"], 120)
    eff_sig = _z(ind["efficiency"], 60)
    # BMA: weight by recent accuracy
    sigs = [trend, rsi_sig, mom_sig, vol_sig, eff_sig]
    total = pd.Series(0.0, index=close.index)
    total_w = pd.Series(0.0, index=close.index)
    for s in sigs:
        acc = (np.sign(s.shift(1)) == np.sign(ret)).astype(float).rolling(60).mean().fillna(0.2)
        total = total + s * acc
        total_w = total_w + acc
    weighted = total / total_w.replace(0, np.nan)
    raw = weighted * 65
    return _clip_signal(raw)


def s452_regime_mom_mr_switch(ind):
    """452 | Regime-Conditional Momentum-Reversion Switch"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    rsi = ind["rsi"]
    # Low vol: momentum; High vol: mean reversion
    low_vol = (vol_z < -0.5).astype(float)
    high_vol = (vol_z > 0.5).astype(float)
    mid = 1 - low_vol - high_vol
    mom_sig = _z(mom, 60) * 55
    mr_sig = -(rsi - 50) / 50 * 55
    blend_sig = ind["trend_score"] * 45
    raw = low_vol * mom_sig + high_vol * mr_sig + mid * blend_sig
    return _clip_signal(raw)


def s453_kalman_adaptive_ma(ind):
    """453 | Kalman-Filtered Adaptive Moving Average"""
    close = ind["close"]
    # Kalman filter: adaptive level estimation
    vals = close.values.astype(float)
    n = len(vals)
    state = np.zeros(n)
    p = np.ones(n) * 1.0  # state variance
    q = 1e-5  # process noise
    r = 0.01  # observation noise
    state[0] = vals[0]
    for i in range(1, n):
        # Predict
        p_pred = p[i-1] + q
        # Update
        k = p_pred / (p_pred + r)
        state[i] = state[i-1] + k * (vals[i] - state[i-1])
        p[i] = (1 - k) * p_pred
    kf = pd.Series(state, index=close.index)
    dist = (close - kf) / ind["atr14"].replace(0, np.nan)
    raw = dist.clip(-3, 3) / 3 * 55
    return _clip_signal(raw)


def s454_ml_feature_rotation(ind):
    """454 | Machine Learning Feature Importance Rotation"""
    close = ind["close"]
    ret = close.pct_change()
    # Rotate features by recent performance
    features = {
        'mom': _z(ind["mom_20"], 60),
        'vol': -_z(ind["vol_20"], 120),
        'trend': ind["trend_score"],
        'rsi': -(ind["rsi"] - 50) / 50,
        'eff': _z(ind["efficiency"], 60),
    }
    total = pd.Series(0.0, index=close.index)
    for name, feat in features.items():
        perf = (np.sign(feat.shift(1)) * ret).rolling(60).mean().fillna(0)
        weight = (perf > 0).astype(float) * perf.abs()
        total = total + feat * weight
    raw = _z(total, 40) * 60
    return _clip_signal(raw)


def s455_wavelet_multiscale(ind):
    """455 | Wavelet Decomposition Multi-Scale Trading"""
    close = ind["close"]
    # Multi-scale: EMA at different timeframes as wavelet proxy
    d1 = close.ewm(span=5, adjust=False).mean() - close.ewm(span=10, adjust=False).mean()
    d2 = close.ewm(span=10, adjust=False).mean() - close.ewm(span=20, adjust=False).mean()
    d3 = close.ewm(span=20, adjust=False).mean() - close.ewm(span=40, adjust=False).mean()
    d4 = close.ewm(span=40, adjust=False).mean() - close.ewm(span=80, adjust=False).mean()
    # Weight by scale
    raw = (_z(d1, 20) * 0.15 + _z(d2, 40) * 0.25 + _z(d3, 60) * 0.35 + _z(d4, 80) * 0.25) * 65
    return _clip_signal(raw)


def s456_entropy_weighted(ind):
    """456 | Entropy-Weighted Portfolio Construction"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Entropy: uncertainty measure
    def rolling_entropy(x):
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))
    entropy = ret.rolling(60, min_periods=20).apply(rolling_entropy, raw=True)
    ent_z = _z(entropy, 120)
    # Low entropy (certain) = higher weight; high entropy = reduce
    certainty = (1 - ent_z.clip(0, 3) / 3)
    raw = certainty * ind["trend_score"] * 55
    return _clip_signal(raw)


def s457_tech_fund_macro(ind):
    """457 | Ensemble of Technical + Fundamental + Macro"""
    # Technical
    tech = ind["trend_score"]
    # Fundamental proxy: efficiency + mean reversion
    fund = _z(ind["efficiency"], 60) * 0.5 + (-_z(ind["close"].pct_change(120), 252)) * 0.5
    # Macro proxy: vol regime + long-term momentum
    macro = -_z(ind["vol_60"], 252) * 0.5 + _z(ind["mom_40"], 120) * 0.5
    raw = (tech * 0.4 + fund * 0.3 + macro * 0.3) * 65
    return _clip_signal(raw)


def s458_copula_tail(ind):
    """458 | Copula-Based Tail Dependence Strategy"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # Tail dependence proxy: extreme return clustering
    left_tail = (ret < ret.rolling(60).quantile(0.05)).astype(float).rolling(20).mean()
    right_tail = (ret > ret.rolling(60).quantile(0.95)).astype(float).rolling(20).mean()
    tail_asym = _z(right_tail - left_tail, 120)
    raw = tail_asym * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s459_hmm_trading(ind):
    """459 | Hidden Markov Model Regime Trading"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # HMM: 3-state (bull, bear, neutral)
    vol_z = _z(vol, 120)
    mom_z = _z(mom, 60)
    bull = (1 / (1 + np.exp(-(2 * mom_z - vol_z))))
    bear = (1 / (1 + np.exp(-(-2 * mom_z + vol_z))))
    state = bull - bear
    raw = state * 60
    return _clip_signal(raw)


def s460_dcc_portfolio(ind):
    """460 | Dynamic Conditional Correlation (DCC) Portfolio"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # DCC proxy: time-varying correlation with own volatility
    dcc = ret.rolling(60).corr(vol.pct_change()).fillna(0)
    dcc_z = _z(dcc, 120)
    # Low correlation = diversified; high = concentrated risk
    raw = -dcc_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s461_quality_junk(ind):
    """461 | Long-Short Quality-Junk Factor"""
    eff = ind["efficiency"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Quality: high efficiency, low vol, positive momentum
    quality = _z(eff, 60) - _z(vol, 120) + _z(mom, 60)
    raw = quality / 3 * 65
    return _clip_signal(raw)


def s462_stoch_vol(ind):
    """462 | Stochastic Volatility Model Signal"""
    vol = ind["vol_20"]
    # SV model: vol follows geometric random walk
    log_vol = np.log(vol.replace(0, np.nan))
    vol_of_vol = log_vol.rolling(20).std()
    vol_trend = log_vol.diff(10)
    z = _z(vol_trend, 60)
    # Falling vol = bullish; rising vol = bearish
    raw = -z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s463_cv_ensemble(ind):
    """463 | Cross-Validated Ensemble Factor Model"""
    close = ind["close"]
    ret = close.pct_change()
    # Cross-validate multiple signals on rolling window
    sigs = [
        ind["trend_score"],
        _z(ind["mom_20"], 60),
        -(ind["rsi"] - 50) / 50,
        -_z(ind["vol_20"], 120),
    ]
    # Simple cross-validation: use first half to select, second half to trade
    total = pd.Series(0.0, index=close.index)
    for s in sigs:
        sharpe = (s.shift(1) * ret).rolling(120).mean() / (s.shift(1) * ret).rolling(120).std().replace(0, np.nan)
        weight = sharpe.clip(0, 3) / 3
        total = total + s * weight.fillna(0.25)
    raw = total / len(sigs) * 80
    return _clip_signal(raw)


def s464_turbulence_insurance(ind):
    """464 | Turbulence Index Portfolio Insurance"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    z_ret = _z(ret, 60)
    turb = (z_ret ** 2).ewm(span=20, adjust=False).mean()
    turb_z = _z(turb, 120)
    # Insurance: reduce exposure when turbulent
    exposure = (1 - turb_z.clip(0, 3) / 3)
    raw = exposure * ind["trend_score"] * 60
    return _clip_signal(raw)


def s465_nn_vol_surface(ind):
    """465 | Neural Network Volatility Surface Arbitrage"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    ret = ind["ret_1"]
    # NN proxy: nonlinear combination of features
    x1 = _z(vol, 60)
    x2 = _z(vol_60, 120)
    x3 = _z(ret.rolling(20).sum(), 60)
    # Tanh activation (neural net proxy)
    hidden = np.tanh(x1 * 0.4 + x2 * 0.3 + x3 * 0.3)
    raw = hidden * 55
    return _clip_signal(raw)


def s466_genetic_algo(ind):
    """466 | Genetic Algorithm Strategy Evolution"""
    # Pre-evolved combination of signals (genetic algorithm result)
    mom_z = _z(ind["mom_20"], 60)
    rsi_n = (ind["rsi"] - 50) / 50
    trend = ind["trend_score"]
    vol_z = -_z(ind["vol_20"], 120)
    bb_n = (ind["bb_pctb"] - 0.5) * 2
    # Evolved weights
    raw = (0.28 * mom_z + 0.22 * trend + 0.18 * vol_z + 0.17 * rsi_n + 0.15 * bb_n) * 70
    return _clip_signal(raw)


def s467_frama_v2(ind):
    """467 | Fractal Adaptive Moving Average (FRAMA) v2"""
    close = ind["close"]
    hurst = ind["hurst"].fillna(0.5)
    fdi = 2 - hurst
    alpha = np.exp(-4.6 * (fdi - 1)).clip(0.01, 1.0)
    vals = close.values.astype(float)
    a = alpha.values.astype(float)
    frama = np.zeros(len(vals))
    frama[0] = vals[0]
    for i in range(1, len(vals)):
        frama[i] = frama[i-1] + a[i] * (vals[i] - frama[i-1])
    f = pd.Series(frama, index=close.index)
    # V2: add momentum overlay
    dist = (close - f) / ind["atr14"].replace(0, np.nan)
    mom_overlay = _z(ind["mom_10"], 40) * 0.3
    raw = dist.clip(-3, 3) / 3 * 40 + mom_overlay * 30
    return _clip_signal(raw)


def s468_rl_agent(ind):
    """468 | Reinforcement Learning Portfolio Agent"""
    close = ind["close"]
    ret = close.pct_change()
    # RL proxy: state -> action based on reward history
    state = _z(ind["mom_20"], 60)
    # Reward: directional accuracy
    reward = (np.sign(state.shift(1)) * ret).rolling(60).mean()
    reward_z = _z(reward, 120)
    # Agent learns: increase signal when reward is positive
    action = state * (1 + reward_z.clip(0, 2))
    raw = action * 45
    return _clip_signal(raw)


def s469_pairs_coint(ind):
    """469 | Pairs Trading with Cointegration"""
    close = ind["close"]
    # Cointegration proxy: mean reversion of spread from long-term trend
    sma_50 = ind["sma_50"]
    sma_200 = ind["sma_200"]
    spread = close / sma_200.replace(0, np.nan) - sma_50 / sma_200.replace(0, np.nan)
    z = _z(spread, 120)
    raw = -z * 50
    return _clip_signal(raw)


def s470_option_adj_mom(ind):
    """470 | Option-Adjusted Momentum"""
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    # Option-adjusted: momentum / vol (Sharpe-like)
    risk_adj_mom = mom / vol.replace(0, np.nan)
    z = _z(risk_adj_mom, 60)
    raw = z * 55
    return _clip_signal(raw)


def s471_factor_budget(ind):
    """471 | Risk Factor Budgeting with Factor Momentum"""
    mom_z = _z(ind["mom_20"], 60)
    vol_z = -_z(ind["vol_20"], 120)
    trend = ind["trend_score"]
    eff_z = _z(ind["efficiency"], 60)
    # Equal risk budget per factor
    factors = [mom_z, vol_z, trend, eff_z]
    inv_vol_weights = []
    for f in factors:
        f_vol = f.rolling(60).std().replace(0, np.nan)
        inv_vol_weights.append(1 / f_vol)
    total_iv = sum(inv_vol_weights)
    raw = sum([f * w / total_iv.replace(0, np.nan) for f, w in zip(factors, inv_vol_weights)]) * 60
    return _clip_signal(raw)


def s472_mcmc_portfolio(ind):
    """472 | Markov Chain Monte Carlo Portfolio Optimization"""
    close = ind["close"]
    ret = close.pct_change()
    vol = ind["vol_20"]
    # MCMC proxy: posterior mean of returns
    prior_mean = ret.rolling(252).mean()
    likelihood = ret.rolling(60).mean()
    # Bayesian update: weighted average
    precision_prior = 1 / vol.rolling(252).std().replace(0, np.nan) ** 2
    precision_like = 1 / vol.replace(0, np.nan) ** 2
    posterior = (precision_prior * prior_mean + precision_like * likelihood) / (precision_prior + precision_like)
    z = _z(posterior, 60)
    raw = z * 55
    return _clip_signal(raw)


def s473_transfer_learning(ind):
    """473 | Transfer Learning Cross-Asset Signals"""
    # Transfer: use multi-timeframe patterns that generalize
    mom_5 = _z(ind["mom_5"], 40)
    mom_20 = _z(ind["mom_20"], 60)
    vol_z = _z(ind["vol_20"], 120)
    bb_z = (ind["bb_pctb"] - 0.5) * 2
    # Robust features that transfer across assets
    raw = (mom_5 * 0.2 + mom_20 * 0.3 + (-vol_z) * 0.25 + bb_z * 0.25) * 60
    return _clip_signal(raw)


def s474_systematic_macro(ind):
    """474 | Systematic Macro Trend Following"""
    close = ind["close"]
    sma_10 = ind["sma_10"]
    sma_50 = ind["sma_50"]
    sma_200 = ind["sma_200"]
    vol = ind["vol_20"]
    # Multi-MA trend scoring
    above_10 = (close > sma_10).astype(float)
    above_50 = (close > sma_50).astype(float)
    above_200 = (close > sma_200).astype(float)
    trend_str = (above_10 + above_50 + above_200) / 3
    vol_target = 0.15 / np.sqrt(252)
    vol_scale = (vol_target / vol.replace(0, np.nan)).clip(0.3, 3)
    raw = (trend_str * 2 - 1) * vol_scale * 55
    return _clip_signal(raw)


def s475_black_litterman(ind):
    """475 | Black-Litterman Bayesian Asset Allocation"""
    close = ind["close"]
    ret = close.pct_change()
    vol = ind["vol_20"]
    # BL: equilibrium (market) + views (momentum)
    equilibrium = ret.rolling(252).mean()  # market implied
    view = ind["mom_20"] / 100  # momentum view
    tau = 0.05
    omega = vol ** 2 / 252
    sigma = vol ** 2 / 252
    # BL posterior
    posterior = ((1/tau * sigma) * equilibrium + (1/omega * view)) / (1/tau * sigma + 1/omega)
    z = _z(posterior, 60)
    raw = z * 55
    return _clip_signal(raw)


def s476_vwap_twap(ind):
    """476 | Optimal Execution VWAP/TWAP Hybrid"""
    close = ind["close"]
    volume = ind["volume"]
    # VWAP: volume-weighted average price
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    twap = close.rolling(20).mean()
    # Price vs VWAP/TWAP
    vwap_dist = (close - vwap) / ind["atr14"].replace(0, np.nan)
    twap_dist = (close - twap) / ind["atr14"].replace(0, np.nan)
    raw = (vwap_dist + twap_dist) / 2
    raw = raw.clip(-3, 3) / 3 * 55
    return _clip_signal(raw)


def s477_dispersion_trading(ind):
    """477 | Dispersion Trading (Index vs Single-Stock Vol)"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Dispersion: vol clustering as correlation proxy
    vol_ratio = vol / vol_60.replace(0, np.nan)
    disp = vol_ratio.rolling(20).std()
    disp_z = _z(disp, 120)
    # High dispersion = low correlation = buy singles
    raw = disp_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s478_smart_beta(ind):
    """478 | Smart Beta Multi-Factor ETF Construction"""
    mom_z = _z(ind["mom_20"], 60)
    vol_z = -_z(ind["vol_20"], 120)
    eff_z = _z(ind["efficiency"], 60)
    value_z = -_z(ind["close"].pct_change(120), 252)
    # Smart beta: equal weight of value, momentum, quality, low vol
    raw = (0.25 * mom_z + 0.25 * vol_z + 0.25 * eff_z + 0.25 * value_z) * 65
    return _clip_signal(raw)


def s479_kalman_regime(ind):
    """479 | Kalman Smoother Regime-Change Detection"""
    close = ind["close"]
    ret = close.pct_change()
    # Kalman smoother: detect regime changes in mean
    vals = ret.fillna(0).values.astype(float)
    n = len(vals)
    state = np.zeros(n)
    p = np.ones(n) * 0.01
    q = 1e-6
    r = 0.001
    state[0] = 0
    for i in range(1, n):
        p_pred = p[i-1] + q
        k = p_pred / (p_pred + r)
        state[i] = state[i-1] + k * (vals[i] - state[i-1])
        p[i] = (1 - k) * p_pred
    mu = pd.Series(state, index=close.index)
    regime_change = mu.diff().abs()
    z = _z(mu, 60)
    raw = z * 45 + _z(-regime_change, 40) * 15
    return _clip_signal(raw)


def s480_vrp_harvest(ind):
    """480 | Volatility Risk Premium Harvesting"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    ret = ind["ret_1"]
    # VRP: implied (vol_60 proxy) - realized (vol_20)
    vrp = _z(vol_60 - vol, 120)
    # Positive VRP = sell vol premium = bullish
    raw = vrp * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s481_sentiment_divergence(ind):
    """481 | Sentiment Divergence Cross-Market Signal"""
    rsi = ind["rsi"]
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    # Sentiment: RSI + volume flow
    sentiment = _z(rsi - 50, 60) + _z(volume / vol_ma - 1, 60)
    # Price divergence: momentum vs sentiment
    price = _z(mom, 60)
    divergence = sentiment - price
    raw = divergence * 35 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s482_intraday_seasonality(ind):
    """482 | Intraday Seasonality Pattern Exploitation"""
    close = ind["close"]
    open_p = ind["open"]
    # Day-of-week seasonality proxy: open-to-close patterns
    oc = (close - open_p) / open_p * 100
    oc_5d = oc.rolling(5).mean()  # weekly pattern
    oc_22d = oc.rolling(22).mean()  # monthly pattern
    raw = _z(oc_5d, 60) * 30 + _z(oc_22d, 120) * 25 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s483_hedge_fund_ensemble(ind):
    """483 | Autonomous Hedge Fund Ensemble Architecture"""
    close = ind["close"]
    ret = close.pct_change()
    vol = ind["vol_20"]
    # 5 sub-strategies
    s1_trend = ind["trend_score"]
    s2_mr = -(ind["rsi"] - 50) / 50
    s3_mom = _z(ind["mom_20"], 60)
    s4_vol = -_z(vol, 120)
    s5_carry = _z(ind["mom_5"] - ind["mom_40"] / 8, 60)
    # Risk-parity weighting
    strats = [s1_trend, s2_mr, s3_mom, s4_vol, s5_carry]
    total = pd.Series(0.0, index=close.index)
    total_w = pd.Series(0.0, index=close.index)
    for s in strats:
        s_vol = s.rolling(60).std().replace(0, np.nan)
        w = 1 / s_vol
        total = total + s * w
        total_w = total_w + w
    raw = (total / total_w.replace(0, np.nan)) * 60
    return _clip_signal(raw)


def s484_ir_max(ind):
    """484 | Information Ratio Maximization Portfolio"""
    close = ind["close"]
    ret = close.pct_change()
    mom = ind["mom_20"]
    # IR: excess return / tracking error
    excess = ret - ret.rolling(252).mean()
    te = excess.rolling(60).std().replace(0, np.nan)
    ir = excess.rolling(60).mean() / te
    ir_z = _z(ir, 120)
    raw = ir_z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s485_transformer_pred(ind):
    """485 | Transformer-Based Price Prediction"""
    close = ind["close"]
    # Attention proxy: weighted historical patterns
    ret_1 = ind["ret_1"]
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    # Self-attention: correlation between timeframes
    attn_1_5 = ret_1.rolling(20).corr(ret_5.shift(1)).fillna(0)
    attn_5_20 = ret_5.rolling(20).corr(ret_20.shift(5)).fillna(0)
    # Weighted prediction
    pred = attn_1_5 * ret_1 + attn_5_20 * ret_5
    z = _z(pred, 40)
    raw = z * 50 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s486_cross_freq_mom(ind):
    """486 | Cross-Frequency Momentum Synthesis"""
    mom_5 = _z(ind["mom_5"], 40)
    mom_10 = _z(ind["mom_10"], 60)
    mom_20 = _z(ind["mom_20"], 60)
    mom_40 = _z(ind["mom_40"], 120)
    # Frequency synthesis: phase-aligned momentum
    agreement = np.sign(mom_5) + np.sign(mom_10) + np.sign(mom_20) + np.sign(mom_40)
    phase_aligned = (agreement.abs() >= 3).astype(float)
    avg = (mom_5 + mom_10 + mom_20 + mom_40) / 4
    raw = avg * (1 + phase_aligned * 0.5) * 50
    return _clip_signal(raw)


def s487_risk_barometer(ind):
    """487 | Risk-On/Risk-Off Regime Barometer"""
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    rsi = ind["rsi"]
    adx = ind["adx"]
    vol_z = _z(vol, 120)
    mom_z = _z(mom, 60)
    # Risk-on: low vol, positive mom, trending
    risk_on = -vol_z * 0.3 + mom_z * 0.3 + (adx / 50 - 0.5) * 0.2 + ((rsi - 50) / 50) * 0.2
    raw = risk_on * 65
    return _clip_signal(raw)


def s488_maxdd_constrained(ind):
    """488 | Maximum Drawdown-Constrained Optimization"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Drawdown from rolling high
    rolling_max = close.rolling(60).max()
    dd = (close - rolling_max) / rolling_max
    dd_z = _z(dd, 120)
    # Reduce exposure during drawdowns
    exposure = (1 + dd_z.clip(-2, 0))  # 0 to 1
    raw = exposure * ind["trend_score"] * 60
    return _clip_signal(raw)


def s489_kalman_heavy_tail(ind):
    """489 | Kalman Filter with Heavy-Tailed Innovations"""
    close = ind["close"]
    ret = close.pct_change().fillna(0)
    # Student-t Kalman: adaptive observation noise
    vals = ret.values.astype(float)
    n = len(vals)
    state = np.zeros(n)
    p = np.ones(n) * 0.01
    q = 1e-6
    for i in range(1, n):
        p_pred = p[i-1] + q
        # Heavy-tailed: scale R by residual magnitude
        resid = abs(vals[i] - state[i-1])
        r = max(0.0005, 0.001 * (1 + resid * 10))  # adaptive R
        k = p_pred / (p_pred + r)
        state[i] = state[i-1] + k * (vals[i] - state[i-1])
        p[i] = (1 - k) * p_pred
    mu = pd.Series(state, index=close.index)
    z = _z(mu, 60)
    raw = z * 55
    return _clip_signal(raw)


def s490_implied_div_growth(ind):
    """490 | Implied Dividend Growth Signal"""
    close = ind["close"]
    sma_200 = ind["sma_200"]
    # Dividend growth proxy: long-term price appreciation
    growth = _z(close.pct_change(252), 252)
    yield_proxy = _z(sma_200 / close.replace(0, np.nan), 252)
    raw = growth * 30 + yield_proxy * 25 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s491_multi_asset_carry(ind):
    """491 | Multi-Asset Carry Portfolio"""
    mom_5 = ind["mom_5"]
    mom_40 = ind["mom_40"]
    vol = ind["vol_20"]
    # Carry: short-term excess over long-term
    carry = _z(mom_5 - mom_40 / 8, 60)
    vol_adj = ind["vol_dampener"]
    raw = carry * vol_adj * 55
    return _clip_signal(raw)


def s492_hrp(ind):
    """492 | Hierarchical Risk Parity (HRP)"""
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    eff = ind["efficiency"]
    # HRP: cluster-aware risk parity
    inv_vol = 1 / vol.replace(0, np.nan)
    quality = _z(eff, 60)
    mom_z = _z(mom, 60)
    # Hierarchical weighting: quality first, then vol, then momentum
    raw = (quality * 0.35 + _z(inv_vol, 60) * 0.35 + mom_z * 0.30) * 60
    return _clip_signal(raw)


def s493_em_gmm(ind):
    """493 | Expectation-Maximization Gaussian Mixture Trading"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # EM-GMM: 2-component mixture
    ret_z = _z(ret, 60)
    vol_z = _z(vol, 120)
    # Component 1: positive mean, low vol (bull)
    # Component 2: negative mean, high vol (bear)
    bull_p = 1 / (1 + np.exp(-(ret_z - vol_z)))
    raw = (bull_p - 0.5) * 120
    return _clip_signal(raw)


def s494_satellite_data(ind):
    """494 | Alternative Data Satellite Imagery Signal"""
    volume = ind["volume"]
    vol_ma = volume.rolling(60).mean().replace(0, np.nan)
    mom = ind["mom_20"]
    # Satellite proxy: volume anomalies as activity indicator
    activity = _z(volume / vol_ma, 60)
    raw = activity * np.sign(mom) * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s495_convexity_harvest(ind):
    """495 | Convexity Harvesting Through Options Structures"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    ret = ind["ret_1"]
    # Convexity: profit from large moves
    gamma_pnl = ret ** 2 - (vol ** 2 / 252)
    gamma_z = _z(gamma_pnl.rolling(20).sum(), 60)
    # Buy convexity (long gamma) when vol is cheap
    vol_z = _z(vol, 120)
    cheap = (vol_z < -0.5).astype(float)
    raw = cheap * gamma_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s496_kelly_bma(ind):
    """496 | Adaptive Position Sizing via Kelly-BMA Integration"""
    close = ind["close"]
    ret = close.pct_change()
    vol = ind["vol_20"]
    # Kelly: f = mu / sigma^2
    mu = ret.rolling(60).mean()
    sigma2 = vol ** 2 / 252
    kelly = mu / sigma2.replace(0, np.nan)
    kelly_z = _z(kelly, 120)
    # BMA: weight by model confidence
    trend = ind["trend_score"]
    raw = kelly_z * 0.4 * 50 + trend * 0.6 * 50
    return _clip_signal(raw)


def s497_spectral_cycle(ind):
    """497 | Spectral Analysis Cycle Trading"""
    close = ind["close"]
    # Spectral proxy: multi-period cycle decomposition
    periods = [10, 20, 40, 60]
    cycle_sum = pd.Series(0.0, index=close.index)
    n_bar = np.arange(len(close))
    for p in periods:
        cycle = np.sin(2 * np.pi * n_bar / p)
        ret_at_phase = (close.pct_change(p) * pd.Series(cycle, index=close.index))
        cycle_sum = cycle_sum + _z(ret_at_phase, p * 2) / len(periods)
    raw = cycle_sum * 50
    return _clip_signal(raw)


def s498_tail_aware_alloc(ind):
    """498 | Tail-Risk-Aware Asset Allocation"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # Tail risk: expected shortfall proxy
    es = ret.rolling(60).quantile(0.05)
    es_z = _z(es, 120)
    # High tail risk (very negative ES): reduce; low tail risk: increase
    risk_adj = es_z.clip(-2, 2) / 2
    raw = (1 + risk_adj) / 2 * ind["trend_score"] * 60
    return _clip_signal(raw)


def s499_unified_fusion(ind):
    """499 | Unified Signal Fusion Architecture"""
    close = ind["close"]
    ret = close.pct_change()
    # Fuse all signal families
    trend = ind["trend_score"]
    mom = _z(ind["mom_20"], 60)
    mr = -(ind["rsi"] - 50) / 50
    vol_sig = -_z(ind["vol_20"], 120)
    vol_flow = _z(ind["vol_flow"], 60) if "vol_flow" in ind else pd.Series(0.0, index=close.index)
    eff = _z(ind["efficiency"], 60)
    # Adaptive fusion: weight by recent IC
    sigs = [trend, mom, mr, vol_sig, vol_flow, eff]
    total = pd.Series(0.0, index=close.index)
    total_w = pd.Series(0.0, index=close.index)
    for s in sigs:
        ic = s.shift(1).rolling(60).corr(ret).fillna(0).abs()
        total = total + s * ic
        total_w = total_w + ic
    fused = total / total_w.replace(0, np.nan)
    raw = fused * 70
    return _clip_signal(raw)


def s500_grand_unified(ind):
    """500 | The Grand Unified Strategy: Adaptive Multi-Regime Meta-Ensemble"""
    close = ind["close"]
    ret = close.pct_change()
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    rsi = ind["rsi"]

    # Regime detection
    low_vol = (vol_z < -0.5).astype(float)
    high_vol = (vol_z > 0.5).astype(float)
    mid_vol = 1 - low_vol - high_vol

    # Sub-strategies
    s_trend = ind["trend_score"]
    s_mom = _z(mom, 60)
    s_mr = -(rsi - 50) / 50
    s_vol = -vol_z
    s_eff = _z(ind["efficiency"], 60)

    # Regime-conditional weights
    # Low vol: momentum + trend dominant
    w_low = s_trend * 0.35 + s_mom * 0.30 + s_eff * 0.20 + s_vol * 0.15
    # High vol: mean reversion + vol signal dominant
    w_high = s_mr * 0.35 + s_vol * 0.30 + s_trend * 0.20 + s_eff * 0.15
    # Mid: balanced
    w_mid = s_trend * 0.25 + s_mom * 0.25 + s_mr * 0.25 + s_eff * 0.15 + s_vol * 0.10

    # Meta-ensemble
    meta = low_vol * w_low + high_vol * w_high + mid_vol * w_mid

    # Adaptive confidence from recent performance
    perf = (np.sign(meta.shift(1)) * ret).rolling(60).mean()
    perf_z = _z(perf, 120)
    confidence = (1 + perf_z.clip(0, 2) * 0.3)

    # Vol targeting
    target_vol = 0.15 / np.sqrt(252)
    vol_scale = (target_vol / vol.replace(0, np.nan)).clip(0.3, 3)

    raw = meta * confidence * vol_scale * 55
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    451: ("BMA Signal Ensemble", s451_bma_signal_ensemble),
    452: ("Regime Mom/MR Switch", s452_regime_mom_mr_switch),
    453: ("Kalman Adaptive MA", s453_kalman_adaptive_ma),
    454: ("ML Feature Rotation", s454_ml_feature_rotation),
    455: ("Wavelet Multi-Scale", s455_wavelet_multiscale),
    456: ("Entropy Weighted", s456_entropy_weighted),
    457: ("Tech+Fund+Macro", s457_tech_fund_macro),
    458: ("Copula Tail Dep", s458_copula_tail),
    459: ("HMM Regime Trade", s459_hmm_trading),
    460: ("DCC Portfolio", s460_dcc_portfolio),
    461: ("Quality-Junk Long/Short", s461_quality_junk),
    462: ("Stochastic Vol", s462_stoch_vol),
    463: ("CV Ensemble Factor", s463_cv_ensemble),
    464: ("Turbulence Insurance", s464_turbulence_insurance),
    465: ("NN Vol Surface", s465_nn_vol_surface),
    466: ("Genetic Algo", s466_genetic_algo),
    467: ("FRAMA v2", s467_frama_v2),
    468: ("RL Portfolio Agent", s468_rl_agent),
    469: ("Pairs Cointegration", s469_pairs_coint),
    470: ("Option-Adj Momentum", s470_option_adj_mom),
    471: ("Factor Risk Budget", s471_factor_budget),
    472: ("MCMC Portfolio", s472_mcmc_portfolio),
    473: ("Transfer Learning", s473_transfer_learning),
    474: ("Systematic Macro", s474_systematic_macro),
    475: ("Black-Litterman", s475_black_litterman),
    476: ("VWAP/TWAP Hybrid", s476_vwap_twap),
    477: ("Dispersion Trading", s477_dispersion_trading),
    478: ("Smart Beta Multi", s478_smart_beta),
    479: ("Kalman Regime", s479_kalman_regime),
    480: ("VRP Harvesting", s480_vrp_harvest),
    481: ("Sentiment Divergence", s481_sentiment_divergence),
    482: ("Intraday Seasonality", s482_intraday_seasonality),
    483: ("HF Ensemble", s483_hedge_fund_ensemble),
    484: ("IR Maximization", s484_ir_max),
    485: ("Transformer Pred", s485_transformer_pred),
    486: ("Cross-Freq Mom", s486_cross_freq_mom),
    487: ("Risk Barometer", s487_risk_barometer),
    488: ("MaxDD Constrained", s488_maxdd_constrained),
    489: ("Kalman Heavy-Tail", s489_kalman_heavy_tail),
    490: ("Implied Div Growth", s490_implied_div_growth),
    491: ("Multi-Asset Carry", s491_multi_asset_carry),
    492: ("Hierarchical RP", s492_hrp),
    493: ("EM-GMM Trading", s493_em_gmm),
    494: ("Satellite Data", s494_satellite_data),
    495: ("Convexity Harvest", s495_convexity_harvest),
    496: ("Kelly-BMA Sizing", s496_kelly_bma),
    497: ("Spectral Cycle", s497_spectral_cycle),
    498: ("Tail-Aware Alloc", s498_tail_aware_alloc),
    499: ("Unified Fusion", s499_unified_fusion),
    500: ("Grand Unified Meta", s500_grand_unified),
}
