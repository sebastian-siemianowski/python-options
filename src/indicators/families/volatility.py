"""
SECTION III: VOLATILITY STRATEGIES (101-150)
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


def _garch_vol(ret, omega=1e-6, alpha=0.08, beta=0.90):
    """Simple GARCH(1,1) variance series."""
    r = ret.fillna(0).values.astype(float)
    n = len(r)
    var = np.zeros(n)
    var[0] = np.var(r[:20]) if n > 20 else 1e-4
    for i in range(1, n):
        var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]
    return pd.Series(np.sqrt(var) * np.sqrt(252), index=ret.index)


def _parkinson_vol(high, low, window=20):
    """Parkinson range-based volatility estimator."""
    ln_hl = np.log(high / low.replace(0, np.nan))
    return np.sqrt((ln_hl ** 2).rolling(window, min_periods=5).mean() / (4 * np.log(2)) * 252)


def _yang_zhang_vol(open_p, high, low, close, window=20):
    """Yang-Zhang volatility estimator."""
    o = np.log(open_p / close.shift(1)).fillna(0)
    c = np.log(close / open_p).fillna(0)
    rs = np.log(high / close).fillna(0) * np.log(high / open_p).fillna(0) + \
         np.log(low / close).fillna(0) * np.log(low / open_p).fillna(0)
    n_days = window
    k = 0.34 / (1.34 + (n_days + 1) / (n_days - 1))
    o_var = o.rolling(window, min_periods=5).var()
    c_var = c.rolling(window, min_periods=5).var()
    rs_var = rs.rolling(window, min_periods=5).mean()
    yz = np.sqrt((o_var + k * c_var + (1 - k) * rs_var).clip(0) * 252)
    return yz


def _garman_klass_vol(open_p, high, low, close, window=20):
    """Garman-Klass volatility estimator."""
    u = np.log(high / open_p)
    d = np.log(low / open_p)
    c_part = np.log(close / open_p)
    gk = 0.5 * (u - d) ** 2 - (2 * np.log(2) - 1) * c_part ** 2
    return np.sqrt(gk.rolling(window, min_periods=5).mean().clip(0) * 252)


# ═════════════════════════════════════════════════════════════════════════════

def s101_garch_vol_breakout(ind):
    """101 | GARCH(1,1) Conditional Volatility Breakout"""
    ret = ind["ret_1"]
    garch = _garch_vol(ret)
    lr_vol = garch.rolling(252, min_periods=60).mean()
    vr = garch / lr_vol.replace(0, np.nan)
    # Low vol = pre-breakout opportunity (long); High vol decaying = short vol
    suppressed = (vr < 0.7).astype(float) * (0.7 - vr) / 0.3
    elevated_decay = ((vr > 1.5) & (vr < vr.shift(5))).astype(float)
    trend = ind["trend_score"]
    raw = suppressed * 40 + elevated_decay * (-30) + trend * 30
    return _clip_signal(raw)


def s102_vix_term_structure(ind):
    """102 | VIX Term Structure Roll Strategy (vol proxy)"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    contango = (vol_60 - vol_20) / vol_60.replace(0, np.nan)
    contango_z = _z(contango, 120)
    raw = contango_z * 45 + ind["mom_20"] * 20
    return _clip_signal(raw)


def s103_rv_iv_convergence(ind):
    """103 | Realized vs. Implied Volatility Convergence (proxy)"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    vrp = vol_60 - vol_20  # proxy for IV - RV
    vrp_z = _z(vrp, 120)
    raw = vrp_z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s104_garman_klass_vol(ind):
    """104 | Garman-Klass High-Low Volatility Estimator"""
    gk = _garman_klass_vol(ind["open"], ind["high"], ind["low"], ind["close"])
    gk_z = _z(gk, 60)
    # Low GK vol = breakout setup; high GK vol = mean revert
    raw = -gk_z * 35 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s105_straddle_condor(ind):
    """105 | Straddle Iron Condor Volatility Harvest (vol proxy)"""
    vol = ind["vol_20"]
    vol_pct = ind["vol_pct"]
    bb_width = ind["bb_width"]
    # High vol + wide BB = sell vol (condor); low vol = buy vol (straddle)
    sell_vol = (vol_pct > 0.8).astype(float)
    buy_vol = (vol_pct < 0.2).astype(float)
    raw = -sell_vol * 40 + buy_vol * 40 + ind["mom_score"] * 30
    return _clip_signal(raw)


def s106_yang_zhang_regime(ind):
    """106 | Yang-Zhang Volatility Regime Detection"""
    yz = _yang_zhang_vol(ind["open"], ind["high"], ind["low"], ind["close"])
    yz_z = _z(yz, 120)
    trend = ind["trend_score"]
    # Low vol regime: trend following; High vol regime: cautious
    low_vol = (yz_z < -0.5).astype(float)
    high_vol = (yz_z > 1.0).astype(float)
    raw = low_vol * trend * 50 - high_vol * 30 + (1 - low_vol - high_vol) * trend * 25
    return _clip_signal(raw)


def s107_butterfly_vol_smile(ind):
    """107 | Butterfly Spread Volatility Smile Trade (proxy)"""
    vol = ind["vol_20"]
    vol_ma = vol.rolling(60).mean()
    smile_proxy = (vol - vol_ma).abs() / vol_ma.replace(0, np.nan)
    z = _z(smile_proxy, 120)
    raw = -z * 40 + ind["mom_20"] * 25
    return _clip_signal(raw)


def s108_atr_channel_breakout(ind):
    """108 | ATR Channel Breakout with Volatility Sizing"""
    close = ind["close"]
    atr = ind["atr14"]
    ema20 = ind["ema_21"]
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    breakout_up = (close > upper).astype(float)
    breakout_dn = (close < lower).astype(float)
    vol_size = ind["vol_dampener"]
    raw = (breakout_up - breakout_dn) * vol_size * 65
    return _clip_signal(raw)


def s109_ewma_vol_cone(ind):
    """109 | EWMA Volatility Cone Strategy"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    p10 = vol_60.rolling(252, min_periods=60).quantile(0.10)
    p90 = vol_60.rolling(252, min_periods=60).quantile(0.90)
    low_cone = (vol_20 < p10).astype(float)
    high_cone = (vol_20 > p90).astype(float)
    raw = low_cone * 50 - high_cone * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s110_variance_swap(ind):
    """110 | Variance Swap Replication Strategy (proxy)"""
    ret = ind["ret_1"]
    rv_20 = ret.rolling(20).std() * np.sqrt(252)
    rv_60 = ret.rolling(60).std() * np.sqrt(252)
    var_spread = rv_60 ** 2 - rv_20 ** 2
    z = _z(var_spread, 120)
    raw = z * 40
    return _clip_signal(raw)


def s111_parkinson_breakout(ind):
    """111 | Parkinson Number Breakout Detection"""
    pk = _parkinson_vol(ind["high"], ind["low"])
    pk_z = _z(pk, 60)
    breakout = (pk_z > 1.5).astype(float) * ind["trend_score"]
    compress = (pk_z < -1.0).astype(float)
    raw = breakout * 50 + compress * 30
    return _clip_signal(raw)


def s112_volga_vol_of_vol(ind):
    """112 | Volga (Vol-of-Vol) Trading Strategy"""
    vol = ind["vol_20"]
    vol_of_vol = vol.rolling(20).std() / vol.rolling(20).mean().replace(0, np.nan)
    vov_z = _z(vol_of_vol, 120)
    # High vov = uncertain regime; low vov = stable regime
    raw = -vov_z * 35 + ind["mom_score"] * 30
    return _clip_signal(raw)


def s113_earnings_straddle(ind):
    """113 | DAX Straddle Around Earnings (vol spike proxy)"""
    vol = ind["vol_20"]
    vol_spike = (vol > vol.rolling(60).mean() + 2 * vol.rolling(60).std()).astype(float)
    fade_spike = vol_spike * -30
    raw = fade_spike + ind["trend_score"] * 35
    return _clip_signal(raw)


def s114_0dte_gamma(ind):
    """114 | Zero-Day-to-Expiry (0DTE) Gamma Scalping (intraday vol proxy)"""
    tr = ind["tr"]
    atr = ind["atr14"]
    gamma_proxy = tr / atr.replace(0, np.nan)
    gamma_z = _z(gamma_proxy, 20)
    raw = -gamma_z * 40 + ind["mom_5"] * 25
    return _clip_signal(raw)


def s115_overnight_vol(ind):
    """115 | Nikkei Overnight Volatility Strategy"""
    gap = (ind["open"] - ind["close"].shift(1)) / ind["close"].shift(1) * 100
    gap_z = _z(gap, 40)
    raw = -gap_z * 45 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s116_vol_targeting(ind):
    """116 | Exponentially Weighted Volatility Targeting"""
    vol = ind["vol_20"]
    target_vol = 0.15  # 15% annualized target
    leverage = (target_vol / (vol * np.sqrt(252)).replace(0, np.nan)).clip(0.2, 2.0)
    trend = ind["trend_score"]
    raw = trend * leverage * 40
    return _clip_signal(raw)


def s117_vol_skew_trading(ind):
    """117 | Volatility Surface Skew Trading (proxy)"""
    ret = ind["ret_1"]
    skew = ret.rolling(60, min_periods=20).apply(lambda x: pd.Series(x).skew(), raw=False)
    skew_z = _z(skew, 120)
    raw = -skew_z * 40 + ind["mom_20"] * 25
    return _clip_signal(raw)


def s118_intraday_smile(ind):
    """118 | Intraday Volatility Smile Dynamics (range proxy)"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    intraday_range = (high - low) / close * 100
    range_z = _z(intraday_range, 40)
    raw = -range_z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s119_cvar_risk_budget(ind):
    """119 | Conditional VaR (CVaR) Risk Budgeting"""
    ret = ind["ret_1"]
    var_5 = ret.rolling(60, min_periods=20).quantile(0.05)
    cvar = ret.rolling(60, min_periods=20).apply(
        lambda x: x[x <= np.quantile(x, 0.05)].mean() if len(x) > 5 else -0.03, raw=True
    )
    cvar_z = _z(cvar, 120)
    raw = cvar_z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s120_iv_rank_percentile(ind):
    """120 | Implied Volatility Rank and Percentile Strategy"""
    vol = ind["vol_20"]
    iv_rank = vol.rolling(252, min_periods=60).rank(pct=True)
    low_iv = (iv_rank < 0.15).astype(float)
    high_iv = (iv_rank > 0.85).astype(float)
    raw = low_iv * 45 - high_iv * 35 + ind["mom_score"] * 25
    return _clip_signal(raw)


def s121_ibov_smile_asymmetry(ind):
    """121 | IBOV Volatility Smile Asymmetry (skew proxy)"""
    ret = ind["ret_1"]
    neg_vol = ret.clip(upper=0).rolling(20).std()
    pos_vol = ret.clip(lower=0).rolling(20).std()
    asymmetry = (neg_vol - pos_vol) / (neg_vol + pos_vol).replace(0, np.nan)
    z = _z(asymmetry, 60)
    raw = -z * 45 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s122_hist_vol_percentile(ind):
    """122 | Historical Volatility Breakout Percentile"""
    vol = ind["vol_20"]
    pct = vol.rolling(252, min_periods=60).rank(pct=True)
    squeeze = (pct < 0.10).astype(float) * 50  # vol breakout setup
    elevated = (pct > 0.90).astype(float) * -30  # vol mean revert
    raw = squeeze + elevated + ind["trend_score"] * 25
    return _clip_signal(raw)


def s123_india_vix_mr(ind):
    """123 | Mumbai VIX (India VIX) Mean Reversion (vol proxy)"""
    vol_z = _z(ind["vol_20"], 60)
    raw = -vol_z * 50 + ind["mom_10"] * 20
    return _clip_signal(raw)


def s124_cross_ccy_vol(ind):
    """124 | Cross-Currency Volatility Smile Trade (vol ratio proxy)"""
    vol_ratio = ind["vol_20"] / ind["vol_60"].replace(0, np.nan)
    z = _z(vol_ratio, 120)
    raw = -z * 40 + ind["mom_score"] * 25
    return _clip_signal(raw)


def s125_rv_signature(ind):
    """125 | Realized Volatility Signature Plot Strategy"""
    ret = ind["ret_1"]
    rv_5 = ret.rolling(5).std() * np.sqrt(252)
    rv_20 = ind["vol_20"] * np.sqrt(252)
    rv_60 = ind["vol_60"] * np.sqrt(252)
    sig = (rv_5 - rv_20) / rv_60.replace(0, np.nan)
    z = _z(sig, 60)
    raw = -z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s126_corr_breakdown_hedge(ind):
    """126 | Correlation Breakdown Volatility Hedge (proxy)"""
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Correlation breakdown proxy: vol spike + momentum crash
    breakdown = ((vol > vol.rolling(60).mean() + 1.5 * vol.rolling(60).std()) & (mom < -0.05)).astype(float)
    raw = breakdown * (-60) + (1 - breakdown) * ind["trend_score"] * 40
    return _clip_signal(raw)


def s127_iv_rv_sector(ind):
    """127 | Implied-Realized Volatility Spread by Sector (proxy)"""
    vol_long = ind["vol_60"]
    vol_short = ind["vol_20"]
    spread = vol_long - vol_short
    z = _z(spread, 120)
    raw = z * 35 + ind["mom_score"] * 30
    return _clip_signal(raw)


def s128_heston_sv(ind):
    """128 | Heston Model Stochastic Volatility Trade"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Heston: vol mean reverts + correlated with returns
    vol_z = _z(vol, 60)
    # Negative vol-price correlation (leverage effect)
    corr = ret.rolling(60, min_periods=20).corr(vol)
    lev_effect = (corr < -0.3).astype(float)
    raw = -vol_z * 30 * (1 + lev_effect * 0.5) + ind["trend_score"] * 30
    return _clip_signal(raw)


def s129_gold_vol_premium(ind):
    """129 | Dubai Gold Volatility Premium Strategy"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    raw = -vol_z * 35 + mom * 50
    return _clip_signal(raw)


def s130_vol_autocorr(ind):
    """130 | OMX Volatility Autocorrelation"""
    vol = ind["vol_20"]
    ac = vol.rolling(60, min_periods=20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.5, raw=True
    )
    ac_z = _z(ac, 120)
    raw = ac_z * 35 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s131_rv_delta_hedge(ind):
    """131 | Realized Variance Swap via Delta Hedging (proxy)"""
    ret = ind["ret_1"]
    rv = ret.rolling(20).std() ** 2 * 252
    lrv = ret.rolling(252, min_periods=60).std() ** 2 * 252
    spread = rv - lrv
    z = _z(spread, 120)
    raw = -z * 40 + ind["mom_score"] * 25
    return _clip_signal(raw)


def s132_vol_cluster_duration(ind):
    """132 | Volatility Clustering Duration Model"""
    vol = ind["vol_20"]
    high_vol = (vol > vol.rolling(60).mean() + vol.rolling(60).std()).astype(float)
    # Duration of high vol cluster
    cluster_len = high_vol.rolling(20).sum()
    # Long clusters tend to mean revert
    z = _z(cluster_len, 60)
    raw = -z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s133_volume_vol_elasticity(ind):
    """133 | Intraday Volume-Volatility Elasticity"""
    vol = ind["vol_20"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    elasticity = vol / vol_ma
    z = _z(elasticity, 60)
    raw = -z * 35 + ind["mom_score"] * 30
    return _clip_signal(raw)


def s134_tase_vol_premium(ind):
    """134 | Tel Aviv TASE Volatility Premium"""
    vol_z = _z(ind["vol_20"], 120)
    above_200 = ind["above_200"].fillna(0.5)
    raw = -vol_z * 35 + (above_200 * 2 - 1) * 30
    return _clip_signal(raw)


def s135_sabr_surface(ind):
    """135 | SABR Model Volatility Surface Arbitrage (proxy)"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    skew = ret.rolling(60, min_periods=20).apply(lambda x: pd.Series(x).skew(), raw=False)
    kurt = ret.rolling(60, min_periods=20).apply(lambda x: pd.Series(x).kurtosis(), raw=False)
    z_vol = _z(vol, 120)
    z_skew = _z(skew, 120)
    raw = -z_vol * 25 - z_skew * 20 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s136_commodity_iv_signal(ind):
    """136 | Toronto Commodity Producer Implied Volatility Signal (proxy)"""
    vol = ind["vol_20"]
    mom = ind["mom_40"]
    vol_z = _z(vol, 120)
    raw = -vol_z * 30 + mom * 50
    return _clip_signal(raw)


def s137_vix_spx_decoupling(ind):
    """137 | Intraday VIX vs. SPX Decoupling Strategy (proxy)"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Normal: vol up when price down. Decoupling: both up or both down
    corr = ret.rolling(20).corr(vol.pct_change())
    decouple = (corr > 0).astype(float)  # abnormal positive correlation
    raw = decouple * (-30) + (1 - decouple) * ind["trend_score"] * 40
    return _clip_signal(raw)


def s138_exp_cone_vol(ind):
    """138 | Exponential Cone Volatility Model"""
    ret = ind["ret_1"]
    garch = _garch_vol(ret)
    yz = _yang_zhang_vol(ind["open"], ind["high"], ind["low"], ind["close"])
    # Blend: exponential weighting of estimators
    blend = 0.6 * garch + 0.4 * yz
    z = _z(blend, 60)
    raw = -z * 35 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s139_volvol_mr(ind):
    """139 | Volatility of Volatility (VolVol) Mean Reversion"""
    vol = ind["vol_20"]
    vov = vol.rolling(20).std() / vol.rolling(20).mean().replace(0, np.nan)
    z = _z(vov, 120)
    raw = -z * 50
    return _clip_signal(raw)


def s140_chaikin_vol(ind):
    """140 | Chaikin Volatility Expansion/Contraction"""
    high = ind["high"]
    low = ind["low"]
    hl_ema = (high - low).ewm(span=10, adjust=False).mean()
    chaikin = hl_ema.pct_change(10) * 100
    z = _z(chaikin, 60)
    # Vol expansion: trend follow; Vol contraction: mean revert setup
    expand = (z > 1.0).astype(float) * ind["trend_score"]
    compress = (z < -1.0).astype(float)
    raw = expand * 40 + compress * 30 + ind["mom_score"] * 20
    return _clip_signal(raw)


def s141_managed_futures_vol(ind):
    """141 | Synthetic Long Volatility via Managed Futures"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    mom = ind["mom_20"]
    # Long vol: reduce when vol high, add when vol low, always follow trend
    raw = ind["trend_score"] * (1 - vol_z.clip(0, 2) / 4) * 60
    return _clip_signal(raw)


def s142_rv_day_of_week(ind):
    """142 | Realized Volatility Premium by Day of Week"""
    close = ind["close"]
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # Weekly cycle proxy using 5-day pattern
    weekly_pos = np.arange(len(close)) % 5
    # Mon/Fri tend to have higher vol
    day_weight = pd.Series(
        np.where((weekly_pos == 0) | (weekly_pos == 4), 1.2, 0.9),
        index=close.index
    )
    raw = ind["trend_score"] * day_weight * 40
    return _clip_signal(raw)


def s143_quanto_vol_arb(ind):
    """143 | Quanto Volatility Arbitrage (proxy)"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 60)
    mom = ind["mom_20"]
    raw = -vol_z * 35 + mom * 40
    return _clip_signal(raw)


def s144_vol_skew_momentum(ind):
    """144 | Volatility Skew Momentum (Term Structure)"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    skew = vol_20 / vol_60.replace(0, np.nan)
    skew_mom = skew.pct_change(10)
    z = _z(skew_mom, 60)
    raw = z * 45 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s145_india_vix_contango(ind):
    """145 | Indian VIX Futures Contango Carry (vol carry proxy)"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    carry = (vol_60 - vol_20) / vol_60.replace(0, np.nan)
    z = _z(carry, 120)
    raw = z * 40 + ind["mom_score"] * 25
    return _clip_signal(raw)


def s146_tr_expansion(ind):
    """146 | Range-Based Volatility Breakout (True Range Expansion)"""
    tr = ind["tr"]
    atr = ind["atr14"]
    ratio = tr / atr.replace(0, np.nan)
    expansion = (ratio > 2.0).astype(float)
    direction = np.sign(ind["close"] - ind["open"])
    raw = expansion * direction * 55 + (1 - expansion) * ind["trend_score"] * 30
    return _clip_signal(raw)


def s147_vix_fomc_seasonality(ind):
    """147 | VIX Seasonality and FOMC Cycle (proxy)"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 60)
    # Monthly cycle proxy
    monthly_pos = np.arange(len(vol)) % 22
    mid_month = ((monthly_pos >= 8) & (monthly_pos <= 14)).astype(float)
    raw = -vol_z * 35 + ind["trend_score"] * mid_month * 40
    return _clip_signal(raw)


def s148_hawkes_jump(ind):
    """148 | Hawkes Process Jump Intensity Model"""
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    jump = (ret.abs() > 3 * vol).astype(float)
    # Hawkes: jumps cluster. Intensity = base + alpha * recent jumps
    intensity = jump.rolling(10).sum() / 10
    z = _z(intensity, 60)
    # High intensity: cautious; Low intensity: follow trend
    raw = -z * 30 + ind["trend_score"] * (1 - z.clip(0, 2) / 3) * 40
    return _clip_signal(raw)


def s149_covered_call(ind):
    """149 | ASX Covered Call Premium Optimization (vol-momentum proxy)"""
    vol_pct = ind["vol_pct"]
    above_200 = ind["above_200"].fillna(0.5)
    # Write calls when: above 200MA + high vol rank (premium capture)
    raw = above_200 * (1 + vol_pct * 0.5) * 55 - 20
    return _clip_signal(raw)


def s150_vol_surface_pca(ind):
    """150 | Volatility Surface PCA Decomposition (proxy)"""
    ret = ind["ret_1"]
    garch = _garch_vol(ret)
    pk = _parkinson_vol(ind["high"], ind["low"])
    yz = _yang_zhang_vol(ind["open"], ind["high"], ind["low"], ind["close"])
    # Blend as PCA proxy: first component is level
    level = (garch + pk + yz) / 3
    level_z = _z(level, 120)
    # Second component: skew (difference between estimators)
    slope = garch - pk
    slope_z = _z(slope, 60)
    raw = -level_z * 25 - slope_z * 20 + ind["trend_score"] * 25
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    101: ("GARCH Vol Breakout", s101_garch_vol_breakout),
    102: ("VIX Term Structure Roll", s102_vix_term_structure),
    103: ("RV-IV Convergence", s103_rv_iv_convergence),
    104: ("Garman-Klass Vol", s104_garman_klass_vol),
    105: ("Straddle Condor Harvest", s105_straddle_condor),
    106: ("Yang-Zhang Regime", s106_yang_zhang_regime),
    107: ("Butterfly Vol Smile", s107_butterfly_vol_smile),
    108: ("ATR Channel Breakout", s108_atr_channel_breakout),
    109: ("EWMA Vol Cone", s109_ewma_vol_cone),
    110: ("Variance Swap", s110_variance_swap),
    111: ("Parkinson Breakout", s111_parkinson_breakout),
    112: ("Volga Vol-of-Vol", s112_volga_vol_of_vol),
    113: ("Earnings Straddle", s113_earnings_straddle),
    114: ("0DTE Gamma Scalp", s114_0dte_gamma),
    115: ("Overnight Vol", s115_overnight_vol),
    116: ("Vol Targeting", s116_vol_targeting),
    117: ("Vol Skew Trading", s117_vol_skew_trading),
    118: ("Intraday Smile", s118_intraday_smile),
    119: ("CVaR Risk Budget", s119_cvar_risk_budget),
    120: ("IV Rank Percentile", s120_iv_rank_percentile),
    121: ("IBOV Smile Asymmetry", s121_ibov_smile_asymmetry),
    122: ("Hist Vol Percentile", s122_hist_vol_percentile),
    123: ("India VIX MR", s123_india_vix_mr),
    124: ("Cross-Ccy Vol", s124_cross_ccy_vol),
    125: ("RV Signature Plot", s125_rv_signature),
    126: ("Corr Breakdown Hedge", s126_corr_breakdown_hedge),
    127: ("IV-RV Sector Spread", s127_iv_rv_sector),
    128: ("Heston Stoch Vol", s128_heston_sv),
    129: ("Gold Vol Premium", s129_gold_vol_premium),
    130: ("Vol Autocorrelation", s130_vol_autocorr),
    131: ("RV Delta Hedge", s131_rv_delta_hedge),
    132: ("Vol Cluster Duration", s132_vol_cluster_duration),
    133: ("Volume-Vol Elasticity", s133_volume_vol_elasticity),
    134: ("TASE Vol Premium", s134_tase_vol_premium),
    135: ("SABR Surface Arb", s135_sabr_surface),
    136: ("Commodity IV Signal", s136_commodity_iv_signal),
    137: ("VIX-SPX Decoupling", s137_vix_spx_decoupling),
    138: ("Exp Cone Vol", s138_exp_cone_vol),
    139: ("VolVol Mean Reversion", s139_volvol_mr),
    140: ("Chaikin Vol", s140_chaikin_vol),
    141: ("Managed Futures Vol", s141_managed_futures_vol),
    142: ("RV Day-of-Week", s142_rv_day_of_week),
    143: ("Quanto Vol Arb", s143_quanto_vol_arb),
    144: ("Vol Skew Momentum", s144_vol_skew_momentum),
    145: ("India VIX Contango", s145_india_vix_contango),
    146: ("TR Expansion Breakout", s146_tr_expansion),
    147: ("VIX FOMC Seasonality", s147_vix_fomc_seasonality),
    148: ("Hawkes Jump Intensity", s148_hawkes_jump),
    149: ("Covered Call Premium", s149_covered_call),
    150: ("Vol Surface PCA", s150_vol_surface_pca),
}
