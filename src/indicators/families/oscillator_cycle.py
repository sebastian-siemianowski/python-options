"""
SECTION VI: OSCILLATOR & CYCLE STRATEGIES (251-300)
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

def s251_ehlers_cyber_cycle(ind):
    """251 | Ehlers Cyber Cycle Oscillator"""
    close = ind["close"]
    # Simplified Cyber Cycle using 2-pole high-pass filter
    alpha = 0.07
    vals = close.values.astype(float)
    n = len(vals)
    cycle = np.zeros(n)
    smooth = np.zeros(n)
    for i in range(6, n):
        smooth[i] = (vals[i] + 2*vals[i-1] + 2*vals[i-2] + vals[i-3]) / 6
    for i in range(6, n):
        cycle[i] = ((1 - 0.5*alpha)**2) * (smooth[i] - 2*smooth[i-1] + smooth[i-2]) + \
                   2*(1-alpha)*cycle[i-1] - (1-alpha)**2 * cycle[i-2]
    cc = pd.Series(cycle, index=close.index)
    z = _z(cc, 40)
    raw = z * 55
    return _clip_signal(raw)


def s252_hilbert_trendline(ind):
    """252 | Hilbert Transform Instantaneous Trendline"""
    close = ind["close"]
    # Hilbert proxy: dual EMA detrending
    ema_short = close.ewm(span=10, adjust=False).mean()
    ema_long = close.ewm(span=30, adjust=False).mean()
    it = 2 * ema_short - ema_long  # instantaneous trendline proxy
    trend = (close > it).astype(float) * 2 - 1
    dist = (close - it) / ind["atr14"].replace(0, np.nan)
    raw = trend * 30 + dist.clip(-2, 2) / 2 * 25
    return _clip_signal(raw)


def s253_schaff_trend_cycle(ind):
    """253 | Schaff Trend Cycle (STC)"""
    close = ind["close"]
    macd = ind["macd_line"] - ind["macd_signal"]
    # Double stochastic on MACD
    ll = macd.rolling(10).min()
    hh = macd.rolling(10).max()
    frac1 = (macd - ll) / (hh - ll).replace(0, np.nan) * 100
    pf = frac1.ewm(span=3, adjust=False).mean()
    ll2 = pf.rolling(10).min()
    hh2 = pf.rolling(10).max()
    frac2 = (pf - ll2) / (hh2 - ll2).replace(0, np.nan) * 100
    stc = frac2.ewm(span=3, adjust=False).mean()
    raw = (stc - 50) / 50 * 60
    return _clip_signal(raw)


def s254_coppock_curve(ind):
    """254 | Coppock Curve Long-Term Buy Signal"""
    close = ind["close"]
    roc14 = close.pct_change(14 * 22) * 100  # 14-month proxy (308 days)
    roc11 = close.pct_change(11 * 22) * 100  # 11-month proxy (242 days)
    roc_sum = roc14.fillna(0) + roc11.fillna(0)
    coppock = roc_sum.ewm(span=10, adjust=False).mean()
    z = _z(coppock, 252)
    raw = z * 50
    return _clip_signal(raw)


def s255_ultimate_oscillator(ind):
    """255 | Ultimate Oscillator (Larry Williams)"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    raw = (uo - 50) / 50 * 60
    return _clip_signal(raw)


def s256_dpo_cycle(ind):
    """256 | Detrended Price Oscillator (DPO) Cycle"""
    close = ind["close"]
    n = 20
    sma = close.rolling(n).mean()
    dpo = close.shift(n // 2 + 1) - sma
    z = _z(dpo, 60)
    raw = z * 50
    return _clip_signal(raw)


def s257_cmo_adaptive(ind):
    """257 | Chande Momentum Oscillator (CMO) Adaptive"""
    close = ind["close"]
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(14).sum()
    loss = (-diff.clip(upper=0)).rolling(14).sum()
    cmo = (gain - loss) / (gain + loss).replace(0, np.nan) * 100
    # Adaptive: scale by volatility
    vol_adj = ind["vol_dampener"]
    raw = cmo * vol_adj * 0.6
    return _clip_signal(raw)


def s258_willr_thrust(ind):
    """258 | Williams %R Multi-Period Thrust"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    willr_5 = (close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min()).replace(0, np.nan) * -100 + 100
    willr_14 = ind["willr"] + 100  # convert from [-100,0] to [0,100]
    willr_21 = (close - low.rolling(21).min()) / (high.rolling(21).max() - low.rolling(21).min()).replace(0, np.nan) * -100 + 100
    # Thrust: all periods overbought/oversold simultaneously
    all_os = ((willr_5 < 20) & (willr_14 < 20) & (willr_21 < 20)).astype(float)
    all_ob = ((willr_5 > 80) & (willr_14 > 80) & (willr_21 > 80)).astype(float)
    raw = all_os * 60 - all_ob * 60
    return _clip_signal(raw)


def s259_cci_trend(ind):
    """259 | Commodity Channel Index (CCI) Trend-Following"""
    cci = ind["cci"]
    # CCI > 100: strong uptrend; CCI < -100: strong downtrend
    strong_up = (cci > 100).astype(float) * (cci - 100) / 100
    strong_dn = (cci < -100).astype(float) * (-100 - cci) / 100
    trend = ind["trend_score"]
    raw = strong_up * trend * 40 - strong_dn * trend * 40 + cci.clip(-200, 200) / 200 * 30
    return _clip_signal(raw)


def s260_kst(ind):
    """260 | Know Sure Thing (KST) Oscillator"""
    close = ind["close"]
    roc1 = close.pct_change(10).rolling(10).mean() * 100
    roc2 = close.pct_change(15).rolling(10).mean() * 100
    roc3 = close.pct_change(20).rolling(10).mean() * 100
    roc4 = close.pct_change(30).rolling(15).mean() * 100
    kst = roc1 + 2 * roc2 + 3 * roc3 + 4 * roc4
    kst_sig = kst.rolling(9).mean()
    z = _z(kst - kst_sig, 40)
    raw = z * 55
    return _clip_signal(raw)


def s261_rvi_divergence(ind):
    """261 | Relative Vigor Index (RVI) Divergence"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    num = (close - open_p + 2*(close.shift(1) - open_p.shift(1)) +
           2*(close.shift(2) - open_p.shift(2)) + (close.shift(3) - open_p.shift(3))) / 6
    den = (high - low + 2*(high.shift(1) - low.shift(1)) +
           2*(high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))) / 6
    rvi = num / den.replace(0, np.nan)
    rvi_sig = (rvi + 2*rvi.shift(1) + 2*rvi.shift(2) + rvi.shift(3)) / 6
    z = _z(rvi - rvi_sig, 40)
    raw = z * 55
    return _clip_signal(raw)


def s262_awesome_osc(ind):
    """262 | Awesome Oscillator (Bill Williams) Saucer"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    mid = (high + low) / 2
    ao = mid.rolling(5).mean() - mid.rolling(34).mean()
    # Saucer: AO > 0, dips then rises
    saucer_bull = ((ao > 0) & (ao > ao.shift(1)) & (ao.shift(1) < ao.shift(2))).astype(float)
    saucer_bear = ((ao < 0) & (ao < ao.shift(1)) & (ao.shift(1) > ao.shift(2))).astype(float)
    z = _z(ao, 40)
    raw = saucer_bull * 40 - saucer_bear * 40 + z * 25
    return _clip_signal(raw)


def s263_stoch_rsi(ind):
    """263 | Stochastic RSI (StochRSI) Extreme Reversal"""
    rsi = ind["rsi"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    k = stoch_rsi.rolling(3).mean()
    d = k.rolling(3).mean()
    oversold = ((k < 20) & (k > d)).astype(float) * (20 - k) / 20
    overbought = ((k > 80) & (k < d)).astype(float) * (k - 80) / 20
    raw = oversold * 55 - overbought * 55
    return _clip_signal(raw)


def s264_ppo_divergence(ind):
    """264 | Percentage Price Oscillator (PPO) Divergence"""
    close = ind["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ppo = (ema12 - ema26) / ema26.replace(0, np.nan) * 100
    ppo_sig = ppo.ewm(span=9, adjust=False).mean()
    hist = ppo - ppo_sig
    # Divergence: price makes new high but PPO doesn't
    z = _z(hist, 40)
    raw = z * 55
    return _clip_signal(raw)


def s265_fisher_transform(ind):
    """265 | Fisher Transform Oscillator"""
    close = ind["close"]
    ll = close.rolling(10).min()
    hh = close.rolling(10).max()
    value = ((close - ll) / (hh - ll).replace(0, np.nan) - 0.5) * 2
    value = value.clip(-0.999, 0.999)
    fisher = (0.5 * np.log((1 + value) / (1 - value))).ewm(span=3, adjust=False).mean()
    z = _z(fisher, 40)
    raw = z * 55
    return _clip_signal(raw)


def s266_sine_wave(ind):
    """266 | Sine Wave Indicator (Ehlers)"""
    close = ind["close"]
    # Sine wave proxy: detrended cycle detection
    n = 20
    detrended = close - close.rolling(n).mean()
    phase = np.arctan2(detrended, detrended.shift(n // 4)) * 180 / np.pi
    sine = np.sin(phase * np.pi / 180)
    lead_sine = np.sin((phase + 45) * np.pi / 180)
    cross = (sine > lead_sine).astype(float) - (sine < lead_sine).astype(float)
    raw = cross * 40 + _z(detrended, 40) * 20
    return _clip_signal(raw)


def s267_klinger_accum(ind):
    """267 | Klinger Volume Accumulation Oscillator"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    hlc = high + low + close
    trend = np.sign(hlc.diff())
    dm = high - low
    cm = dm.copy().astype(float)
    vals = dm.values.astype(float)
    t = trend.values.astype(float)
    c = np.zeros(len(vals))
    c[0] = vals[0]
    for i in range(1, len(vals)):
        if t[i] == t[i-1]:
            c[i] = c[i-1] + vals[i]
        else:
            c[i] = vals[i]
    cm = pd.Series(c, index=close.index)
    vf = volume * (2 * dm / cm.replace(0, np.nan) - 1) * trend * 100
    kvo = vf.ewm(span=34, adjust=False).mean() - vf.ewm(span=55, adjust=False).mean()
    sig = kvo.ewm(span=13, adjust=False).mean()
    raw = _z(kvo - sig, 40) * 50
    return _clip_signal(raw)


def s268_ergodic_osc(ind):
    """268 | Ergodic Oscillator (William Blau)"""
    close = ind["close"]
    mom = close.diff()
    ema1 = mom.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=5, adjust=False).mean()
    abs_mom = mom.abs()
    abs_ema1 = abs_mom.ewm(span=20, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=5, adjust=False).mean()
    ergo = ema2 / abs_ema2.replace(0, np.nan) * 100
    sig = ergo.ewm(span=5, adjust=False).mean()
    raw = _z(ergo - sig, 40) * 50 + ergo.clip(-100, 100) * 0.2
    return _clip_signal(raw)


def s269_mcginley_dynamic(ind):
    """269 | McGinley Dynamic Oscillator"""
    close = ind["close"]
    n = 14
    vals = close.values.astype(float)
    md = np.zeros(len(vals))
    md[0] = vals[0]
    for i in range(1, len(vals)):
        ratio = vals[i] / md[i-1] if md[i-1] != 0 else 1
        md[i] = md[i-1] + (vals[i] - md[i-1]) / (n * ratio**4)
    mg = pd.Series(md, index=close.index)
    dist = (close - mg) / ind["atr14"].replace(0, np.nan)
    raw = dist.clip(-3, 3) / 3 * 55
    return _clip_signal(raw)


def s270_ha_oscillator(ind):
    """270 | Heikin-Ashi Oscillator"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    ha_close = (open_p + high + low + close) / 4
    vals = ha_close.values.astype(float)
    o = open_p.values.astype(float)
    ho = np.zeros(len(vals))
    ho[0] = o[0]
    for i in range(1, len(vals)):
        ho[i] = (ho[i-1] + vals[i-1]) / 2
    ha_open = pd.Series(ho, index=close.index)
    ha_body = ha_close - ha_open
    ha_body_z = _z(ha_body, 40)
    raw = ha_body_z * 55
    return _clip_signal(raw)


def s271_gann_hilo_adx(ind):
    """271 | Gann Hi-Lo Activator with ADX Filter"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    sma_high = high.rolling(13).mean()
    sma_low = low.rolling(13).mean()
    # Gann Hi-Lo: above sma_high = up, below sma_low = down
    up = (close > sma_high).astype(float)
    dn = (close < sma_low).astype(float)
    adx = ind["adx"]
    strong_trend = (adx > 25).astype(float)
    raw = (up - dn) * (0.4 + 0.6 * strong_trend) * 55
    return _clip_signal(raw)


def s272_trix(ind):
    """272 | TRIX Triple Smoothed Momentum"""
    close = ind["close"]
    e1 = close.ewm(span=15, adjust=False).mean()
    e2 = e1.ewm(span=15, adjust=False).mean()
    e3 = e2.ewm(span=15, adjust=False).mean()
    trix = e3.pct_change() * 10000
    trix_sig = trix.rolling(9).mean()
    z = _z(trix - trix_sig, 40)
    raw = z * 55
    return _clip_signal(raw)


def s273_chaikin_vol_osc(ind):
    """273 | Chaikin Volatility Oscillator Expansion"""
    high = ind["high"]
    low = ind["low"]
    hl_ema = (high - low).ewm(span=10, adjust=False).mean()
    cvo = hl_ema.pct_change(10) * 100
    z = _z(cvo, 60)
    trend = ind["trend_score"]
    expand = (z > 1.0).astype(float) * trend
    compress = (z < -1.0).astype(float)
    raw = expand * 40 + compress * 30 + trend * 20
    return _clip_signal(raw)


def s274_mama(ind):
    """274 | MESA Adaptive Moving Average (MAMA)"""
    close = ind["close"]
    # Simplified MAMA: adaptive EMA
    fast_limit = 0.5
    slow_limit = 0.05
    hurst = ind["hurst"].fillna(0.5)
    alpha = slow_limit + (fast_limit - slow_limit) * (1 - hurst)
    vals = close.values.astype(float)
    a = alpha.values.astype(float)
    mama = np.zeros(len(vals))
    fama = np.zeros(len(vals))
    mama[0] = vals[0]
    fama[0] = vals[0]
    for i in range(1, len(vals)):
        mama[i] = mama[i-1] + a[i] * (vals[i] - mama[i-1])
        fama[i] = fama[i-1] + 0.5 * a[i] * (mama[i] - fama[i-1])
    mama_s = pd.Series(mama, index=close.index)
    fama_s = pd.Series(fama, index=close.index)
    raw = _z(mama_s - fama_s, 40) * 55
    return _clip_signal(raw)


def s275_fib_time_zone(ind):
    """275 | Fibonacci Time Zone Cyclical Trading"""
    close = ind["close"]
    # Fib time zones proxy: cyclical detection at fib intervals
    n_bar = np.arange(len(close))
    fib_periods = [8, 13, 21, 34, 55]
    cycle_strength = sum([(np.sin(2 * np.pi * n_bar / p)) for p in fib_periods])
    cycle = pd.Series(cycle_strength / len(fib_periods), index=close.index)
    raw = cycle * ind["trend_score"] * 50
    return _clip_signal(raw)


def s276_aroon_osc(ind):
    """276 | Aroon Oscillator Trend Strength"""
    high = ind["high"]
    low = ind["low"]
    n = 25
    aroon_up = high.rolling(n).apply(lambda x: x.argmax() / (n-1) * 100, raw=True)
    aroon_dn = low.rolling(n).apply(lambda x: x.argmin() / (n-1) * 100, raw=True)
    aroon_osc = aroon_up - aroon_dn
    z = _z(aroon_osc, 40)
    raw = z * 50
    return _clip_signal(raw)


def s277_bop(ind):
    """277 | Balance of Power (BOP) Indicator"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    bop = (close - open_p) / (high - low).replace(0, np.nan)
    bop_sma = bop.rolling(14).mean()
    z = _z(bop_sma, 40)
    raw = z * 55
    return _clip_signal(raw)


def s278_csi_strategy(ind):
    """278 | Commodity Selection Index (CSI) Strategy"""
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    atr = ind["atr_pct"]
    # CSI: trend strength * volatility (tradability score)
    csi = mom.abs() * atr * 100
    z = _z(csi, 60)
    direction = np.sign(mom)
    raw = z * direction * 45
    return _clip_signal(raw)


def s279_ado(ind):
    """279 | Accumulation/Distribution Oscillator (ADO)"""
    ad = ind["ad_score"]
    fast = ad.ewm(span=3, adjust=False).mean()
    slow = ad.ewm(span=10, adjust=False).mean()
    ado = fast - slow
    z = _z(ado, 40)
    raw = z * 55
    return _clip_signal(raw)


def s280_connors_rsi(ind):
    """280 | Connors RSI (CRSI) Mean Reversion"""
    close = ind["close"]
    rsi = ind["rsi"]
    # Streak RSI: RSI of up/down streak length
    ret = close.diff()
    streak = ret.copy() * 0
    s_vals = np.zeros(len(ret))
    r = ret.fillna(0).values
    for i in range(1, len(r)):
        if r[i] > 0:
            s_vals[i] = max(s_vals[i-1], 0) + 1
        elif r[i] < 0:
            s_vals[i] = min(s_vals[i-1], 0) - 1
    streak = pd.Series(s_vals, index=close.index)
    streak_rsi = _norm(streak, 60) * 50 + 50
    # Percentile rank of returns
    pct_rank = ret.rolling(60, min_periods=20).rank(pct=True) * 100
    crsi = (rsi + streak_rsi + pct_rank) / 3
    raw = (50 - crsi) / 50 * -60
    return _clip_signal(raw)


def s281_demand_index(ind):
    """281 | Demand Index (DI) Oscillator"""
    close = ind["close"]
    volume = ind["volume"]
    high = ind["high"]
    low = ind["low"]
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    sp = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - close
    demand = bp * volume
    supply = sp * volume
    di = (demand - supply) / (demand + supply).replace(0, np.nan)
    z = _z(di.rolling(10).mean(), 40)
    raw = z * 55
    return _clip_signal(raw)


def s282_elder_ray(ind):
    """282 | Elder Ray Bull/Bear Power"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    ema13 = close.ewm(span=13, adjust=False).mean()
    bull = high - ema13
    bear = low - ema13
    bull_z = _z(bull, 40)
    bear_z = _z(bear, 40)
    raw = (bull_z + bear_z) / 2 * 55
    return _clip_signal(raw)


def s283_smi(ind):
    """283 | Stochastic Momentum Index (SMI)"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    hh = high.rolling(13).max()
    ll = low.rolling(13).min()
    mid = (hh + ll) / 2
    d = close - mid
    d_smooth = d.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    r = (hh - ll).ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    smi = d_smooth / (r / 2).replace(0, np.nan) * 100
    sig = smi.ewm(span=3, adjust=False).mean()
    raw = _z(smi - sig, 40) * 50 + smi.clip(-100, 100) * 0.15
    return _clip_signal(raw)


def s284_pfe(ind):
    """284 | Polarized Fractal Efficiency (PFE)"""
    close = ind["close"]
    n = 10
    price_change = (close - close.shift(n)).abs()
    path_length = close.diff().abs().rolling(n).sum()
    pfe = price_change / path_length.replace(0, np.nan) * 100 * np.sign(close.diff(n))
    pfe_smooth = pfe.ewm(span=5, adjust=False).mean()
    z = _z(pfe_smooth, 40)
    raw = z * 55
    return _clip_signal(raw)


def s285_premier_stoch(ind):
    """285 | Premier Stochastic Oscillator"""
    stoch = ind["stoch_k"]
    # Normalize and smooth
    norm = (stoch - 50) / 50
    smooth1 = norm.ewm(span=8, adjust=False).mean()
    smooth2 = smooth1.ewm(span=5, adjust=False).mean()
    premier = (np.exp(smooth2) - np.exp(-smooth2)) / (np.exp(smooth2) + np.exp(-smooth2))
    raw = premier * 60
    return _clip_signal(raw)


def s286_qstick(ind):
    """286 | Qstick Oscillator (Chan)"""
    close = ind["close"]
    open_p = ind["open"]
    body = close - open_p
    qstick = body.rolling(14).mean()
    z = _z(qstick, 40)
    raw = z * 55
    return _clip_signal(raw)


def s287_vortex(ind):
    """287 | Vortex Indicator (VI) Trend Confirmation"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr = ind["tr"]
    n = 14
    vi_plus = vm_plus.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    vi_minus = vm_minus.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    raw = _z(vi_plus - vi_minus, 40) * 55
    return _clip_signal(raw)


def s288_rmi(ind):
    """288 | Relative Momentum Index (RMI)"""
    close = ind["close"]
    n = 5  # momentum period
    diff = close.diff(n)
    gain = diff.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rmi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    raw = (rmi - 50) / 50 * 60
    return _clip_signal(raw)


def s289_murrey_math(ind):
    """289 | Murrey Math Lines (MML) Grid"""
    close = ind["close"]
    hh = ind["hh"]
    ll = ind["ll"]
    rng = hh - ll
    # MML levels: 0/8 to 8/8
    level = (close - ll) / rng.replace(0, np.nan)
    # Extreme zones: below 1/8 or above 7/8
    oversold = (level < 0.125).astype(float) * (0.125 - level) / 0.125
    overbought = (level > 0.875).astype(float) * (level - 0.875) / 0.125
    # Pivot zones: 3/8 to 5/8
    pivot = ((level > 0.375) & (level < 0.625)).astype(float)
    raw = oversold * 55 - overbought * 55 + pivot * ind["trend_score"] * 20
    return _clip_signal(raw)


def s290_choppiness(ind):
    """290 | Choppiness Index (CI) Regime Filter"""
    atr = ind["atr14"]
    high = ind["high"]
    low = ind["low"]
    n = 14
    atr_sum = atr.rolling(n).sum()
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    rng = (hh - ll).replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / rng) / np.log10(n)
    # CI > 61.8: choppy (range); CI < 38.2: trending
    trending = (ci < 38.2).astype(float)
    choppy = (ci > 61.8).astype(float)
    raw = trending * ind["trend_score"] * 55 - choppy * _z(ind["close"], 40) * 30
    return _clip_signal(raw)


def s291_macd_hist_reversal(ind):
    """291 | MACD Histogram Reversal Pattern"""
    hist = ind["macd_hist"]
    # Histogram reversal: histogram changes direction
    reversal_up = ((hist > hist.shift(1)) & (hist.shift(1) < hist.shift(2))).astype(float)
    reversal_dn = ((hist < hist.shift(1)) & (hist.shift(1) > hist.shift(2))).astype(float)
    # Strength: how far from zero
    strength = hist.abs() / hist.rolling(60).std().replace(0, np.nan)
    raw = reversal_up * strength * 30 - reversal_dn * strength * 30 + _z(hist, 40) * 20
    return _clip_signal(raw)


def s292_chandelier_exit(ind):
    """292 | Chandelier Exit System"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    atr = ind["atr14"]
    hh22 = high.rolling(22).max()
    ll22 = low.rolling(22).min()
    long_stop = hh22 - 3 * atr
    short_stop = ll22 + 3 * atr
    long_sig = (close > long_stop).astype(float)
    short_sig = (close < short_stop).astype(float)
    raw = (long_sig - short_sig) * 55
    return _clip_signal(raw)


def s293_squeeze_momentum(ind):
    """293 | Squeeze Momentum Indicator (LazyBear)"""
    close = ind["close"]
    bb_squeeze = ind["bb_squeeze"]
    # Momentum during squeeze
    mom = close - close.rolling(20).mean()
    mom_z = _z(mom, 40)
    squeeze_on = (bb_squeeze > 0.5).astype(float)
    raw = squeeze_on * mom_z * 40 + (1 - squeeze_on) * mom_z * 55
    return _clip_signal(raw)


def s294_klinger_macd_dual(ind):
    """294 | Klinger + MACD Dual Confirmation"""
    macd_z = _z(ind["macd_hist"], 40)
    # Klinger proxy
    close = ind["close"]
    volume = ind["volume"]
    trend = np.sign(close.diff())
    vf = volume * trend * (ind["high"] - ind["low"])
    kvo = vf.ewm(span=34, adjust=False).mean() - vf.ewm(span=55, adjust=False).mean()
    kvo_z = _z(kvo, 40)
    # Both agree = strong signal
    agreement = np.sign(macd_z) == np.sign(kvo_z)
    raw = ((macd_z + kvo_z) / 2 * agreement.astype(float) * 50 +
           (macd_z + kvo_z) / 2 * (1 - agreement.astype(float)) * 20)
    return _clip_signal(raw)


def s295_waddah_attar(ind):
    """295 | Waddah Attar Explosion Indicator"""
    close = ind["close"]
    macd_diff = ind["macd_line"] - ind["macd_signal"]
    bb_upper = ind["bb_upper"]
    bb_lower = ind["bb_lower"]
    explosion = bb_upper - bb_lower
    trend_up = (macd_diff > 0).astype(float) * macd_diff
    trend_dn = (macd_diff < 0).astype(float) * macd_diff.abs()
    dead_zone = explosion.rolling(60).mean() * 0.5
    active = (explosion > dead_zone).astype(float)
    raw = (trend_up - trend_dn) / explosion.replace(0, np.nan) * active * 70
    return _clip_signal(raw)


def s296_cycle_adaptive_rsi(ind):
    """296 | Cycle-Adaptive RSI (Ehlers)"""
    close = ind["close"]
    hurst = ind["hurst"].fillna(0.5)
    # Adaptive RSI period based on Hurst/cycle
    period = (10 + hurst * 20).astype(int).clip(5, 30)
    # Simplified: use fixed RSI but scale by cycle
    rsi = ind["rsi"]
    cycle_adj = (1 - hurst) * 2  # trending: less aggressive; mean-reverting: more
    raw = (50 - rsi) / 50 * cycle_adj * -55
    return _clip_signal(raw)


def s297_hurst_band(ind):
    """297 | Hurst Band Cyclical Envelope"""
    close = ind["close"]
    hurst = ind["hurst"].fillna(0.5)
    sma = ind["sma_50"]
    vol = ind["vol_20"] * close
    upper = sma + 2 * vol * hurst
    lower = sma - 2 * vol * hurst
    dist = (close - sma) / (upper - lower).replace(0, np.nan) * 2
    raw = dist.clip(-1, 1) * 55
    return _clip_signal(raw)


def s298_zweig_thrust(ind):
    """298 | Zweig Breadth Thrust"""
    # Breadth proxy: momentum of advancing vs declining signals
    ret = ind["ret_1"]
    advancing = (ret > 0).astype(float)
    thrust = advancing.rolling(10).mean()
    # Zweig thrust: breadth moves from <40% to >61.5% in 10 days
    low_breadth = (thrust.rolling(10).min() < 0.4).astype(float)
    high_breadth = (thrust > 0.615).astype(float)
    signal = low_breadth * high_breadth
    raw = signal * 60 + (thrust - 0.5) * 60
    return _clip_signal(raw)


def s299_imi(ind):
    """299 | Intraday Momentum Index (IMI)"""
    close = ind["close"]
    open_p = ind["open"]
    up_days = (close > open_p).astype(float) * (close - open_p)
    dn_days = (close < open_p).astype(float) * (open_p - close)
    imi = up_days.rolling(14).sum() / (up_days.rolling(14).sum() + dn_days.rolling(14).sum()).replace(0, np.nan) * 100
    raw = (imi - 50) / 50 * 60
    return _clip_signal(raw)


def s300_mtf_osc_consensus(ind):
    """300 | Multi-Timeframe Oscillator Consensus"""
    rsi = ind["rsi"]
    stoch = ind["stoch_k"]
    cci = ind["cci"]
    willr = ind["willr"]
    mfi = ind["mfi"]
    # Normalize all to [-1, 1]
    rsi_n = (rsi - 50) / 50
    stoch_n = (stoch - 50) / 50
    cci_n = (cci / 200).clip(-1, 1)
    willr_n = (willr + 50) / 50
    mfi_n = (mfi - 50) / 50
    consensus = (rsi_n + stoch_n + cci_n + willr_n + mfi_n) / 5
    vol_adj = ind["vol_dampener"]
    raw = consensus * vol_adj * 70
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    251: ("Ehlers Cyber Cycle", s251_ehlers_cyber_cycle),
    252: ("Hilbert Trendline", s252_hilbert_trendline),
    253: ("Schaff Trend Cycle", s253_schaff_trend_cycle),
    254: ("Coppock Curve", s254_coppock_curve),
    255: ("Ultimate Oscillator", s255_ultimate_oscillator),
    256: ("DPO Cycle", s256_dpo_cycle),
    257: ("CMO Adaptive", s257_cmo_adaptive),
    258: ("Williams %R Thrust", s258_willr_thrust),
    259: ("CCI Trend", s259_cci_trend),
    260: ("KST Oscillator", s260_kst),
    261: ("RVI Divergence", s261_rvi_divergence),
    262: ("Awesome Osc Saucer", s262_awesome_osc),
    263: ("StochRSI Extreme", s263_stoch_rsi),
    264: ("PPO Divergence", s264_ppo_divergence),
    265: ("Fisher Transform", s265_fisher_transform),
    266: ("Sine Wave Ehlers", s266_sine_wave),
    267: ("Klinger Accumulation", s267_klinger_accum),
    268: ("Ergodic Oscillator", s268_ergodic_osc),
    269: ("McGinley Dynamic", s269_mcginley_dynamic),
    270: ("Heikin-Ashi Osc", s270_ha_oscillator),
    271: ("Gann HiLo + ADX", s271_gann_hilo_adx),
    272: ("TRIX Momentum", s272_trix),
    273: ("Chaikin Vol Osc", s273_chaikin_vol_osc),
    274: ("MESA MAMA", s274_mama),
    275: ("Fib Time Zone", s275_fib_time_zone),
    276: ("Aroon Oscillator", s276_aroon_osc),
    277: ("Balance of Power", s277_bop),
    278: ("Commodity Selection", s278_csi_strategy),
    279: ("AD Oscillator", s279_ado),
    280: ("Connors RSI", s280_connors_rsi),
    281: ("Demand Index", s281_demand_index),
    282: ("Elder Ray", s282_elder_ray),
    283: ("Stochastic Momentum", s283_smi),
    284: ("Polarized Fractal Eff", s284_pfe),
    285: ("Premier Stochastic", s285_premier_stoch),
    286: ("Qstick Oscillator", s286_qstick),
    287: ("Vortex Indicator", s287_vortex),
    288: ("Relative Momentum", s288_rmi),
    289: ("Murrey Math Lines", s289_murrey_math),
    290: ("Choppiness Index", s290_choppiness),
    291: ("MACD Hist Reversal", s291_macd_hist_reversal),
    292: ("Chandelier Exit", s292_chandelier_exit),
    293: ("Squeeze Momentum", s293_squeeze_momentum),
    294: ("Klinger + MACD Dual", s294_klinger_macd_dual),
    295: ("Waddah Attar", s295_waddah_attar),
    296: ("Cycle-Adaptive RSI", s296_cycle_adaptive_rsi),
    297: ("Hurst Band Envelope", s297_hurst_band),
    298: ("Zweig Breadth Thrust", s298_zweig_thrust),
    299: ("Intraday Momentum", s299_imi),
    300: ("MTF Osc Consensus", s300_mtf_osc_consensus),
}
