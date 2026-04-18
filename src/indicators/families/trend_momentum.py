"""
SECTION I: TREND AND MOMENTUM STRATEGIES (001-050)
Each function takes ind dict -> pd.Series[-100, +100].
"""

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────
def _safe(s, fill=0.0):
    return s.fillna(fill) if isinstance(s, pd.Series) else pd.Series(s).fillna(fill)


def _norm(s, span=60):
    """Normalise to [-1,1] via rolling min/max."""
    lo = s.rolling(span, min_periods=10).min()
    hi = s.rolling(span, min_periods=10).max()
    rng = (hi - lo).replace(0, np.nan)
    return ((s - lo) / rng * 2 - 1).clip(-1, 1).fillna(0)


def _z(s, span=60):
    """Rolling z-score."""
    m = s.rolling(span, min_periods=10).mean()
    sd = s.rolling(span, min_periods=10).std().replace(0, np.nan)
    return ((s - m) / sd).clip(-4, 4).fillna(0)


def _wma(s, n):
    """Weighted moving average."""
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hma(s, n):
    """Hull Moving Average."""
    half_n = max(int(n / 2), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    return _wma(2 * _wma(s, half_n) - _wma(s, n), sqrt_n)


def _alma(s, n=9, offset=0.85, sigma=6.0):
    """Arnaud Legoux Moving Average."""
    m = int(offset * (n - 1))
    ss = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * ss * ss))
    w = w / w.sum()
    return s.rolling(n).apply(lambda x: np.dot(x, w), raw=True)


def _crossover(a, b):
    """Series: 1 where a crosses above b."""
    return ((a > b) & (a.shift(1) <= b.shift(1))).astype(float)


def _crossunder(a, b):
    """Series: 1 where a crosses below b."""
    return ((a < b) & (a.shift(1) >= b.shift(1))).astype(float)


def _clip_signal(raw, smooth=3):
    """Clip to [-100,100] and smooth."""
    return raw.clip(-100, 100).ewm(span=smooth, adjust=False).mean()


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGIES 001-050
# ═════════════════════════════════════════════════════════════════════════════

def s001_ehlers_trendline(ind):
    """001 | Ehlers Instantaneous Trendline Filter"""
    close = ind["close"]
    n = 20
    a = 2.0 / (n + 1)
    it = close.copy().astype(float)
    vals = close.values.astype(float)
    out = np.full(len(vals), np.nan)
    out[0] = vals[0]
    out[1] = vals[1] if len(vals) > 1 else vals[0]
    for i in range(2, len(vals)):
        out[i] = (a - a*a/4)*vals[i] + (a*a/2)*vals[i-1] - (a - 3*a*a/4)*vals[i-2] + 2*(1-a)*out[i-1] - (1-a)**2*out[i-2]
    it = pd.Series(out, index=close.index)
    slope = (it - it.shift(3)) / it.shift(3) * 100
    above = (close > it).astype(float) * 2 - 1
    raw = above * slope.abs().clip(0, 5) / 5 * 80
    return _clip_signal(raw)


def s002_kama_momentum(ind):
    """002 | Kaufman Adaptive Moving Average Momentum"""
    close = ind["close"]
    n = 10
    fast_sc = 2.0 / 3.0
    slow_sc = 2.0 / 31.0
    er = ind["efficiency"]
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = close.copy().astype(float)
    vals = close.values.astype(float)
    sc_vals = sc.fillna(slow_sc**2).values
    out = np.full(len(vals), np.nan)
    out[n] = vals[n] if len(vals) > n else vals[0]
    for i in range(n + 1, len(vals)):
        if np.isnan(out[i-1]):
            out[i] = vals[i]
        else:
            out[i] = out[i-1] + sc_vals[i] * (vals[i] - out[i-1])
    kama = pd.Series(out, index=close.index)
    slope = kama.pct_change(3)
    dist = (close - kama) / ind["atr14"].replace(0, np.nan)
    trend = np.sign(slope) * dist.clip(-3, 3) / 3
    er_filter = (er > 0.3).astype(float) * 0.7 + 0.3
    raw = trend * er_filter * 80
    return _clip_signal(raw)


def s003_triple_hma_regime(ind):
    """003 | Triple Hull Moving Average Regime System"""
    close = ind["close"]
    hma_f = _hma(close, 16)
    hma_m = _hma(close, 36)
    hma_s = _hma(close, 64)
    regime = np.sign(hma_f - hma_m) + np.sign(hma_m - hma_s)
    raw = regime / 2.0 * 80
    return _clip_signal(pd.Series(raw, index=close.index))


def s004_ichimoku_extended(ind):
    """004 | Ichimoku Kinko Hyo Extended Cloud System"""
    close = ind["close"]
    above_cloud = ind["ichi_above_cloud"]
    tk = ind["ichi_tk"]
    mom = ind["mom_10"]
    raw = (0.4 * (above_cloud * 2 - 1) + 0.35 * tk / 5.0 + 0.25 * mom) * 80
    return _clip_signal(raw)


def s005_dax_momentum_oscillator(ind):
    """005 | DAX Ordnungssystem Momentum Oscillator"""
    close = ind["close"]
    n = 20
    roc = close.pct_change(n)
    sigma = n / 3.0
    w = np.array([np.exp(-i**2 / (2*sigma**2)) for i in range(2*n+1)])
    w = w / w.sum()
    g_roc = roc.rolling(2*n+1, min_periods=n).apply(lambda x: np.dot(x[-len(w):], w[-len(x):]) / w[-len(x):].sum() if len(x) >= n else np.nan, raw=True)
    daxmo = _z(g_roc, 60)
    raw = daxmo.clip(-3, 3) / 3 * 80
    return _clip_signal(raw)


def s006_carry_momentum_filter(ind):
    """006 | Swiss Precision Carry-Momentum Filter (OHLCV proxy: yield proxy from vol)"""
    close = ind["close"]
    mom_13w = ind["mom_40"]
    vol_inv = 1.0 / ind["vol_20"].replace(0, np.nan)
    carry_proxy = _norm(vol_inv, 60)
    mom_rank = _norm(mom_13w, 60)
    vol_rank = _norm(ind["vol_20"], 60)
    composite = 0.40 * carry_proxy + 0.40 * mom_rank - 0.20 * vol_rank
    raw = composite * 70
    return _clip_signal(raw)


def s007_liquidity_pulse(ind):
    """007 | Shanghai Liquidity Pulse Detector"""
    close = ind["close"]
    volume = ind["volume"]
    vol_pulse = volume / volume.ewm(span=20, adjust=False).mean() - 1
    ret = ind["ret_1"].abs()
    vol_median = volume.rolling(20).median().replace(0, np.nan)
    price_impact = ret / (volume / vol_median).replace(0, np.nan)
    price_impact_n = -_z(price_impact, 60)
    vwap_proxy = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    ofi = ((close - vwap_proxy) / close * volume).rolling(20).sum()
    ofi = ofi / volume.rolling(20).sum()
    ofi_n = _z(ofi, 60)
    liq = 0.30 * _z(vol_pulse, 60) + 0.25 * price_impact_n + 0.25 * _z(ind["efficiency"], 60) + 0.20 * ofi_n
    raw = liq.clip(-3, 3) / 3 * 70
    return _clip_signal(raw)


def s008_breakout_range(ind):
    """008 | London FX Breakout Range System (daily proxy: Donchian breakout)"""
    close = ind["close"]
    hi20 = ind["donch_high"]
    lo20 = ind["donch_low"]
    rng = hi20 - lo20
    med_rng = rng.rolling(20).median()
    tight = (rng < 1.5 * med_rng).astype(float)
    breakout_up = ((close > hi20.shift(1)) & (tight > 0)).astype(float)
    breakout_dn = ((close < lo20.shift(1)) & (tight > 0)).astype(float)
    raw = (breakout_up - breakout_dn).rolling(5).sum() * 30
    return _clip_signal(raw)


def s009_hurst_adaptive(ind):
    """009 | Hurst Exponent Adaptive Trend/MR Classifier"""
    hurst = ind["hurst"]
    mom = ind["mom_score"]
    rsi_mr = (50 - ind["rsi"]) / 50
    trending = (hurst > 0.55).astype(float)
    mr = (hurst < 0.45).astype(float)
    neutral = 1 - trending - mr
    signal = trending * mom * 80 + mr * rsi_mr * 60 + neutral * mom * 20
    return _clip_signal(signal)


def s010_kalman_trend(ind):
    """010 | Kalman Filter State-Space Trend Extraction"""
    close = ind["close"]
    vol = ind["vol_20"].fillna(0.01)
    vals = close.values.astype(float)
    n = len(vals)
    mu = np.zeros(n)
    drift = np.zeros(n)
    p11 = np.zeros(n)
    p22 = np.zeros(n)
    mu[0] = vals[0]; drift[0] = 0; p11[0] = 1.0; p22[0] = 0.001
    q1, q2 = 0.001, 1e-5
    for i in range(1, n):
        mu_pred = mu[i-1] + drift[i-1]
        d_pred = drift[i-1]
        p11_pred = p11[i-1] + q1
        p22_pred = p22[i-1] + q2
        v = vol.iloc[i] if not np.isnan(vol.iloc[i]) else 0.01
        r = max(v**2 * 252, 0.01)
        k1 = p11_pred / (p11_pred + r)
        innov = vals[i] - mu_pred
        mu[i] = mu_pred + k1 * innov
        drift[i] = d_pred + (p22_pred / (p11_pred + r)) * innov
        p11[i] = (1 - k1) * p11_pred
        p22[i] = p22_pred - p22_pred**2 / (p11_pred + r)
    drift_s = pd.Series(drift, index=close.index)
    sig_ratio = drift_s / np.sqrt(pd.Series(p22, index=close.index).replace(0, np.nan))
    raw = sig_ratio.clip(-3, 3) / 3 * 80
    return _clip_signal(raw)


def s011_supertrend(ind):
    """011 | Supertrend ATR Band System"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    atr = ind["atr14"]
    mult = 3.0
    hl2 = (high + low) / 2
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    n = len(close)
    st = np.zeros(n)
    direction = np.ones(n)
    ub_v = ub.values.astype(float)
    lb_v = lb.values.astype(float)
    c = close.values.astype(float)
    for i in range(1, n):
        if lb_v[i] > lb_v[i-1] or c[i-1] < lb_v[i-1]:
            pass
        else:
            lb_v[i] = lb_v[i-1]
        if ub_v[i] < ub_v[i-1] or c[i-1] > ub_v[i-1]:
            pass
        else:
            ub_v[i] = ub_v[i-1]
        if direction[i-1] == 1:
            if c[i] < lb_v[i]:
                direction[i] = -1
                st[i] = ub_v[i]
            else:
                direction[i] = 1
                st[i] = lb_v[i]
        else:
            if c[i] > ub_v[i]:
                direction[i] = 1
                st[i] = lb_v[i]
            else:
                direction[i] = -1
                st[i] = ub_v[i]
    raw = pd.Series(direction, index=close.index) * 60
    dist = (close - pd.Series(st, index=close.index)) / atr.replace(0, np.nan)
    raw = raw + dist.clip(-3, 3) * 10
    return _clip_signal(raw)


def s012_donchian_turtle(ind):
    """012 | Donchian Channel Turtle System (Modified)"""
    close = ind["close"]
    dp = ind["donch_pct"]
    exit_hi = ind["high"].rolling(10).max()
    exit_lo = ind["low"].rolling(10).min()
    breakout_long = (close > ind["donch_high"].shift(1)).astype(float)
    breakout_short = (close < ind["donch_low"].shift(1)).astype(float)
    exit_long = (close < exit_lo.shift(1)).astype(float)
    exit_short = (close > exit_hi.shift(1)).astype(float)
    raw = (breakout_long * 60 - breakout_short * 60).rolling(5).sum()
    raw = raw - exit_long * 40 + exit_short * 40
    return _clip_signal(raw.clip(-100, 100))


def s013_vortex_trend(ind):
    """013 | Vortex Indicator Trend Strength"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    tr = ind["tr"]
    n = 14
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    vi_plus = vm_plus.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    vi_minus = vm_minus.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    vortex_diff = vi_plus - vi_minus
    cross_up = _crossover(vi_plus, vi_minus)
    cross_dn = _crossunder(vi_plus, vi_minus)
    raw = _z(vortex_diff, 40) * 50 + cross_up * 30 - cross_dn * 30
    return _clip_signal(raw)


def s014_chande_momentum(ind):
    """014 | Chande Momentum Oscillator Divergence"""
    close = ind["close"]
    diff = close.diff()
    su = diff.clip(lower=0).rolling(14).sum()
    sd = (-diff.clip(upper=0)).rolling(14).sum()
    cmo = 100 * (su - sd) / (su + sd).replace(0, np.nan)
    cmo_z = _z(cmo, 60)
    raw = cmo_z * 40
    ob = (cmo > 50).astype(float) * (-20)
    os_s = (cmo < -50).astype(float) * 20
    raw = raw + ob + os_s
    return _clip_signal(raw)


def s015_parabolic_sar(ind):
    """015 | Parabolic SAR Acceleration System"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    n = len(close)
    c = close.values.astype(float)
    h = high.values.astype(float)
    l = low.values.astype(float)
    sar = np.zeros(n)
    direction = np.ones(n)
    af = 0.02
    ep = h[0]
    sar[0] = l[0]
    for i in range(1, n):
        if direction[i-1] == 1:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[max(0,i-2)])
            if c[i] < sar[i]:
                direction[i] = -1
                sar[i] = ep
                ep = l[i]
                af = 0.02
            else:
                direction[i] = 1
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + 0.02, 0.20)
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[max(0,i-2)])
            if c[i] > sar[i]:
                direction[i] = 1
                sar[i] = ep
                ep = h[i]
                af = 0.02
            else:
                direction[i] = -1
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + 0.02, 0.20)
    adx_filter = (ind["adx"] > 25).astype(float) * 0.6 + 0.4
    raw = pd.Series(direction, index=close.index) * 60 * adx_filter
    return _clip_signal(raw)


def s016_klinger_oscillator(ind):
    """016 | Klinger Oscillator Volume Trend"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    hlc = high + low + close
    trend = np.sign(hlc - hlc.shift(1)).fillna(1)
    dm = high - low
    cm = dm.copy()
    for i in range(1, len(cm)):
        if trend.iloc[i] == trend.iloc[i-1]:
            cm.iloc[i] = cm.iloc[i-1] + dm.iloc[i]
        else:
            cm.iloc[i] = dm.iloc[i-1] + dm.iloc[i]
    cm = cm.replace(0, np.nan)
    vf = volume * (2 * dm / cm - 1).abs() * trend * 100
    kvo = vf.ewm(span=34, adjust=False).mean() - vf.ewm(span=55, adjust=False).mean()
    sig = kvo.ewm(span=13, adjust=False).mean()
    hist = kvo - sig
    raw = _z(hist, 40) * 50
    return _clip_signal(raw)


def s017_alma_trend(ind):
    """017 | Arnaud Legoux Moving Average Trend"""
    close = ind["close"]
    alma_f = _alma(close, 9, 0.85, 6)
    alma_s = _alma(close, 21, 0.85, 6)
    diff = alma_f - alma_s
    sig = _alma(diff, 5, 0.85, 6)
    hist = diff - sig
    strength = diff.abs() / ind["atr14"].replace(0, np.nan)
    raw = _z(hist, 30) * 40 + np.sign(diff) * strength.clip(0, 3) / 3 * 30
    return _clip_signal(raw)


def s018_mcginley_dynamic(ind):
    """018 | McGinley Dynamic Line"""
    close = ind["close"]
    n_per = 14
    k = 0.6
    vals = close.values.astype(float)
    md = np.full(len(vals), np.nan)
    md[0] = vals[0]
    for i in range(1, len(vals)):
        if np.isnan(md[i-1]) or md[i-1] == 0:
            md[i] = vals[i]
        else:
            ratio = vals[i] / md[i-1]
            denom = k * n_per * ratio**4
            if denom == 0:
                md[i] = vals[i]
            else:
                md[i] = md[i-1] + (vals[i] - md[i-1]) / denom
    md_s = pd.Series(md, index=close.index)
    slope = md_s.pct_change(3)
    above = (close > md_s).astype(float) * 2 - 1
    vol_confirm = (ind["vol_rel"] > 1.5).astype(float) * 0.4 + 0.6
    raw = above * slope.abs().clip(0, 0.05) / 0.05 * 60 * vol_confirm
    return _clip_signal(raw)


def s019_bb_squeeze_breakout(ind):
    """019 | Bollinger BandWidth Squeeze Breakout"""
    squeeze = ind["bb_squeeze"]
    pctb = ind["bb_pctb"]
    mom = ind["mom_5"]
    in_squeeze = (squeeze < 0.2).astype(float)
    breakout_up = (pctb > 1.0) & (in_squeeze.shift(5) > 0)
    breakout_dn = (pctb < 0.0) & (in_squeeze.shift(5) > 0)
    direction = breakout_up.astype(float) - breakout_dn.astype(float)
    raw = direction.rolling(5).sum() * 30 + mom * 30
    return _clip_signal(raw)


def s020_keltner_mean_reversion(ind):
    """020 | Keltner Channel Mean Reversion"""
    kc_pct = ind["kc_pct"]
    rsi = ind["rsi"]
    oversold = ((kc_pct < 0.1) & (rsi < 35)).astype(float)
    overbought = ((kc_pct > 0.9) & (rsi > 65)).astype(float)
    raw = oversold * 60 - overbought * 60
    trend = ind["trend_score"]
    raw = raw + trend * 20
    return _clip_signal(raw)


def s021_alligator_fractal(ind):
    """021 | Williams Alligator Fractal Strategy"""
    close = ind["close"]
    jaw = ind["sma_200"].shift(8) if ind["n"] > 208 else ind["sma_50"]
    teeth = ind["ema_21"].shift(5)
    lips = ind["ema_10"].shift(3)
    spread = (lips - jaw) / close * 100
    awake = (lips > teeth) & (teeth > jaw)
    sleeping = lips.rolling(10).std() / close * 100 < 0.1
    raw = awake.astype(float) * spread.clip(-5, 5) / 5 * 70
    raw = raw - sleeping.astype(float) * 20
    return _clip_signal(raw)


def s022_range_expansion(ind):
    """022 | Tokyo Range Expansion Index (REI)"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    h1 = high - high.shift(2)
    l1 = low - low.shift(2)
    cond_h = ((high.shift(1) >= close.shift(3)) | (high.shift(2) >= close.shift(4))).astype(float)
    cond_l = ((low.shift(1) <= close.shift(3)) | (low.shift(2) <= close.shift(4))).astype(float)
    rei_num = (h1 * cond_h + l1 * cond_l).rolling(5).sum()
    rei_den = (h1.abs() + l1.abs()).rolling(5).sum().replace(0, np.nan)
    rei = (rei_num / rei_den * 100).clip(-100, 100)
    raw = rei * 0.7
    return _clip_signal(raw)


def s023_dmi_multiperiod(ind):
    """023 | Directional Movement Index Multi-Period"""
    adx = ind["adx"]
    trend_adx = ind["trend_adx"]
    adx_slope = adx.diff(5)
    strength = ind["adx_regime"]
    raw = trend_adx * 50 + (adx_slope > 0).astype(float) * strength * 30
    return _clip_signal(raw)


def s024_cci_mean_reversion(ind):
    """024 | Commodity Channel Index Mean Reversion"""
    cci = ind["cci"]
    oversold = (cci < -100).astype(float) * ((-100 - cci) / 200).clip(0, 1)
    overbought = (cci > 100).astype(float) * ((cci - 100) / 200).clip(0, 1)
    cross_up = _crossover(cci, pd.Series(-100, index=cci.index))
    cross_dn = _crossunder(cci, pd.Series(100, index=cci.index))
    raw = oversold * 50 - overbought * 50 + cross_up * 30 - cross_dn * 30
    return _clip_signal(raw)


def s025_stoch_rsi_bounce(ind):
    """025 | Stochastic RSI Oversold Bounce"""
    rsi = ind["rsi"]
    rsi_lo = rsi.rolling(14).min()
    rsi_hi = rsi.rolling(14).max()
    stoch_rsi = ((rsi - rsi_lo) / (rsi_hi - rsi_lo).replace(0, np.nan) * 100).rolling(3).mean()
    oversold = (stoch_rsi < 20).astype(float) * (20 - stoch_rsi) / 20
    overbought = (stoch_rsi > 80).astype(float) * (stoch_rsi - 80) / 20
    bounce = _crossover(stoch_rsi, pd.Series(20, index=rsi.index))
    raw = oversold * 40 - overbought * 40 + bounce * 40
    return _clip_signal(raw)


def s026_zero_lag_dema(ind):
    """026 | Zero-Lag DEMA Momentum"""
    close = ind["close"]
    ema1 = close.ewm(span=21, adjust=False).mean()
    ema2 = ema1.ewm(span=21, adjust=False).mean()
    dema = 2 * ema1 - ema2
    zl = close + (close - dema)
    zl_slope = zl.pct_change(3)
    raw = _z(zl_slope, 40) * 50
    return _clip_signal(raw)


def s027_connors_rsi(ind):
    """027 | Connors RSI Short-Term Reversal"""
    close = ind["close"]
    rsi3 = close.diff().pipe(lambda d: d.clip(lower=0).rolling(3).mean() / (-d.clip(upper=0)).rolling(3).mean().replace(0, np.nan)).pipe(lambda rs: 100 - 100 / (1 + rs))
    streak = np.zeros(len(close))
    c = close.values.astype(float)
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            streak[i] = max(streak[i-1], 0) + 1
        elif c[i] < c[i-1]:
            streak[i] = min(streak[i-1], 0) - 1
    streak_s = pd.Series(streak, index=close.index)
    streak_rsi = streak_s.rolling(14).apply(lambda x: (x[-1] - x.min()) / max(x.max() - x.min(), 1) * 100, raw=True)
    pct_rank = close.pct_change(1).rolling(100, min_periods=20).rank(pct=True) * 100
    crsi = (rsi3 + streak_rsi + pct_rank) / 3
    raw = (50 - crsi) * 1.5
    return _clip_signal(raw)


def s028_elder_triple_screen(ind):
    """028 | Elder Triple Screen System"""
    trend_w = ind["ma50_slope_n"]
    macd_h = ind["macd_hist"]
    force = ind["vol_rel"] * ind["ret_1"]
    force_ema = force.ewm(span=2, adjust=False).mean()
    screen1 = np.sign(trend_w)
    screen2 = np.sign(macd_h)
    screen3 = np.sign(force_ema)
    agreement = (screen1 + screen2 + screen3) / 3
    raw = agreement * 70
    return _clip_signal(raw)


def s029_aroon_oscillator(ind):
    """029 | Aroon Oscillator Trend Detection"""
    high = ind["high"]
    low = ind["low"]
    n = 25
    aroon_up = high.rolling(n + 1).apply(lambda x: x.argmax() / n * 100, raw=True)
    aroon_dn = low.rolling(n + 1).apply(lambda x: x.argmin() / n * 100, raw=True)
    aroon_osc = aroon_up - aroon_dn
    raw = aroon_osc * 0.7
    return _clip_signal(raw)


def s030_relative_strength_rotation(ind):
    """030 | Mumbai Relative Strength Rotation"""
    close = ind["close"]
    rs_mom = close.pct_change(20) / ind["vol_20"].replace(0, np.nan)
    rs_accel = rs_mom - rs_mom.shift(10)
    rs_z = _z(rs_mom, 60)
    accel_z = _z(rs_accel, 60)
    quadrant = np.where((rs_z > 0) & (accel_z > 0), 1,
               np.where((rs_z > 0) & (accel_z < 0), 0.3,
               np.where((rs_z < 0) & (accel_z > 0), -0.3, -1)))
    raw = pd.Series(quadrant, index=close.index).astype(float) * 60
    return _clip_signal(raw)


def s031_fibonacci_confluence(ind):
    """031 | Fibonacci Confluence Zone Strategy"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    hi52 = high.rolling(52 * 5, min_periods=60).max()
    lo52 = low.rolling(52 * 5, min_periods=60).min()
    rng = hi52 - lo52
    fib_382 = hi52 - 0.382 * rng
    fib_500 = hi52 - 0.500 * rng
    fib_618 = hi52 - 0.618 * rng
    near_382 = (close - fib_382).abs() / close < 0.02
    near_500 = (close - fib_500).abs() / close < 0.02
    near_618 = (close - fib_618).abs() / close < 0.02
    confluence = near_382.astype(float) + near_500.astype(float) + near_618.astype(float)
    above = (close > fib_500).astype(float) * 2 - 1
    raw = confluence * above * 30 + ind["mom_score"] * 40
    return _clip_signal(raw)


def s032_momentum_carry(ind):
    """032 | Singapore Dollar Momentum Carry (OHLCV proxy)"""
    close = ind["close"]
    mom = ind["mom_20"]
    carry_proxy = -_z(ind["vol_20"], 60)
    raw = (0.50 * mom + 0.50 * carry_proxy) * 70
    return _clip_signal(raw)


def s033_chaikin_money_flow(ind):
    """033 | Chaikin Money Flow Accumulation"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    cmf = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    cmf_z = _z(cmf, 60)
    raw = cmf_z * 50
    return _clip_signal(raw)


def s034_yield_curve_proxy(ind):
    """034 | German Bund Yield Curve Steepener (vol regime proxy)"""
    vol_ratio = ind["vol_20"] / ind["vol_60"].replace(0, np.nan)
    vol_slope = vol_ratio.pct_change(5)
    trend = ind["trend_score"]
    raw = _z(vol_slope, 40) * 30 + trend * 40
    return _clip_signal(raw)


def s035_weis_wave_volume(ind):
    """035 | Weis Wave Volume Analysis"""
    close = ind["close"]
    volume = ind["volume"]
    direction = np.sign(close.diff()).fillna(0)
    wave_vol = (direction * volume).rolling(10).sum()
    wave_norm = _z(wave_vol, 40)
    price_conf = np.sign(close.diff(5))
    vol_conf = (np.sign(wave_vol) == price_conf).astype(float) * 0.4 + 0.6
    raw = wave_norm * vol_conf * 50
    return _clip_signal(raw)


def s036_elder_force_index(ind):
    """036 | Elder Force Index Impulse System"""
    close = ind["close"]
    volume = ind["volume"]
    fi = close.diff() * volume
    fi2 = fi.ewm(span=2, adjust=False).mean()
    fi13 = fi.ewm(span=13, adjust=False).mean()
    ema_slope = ind["ema_21"].pct_change(1)
    impulse_green = ((ema_slope > 0) & (fi2 > 0)).astype(float)
    impulse_red = ((ema_slope < 0) & (fi2 < 0)).astype(float)
    raw = impulse_green * 50 - impulse_red * 50 + _z(fi13, 40) * 30
    return _clip_signal(raw)


def s037_vol_regime_switch(ind):
    """037 | Zurich Volatility Regime Switch"""
    vol_pct = ind["vol_pct"]
    mom = ind["mom_score"]
    rsi_mr = (50 - ind["rsi"]) / 50
    low_vol = (vol_pct < 0.3).astype(float)
    high_vol = (vol_pct > 0.7).astype(float)
    mid_vol = 1 - low_vol - high_vol
    signal = low_vol * mom * 70 + mid_vol * mom * 50 + high_vol * rsi_mr * 40
    return _clip_signal(signal)


def s038_opening_gap(ind):
    """038 | Hong Kong HSI Opening Gap Strategy (daily proxy)"""
    close = ind["close"]
    open_p = ind["open"]
    gap = (open_p - close.shift(1)) / close.shift(1) * 100
    gap_z = _z(gap, 60)
    fade = -gap_z * 30
    continuation = gap_z * (ind["mom_5"] > 0).astype(float) * 20
    raw = np.where(gap.abs() > gap.abs().rolling(20).quantile(0.8), fade, continuation)
    return _clip_signal(pd.Series(raw, index=close.index))


def s039_anchored_vwap(ind):
    """039 | Anchored VWAP Institutional Level Strategy"""
    close = ind["close"]
    volume = ind["volume"]
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    dist = (close - vwap) / ind["atr14"].replace(0, np.nan)
    raw = dist.clip(-3, 3) / 3 * 60 * ind["vol_dampener"]
    return _clip_signal(raw)


def s040_momentum_carry_br(ind):
    """040 | Brazilian Real Momentum-Carry Hybrid (OHLCV proxy)"""
    mom = ind["mom_20"]
    vol_inv = -_z(ind["vol_20"], 60)
    trend = ind["trend_score"]
    raw = 0.35 * mom * 70 + 0.35 * vol_inv * 50 + 0.30 * trend * 60
    return _clip_signal(raw)


def s041_pivot_confluence(ind):
    """041 | Pivot Point Confluence Trading"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    pp = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    r1 = 2 * pp - low.shift(1)
    s1 = 2 * pp - high.shift(1)
    r2 = pp + (high.shift(1) - low.shift(1))
    s2 = pp - (high.shift(1) - low.shift(1))
    near_s1 = ((close - s1).abs() / close < 0.005).astype(float)
    near_r1 = ((close - r1).abs() / close < 0.005).astype(float)
    above_pp = (close > pp).astype(float) * 2 - 1
    raw = above_pp * 30 + near_s1 * 30 - near_r1 * 30 + ind["mom_score"] * 30
    return _clip_signal(raw)


def s042_risk_appetite_barometer(ind):
    """042 | Swedish Krona Risk Appetite Barometer (OHLCV proxy)"""
    mom = ind["mom_10"]
    vol_z = _z(ind["vol_20"], 120)
    risk_on = (mom > 0) & (vol_z < 0)
    risk_off = (mom < 0) & (vol_z > 0)
    raw = risk_on.astype(float) * 50 - risk_off.astype(float) * 50
    raw = raw + ind["trend_score"] * 30
    return _clip_signal(raw)


def s043_vwmo(ind):
    """043 | Volume-Weighted Momentum Oscillator"""
    close = ind["close"]
    volume = ind["volume"]
    ret = close.pct_change(1)
    vw_ret = (ret * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    vwmo = _z(vw_ret, 60)
    raw = vwmo * 50
    return _clip_signal(raw)


def s044_adaptive_rsi_hilbert(ind):
    """044 | Adaptive RSI with Hilbert Transform Period (simplified)"""
    close = ind["close"]
    rsi = ind["rsi"]
    er = ind["efficiency"]
    adaptive_period = (14 * (1 - er) + 5 * er).clip(5, 30).astype(int)
    rsi_z = _z(rsi, 60)
    raw = (50 - rsi) / 50 * 40 + rsi_z * (-30)
    return _clip_signal(raw)


def s045_dual_momentum(ind):
    """045 | Tel Aviv Dual Momentum"""
    abs_mom = ind["mom_20"]
    rel_mom = ind["mom_40"]
    dual = np.where((abs_mom > 0) & (rel_mom > 0), 1,
           np.where((abs_mom < 0) & (rel_mom < 0), -1, 0))
    strength = (abs_mom.abs() + rel_mom.abs()) / 2
    raw = pd.Series(dual, index=ind["close"].index).astype(float) * strength * 80
    return _clip_signal(raw)


def s046_trix_divergence(ind):
    """046 | Triple Exponential Average (TRIX) Divergence"""
    close = ind["close"]
    ema1 = close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = ema3.pct_change(1) * 10000
    trix_signal = trix.ewm(span=9, adjust=False).mean()
    hist = trix - trix_signal
    raw = _z(hist, 40) * 50
    return _clip_signal(raw)


def s047_mass_index_reversal(ind):
    """047 | Mass Index Reversal Bulge"""
    high = ind["high"]
    low = ind["low"]
    rng = high - low
    ema_rng = rng.ewm(span=9, adjust=False).mean()
    dema_rng = ema_rng.ewm(span=9, adjust=False).mean()
    ratio = ema_rng / dema_rng.replace(0, np.nan)
    mass = ratio.rolling(25).sum()
    bulge = (mass > 27).astype(float)
    reversal = bulge & (mass < mass.shift(1))
    trend_dir = ind["trend_score"]
    raw = reversal.astype(float) * (-np.sign(trend_dir)) * 60
    return _clip_signal(raw)


def s048_gold_momentum(ind):
    """048 | Dubai Gold Dinar Momentum (OHLCV proxy)"""
    mom = ind["mom_20"]
    vol_adj = ind["vol_dampener"]
    above_200 = ind["above_200"]
    raw = mom * vol_adj * 60 + (above_200 * 2 - 1) * 20
    return _clip_signal(raw)


def s049_consecutive_days(ind):
    """049 | Consecutive Day Pattern with Adaptive Scaling"""
    close = ind["close"]
    c = close.values.astype(float)
    streak = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            streak[i] = max(streak[i-1], 0) + 1
        elif c[i] < c[i-1]:
            streak[i] = min(streak[i-1], 0) - 1
    streak_s = pd.Series(streak, index=close.index)
    vol_adj = 1.0 / (1.0 + ind["vol_pct"])
    reversal = -streak_s.clip(-5, 5) / 5
    raw = reversal * vol_adj * 60
    return _clip_signal(raw)


def s050_sector_rotation(ind):
    """050 | Toronto Resource Sector Rotation (OHLCV proxy)"""
    mom_short = ind["mom_5"]
    mom_long = ind["mom_40"]
    vol_flow = ind["vol_flow"]
    rotation = np.where((mom_short > 0) & (mom_long > 0) & (vol_flow > 0), 1,
               np.where((mom_short < 0) & (mom_long < 0) & (vol_flow < 0), -1, 0))
    raw = pd.Series(rotation, index=ind["close"].index).astype(float) * 60
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    1: ("Ehlers Instantaneous Trendline", s001_ehlers_trendline),
    2: ("KAMA Momentum", s002_kama_momentum),
    3: ("Triple HMA Regime", s003_triple_hma_regime),
    4: ("Ichimoku Extended Cloud", s004_ichimoku_extended),
    5: ("DAX Momentum Oscillator", s005_dax_momentum_oscillator),
    6: ("Carry-Momentum Filter", s006_carry_momentum_filter),
    7: ("Liquidity Pulse Detector", s007_liquidity_pulse),
    8: ("Breakout Range System", s008_breakout_range),
    9: ("Hurst Adaptive Classifier", s009_hurst_adaptive),
    10: ("Kalman Trend Extraction", s010_kalman_trend),
    11: ("Supertrend ATR Band", s011_supertrend),
    12: ("Donchian Turtle System", s012_donchian_turtle),
    13: ("Vortex Trend Strength", s013_vortex_trend),
    14: ("Chande Momentum Oscillator", s014_chande_momentum),
    15: ("Parabolic SAR Acceleration", s015_parabolic_sar),
    16: ("Klinger Volume Oscillator", s016_klinger_oscillator),
    17: ("ALMA Trend", s017_alma_trend),
    18: ("McGinley Dynamic", s018_mcginley_dynamic),
    19: ("BB Squeeze Breakout", s019_bb_squeeze_breakout),
    20: ("Keltner Mean Reversion", s020_keltner_mean_reversion),
    21: ("Alligator Fractal", s021_alligator_fractal),
    22: ("Range Expansion Index", s022_range_expansion),
    23: ("DMI Multi-Period", s023_dmi_multiperiod),
    24: ("CCI Mean Reversion", s024_cci_mean_reversion),
    25: ("Stochastic RSI Bounce", s025_stoch_rsi_bounce),
    26: ("Zero-Lag DEMA", s026_zero_lag_dema),
    27: ("Connors RSI Reversal", s027_connors_rsi),
    28: ("Elder Triple Screen", s028_elder_triple_screen),
    29: ("Aroon Oscillator", s029_aroon_oscillator),
    30: ("Relative Strength Rotation", s030_relative_strength_rotation),
    31: ("Fibonacci Confluence", s031_fibonacci_confluence),
    32: ("Momentum-Carry Hybrid", s032_momentum_carry),
    33: ("Chaikin Money Flow", s033_chaikin_money_flow),
    34: ("Yield Curve Proxy", s034_yield_curve_proxy),
    35: ("Weis Wave Volume", s035_weis_wave_volume),
    36: ("Elder Force Index", s036_elder_force_index),
    37: ("Vol Regime Switch", s037_vol_regime_switch),
    38: ("Opening Gap Strategy", s038_opening_gap),
    39: ("Anchored VWAP", s039_anchored_vwap),
    40: ("Momentum-Carry BR", s040_momentum_carry_br),
    41: ("Pivot Confluence", s041_pivot_confluence),
    42: ("Risk Appetite Barometer", s042_risk_appetite_barometer),
    43: ("VWMO", s043_vwmo),
    44: ("Adaptive RSI Hilbert", s044_adaptive_rsi_hilbert),
    45: ("Dual Momentum", s045_dual_momentum),
    46: ("TRIX Divergence", s046_trix_divergence),
    47: ("Mass Index Reversal", s047_mass_index_reversal),
    48: ("Gold Momentum", s048_gold_momentum),
    49: ("Consecutive Day Pattern", s049_consecutive_days),
    50: ("Sector Rotation", s050_sector_rotation),
}
