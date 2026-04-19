"""
SECTION IV: VOLUME & MICROSTRUCTURE STRATEGIES (151-200)
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

def s151_obv_divergence(ind):
    """151 | On-Balance Volume Divergence"""
    close = ind["close"]
    volume = ind["volume"]
    ret = close.diff()
    obv = (volume * np.sign(ret)).cumsum()
    obv_slope = obv.diff(10) / obv.rolling(20).std().replace(0, np.nan)
    price_slope = close.diff(10) / close.rolling(20).std().replace(0, np.nan)
    divergence = obv_slope - price_slope
    z = _z(divergence, 40)
    raw = z * 50
    return _clip_signal(raw)


def s152_vwap_institutional(ind):
    """152 | VWAP Institutional Flow"""
    close = ind["close"]
    volume = ind["volume"]
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    dist = (close - vwap) / ind["atr14"].replace(0, np.nan)
    trend = ind["above_200"].fillna(0.5)
    raw = dist.clip(-3, 3) / 3 * trend * 60
    return _clip_signal(raw)


def s153_cvd_strategy(ind):
    """153 | Cumulative Volume Delta (CVD) Strategy"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    # Approximate buy/sell volume using close position in range
    rng = (high - low).replace(0, np.nan)
    buy_pct = (close - low) / rng
    sell_pct = (high - close) / rng
    buy_vol = volume * buy_pct
    sell_vol = volume * sell_pct
    delta = buy_vol - sell_vol
    cvd = delta.cumsum()
    cvd_z = _z(cvd.diff(10), 40)
    raw = cvd_z * 55
    return _clip_signal(raw)


def s154_volume_poc(ind):
    """154 | Volume Profile Point of Control (POC) Trading"""
    close = ind["close"]
    volume = ind["volume"]
    # POC proxy: volume-weighted median price over 20 days
    vwap_20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    dist = (close - vwap_20) / ind["atr14"].replace(0, np.nan)
    # Price near POC = range; far from POC = breakout
    near_poc = (dist.abs() < 0.5).astype(float)
    breakout = (dist.abs() > 2.0).astype(float) * np.sign(dist)
    raw = breakout * 50 + near_poc * ind["trend_score"] * 20
    return _clip_signal(raw)


def s155_market_profile_tpo(ind):
    """155 | Market Profile TPO Distribution"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Value area proxy: 70% of volume within 1 ATR of VWAP
    atr = ind["atr14"]
    ema = ind["ema_21"]
    dist = (close - ema) / atr.replace(0, np.nan)
    in_value = (dist.abs() < 1.0).astype(float)
    above_value = (dist > 1.5).astype(float)
    below_value = (dist < -1.5).astype(float)
    raw = below_value * 40 - above_value * 30 + in_value * ind["trend_score"] * 25
    return _clip_signal(raw)


def s156_wyckoff_vsa(ind):
    """156 | Wyckoff Volume Spread Analysis (VSA)"""
    close = ind["close"]
    volume = ind["volume"]
    spread = ind["high"] - ind["low"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    spread_ma = spread.rolling(20).mean().replace(0, np.nan)
    # Effort vs Result
    high_vol_narrow = ((volume > 1.5 * vol_ma) & (spread < 0.7 * spread_ma)).astype(float)
    low_vol_wide = ((volume < 0.7 * vol_ma) & (spread > 1.5 * spread_ma)).astype(float)
    ret = ind["ret_1"]
    # High vol + narrow spread + down = accumulation (bullish)
    accum = high_vol_narrow * (ret < 0).astype(float)
    # High vol + narrow spread + up = distribution (bearish)
    distrib = high_vol_narrow * (ret > 0).astype(float)
    raw = accum * 55 - distrib * 55 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s157_twap_deviation(ind):
    """157 | TWAP Deviation Strategy"""
    close = ind["close"]
    twap = close.rolling(20).mean()
    dev = (close - twap) / ind["atr14"].replace(0, np.nan)
    z = _z(dev, 40)
    raw = z * 45 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s158_volume_climax(ind):
    """158 | Volume Climax Exhaustion"""
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean()
    climax = (volume > 3 * vol_ma).astype(float)
    ret = ind["ret_1"]
    # Climax on up day = potential top; climax on down day = potential bottom
    bull_climax = climax * (ret > 0).astype(float)
    bear_climax = climax * (ret < 0).astype(float)
    raw = bear_climax * 50 - bull_climax * 40
    return _clip_signal(raw, smooth=5)


def s159_ad_institutional(ind):
    """159 | Accumulation/Distribution Institutional Flow"""
    ad = ind["ad_score"]
    ad_z = _z(ad, 40)
    mom = ind["mom_10"]
    divergence = ad_z - _z(mom, 40)
    raw = divergence * 30 + ad_z * 30
    return _clip_signal(raw)


def s160_lob_imbalance(ind):
    """160 | Limit Order Book Imbalance Strategy (volume proxy)"""
    volume = ind["volume"]
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Buy pressure: close near high on volume
    rng = (high - low).replace(0, np.nan)
    buy_pres = ((close - low) / rng) * volume
    sell_pres = ((high - close) / rng) * volume
    imbalance = (buy_pres - sell_pres) / (buy_pres + sell_pres).replace(0, np.nan)
    z = _z(imbalance, 20)
    raw = z * 55
    return _clip_signal(raw)


def s161_vpin(ind):
    """161 | Volume-Synchronized Probability of Informed Trading (VPIN)"""
    close = ind["close"]
    volume = ind["volume"]
    ret = ind["ret_1"]
    # VPIN proxy: abs(return) * volume as informed trade proxy
    informed = ret.abs() * volume
    total = volume
    vpin = informed.rolling(20).sum() / total.rolling(20).sum().replace(0, np.nan)
    z = _z(vpin, 60)
    # High VPIN = toxic flow, cautious
    raw = -z * 35 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s162_opening_auction(ind):
    """162 | Shanghai Opening Auction Volume Signal"""
    gap = (ind["open"] - ind["close"].shift(1)) / ind["close"].shift(1) * 100
    vol_rel = ind["vol_rel"]
    raw = _z(gap, 40) * vol_rel.clip(0.5, 3) * 35
    return _clip_signal(raw)


def s163_mfi_divergence(ind):
    """163 | Money Flow Index (MFI) Divergence System"""
    mfi = ind["mfi"]
    close = ind["close"]
    mfi_slope = mfi.diff(10)
    price_slope = close.pct_change(10) * 100
    divergence = _z(mfi_slope, 40) - _z(price_slope, 40)
    raw = divergence * 40 + (50 - mfi) / 50 * 20
    return _clip_signal(raw)


def s164_tick_vol_momentum(ind):
    """164 | Tick Volume Momentum Oscillator"""
    volume = ind["volume"]
    ret = ind["ret_1"]
    tick_mom = (volume * np.sign(ret)).rolling(14).sum()
    z = _z(tick_mom, 40)
    raw = z * 55
    return _clip_signal(raw)


def s165_force_index(ind):
    """165 | Force Index (Elder) Exhaustion Pattern"""
    close = ind["close"]
    volume = ind["volume"]
    fi = close.diff() * volume
    fi_13 = fi.ewm(span=13, adjust=False).mean()
    fi_z = _z(fi_13, 40)
    # Extreme force = exhaustion
    exhaust_up = (fi_z > 2.0).astype(float) * (fi_z - 2.0)
    exhaust_dn = (fi_z < -2.0).astype(float) * (-2.0 - fi_z)
    raw = -exhaust_up * 30 + exhaust_dn * 30 + fi_z.clip(-2, 2) / 2 * 40
    return _clip_signal(raw)


def s166_vol_spread_divergence(ind):
    """166 | Volume Spread Divergence (Market Maker Absorption)"""
    spread = ind["high"] - ind["low"]
    volume = ind["volume"]
    effort = volume / spread.replace(0, np.nan)
    effort_z = _z(effort, 40)
    ret = ind["ret_1"]
    # High effort (vol) + low result (spread) = absorption
    raw = -effort_z * np.sign(ret) * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s167_rvol(ind):
    """167 | Relative Volume at Time of Day (RVOL)"""
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    rvol = volume / vol_ma
    rvol_z = _z(rvol, 40)
    ret = ind["ret_1"]
    raw = rvol_z * np.sign(ret) * 45
    return _clip_signal(raw)


def s168_klinger(ind):
    """168 | Klinger Volume Oscillator (KVO) Momentum"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    hlc = high + low + close
    trend = np.sign(hlc.diff())
    dm = high - low
    vol_force = volume * trend * dm * 2 / hlc.replace(0, np.nan) * 100
    kvo = vol_force.ewm(span=34, adjust=False).mean() - vol_force.ewm(span=55, adjust=False).mean()
    kvo_sig = kvo.ewm(span=13, adjust=False).mean()
    z = _z(kvo - kvo_sig, 40)
    raw = z * 55
    return _clip_signal(raw)


def s169_dark_pool(ind):
    """169 | Dark Pool Volume Detection (low-vol large-move proxy)"""
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    ret = ind["ret_1"].abs()
    ret_ma = ret.rolling(20).mean().replace(0, np.nan)
    # Large price move on low volume = potential dark pool activity
    dark_proxy = (ret / ret_ma) / (volume / vol_ma).replace(0, np.nan)
    z = _z(dark_proxy, 60)
    direction = np.sign(ind["ret_1"])
    raw = z * direction * 40
    return _clip_signal(raw)


def s170_cmf_persistence(ind):
    """170 | Chaikin Money Flow (CMF) Persistence"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    cmf = mfv.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    # Persistence: CMF positive for consecutive periods
    cmf_pos = (cmf > 0.05).astype(float)
    persist = cmf_pos.rolling(10).mean()
    raw = (persist - 0.5) * 100 + cmf * 30
    return _clip_signal(raw)


def s171_vroc_breakout(ind):
    """171 | Volume Rate of Change (VROC) Breakout Filter"""
    volume = ind["volume"]
    vroc = volume.pct_change(14) * 100
    vroc_z = _z(vroc, 40)
    trend = ind["trend_score"]
    raw = vroc_z * trend * 50
    return _clip_signal(raw)


def s172_ease_of_movement(ind):
    """172 | Ease of Movement (EMV) Indicator"""
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    dm = ((high + low) / 2).diff()
    box_ratio = (volume / 1e6) / (high - low).replace(0, np.nan)
    emv = dm / box_ratio.replace(0, np.nan)
    emv_14 = emv.rolling(14).mean()
    z = _z(emv_14, 40)
    raw = z * 55
    return _clip_signal(raw)


def s173_net_vol_pressure(ind):
    """173 | Net Volume Pressure Oscillator"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    rng = (high - low).replace(0, np.nan)
    buy_p = ((close - low) / rng) * volume
    sell_p = ((high - close) / rng) * volume
    net = buy_p - sell_p
    net_osc = net.ewm(span=14, adjust=False).mean() - net.ewm(span=28, adjust=False).mean()
    z = _z(net_osc, 40)
    raw = z * 55
    return _clip_signal(raw)


def s174_trin_reversal(ind):
    """174 | Arms Index (TRIN) Extreme Reversal (proxy)"""
    # TRIN proxy: vol_flow as breadth indicator
    vf = ind["vol_flow"]
    z = _z(vf, 40)
    extreme_sell = (z < -2.0).astype(float) * (-2.0 - z)
    extreme_buy = (z > 2.0).astype(float) * (z - 2.0)
    raw = extreme_sell * 40 - extreme_buy * 40 + vf * 30
    return _clip_signal(raw)


def s175_vzo(ind):
    """175 | Volume Zone Oscillator (VZO)"""
    close = ind["close"]
    volume = ind["volume"]
    sign_vol = volume * np.sign(close.diff())
    vzo = sign_vol.ewm(span=14, adjust=False).mean() / volume.ewm(span=14, adjust=False).mean().replace(0, np.nan) * 100
    oversold = (vzo < -40).astype(float) * (-40 - vzo) / 60
    overbought = (vzo > 40).astype(float) * (vzo - 40) / 60
    raw = oversold * 50 - overbought * 50 + _z(vzo, 40) * 20
    return _clip_signal(raw)


def s176_footprint_delta(ind):
    """176 | Footprint Chart Delta Imbalance (proxy)"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    rng = (high - low).replace(0, np.nan)
    delta = ((2 * close - high - low) / rng) * volume
    delta_z = _z(delta.rolling(5).sum(), 20)
    raw = delta_z * 55
    return _clip_signal(raw)


def s177_weis_wave(ind):
    """177 | Weis Wave Volume Analysis"""
    close = ind["close"]
    volume = ind["volume"]
    direction = np.sign(close.diff())
    # Wave volume: cumulate volume by direction
    wave = (volume * direction).rolling(10).sum()
    wave_z = _z(wave, 40)
    raw = wave_z * 55
    return _clip_signal(raw)


def s178_vwm(ind):
    """178 | Volume Weighted Momentum (VWM)"""
    ret = ind["ret_1"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    vw_ret = ret * (volume / vol_ma)
    vwm = vw_ret.rolling(20).sum()
    z = _z(vwm, 40)
    raw = z * 55
    return _clip_signal(raw)


def s179_nvi_smart_money(ind):
    """179 | Negative Volume Index (NVI) Smart Money Tracker"""
    close = ind["close"]
    volume = ind["volume"]
    ret = close.pct_change()
    vol_dn = volume < volume.shift(1)
    nvi = pd.Series(1000.0, index=close.index)
    vals = np.full(len(close), 1000.0)
    r = ret.fillna(0).values
    vd = vol_dn.fillna(False).values
    for i in range(1, len(vals)):
        vals[i] = vals[i-1] * (1 + r[i]) if vd[i] else vals[i-1]
    nvi = pd.Series(vals, index=close.index)
    nvi_ma = nvi.ewm(span=50, adjust=False).mean()
    raw = _z(nvi - nvi_ma, 60) * 50
    return _clip_signal(raw)


def s180_vol_osc_confirm(ind):
    """180 | Volume Oscillator Breakout Confirmation"""
    volume = ind["volume"]
    fast = volume.ewm(span=5, adjust=False).mean()
    slow = volume.ewm(span=20, adjust=False).mean()
    osc = (fast - slow) / slow.replace(0, np.nan) * 100
    z = _z(osc, 40)
    raw = z * ind["trend_score"] * 50
    return _clip_signal(raw)


def s181_gap_volume_filter(ind):
    """181 | Tokyo Opening Gap Volume Filter"""
    gap = (ind["open"] - ind["close"].shift(1)) / ind["close"].shift(1) * 100
    vol_rel = ind["vol_rel"]
    # Gap + high volume = continuation; gap + low volume = fade
    gap_z = _z(gap, 40)
    raw = gap_z * (vol_rel > 1.5).astype(float) * 40 - gap_z * (vol_rel < 0.7).astype(float) * 30
    return _clip_signal(raw)


def s182_asi(ind):
    """182 | Accumulation Swing Index (ASI)"""
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    # Simplified Swing Index
    c1 = close.shift(1)
    k = pd.concat([high - c1, (low - c1).abs()], axis=1).max(axis=1)
    er = (close - c1) / k.replace(0, np.nan)
    si = 50 * er * (close - c1 + 0.5 * (close - open_p) + 0.25 * (c1 - open_p.shift(1))) / k.replace(0, np.nan)
    asi = si.cumsum()
    z = _z(asi.diff(10), 40)
    raw = z * 50
    return _clip_signal(raw)


def s183_vpt(ind):
    """183 | Volume Price Trend (VPT) System"""
    close = ind["close"]
    volume = ind["volume"]
    vpt = (volume * close.pct_change()).cumsum()
    vpt_ma = vpt.ewm(span=20, adjust=False).mean()
    z = _z(vpt - vpt_ma, 40)
    raw = z * 55
    return _clip_signal(raw)


def s184_vw_rsi(ind):
    """184 | Volume Weighted RSI"""
    close = ind["close"]
    volume = ind["volume"]
    ret = close.diff()
    vw_gain = (ret.clip(lower=0) * volume).rolling(14).sum()
    vw_loss = ((-ret.clip(upper=0)) * volume).rolling(14).sum()
    vw_rs = vw_gain / vw_loss.replace(0, np.nan)
    vw_rsi = 100 - 100 / (1 + vw_rs)
    raw = (50 - vw_rsi) / 50 * -60
    return _clip_signal(raw)


def s185_vol_profile_migration(ind):
    """185 | Intraday Volume Profile Value Area Migration"""
    close = ind["close"]
    volume = ind["volume"]
    vwap_5 = (close * volume).rolling(5).sum() / volume.rolling(5).sum().replace(0, np.nan)
    vwap_20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    migration = (vwap_5 - vwap_20) / ind["atr14"].replace(0, np.nan)
    z = _z(migration, 40)
    raw = z * 55
    return _clip_signal(raw)


def s186_buy_sell_heatmap(ind):
    """186 | Buy/Sell Imbalance Heatmap Strategy"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    rng = (high - low).replace(0, np.nan)
    buy_pct = (close - low) / rng
    imbalance = buy_pct * 2 - 1  # -1 to +1
    vol_weight = volume / volume.rolling(20).mean().replace(0, np.nan)
    weighted = imbalance * vol_weight
    z = _z(weighted.rolling(5).mean(), 20)
    raw = z * 55
    return _clip_signal(raw)


def s187_cumulative_tick(ind):
    """187 | Intraday Cumulative Tick Index Strategy (proxy)"""
    ret = ind["ret_1"]
    volume = ind["volume"]
    tick_proxy = np.sign(ret) * volume
    cum_tick = tick_proxy.rolling(10).sum()
    z = _z(cum_tick, 20)
    raw = z * 50
    return _clip_signal(raw)


def s188_vol_at_price_sr(ind):
    """188 | Volume-at-Price Support/Resistance"""
    close = ind["close"]
    volume = ind["volume"]
    # Support: high volume below current price; Resistance: high volume above
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    dist = close - vwap
    atr = ind["atr14"]
    dist_norm = dist / atr.replace(0, np.nan)
    # Near VWAP = support/resistance interaction
    near_vwap = (dist_norm.abs() < 0.5).astype(float)
    breakout = (dist_norm > 1.5).astype(float) - (dist_norm < -1.5).astype(float)
    raw = breakout * 45 + near_vwap * ind["trend_score"] * 25
    return _clip_signal(raw)


def s189_hk_connect_flow(ind):
    """189 | Hong Kong Connect Flow Signal (vol proxy)"""
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean()
    vol_surge = (volume > 2 * vol_ma).astype(float)
    direction = np.sign(ind["ret_1"])
    raw = vol_surge * direction * 45 + (1 - vol_surge) * ind["trend_score"] * 25
    return _clip_signal(raw)


def s190_bid_ask_regime(ind):
    """190 | Bid-Ask Spread Regime Strategy (range proxy)"""
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    spread_proxy = (high - low) / close * 10000  # bps
    z = _z(spread_proxy, 60)
    # Wide spread = illiquid (cautious); narrow = liquid (trend follow)
    raw = -z * 25 + ind["trend_score"] * (1 - z.clip(0, 2) / 3) * 40
    return _clip_signal(raw)


def s191_pc_oi_imbalance(ind):
    """191 | Put/Call Open Interest Imbalance (vol proxy)"""
    vol = ind["vol_20"]
    vol_z = _z(vol, 120)
    # High implied fear (high vol) = contrarian buy
    raw = -vol_z * 40 + ind["mom_score"] * 25
    return _clip_signal(raw)


def s192_liquidity_concentration(ind):
    """192 | Dubai Liquidity Concentration Index"""
    volume = ind["volume"]
    vol_std = volume.rolling(20).std()
    vol_mean = volume.rolling(20).mean().replace(0, np.nan)
    concentration = vol_std / vol_mean  # CV of volume
    z = _z(concentration, 60)
    raw = -z * 30 + ind["trend_score"] * 35
    return _clip_signal(raw)


def s193_mm_inventory(ind):
    """193 | Market Maker Inventory Model"""
    close = ind["close"]
    volume = ind["volume"]
    ret = ind["ret_1"]
    # Inventory builds: large volume with small price change
    impact = ret.abs() / (volume / volume.rolling(20).mean()).replace(0, np.nan)
    z = _z(impact, 40)
    # Low impact = inventory absorption; high impact = normal trading
    raw = -z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s194_vol_adj_bb(ind):
    """194 | Volume-Adjusted Bollinger Band Width"""
    bb_width = ind["bb_width"]
    vol_rel = ind["vol_rel"]
    adj_width = bb_width * vol_rel
    z = _z(adj_width, 60)
    squeeze = (z < -1.0).astype(float)
    expand = (z > 1.5).astype(float)
    raw = squeeze * 40 + expand * ind["trend_score"] * 35
    return _clip_signal(raw)


def s195_equity_pc_ratio(ind):
    """195 | Equity Put/Call Volume Ratio Extremes (vol proxy)"""
    vol_z = _z(ind["vol_20"], 60)
    extreme_fear = (vol_z > 2.0).astype(float) * (vol_z - 2.0)
    extreme_greed = (vol_z < -1.5).astype(float) * (-1.5 - vol_z)
    raw = extreme_fear * 40 - extreme_greed * 40 + ind["mom_score"] * 20
    return _clip_signal(raw)


def s196_turnover_velocity(ind):
    """196 | Singapore STI Turnover Velocity"""
    volume = ind["volume"]
    close = ind["close"]
    turnover = volume * close
    velocity = turnover.pct_change(5)
    z = _z(velocity, 40)
    raw = z * ind["trend_score"] * 50
    return _clip_signal(raw)


def s197_max_pain_gamma(ind):
    """197 | Options Max Pain and Gamma Exposure (proxy)"""
    close = ind["close"]
    sma50 = ind["sma_50"]
    # Max pain proxy: price gravitates toward SMA50
    dist = (close - sma50) / ind["atr14"].replace(0, np.nan)
    z = _z(dist, 40)
    raw = -z * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s198_vol_spike_reversion(ind):
    """198 | Volume Spike Reversion Trading"""
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean()
    spike = (volume > 3 * vol_ma).astype(float)
    ret = ind["ret_1"]
    # Fade the spike: high volume exhaustion
    raw = -spike * np.sign(ret) * 50 + (1 - spike) * ind["trend_score"] * 30
    return _clip_signal(raw, smooth=5)


def s199_fx_session_vol(ind):
    """199 | London FX Session Volume Dynamics (daily proxy)"""
    volume = ind["volume"]
    vol_z = _z(volume, 20)
    mom = ind["mom_5"]
    raw = vol_z * np.sign(mom) * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s200_vpin_regime(ind):
    """200 | Order Flow Toxicity (VPIN) Regime Filter"""
    ret = ind["ret_1"]
    volume = ind["volume"]
    toxicity = ret.abs() * volume
    total = volume
    vpin = toxicity.rolling(20).sum() / total.rolling(20).sum().replace(0, np.nan)
    vpin_z = _z(vpin, 60)
    # High toxicity = reduced exposure; Low toxicity = full exposure
    raw = ind["trend_score"] * (1 - vpin_z.clip(0, 2) / 3) * 55
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    151: ("OBV Divergence", s151_obv_divergence),
    152: ("VWAP Institutional", s152_vwap_institutional),
    153: ("CVD Strategy", s153_cvd_strategy),
    154: ("Volume POC", s154_volume_poc),
    155: ("Market Profile TPO", s155_market_profile_tpo),
    156: ("Wyckoff VSA", s156_wyckoff_vsa),
    157: ("TWAP Deviation", s157_twap_deviation),
    158: ("Volume Climax", s158_volume_climax),
    159: ("AD Institutional", s159_ad_institutional),
    160: ("LOB Imbalance", s160_lob_imbalance),
    161: ("VPIN Informed Trading", s161_vpin),
    162: ("Opening Auction Vol", s162_opening_auction),
    163: ("MFI Divergence", s163_mfi_divergence),
    164: ("Tick Vol Momentum", s164_tick_vol_momentum),
    165: ("Force Index Elder", s165_force_index),
    166: ("Vol Spread Divergence", s166_vol_spread_divergence),
    167: ("RVOL", s167_rvol),
    168: ("Klinger KVO", s168_klinger),
    169: ("Dark Pool Detection", s169_dark_pool),
    170: ("CMF Persistence", s170_cmf_persistence),
    171: ("VROC Breakout", s171_vroc_breakout),
    172: ("Ease of Movement", s172_ease_of_movement),
    173: ("Net Vol Pressure", s173_net_vol_pressure),
    174: ("TRIN Reversal", s174_trin_reversal),
    175: ("VZO", s175_vzo),
    176: ("Footprint Delta", s176_footprint_delta),
    177: ("Weis Wave", s177_weis_wave),
    178: ("Vol Weighted Mom", s178_vwm),
    179: ("NVI Smart Money", s179_nvi_smart_money),
    180: ("Vol Osc Confirm", s180_vol_osc_confirm),
    181: ("Gap Volume Filter", s181_gap_volume_filter),
    182: ("Accum Swing Index", s182_asi),
    183: ("Volume Price Trend", s183_vpt),
    184: ("Vol Weighted RSI", s184_vw_rsi),
    185: ("Vol Profile Migration", s185_vol_profile_migration),
    186: ("Buy/Sell Heatmap", s186_buy_sell_heatmap),
    187: ("Cumulative Tick", s187_cumulative_tick),
    188: ("Vol at Price SR", s188_vol_at_price_sr),
    189: ("HK Connect Flow", s189_hk_connect_flow),
    190: ("Bid-Ask Regime", s190_bid_ask_regime),
    191: ("PC OI Imbalance", s191_pc_oi_imbalance),
    192: ("Liquidity Concentration", s192_liquidity_concentration),
    193: ("MM Inventory", s193_mm_inventory),
    194: ("Vol Adj BB Width", s194_vol_adj_bb),
    195: ("Equity PC Ratio", s195_equity_pc_ratio),
    196: ("Turnover Velocity", s196_turnover_velocity),
    197: ("Max Pain Gamma", s197_max_pain_gamma),
    198: ("Vol Spike Reversion", s198_vol_spike_reversion),
    199: ("FX Session Vol", s199_fx_session_vol),
    200: ("VPIN Regime Filter", s200_vpin_regime),
}
