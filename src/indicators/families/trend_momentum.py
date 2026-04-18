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
    """001 | Ehlers Instantaneous Trendline Filter
    Two-pole super-smoother Butterworth filter with ~2 bar group delay.
    Signal: crossover with slope persistence confirmation + flat zone detection.
    """
    close = ind["close"]
    atr = ind["atr14"]
    n = 20
    a = 2.0 / (n + 1)

    # -- Butterworth two-pole super-smoother filter --
    vals = close.values.astype(float)
    length = len(vals)
    it_arr = np.full(length, np.nan)
    it_arr[0] = vals[0]
    if length > 1:
        it_arr[1] = vals[1]
    for i in range(2, length):
        it_arr[i] = (
            (a - a * a / 4) * vals[i]
            + (a * a / 2) * vals[i - 1]
            - (a - 3 * a * a / 4) * vals[i - 2]
            + 2 * (1 - a) * it_arr[i - 1]
            - (1 - a) ** 2 * it_arr[i - 2]
        )
    it = pd.Series(it_arr, index=close.index)

    # -- Slope: 3-bar rate of change of the filter, normalized by ATR --
    it_diff = it - it.shift(3)
    slope_raw = it_diff / atr.replace(0, np.nan)
    slope = slope_raw.fillna(0.0)

    # -- Slope persistence: require slope same sign for 2 consecutive bars --
    slope_sign = np.sign(slope)
    slope_persistent = (
        (slope_sign == slope_sign.shift(1)) & (slope_sign != 0)
    ).astype(float)

    # -- Crossover detection: close crosses above/below the filter line --
    cross_above = _crossover(close, it)
    cross_below = _crossunder(close, it)

    # -- Directional state: builds conviction via confirmed crossovers --
    # +1 when close > IT with persistent positive slope; -1 mirror; 0 ranging
    position_above = (close > it).astype(float)
    position_below = (close < it).astype(float)

    bull_confirmed = position_above * slope_persistent * (slope > 0).astype(float)
    bear_confirmed = position_below * slope_persistent * (slope < 0).astype(float)

    # -- Flat zone: |slope| below threshold means ranging market --
    slope_abs = slope.abs()
    slope_median = slope_abs.rolling(60, min_periods=20).median().fillna(0.01)
    flat_threshold = 0.3 * slope_median
    is_flat = (slope_abs < flat_threshold).astype(float)

    # -- Conviction ramp: recent crossover adds initial burst, then slope drives --
    cross_impulse_bull = cross_above.rolling(5, min_periods=1).sum().clip(0, 1) * 0.3
    cross_impulse_bear = cross_below.rolling(5, min_periods=1).sum().clip(0, 1) * 0.3

    # -- Trend strength: normalized slope magnitude [0, 1] --
    slope_strength = (slope_abs / slope_abs.rolling(120, min_periods=20).quantile(0.95).replace(0, np.nan)).clip(0, 1).fillna(0)

    # -- Composite signal --
    bull_signal = bull_confirmed * (0.7 * slope_strength + cross_impulse_bull)
    bear_signal = bear_confirmed * (0.7 * slope_strength + cross_impulse_bear)
    raw = (bull_signal - bear_signal) * 100
    raw = raw * (1 - 0.8 * is_flat)  # dampen in flat zones

    return _clip_signal(raw, smooth=3)


def s002_kama_momentum(ind):
    """002 | Kaufman Adaptive Moving Average Momentum
    Self-adapting MA via Efficiency Ratio. Fast in trends, flat in noise.
    Signal: KAMA slope + ATR buffer distance + ER regime filter.
    """
    close = ind["close"]
    atr = ind["atr14"]
    er = ind["efficiency"]

    # -- KAMA construction with proper n=10 lookback --
    n = 10
    fast_sc = 2.0 / (2 + 1)   # 0.6667
    slow_sc = 2.0 / (30 + 1)  # 0.0645
    sc = (er.fillna(0) * (fast_sc - slow_sc) + slow_sc) ** 2

    vals = close.values.astype(float)
    sc_vals = sc.values.astype(float)
    length = len(vals)
    kama_arr = np.full(length, np.nan)
    # Seed at bar n with the close value
    if length > n:
        kama_arr[n] = vals[n]
    for i in range(n + 1, length):
        prev = kama_arr[i - 1]
        if np.isnan(prev):
            kama_arr[i] = vals[i]
        else:
            s = sc_vals[i] if not np.isnan(sc_vals[i]) else slow_sc ** 2
            kama_arr[i] = prev + s * (vals[i] - prev)
    kama = pd.Series(kama_arr, index=close.index)

    # -- KAMA slope: 5-bar rate of change, normalized by price level --
    kama_slope = (kama - kama.shift(5)) / kama.shift(5).replace(0, np.nan)
    slope_direction = np.sign(kama_slope).fillna(0)

    # -- ATR buffer: distance from KAMA in ATR units --
    # Long signal requires Close > KAMA + 0.5*ATR(14)
    # Short signal requires Close < KAMA - 0.5*ATR(14)
    atr_safe = atr.replace(0, np.nan).fillna(method="ffill").fillna(0.01)
    buffer = 0.5 * atr_safe
    dist_from_kama = close - kama
    dist_atr = dist_from_kama / atr_safe

    # -- Directional signals with buffer confirmation --
    long_zone = (dist_from_kama > buffer).astype(float)   # Close > KAMA + 0.5*ATR
    short_zone = (dist_from_kama < -buffer).astype(float)  # Close < KAMA - 0.5*ATR
    neutral_zone = 1 - long_zone - short_zone

    # -- Slope confirmation: must align with distance --
    long_confirmed = long_zone * (slope_direction > 0).astype(float)
    short_confirmed = short_zone * (slope_direction < 0).astype(float)

    # -- ER filter: ER > 0.3 means directional, below = choppy --
    # In choppy (ER < 0.3): suppress signal by 70%
    er_quality = er.fillna(0)
    er_gate = np.where(er_quality > 0.3, 1.0, 0.3)
    er_gate = pd.Series(er_gate, index=close.index)

    # -- Signal strength from distance magnitude --
    # Farther from KAMA = stronger conviction, capped at 3 ATR
    strength = dist_atr.abs().clip(0, 3) / 3.0

    # -- Momentum quality: slope magnitude relative to its own history --
    slope_mag = kama_slope.abs()
    slope_norm = (slope_mag / slope_mag.rolling(60, min_periods=20).quantile(0.90).replace(0, np.nan)).clip(0, 1).fillna(0)

    # -- Composite --
    raw = (long_confirmed - short_confirmed) * (0.5 * strength + 0.5 * slope_norm) * er_gate * 100
    # Add small signal in neutral zone based on slope
    neutral_signal = neutral_zone * slope_direction * slope_norm * er_gate * 25
    raw = raw + neutral_signal

    return _clip_signal(raw, smooth=3)


def s003_triple_hma_regime(ind):
    """003 | Triple Hull Moving Average Regime System
    Three HMA channels classify 5 regimes (+2 Full Bull to -2 Full Bear).
    Signal: regime transitions trigger entries; sizing scales with regime strength.
    """
    close = ind["close"]
    atr = ind["atr14"]

    # -- Three Hull Moving Averages at different timescales --
    hma_fast = _hma(close, 16)   # Fast trend (weeks)
    hma_med = _hma(close, 36)    # Medium trend (1-2 months)
    hma_slow = _hma(close, 64)   # Slow trend (quarter)

    # -- Regime classification: +2 Full Bull, +1 Early Bull, 0 Transition, etc --
    regime = np.sign(hma_fast - hma_med) + np.sign(hma_med - hma_slow)
    regime = pd.Series(regime, index=close.index).fillna(0)

    # -- Regime transitions: detect when regime improves or deteriorates --
    prev_regime = regime.shift(1).fillna(0)
    regime_change = regime - prev_regime  # +1 or +2 = bullish transition

    # -- Transition confirmation: regime must hold for 2 bars --
    regime_held = (regime == regime.shift(1)).astype(float)

    # -- HMA slope quality: fast HMA slope normalized by ATR --
    hma_fast_slope = (hma_fast - hma_fast.shift(3)) / atr.replace(0, np.nan)
    hma_med_slope = (hma_med - hma_med.shift(5)) / atr.replace(0, np.nan)
    slope_alignment = np.sign(hma_fast_slope) * np.sign(hma_med_slope)
    slopes_agree = (slope_alignment > 0).astype(float)

    # -- Spacing quality: wider spacing = stronger trend --
    fast_med_spread = (hma_fast - hma_med).abs() / atr.replace(0, np.nan)
    med_slow_spread = (hma_med - hma_slow).abs() / atr.replace(0, np.nan)
    spread_score = (fast_med_spread + med_slow_spread).clip(0, 6) / 6.0

    # -- Regime-based position sizing --
    # Full Bull (+2) or Full Bear (-2): full signal
    # Early Bull (+1) or Early Bear (-1): half signal
    # Transition (0): minimal signal
    regime_strength = regime.abs() / 2.0  # 0, 0.5, or 1.0
    regime_direction = np.sign(regime)

    # -- Transition impulse: boost signal on fresh regime entries --
    bullish_transition = (regime_change > 0).astype(float) * regime_change * 0.15
    bearish_transition = (regime_change < 0).astype(float) * regime_change.abs() * 0.15
    transition_impulse = bullish_transition - bearish_transition

    # -- Composite signal --
    base_signal = regime_direction * regime_strength  # [-1, +1]
    quality = 0.5 * slopes_agree + 0.3 * spread_score + 0.2 * regime_held
    raw = (base_signal * quality + transition_impulse) * 100

    return _clip_signal(raw, smooth=3)


def s004_ichimoku_extended(ind):
    """004 | Ichimoku Kinko Hyo Extended Cloud System
    Full 5-component Ichimoku with cloud thickness, expansion, Kumo twist,
    and multi-factor confluence scoring. Hosoda's complete system.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]

    # -- Five Ichimoku components computed from scratch --
    # Tenkan-sen (Conversion Line): 9-period midpoint
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2

    # Kijun-sen (Base Line): 26-period midpoint
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Senkou Span A (Leading Span A): midpoint of Tenkan/Kijun, displaced 26 forward
    # For signal purposes we use the current value (displacement is for plotting)
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): 52-period midpoint, displaced 26 forward
    senkou_b = (high.rolling(52, min_periods=26).max() + low.rolling(52, min_periods=26).min()) / 2

    # Chikou Span (Lagging Span): current close vs cloud 26 periods ago
    chikou_ref_cloud_top = pd.concat([senkou_a.shift(26), senkou_b.shift(26)], axis=1).max(axis=1)
    chikou_ref_cloud_bot = pd.concat([senkou_a.shift(26), senkou_b.shift(26)], axis=1).min(axis=1)

    # -- Cloud boundaries (current) --
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    # -- Cloud metrics --
    cloud_thickness = (cloud_top - cloud_bot).fillna(0)
    ct_norm = cloud_thickness / close.replace(0, np.nan) * 100  # as % of price
    cloud_delta = ct_norm.pct_change(5).clip(-2, 2)  # expansion/contraction rate

    cloud_is_green = (senkou_a > senkou_b).astype(float)  # 1 = bullish cloud
    cloud_is_red = (senkou_a < senkou_b).astype(float)

    # -- Kumo twist: Senkou A crosses Senkou B = regime change warning --
    kumo_twist_up = _crossover(senkou_a, senkou_b)
    kumo_twist_dn = _crossunder(senkou_a, senkou_b)
    kumo_twist = ((kumo_twist_up + kumo_twist_dn) > 0).astype(float)
    recent_twist = kumo_twist.rolling(5, min_periods=1).sum().clip(0, 1)

    # -- Confluence factors (each scored [-1, +1]) --

    # Factor 1: Price vs Cloud position
    above_cloud = (close > cloud_top).astype(float)
    below_cloud = (close < cloud_bot).astype(float)
    in_cloud = 1 - above_cloud - below_cloud
    price_cloud = above_cloud - below_cloud  # +1 above, -1 below, 0 inside

    # Factor 2: Tenkan-Kijun relationship
    tk_diff = (tenkan - kijun) / close.replace(0, np.nan) * 100
    tk_signal = tk_diff.clip(-2, 2) / 2  # normalized [-1, +1]

    # Factor 3: Chikou Span vs historical cloud
    chikou_above = (close > chikou_ref_cloud_top).astype(float)
    chikou_below = (close < chikou_ref_cloud_bot).astype(float)
    chikou_signal = chikou_above - chikou_below

    # Factor 4: Cloud color (green = bullish, red = bearish)
    cloud_color = cloud_is_green - cloud_is_red  # +1 green, -1 red

    # Factor 5: Cloud expanding (positive delta = strengthening trend)
    cloud_expanding = (cloud_delta > 0).astype(float) * cloud_delta.clip(0, 1)
    cloud_contracting = (cloud_delta < 0).astype(float) * cloud_delta.clip(-1, 0)
    expansion_signal = cloud_expanding + cloud_contracting

    # -- Weighted confluence score --
    confluence = (
        0.30 * price_cloud       # Price above/below cloud is dominant
        + 0.20 * tk_signal       # Tenkan-Kijun cross
        + 0.20 * chikou_signal   # Chikou confirmation
        + 0.15 * cloud_color     # Cloud color
        + 0.15 * expansion_signal  # Cloud expansion
    )

    # -- Count how many factors agree (agreement quality) --
    factor_signs = pd.concat([
        np.sign(price_cloud), np.sign(tk_signal),
        np.sign(chikou_signal), np.sign(cloud_color)
    ], axis=1)
    agreement = factor_signs.sum(axis=1).abs() / 4.0  # 0 to 1
    agreement_boost = 0.5 + 0.5 * agreement  # 0.5 to 1.0

    # -- Kumo twist penalty: reduce size by 50% near twist --
    twist_penalty = 1 - 0.5 * recent_twist

    # -- Final signal --
    raw = confluence * agreement_boost * twist_penalty * 100

    return _clip_signal(raw, smooth=3)


def s005_dax_momentum_oscillator(ind):
    """005 | DAX Ordnungssystem Momentum Oscillator
    Gaussian-weighted ROC with z-score normalization, threshold crossover,
    and momentum divergence detection. Frankfurt school precision.
    """
    close = ind["close"]
    high = ind["high"]

    n = 20
    sigma = n / 3.0  # optimal bandwidth per spec

    # -- Rate of change --
    roc = close.pct_change(n)

    # -- Gaussian kernel for smoothing --
    kernel_len = 2 * n + 1
    kernel_idx = np.arange(kernel_len)
    gauss_weights = np.exp(-(kernel_idx ** 2) / (2 * sigma * sigma))
    gauss_weights = gauss_weights / gauss_weights.sum()

    # -- Convolve ROC with Gaussian kernel (proper convolution, not rolling apply) --
    roc_vals = roc.fillna(0).values.astype(float)
    gauss_roc = np.convolve(roc_vals, gauss_weights, mode="same")
    gauss_roc_s = pd.Series(gauss_roc, index=close.index)
    # Mask the warmup period where convolution is unreliable
    gauss_roc_s.iloc[:kernel_len] = np.nan

    # -- Z-score normalization (rolling 60-bar) --
    daxmo = _z(gauss_roc_s, 60)

    # -- Threshold crossover detection --
    # Long: DAXMO crosses above +0.5 from below (momentum ignition)
    # Short: DAXMO crosses below -0.5 from above
    thresh_up = 0.5
    thresh_dn = -0.5
    cross_bull = _crossover(daxmo, pd.Series(thresh_up, index=close.index))
    cross_bear = _crossunder(daxmo, pd.Series(thresh_dn, index=close.index))

    # -- Sustain signal while in bullish/bearish territory --
    in_bull_zone = (daxmo > thresh_up).astype(float)
    in_bear_zone = (daxmo < thresh_dn).astype(float)
    in_neutral = 1 - in_bull_zone - in_bear_zone

    # -- Momentum divergence: price new high but DAXMO lower --
    price_high_20 = high.rolling(20).max()
    price_new_high = (high >= price_high_20).astype(float)
    daxmo_peak_20 = daxmo.rolling(20).max()
    daxmo_below_peak = (daxmo < daxmo_peak_20 * 0.8).astype(float)  # DAXMO < 80% of its peak
    bearish_divergence = (price_new_high * daxmo_below_peak * in_bull_zone)

    # Similarly: price new low but DAXMO higher
    price_low_20 = ind["low"].rolling(20).min()
    price_new_low = (ind["low"] <= price_low_20).astype(float)
    daxmo_trough_20 = daxmo.rolling(20).min()
    daxmo_above_trough = (daxmo > daxmo_trough_20 * 0.8).astype(float)
    bullish_divergence = (price_new_low * daxmo_above_trough * in_bear_zone)

    # -- Signal strength: DAXMO magnitude beyond threshold --
    bull_strength = ((daxmo - thresh_up) / 2.0).clip(0, 1)  # 0 at +0.5, 1 at +2.5
    bear_strength = ((thresh_dn - daxmo) / 2.0).clip(0, 1)

    # -- Cross impulse for fresh entries --
    cross_bull_impulse = cross_bull.rolling(3, min_periods=1).sum().clip(0, 1) * 0.2
    cross_bear_impulse = cross_bear.rolling(3, min_periods=1).sum().clip(0, 1) * 0.2

    # -- Composite signal --
    bull_signal = in_bull_zone * (bull_strength + cross_bull_impulse)
    bear_signal = in_bear_zone * (bear_strength + cross_bear_impulse)

    # Divergence reduces conviction by 50% per spec
    bull_signal = bull_signal * (1 - 0.5 * bearish_divergence)
    bear_signal = bear_signal * (1 - 0.5 * bullish_divergence)

    # Neutral zone: weak signal from DAXMO direction
    neutral_signal = in_neutral * daxmo.clip(-1, 1) * 0.15

    raw = (bull_signal - bear_signal + neutral_signal) * 100

    return _clip_signal(raw, smooth=3)


def s006_carry_momentum_filter(ind):
    """006 | Swiss Precision Carry-Momentum Filter
    Cross-sectional carry-momentum composite adapted for single-asset OHLCV.
    Carry proxy from yield curve slope (vol term structure), risk-adjusted momentum,
    and inverse volatility weighting. Zurich wealth management approach.
    """
    close = ind["close"]
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]

    # -- Carry proxy: vol term structure slope --
    # When short-term vol < long-term vol (contango), it's a "carry" environment
    # When short-term vol > long-term vol (backwardation), carry is negative
    vol_20_safe = vol_20.replace(0, np.nan).fillna(method="ffill").fillna(0.01)
    vol_60_safe = vol_60.replace(0, np.nan).fillna(method="ffill").fillna(0.01)
    vol_term_slope = (vol_60_safe - vol_20_safe) / vol_60_safe  # positive = contango
    carry_score = _z(vol_term_slope, 60).clip(-3, 3) / 3  # z-scored and normalized

    # -- Risk-adjusted momentum: 13-week return / 13-week vol --
    ret_65 = close.pct_change(65)  # ~13 weeks
    vol_65 = ind["ret_1"].rolling(65, min_periods=20).std().replace(0, np.nan)
    risk_adj_mom = (ret_65 / (vol_65 * np.sqrt(65))).clip(-3, 3) / 3

    # -- Volatility rank: penalize high-vol assets (vol drag) --
    vol_rank = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    vol_penalty = -(vol_rank - 0.5) * 2  # -1 to +1: low vol = positive, high vol = negative

    # -- Composite per Swiss specification: 40% carry, 40% momentum, 20% vol --
    composite = 0.40 * carry_score + 0.40 * risk_adj_mom + 0.20 * vol_penalty

    # -- Turnover control: smooth the composite to simulate weekly rebalance --
    composite_smooth = composite.ewm(span=5, adjust=False).mean()

    # -- Cross-sectional threshold: only trade when conviction is strong --
    # Long above 70th percentile equivalent, Short below 30th
    comp_pct = composite_smooth.rolling(120, min_periods=30).rank(pct=True).fillna(0.5)
    long_zone = (comp_pct > 0.70).astype(float)
    short_zone = (comp_pct < 0.30).astype(float)
    neutral = 1 - long_zone - short_zone

    # -- Signal with conviction scaling --
    long_strength = (comp_pct - 0.70).clip(0, 0.30) / 0.30  # 0 at 70th, 1 at 100th
    short_strength = (0.30 - comp_pct).clip(0, 0.30) / 0.30

    raw = (long_zone * long_strength - short_zone * short_strength) * 85
    raw = raw + neutral * composite_smooth * 20  # weak signal in neutral

    return _clip_signal(raw, smooth=5)  # weekly smoothing


def s007_liquidity_pulse(ind):
    """007 | Shanghai Liquidity Pulse Detector
    Volume-price elasticity, Amihud illiquidity, and Order Flow Imbalance
    relative to VWAP. Detects institutional accumulation/distribution surges.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    ret = ind["ret_1"]

    # -- Volume Pulse: relative volume surge vs 20-bar EMA --
    vol_ema20 = volume.ewm(span=20, adjust=False).mean().replace(0, np.nan)
    volume_pulse = (volume / vol_ema20 - 1).fillna(0)

    # -- Amihud Price Impact: |return| / dollar volume (illiquidity measure) --
    vol_median = volume.rolling(20, min_periods=5).median().replace(0, np.nan)
    rel_volume = (volume / vol_median).replace(0, np.nan)
    price_impact = ret.abs() / rel_volume  # higher = more illiquid
    # Invert and z-score: low impact (high liquidity) = positive
    price_impact_z = -_z(price_impact, 60)

    # -- Volume-Price Elasticity: regression of returns on log volume (10-bar) --
    log_vol = np.log1p(volume).fillna(0)
    ret_vals = ret.fillna(0).values
    log_vol_vals = log_vol.values
    n_bars = len(ret_vals)
    elasticity = np.full(n_bars, np.nan)
    window = 10
    for i in range(window, n_bars):
        y = ret_vals[i - window:i]
        x = log_vol_vals[i - window:i]
        x_dm = x - x.mean()
        denom = np.dot(x_dm, x_dm)
        if denom > 1e-12:
            elasticity[i] = np.dot(x_dm, y - y.mean()) / denom
    elasticity_s = pd.Series(elasticity, index=close.index)
    elasticity_z = _z(elasticity_s, 60)

    # -- Order Flow Imbalance (OFI) relative to VWAP --
    # VWAP proxy from typical price * volume
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(20, min_periods=5).sum() / volume.rolling(20, min_periods=5).sum()
    # OFI: signed volume relative to VWAP
    sign_vs_vwap = np.sign(close - vwap)
    ofi_raw = (sign_vs_vwap * volume).rolling(20, min_periods=5).sum()
    ofi_norm = ofi_raw / volume.rolling(20, min_periods=5).sum().replace(0, np.nan)
    ofi_z = _z(ofi_norm, 60)

    # -- Composite Liquidity Score (spec weights) --
    liq_score = (
        0.30 * _z(volume_pulse, 60)
        + 0.25 * price_impact_z
        + 0.25 * elasticity_z
        + 0.20 * ofi_z
    )

    # -- Threshold signals --
    # Long: liq_score > +1.5 sigma AND volume_pulse > 2.0 (institutional buying)
    # Short: liq_score < -1.5 sigma AND volume_pulse > 1.5 (panic selling)
    strong_vol = (volume_pulse > 2.0).astype(float)
    panic_vol = (volume_pulse > 1.5).astype(float)

    bull_pulse = (liq_score > 1.5).astype(float) * strong_vol
    bear_pulse = (liq_score < -1.5).astype(float) * panic_vol

    # -- Sustain signal with decay --
    bull_sustained = bull_pulse.rolling(5, min_periods=1).max() * 0.7
    bear_sustained = bear_pulse.rolling(5, min_periods=1).max() * 0.7
    # Fresh pulse gets full weight
    bull_signal = np.maximum(bull_pulse, bull_sustained)
    bear_signal = np.maximum(bear_pulse, bear_sustained)

    # -- Moderate zone: directional bias from OFI even without pulse --
    moderate = (1 - bull_signal - bear_signal).clip(0, 1)
    moderate_signal = moderate * liq_score.clip(-3, 3) * 0.70

    # -- Strength scaling from liquidity score magnitude --
    strength = liq_score.abs().clip(0.5, 3) / 3  # floor at 0.5 for pulse signals

    raw = (bull_signal * 1.0 - bear_signal * 1.0 + moderate_signal) * 100

    return _clip_signal(raw, smooth=3)


def s008_breakout_range(ind):
    """008 | London FX Breakout Range System (daily OHLCV proxy)
    Tight consolidation detection via range compression, then breakout with
    buffer confirmation. Inspired by Asian range / London breakout concept
    adapted for daily data via Donchian ranges and ATR filtering.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    atr = ind["atr14"]

    # -- Consolidation range: 5-day range (proxy for "Asian session" on daily) --
    range_5d = high.rolling(5).max() - low.rolling(5).min()
    range_20d_median = range_5d.rolling(20, min_periods=5).median()

    # -- Tight consolidation filter: range must be < 1.5x median --
    # Coiled spring effect: tighter consolidation = better breakout
    range_ratio = range_5d / range_20d_median.replace(0, np.nan)
    is_tight = (range_ratio < 1.5).astype(float)
    tightness_quality = (1.5 - range_ratio).clip(0, 1)  # tighter = higher quality

    # -- ATR filter: must have sufficient volatility to be worth trading --
    atr_pct = (atr / close * 100).replace(0, np.nan)
    atr_sufficient = (atr_pct > 0.5).astype(float)  # at least 0.5% daily ATR

    # -- Breakout levels with buffer (10% of range above/below) --
    hi_5d = high.rolling(5).max()
    lo_5d = low.rolling(5).min()
    range_buf = range_5d * 0.10
    breakout_up_level = hi_5d.shift(1) + range_buf.shift(1)
    breakout_dn_level = lo_5d.shift(1) - range_buf.shift(1)

    # -- Breakout detection --
    breakout_long = (close > breakout_up_level).astype(float) * is_tight * atr_sufficient
    breakout_short = (close < breakout_dn_level).astype(float) * is_tight * atr_sufficient

    # -- Sustain: breakout signal holds while price stays above/below level --
    still_above = (close > hi_5d.shift(1)).astype(float)
    still_below = (close < lo_5d.shift(1)).astype(float)

    # -- Breakout strength: how far past the level (in ATR units) --
    dist_up = ((close - breakout_up_level) / atr.replace(0, np.nan)).clip(0, 3) / 3
    dist_dn = ((breakout_dn_level - close) / atr.replace(0, np.nan)).clip(0, 3) / 3

    # -- Fresh breakout impulse --
    breakout_fresh_long = _crossover(close, breakout_up_level)
    breakout_fresh_short = _crossunder(close, breakout_dn_level)

    # Impulse decays over 10 bars
    impulse_long = breakout_fresh_long.rolling(10, min_periods=1).sum().clip(0, 2) / 2
    impulse_short = breakout_fresh_short.rolling(10, min_periods=1).sum().clip(0, 2) / 2

    # -- Composite: fresh breakout + sustained position + distance --
    raw_long = (
        0.35 * breakout_long
        + 0.25 * still_above * tightness_quality
        + 0.25 * dist_up
        + 0.15 * impulse_long
    )
    raw_short = (
        0.35 * breakout_short
        + 0.25 * still_below * tightness_quality
        + 0.25 * dist_dn
        + 0.15 * impulse_short
    )

    # -- Also provide Donchian 20-day breakout as baseline signal --
    donch_hi = ind["donch_high"]
    donch_lo = ind["donch_low"]
    donch_break_up = (close > donch_hi.shift(1)).astype(float) * 0.5
    donch_break_dn = (close < donch_lo.shift(1)).astype(float) * 0.5

    raw = ((raw_long + donch_break_up) - (raw_short + donch_break_dn)) * 80

    return _clip_signal(raw, smooth=3)


def s009_hurst_adaptive(ind):
    """009 | Hurst Exponent Adaptive Trend/MR Classifier
    Uses Hurst exponent for regime classification.
    H > 0.55 activates momentum overlay; H < 0.45 activates mean-reversion.
    Soft blending via sigmoid at boundaries. Efficiency ratio as persistence proxy.
    """
    close = ind["close"]
    rsi = ind["rsi"]
    mom = ind["mom_score"]

    # -- Hurst proxy: combine base hurst with efficiency ratio --
    # Base hurst has many NaNs, so blend with ER as persistence proxy
    hurst_raw = ind["hurst"].copy()
    er = ind["efficiency"].fillna(0.5)
    # ER > 0.5 suggests trending (H > 0.5), ER < 0.3 suggests noise
    er_hurst_proxy = 0.35 + 0.30 * er  # maps ER [0,1] -> H [0.35, 0.65]
    # Blend: use actual hurst where available, proxy where NaN
    hurst_blended = hurst_raw.fillna(er_hurst_proxy)
    hurst_smooth = hurst_blended.ewm(span=10, min_periods=1, adjust=False).mean()

    # -- Soft regime blending via sigmoid --
    trend_weight = ((hurst_smooth - 0.50) * 10).clip(-3, 3)
    trend_bias = 1.0 / (1.0 + np.exp(-trend_weight))
    trend_bias = pd.Series(trend_bias, index=close.index).fillna(0.5)
    mr_bias = 1.0 - trend_bias

    # -- Momentum overlay for trending regime --
    macd_signal = ind["trend_macd"].fillna(0)
    trend_signal = (0.60 * mom + 0.40 * macd_signal).fillna(0)

    # -- Mean-reversion overlay for MR regime --
    rsi_mr = ((50 - rsi) / 50).fillna(0)
    bb_mr = ((0.5 - ind["bb_pctb"]) * 2).fillna(0)
    mr_signal = 0.55 * rsi_mr + 0.45 * bb_mr

    # -- Direct blending --
    # Use dominant regime signal at full strength, minor regime at reduced
    dominant_is_trend = (trend_bias > 0.55).astype(float)
    dominant_is_mr = (trend_bias < 0.45).astype(float)
    neutral = 1 - dominant_is_trend - dominant_is_mr

    raw = (
        dominant_is_trend * trend_signal * 90
        + dominant_is_mr * mr_signal * 90
        + neutral * (trend_signal * 50 + mr_signal * 40)
    )

    # -- Regime transition smoothing --
    raw_smooth = raw.ewm(span=5, adjust=False).mean()

    return _clip_signal(raw_smooth, smooth=3)


def s010_kalman_trend(ind):
    """010 | Kalman Filter State-Space Trend Extraction
    Full 2D state-space model [level, drift] with proper Riccati equations,
    adaptive observation noise from realized vol, and drift significance ratio.
    Signal: drift_t / sqrt(P_22_t) with zone-based position sizing.
    """
    close = ind["close"]
    vol = ind["vol_20"].fillna(0.01)
    vals = close.values.astype(float)
    n = len(vals)

    # -- State: [mu_t, drift_t], Transition: F = [[1,1],[0,1]] --
    mu = np.zeros(n)
    drift = np.zeros(n)

    # -- Full 2x2 covariance matrix P --
    p11 = np.zeros(n)  # var(mu)
    p12 = np.zeros(n)  # cov(mu, drift)
    p22 = np.zeros(n)  # var(drift)

    # -- Process noise Q: [[q1, 0], [0, q2]] --
    q1 = 0.001    # level process noise
    q2 = 1e-5     # drift process noise

    # -- Initialize --
    mu[0] = vals[0]
    drift[0] = 0.0
    p11[0] = 1.0
    p12[0] = 0.0
    p22[0] = 0.001

    for i in range(1, n):
        # -- Predict step --
        mu_pred = mu[i - 1] + drift[i - 1]
        d_pred = drift[i - 1]

        # Predicted covariance: P_pred = F * P * F' + Q
        p11_pred = p11[i - 1] + 2 * p12[i - 1] + p22[i - 1] + q1
        p12_pred = p12[i - 1] + p22[i - 1]
        p22_pred = p22[i - 1] + q2

        # -- Observation noise: R = c * realized_variance * 252 --
        v = vol.iloc[i]
        if np.isnan(v) or v <= 0:
            v = 0.01
        r = max(v ** 2 * 252, 0.01)  # annualized variance

        # -- Innovation --
        innov = vals[i] - mu_pred
        s = p11_pred + r  # innovation variance

        # -- Kalman gain: K = P_pred * H' / S where H = [1, 0] --
        k1 = p11_pred / s
        k2 = p12_pred / s

        # -- Update step --
        mu[i] = mu_pred + k1 * innov
        drift[i] = d_pred + k2 * innov

        # -- Updated covariance: P = (I - K*H) * P_pred --
        p11[i] = (1 - k1) * p11_pred
        p12[i] = (1 - k1) * p12_pred
        p22[i] = p22_pred - k2 * p12_pred

    drift_s = pd.Series(drift, index=close.index)
    drift_var = pd.Series(p22, index=close.index).replace(0, np.nan)

    # -- Signal Ratio: drift significance = drift / sqrt(var(drift)) --
    sig_ratio = drift_s / np.sqrt(drift_var)

    # -- Zone-based position sizing per spec --
    # Long: sig_ratio > +1.5 (significantly positive drift)
    # Short: sig_ratio < -1.5
    # Flat: |sig_ratio| < 0.5
    # Size proportional to |sig_ratio| capped at 3.0

    long_zone = (sig_ratio > 1.5).astype(float)
    short_zone = (sig_ratio < -1.5).astype(float)
    flat_zone = (sig_ratio.abs() < 0.5).astype(float)
    transition_zone = 1 - long_zone - short_zone - flat_zone

    # Position sizing: linear scale from threshold to cap
    long_strength = ((sig_ratio - 1.5) / 1.5).clip(0, 1)  # 0 at 1.5, 1 at 3.0
    short_strength = ((-1.5 - sig_ratio) / 1.5).clip(0, 1)

    # Transition zone: weak signal proportional to sig_ratio
    trans_signal = transition_zone * sig_ratio.clip(-1.5, 1.5) / 1.5 * 0.3

    # -- Drift acceleration: change in drift significance --
    sig_ratio_accel = sig_ratio - sig_ratio.shift(5)
    accel_boost = sig_ratio_accel.clip(-1, 1) * 0.15

    # -- Stop signal: when sig_ratio crosses zero, flatten --
    cross_zero_up = _crossover(sig_ratio, pd.Series(0.0, index=close.index))
    cross_zero_dn = _crossunder(sig_ratio, pd.Series(0.0, index=close.index))
    zero_cross_recent = (cross_zero_up + cross_zero_dn).rolling(3, min_periods=1).sum().clip(0, 1)
    zero_cross_penalty = 1 - 0.5 * zero_cross_recent

    # -- Composite --
    raw = (
        long_zone * (0.85 * long_strength + accel_boost)
        - short_zone * (0.85 * short_strength - accel_boost)
        + trans_signal
    ) * zero_cross_penalty * 100

    return _clip_signal(raw, smooth=3)


def s011_supertrend(ind):
    """011 | Supertrend ATR Band System (Sezer)
    ATR-based trailing stop defining trend direction.
    HL_median +/- multiplier*ATR with ratchet mechanism.
    period=10, multiplier=3.0. Signal on trend flips + distance scaling.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    tr = ind["tr"]

    # -- Compute ATR(10) per spec (base.py only provides ATR(14)) --
    atr10 = tr.rolling(10, min_periods=5).mean()
    mult = 3.0
    hl2 = (high + low) / 2

    ub = hl2 + mult * atr10
    lb = hl2 - mult * atr10

    n = len(close)
    st = np.zeros(n)
    direction = np.ones(n)
    ub_v = ub.values.astype(float).copy()
    lb_v = lb.values.astype(float).copy()
    c = close.values.astype(float)

    # -- Supertrend loop with ratchet mechanism --
    for i in range(1, n):
        # Ratchet: lower band can only go UP in bullish, upper band only DOWN in bearish
        if not np.isnan(lb_v[i]) and not np.isnan(lb_v[i - 1]):
            if c[i - 1] > lb_v[i - 1]:
                lb_v[i] = max(lb_v[i], lb_v[i - 1])
        if not np.isnan(ub_v[i]) and not np.isnan(ub_v[i - 1]):
            if c[i - 1] < ub_v[i - 1]:
                ub_v[i] = min(ub_v[i], ub_v[i - 1])

        # Direction flip logic
        if direction[i - 1] == 1:  # bullish
            if c[i] < lb_v[i]:
                direction[i] = -1
                st[i] = ub_v[i]
            else:
                direction[i] = 1
                st[i] = lb_v[i]
        else:  # bearish
            if c[i] > ub_v[i]:
                direction[i] = 1
                st[i] = lb_v[i]
            else:
                direction[i] = -1
                st[i] = ub_v[i]

    dir_s = pd.Series(direction, index=close.index)
    st_s = pd.Series(st, index=close.index)

    # -- Detect trend flips for entry signals --
    flip_bull = (dir_s == 1) & (dir_s.shift(1) == -1)
    flip_bear = (dir_s == -1) & (dir_s.shift(1) == 1)

    # -- Distance from supertrend (normalized by ATR) for conviction --
    atr_safe = atr10.replace(0, np.nan)
    dist_norm = ((close - st_s) / atr_safe).clip(-4, 4)

    # -- Persistence: how many bars in current direction --
    dir_change = (dir_s != dir_s.shift(1)).astype(int)
    group = dir_change.cumsum()
    persist = group.groupby(group).cumcount() + 1
    persist_factor = (persist.clip(1, 20) / 20).clip(0.3, 1.0)

    # -- Build signal: base direction + flip spike + distance scaling --
    base = dir_s * 45  # base direction
    flip_spike = flip_bull.astype(float) * 25 - flip_bear.astype(float) * 25
    dist_component = dist_norm * 10 * persist_factor

    raw = base + flip_spike + dist_component

    return _clip_signal(raw, smooth=3)


def s012_donchian_turtle(ind):
    """012 | Donchian Channel Turtle System (Modified)
    Classic breakout system: Enter on 20-day high/low, Exit on 10-day.
    N-based position sizing with pyramiding up to 4 units.
    System 1 filter: skip if previous same-direction trade was profitable.
    Asymmetric entry(20) / exit(10) channels.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    tr = ind["tr"]

    # -- Entry channel (20-day) from base.py --
    entry_hi = ind["donch_high"]  # 20-day highest high
    entry_lo = ind["donch_low"]   # 20-day lowest low

    # -- Exit channel (10-day) --
    exit_hi = high.rolling(10, min_periods=5).max()
    exit_lo = low.rolling(10, min_periods=5).min()

    # -- ATR(20) for N-based sizing --
    atr20 = tr.rolling(20, min_periods=10).mean()
    atr_safe = atr20.replace(0, np.nan)

    # -- Breakout signals: new 20-day high/low --
    breakout_long = (close > entry_hi.shift(1)).astype(float)
    breakout_short = (close < entry_lo.shift(1)).astype(float)

    # -- Exit signals: 10-day channel exit --
    exit_long = (close < exit_lo.shift(1)).astype(float)
    exit_short = (close > exit_hi.shift(1)).astype(float)

    # -- Penetration depth: how far above/below channel --
    pen_long = ((close - entry_hi.shift(1)) / atr_safe).clip(0, 4)
    pen_short = ((entry_lo.shift(1) - close) / atr_safe).clip(0, 4)

    # -- Pyramiding: count consecutive breakout bars (up to 4 units, add every 0.5N) --
    # Sustained breakout = stronger signal (simulates adding units)
    long_streak = breakout_long.rolling(10, min_periods=1).sum().clip(0, 4)
    short_streak = breakout_short.rolling(10, min_periods=1).sum().clip(0, 4)
    pyramid_long = long_streak / 4  # 0 to 1
    pyramid_short = short_streak / 4

    # -- Trend confirmation via ADX --
    adx = ind["adx"].fillna(15)
    trend_confirm = ((adx - 20) / 20).clip(0, 1)  # 0 at ADX=20, 1 at ADX=40

    # -- System 1 filter proxy: suppress breakout if recent same-direction trade --
    # Use momentum to determine if previous breakout was profitable
    ret_20 = close.pct_change(20).fillna(0)
    s1_suppress_long = ((ret_20 > 0.05) & breakout_long.astype(bool)).astype(float) * 0.4
    s1_suppress_short = ((ret_20 < -0.05) & breakout_short.astype(bool)).astype(float) * 0.4

    # -- Build signal --
    long_sig = breakout_long * 50 * (1 + pyramid_long * 0.5) * (0.5 + 0.5 * trend_confirm)
    short_sig = breakout_short * 50 * (1 + pyramid_short * 0.5) * (0.5 + 0.5 * trend_confirm)

    # Apply System 1 filter (reduce signal on previously profitable direction)
    long_sig = long_sig * (1 - s1_suppress_long)
    short_sig = short_sig * (1 - s1_suppress_short)

    # Entry + exit combined
    raw = long_sig - short_sig - exit_long * 35 + exit_short * 35

    # Add penetration depth bonus
    raw = raw + pen_long * 8 - pen_short * 8

    # Smooth to sustain position
    raw = raw.rolling(5, min_periods=1).mean()

    return _clip_signal(raw, smooth=3)


def s013_vortex_trend(ind):
    """013 | Vortex Indicator Trend Strength (Botes & Siepman)
    Measures positive/negative trend movement via high-low range structure.
    VI+ = sum(|H_t - L_{t-1}|, 14) / sum(TR, 14)
    VI- = sum(|L_t - H_{t-1}|, 14) / sum(TR, 14)
    Signal on crossovers with diff > 0.10 threshold, fresh cross < 5 bars.
    Range filter: |diff| < 0.05 for 10+ bars = no trade.
    """
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    tr = ind["tr"]
    n = 14

    # -- Vortex movement per spec --
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    # -- Rolling sums for VI computation --
    tr_sum = tr.rolling(n, min_periods=n).sum().replace(0, np.nan)
    vi_plus = vm_plus.rolling(n, min_periods=n).sum() / tr_sum
    vi_minus = vm_minus.rolling(n, min_periods=n).sum() / tr_sum

    # -- Vortex Difference (trend strength) --
    vortex_diff = (vi_plus - vi_minus).fillna(0)

    # -- Crossover detection --
    cross_bull = _crossover(vi_plus, vi_minus)
    cross_bear = _crossunder(vi_plus, vi_minus)

    # -- Cross age: bars since last crossover --
    any_cross = ((cross_bull > 0.5) | (cross_bear > 0.5)).astype(int)
    cross_groups = any_cross.cumsum()
    cross_age = cross_groups.groupby(cross_groups).cumcount()

    # -- Fresh crossover filter: only trade within 5 bars of cross --
    fresh = (cross_age < 5).astype(float)
    freshness_decay = (1.0 - cross_age / 5).clip(0, 1)  # decays from 1 to 0

    # -- Strength threshold: |diff| > 0.10 for strong signals --
    strong = (vortex_diff.abs() > 0.10).astype(float)
    moderate = (vortex_diff.abs() > 0.05).astype(float)

    # -- Range-bound filter: |diff| < 0.05 for 10+ bars = avoid --
    range_bars = (vortex_diff.abs() < 0.05).astype(int)
    range_count = range_bars.rolling(15, min_periods=1).sum()
    range_suppress = (1 - (range_count / 15).clip(0, 0.7))  # suppress up to 70%

    # -- Build signal components --
    # 1) Base direction from vortex diff (z-scored)
    vd_z = _z(vortex_diff, 40)
    base_dir = vd_z * 30

    # 2) Crossover spikes (fresh and strong only)
    cross_spike = (cross_bull * 35 - cross_bear * 35) * freshness_decay * strong

    # 3) Sustained trend bonus when diff is large and persistent
    sustained = vortex_diff.clip(-0.3, 0.3) / 0.3 * 20 * moderate

    # -- Combine with range suppression --
    raw = (base_dir + cross_spike + sustained) * range_suppress

    return _clip_signal(raw, smooth=3)


def s014_chande_momentum(ind):
    """014 | Chande Momentum Oscillator Divergence (Tushar Chande)
    CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down) over n=14.
    Unsmoothed unlike RSI, giving sharper divergences.
    Bearish div: price higher high + CMO lower high, confirmed by CMO < 0.
    Bullish div: price lower low + CMO higher low, confirmed by CMO > 0.
    Extreme zones: CMO > 50 overbought, CMO < -50 oversold.
    """
    close = ind["close"]
    n_cmo = 14

    # -- CMO computation per spec --
    diff = close.diff()
    su = diff.clip(lower=0).rolling(n_cmo, min_periods=n_cmo).sum()
    sd = (-diff.clip(upper=0)).rolling(n_cmo, min_periods=n_cmo).sum()
    denom = (su + sd).replace(0, np.nan)
    cmo = (100 * (su - sd) / denom).fillna(0)

    # -- Peak/trough detection for divergence (order=5 per spec) --
    c = close.values.astype(float)
    cmo_v = cmo.values.astype(float)
    order = 5
    nn = len(c)

    # Find local maxima/minima in price and CMO
    price_peaks = np.zeros(nn, dtype=bool)
    price_troughs = np.zeros(nn, dtype=bool)
    cmo_peaks = np.zeros(nn, dtype=bool)
    cmo_troughs = np.zeros(nn, dtype=bool)

    for i in range(order, nn - order):
        # Price peaks/troughs
        if all(c[i] >= c[i - j] for j in range(1, order + 1)) and \
           all(c[i] >= c[i + j] for j in range(1, order + 1)):
            price_peaks[i] = True
        if all(c[i] <= c[i - j] for j in range(1, order + 1)) and \
           all(c[i] <= c[i + j] for j in range(1, order + 1)):
            price_troughs[i] = True
        # CMO peaks/troughs
        if all(cmo_v[i] >= cmo_v[i - j] for j in range(1, order + 1)) and \
           all(cmo_v[i] >= cmo_v[i + j] for j in range(1, order + 1)):
            cmo_peaks[i] = True
        if all(cmo_v[i] <= cmo_v[i - j] for j in range(1, order + 1)) and \
           all(cmo_v[i] <= cmo_v[i + j] for j in range(1, order + 1)):
            cmo_troughs[i] = True

    # -- Detect divergences --
    bearish_div = np.zeros(nn)
    bullish_div = np.zeros(nn)

    # Track last peak/trough values and locations
    last_price_peak_val = np.nan
    last_price_peak_idx = 0
    last_cmo_peak_val = np.nan
    last_price_trough_val = np.nan
    last_price_trough_idx = 0
    last_cmo_trough_val = np.nan

    for i in range(order, nn):
        # Bearish divergence: price higher high + CMO lower high
        if price_peaks[i] and cmo_peaks[i]:
            if not np.isnan(last_price_peak_val) and not np.isnan(last_cmo_peak_val):
                if c[i] > last_price_peak_val and cmo_v[i] < last_cmo_peak_val:
                    # Spread the divergence signal over a few bars for smoothness
                    end = min(i + 8, nn)
                    bearish_div[i:end] = np.maximum(
                        bearish_div[i:end],
                        np.linspace(1.0, 0.3, end - i)
                    )
            last_price_peak_val = c[i]
            last_price_peak_idx = i
            last_cmo_peak_val = cmo_v[i]
        elif price_peaks[i]:
            last_price_peak_val = c[i]
            last_price_peak_idx = i
        elif cmo_peaks[i]:
            last_cmo_peak_val = cmo_v[i]

        # Bullish divergence: price lower low + CMO higher low
        if price_troughs[i] and cmo_troughs[i]:
            if not np.isnan(last_price_trough_val) and not np.isnan(last_cmo_trough_val):
                if c[i] < last_price_trough_val and cmo_v[i] > last_cmo_trough_val:
                    end = min(i + 8, nn)
                    bullish_div[i:end] = np.maximum(
                        bullish_div[i:end],
                        np.linspace(1.0, 0.3, end - i)
                    )
            last_price_trough_val = c[i]
            last_price_trough_idx = i
            last_cmo_trough_val = cmo_v[i]
        elif price_troughs[i]:
            last_price_trough_val = c[i]
            last_price_trough_idx = i
        elif cmo_troughs[i]:
            last_cmo_trough_val = cmo_v[i]

    bearish_s = pd.Series(bearish_div, index=close.index)
    bullish_s = pd.Series(bullish_div, index=close.index)

    # -- CMO zero-cross confirmation per spec --
    cmo_below_zero = (cmo < 0).astype(float)
    cmo_above_zero = (cmo > 0).astype(float)

    # -- Extreme zones --
    overbought = (cmo > 50).astype(float)
    oversold = (cmo < -50).astype(float)

    # -- CMO base direction (z-scored) --
    cmo_z = _z(cmo, 60)

    # -- Build signal --
    # 1) Base CMO direction
    base = cmo_z * 25

    # 2) Divergence signals (confirmed by CMO crossing zero)
    div_sell = bearish_s * cmo_below_zero * (-45)
    div_buy = bullish_s * cmo_above_zero * 45

    # 3) Extreme zone signals (counter-trend)
    extreme = -overbought * 15 + oversold * 15

    raw = base + div_sell + div_buy + extreme

    return _clip_signal(raw, smooth=3)


def s015_parabolic_sar(ind):
    """015 | Parabolic SAR Acceleration System (Wilder 1978)
    SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1})
    AF starts 0.02, increments 0.02 per new EP, max 0.20.
    Reversal when price crosses SAR; new SAR = EP of previous trend.
    Modified: volatility filter ATR(5) > 0.8*ATR(20) for genuine moves.
    Enhanced: require ADX(14) > 25 for trend confirmation.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    tr = ind["tr"]

    n = len(close)
    c = close.values.astype(float)
    h = high.values.astype(float)
    lo = low.values.astype(float)

    sar = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    af_arr = np.zeros(n)
    ep_arr = np.zeros(n)
    flip = np.zeros(n)  # 1 = bullish flip, -1 = bearish flip

    # Initialize
    af = 0.02
    ep = h[0]
    sar[0] = lo[0]
    af_arr[0] = af
    ep_arr[0] = ep

    for i in range(1, n):
        if direction[i - 1] == 1:  # bullish
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            # SAR can't be above prior two lows
            sar[i] = min(sar[i], lo[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], lo[i - 2])

            if c[i] < sar[i]:
                # Reversal to bearish
                direction[i] = -1
                sar[i] = ep  # new SAR = EP of previous trend
                ep = lo[i]
                af = 0.02
                flip[i] = -1
            else:
                direction[i] = 1
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + 0.02, 0.20)
        else:  # bearish
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            # SAR can't be below prior two highs
            sar[i] = max(sar[i], h[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], h[i - 2])

            if c[i] > sar[i]:
                # Reversal to bullish
                direction[i] = 1
                sar[i] = ep
                ep = h[i]
                af = 0.02
                flip[i] = 1
            else:
                direction[i] = -1
                if lo[i] < ep:
                    ep = lo[i]
                    af = min(af + 0.02, 0.20)

        af_arr[i] = af
        ep_arr[i] = ep

    dir_s = pd.Series(direction, index=close.index)
    sar_s = pd.Series(sar, index=close.index)
    flip_s = pd.Series(flip, index=close.index)
    af_s = pd.Series(af_arr, index=close.index)

    # -- Volatility filter per spec: ATR(5) > 0.8 * ATR(20) for genuine moves --
    atr5 = tr.rolling(5, min_periods=3).mean()
    atr20 = tr.rolling(20, min_periods=10).mean()
    vol_expanding = (atr5 > 0.8 * atr20).astype(float)
    vol_filter = 0.5 + 0.5 * vol_expanding  # 0.5 base, 1.0 when expanding

    # -- ADX confirmation per spec: ADX > 25 --
    adx = ind["adx"].fillna(15)
    adx_filter = ((adx - 15) / 25).clip(0.3, 1.0)  # 0.3 floor, ramps to 1.0 at ADX=40

    # -- SAR distance (how far price is from SAR, normalized) --
    atr_safe = ind["atr14"].replace(0, np.nan)
    sar_dist = ((close - sar_s) / atr_safe).clip(-4, 4)

    # -- AF acceleration: higher AF = more momentum conviction --
    af_boost = (af_s / 0.20).clip(0, 1)  # 0 at af=0, 1 at max af

    # -- Direction persistence --
    dir_change = (dir_s != dir_s.shift(1)).astype(int)
    group = dir_change.cumsum()
    persist = group.groupby(group).cumcount() + 1
    persist_factor = (persist.clip(1, 15) / 15).clip(0.4, 1.0)

    # -- Build signal --
    # Base direction
    base = dir_s * 40

    # Flip spikes (fresh reversals, filtered by volatility)
    flip_spike = flip_s * 25 * vol_filter

    # Distance scaling (confirms trend strength)
    dist_component = sar_dist * 8 * af_boost

    # Persistence bonus
    persist_bonus = dir_s * persist_factor * 10

    raw = (base + flip_spike + dist_component + persist_bonus) * adx_filter

    return _clip_signal(raw, smooth=3)


def s016_klinger_oscillator(ind):
    """016 | Klinger Oscillator Volume Trend (Stephen Klinger)
    Volume Force = Volume * |2*(dm/cm) - 1| * Trend * 100
    where cm = cumulative dm (reset on trend change).
    KVO = EMA(VF, 34) - EMA(VF, 55), Signal = EMA(KVO, 13).
    Long on KVO cross above signal with expanding histogram.
    Divergence: price new high + KVO declining = distribution.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]

    # -- Trend direction per spec: based on HLC sum --
    hlc = high + low + close
    trend = np.sign(hlc - hlc.shift(1)).fillna(1)

    # -- Daily range and cumulative range --
    dm = high - low
    cm = dm.values.astype(float).copy()
    trend_v = trend.values.astype(float)
    for i in range(1, len(cm)):
        if trend_v[i] == trend_v[i - 1]:
            cm[i] = cm[i - 1] + dm.values[i]
        else:
            cm[i] = dm.values[i - 1] + dm.values[i]
    cm_s = pd.Series(cm, index=close.index).replace(0, np.nan)

    # -- Volume Force per spec --
    vf = volume * (2 * dm / cm_s - 1).abs() * trend * 100

    # -- KVO and Signal line --
    kvo = vf.ewm(span=34, adjust=False).mean() - vf.ewm(span=55, adjust=False).mean()
    signal_line = kvo.ewm(span=13, adjust=False).mean()
    kvo_hist = kvo - signal_line

    # -- Crossover detection --
    cross_bull = _crossover(kvo, signal_line)
    cross_bear = _crossunder(kvo, signal_line)

    # -- Histogram expansion/contraction --
    hist_expanding = (kvo_hist.abs() > kvo_hist.abs().shift(1)).astype(float)
    hist_contracting = 1 - hist_expanding

    # -- Divergence detection: price making new highs but KVO declining --
    price_hh = (close == close.rolling(20, min_periods=10).max()).astype(float)
    price_ll = (close == close.rolling(20, min_periods=10).min()).astype(float)
    kvo_declining = (kvo < kvo.shift(5)).astype(float)
    kvo_rising = (kvo > kvo.shift(5)).astype(float)

    bearish_div = price_hh * kvo_declining  # distribution
    bullish_div = price_ll * kvo_rising  # accumulation

    # -- Build signal --
    # 1) Base momentum from KVO histogram z-score
    hist_z = _z(kvo_hist, 40)
    base = hist_z * 30

    # 2) Crossover spikes (confirmed by histogram expansion)
    cross_component = (cross_bull * 35 - cross_bear * 35) * (0.6 + 0.4 * hist_expanding)

    # 3) Divergence warnings
    div_component = bullish_div * 20 - bearish_div * 20

    # 4) KVO direction (longer-term flow)
    kvo_dir = np.sign(kvo) * (kvo.abs() / kvo.abs().rolling(60, min_periods=20).mean().replace(0, np.nan)).clip(0, 2) * 10

    raw = base + cross_component + div_component + kvo_dir

    return _clip_signal(raw, smooth=3)


def s017_alma_trend(ind):
    """017 | Arnaud Legoux Moving Average Trend (Paris School)
    Gaussian-weighted MA with offset=0.85 for near-zero lag.
    Dual ALMA: fast(9) - slow(21) crossover system.
    ALMA_signal = ALMA(diff, 5, 0.85, 6).
    Strength scaled by |diff| / ATR(14).
    """
    close = ind["close"]
    atr = ind["atr14"].replace(0, np.nan)

    # -- Dual ALMA per spec --
    alma_fast = _alma(close, 9, 0.85, 6)
    alma_slow = _alma(close, 21, 0.85, 6)
    alma_diff = alma_fast - alma_slow
    alma_signal = _alma(alma_diff, 5, 0.85, 6)
    alma_hist = alma_diff - alma_signal

    # -- Crossover detection --
    cross_bull = _crossover(alma_diff, alma_signal)
    cross_bear = _crossunder(alma_diff, alma_signal)

    # -- Strength: |ALMA_diff| / ATR(14) as conviction multiplier per spec --
    strength = (alma_diff.abs() / atr).clip(0, 3)
    strength_factor = 0.4 + 0.6 * (strength / 3)  # 0.4 base, up to 1.0

    # -- ALMA slope for trend persistence --
    alma_fast_slope = alma_fast.diff(3) / atr
    slope_dir = np.sign(alma_fast_slope)

    # -- Histogram momentum (expanding vs contracting) --
    hist_accel = alma_hist - alma_hist.shift(2)
    hist_expanding = (hist_accel * alma_hist > 0).astype(float)  # same direction = expanding

    # -- Build signal --
    # 1) Histogram z-score base
    hist_z = _z(alma_hist, 30)
    base = hist_z * 30

    # 2) Crossover spikes (must confirm ALMA_diff side per spec)
    diff_positive = (alma_diff > 0).astype(float)
    diff_negative = (alma_diff < 0).astype(float)
    cross_component = (cross_bull * diff_positive * 30 - cross_bear * diff_negative * 30)

    # 3) Direction + strength
    dir_component = slope_dir * strength_factor * 20

    # 4) Histogram expansion bonus
    expansion_bonus = np.sign(alma_hist) * hist_expanding * 10

    raw = (base + cross_component + dir_component + expansion_bonus)

    return _clip_signal(raw, smooth=3)


def s018_mcginley_dynamic(ind):
    """018 | McGinley Dynamic Line (John McGinley)
    Self-adjusting exponential: MD_t = MD_{t-1} + (P - MD_{t-1}) / (k*N*(P/MD)^4)
    k=0.6, N=14. The quartic term speeds up during breakouts, slows during pullbacks.
    Long: Close > MD AND MD rising (3-bar slope > 0).
    Crossover with volume > 1.5x average = high-conviction.
    """
    close = ind["close"]
    volume = ind["volume"]
    n_per = 14
    k = 0.6

    # -- McGinley Dynamic computation per spec --
    vals = close.values.astype(float)
    md = np.full(len(vals), np.nan)
    md[0] = vals[0]
    for i in range(1, len(vals)):
        if np.isnan(md[i - 1]) or md[i - 1] == 0:
            md[i] = vals[i]
        else:
            ratio = vals[i] / md[i - 1]
            denom = k * n_per * ratio ** 4
            if denom == 0:
                md[i] = vals[i]
            else:
                md[i] = md[i - 1] + (vals[i] - md[i - 1]) / denom
    md_s = pd.Series(md, index=close.index)

    # -- MD slope (3-bar) per spec --
    md_slope = md_s.diff(3)
    md_rising = (md_slope > 0).astype(float)
    md_falling = (md_slope < 0).astype(float)

    # -- Price position relative to MD --
    above = (close > md_s).astype(float)
    below = (close < md_s).astype(float)

    # -- Distance from MD, normalized by ATR --
    atr = ind["atr14"].replace(0, np.nan)
    dist = ((close - md_s) / atr).clip(-4, 4)

    # -- Crossover detection --
    price_cross_above = (above > above.shift(1)).astype(float)
    price_cross_below = (below > below.shift(1)).astype(float)

    # -- Volume confirmation: volume > 1.5x average per spec --
    vol_avg = volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    vol_surge = (volume / vol_avg).clip(0.5, 3.0)
    high_vol = (vol_surge > 1.5).astype(float)

    # -- Slope magnitude for conviction --
    slope_mag = (md_slope.abs() / atr).clip(0, 0.1) / 0.1  # normalized 0-1

    # -- Build signal --
    # 1) Direction: above/below MD with slope confirmation per spec
    dir_component = (above * md_rising - below * md_falling) * 40

    # 2) Distance scaling (further from MD = more conviction)
    dist_component = dist * 10

    # 3) Crossover spikes (high conviction with volume)
    cross_vol_factor = 0.6 + 0.4 * high_vol
    cross_component = (price_cross_above * 25 - price_cross_below * 25) * cross_vol_factor

    # 4) Slope magnitude bonus
    slope_bonus = np.sign(md_slope) * slope_mag * 15

    raw = dir_component + dist_component + cross_component + slope_bonus

    return _clip_signal(raw, smooth=3)


def s019_bb_squeeze_breakout(ind):
    """019 | Bollinger BandWidth Squeeze Breakout (John Bollinger)
    BandWidth = (Upper - Lower) / Mid * 100.
    Squeeze = BW in lowest 10% of 120-bar range (6-month percentile).
    Breakout: Close > BB_Upper after squeeze duration >= 5 bars.
    Size by squeeze duration (longer = more energy stored).
    Volume > 1.5x average confirms genuine breakout.
    """
    close = ind["close"]
    volume = ind["volume"]

    # -- Bollinger components from base.py --
    bb_upper = ind["bb_upper"]
    bb_lower = ind["bb_lower"]
    bb_pctb = ind["bb_pctb"]
    bb_width = ind["bb_width"]

    # -- BandWidth percentile per spec (120-bar rolling rank) --
    bw_rank = bb_width.rolling(120, min_periods=30).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )

    # -- Squeeze detection: BW percentile < 20% (relaxed from strict 10%) --
    in_squeeze = (bw_rank < 0.20).astype(float)

    # -- Squeeze duration: consecutive bars in squeeze --
    squeeze_groups = (in_squeeze != in_squeeze.shift(1)).cumsum()
    squeeze_duration = in_squeeze.groupby(squeeze_groups).cumcount() + 1
    squeeze_duration = squeeze_duration * in_squeeze  # zero when not in squeeze

    # -- Rolling max squeeze duration (recent memory) --
    recent_squeeze = squeeze_duration.rolling(20, min_periods=1).max()

    # -- Breakout detection: close beyond bands OR strong momentum after squeeze --
    had_squeeze = (recent_squeeze >= 3).astype(float)  # relaxed from 5
    breakout_up = ((bb_pctb > 0.95) & (had_squeeze.shift(1) > 0)).astype(float)
    breakout_dn = ((bb_pctb < 0.05) & (had_squeeze.shift(1) > 0)).astype(float)

    # -- Volume confirmation per spec: volume > 1.5x average --
    vol_avg = volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    vol_surge = (volume / vol_avg).clip(0.5, 3.0)
    high_vol = (vol_surge > 1.5).astype(float)
    vol_factor = 0.5 + 0.5 * high_vol

    # -- Size by squeeze duration per spec (longer = more energy) --
    squeeze_energy = (recent_squeeze / 20).clip(0, 1)  # normalized 0-1
    energy_factor = 0.5 + 0.5 * squeeze_energy

    # -- Sustain breakout signal for several bars --
    breakout_long = breakout_up.rolling(8, min_periods=1).max()
    breakout_short = breakout_dn.rolling(8, min_periods=1).max()

    # -- Momentum confirmation --
    mom = ind["mom_5"].fillna(0)
    mom_confirm = (np.sign(mom) * breakout_long > 0).astype(float) * 0.3 + 0.7

    # -- Build signal --
    raw = (breakout_long * 55 - breakout_short * 55) * vol_factor * energy_factor * mom_confirm

    # -- During squeeze, add mild mean-reversion --
    squeeze_mr = in_squeeze * (0.5 - bb_pctb) * 20

    # -- Non-squeeze BB direction (primary signal source between squeezes) --
    bb_dir = (bb_pctb - 0.5).clip(-0.5, 0.5) * 2  # -1 to +1
    mom_s = ind["mom_score"].fillna(0)
    trend_dir = ind["trend_score"].fillna(0)
    # Blend BB position with momentum for persistent direction
    non_squeeze_dir = bb_dir * 30 + mom_s * 20 + trend_dir * 15

    raw = raw + squeeze_mr + non_squeeze_dir

    return _clip_signal(raw, smooth=3)


def s020_keltner_mean_reversion(ind):
    """020 | Keltner Channel Mean Reversion (Keltner/Raschke)
    KC_Mid = EMA(20), KC_Upper/Lower = EMA(20) +/- mult*ATR.
    Stretch = normalized distance from center, range ~[-1, +1].
    Reversion: oversold (price below lower KC AND first up bar) = long.
    Filter: ADX < 25 preferred (non-trending environment).
    Target: Keltner_Mid (EMA 20).
    """
    close = ind["close"]
    rsi = ind["rsi"].fillna(50)

    # -- Use pre-computed KC from base.py (1.5x ATR, tighter for better signals) --
    kc_pct = ind["kc_pct"]  # 0 at lower, 1 at upper, clipped [-0.5, 1.5]

    # -- Stretch: distance from center, normalized --
    stretch = (kc_pct - 0.5) * 2  # maps 0->-1, 0.5->0, 1.0->+1

    # -- Oversold/overbought --
    oversold = (kc_pct < 0.1)
    overbought = (kc_pct > 0.9)

    # -- First reversal bar per spec --
    first_up = close > close.shift(1)
    first_down = close < close.shift(1)

    reversion_long = (oversold & first_up).astype(float)
    reversion_short = (overbought & first_down).astype(float)

    # -- ADX filter: prefer non-trending but keep minimum strength --
    adx = ind["adx"].fillna(20)
    adx_filter = ((40 - adx) / 25).clip(0.4, 1.0)

    # -- Stretch magnitude for conviction --
    conviction = stretch.abs().clip(0, 2) / 2

    # -- Sustain reversion signal --
    reversion_long_sustained = reversion_long.rolling(12, min_periods=1).max()
    reversion_short_sustained = reversion_short.rolling(12, min_periods=1).max()

    # Decay as price approaches target
    long_decay = (0.5 - kc_pct).clip(0, 1)  # higher when below mid
    short_decay = (kc_pct - 0.5).clip(0, 1)

    # -- RSI confirmation --
    rsi_oversold = ((40 - rsi) / 20).clip(0, 1)
    rsi_overbought = ((rsi - 60) / 20).clip(0, 1)

    # -- Build signal --
    # 1) Reversion spikes
    spike_long = reversion_long * 50 * (0.5 + 0.5 * conviction)
    spike_short = reversion_short * 50 * (0.5 + 0.5 * conviction)

    # 2) Sustained reversion
    sustained_long = reversion_long_sustained * long_decay * 35 * rsi_oversold
    sustained_short = reversion_short_sustained * short_decay * 35 * rsi_overbought

    # 3) General stretch mean-reversion (primary persistent signal)
    mild_mr = -stretch.clip(-2, 2) / 2 * 45

    # 4) RSI mean-reversion component
    rsi_mr = ((50 - rsi) / 50).fillna(0) * 20

    raw = (spike_long - spike_short + sustained_long - sustained_short + mild_mr + rsi_mr) * adx_filter

    return _clip_signal(raw, smooth=3)


def s021_alligator_fractal(ind):
    """021 | Williams Alligator Fractal Strategy (Bill Williams)
    Three smoothed moving averages (SMMA) with displacement:
    Jaw = SMMA(13, displaced 8), Teeth = SMMA(8, displaced 5), Lips = SMMA(5, displaced 3).
    Sleeping: lines intertwined (spread < threshold). Eating: lines diverging in order.
    Fractal breakout above/below jaw confirms entry.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]

    # -- SMMA (Smoothed MA) computation per spec --
    # SMMA(n) = [prev_smma * (n-1) + close] / n
    def smma(series, n):
        vals = series.values.astype(float)
        out = np.full(len(vals), np.nan)
        # Initialize with SMA
        if len(vals) >= n:
            out[n - 1] = np.mean(vals[:n])
            for i in range(n, len(vals)):
                out[i] = (out[i - 1] * (n - 1) + vals[i]) / n
        return pd.Series(out, index=series.index)

    # -- Alligator lines per spec (median price = (H+L)/2) --
    median_price = (high + low) / 2
    jaw = smma(median_price, 13).shift(8)     # Blue line, displaced 8
    teeth = smma(median_price, 8).shift(5)    # Red line, displaced 5
    lips = smma(median_price, 5).shift(3)     # Green line, displaced 3

    # -- State detection per spec --
    # Sleeping: lines intertwined, low spread
    spread = (lips - jaw).abs() / close * 100
    spread_smooth = spread.rolling(10, min_periods=3).mean()
    sleeping = (spread_smooth < 0.3).astype(float)

    # Eating up: Lips > Teeth > Jaw (bullish order)
    eating_up = ((lips > teeth) & (teeth > jaw)).astype(float)
    # Eating down: Lips < Teeth < Jaw (bearish order)
    eating_down = ((lips < teeth) & (teeth < jaw)).astype(float)

    # -- Fractal detection (5-bar high/low patterns) per spec --
    # Up fractal: bar[i-2] has highest high among 5 bars
    up_fractal = np.zeros(len(close))
    dn_fractal = np.zeros(len(close))
    h_v = high.values.astype(float)
    l_v = low.values.astype(float)
    for i in range(2, len(close) - 2):
        if h_v[i] > h_v[i - 1] and h_v[i] > h_v[i - 2] and h_v[i] > h_v[i + 1] and h_v[i] > h_v[i + 2]:
            up_fractal[i] = h_v[i]
        if l_v[i] < l_v[i - 1] and l_v[i] < l_v[i - 2] and l_v[i] < l_v[i + 1] and l_v[i] < l_v[i + 2]:
            dn_fractal[i] = l_v[i]

    up_frac = pd.Series(up_fractal, index=close.index)
    dn_frac = pd.Series(dn_fractal, index=close.index)

    # Forward fill fractal levels for breakout detection
    last_up_frac = up_frac.replace(0, np.nan).ffill()
    last_dn_frac = dn_frac.replace(0, np.nan).ffill()

    # -- Fractal breakout: close above last up fractal --
    frac_break_up = (close > last_up_frac).astype(float)
    frac_break_dn = (close < last_dn_frac).astype(float)

    # -- Directional spread (signed) --
    dir_spread = (lips - jaw) / close * 100

    # -- Build signal --
    # 1) Base direction from eating state
    eat_signal = eating_up * 40 - eating_down * 40

    # 2) Spread magnitude conviction (wider = stronger trend)
    spread_conviction = dir_spread.clip(-3, 3) / 3 * 15

    # 3) Fractal breakout confirmation (only when eating in same direction)
    frac_long = frac_break_up * eating_up * 25
    frac_short = frac_break_dn * eating_down * 25

    # 4) Sleeping penalty (reduce position during consolidation)
    sleep_penalty = sleeping * (-15) * np.sign(eat_signal.shift(1).fillna(0))

    raw = eat_signal + spread_conviction + frac_long - frac_short + sleep_penalty

    return _clip_signal(raw, smooth=3)


def s022_range_expansion(ind):
    """022 | Tokyo Range Expansion Index (DeMark REI)
    REI measures range expansion via conditional price changes over 5 bars.
    Overbought >+60 declining -> sell exhaustion.
    Oversold <-60 rising -> buy exhaustion.
    Neutral zone -40 to +40 -> no signal.
    """
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]

    # -- DeMark REI computation per spec --
    # Conditional: high[t-1]>=close[t-3] OR high[t-2]>=close[t-4]
    # (same for lows)
    h1 = high - high.shift(2)
    l1 = low - low.shift(2)

    cond_h = ((high.shift(1) >= close.shift(3)) | (high.shift(2) >= close.shift(4)))
    cond_l = ((low.shift(1) <= close.shift(3)) | (low.shift(2) <= close.shift(4)))

    # Numerator: conditional sum of range changes
    num_h = h1.where(cond_h, 0.0)
    num_l = l1.where(cond_l, 0.0)
    rei_num = (num_h + num_l).rolling(5, min_periods=3).sum()

    # Denominator: absolute range changes (always)
    rei_den = (h1.abs() + l1.abs()).rolling(5, min_periods=3).sum().replace(0, np.nan)

    # REI = (num / den) * 100, bounded [-100, +100]
    rei = (rei_num / rei_den * 100).clip(-100, 100)
    rei_smooth = rei.ewm(span=3, min_periods=1).mean()

    # -- Slope for exhaustion detection --
    rei_slope = rei_smooth.diff(3)
    rei_slope_smooth = rei_slope.ewm(span=3, min_periods=1).mean()

    # -- Exhaustion signals per spec --
    # Overbought exhaustion: REI > +60 AND slope declining
    ob_zone = (rei_smooth > 60).astype(float)
    ob_exhaustion = ob_zone * (rei_slope_smooth < 0).astype(float)

    # Oversold exhaustion: REI < -60 AND slope rising
    os_zone = (rei_smooth < -60).astype(float)
    os_exhaustion = os_zone * (rei_slope_smooth > 0).astype(float)

    # -- Signal construction --
    # 1) Exhaustion reversal (primary): sell when overbought exhausting, buy when oversold exhausting
    exhaustion_signal = os_exhaustion * 55 - ob_exhaustion * 55

    # 2) Extreme conviction scaling
    extreme_bull = ((rei_smooth < -80).astype(float)) * 20
    extreme_bear = ((rei_smooth > 80).astype(float)) * (-20)

    # 3) Momentum continuation (trending zone beyond neutral)
    trend_bull = ((rei_smooth > 40) & (rei_slope_smooth > 0)).astype(float) * 20
    trend_bear = ((rei_smooth < -40) & (rei_slope_smooth < 0)).astype(float) * 20

    # 4) Directional zone signal (proportional to REI level beyond neutral)
    zone_signal = rei_smooth.clip(-80, 80) / 80 * 15

    # 5) Neutral zone suppression (lighter touch)
    in_neutral = ((rei_smooth > -40) & (rei_smooth < 40)).astype(float)
    neutral_dampen = 1 - in_neutral * 0.5

    raw = (exhaustion_signal + extreme_bull + extreme_bear + trend_bull - trend_bear + zone_signal) * neutral_dampen

    return _clip_signal(raw, smooth=3)


def s023_dmi_multiperiod(ind):
    """023 | Directional Movement Index Multi-Period Composite
    Computes ADX at multiple periods (7, 14, 28) from +DI/-DI.
    Trend_Strength = weighted average of ADX values.
    Trend_Acceleration = ADX_fast(7) - ADX_slow(28).
    Entry: strength > 25 AND acceleration > 5.
    Exhaustion: strength > 40 AND acceleration < -3 (trend fading).
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    tr = ind["tr"]

    # -- Compute +DM and -DM per spec --
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=close.index)

    # -- ADX at multiple periods --
    def compute_adx(period):
        atr_p = tr.ewm(span=period, min_periods=period, adjust=False).mean()
        plus_di = (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr_p.replace(0, np.nan)) * 100
        minus_di = (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr_p.replace(0, np.nan)) * 100
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx_val = dx.ewm(span=period, min_periods=period, adjust=False).mean()
        direction = plus_di - minus_di  # positive = bullish
        return adx_val, direction

    adx_7, dir_7 = compute_adx(7)
    adx_14, dir_14 = compute_adx(14)
    adx_28, dir_28 = compute_adx(28)

    # -- Trend strength: weighted average per spec (fast gets more weight) --
    trend_strength = adx_7 * 0.45 + adx_14 * 0.35 + adx_28 * 0.20

    # -- Trend acceleration: fast - slow --
    trend_accel = adx_7 - adx_28
    trend_accel_smooth = trend_accel.ewm(span=5, min_periods=2).mean()

    # -- Composite direction (weighted) --
    direction = dir_7 * 0.45 + dir_14 * 0.35 + dir_28 * 0.20
    dir_sign = np.sign(direction)

    # -- Entry conditions per spec --
    # Strong trend entry: strength > 25 AND accel > 5
    strong_entry = ((trend_strength > 25) & (trend_accel_smooth > 5)).astype(float)
    # Moderate trend: strength > 20
    moderate_trend = ((trend_strength > 20) & (trend_strength <= 35)).astype(float)

    # -- Exhaustion per spec: strength > 40 AND accel < -3 --
    exhaustion = ((trend_strength > 40) & (trend_accel_smooth < -3)).astype(float)

    # -- Trend intensity scaling --
    intensity = (trend_strength / 50).clip(0, 1)

    # -- Build signal --
    # 1) Strong entry: full conviction directional
    entry_signal = strong_entry * dir_sign * intensity * 50

    # 2) Moderate trending (reduced conviction)
    moderate_signal = moderate_trend * (1 - strong_entry) * dir_sign * intensity * 25

    # 3) Acceleration bonus (accelerating trends)
    accel_bonus = (trend_accel_smooth.clip(0, 15) / 15) * dir_sign * 10

    # 4) Exhaustion fade (counter-trend when trend fading)
    exh_signal = exhaustion * (-dir_sign) * 20

    raw = entry_signal + moderate_signal + accel_bonus + exh_signal

    return _clip_signal(raw, smooth=3)


def s024_cci_mean_reversion(ind):
    """024 | Commodity Channel Index Mean Reversion
    CCI = (Typical_Price - SMA(TP, 20)) / (0.015 * Mean_Deviation)
    Extreme oversold: CCI < -200 with slope reversing up -> buy.
    Extreme overbought: CCI > +200 with slope reversing down -> sell.
    Recovery: CCI was recently extreme and is now returning toward zero -> sustained signal.
    Exit at zero-line crossing.
    Risk: vol-scaled position sizing via vol_dampener.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]

    # -- Full CCI computation per spec --
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20, min_periods=10).mean()
    mean_dev = tp.rolling(20, min_periods=10).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    mean_dev = mean_dev.replace(0, np.nan)
    cci = (tp - tp_sma) / (0.015 * mean_dev)

    # -- CCI dynamics per spec --
    cci_slope = cci.diff(3)
    cci_slope_smooth = cci_slope.ewm(span=5, min_periods=2).mean()

    # -- Zone detection --
    extreme_os = (cci < -200).astype(float)
    extreme_ob = (cci > 200).astype(float)
    std_os = (cci < -100).astype(float)
    std_ob = (cci > 100).astype(float)

    # -- Recovery detection: CCI was extreme recently, now returning --
    # (sustained signal while price reverts from extreme)
    was_extreme_os = (cci.rolling(10, min_periods=3).min() < -100).astype(float)
    was_extreme_ob = (cci.rolling(10, min_periods=3).max() > 100).astype(float)
    returning_bull = was_extreme_os * (cci > cci.shift(1)).astype(float) * (cci < 0).astype(float)
    returning_bear = was_extreme_ob * (cci < cci.shift(1)).astype(float) * (cci > 0).astype(float)

    # -- Deep extreme recovery (even stronger) --
    was_deep_os = (cci.rolling(15, min_periods=5).min() < -200).astype(float)
    was_deep_ob = (cci.rolling(15, min_periods=5).max() > 200).astype(float)
    deep_returning_bull = was_deep_os * (cci > cci.shift(1)).astype(float) * (cci < 50).astype(float)
    deep_returning_bear = was_deep_ob * (cci < cci.shift(1)).astype(float) * (cci > -50).astype(float)

    # -- Reversal detection: in zone + slope turning --
    reversal_bull = std_os * (cci_slope_smooth > 0).astype(float)
    reversal_bear = std_ob * (cci_slope_smooth < 0).astype(float)

    # -- Zero-line crossing (exit / confirmation) --
    zero_cross_up = _crossover(cci, pd.Series(0.0, index=cci.index))
    zero_cross_dn = _crossunder(cci, pd.Series(0.0, index=cci.index))

    # -- Vol dampener --
    dampener = ind.get("vol_dampener", pd.Series(1.0, index=close.index))

    # -- Build signal --
    # 1) In-zone reversal (high conviction burst)
    zone_signal = reversal_bull * 45 - reversal_bear * 45

    # 2) Recovery signal (sustained while returning from extreme)
    recovery_signal = returning_bull * 35 - returning_bear * 35

    # 3) Deep extreme recovery bonus
    deep_signal = deep_returning_bull * 15 - deep_returning_bear * 15

    # 4) Zero-line confirmation
    zero_signal = zero_cross_up * 15 - zero_cross_dn * 15

    raw = (zone_signal + recovery_signal + deep_signal + zero_signal) * dampener

    return _clip_signal(raw, smooth=3)


def s025_stoch_rsi_bounce(ind):
    """025 | Stochastic RSI Oversold Bounce
    StochRSI = (RSI - RSI_low(14)) / (RSI_high(14) - RSI_low(14))
    K = SMA(StochRSI, 3), D = SMA(K, 3).
    Buy: K < 0.10 AND K crosses above D AND price > SMA(200).
    Exit: K > 0.80.
    Double-bottom divergence: price makes lower low but StochRSI makes higher low.
    """
    close = ind["close"]
    rsi = ind["rsi"]
    sma_200 = ind["sma_200"]

    # -- StochRSI computation per spec --
    rsi_lo = rsi.rolling(14, min_periods=7).min()
    rsi_hi = rsi.rolling(14, min_periods=7).max()
    rsi_range = (rsi_hi - rsi_lo).replace(0, np.nan)
    stoch_rsi = (rsi - rsi_lo) / rsi_range  # normalized [0, 1]

    # -- K and D lines per spec --
    k_line = stoch_rsi.rolling(3, min_periods=1).mean()
    d_line = k_line.rolling(3, min_periods=1).mean()

    # -- Uptrend filter per spec: price > SMA(200) --
    uptrend = (close > sma_200).astype(float)

    # -- Oversold / overbought zones per spec --
    deep_oversold = (k_line < 0.10).astype(float)
    oversold = (k_line < 0.20).astype(float)
    overbought = (k_line > 0.80).astype(float)
    deep_overbought = (k_line > 0.90).astype(float)

    # -- K/D crossovers per spec --
    k_cross_up_d = _crossover(k_line, d_line)
    k_cross_dn_d = _crossunder(k_line, d_line)

    # -- Double-bottom divergence per spec --
    # Price makes lower low but StochRSI makes higher low over 20-bar window
    price_lo_20 = close.rolling(20, min_periods=10).min()
    srsi_lo_20 = k_line.rolling(20, min_periods=10).min()

    # Current is near low but StochRSI has been rising off lows
    price_near_low = (close < price_lo_20 * 1.02).astype(float)
    srsi_above_prev_low = (k_line > srsi_lo_20 + 0.05).astype(float)
    divergence_bull = price_near_low * srsi_above_prev_low * oversold

    # -- Build signal --
    # 1) Primary: K<0.10 + K cross above D + uptrend = strong buy
    primary_buy = deep_oversold * k_cross_up_d * uptrend * 50

    # 2) Standard oversold bounce (K<0.20 + K>D + uptrend)
    std_bounce = oversold * (k_line > d_line).astype(float) * uptrend * 25

    # 3) Divergence buy (high conviction when uptrend)
    div_signal = divergence_bull * 25 * (0.5 + uptrend * 0.5)

    # 4) Overbought exit / short signal (less aggressive without downtrend filter)
    ob_signal = -overbought * k_cross_dn_d * 30
    deep_ob_signal = -deep_overbought * 15

    # 5) K/D momentum in mid-range (trending)
    mid_range = ((k_line > 0.30) & (k_line < 0.70)).astype(float)
    mid_trend = mid_range * (k_line - d_line).clip(-0.2, 0.2) / 0.2 * 15 * uptrend

    raw = primary_buy + std_bounce + div_signal + ob_signal + deep_ob_signal + mid_trend

    return _clip_signal(raw, smooth=3)


def s026_zero_lag_dema(ind):
    """026 | Zero-Lag DEMA Momentum (Patrick Mulloy)
    DEMA(n) = 2*EMA(n) - EMA(EMA(n), n)
    TEMA(n) = 3*EMA - 3*EMA2 + EMA3
    ZL_Mom = TEMA(8) - DEMA(21), ZL_Signal = EMA(ZL_Mom, 5)
    ZL_Hist = ZL_Mom - ZL_Signal
    Impulse: Hist>0 AND increasing. Decay: Hist>0 AND decreasing.
    Reversal: histogram crosses zero.
    """
    close = ind["close"]

    # -- DEMA(21) per spec --
    ema21_1 = close.ewm(span=21, adjust=False).mean()
    ema21_2 = ema21_1.ewm(span=21, adjust=False).mean()
    dema_21 = 2 * ema21_1 - ema21_2

    # -- TEMA(8) per spec --
    ema8_1 = close.ewm(span=8, adjust=False).mean()
    ema8_2 = ema8_1.ewm(span=8, adjust=False).mean()
    ema8_3 = ema8_2.ewm(span=8, adjust=False).mean()
    tema_8 = 3 * ema8_1 - 3 * ema8_2 + ema8_3

    # -- Zero-Lag momentum histogram per spec --
    zl_mom = tema_8 - dema_21
    zl_signal = zl_mom.ewm(span=5, adjust=False).mean()
    zl_hist = zl_mom - zl_signal

    # Normalize histogram by price level for cross-asset comparability
    zl_hist_pct = zl_hist / close * 1000

    # -- Phase classification per spec --
    hist_rising = (zl_hist > zl_hist.shift(1)).astype(float)
    hist_falling = (zl_hist < zl_hist.shift(1)).astype(float)
    hist_positive = (zl_hist > 0).astype(float)
    hist_negative = (zl_hist < 0).astype(float)

    impulse_up = hist_positive * hist_rising        # accelerating up
    impulse_dn = hist_negative * hist_falling       # accelerating down
    decay_up = hist_positive * hist_falling          # decelerating up
    decay_dn = hist_negative * hist_rising           # decelerating down

    # -- Zero-line crossovers --
    zero_cross_up = _crossover(zl_hist, pd.Series(0.0, index=close.index))
    zero_cross_dn = _crossunder(zl_hist, pd.Series(0.0, index=close.index))

    # -- Histogram magnitude for conviction --
    hist_z = _z(zl_hist_pct, 40)

    # -- Build signal --
    # 1) Impulse phase (strong directional, full conviction)
    impulse_signal = impulse_up * 40 - impulse_dn * 40

    # 2) Decay phase (reduced conviction, tightening)
    decay_signal = decay_up * 15 - decay_dn * 15

    # 3) Zero-line crossover (reversal confirmation)
    cross_signal = zero_cross_up * 20 - zero_cross_dn * 20

    # 4) Histogram magnitude scaling
    mag_signal = hist_z.clip(-1, 1) * 15

    raw = impulse_signal + decay_signal + cross_signal + mag_signal

    return _clip_signal(raw, smooth=3)


def s027_connors_rsi(ind):
    """027 | Connors RSI Short-Term Reversal (Larry Connors)
    CRSI = (RSI(3) + PercentRank(streak, 100) + PercentRank(ret_1, 100)) / 3
    Long: CRSI < 10 AND Close > SMA(200) (oversold in uptrend).
    Short: CRSI > 90 AND Close < SMA(200) (overbought in downtrend).
    Exit: CRSI crosses 50, or max 5-bar hold.
    """
    close = ind["close"]
    sma_200 = ind["sma_200"]

    # -- RSI(3) per spec (ultra-short RSI) --
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=3, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=3, adjust=False).mean().replace(0, np.nan)
    rsi_3 = 100 - 100 / (1 + gain / loss)

    # -- Up/Down Streak per spec --
    streak = np.zeros(len(close))
    c = close.values.astype(float)
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            streak[i] = max(streak[i - 1], 0) + 1
        elif c[i] < c[i - 1]:
            streak[i] = min(streak[i - 1], 0) - 1
        else:
            streak[i] = 0
    streak_s = pd.Series(streak, index=close.index)

    # -- PercentRank of streak over 100 bars --
    streak_pctrank = streak_s.rolling(100, min_periods=20).rank(pct=True) * 100

    # -- PercentRank of 1-day return over 100 bars --
    ret1_pctrank = close.pct_change(1).rolling(100, min_periods=20).rank(pct=True) * 100

    # -- Connors RSI per spec --
    crsi = (rsi_3 + streak_pctrank + ret1_pctrank) / 3

    # -- Trend filter per spec (softer: also allow neutral) --
    uptrend = (close > sma_200).astype(float)
    downtrend = (close < sma_200).astype(float)
    trend_factor_buy = uptrend + (1 - uptrend) * (1 - downtrend) * 0.5
    trend_factor_sell = downtrend + (1 - downtrend) * (1 - uptrend) * 0.5

    # -- CRSI zones (widened for actionable frequency) --
    # Was-recently-extreme over lookback window (persistence-based like S024)
    was_os_deep = (crsi < 10).rolling(5, min_periods=1).max()    # was < 10 within 5 bars
    was_os_std = (crsi < 20).rolling(5, min_periods=1).max()     # was < 20 within 5 bars
    was_os_mild = (crsi < 30).rolling(5, min_periods=1).max()    # was < 30 within 5 bars
    was_ob_deep = (crsi > 90).rolling(5, min_periods=1).max()
    was_ob_std = (crsi > 80).rolling(5, min_periods=1).max()
    was_ob_mild = (crsi > 70).rolling(5, min_periods=1).max()

    # Recovery from extreme (CRSI moving back toward 50)
    crsi_slope = crsi - crsi.shift(2)
    recovering_up = (crsi_slope > 0).astype(float)    # bouncing from OS
    recovering_dn = (crsi_slope < 0).astype(float)    # falling from OB

    # -- Continuous CRSI deviation signal (always-on component) --
    crsi_dev = (50 - crsi) / 50  # positive when oversold, negative when overbought
    crsi_dev_smooth = crsi_dev.ewm(span=5, adjust=False).mean()

    # -- Build signal --
    # 1) Deep OS recovery + trend (highest conviction)
    deep_buy = was_os_deep * recovering_up * trend_factor_buy * 55

    # 2) Standard OS recovery + trend
    std_buy = was_os_std * (1 - was_os_deep) * recovering_up * trend_factor_buy * 40

    # 3) Mild OS recovery
    mild_buy = was_os_mild * (1 - was_os_std) * recovering_up * trend_factor_buy * 20

    # 4) Deep OB recovery + downtrend
    deep_sell = was_ob_deep * recovering_dn * trend_factor_sell * 55

    # 5) Standard OB recovery + downtrend
    std_sell = was_ob_std * (1 - was_ob_deep) * recovering_dn * trend_factor_sell * 40

    # 6) Mild OB recovery
    mild_sell = was_ob_mild * (1 - was_ob_std) * recovering_dn * trend_factor_sell * 20

    # 7) Continuous deviation (baseline always-on, weaker)
    cont_signal = crsi_dev_smooth * 15

    raw = deep_buy + std_buy + mild_buy - deep_sell - std_sell - mild_sell + cont_signal

    return _clip_signal(raw, smooth=3)


def s028_elder_triple_screen(ind):
    """028 | Elder Triple Screen System (Dr. Alexander Elder)
    Screen 1 (Tide/Weekly): MACD histogram slope on higher timeframe.
    Screen 2 (Wave/Daily): Stochastic pullback + Force Index.
    Screen 3 (Ripple): Trailing buy/sell stop for entry timing.
    Each screen contributes independently; alignment provides bonus.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]

    # -- Screen 1: Weekly Tide (simulated via 5x daily smoothing) --
    macd_hist = ind["macd_hist"]
    weekly_macd_hist = macd_hist.rolling(5, min_periods=3).mean()
    weekly_hist_slope = weekly_macd_hist - weekly_macd_hist.shift(1)

    # Tide direction: rising weekly histogram = bullish, falling = bearish
    tide_bull = (weekly_hist_slope > 0).astype(float)
    tide_bear = (weekly_hist_slope < 0).astype(float)
    tide_score = tide_bull - tide_bear  # -1 to +1

    # -- Screen 2: Daily Wave --
    stoch_k = ind["stoch_k"]
    stoch_d = ind["stoch_d"]

    # Force Index (2-period EMA) per Elder's spec
    force_raw = close.diff() * ind["volume"]
    force_2 = force_raw.ewm(span=2, adjust=False).mean()
    force_13 = force_raw.ewm(span=13, adjust=False).mean()
    force_z = _z(force_2, 40)

    # Stochastic position and crossovers
    stoch_os = ((stoch_k < 30) & (stoch_k > stoch_k.shift(1))).astype(float)  # OS and turning up
    stoch_ob = ((stoch_k > 70) & (stoch_k < stoch_k.shift(1))).astype(float)  # OB and turning down
    stoch_cross_up = _crossover(stoch_k, stoch_d)
    stoch_cross_dn = _crossunder(stoch_k, stoch_d)

    # Wave score: combines stoch and force
    wave_buy = stoch_os * 0.4 + (force_z < -0.5).astype(float) * 0.3 + stoch_cross_up * 0.3
    wave_sell = stoch_ob * 0.4 + (force_z > 0.5).astype(float) * 0.3 + stoch_cross_dn * 0.3
    wave_score = wave_buy - wave_sell  # -1 to +1

    # -- Screen 3: Entry Timing (trailing stop concept) --
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    break_above = (close > prev_high).astype(float)
    break_below = (close < prev_low).astype(float)
    ripple_score = break_above - break_below  # -1 to +1

    # -- Force Index 13 trend (additional confirmation) --
    force_13_z = _z(force_13, 40).clip(-1, 1)

    # -- Build signal: Each screen contributes independently --
    # Screen 1: Tide (highest weight - weekly direction)
    tide_signal = tide_score * 25

    # Screen 2: Wave (pullback/recovery detection)
    wave_signal = wave_score * 20

    # Screen 3: Ripple (entry timing)
    ripple_signal = ripple_score * 10

    # Force 13 confirmation
    force_signal = force_13_z * 10

    # -- Alignment bonus: when all 3 screens agree, boost conviction --
    all_bull = (tide_score > 0).astype(float) * (wave_score > 0).astype(float) * (ripple_score > 0).astype(float)
    all_bear = (tide_score < 0).astype(float) * (wave_score < 0).astype(float) * (ripple_score < 0).astype(float)
    alignment_bonus = (all_bull - all_bear) * 20

    # -- Counter-trend dampening: wave against tide gets reduced --
    counter_trend = ((tide_score > 0) & (wave_score < 0) | (tide_score < 0) & (wave_score > 0)).astype(float)
    dampener = 1.0 - counter_trend * 0.3

    raw = (tide_signal + wave_signal + ripple_signal + force_signal + alignment_bonus) * dampener

    return _clip_signal(raw, smooth=3)


def s029_aroon_oscillator(ind):
    """029 | Aroon Oscillator Trend Detection (Tushar Chande)
    Aroon_Up(25) = 100 * (25 - bars_since_highest) / 25
    Aroon_Down(25) = 100 * (25 - bars_since_lowest) / 25
    Aroon_Osc = Aroon_Up - Aroon_Down
    Strong_Up: Osc > +70 AND Aroon_Up > 90.
    Emerging_Up: Osc crosses +50 from below.
    Consolidation: |Osc| < 30 for 10+ bars, then breakout.
    """
    high = ind["high"]
    low = ind["low"]
    close = ind["close"]
    n = 25

    # -- Proper Aroon calculation per spec --
    aroon_up = high.rolling(n + 1).apply(lambda x: (n - (n - x.argmax())) / n * 100, raw=True)
    aroon_dn = low.rolling(n + 1).apply(lambda x: (n - (n - x.argmin())) / n * 100, raw=True)
    aroon_osc = aroon_up - aroon_dn

    # -- Phase classification per spec --
    # Strong trend: Osc extreme AND dominant Aroon > 90
    strong_up = ((aroon_osc > 70) & (aroon_up > 90)).astype(float)
    strong_dn = ((aroon_osc < -70) & (aroon_dn > 90)).astype(float)

    # Moderate trend
    mod_up = ((aroon_osc > 50) & (aroon_osc <= 70)).astype(float)
    mod_dn = ((aroon_osc < -50) & (aroon_osc >= -70)).astype(float)

    # Emerging trend: Osc crosses +/-50
    osc_50_line = pd.Series(50.0, index=close.index)
    osc_neg50_line = pd.Series(-50.0, index=close.index)
    emerging_up = _crossover(aroon_osc, osc_50_line)
    emerging_dn = _crossunder(aroon_osc, osc_neg50_line)

    # Zero-line crossover
    zero_line = pd.Series(0.0, index=close.index)
    cross_zero_up = _crossover(aroon_osc, zero_line)
    cross_zero_dn = _crossunder(aroon_osc, zero_line)

    # -- Consolidation detection per spec: |Osc| < 30 for 10+ bars --
    narrow_range = (aroon_osc.abs() < 30).astype(float)
    consol_duration = narrow_range.rolling(10, min_periods=1).sum()
    is_consolidating = (consol_duration >= 10).astype(float)

    # Breakout from consolidation
    was_consolidating = is_consolidating.shift(1).fillna(0)
    breakout_up = was_consolidating * (aroon_osc > 30).astype(float)
    breakout_dn = was_consolidating * (aroon_osc < -30).astype(float)

    # -- Aroon Up/Down divergence (additional conviction) --
    # Both Aroon lines above 70 = strong market, below 30 = no trend
    both_strong = ((aroon_up > 70) & (aroon_dn > 70)).astype(float)
    no_trend = ((aroon_up < 30) & (aroon_dn < 30)).astype(float)
    conviction_dampener = 1.0 - both_strong * 0.3 - no_trend * 0.5

    # -- Build signal --
    # 1) Strong trend (highest conviction)
    strong_signal = strong_up * 40 - strong_dn * 40

    # 2) Moderate trend
    mod_signal = mod_up * 25 - mod_dn * 25

    # 3) Emerging trend (crossover events)
    emerging_signal = emerging_up * 20 - emerging_dn * 20

    # 4) Zero-line crosses
    zero_signal = cross_zero_up * 12 - cross_zero_dn * 12

    # 5) Consolidation breakout (tactical)
    breakout_signal = breakout_up * 25 - breakout_dn * 25

    raw = (strong_signal + mod_signal + emerging_signal + zero_signal + breakout_signal) * conviction_dampener

    return _clip_signal(raw, smooth=3)


def s030_relative_strength_rotation(ind):
    """030 | Relative Strength Rotation (RRG / Mumbai Style)
    RS_Ratio = 100 + (SMA(RelPrice, 10) / SMA(RelPrice, 30) - 1) * 100
    RS_Momentum = 100 + (SMA(RS_Ratio, 10) / SMA(RS_Ratio, 30) - 1) * 100
    Quadrants: Leading (both>100), Weakening (ratio>100,mom<100),
               Lagging (both<100), Improving (ratio<100,mom>100).
    Buy: Improving -> Leading transition. Sell: Weakening -> Lagging.
    """
    close = ind["close"]
    sma_200 = ind["sma_200"]

    # -- Relative price vs own long-term MA (self-relative strength) --
    # Using close/SMA(200) as relative strength proxy for single-asset
    rel_price = close / sma_200.replace(0, np.nan)

    # -- RS_Ratio per spec: ratio of short vs long SMA of relative price --
    rs_sma_short = rel_price.rolling(10, min_periods=5).mean()
    rs_sma_long = rel_price.rolling(30, min_periods=10).mean()
    rs_ratio = 100 + (rs_sma_short / rs_sma_long.replace(0, np.nan) - 1) * 100

    # -- RS_Momentum per spec: ratio of short vs long SMA of RS_Ratio --
    rs_ratio_sma_short = rs_ratio.rolling(10, min_periods=5).mean()
    rs_ratio_sma_long = rs_ratio.rolling(30, min_periods=10).mean()
    rs_momentum = 100 + (rs_ratio_sma_short / rs_ratio_sma_long.replace(0, np.nan) - 1) * 100

    # -- Quadrant classification per spec --
    ratio_above = (rs_ratio > 100).astype(float)
    ratio_below = (rs_ratio <= 100).astype(float)
    mom_above = (rs_momentum > 100).astype(float)
    mom_below = (rs_momentum <= 100).astype(float)

    leading = ratio_above * mom_above        # NE quadrant
    weakening = ratio_above * mom_below       # SE quadrant
    lagging = ratio_below * mom_below         # SW quadrant
    improving = ratio_below * mom_above       # NW quadrant

    # -- Transition detection per spec --
    prev_improving = improving.shift(1).fillna(0)
    prev_weakening = weakening.shift(1).fillna(0)
    prev_lagging = lagging.shift(1).fillna(0)
    prev_leading = leading.shift(1).fillna(0)

    # Key transitions:
    # Improving -> Leading = strongest buy
    imp_to_lead = prev_improving * leading
    # Lagging -> Improving = early buy
    lag_to_imp = prev_lagging * improving
    # Weakening -> Lagging = strongest sell
    weak_to_lag = prev_weakening * lagging
    # Leading -> Weakening = early sell
    lead_to_weak = prev_leading * weakening

    # -- Rotation velocity (distance from 100,100 center) --
    dist_from_center = np.sqrt((rs_ratio - 100) ** 2 + (rs_momentum - 100) ** 2)
    dist_z = _z(dist_from_center, 40).clip(0, 2)

    # -- Within-quadrant momentum (direction of movement) --
    ratio_slope = rs_ratio - rs_ratio.shift(3)
    mom_slope = rs_momentum - rs_momentum.shift(3)

    ratio_improving = (ratio_slope > 0).astype(float)
    mom_improving = (mom_slope > 0).astype(float)

    # -- Build signal --
    # 1) Quadrant base signal
    quad_signal = leading * 25 + improving * 10 - weakening * 10 - lagging * 25

    # 2) Transition events (high conviction)
    trans_buy = imp_to_lead * 35 + lag_to_imp * 20
    trans_sell = weak_to_lag * 35 + lead_to_weak * 20

    # 3) Within-quadrant momentum bonus
    intra_buy = leading * ratio_improving * mom_improving * 10
    intra_sell = lagging * (1 - ratio_improving) * (1 - mom_improving) * 10

    # 4) Distance from center scaling (stronger at extremes)
    dist_bonus = dist_z * 5 * (leading - lagging)

    raw = quad_signal + trans_buy - trans_sell + intra_buy - intra_sell + dist_bonus

    return _clip_signal(raw, smooth=3)


def s031_fibonacci_confluence(ind):
    """031 | Fibonacci Confluence Zone Strategy (Leonardo of Pisa)
    Multiple swing pairs generate Fibonacci retracement levels.
    Confluence = 2+ Fib levels converging within proximity band.
    Long: Price near strong Fib support with bullish context.
    Short: Price near strong Fib resistance with bearish context.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]

    # -- Multiple swing pairs for confluence --
    hi_252 = high.rolling(252, min_periods=60).max()
    lo_252 = low.rolling(252, min_periods=60).min()
    rng_252 = (hi_252 - lo_252).replace(0, np.nan)

    hi_126 = high.rolling(126, min_periods=40).max()
    lo_126 = low.rolling(126, min_periods=40).min()
    rng_126 = (hi_126 - lo_126).replace(0, np.nan)

    hi_63 = high.rolling(63, min_periods=20).max()
    lo_63 = low.rolling(63, min_periods=20).min()
    rng_63 = (hi_63 - lo_63).replace(0, np.nan)

    # -- Fib levels per spec with wider proximity band --
    fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    band = 0.015  # 1.5% band (widened from 0.5% for actionable frequency)

    confluence = pd.Series(0.0, index=close.index)
    # Also track proximity as continuous measure (closer = stronger)
    proximity_score = pd.Series(0.0, index=close.index)

    for ratio in fib_ratios:
        for hi_n, rng_n in [(hi_252, rng_252), (hi_126, rng_126), (hi_63, rng_63)]:
            fib_level = hi_n - ratio * rng_n
            dist_pct = (close - fib_level).abs() / close.replace(0, np.nan)
            near = (dist_pct < band).astype(float)
            confluence = confluence + near
            # Continuous proximity: closer to fib level = higher score
            prox = (1.0 - dist_pct / 0.03).clip(0, 1)  # full score within 0%, decay to 0 at 3%
            proximity_score = proximity_score + prox

    # -- Extensions --
    for hi_n, lo_n, rng_n in [(hi_252, lo_252, rng_252), (hi_126, lo_126, rng_126)]:
        for ext in [1.272, 1.618]:
            ext_level = lo_n + ext * rng_n
            dist_pct = (close - ext_level).abs() / close.replace(0, np.nan)
            near = (dist_pct < band).astype(float)
            confluence = confluence + near
            prox = (1.0 - dist_pct / 0.03).clip(0, 1)
            proximity_score = proximity_score + prox

    # Normalize proximity (19 total fib levels: 5 ratios * 3 timeframes + 2 ext * 2 tf)
    proximity_norm = (proximity_score / 10).clip(0, 2)

    # -- Zone classification --
    strong_zone = (confluence >= 3).astype(float)
    mod_zone = ((confluence >= 2) & (confluence < 3)).astype(float)
    any_zone = (confluence >= 1).astype(float)

    # -- Position relative to Fib midpoint --
    fib_500_252 = hi_252 - 0.500 * rng_252
    above_mid = (close > fib_500_252).astype(float)
    below_mid = 1.0 - above_mid

    # -- Reversal candle detection --
    body = (close - ind["open"]).abs()
    lower_wick = pd.concat([close, ind["open"]], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([close, ind["open"]], axis=1).max(axis=1)

    hammer = ((lower_wick > 1.5 * body) & (close > ind["open"])).astype(float)
    star = ((upper_wick > 1.5 * body) & (close < ind["open"])).astype(float)
    reversal_bull = hammer.rolling(3, min_periods=1).max()
    reversal_bear = star.rolling(3, min_periods=1).max()

    # -- Momentum context --
    mom_bull = (ind["mom_10"] > 0).astype(float)
    mom_bear = (ind["mom_10"] < 0).astype(float)
    trend_ctx = ind["trend_score"]

    # -- Build signal --
    # 1) Strong confluence + context
    strong_buy = strong_zone * (reversal_bull * 0.5 + mom_bull * 0.3 + 0.2) * 55
    strong_sell = strong_zone * (reversal_bear * 0.5 + mom_bear * 0.3 + 0.2) * 55

    # 2) Moderate confluence (doesn't require below/above mid)
    mod_buy = mod_zone * (reversal_bull * 0.3 + mom_bull * 0.4 + 0.3) * 35
    mod_sell = mod_zone * (reversal_bear * 0.3 + mom_bear * 0.4 + 0.3) * 35

    # Direction from midpoint (which confluence is support vs resistance)
    direction = below_mid * 2 - 1  # +1 if below mid (support), -1 if above (resistance)

    # 3) Retracement depth signal (always-on, where price sits in Fib range)
    fib_position = (close - lo_252) / rng_252  # 0 at low, 1 at high
    # Golden zone: 0.382-0.618 retracement = strong support in uptrend
    in_golden = ((fib_position > 0.382) & (fib_position < 0.618)).astype(float)
    depth_signal = (0.5 - fib_position).clip(-0.5, 0.5) * 40  # ranges -20 to +20

    # 4) Bounce from recent Fib level (was near in last 5 bars, now moving)
    was_near_fib = any_zone.rolling(5, min_periods=1).max()
    bouncing_up = was_near_fib * (close > close.shift(2)).astype(float) * below_mid
    bouncing_dn = was_near_fib * (close < close.shift(2)).astype(float) * above_mid

    bounce_signal = bouncing_up * 25 - bouncing_dn * 25

    # 5) Golden zone + momentum alignment
    golden_buy = in_golden * mom_bull * 15
    golden_sell = in_golden * mom_bear * 15

    raw = ((strong_buy + mod_buy + golden_buy) * direction.clip(0, 1)
           - (strong_sell + mod_sell + golden_sell) * (-direction).clip(0, 1)
           + depth_signal + bounce_signal)

    return _clip_signal(raw, smooth=3)


def s032_momentum_carry(ind):
    """032 | Momentum Carry Strategy (Singapore Dollar style, OHLCV proxy)
    Composite signal combining:
    1. Policy momentum proxy: rolling regression slope of smoothed price (band slope)
    2. Carry proxy: inverse vol (low vol = positive carry environment)
    3. Price momentum: 60-day return (trend following component)
    Weights: 50% policy momentum, 30% carry, 20% price momentum per spec.
    """
    close = ind["close"]
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]

    # -- Policy Momentum Proxy: rolling regression slope of smoothed price --
    # Simulates the MAS band slope estimate from 3-month rolling regression
    smooth_price = close.ewm(span=10, adjust=False).mean()
    # Rolling 63-bar (3 month) regression slope approximated via change
    price_slope_63 = (smooth_price - smooth_price.shift(63)) / smooth_price.shift(63).replace(0, np.nan)
    policy_mom = _z(price_slope_63, 60)

    # -- Carry Proxy: low volatility = favorable carry environment --
    # In FX, carry works best when vol is low and stable
    vol_ratio = vol_20 / vol_60.replace(0, np.nan)
    vol_level_z = _z(vol_20, 120)  # long-term vol percentile
    carry_signal = -vol_level_z  # low vol = positive carry
    # Stable vol (ratio near 1) enhances carry
    vol_stability = 1.0 - (vol_ratio - 1.0).abs().clip(0, 1)

    carry_composite = carry_signal * (0.6 + vol_stability * 0.4)

    # -- Price Momentum: 60-day returns --
    mom_60 = close.pct_change(60) / vol_60.replace(0, np.nan)  # risk-adjusted
    mom_z = _z(mom_60, 60)

    # -- MAS policy meeting proxy (April/October = ~trading days 80, 210) --
    # Semi-annual cycle: detect periods when trend alignment strengthens
    mom_20 = close.pct_change(20) / vol_20.replace(0, np.nan)
    mom_align = (np.sign(mom_60) == np.sign(mom_20)).astype(float)

    # -- Composite per spec: 50/30/20 weighting --
    raw_composite = (0.50 * policy_mom.clip(-2, 2) +
                     0.30 * carry_composite.clip(-2, 2) +
                     0.20 * mom_z.clip(-2, 2))

    # -- Threshold per spec: Composite > +0.5 = long, < -0.5 = short --
    strong_long = (raw_composite > 1.0).astype(float)
    mod_long = ((raw_composite > 0.5) & (raw_composite <= 1.0)).astype(float)
    strong_short = (raw_composite < -1.0).astype(float)
    mod_short = ((raw_composite < -0.5) & (raw_composite >= -1.0)).astype(float)

    # Momentum alignment bonus
    align_bonus = mom_align * 0.2

    # -- Build signal --
    discrete = (strong_long * 45 + mod_long * 30 - strong_short * 45 - mod_short * 30)
    continuous = raw_composite.clip(-2, 2) * 15  # baseline always-on
    alignment = (strong_long + mod_long - strong_short - mod_short) * align_bonus * 15

    raw = discrete + continuous + alignment

    return _clip_signal(raw, smooth=5)


def s033_chaikin_money_flow(ind):
    """033 | Chaikin Money Flow Accumulation (Marc Chaikin)
    CLV = ((Close - Low) - (High - Close)) / (High - Low), ranges [-1, +1]
    CMF(21) = sum(CLV * Volume, 21) / sum(Volume, 21)
    CMF_Trend = EMA(CMF, 5) - EMA(CMF, 13) (money flow acceleration)
    Divergence: Price new high + CMF < 0 = bearish; Price new low + CMF > 0 = bullish.
    Long: CMF > +0.10 AND CMF_Trend > 0. Short: CMF < -0.10 AND CMF_Trend < 0.
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]

    # -- Close Location Value per spec --
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range

    # -- CMF(21) per spec --
    cmf_21 = (clv * volume).rolling(21, min_periods=10).sum() / volume.rolling(21, min_periods=10).sum().replace(0, np.nan)

    # -- CMF Trend per spec: EMA(5) - EMA(13) acceleration --
    cmf_fast = cmf_21.ewm(span=5, adjust=False).mean()
    cmf_slow = cmf_21.ewm(span=13, adjust=False).mean()
    cmf_trend = cmf_fast - cmf_slow

    # -- Divergence detection per spec --
    # Price new 20-day high but CMF < 0 = bearish divergence
    price_hh = (close >= close.rolling(20, min_periods=10).max()).astype(float)
    price_ll = (close <= close.rolling(20, min_periods=10).min()).astype(float)

    bear_div = price_hh * (cmf_21 < 0).astype(float)
    bull_div = price_ll * (cmf_21 > 0).astype(float)

    # Persistent divergence (was divergent in last 5 bars)
    bear_div_persist = bear_div.rolling(5, min_periods=1).max()
    bull_div_persist = bull_div.rolling(5, min_periods=1).max()

    # -- Zone classification per spec --
    strong_accum = (cmf_21 > 0.15).astype(float)
    accum = ((cmf_21 > 0.10) & (cmf_21 <= 0.15)).astype(float)
    mild_accum = ((cmf_21 > 0.05) & (cmf_21 <= 0.10)).astype(float)

    strong_distrib = (cmf_21 < -0.15).astype(float)
    distrib = ((cmf_21 < -0.10) & (cmf_21 >= -0.15)).astype(float)
    mild_distrib = ((cmf_21 < -0.05) & (cmf_21 >= -0.10)).astype(float)

    # -- CMF trend direction --
    trend_bull = (cmf_trend > 0).astype(float)
    trend_bear = (cmf_trend < 0).astype(float)
    trend_strength = _z(cmf_trend.abs(), 40).clip(0, 1.5)

    # -- Build signal --
    # 1) Strong accumulation + rising trend (per spec: CMF > 0.10 AND CMF_Trend > 0)
    buy_strong = strong_accum * trend_bull * (40 + trend_strength * 10)
    buy_std = accum * trend_bull * (30 + trend_strength * 5)
    buy_mild = mild_accum * trend_bull * 15

    # 2) Strong distribution + falling trend
    sell_strong = strong_distrib * trend_bear * (40 + trend_strength * 10)
    sell_std = distrib * trend_bear * (30 + trend_strength * 5)
    sell_mild = mild_distrib * trend_bear * 15

    # 3) Divergence signals (reversal warning per spec)
    div_buy = bull_div_persist * 20
    div_sell = bear_div_persist * 20

    # 4) Continuous CMF signal (always-on baseline)
    cmf_z = _z(cmf_21, 40)
    continuous = cmf_z.clip(-1.5, 1.5) * 10

    raw = (buy_strong + buy_std + buy_mild + div_buy
           - sell_strong - sell_std - sell_mild - div_sell
           + continuous)

    return _clip_signal(raw, smooth=3)


def s034_yield_curve_proxy(ind):
    """034 | Yield Curve Steepener Signal (Frankfurt style, vol regime proxy)
    Since we don't have bond yield data, we proxy the curve slope concept:
    - Curve_Slope proxy: short vol vs long vol ratio (term structure of vol)
    - Curve_Momentum: rate of change of the slope
    - Curve_Acceleration: second derivative (per spec)
    - Policy proxy: trend score (hawkish = strong trend, dovish = range)
    Z-score approach to identify extreme dislocations per spec.
    """
    close = ind["close"]
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]

    # -- Curve Slope Proxy: vol term structure --
    # Short vol / long vol ratio mimics the 2Y/10Y spread behavior
    vol_ratio = vol_20 / vol_60.replace(0, np.nan)
    curve_slope = vol_ratio - 1.0  # centered at 0: positive = inverted (short > long)

    # -- Z-score of slope per spec --
    curve_slope_z = _z(curve_slope, 60)

    # -- Curve Momentum per spec: 5-bar change in slope --
    curve_momentum = curve_slope - curve_slope.shift(5)
    curve_mom_z = _z(curve_momentum, 40)

    # -- Curve Acceleration per spec: second derivative --
    curve_accel = curve_momentum - curve_momentum.shift(5)
    curve_accel_z = _z(curve_accel, 40)

    # -- Policy Proxy: trend score as ECB/monetary policy stand-in --
    trend = ind["trend_score"]
    policy_dovish = (trend < -0.3).astype(float)
    policy_hawkish = (trend > 0.3).astype(float)

    # -- Real rate proxy: momentum-adjusted vol --
    # Rising prices with falling vol = positive real rate environment
    real_rate_proxy = ind["mom_20"] - _z(vol_20, 60)
    real_rate_trend = real_rate_proxy.ewm(span=20, adjust=False).mean() - real_rate_proxy.ewm(span=60, adjust=False).mean()

    # -- Steepener signal per spec: slope too flat + starting to steepen + dovish --
    # Curve_Slope_Z < -1.5 AND Acceleration > 0 AND policy dovish
    steep_setup = ((curve_slope_z < -1.5) & (curve_accel > 0)).astype(float)
    steep_buy = steep_setup * (1.0 + policy_dovish * 0.5)

    # -- Flattener signal per spec: slope too steep + starting to flatten + hawkish --
    flat_setup = ((curve_slope_z > 1.5) & (curve_accel < 0)).astype(float)
    flat_sell = flat_setup * (1.0 + policy_hawkish * 0.5)

    # -- Moderate signals at lower thresholds --
    mod_steep = ((curve_slope_z < -1.0) & (curve_slope_z >= -1.5) & (curve_accel > 0)).astype(float)
    mod_flat = ((curve_slope_z > 1.0) & (curve_slope_z <= 1.5) & (curve_accel < 0)).astype(float)

    # -- Mean reversion of slope z-score (always-on, targeting z=0) --
    mean_rev = -curve_slope_z.clip(-2, 2) * 10

    # -- Build signal --
    # 1) Extreme dislocations (per spec: z > 1.5)
    extreme = steep_buy * 45 - flat_sell * 45

    # 2) Moderate dislocations
    moderate = mod_steep * 25 - mod_flat * 25

    # 3) Real rate trend context
    real_bonus = real_rate_trend.clip(-1, 1) * 10

    # 4) Continuous mean-reversion component
    raw = extreme + moderate + mean_rev + real_bonus

    return _clip_signal(raw, smooth=3)


def s035_weis_wave_volume(ind):
    """035 | Weis Wave Volume Analysis (David Weis / Wyckoff)
    Wave_Direction: determined by close-to-close direction changes.
    Wave_Volume: accumulated volume within each directional wave.
    Effort_vs_Result: volume per unit price move.
    Increasing effort + decreasing result = absorption (distribution at top).
    Decreasing effort + increasing result = exhaustion (accumulation at bottom).
    Long: Selling exhaustion followed by increasing volume on up wave.
    Short: Buying absorption followed by volume decrease on down wave.
    """
    close = ind["close"]
    volume = ind["volume"]
    high = ind["high"]
    low = ind["low"]

    c = close.values.astype(float)
    v = volume.values.astype(float)
    h = high.values.astype(float)
    l = low.values.astype(float)
    n = len(c)

    # -- Wave segmentation per spec --
    # Track direction changes and accumulate volume within each wave
    wave_dir = np.zeros(n)       # +1 up, -1 down
    wave_vol = np.zeros(n)       # accumulated vol in current wave
    wave_range = np.zeros(n)     # price range of current wave
    wave_start_price = np.zeros(n)

    # Previous wave stats for comparison
    prev_up_vol = np.zeros(n)
    prev_dn_vol = np.zeros(n)
    prev_up_range = np.zeros(n)
    prev_dn_range = np.zeros(n)

    curr_dir = 0.0
    curr_vol = 0.0
    curr_hi = c[0] if n > 0 else 0.0
    curr_lo = c[0] if n > 0 else 0.0
    last_up_vol = 0.0
    last_dn_vol = 0.0
    last_up_range = 0.0
    last_dn_range = 0.0

    for i in range(1, n):
        new_dir = 1.0 if c[i] > c[i - 1] else (-1.0 if c[i] < c[i - 1] else curr_dir)

        if new_dir == curr_dir or curr_dir == 0.0:
            # Continue current wave
            curr_vol += v[i]
            curr_hi = max(curr_hi, h[i])
            curr_lo = min(curr_lo, l[i])
        else:
            # Wave changed direction - save previous wave stats
            wave_rng = curr_hi - curr_lo
            if curr_dir > 0:
                last_up_vol = curr_vol
                last_up_range = wave_rng
            elif curr_dir < 0:
                last_dn_vol = curr_vol
                last_dn_range = wave_rng

            # Start new wave
            curr_vol = v[i]
            curr_hi = h[i]
            curr_lo = l[i]

        curr_dir = new_dir
        wave_dir[i] = curr_dir
        wave_vol[i] = curr_vol
        wave_range[i] = curr_hi - curr_lo
        prev_up_vol[i] = last_up_vol
        prev_dn_vol[i] = last_dn_vol
        prev_up_range[i] = last_up_range
        prev_dn_range[i] = last_dn_range

    wave_dir_s = pd.Series(wave_dir, index=close.index)
    wave_vol_s = pd.Series(wave_vol, index=close.index)
    wave_range_s = pd.Series(wave_range, index=close.index)
    prev_up_vol_s = pd.Series(prev_up_vol, index=close.index)
    prev_dn_vol_s = pd.Series(prev_dn_vol, index=close.index)
    prev_up_range_s = pd.Series(prev_up_range, index=close.index)
    prev_dn_range_s = pd.Series(prev_dn_range, index=close.index)

    # -- Effort vs Result per spec --
    # Effort = volume, Result = price range
    effort_result = wave_vol_s / wave_range_s.replace(0, np.nan)
    effort_z = _z(effort_result, 40)

    # -- Key Wyckoff patterns per spec --
    is_up_wave = (wave_dir_s > 0).astype(float)
    is_dn_wave = (wave_dir_s < 0).astype(float)

    # Buying absorption: increasing effort on up wave + decreasing result
    # (volume rising but price range shrinking = buyers being absorbed at top)
    buy_absorb = is_up_wave * (wave_vol_s > prev_up_vol_s * 1.1).astype(float) * \
                 (wave_range_s < prev_up_range_s * 0.9).astype(float)

    # Selling exhaustion: decreasing effort on down wave + result maintaining
    # (volume falling but price still moving = sellers drying up)
    sell_exhaust = is_dn_wave * (wave_vol_s < prev_dn_vol_s * 0.9).astype(float) * \
                   (wave_range_s > prev_dn_range_s * 0.5).astype(float)

    # -- Volume confirmation per spec --
    # Strong up wave volume vs down wave volume
    vol_up_vs_dn = (prev_up_vol_s / prev_dn_vol_s.replace(0, np.nan)).fillna(1.0)
    vol_bias = _z(vol_up_vs_dn, 40).clip(-2, 2)

    # -- Wave direction change detection (reversal point) --
    dir_change = (wave_dir_s != wave_dir_s.shift(1)).astype(float)
    new_up_wave = dir_change * is_up_wave
    new_dn_wave = dir_change * is_dn_wave

    # -- Build signal per spec --
    # 1) Selling exhaustion + new up wave = strongest buy
    exhaust_buy = sell_exhaust.rolling(3, min_periods=1).max() * new_up_wave * 40

    # 2) Buying absorption + new down wave = strongest sell
    absorb_sell = buy_absorb.rolling(3, min_periods=1).max() * new_dn_wave * 40

    # 3) Volume bias (up vol > down vol = accumulation)
    bias_signal = vol_bias * 15

    # 4) Effort-result divergence continuous
    effort_signal = -effort_z.clip(-1.5, 1.5) * is_up_wave * 8 + effort_z.clip(-1.5, 1.5) * is_dn_wave * 8

    # 5) Wave direction base
    dir_base = wave_dir_s * 10

    raw = exhaust_buy - absorb_sell + bias_signal + effort_signal + dir_base

    return _clip_signal(raw, smooth=3)


def s036_elder_force_index(ind):
    """036 | Elder Force Index Impulse System

    Alexander Elder's Force Index + Impulse System.
    Force_Index = Close_diff * Volume  (directional force measurement).
    EFI_2  = EMA(FI, 2)   -- short-term force for timing
    EFI_13 = EMA(FI, 13)  -- intermediate force for trend confirmation

    Impulse System (Green / Red / Blue bars):
      Green: EMA_13_slope > 0 AND MACD_Hist_slope > 0  (bulls in control)
      Red:   EMA_13_slope < 0 AND MACD_Hist_slope < 0  (bears in control)
      Blue:  mixed signals -- stay out

    Force Confirmation:
      Strong_Bull: EFI_2 > 0 AND EFI_13 > 0 AND EFI_13 rising
      Strong_Bear: EFI_2 < 0 AND EFI_13 < 0 AND EFI_13 falling

    Signal:
      Long : Green bar AND Strong_Bull_Force
      Short: Red bar   AND Strong_Bear_Force
      Exit : First Blue bar after streak
    """
    close = ind["close"]
    volume = ind["volume"]

    # --- Force Index ---
    force_index = close.diff() * volume
    efi_2 = force_index.ewm(span=2, adjust=False).mean()
    efi_13 = force_index.ewm(span=13, adjust=False).mean()
    efi_13_slope = efi_13.diff()

    # --- Impulse System ---
    ema13 = close.ewm(span=13, adjust=False).mean()
    ema13_slope = ema13.diff()
    macd_hist = ind["macd_hist"]
    macd_hist_slope = macd_hist.diff()

    green_bar = (ema13_slope > 0) & (macd_hist_slope > 0)
    red_bar = (ema13_slope < 0) & (macd_hist_slope < 0)

    # --- Force Confirmation ---
    strong_bull = (efi_2 > 0) & (efi_13 > 0) & (efi_13_slope > 0)
    strong_bear = (efi_2 < 0) & (efi_13 < 0) & (efi_13_slope < 0)

    # --- Impulse + Force alignment ---
    bull_impulse = (green_bar & strong_bull).astype(float)
    bear_impulse = (red_bar & strong_bear).astype(float)

    # Impulse streak: consecutive Green/Red bars (persistence)
    green_streak = green_bar.astype(float)
    green_streak = green_streak.rolling(5, min_periods=1).sum()  # how many of last 5 bars green
    red_streak = red_bar.astype(float)
    red_streak = red_streak.rolling(5, min_periods=1).sum()

    # Blue bar = recent streak breaking (exit signal)
    was_green = green_streak.shift(1) >= 3
    was_red = red_streak.shift(1) >= 3
    blue_now = (~green_bar) & (~red_bar)
    exit_from_long = (was_green & blue_now).astype(float)
    exit_from_short = (was_red & blue_now).astype(float)

    # --- Continuous force z-score for baseline ---
    efi_13_z = _z(efi_13, 40)

    # --- Composite signal ---
    # Strong impulse events: +/- 55 points
    # Streak quality bonus: up to +/- 15
    # Continuous force z-score: +/- 20 baseline
    # Exit events dampen toward zero
    impulse_signal = bull_impulse * 55 - bear_impulse * 55
    streak_bonus = (green_streak / 5).clip(0, 1) * 15 - (red_streak / 5).clip(0, 1) * 15
    force_base = efi_13_z.clip(-2, 2) / 2 * 20

    raw = impulse_signal + streak_bonus + force_base
    # Exit: blue bars after green streak pull toward zero
    raw = raw - exit_from_long * 40 + exit_from_short * 40

    return _clip_signal(raw)


def s037_vol_regime_switch(ind):
    """037 | Zurich Volatility Regime Switch

    Swiss Re / ETH Zurich style volatility regime allocation.
    RV_fast = sqrt(252 * EMA(ret^2, 10))   -- 10-day annualized realized vol
    RV_slow = sqrt(252 * EMA(ret^2, 60))   -- 60-day annualized realized vol
    Vol_Ratio = RV_fast / RV_slow

    Regime Classification (HMM proxy via thresholds):
      Low_Vol:  Vol_Ratio < 0.8 AND RV_fast < median(RV_fast, 252)
      Normal:   0.8 <= Vol_Ratio <= 1.3
      High_Vol: Vol_Ratio > 1.3 OR RV_fast > 90th pctile
      Crisis:   RV_fast > 2 * RV_slow AND RV_fast > 80th pctile

    Strategy Allocation per Regime:
      Low_Vol:  Trend following (60%) + Carry (30%) + MR (10%)
      Normal:   Trend (40%) + Carry (20%) + MR (40%)
      High_Vol: Mean reversion (70%) + Defensive (30%)
      Crisis:   Defensive (80%) + vol-of-vol short (20%)
    """
    close = ind["close"]
    ret = close.pct_change(1)
    ret_sq = ret ** 2

    # --- Realized Volatility (annualized) ---
    rv_fast = np.sqrt(252 * ret_sq.ewm(span=10, adjust=False).mean())
    rv_slow = np.sqrt(252 * ret_sq.ewm(span=60, adjust=False).mean())
    vol_ratio = (rv_fast / rv_slow.replace(0, np.nan)).fillna(1.0)

    # --- Regime percentiles ---
    rv_median = rv_fast.rolling(252, min_periods=60).median()
    rv_p90 = rv_fast.rolling(252, min_periods=60).quantile(0.90)
    rv_p80 = rv_fast.rolling(252, min_periods=60).quantile(0.80)

    # --- Regime Classification ---
    crisis = (rv_fast > 2 * rv_slow) & (rv_fast > rv_p80)
    high_vol = (~crisis) & ((vol_ratio > 1.3) | (rv_fast > rv_p90))
    low_vol = (~crisis) & (~high_vol) & (vol_ratio < 0.8) & (rv_fast < rv_median)
    normal = (~crisis) & (~high_vol) & (~low_vol)

    # --- Per-Regime Dominant Signals (not fractional allocation) ---
    mom = ind["mom_score"]
    trend_sig = mom * 100
    rsi_mr = (50 - ind["rsi"]) / 50 * 80
    carry_bias = (ind["above_200"].astype(float) * 2 - 1) * 25

    # Low_Vol: trend following dominates + carry base
    low_vol_sig = trend_sig + carry_bias
    # Normal: balanced blend at full scale
    normal_sig = 0.55 * trend_sig + 0.45 * rsi_mr + carry_bias * 0.5
    # High_Vol: mean reversion dominates
    high_vol_sig = rsi_mr + carry_bias * 0.3
    # Crisis: reduced mean reversion only
    crisis_sig = rsi_mr * 0.5

    raw = (low_vol.astype(float) * low_vol_sig
           + normal.astype(float) * normal_sig
           + high_vol.astype(float) * high_vol_sig
           + crisis.astype(float) * crisis_sig)

    # Regime transition boost
    vol_ratio_chg = vol_ratio.diff().abs()
    transition_z = _z(vol_ratio_chg, 40).clip(0, 2)
    raw = raw * (1 + transition_z * 0.2)

    return _clip_signal(raw)


def s038_opening_gap(ind):
    """038 | Hong Kong HSI Opening Gap Strategy
    School: Hong Kong (HKEX Proprietary)

    Gap = Open - Close_prev.  Gap_Size_Z = Gap_Pct / StdDev(Gap_Pct, 60).
    Fill_Probability = logistic(Gap_Size_Z).
    Small gaps (fill_prob > 0.65): fade gap direction.
    Large gaps (fill_prob < 0.35, Z > 2): trade WITH gap if momentum confirms.
    """
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    prev_close = close.shift(1)
    gap = open_p - prev_close
    gap_pct = gap / prev_close.replace(0, np.nan) * 100
    gap_std = gap_pct.rolling(60, min_periods=20).std().replace(0, np.nan)
    gap_z = (gap_pct / gap_std).fillna(0)

    # --- Fill Probability (logistic proxy) ---
    # Higher |gap_z| -> lower fill probability
    # fill_prob = 1 / (1 + exp(1.5 * (|gap_z| - 1.0)))
    fill_prob = 1 / (1 + np.exp(1.5 * (gap_z.abs() - 1.0)))

    # --- Gap Fill Detection (daily proxy) ---
    # Did the gap fill within the day? (price touched prev_close level)
    gap_up = gap > 0
    gap_dn = gap < 0
    gap_filled = ((gap_up & (low <= prev_close)) | (gap_dn & (high >= prev_close))).astype(float)

    # Partial fill: how much of the gap was closed by end of day
    gap_remaining = (close - prev_close) / gap.replace(0, np.nan)
    gap_remaining = gap_remaining.clip(-1, 2).fillna(0)  # 0 = fully filled, 1 = not at all

    # --- Fade Signal (small gaps, high fill prob) ---
    # Fade = enter against gap direction when fill is likely
    fade_strength = fill_prob.clip(0.3, 1)
    fade_signal = -np.sign(gap_pct) * fade_strength * 55

    # --- Continuation Signal (large gaps, low fill prob + momentum) ---
    mom_confirm = ind["mom_5"]
    cont_strength = (1 - fill_prob).clip(0, 1) * (gap_z.abs() > 1.0).astype(float)
    # Only continue if momentum agrees with gap direction
    mom_agrees = (np.sign(gap_pct) * np.sign(mom_confirm) > 0).astype(float)
    cont_signal = np.sign(gap_pct) * cont_strength * (0.5 + 0.5 * mom_agrees) * 60

    # --- Gap Fill Feedback (previous gaps that filled = reinforcement) ---
    fill_rate = gap_filled.rolling(20, min_periods=5).mean().fillna(0.5)
    fade_confidence = (fill_rate - 0.4) * 2.5  # -1 to +1.5, shifted so fading is default

    # --- Always-on gap z-score baseline ---
    # Even on non-gap days, gap_z captures overnight sentiment
    gap_baseline = -gap_z.clip(-2.5, 2.5) / 2.5 * 25  # fade bias baseline

    # --- Composite ---
    gap_event = fade_signal * (0.6 + 0.4 * fade_confidence.clip(-1, 1)) + cont_signal
    # Use gap events when gap is significant, otherwise use baseline
    has_gap = (gap_pct.abs() > gap_pct.abs().rolling(20, min_periods=5).quantile(0.3)).astype(float)
    raw = has_gap * gap_event + (1 - has_gap) * gap_baseline

    # Volume confirmation: larger volume = more conviction (mild boost)
    vol_surge = (ind["volume"] / ind["volume"].rolling(20).mean().replace(0, np.nan)).fillna(1)
    vol_mult = vol_surge.clip(0.7, 1.5)
    raw = raw * vol_mult

    return _clip_signal(raw)


def s039_anchored_vwap(ind):
    """039 | Anchored VWAP Institutional Level Strategy

    Brian Shannon's Anchored VWAP -- institutional reference levels.

    VWAP(anchor) = sum(Price * Volume, from anchor) / sum(Volume, from anchor)

    Anchor Points (proxy since no earnings dates):
      Quarterly_VWAP: anchored every ~63 trading days (earnings proxy)
      High_VWAP: anchored at 52-week high date
      Low_VWAP: anchored at 52-week low date
      Month_VWAP: anchored at first trading day of month

    Signal:
      Long : price pulls back to VWAP from above (institutions defend avg entry)
      Short: price rallies to VWAP from below and fails (institutions sell at breakeven)
      Breakout: crosses VWAP with volume > 2x average -> confirms new trend
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    n = len(close)
    typical_price = (high + low + close) / 3
    pv = typical_price * volume

    # --- Monthly VWAP (anchored at month boundaries) ---
    # Detect month changes via index
    idx = close.index
    month_change = pd.Series(False, index=idx)
    if hasattr(idx, 'month'):
        month_change = pd.Series(idx.month, index=idx).diff().fillna(0) != 0
    else:
        # Fallback: approximate every 21 bars
        counter = pd.Series(range(n), index=idx)
        month_change = (counter % 21 == 0)
    month_change.iloc[0] = True  # first bar is always an anchor

    # Compute anchored VWAP from last month boundary
    cum_pv = pd.Series(0.0, index=idx)
    cum_vol = pd.Series(0.0, index=idx)
    for i in range(n):
        if month_change.iloc[i]:
            cum_pv.iloc[i] = pv.iloc[i]
            cum_vol.iloc[i] = volume.iloc[i]
        else:
            cum_pv.iloc[i] = cum_pv.iloc[i-1] + pv.iloc[i]
            cum_vol.iloc[i] = cum_vol.iloc[i-1] + volume.iloc[i]
    month_vwap = cum_pv / cum_vol.replace(0, np.nan)

    # --- Quarterly VWAP (earnings proxy, ~63 bars) ---
    cum_pv_q = pd.Series(0.0, index=idx)
    cum_vol_q = pd.Series(0.0, index=idx)
    counter_arr = np.arange(n)
    qtr_anchor = (counter_arr % 63 == 0)
    for i in range(n):
        if qtr_anchor[i]:
            cum_pv_q.iloc[i] = pv.iloc[i]
            cum_vol_q.iloc[i] = volume.iloc[i]
        else:
            cum_pv_q.iloc[i] = cum_pv_q.iloc[i-1] + pv.iloc[i]
            cum_vol_q.iloc[i] = cum_vol_q.iloc[i-1] + volume.iloc[i]
    qtr_vwap = cum_pv_q / cum_vol_q.replace(0, np.nan)

    # --- Distance from VWAPs (normalized by ATR) ---
    atr = ind["atr14"].replace(0, np.nan)
    dist_month = (close - month_vwap) / atr
    dist_qtr = (close - qtr_vwap) / atr

    # --- Pullback-to-VWAP Detection ---
    # Price near VWAP = within 0.5 ATR
    near_month = (dist_month.abs() < 0.5).astype(float)
    near_qtr = (dist_qtr.abs() < 0.5).astype(float)

    # Was above/below before touching VWAP (pullback direction)
    was_above_month = (dist_month.shift(3) > 1.0).astype(float)
    was_below_month = (dist_month.shift(3) < -1.0).astype(float)
    was_above_qtr = (dist_qtr.shift(3) > 1.0).astype(float)
    was_below_qtr = (dist_qtr.shift(3) < -1.0).astype(float)

    # Pullback long: was above, came back to VWAP (institutions defend)
    pullback_long = near_month * was_above_month + near_qtr * was_above_qtr
    # Pullback short: was below, rallied to VWAP (institutions sell breakeven)
    pullback_short = near_month * was_below_month + near_qtr * was_below_qtr

    # --- Breakout Confirmation ---
    vol_avg = volume.rolling(20).mean().replace(0, np.nan)
    vol_surge = (volume / vol_avg).fillna(1)
    breakout_vol = (vol_surge > 2.0).astype(float)

    # Breakout above VWAP with volume
    break_above = ((dist_month > 0.5) & (dist_month.shift(1) < 0.5)).astype(float) * breakout_vol
    break_below = ((dist_month < -0.5) & (dist_month.shift(1) > -0.5)).astype(float) * breakout_vol

    # --- VWAP Slope (trend of institutional level) ---
    vwap_slope = _z(month_vwap.pct_change(5), 40)

    # --- Composite Signal ---
    pullback_sig = pullback_long * 35 - pullback_short * 35
    breakout_sig = break_above * 40 - break_below * 40
    trend_bias = vwap_slope.clip(-2, 2) / 2 * 15
    # Continuous: distance from VWAP as mild directional signal
    dist_sig = dist_month.clip(-3, 3) / 3 * 10

    raw = pullback_sig + breakout_sig + trend_bias + dist_sig

    return _clip_signal(raw)


def s040_momentum_carry_br(ind):
    """040 | Brazilian Real Momentum-Carry Hybrid (OHLCV proxy)

    Itau BBA Sao Paulo quant desk style EM FX composite.
    Original factors:
      Carry_Z   = (Selic - FFR) z-scored over 252 days
      Momentum  = ret(60d) / vol(60d)  (risk-adjusted)
      CDS_Signal = -normalize(5Y CDS)  (credit risk)
      Terms_of_Trade = normalize(Iron_Ore + Soybean + Coffee)

    Composite = 0.30*Carry_Z + 0.25*Momentum + 0.25*(-CDS) + 0.20*ToT
    Long BRL: Composite > +0.8 AND Carry > 5%
    Short BRL: Composite < -0.8 OR CDS > 300bps

    OHLCV Proxy:
      Carry -> above long-term MA (persistent uptrend = positive carry analog)
      Momentum -> risk-adjusted 60-day return
      CDS -> realized vol z-score inverted (high vol = high credit risk)
      ToT -> relative strength vs SMA_200 (export competitiveness proxy)
    """
    close = ind["close"]

    # --- Carry Proxy ---
    # Persistent above-200 SMA = positive carry environment
    # Distance above SMA_200 normalized as carry attractiveness
    sma200 = ind["sma_200"]
    carry_raw = (close - sma200) / sma200.replace(0, np.nan)
    carry_z = _z(carry_raw, 252)
    # Carry regime: only attractive when clearly positive
    carry_positive = (carry_z > 0).astype(float)

    # --- Momentum (risk-adjusted 60-day) ---
    ret_60 = close.pct_change(60)
    vol_60 = ind["vol_60"].replace(0, np.nan)
    momentum = (ret_60 / vol_60).fillna(0)
    momentum_z = _z(momentum, 120)

    # --- CDS Proxy (credit risk = inverted volatility z-score) ---
    # High realized vol = high credit risk (negative for the asset)
    vol_20 = ind["vol_20"]
    cds_proxy = -_z(vol_20, 252)  # inverted: high vol = negative signal

    # --- Terms of Trade Proxy (export competitiveness) ---
    # Use relative strength of recent vs long-term price as commodity demand proxy
    sma50 = ind["sma_50"]
    sma20 = ind["sma_20"]
    tot_raw = (sma20 - sma50) / sma50.replace(0, np.nan)
    tot_z = _z(tot_raw, 120)

    # --- Multi-Factor Composite ---
    composite = (0.30 * carry_z.clip(-3, 3)
                 + 0.25 * momentum_z.clip(-3, 3)
                 + 0.25 * cds_proxy.clip(-3, 3)
                 + 0.20 * tot_z.clip(-3, 3))

    # --- Signal Generation ---
    # Strong long: composite > 0.8 AND carry positive
    strong_long = ((composite > 0.8) & (carry_positive > 0)).astype(float)
    # Strong short: composite < -0.8 OR extreme vol (CDS crisis proxy)
    vol_crisis = (_z(vol_20, 120) > 2.0).astype(float)
    strong_short = ((composite < -0.8) | (vol_crisis > 0)).astype(float)

    # Continuous composite as baseline
    base_signal = composite.clip(-2, 2) / 2 * 35

    # Event overlays
    event_signal = strong_long * 45 - strong_short * 45

    # --- Risk Dampening ---
    # Scale by inverse of recent vol (position size = 1% / RealizedVol)
    vol_dampener = ind["vol_dampener"]

    raw = (base_signal + event_signal) * vol_dampener

    # Smooth to reduce whipsaw (weekly rebalance proxy)
    raw = raw.rolling(5, min_periods=1).mean()

    return _clip_signal(raw)


def s041_pivot_confluence(ind):
    """041 | Pivot Point Confluence Trading

    Chicago Floor Traders: Support/Resistance from universal pivot calculations.

    Three pivot systems computed from prior bar's H/L/C:
      Standard:  PP = (H+L+C)/3, R1 = 2PP-L, S1 = 2PP-H, R2 = PP+(H-L), S2 = PP-(H-L),
                 R3 = H+2(PP-L), S3 = L-2(H-PP)
      Fibonacci: R1 = PP+0.382*(H-L), S1 = PP-0.382*(H-L), R2/S2 at 0.618, R3/S3 at 1.000
      Camarilla: R4 = C+1.1*(H-L)/2, S4 = C-1.1*(H-L)/2

    Confluence = zone where 2+ pivot types converge within 0.3% band.
    Confluence_Strength = count of converging levels at that zone.

    Signal:
      Long  at support confluence: price near support zone with strength >= 3
      Short at resistance confluence: price near resistance zone with strength >= 3
      Breakout: price closes beyond R3/S3 with volume = trend continuation
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]

    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)
    prev_range = prev_h - prev_l

    # --- Standard Pivots ---
    pp = (prev_h + prev_l + prev_c) / 3
    std_r1 = 2 * pp - prev_l
    std_s1 = 2 * pp - prev_h
    std_r2 = pp + prev_range
    std_s2 = pp - prev_range
    std_r3 = prev_h + 2 * (pp - prev_l)
    std_s3 = prev_l - 2 * (prev_h - pp)

    # --- Fibonacci Pivots ---
    fib_r1 = pp + 0.382 * prev_range
    fib_s1 = pp - 0.382 * prev_range
    fib_r2 = pp + 0.618 * prev_range
    fib_s2 = pp - 0.618 * prev_range
    fib_r3 = pp + 1.000 * prev_range
    fib_s3 = pp - 1.000 * prev_range

    # --- Camarilla Pivots ---
    cam_r4 = prev_c + 1.1 * prev_range / 2
    cam_s4 = prev_c - 1.1 * prev_range / 2

    # --- Confluence detection ---
    # For each bar, count how many pivot levels are within 0.3% of close
    # Group into support levels (below close) and resistance levels (above close)
    band = 0.003 * close  # 0.3% band

    support_levels = [std_s1, std_s2, std_s3, fib_s1, fib_s2, fib_s3, cam_s4, pp]
    resist_levels = [std_r1, std_r2, std_r3, fib_r1, fib_r2, fib_r3, cam_r4]

    # Count support levels within band of current price (price near support = bounce up)
    sup_count = pd.Series(0.0, index=close.index)
    for lvl in support_levels:
        near = ((close - lvl).abs() < band) & (lvl <= close)
        sup_count = sup_count + near.astype(float)

    # Count resistance levels within band
    res_count = pd.Series(0.0, index=close.index)
    for lvl in resist_levels:
        near = ((close - lvl).abs() < band) & (lvl >= close)
        res_count = res_count + near.astype(float)

    # --- Breakout detection: close beyond R3/S3 ---
    vol_ma = volume.rolling(20, min_periods=5).mean()
    vol_surge = volume > (1.3 * vol_ma)
    breakout_up = (close > std_r3) & vol_surge
    breakout_dn = (close < std_s3) & vol_surge

    # --- Above/below pivot baseline ---
    above_pp = (close > pp).astype(float) * 2 - 1  # +1 above, -1 below

    # --- Composite signal ---
    # Support confluence (sup_count >= 2): bullish bounce (+25 per level above 1)
    # Resistance confluence (res_count >= 2): bearish rejection (-25 per level above 1)
    # Breakout: +40 / -40
    # Above/below PP: +10 / -10 continuous baseline
    # Trend context: momentum alignment
    sup_signal = (sup_count - 1).clip(0, 4) * 25  # 0 if count<=1, 25/50/75/100 for 2/3/4/5
    res_signal = (res_count - 1).clip(0, 4) * 25

    breakout_signal = breakout_up.astype(float) * 40 - breakout_dn.astype(float) * 40

    # Recent support/resistance persistence (was near confluence recently?)
    sup_recent = sup_signal.rolling(3, min_periods=1).max()
    res_recent = res_signal.rolling(3, min_periods=1).max()

    raw = (sup_recent - res_recent
           + breakout_signal
           + above_pp * 10
           + ind["mom_score"] * 15)


    return _clip_signal(raw)


def s042_risk_appetite_barometer(ind):
    """042 | Swedish Krona Risk Appetite Barometer (OHLCV proxy)

    Stockholm (Riksbank/SEB Quantitative): Risk Sentiment via OHLCV proxy.

    SEK is a high-beta small-open-economy currency that amplifies global risk appetite.

    Risk_On_Score composite (OHLCV proxy):
      0.25 * norm(5d_return)         -- equity momentum proxy
      0.20 * norm(-vol_change_5d)    -- VIX compression proxy
      0.20 * norm(obv_signal)        -- flow proxy (HY spread)
      0.15 * norm(mom_20)            -- EM FX / broader momentum
      0.10 * norm(vol_flow)          -- copper / commodity proxy
      0.10 * norm(-vol_ratio)        -- USD weakness proxy

    Fair_Value via 60d rolling regression of returns on Risk_On_Score.
    Mispricing_Z = (actual - predicted) / std.

    Signal:
      Long : Risk_On > +1.0 AND Mispricing_Z < -1.0 (risk-on but asset hasn't caught up)
      Short: Risk_On < -1.0 AND Mispricing_Z > +1.0
      Baseline: trend_score alignment
    """
    close = ind["close"]

    # --- Risk-On Score composite (OHLCV proxy for global risk) ---
    ret_5 = close.pct_change(5)
    vol_20 = ind["vol_20"]
    vol_change_5d = vol_20 - vol_20.shift(5)

    risk_on_score = (
        0.25 * _z(ret_5, 60)
        + 0.20 * _z(-vol_change_5d, 60)
        + 0.20 * _z(ind["obv_signal"], 60)
        + 0.15 * _z(ind["mom_20"], 60)
        + 0.10 * _z(ind["vol_flow"], 60)
        + 0.10 * _z(-ind["vol_ratio"], 60)
    ).fillna(0)

    # --- Fair value via rolling regression proxy ---
    # Use 60-bar rolling correlation of returns with risk_on_score as beta
    ret_1 = ind["ret_1"]
    roll_cov = ret_1.rolling(60, min_periods=20).cov(risk_on_score)
    roll_var = risk_on_score.rolling(60, min_periods=20).var().replace(0, np.nan)
    beta = (roll_cov / roll_var).clip(-3, 3).fillna(0)
    alpha = ret_1.rolling(60, min_periods=20).mean() - beta * risk_on_score.rolling(60, min_periods=20).mean()
    predicted = alpha + beta * risk_on_score
    mispricing = ret_1 - predicted
    mispricing_z = _z(mispricing, 60)

    # --- Signal logic ---
    # Strong risk-on + underperforming = long (catch-up trade)
    long_setup = (risk_on_score > 1.0) & (mispricing_z < -1.0)
    short_setup = (risk_on_score < -1.0) & (mispricing_z > 1.0)

    # Moderate signals for weaker conditions
    mild_long = (risk_on_score > 0.5) & (mispricing_z < -0.5)
    mild_short = (risk_on_score < -0.5) & (mispricing_z > 0.5)

    # Persistence: was in setup recently?
    long_recent = long_setup.astype(float).rolling(5, min_periods=1).max()
    short_recent = short_setup.astype(float).rolling(5, min_periods=1).max()

    # --- Composite ---
    event_signal = (long_setup.astype(float) * 50 - short_setup.astype(float) * 50
                    + mild_long.astype(float) * 25 - mild_short.astype(float) * 25)
    persist_signal = long_recent * 15 - short_recent * 15
    baseline = risk_on_score.clip(-2, 2) / 2 * 15 + ind["trend_score"] * 10

    raw = event_signal + persist_signal + baseline

    return _clip_signal(raw)


def s043_vwmo(ind):
    """043 | Volume-Weighted Momentum Oscillator

    TradingView Community: Volume-Enhanced Momentum.

    VWMO = sum(Volume * ret, 14) / sum(Volume, 14)
      volume-weighted average return over n periods.

    VWMO_Signal = EMA(VWMO, 9)
    VWMO_Hist   = VWMO - VWMO_Signal  (histogram for timing)
    VWMO_Z      = VWMO / rolling_std(VWMO, 60)  (normalized strength)

    Volume_Thrust:
      VT = sum(Volume * (ret > 0), 5) / sum(Volume, 5)  (% vol on up days)
      Bullish_Thrust = VT > 0.70
      Bearish_Thrust = VT < 0.30

    Divergence: price new high but VWMO declining = smart money exit.

    Signal:
      Long : VWMO_Z > +1.0 AND Bullish_Thrust AND VWMO_Hist > 0
      Short: VWMO_Z < -1.0 AND Bearish_Thrust AND VWMO_Hist < 0
      Baseline: continuous VWMO_Z + histogram direction
    """
    close = ind["close"]
    volume = ind["volume"]
    ret = close.pct_change(1).fillna(0)

    # --- VWMO: volume-weighted average return ---
    n_vwmo = 14
    vw_ret = (ret * volume).rolling(n_vwmo, min_periods=5).sum()
    vol_sum = volume.rolling(n_vwmo, min_periods=5).sum().replace(0, np.nan)
    vwmo = vw_ret / vol_sum

    # Signal line and histogram
    vwmo_signal = vwmo.ewm(span=9, adjust=False).mean()
    vwmo_hist = vwmo - vwmo_signal

    # Normalized z-score
    vwmo_z = _z(vwmo, 60)

    # --- Volume Thrust ---
    up_vol = (volume * (ret > 0).astype(float))
    vt = up_vol.rolling(5, min_periods=2).sum() / volume.rolling(5, min_periods=2).sum().replace(0, np.nan)
    vt = vt.fillna(0.5)
    bullish_thrust = vt > 0.70
    bearish_thrust = vt < 0.30

    # --- Divergence detection ---
    # Price at 20-bar high but VWMO declining
    price_hh = close >= close.rolling(20, min_periods=5).max()
    vwmo_declining = vwmo < vwmo.shift(5)
    bearish_div = (price_hh & vwmo_declining).astype(float)

    price_ll = close <= close.rolling(20, min_periods=5).min()
    vwmo_rising = vwmo > vwmo.shift(5)
    bullish_div = (price_ll & vwmo_rising).astype(float)

    # --- Signal composition ---
    # Strong setups: z-score + thrust + histogram alignment
    strong_long = ((vwmo_z > 1.0) & bullish_thrust & (vwmo_hist > 0)).astype(float)
    strong_short = ((vwmo_z < -1.0) & bearish_thrust & (vwmo_hist < 0)).astype(float)

    # Moderate: z-score + histogram only
    mod_long = ((vwmo_z > 0.5) & (vwmo_hist > 0)).astype(float)
    mod_short = ((vwmo_z < -0.5) & (vwmo_hist < 0)).astype(float)

    # Persistence for strong signals
    strong_long_p = strong_long.rolling(3, min_periods=1).max()
    strong_short_p = strong_short.rolling(3, min_periods=1).max()

    # Volume thrust continuous score
    vt_score = _z(vt - 0.5, 40).clip(-2, 2) / 2  # normalized around neutral 0.5

    raw = (strong_long_p * 45 - strong_short_p * 45
           + mod_long * 20 - mod_short * 20
           + bullish_div * 15 - bearish_div * 15
           + vt_score * 15
           + vwmo_z.clip(-2, 2) / 2 * 10)

    return _clip_signal(raw)


def s044_adaptive_rsi_hilbert(ind):
    """044 | Adaptive RSI with Hilbert Transform Period

    New York DSP (Ehlers): Adaptive Oscillator.

    Hilbert Transform extracts dominant cycle period from price:
      Smooth = (4*P + 3*P1 + 2*P2 + P3) / 10
      Detrender uses Hilbert coefficients with adaptive period.
      Q1, I1 = quadrature and in-phase from Hilbert Transform.
      Period = 360 / arctan(Q1/I1), constrained to [6, 50].

    Adaptive RSI:
      RSI_period = round(Period / 2)  (half the dominant cycle)
      ARSI = Wilder RSI with dynamic period = RSI_period

    Signal:
      Long : ARSI < 25 in uptrend (oversold relative to current market cycle)
      Short: ARSI > 75 in downtrend
      Cycle info: short periods = fast market, long periods = slow market
    """
    close = ind["close"]
    n = len(close)
    vals = close.values.astype(float)

    # --- Simplified Hilbert Transform dominant cycle extraction ---
    smooth = np.full(n, np.nan)
    period = np.full(n, 14.0)  # default period
    for i in range(6, n):
        # 4-bar weighted smoother
        smooth[i] = (4 * vals[i] + 3 * vals[i-1] + 2 * vals[i-2] + vals[i-3]) / 10.0

    # Estimate dominant period via autocorrelation on smoothed series
    sm = pd.Series(smooth, index=close.index)
    sm_diff = sm.diff().fillna(0)

    # Zero-crossing period: count bars between sign changes of sm_diff
    sign_changes = (sm_diff > 0).astype(float).diff().abs().fillna(0)
    # Rolling count of sign changes in last 40 bars
    sc_count = sign_changes.rolling(40, min_periods=10).sum().replace(0, np.nan)
    # Approximate half-cycles in 40 bars -> period = 40 / (sc_count / 2)
    raw_period = 80.0 / sc_count
    dom_period = raw_period.clip(6, 50).fillna(14.0)

    # --- Adaptive RSI with dynamic period ---
    rsi_period = (dom_period / 2).round().clip(3, 25).astype(int)

    # Compute RSI with variable lookback using vectorized approximation:
    # Use EMA-based RSI where alpha = 1/period
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Multiple fixed-period RSIs, then select based on adaptive period
    rsi_vals = np.full(n, 50.0)
    period_bins = [5, 7, 10, 14, 20, 25]
    rsi_by_period = {}
    for p in period_bins:
        avg_gain = gain.ewm(span=p, adjust=False).mean()
        avg_loss = loss.ewm(span=p, adjust=False).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi_by_period[p] = (100 - 100 / (1 + rs)).fillna(50)

    # Select closest period bin for each bar
    for i in range(n):
        rp = rsi_period.iloc[i] if i < n else 14
        # Find closest bin
        best_bin = min(period_bins, key=lambda x: abs(x - rp))
        rsi_vals[i] = rsi_by_period[best_bin].iloc[i]

    arsi = pd.Series(rsi_vals, index=close.index)

    # --- Trend detection ---
    trend_up = ind["above_50"] > 0
    trend_dn = ind["above_50"] < 1

    # --- Signal logic ---
    # Strong: ARSI extreme + trend alignment
    strong_long = ((arsi < 25) & trend_up).astype(float)
    strong_short = ((arsi > 75) & trend_dn).astype(float)

    # Moderate: ARSI approaching extreme
    mod_long = ((arsi < 35) & trend_up).astype(float)
    mod_short = ((arsi > 65) & trend_dn).astype(float)

    # Persistence
    strong_long_p = strong_long.rolling(5, min_periods=1).max()
    strong_short_p = strong_short.rolling(5, min_periods=1).max()

    # Continuous: deviation from 50 scaled by trend
    arsi_dev = (50 - arsi) / 50  # positive when oversold, negative when overbought
    trend_bias = ind["trend_score"]

    # Cycle speed info: short period = wider bands, faster market
    cycle_fast = (dom_period < 12).astype(float)
    cycle_slow = (dom_period > 30).astype(float)
    cycle_bonus = cycle_fast * 5 - cycle_slow * 5  # fast markets get small long bias

    raw = (strong_long_p * 40 - strong_short_p * 40
           + mod_long * 20 - mod_short * 20
           + arsi_dev * 20
           + trend_bias * 10
           + cycle_bonus)

    return _clip_signal(raw)


def s045_dual_momentum(ind):
    """045 | Tel Aviv Dual Momentum

    Tel Aviv (Israeli Quant Hedge Fund): Absolute + Relative Momentum.

    Based on Gary Antonacci's Dual Momentum framework:

    Absolute Momentum (Time-Series):
      AM = long-term return > 0  (asset in positive momentum = investable)
      Proxy: above_200 (price above 200 SMA) AND mom_40 > 0

    Relative Momentum (Cross-Sectional):
      RM = rank by multi-horizon return strength
      Proxy: momentum quality = risk-adjusted momentum

    Dual Momentum Score:
      DM = AM_flag * RM_strength
      Long when DM strong (both absolute and relative positive)
      Defensive when AM fails (rotate to safety)

    Geopolitical Filter proxy: if vol spikes (VIX analog) + trend breakdown
      -> reduce equity exposure

    Signal:
      Long : AM positive AND RM strong (top momentum with trend)
      Neutral: AM negative (defensive mode, no position)
      Short: Both negative with strength (strong downtrend)
    """
    close = ind["close"]

    # --- Absolute Momentum (time-series) ---
    # Asset in positive absolute momentum: above 200 SMA AND positive long return
    above_200 = ind["above_200"]
    mom_40 = ind["mom_40"]
    mom_20 = ind["mom_20"]
    am_positive = (above_200 > 0) & (mom_40 > 0)
    am_negative = (above_200 < 1) & (mom_40 < 0)

    # --- Relative Momentum (cross-sectional quality) ---
    # Risk-adjusted momentum: momentum / volatility = momentum quality
    vol_20 = ind["vol_20"].replace(0, np.nan)
    rm_quality = (mom_20 / vol_20).fillna(0)
    rm_z = _z(rm_quality, 60)

    # Multi-horizon confirmation: 5d, 10d, 20d, 40d all positive
    multi_mom = (
        (ind["mom_5"] > 0).astype(float)
        + (ind["mom_10"] > 0).astype(float)
        + (mom_20 > 0).astype(float)
        + (mom_40 > 0).astype(float)
    ) / 4.0  # 0 to 1: fraction of horizons positive

    # --- Dual Momentum Score ---
    dm_long = (am_positive & (rm_z > 0.5) & (multi_mom >= 0.75)).astype(float)
    dm_short = (am_negative & (rm_z < -0.5) & (multi_mom <= 0.25)).astype(float)

    # Moderate signals
    mild_long = (am_positive & (rm_z > 0)).astype(float)
    mild_short = (am_negative & (rm_z < 0)).astype(float)

    # --- Geopolitical Filter (vol spike = defensive) ---
    vol_z = _z(ind["vol_20"], 120)
    geo_stress = (vol_z > 2.0) & (ind["trend_score"] < -0.3)
    defensive_dampen = 1 - geo_stress.astype(float) * 0.5  # halve signal in stress

    # --- Persistence (monthly rebalance proxy: hold for 20 bars) ---
    dm_long_p = dm_long.rolling(10, min_periods=1).max()
    dm_short_p = dm_short.rolling(10, min_periods=1).max()

    # --- Continuous baseline ---
    # Multi-horizon momentum fraction as continuous signal
    mom_baseline = (multi_mom - 0.5) * 2  # [-1, 1]

    # AM contribution: above 200 SMA bias
    am_bias = above_200.astype(float) * 2 - 1  # +1 above, -1 below

    raw = ((dm_long_p * 40 - dm_short_p * 40
            + mild_long * 20 - mild_short * 20
            + mom_baseline * 15
            + am_bias * 10
            + rm_z.clip(-2, 2) / 2 * 10)
           * defensive_dampen)

    return _clip_signal(raw)


def s046_trix_divergence(ind):
    """046 | Triple Exponential Average (TRIX) Divergence

    New York (Jack Hutson): Smoothed Momentum.

    TRIX = 1-period rate of change of triple-smoothed EMA(15):
      EMA1 = EMA(Close, 15)
      EMA2 = EMA(EMA1, 15)
      EMA3 = EMA(EMA2, 15)
      TRIX = (EMA3_t - EMA3_{t-1}) / EMA3_{t-1} * 100

    TRIX_Signal = EMA(TRIX, 9)
    TRIX_Hist   = TRIX - TRIX_Signal

    Zero-Line Cross:
      Bullish: TRIX crosses above 0
      Bearish: TRIX crosses below 0

    Divergence (early warning, leads by 5-15 bars):
      Bullish: Price lower low, TRIX higher low
      Bearish: Price higher high, TRIX lower high

    Signal:
      Long : TRIX crosses above zero AND TRIX_Hist > 0
      Short: TRIX crosses below zero AND TRIX_Hist < 0
      Exit : TRIX_Hist crosses zero against position
    """
    close = ind["close"]

    # --- Triple EMA smoothing ---
    ema1 = close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()

    # TRIX: rate of change of triple EMA (x10000 for scale)
    trix = ema3.pct_change(1) * 10000
    trix_signal = trix.ewm(span=9, adjust=False).mean()
    trix_hist = trix - trix_signal

    # --- Zero-line crossovers ---
    trix_cross_up = _crossover(trix, pd.Series(0.0, index=close.index))
    trix_cross_dn = _crossunder(trix, pd.Series(0.0, index=close.index))

    # --- Histogram crossovers (exit signals) ---
    hist_cross_up = _crossover(trix_hist, pd.Series(0.0, index=close.index))
    hist_cross_dn = _crossunder(trix_hist, pd.Series(0.0, index=close.index))

    # --- Divergence detection ---
    # 20-bar rolling min/max for swing detection
    price_20_high = close.rolling(20, min_periods=5).max()
    price_20_low = close.rolling(20, min_periods=5).min()
    trix_20_high = trix.rolling(20, min_periods=5).max()
    trix_20_low = trix.rolling(20, min_periods=5).min()

    # Bearish divergence: price at 20-bar high but TRIX 20-bar high is lower than prev
    price_at_high = (close >= price_20_high * 0.998)
    trix_high_declining = trix_20_high < trix_20_high.shift(10)
    bearish_div = (price_at_high & trix_high_declining).astype(float)

    # Bullish divergence: price at 20-bar low but TRIX 20-bar low is rising
    price_at_low = (close <= price_20_low * 1.002)
    trix_low_rising = trix_20_low > trix_20_low.shift(10)
    bullish_div = (price_at_low & trix_low_rising).astype(float)

    # --- Signal composition ---
    # Zero-line cross: strong event (+/- 40)
    cross_signal = trix_cross_up * 40 - trix_cross_dn * 40

    # Persistence: recent crossover holds for 10 bars
    cross_long_p = trix_cross_up.rolling(10, min_periods=1).max()
    cross_short_p = trix_cross_dn.rolling(10, min_periods=1).max()
    persist_signal = cross_long_p * 20 - cross_short_p * 20

    # TRIX position: above/below zero = continuous bias
    trix_pos = (trix > 0).astype(float) * 2 - 1  # +1/-1

    # Histogram direction
    hist_z = _z(trix_hist, 40).clip(-2, 2) / 2

    # Divergence: early warning
    div_signal = bullish_div * 20 - bearish_div * 20

    raw = (cross_signal
           + persist_signal
           + trix_pos * 10
           + hist_z * 15
           + div_signal)

    return _clip_signal(raw)


def s047_mass_index_reversal(ind):
    """047 | Mass Index Reversal Bulge

    Chicago (Donald Dorsey): Reversal Detection.

    Mass_Index measures range expansion/contraction:
      EMA_Range       = EMA(High - Low, 9)
      Double_EMA_Range = EMA(EMA_Range, 9)
      Mass_Ratio      = EMA_Range / Double_EMA_Range
      Mass_Index(25)  = sum(Mass_Ratio, 25)

    Reversal Bulge:
      Mass_Index rises above 27.0 (expansion)
      Then falls below 26.5 (contraction = "reversal bulge")
      This indicates H-L range expanded then contracted -> potential reversal.

    Mass Index detects THAT a reversal is likely, not direction.
    Direction confirmed by EMA(9) vs EMA(18) crossover.

    Signal:
      Setup: Mass Index bulge detected (rose above 27, fell below 26.5)
      Long : Setup + prior downtrend + EMA(9) crosses above EMA(18)
      Short: Setup + prior uptrend + EMA(9) crosses below EMA(18)
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]

    # --- Mass Index ---
    rng = high - low
    ema_rng = rng.ewm(span=9, adjust=False).mean()
    dema_rng = ema_rng.ewm(span=9, adjust=False).mean().replace(0, np.nan)
    mass_ratio = ema_rng / dema_rng
    mass_index = mass_ratio.rolling(25, min_periods=10).sum()

    # --- Reversal Bulge detection ---
    # Classic bulge: above 27 then below 26.5
    was_above_27 = (mass_index > 27.0).astype(float).rolling(10, min_periods=1).max()
    below_26_5 = mass_index < 26.5
    classic_bulge = (was_above_27 > 0) & below_26_5

    # Softer bulge: above 26.2 then below 25.8 (catches more events)
    was_above_26_2 = (mass_index > 26.2).astype(float).rolling(10, min_periods=1).max()
    below_25_8 = mass_index < 25.8
    soft_bulge = (was_above_26_2 > 0) & below_25_8

    # Combined: classic bulge gets full weight, soft gets partial
    bulge_classic = classic_bulge.astype(float)
    bulge_soft = soft_bulge.astype(float)
    bulge_persist = bulge_classic.rolling(8, min_periods=1).max()
    soft_persist = bulge_soft.rolling(6, min_periods=1).max()

    # --- Directional confirmation: EMA(9) vs EMA(18) ---
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema18 = close.ewm(span=18, adjust=False).mean()
    ema_cross_up = _crossover(ema9, ema18)
    ema_cross_dn = _crossunder(ema9, ema18)
    ema_recent_up = ema_cross_up.rolling(8, min_periods=1).max()
    ema_recent_dn = ema_cross_dn.rolling(8, min_periods=1).max()

    # --- Prior trend detection ---
    trend_score = ind["trend_score"]
    was_uptrend = trend_score.rolling(20, min_periods=5).mean() > 0.1
    was_downtrend = trend_score.rolling(20, min_periods=5).mean() < -0.1

    # --- Reversal signals ---
    # Classic reversal: bulge + trend + EMA cross
    bull_rev = ((bulge_persist > 0) | (soft_persist > 0)) & was_downtrend & (ema_recent_up > 0)
    bear_rev = ((bulge_persist > 0) | (soft_persist > 0)) & was_uptrend & (ema_recent_dn > 0)

    # Classic bulge reversals: strong event
    bull_rev_classic = (bulge_persist > 0) & was_downtrend & (ema_recent_up > 0)
    bear_rev_classic = (bulge_persist > 0) & was_uptrend & (ema_recent_dn > 0)

    # Persistence: hold reversal signals
    bull_rev_p = bull_rev.astype(float).rolling(12, min_periods=1).max()
    bear_rev_p = bear_rev.astype(float).rolling(12, min_periods=1).max()
    bull_classic_p = bull_rev_classic.astype(float).rolling(12, min_periods=1).max()
    bear_classic_p = bear_rev_classic.astype(float).rolling(12, min_periods=1).max()

    # --- Continuous baseline ---
    # Mass Index z-score: expansion/contraction as regime signal
    mass_z = _z(mass_index, 60)
    # Range expansion contrarian: high mass + trending = reversal expected
    expansion_contrarian = (mass_z > 1.0).astype(float) * (-trend_score.clip(-1, 1))

    # EMA position: always-on directional baseline
    ema_pos = (ema9 > ema18).astype(float) * 2 - 1

    # EMA distance z-score for amplitude
    ema_diff_z = _z(ema9 - ema18, 40).clip(-2, 2) / 2

    raw = (bull_classic_p * 50 - bear_classic_p * 50
           + bull_rev_p * 25 - bear_rev_p * 25
           + expansion_contrarian * 15
           + ema_pos * 20
           + ema_diff_z * 15
           + ind["mom_score"] * 10)

    return _clip_signal(raw)


def s048_gold_momentum(ind):
    """048 | Dubai Gold Dinar Momentum (OHLCV proxy)

    Dubai (DIFC Quantitative): Gold-FX Hybrid via OHLCV proxy.

    Composite_Gold from multiple factors (OHLCV proxies):
      0.35 * Gold_Mom       -- risk-adjusted momentum (mom_20 / vol_20)
      0.25 * (-USD_Mom)     -- inverse dollar strength proxy (negative correlation)
      0.15 * (-Real_Rate_Z) -- negative real rate proxy (vol regime inverted)
      0.15 * Gulf_Premium_Z -- physical demand proxy (volume flow)
      0.10 * Oil_Mom        -- commodity complex proxy (roc_20 / vol_20)

    Signal:
      Long : Composite > +0.8 AND negative real rate environment
      Short: Composite < -0.8 AND rising real rate environment
      Enhanced: volume flow spike = physical demand surge (very bullish)
    """
    close = ind["close"]
    vol_20 = ind["vol_20"].replace(0, np.nan)

    # --- Factor 1: Risk-adjusted momentum (Gold_Mom proxy) ---
    gold_mom = ind["mom_20"] / vol_20
    gold_mom_z = _z(gold_mom, 60)

    # --- Factor 2: Inverse dollar strength (USD_Mom proxy) ---
    # Use negative of broad market momentum as USD weakness proxy
    # (gold inversely correlates with USD)
    usd_proxy = -ind["mom_10"]  # negative short-term momentum = USD weak
    usd_z = _z(usd_proxy, 60)

    # --- Factor 3: Real rate proxy (negative = gold bullish) ---
    # Low vol + downtrend = deflationary (bearish gold)
    # High vol + stress = negative real rates (bullish gold)
    real_rate_z = _z(ind["vol_20"], 120)
    neg_real = -real_rate_z  # higher vol = more negative real rates proxy

    # --- Factor 4: Physical demand proxy (Gulf_Premium) ---
    # Volume flow as proxy for physical demand surge
    gulf_premium_z = _z(ind["vol_flow"], 60)

    # --- Factor 5: Oil momentum (commodity complex) ---
    oil_mom = ind["roc_20"] / vol_20
    oil_z = _z(oil_mom, 60)

    # --- Composite Gold Score ---
    composite = (
        0.35 * gold_mom_z.clip(-3, 3)
        + 0.25 * usd_z.clip(-3, 3)
        + 0.15 * neg_real.clip(-3, 3)
        + 0.15 * gulf_premium_z.clip(-3, 3)
        + 0.10 * oil_z.clip(-3, 3)
    ).fillna(0)

    # --- Signal logic ---
    strong_long = (composite > 0.8) & (neg_real > -0.5)
    strong_short = (composite < -0.8) & (neg_real < 0.5)

    mod_long = (composite > 0.4) & (composite <= 0.8)
    mod_short = (composite < -0.4) & (composite >= -0.8)

    # Enhanced: physical demand surge
    demand_surge = (gulf_premium_z > 1.5).astype(float)

    # Persistence
    strong_long_p = strong_long.astype(float).rolling(5, min_periods=1).max()
    strong_short_p = strong_short.astype(float).rolling(5, min_periods=1).max()

    # --- Above 200 SMA baseline ---
    above_200 = ind["above_200"]
    trend_bias = above_200.astype(float) * 2 - 1

    raw = (strong_long_p * 40 - strong_short_p * 40
           + mod_long.astype(float) * 20 - mod_short.astype(float) * 20
           + demand_surge * 15
           + composite.clip(-2, 2) / 2 * 15
           + trend_bias * 10)

    return _clip_signal(raw)


def s049_consecutive_days(ind):
    """049 | Consecutive Day Pattern with Adaptive Scaling

    Quantitative (Our Research, v97): Pattern + Adaptive.
    Grand champion of 100 tested variants (C=0.735, Sharpe 0.648).

    Consecutive Up/Down Detection:
      up_runs_t = up_runs_{t-1} + 1 if close > prev_close, else 0
      dn_runs_t = dn_runs_{t-1} + 1 if close < prev_close, else 0

    Exhaustion (Mean Reversion): runs >= 4 -> contrarian
      up_exhaust = max(0, (up_runs - 3) / 3) clipped at 1
      dn_exhaust = max(0, (dn_runs - 3) / 3) clipped at 1

    Momentum Continuation: 2-3 day runs -> continuation
      up_mom = 1 if 2 <= up_runs <= 3
      dn_mom = 1 if 2 <= dn_runs <= 3

    Kaufman Efficiency Ratio:
      ER = |P_t - P_{t-10}| / sum(|P_i - P_{i-1}|, i=t-9..t)
      adaptive_scale = 1.2 - 0.4 * ER  (1.2 mean-reverting, 0.8 trending)

    CSI = (0.7*dn_exhaust - 0.7*up_exhaust + 0.2*up_mom - 0.2*dn_mom
           + 0.15*vol_flow + 0.15*trend_score) * 85 * adaptive_scale
           + 5*above_200 + 1

    NO EMA smoothing (preserves crisp buy/sell boundaries).
    """
    close = ind["close"]
    n = len(close)
    c = close.values.astype(float)

    # --- Consecutive run detection ---
    up_runs = np.zeros(n)
    dn_runs = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i-1]:
            up_runs[i] = up_runs[i-1] + 1
            dn_runs[i] = 0
        elif c[i] < c[i-1]:
            dn_runs[i] = dn_runs[i-1] + 1
            up_runs[i] = 0
        else:
            up_runs[i] = 0
            dn_runs[i] = 0

    up_runs_s = pd.Series(up_runs, index=close.index)
    dn_runs_s = pd.Series(dn_runs, index=close.index)

    # --- Exhaustion (mean reversion): runs >= 4 ---
    up_exhaust = ((up_runs_s - 3) / 3).clip(0, 1)  # 0 if <4, ramps to 1 at 6
    dn_exhaust = ((dn_runs_s - 3) / 3).clip(0, 1)

    # --- Momentum continuation: 2-3 day runs ---
    up_mom = ((up_runs_s >= 2) & (up_runs_s <= 3)).astype(float)
    dn_mom = ((dn_runs_s >= 2) & (dn_runs_s <= 3)).astype(float)

    # --- Kaufman Efficiency Ratio ---
    er = ind["efficiency"]  # already computed: |net move| / sum(|steps|)
    adaptive_scale = 1.2 - 0.4 * er  # 1.2 in choppy, 0.8 in trending

    # --- CSI composite (NO smoothing) ---
    # dn_exhaust = buy (contrarian after selloff)
    # up_exhaust = sell (contrarian after rally)
    # up_mom = mild long, dn_mom = mild short
    vol_flow = ind["vol_flow"]
    trend_score = ind["trend_score"]
    above_200 = ind["above_200"]

    csi = (0.7 * dn_exhaust - 0.7 * up_exhaust
           + 0.2 * up_mom - 0.2 * dn_mom
           + 0.15 * _safe(vol_flow).clip(-1, 1)
           + 0.15 * _safe(trend_score).clip(-1, 1)
           ) * 85 * adaptive_scale + 5 * above_200.astype(float) + 1

    # NO _clip_signal smoothing -- preserve crisp boundaries per spec
    return csi.clip(-100, 100)


def s050_sector_rotation(ind):
    """050 | Toronto Resource Sector Rotation (OHLCV proxy)

    Toronto (TSX Mining/Energy): Sector Rotation via OHLCV proxy.

    Commodity_Cycle_Phase (OHLCV proxy):
      Metals_Mom  = risk-adjusted 13w momentum (mom_40 / vol_20)
      Energy_Mom  = shorter-term momentum proxy (roc_20 / vol_20)
      Cycle_Score = 0.55 * Metals_Mom + 0.45 * Energy_Mom

    TSX Sector Relative Strength:
      RS = multi-timeframe momentum quality
      MQ = RS / vol(RS)  (risk-adjusted)

    CAD Sensitivity:
      Commodity up -> CAD up -> inverse USD proxy

    Composite = 0.40*MQ + 0.30*Cycle_Score + 0.30*CAD_adjusted

    Signal:
      Long : Cycle_Score > 0 AND top momentum quality
      Defensive: Cycle_Score < -0.5 -> reduce exposure
      Rebalance: bi-weekly proxy (use slower smoothing)
    """
    close = ind["close"]
    vol_20 = ind["vol_20"].replace(0, np.nan)

    # --- Commodity Cycle Phase ---
    # Metals momentum: long-term (40-day = ~2 months proxy for 13 weeks)
    metals_mom = ind["mom_40"] / vol_20
    metals_z = _z(metals_mom, 60)

    # Energy momentum: shorter-term
    energy_mom = ind["roc_20"] / vol_20
    energy_z = _z(energy_mom, 60)

    # Cycle Score
    cycle_score = (0.55 * metals_z + 0.45 * energy_z).fillna(0)

    # --- Momentum Quality (risk-adjusted relative strength) ---
    # Multi-timeframe: 5d, 10d, 20d, 40d momentum agreement
    mom_5_z = _z(ind["mom_5"], 40)
    mom_10_z = _z(ind["mom_10"], 40)
    mom_20_z = _z(ind["mom_20"], 40)
    mom_40_z = _z(ind["mom_40"], 40)

    mq = (0.15 * mom_5_z + 0.25 * mom_10_z + 0.35 * mom_20_z + 0.25 * mom_40_z)
    mq = mq.clip(-3, 3)

    # --- CAD sensitivity proxy ---
    # Commodity up -> CAD strength -> good for resource sector
    # Use volume flow as proxy for commodity demand
    vol_flow = ind["vol_flow"]
    cad_proxy = _z(vol_flow, 60).clip(-2, 2)

    # --- Composite ---
    composite = (0.40 * mq + 0.30 * cycle_score.clip(-2, 2) + 0.30 * cad_proxy)

    # --- Signal logic ---
    strong_long = (cycle_score > 0.5) & (mq > 0.5) & (cad_proxy > 0)
    strong_short = (cycle_score < -0.5) & (mq < -0.5) & (cad_proxy < 0)

    mod_long = (cycle_score > 0) & (mq > 0)
    mod_short = (cycle_score < 0) & (mq < 0)

    # Defensive mode: deep negative cycle
    defensive = (cycle_score < -1.0).astype(float)

    # Persistence (bi-weekly rebalance proxy = longer hold)
    strong_long_p = strong_long.astype(float).rolling(10, min_periods=1).max()
    strong_short_p = strong_short.astype(float).rolling(10, min_periods=1).max()

    # Above 200 SMA trend filter
    above_200 = ind["above_200"]
    trend_bias = above_200.astype(float) * 2 - 1

    raw = (strong_long_p * 35 - strong_short_p * 35
           + mod_long.astype(float) * 20 - mod_short.astype(float) * 20
           + composite.clip(-2, 2) / 2 * 20
           + trend_bias * 10
           - defensive * 15)

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
