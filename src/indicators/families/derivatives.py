"""
SECTION VIII: DERIVATIVES-INSPIRED STRATEGIES (351-400)
Each function takes ind dict -> pd.Series[-100, +100].
All options/derivatives concepts are proxied from OHLCV data.
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

def s351_variance_swap(ind):
    """351 | Variance Swap Replication Signal"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Variance swap: realized vol vs implied vol (proxy: short vs long vol)
    var_spread = vol ** 2 - vol_60 ** 2
    z = _z(var_spread, 120)
    # Sell variance when spread is high (reversion)
    raw = -z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s352_gamma_scalp(ind):
    """352 | Gamma Scalping Delta-Hedged"""
    close = ind["close"]
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Gamma P&L proxy: (realized move)^2 - implied variance
    gamma_pnl = ret ** 2 - (vol ** 2 / 252)
    gamma_z = _z(gamma_pnl.rolling(20).sum(), 60)
    # Positive gamma scalp P&L = high realized vol vs implied
    raw = gamma_z * 30 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s353_skew_risk_reversal(ind):
    """353 | Skew Trading via Risk Reversal"""
    close = ind["close"]
    ret = ind["ret_1"]
    # Skew proxy: asymmetry of recent returns
    skew = ret.rolling(60).apply(lambda x: pd.Series(x).skew(), raw=False).fillna(0)
    skew_z = _z(skew, 120)
    # Negative skew = put premium > call premium; bullish contrarian
    raw = -skew_z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s354_term_structure_roll(ind):
    """354 | Term Structure Roll-Down Strategy"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Term structure: contango/backwardation in vol
    contango = (vol_60 > vol).astype(float)
    backwardation = (vol < vol_60).astype(float)
    spread = _z(vol_60 - vol, 120)
    raw = spread * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s355_put_write(ind):
    """355 | Put-Write Strategy (Systematic Premium Collection)"""
    close = ind["close"]
    vol = ind["vol_20"]
    rsi = ind["rsi"]
    # Put write: sell puts = bullish + collect premium
    # Better when vol is high (more premium) and market not crashing
    vol_z = _z(vol, 120)
    high_vol_premium = (vol_z > 0.5).astype(float) * vol_z
    not_crash = (rsi > 25).astype(float)
    raw = high_vol_premium * not_crash * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s356_iron_condor(ind):
    """356 | Iron Condor Adaptive Width"""
    close = ind["close"]
    vol = ind["vol_20"]
    bb_width = ind["bb_width"]
    # Iron condor profits from range-bound markets
    range_bound = (bb_width < bb_width.rolling(60).quantile(0.3)).astype(float)
    trending = (ind["adx"] > 25).astype(float)
    vol_z = _z(vol, 120)
    raw = range_bound * 30 - trending * 30 + (1 - vol_z.clip(0, 3) / 3) * 25
    return _clip_signal(raw)


def s357_calendar_spread(ind):
    """357 | Calendar Spread Theta Harvesting"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Calendar spread profits when near-term vol > far-term vol
    near_over_far = _z(vol / vol_60.replace(0, np.nan), 120)
    # Sell near, buy far when term structure is inverted
    raw = near_over_far * 35 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s358_butterfly_pinning(ind):
    """358 | Butterfly Spread Pinning Strategy"""
    close = ind["close"]
    vol = ind["vol_20"]
    bb_pctb = ind["bb_pctb"]
    # Butterfly profits near center strike; low vol helps
    near_center = (1 - 2 * (bb_pctb - 0.5).abs())
    low_vol = (1 - _z(vol, 120).clip(0, 3) / 3)
    raw = near_center * low_vol * 55
    return _clip_signal(raw)


def s359_dispersion(ind):
    """359 | Dispersion Premium via Correlation Swaps"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Dispersion: vol of vol as correlation proxy
    vol_ratio = vol / vol_60.replace(0, np.nan)
    dispersion = _z(vol_ratio.rolling(20).std(), 120)
    # High dispersion = low correlation = sell index vol, buy single stock vol
    raw = -dispersion * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s360_straddle_momentum(ind):
    """360 | Straddle Momentum (Post-Earnings Vol Persistence)"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Straddle momentum: vol persistence after events
    vol_change = vol.pct_change(5)
    large_vol_event = (vol_change > vol_change.rolling(60).quantile(0.9)).astype(float)
    direction = np.sign(ret.rolling(5).sum())
    raw = large_vol_event * direction * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s361_collar_overlay(ind):
    """361 | Covered Call Collar Dynamic Overlay"""
    close = ind["close"]
    vol = ind["vol_20"]
    trend = ind["trend_score"]
    vol_z = _z(vol, 120)
    # High vol: tighter collar; Low vol: wider collar
    protection = vol_z.clip(0, 3) / 3
    raw = trend * (1 - protection * 0.5) * 55
    return _clip_signal(raw)


def s362_backspread(ind):
    """362 | Ratio Backspread Convexity"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Backspread: buy convexity when vol is low and trending
    low_vol = (1 - _z(vol, 120).clip(0, 3) / 3)
    trending = (ind["adx"] > 20).astype(float)
    direction = np.sign(mom)
    raw = low_vol * trending * direction * 50
    return _clip_signal(raw)


def s363_jade_lizard(ind):
    """363 | Jade Lizard Premium Strategy"""
    close = ind["close"]
    vol = ind["vol_20"]
    rsi = ind["rsi"]
    # Jade lizard: sell put spread + sell call = premium collection
    # Works best in mild bullish markets
    mild_bull = ((rsi > 45) & (rsi < 65)).astype(float)
    vol_z = _z(vol, 120)
    premium = vol_z.clip(0, 3) / 3
    raw = mild_bull * premium * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s364_broken_wing_butterfly(ind):
    """364 | Broken Wing Butterfly Income"""
    vol = ind["vol_20"]
    bb_pctb = ind["bb_pctb"]
    trend = ind["trend_score"]
    # Broken wing: directional butterfly, skewed to trend direction
    raw = (bb_pctb - 0.5) * trend * 50 + (1 - _z(vol, 120).clip(0, 3) / 3) * 20
    return _clip_signal(raw)


def s365_zebra(ind):
    """365 | ZEBRA (Zero Extrinsic Back Ratio)"""
    close = ind["close"]
    sma_50 = ind["sma_50"]
    vol = ind["vol_20"]
    # ZEBRA: synthetic long with no extrinsic cost
    above_ma = (close > sma_50).astype(float)
    low_vol = (1 - _z(vol, 120).clip(0, 3) / 3)
    raw = above_ma * low_vol * 40 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s366_vanna_volga(ind):
    """366 | Vanna-Volga Implied Volatility Model"""
    vol = ind["vol_20"]
    close = ind["close"]
    ret = ind["ret_1"]
    # Vanna: sensitivity of delta to vol; Volga: sensitivity of vega to vol
    # Proxy: vol-return correlation
    vanna_proxy = ret.rolling(20).corr(vol.pct_change()).fillna(0)
    volga_proxy = _z(vol.rolling(20).std(), 60)
    raw = -vanna_proxy * 30 + volga_proxy * 25 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s367_convexity_adj(ind):
    """367 | Convexity Adjustment Trading in Rates"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Convexity: second derivative of price/yield
    price_accel = close.diff().diff()
    convexity = _z(price_accel.rolling(20).mean(), 60)
    vol_adj = ind["vol_dampener"]
    raw = convexity * vol_adj * 50
    return _clip_signal(raw)


def s368_cds_bond_basis(ind):
    """368 | CDS-Bond Basis Trade"""
    close = ind["close"]
    vol = ind["vol_20"]
    # CDS-bond basis proxy: risk premium from vol excess
    excess_vol = _z(vol, 120)
    mom = _z(ind["mom_20"], 60)
    # Basis trade: buy bond (positive carry) when risk premium high
    raw = excess_vol * 25 + mom * 30
    return _clip_signal(raw)


def s369_vol_smile(ind):
    """369 | Vol Smile Dynamics (Sticky Strike vs Sticky Delta)"""
    close = ind["close"]
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Sticky delta: vol follows price (negative correlation)
    vol_ret_corr = ret.rolling(60).corr(vol.pct_change()).fillna(0)
    # Sticky strike: vol independent of price
    sticky_delta = (vol_ret_corr < -0.3).astype(float)
    sticky_strike = (vol_ret_corr.abs() < 0.1).astype(float)
    raw = sticky_delta * ind["trend_score"] * 40 + sticky_strike * _z(ind["mom_20"], 60) * 30
    return _clip_signal(raw)


def s370_synthetic_long(ind):
    """370 | Synthetic Long via Deep ITM Calls + Cash"""
    close = ind["close"]
    sma_50 = ind["sma_50"]
    sma_200 = ind["sma_200"]
    vol = ind["vol_20"]
    # Synthetic long: bullish bias with vol-adjusted sizing
    bull = ((close > sma_50) & (sma_50 > sma_200)).astype(float)
    vol_adj = (1 - _z(vol, 120).clip(0, 2) / 2)
    raw = bull * vol_adj * 55
    return _clip_signal(raw)


def s371_pin_risk_gamma(ind):
    """371 | Pin Risk Gamma Squeeze Detector"""
    close = ind["close"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    # Gamma squeeze: price moves rapidly with extreme volume
    ret_5 = close.pct_change(5)
    vol_surge = (volume > 2.5 * vol_ma).astype(float)
    rapid_move = (ret_5.abs() > ret_5.rolling(60).std() * 2).astype(float)
    squeeze = vol_surge * rapid_move
    direction = np.sign(ret_5)
    raw = squeeze * direction * 45 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s372_vol_surface_arb(ind):
    """372 | Volatility Surface Arbitrage (Calendar + Skew)"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    ret = ind["ret_1"]
    # Calendar: vol term structure
    cal = _z(vol - vol_60, 120)
    # Skew: asymmetry
    skew = ret.rolling(60).apply(lambda x: pd.Series(x).skew(), raw=False).fillna(0)
    skew_z = _z(skew, 120)
    raw = cal * 25 + skew_z * 25 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s373_delta_adj_momentum(ind):
    """373 | Delta-Adjusted Momentum (Options-Informed)"""
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    close = ind["close"]
    # Delta proxy: probability of price staying above entry
    sma = ind["sma_20"]
    dist = (close - sma) / (vol * close).replace(0, np.nan)
    delta_proxy = 1 / (1 + np.exp(-dist * np.sqrt(252)))
    delta_adj_mom = _z(mom, 60) * delta_proxy
    raw = delta_adj_mom * 60
    return _clip_signal(raw)


def s374_implied_dividend(ind):
    """374 | Implied Dividend Extraction"""
    close = ind["close"]
    # Dividend proxy: periodic drops in price
    ret = ind["ret_1"]
    large_drop = (ret < ret.rolling(60).quantile(0.05)).astype(float)
    recovery = ret.shift(-5).rolling(5).sum().shift(5)  # post-drop recovery
    recovery_z = _z(recovery.fillna(0), 60)
    raw = large_drop * recovery_z * 30 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s375_tail_hedge_vix(ind):
    """375 | Tail Hedge Timing via VIX Term Structure"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # VIX term structure proxy: vol curve
    inverted = (vol > vol_60 * 1.1).astype(float)
    contango = (vol_60 > vol * 1.1).astype(float)
    # Inverted = stress, hedge; Contango = calm, risk-on
    raw = contango * 40 - inverted * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s376_var_swap_mtm(ind):
    """376 | Variance Swap Mark-to-Market Signal"""
    vol = ind["vol_20"]
    realized_var = vol ** 2
    implied_var = vol_60_sq = ind["vol_60"] ** 2
    pnl = realized_var - implied_var
    z = _z(pnl, 120)
    raw = z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s377_barrier_hedge(ind):
    """377 | Barrier Option Hedging Flow Signal"""
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    # Barrier proxy: price near round numbers or recent extremes
    hh = ind["hh"]
    ll = ind["ll"]
    near_high = ((hh - close) / ind["atr14"].replace(0, np.nan) < 1.0).astype(float)
    near_low = ((close - ll) / ind["atr14"].replace(0, np.nan) < 1.0).astype(float)
    # Barrier hedging flow pushes price away from barrier
    raw = near_low * 35 - near_high * 35 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s378_quanto_adj(ind):
    """378 | Quanto Adjustment Cross-Currency Signal"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Quanto proxy: correlation between asset and FX (approximated by vol comovement)
    vol_mom = vol.pct_change(10)
    price_mom = close.pct_change(10)
    corr = vol_mom.rolling(60).corr(price_mom).fillna(0)
    raw = _z(corr, 120) * 30 + _z(mom, 60) * 30
    return _clip_signal(raw)


def s379_swaption_straddle(ind):
    """379 | Swaption Straddle Breakeven Analysis"""
    vol = ind["vol_20"]
    atr = ind["atr_pct"]
    # Straddle breakeven: ATR vs vol
    breakeven = atr * np.sqrt(30)  # 30-day breakeven
    realized = vol * np.sqrt(30 / 252)
    edge = _z(realized - breakeven, 120)
    raw = edge * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s380_convert_arb(ind):
    """380 | Convertible Bond Arbitrage"""
    close = ind["close"]
    vol = ind["vol_20"]
    sma = ind["sma_50"]
    # Convert arb: long convert (delta ~0.5), short stock
    # Proxy: mean reversion with vol premium
    dist = _z(close - sma, 60)
    vol_premium = _z(vol, 120)
    raw = -dist * 30 + vol_premium * 25 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s381_realized_skew(ind):
    """381 | Realized Skewness Factor"""
    ret = ind["ret_1"]
    skew = ret.rolling(60).apply(lambda x: pd.Series(x).skew(), raw=False).fillna(0)
    skew_z = _z(skew, 120)
    # Negative skew = crash risk; positive skew = upside potential
    raw = skew_z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s382_xs_options_momentum(ind):
    """382 | Cross-Asset Momentum via Options Signals"""
    mom_5 = ind["mom_5"]
    mom_20 = ind["mom_20"]
    vol = ind["vol_20"]
    # Options-informed momentum: weight by vol (higher vol = stronger signal)
    vol_weight = vol / vol.rolling(60).mean().replace(0, np.nan)
    raw = _z(mom_5 + mom_20, 60) * vol_weight.clip(0.5, 2) * 45
    return _clip_signal(raw)


def s383_iv_mean_revert(ind):
    """383 | Implied Volatility Mean Reversion Calendar"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # IV mean reversion: vol reverts to longer-term average
    dev = _z(vol - vol_60, 120)
    raw = -dev * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s384_delta_gamma_vega(ind):
    """384 | Delta-Gamma-Vega Neutral Income"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    close = ind["close"]
    # Neutral income: low gamma (stable), positive theta
    gamma_proxy = ret.rolling(5).std() / vol.replace(0, np.nan)
    stable = (gamma_proxy < gamma_proxy.rolling(60).quantile(0.3)).astype(float)
    raw = stable * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s385_vol_carry(ind):
    """385 | Volatility Carry Cross-Asset"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Vol carry: sell implied (high), buy realized (low)
    carry = _z(vol_60 - vol, 120)
    raw = carry * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s386_mm_inventory(ind):
    """386 | Options Market Maker Inventory Signal"""
    close = ind["close"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    ret = ind["ret_1"]
    # MM inventory: contrarian at extremes with volume confirmation
    vol_imbalance = (volume - vol_ma) / vol_ma
    z_ret = _z(ret, 20)
    raw = -z_ret * (1 + vol_imbalance.clip(0, 3)) * 30
    return _clip_signal(raw)


def s387_xccy_basis(ind):
    """387 | Cross-Currency Basis Swap Signal"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_40"]
    # XCCY basis proxy: long-term momentum with vol adjustment
    raw = _z(mom, 120) * (1 - _z(vol, 120).clip(0, 2) / 2) * 55
    return _clip_signal(raw)


def s388_letf_rebalance(ind):
    """388 | Leveraged ETF Rebalancing Flow"""
    close = ind["close"]
    ret = ind["ret_1"]
    vol = ind["vol_20"]
    # LETF rebalancing: forced buying/selling at close
    large_ret = (ret.abs() > ret.rolling(60).std() * 1.5).astype(float)
    # Rebalancing flow is in same direction as return
    flow_dir = np.sign(ret) * large_ret
    # Fade next day
    raw = -flow_dir.shift(1) * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s389_synthetic_cdo(ind):
    """389 | Synthetic CDO Tranche Trading"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # CDO tranche proxy: tail risk vs mezzanine
    tail_risk = _z(vol, 120)
    correlation = vol / vol_60.replace(0, np.nan)
    corr_z = _z(correlation, 120)
    raw = -tail_risk * 25 + corr_z * 25 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s390_forward_start(ind):
    """390 | Forward Starting Options Signal"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    mom = ind["mom_20"]
    # Forward starting: vol term structure forward rate
    forward_vol = _z(vol_60 - vol * 0.5, 120)
    raw = forward_vol * 30 + _z(mom, 60) * 30
    return _clip_signal(raw)


def s391_letf_decay(ind):
    """391 | LETF Decay Alpha (Volatility Drag Capture)"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Vol drag: -0.5 * leverage^2 * variance
    vol_drag = -0.5 * vol ** 2 * 252
    drag_z = _z(vol_drag, 120)
    # When drag is extreme, fade the leveraged product
    raw = drag_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s392_vix_basis_momentum(ind):
    """392 | VIX Futures Basis Momentum"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # VIX basis = term structure slope
    basis = vol_60 - vol
    basis_mom = basis.diff(10)
    z = _z(basis_mom, 60)
    raw = z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s393_cms_curve(ind):
    """393 | Constant Maturity Swap Curve Signal"""
    close = ind["close"]
    # CMS proxy: multi-horizon return spread
    ret_20 = close.pct_change(20)
    ret_60 = close.pct_change(60)
    curve = _z(ret_60 - ret_20, 120)
    raw = curve * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s394_trs_funding(ind):
    """394 | Total Return Swap Funding Arbitrage"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_40"]
    # TRS funding: carry trade proxy
    carry = _z(mom, 120)
    funding_cost = _z(vol, 120)
    raw = carry * 30 - funding_cost * 20 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s395_autocallable_hedge(ind):
    """395 | Autocallable Hedging Flow Signal"""
    close = ind["close"]
    sma_50 = ind["sma_50"]
    vol = ind["vol_20"]
    # Autocallable: barrier knock-in near strike; delta hedging accelerates near barrier
    near_barrier = ((close / sma_50.replace(0, np.nan) - 1).abs() < 0.05).astype(float)
    vol_adj = _z(vol, 120)
    direction = np.sign(close - sma_50)
    raw = near_barrier * (-direction) * vol_adj * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s396_vol_var_spread(ind):
    """396 | Volatility Swap vs Variance Swap Spread"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Convexity adjustment: variance > vol^2 when kurtosis > 3
    kurt = ret.rolling(60).apply(lambda x: pd.Series(x).kurtosis(), raw=False).fillna(0)
    kurt_z = _z(kurt, 120)
    # High kurtosis: variance swap > vol swap; trade the spread
    raw = -kurt_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s397_options_beta(ind):
    """397 | Options-Implied Beta Trading"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Beta proxy: vol relative to market
    vol_rel = vol / vol.rolling(120).mean().replace(0, np.nan)
    beta_z = _z(vol_rel, 120)
    # Low beta outperformance in stress; high beta in calm
    vol_regime = _z(vol, 120)
    raw = (1 - vol_regime) * beta_z * 25 + _z(mom, 60) * 30
    return _clip_signal(raw)


def s398_vol_risk_parity(ind):
    """398 | Volatility Risk Parity Portfolio"""
    vol = ind["vol_20"]
    trend = ind["trend_score"]
    # Risk parity: inverse vol allocation
    inv_vol = 1 / vol.replace(0, np.nan)
    inv_vol_z = _z(inv_vol, 60)
    raw = inv_vol_z * 25 + trend * 35
    return _clip_signal(raw)


def s399_reverse_convert(ind):
    """399 | Reverse Convertible Hedging Signal"""
    close = ind["close"]
    sma = ind["sma_50"]
    vol = ind["vol_20"]
    # Reverse convertible: short put embedded; delta hedging
    dist = (close - sma) / (vol * close + 1e-10)
    dist_z = _z(dist, 60)
    raw = dist_z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s400_greeks_dashboard(ind):
    """400 | Greeks-Based Multi-Asset Risk Dashboard"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    ret = ind["ret_1"]
    # Delta: trend direction
    delta = ind["trend_score"]
    # Gamma: rate of change of delta
    gamma = _z(mom.diff(5), 40)
    # Vega: vol sensitivity
    vega = _z(vol, 120)
    # Theta: time decay (mean reversion pressure)
    theta = -_z(close - ind["sma_50"], 60)
    raw = delta * 30 + gamma * 15 + (-vega) * 15 + theta * 10
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    351: ("Variance Swap", s351_variance_swap),
    352: ("Gamma Scalping", s352_gamma_scalp),
    353: ("Skew Risk Reversal", s353_skew_risk_reversal),
    354: ("Term Structure Roll", s354_term_structure_roll),
    355: ("Put-Write Premium", s355_put_write),
    356: ("Iron Condor", s356_iron_condor),
    357: ("Calendar Spread", s357_calendar_spread),
    358: ("Butterfly Pinning", s358_butterfly_pinning),
    359: ("Dispersion Premium", s359_dispersion),
    360: ("Straddle Momentum", s360_straddle_momentum),
    361: ("Collar Overlay", s361_collar_overlay),
    362: ("Backspread Convex", s362_backspread),
    363: ("Jade Lizard", s363_jade_lizard),
    364: ("Broken Wing Bfly", s364_broken_wing_butterfly),
    365: ("ZEBRA Synthetic", s365_zebra),
    366: ("Vanna-Volga", s366_vanna_volga),
    367: ("Convexity Adj", s367_convexity_adj),
    368: ("CDS-Bond Basis", s368_cds_bond_basis),
    369: ("Vol Smile Dynamics", s369_vol_smile),
    370: ("Synthetic Long", s370_synthetic_long),
    371: ("Pin Risk Gamma", s371_pin_risk_gamma),
    372: ("Vol Surface Arb", s372_vol_surface_arb),
    373: ("Delta-Adj Momentum", s373_delta_adj_momentum),
    374: ("Implied Dividend", s374_implied_dividend),
    375: ("Tail Hedge VIX", s375_tail_hedge_vix),
    376: ("Var Swap MTM", s376_var_swap_mtm),
    377: ("Barrier Hedge Flow", s377_barrier_hedge),
    378: ("Quanto Adjustment", s378_quanto_adj),
    379: ("Swaption Straddle", s379_swaption_straddle),
    380: ("Convertible Arb", s380_convert_arb),
    381: ("Realized Skewness", s381_realized_skew),
    382: ("XS Options Mom", s382_xs_options_momentum),
    383: ("IV Mean Reversion", s383_iv_mean_revert),
    384: ("Delta-Gamma-Vega", s384_delta_gamma_vega),
    385: ("Vol Carry", s385_vol_carry),
    386: ("MM Inventory", s386_mm_inventory),
    387: ("XCCY Basis", s387_xccy_basis),
    388: ("LETF Rebalance", s388_letf_rebalance),
    389: ("Synthetic CDO", s389_synthetic_cdo),
    390: ("Forward Start", s390_forward_start),
    391: ("LETF Decay Alpha", s391_letf_decay),
    392: ("VIX Basis Mom", s392_vix_basis_momentum),
    393: ("CMS Curve Signal", s393_cms_curve),
    394: ("TRS Funding Arb", s394_trs_funding),
    395: ("Autocallable Hedge", s395_autocallable_hedge),
    396: ("Vol-Var Spread", s396_vol_var_spread),
    397: ("Options Beta", s397_options_beta),
    398: ("Vol Risk Parity", s398_vol_risk_parity),
    399: ("Reverse Convert", s399_reverse_convert),
    400: ("Greeks Dashboard", s400_greeks_dashboard),
}
