"""
SECTION IX: CROSS-ASSET & MACRO STRATEGIES (401-450)
Each function takes ind dict -> pd.Series[-100, +100].
All macro/cross-asset concepts are proxied from single-asset OHLCV data.
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

def s401_global_macro_regime(ind):
    """401 | Global Macro Regime Allocation Framework"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    mom = ind["mom_40"]
    trend = ind["trend_score"]
    # Macro regime: vol level + trend + vol term structure
    vol_regime = _z(vol_60, 252)
    term = _z(vol_60 - vol, 120)
    raw = (-vol_regime * 0.3 + _z(mom, 120) * 0.4 + term * 0.3) * 70
    return _clip_signal(raw)


def s402_dollar_smile(ind):
    """402 | Dollar Smile Framework (FX Macro)"""
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Dollar smile: USD strengthens in risk-off AND strong growth
    vol_z = _z(vol, 120)
    risk_off = (vol_z > 1.0).astype(float)
    strong_growth = (_z(mom, 120) > 1.0).astype(float)
    # Middle: USD weak
    raw = (risk_off + strong_growth - 1) * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s403_carry_universal(ind):
    """403 | Carry Factor Universal (Multi-Asset)"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom_5 = ind["mom_5"]
    mom_40 = ind["mom_40"]
    # Carry: short-term vs long-term momentum differential
    carry = _z(mom_5 - mom_40 / 8, 60)
    vol_adj = ind["vol_dampener"]
    raw = carry * vol_adj * 55
    return _clip_signal(raw)


def s404_global_liquidity(ind):
    """404 | Global Liquidity Cycle Trading"""
    vol = ind["vol_20"]
    volume = ind["volume"]
    vol_ma = volume.rolling(60).mean().replace(0, np.nan)
    # Liquidity proxy: volume trend + volatility
    vol_trend = _z(volume / vol_ma, 120)
    vol_regime = -_z(vol, 120)
    raw = (vol_trend * 0.4 + vol_regime * 0.3 + ind["trend_score"] * 0.3) * 65
    return _clip_signal(raw)


def s405_copper_gold_ratio(ind):
    """405 | Copper/Gold Ratio Growth Signal"""
    close = ind["close"]
    mom = ind["mom_20"]
    vol = ind["vol_20"]
    # Growth proxy: momentum relative to vol (cyclical vs defensive)
    growth = _z(mom / vol.replace(0, np.nan), 120)
    raw = growth * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s406_em_contagion(ind):
    """406 | Emerging Market Contagion Early Warning"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Contagion proxy: vol clustering + tail events
    tail = (ret.abs() > ret.rolling(60).std() * 2.5).astype(float).rolling(10).sum()
    vol_cluster = vol.rolling(10).std() / vol.rolling(60).std().replace(0, np.nan)
    contagion = _z(tail + vol_cluster, 120)
    raw = -contagion * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s407_pmi_momentum(ind):
    """407 | Global PMI Momentum Cross-Asset"""
    close = ind["close"]
    # PMI proxy: breadth of positive momentum periods
    mom_5 = ind["mom_5"]
    mom_10 = ind["mom_10"]
    mom_20 = ind["mom_20"]
    breadth = ((mom_5 > 0).astype(float) + (mom_10 > 0).astype(float) +
               (mom_20 > 0).astype(float)) / 3
    breadth_z = _z(breadth, 120)
    raw = breadth_z * 55
    return _clip_signal(raw)


def s408_ted_spread(ind):
    """408 | TED Spread Systemic Risk Signal"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # TED spread proxy: short-term funding stress from vol
    stress = _z(vol / vol_60.replace(0, np.nan), 120)
    # High TED = systemic risk; reduce exposure
    raw = -stress * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s409_risk_appetite(ind):
    """409 | Risk Appetite Indicator Composite"""
    rsi = ind["rsi"]
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    adx = ind["adx"]
    # Risk appetite: positive momentum + low vol + trending
    risk_on = _z(mom, 60) * 0.3 + (-_z(vol, 120)) * 0.3 + (adx / 50 - 0.5) * 0.2 + ((rsi - 50) / 50) * 0.2
    raw = risk_on * 65
    return _clip_signal(raw)


def s410_fiscal_impulse(ind):
    """410 | Fiscal Impulse Trading Signal"""
    close = ind["close"]
    mom = ind["mom_40"]
    vol = ind["vol_60"]
    # Fiscal impulse proxy: acceleration of long-term momentum
    impulse = mom.diff(20)
    z = _z(impulse, 120)
    raw = z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s411_ism_new_orders(ind):
    """411 | ISM New Orders/Inventories Spread"""
    close = ind["close"]
    # ISM proxy: short-term momentum acceleration
    mom_5 = ind["mom_5"]
    mom_20 = ind["mom_20"]
    new_orders = _z(mom_5, 40)  # leading indicator
    inventories = _z(mom_20.shift(10), 40)  # lagging
    spread = new_orders - inventories
    raw = spread * 45
    return _clip_signal(raw)


def s412_baltic_dry(ind):
    """412 | Baltic Dry Index Shipping Signal"""
    close = ind["close"]
    volume = ind["volume"]
    # Shipping proxy: volume momentum as demand indicator
    vol_mom = volume.pct_change(20)
    z = _z(vol_mom, 120)
    raw = z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s413_tips_breakeven(ind):
    """413 | TIPS Breakeven Inflation Trading"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Inflation proxy: positive momentum with low vol = reflation
    reflation = _z(ind["mom_40"], 120) * (1 - _z(vol, 120).clip(0, 2) / 2)
    raw = reflation * 45 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s414_term_premium(ind):
    """414 | Term Premium Decomposition Signal"""
    vol_20 = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Term premium proxy: vol term structure
    term = _z(vol_60 - vol_20, 120)
    mom = _z(ind["mom_40"], 120)
    raw = term * 30 + mom * 30
    return _clip_signal(raw)


def s415_real_rate_diff(ind):
    """415 | Real Rate Differential FX Strategy"""
    close = ind["close"]
    vol = ind["vol_20"]
    mom = ind["mom_40"]
    # Real rate proxy: momentum adjusted for vol
    real_rate = _z(mom, 120) - _z(vol, 120) * 0.5
    raw = real_rate * 50
    return _clip_signal(raw)


def s416_fed_funds_implied(ind):
    """416 | Fed Funds Futures Implied Policy Path"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Policy path proxy: vol term structure slope changes
    slope = vol_60 - vol
    slope_mom = slope.diff(20)
    z = _z(slope_mom, 120)
    raw = z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s417_vol_transmission(ind):
    """417 | Cross-Market Volatility Transmission"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Vol transmission proxy: persistence of vol shocks
    vol_auto = vol.rolling(20).corr(vol.shift(5)).fillna(0)
    persistence = _z(vol_auto, 120)
    raw = -persistence * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s418_sofr_spread(ind):
    """418 | Eurodollar/SOFR Spread Arbitrage"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Spread proxy: short vs long vol as funding rate proxy
    spread = _z(vol / vol_60.replace(0, np.nan) - 1, 120)
    raw = -spread * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s419_earnings_yield_spread(ind):
    """419 | Cross-Country Equity Earnings Yield Spread"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Earnings yield proxy: inverse of P/E ~ inverse of momentum
    ey_proxy = -_z(close.pct_change(252), 252)
    vol_adj = ind["vol_dampener"]
    raw = ey_proxy * vol_adj * 50
    return _clip_signal(raw)


def s420_oil_equity_corr(ind):
    """420 | Oil-Equity Correlation Regime"""
    close = ind["close"]
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Correlation regime proxy: vol-return relationship
    corr = ret.rolling(60).corr(vol.pct_change()).fillna(0)
    corr_z = _z(corr, 120)
    raw = -corr_z * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s421_yen_risk_off(ind):
    """421 | Japanese Yen Risk-Off Signal"""
    vol = ind["vol_20"]
    mom = ind["mom_20"]
    # Risk-off proxy: vol spike + negative momentum
    risk_off = (_z(vol, 120) > 1.0).astype(float) * (mom < 0).astype(float)
    risk_on = (_z(vol, 120) < -0.5).astype(float) * (mom > 0).astype(float)
    raw = risk_on * 45 - risk_off * 45
    return _clip_signal(raw)


def s422_gold_silver_ratio(ind):
    """422 | Gold/Silver Ratio Macro Signal"""
    close = ind["close"]
    vol = ind["vol_20"]
    # G/S ratio proxy: relative momentum strength
    long_mom = _z(ind["mom_40"], 120)
    short_mom = _z(ind["mom_10"], 60)
    ratio_signal = long_mom - short_mom  # expansion = risk-off
    raw = -ratio_signal * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s423_credit_spread_mom(ind):
    """423 | Credit Spread Momentum Cross-Asset"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    # Credit spread proxy: vol excess over baseline
    spread = vol - vol_60 * 0.7
    spread_mom = spread.diff(10)
    z = _z(spread_mom, 60)
    # Tightening spreads (negative momentum) = bullish
    raw = -z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s424_vix_equity_regime(ind):
    """424 | VIX-Equity Correlation Regime Switch"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # VIX-equity: typically negative correlation
    corr = ret.rolling(60).corr(vol.diff()).fillna(0)
    regime = _z(corr, 120)
    # Normal (negative corr): trend follow; Decorrelation: reduce
    normal = (corr < -0.2).astype(float)
    raw = normal * ind["trend_score"] * 45 + (1 - normal) * _z(ind["mom_10"], 60) * 25
    return _clip_signal(raw)


def s425_intermarket_quad(ind):
    """425 | Intermarket Analysis Quad-Screen"""
    mom_5 = ind["mom_5"]
    mom_20 = ind["mom_20"]
    vol = ind["vol_20"]
    volume = ind["volume"]
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    # Four screens: momentum, vol, volume, trend
    screen1 = _z(mom_20, 60)
    screen2 = -_z(vol, 120)
    screen3 = _z(volume / vol_ma, 60)
    screen4 = ind["trend_score"]
    raw = (screen1 * 0.3 + screen2 * 0.2 + screen3 * 0.2 + screen4 * 0.3) * 65
    return _clip_signal(raw)


def s426_commodity_supercycle(ind):
    """426 | Commodity Super-Cycle Positioning"""
    close = ind["close"]
    # Super-cycle: very long-term trend
    sma_200 = ind["sma_200"]
    above_200 = (close > sma_200).astype(float)
    mom_long = _z(close.pct_change(120), 252)
    raw = mom_long * 35 + above_200 * 25
    return _clip_signal(raw)


def s427_yen_carry(ind):
    """427 | Yen Carry Trade Monitoring Dashboard"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    mom = ind["mom_40"]
    # Carry trade: profit from low vol, lose from vol spikes
    carry_env = -_z(vol, 120)
    vol_shock = (vol > vol.rolling(60).quantile(0.9)).astype(float)
    raw = carry_env * (1 - vol_shock * 0.8) * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s428_ppp_valuation(ind):
    """428 | Purchasing Power Parity FX Valuation"""
    close = ind["close"]
    # PPP proxy: long-term mean reversion
    sma_200 = ind["sma_200"]
    dev = _z(close / sma_200.replace(0, np.nan) - 1, 252)
    # Fade extremes, trend follow in middle
    extreme = (dev.abs() > 1.5).astype(float)
    raw = extreme * (-dev) * 30 + (1 - extreme) * ind["trend_score"] * 40
    return _clip_signal(raw)


def s429_current_account(ind):
    """429 | Current Account Imbalance Signal"""
    close = ind["close"]
    # CA proxy: long-term momentum as trade balance indicator
    long_mom = _z(close.pct_change(120), 252)
    vol_adj = ind["vol_dampener"]
    raw = long_mom * vol_adj * 55
    return _clip_signal(raw)


def s430_semi_cycle(ind):
    """430 | Semiconductor Cycle Leading Indicator"""
    close = ind["close"]
    mom = ind["mom_20"]
    eff = ind["efficiency"]
    # Semi cycle: cyclical momentum + efficiency
    cycle = _z(mom, 60) * 0.5 + _z(eff, 60) * 0.3 + ind["trend_score"] * 0.2
    raw = cycle * 65
    return _clip_signal(raw)


def s431_china_credit(ind):
    """431 | China Credit Impulse Global Signal"""
    vol = ind["vol_20"]
    mom = ind["mom_40"]
    # Credit impulse proxy: acceleration of momentum
    impulse = mom.diff(20)
    z = _z(impulse, 120)
    raw = z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s432_fund_flow(ind):
    """432 | Global Fund Flow Momentum"""
    volume = ind["volume"]
    vol_ma = volume.rolling(60).mean().replace(0, np.nan)
    ret = ind["ret_1"]
    # Fund flow proxy: directional volume
    flow = volume * np.sign(ret)
    flow_ma = flow.rolling(20).mean()
    z = _z(flow_ma, 60)
    raw = z * 45 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s433_sovereign_cds(ind):
    """433 | Sovereign CDS Contagion Network"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # CDS proxy: tail risk clustering
    tail = (ret < ret.rolling(60).quantile(0.05)).astype(float)
    cluster = tail.rolling(20).sum()
    z = _z(cluster, 120)
    raw = -z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s434_lei_momentum(ind):
    """434 | Leading Economic Index Momentum"""
    close = ind["close"]
    mom_10 = ind["mom_10"]
    mom_40 = ind["mom_40"]
    # LEI proxy: combination of leading indicators
    lei = (0.4 * _z(mom_10, 60) + 0.3 * _z(mom_40, 120) +
           0.3 * -_z(ind["vol_20"], 120))
    raw = lei * 60
    return _clip_signal(raw)


def s435_commodity_roll(ind):
    """435 | Commodity Term Structure Roll Yield"""
    close = ind["close"]
    mom_5 = ind["mom_5"]
    mom_20 = ind["mom_20"]
    # Roll yield proxy: short-term vs medium-term momentum differential
    roll = _z(mom_5 - mom_20 * 0.25, 60)
    raw = roll * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s436_us_china_decouple(ind):
    """436 | US-China Decoupling Basket"""
    close = ind["close"]
    vol = ind["vol_20"]
    # Decoupling proxy: regime-dependent momentum
    vol_z = _z(vol, 120)
    mom = _z(ind["mom_20"], 60)
    # High vol divergence: momentum matters more
    raw = mom * (1 + vol_z.clip(0, 2) * 0.3) * 45
    return _clip_signal(raw)


def s437_multi_horizon_mom(ind):
    """437 | Multi-Horizon Momentum Ensemble"""
    mom_5 = _z(ind["mom_5"], 40)
    mom_10 = _z(ind["mom_10"], 60)
    mom_20 = _z(ind["mom_20"], 60)
    mom_40 = _z(ind["mom_40"], 120)
    # Horizon-weighted ensemble
    raw = (0.15 * mom_5 + 0.25 * mom_10 + 0.35 * mom_20 + 0.25 * mom_40) * 65
    return _clip_signal(raw)


def s438_overnight_intraday_decomp(ind):
    """438 | Overnight-Intraday Return Decomposition"""
    close = ind["close"]
    open_p = ind["open"]
    overnight = (open_p - close.shift(1)) / close.shift(1) * 100
    intraday = (close - open_p) / open_p * 100
    on_cum = overnight.rolling(20).sum()
    id_cum = intraday.rolling(20).sum()
    # Overnight premium: institutional; Intraday: retail
    raw = _z(on_cum, 60) * 30 + _z(id_cum, 60) * 25 + ind["trend_score"] * 15
    return _clip_signal(raw)


def s439_central_bank_sentiment(ind):
    """439 | Central Bank Communication Sentiment"""
    vol = ind["vol_20"]
    mom = ind["mom_40"]
    # CB sentiment proxy: vol direction after large moves
    vol_change = vol.pct_change(10)
    mom_z = _z(mom, 120)
    # Hawkish (vol rising, mom falling) vs Dovish (vol falling, mom rising)
    sentiment = -_z(vol_change, 60) + mom_z
    raw = sentiment / 2 * 55
    return _clip_signal(raw)


def s440_equity_credit_basis(ind):
    """440 | Equity-Credit Basis Trade"""
    vol = ind["vol_20"]
    close = ind["close"]
    # Equity-credit: vol as credit spread proxy
    vol_z = _z(vol, 120)
    price_z = _z(close, 120)
    # Basis: price vs vol divergence
    basis = price_z + vol_z  # positive = equity rich, credit cheap
    raw = -basis * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s441_dividend_futures(ind):
    """441 | Dividend Futures Curve Signal"""
    close = ind["close"]
    # Dividend proxy: mean reversion to 200MA
    sma_200 = ind["sma_200"]
    yield_proxy = _z(sma_200 / close.replace(0, np.nan) - 1, 252)
    raw = yield_proxy * 35 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s442_tail_hedge_put(ind):
    """442 | Tail Risk Hedging via Systematic Put Buying"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    ret = ind["ret_1"]
    # Tail hedge timing: buy protection when cheap (low vol), sell when expensive
    vol_z = _z(vol, 120)
    cheap_prot = (vol_z < -0.5).astype(float)
    expensive_prot = (vol_z > 1.0).astype(float)
    raw = cheap_prot * 35 - expensive_prot * 20 + ind["trend_score"] * 25
    return _clip_signal(raw)


def s443_capital_flow(ind):
    """443 | Cross-Border Capital Flow Tracking"""
    volume = ind["volume"]
    vol_ma = volume.rolling(60).mean().replace(0, np.nan)
    mom = ind["mom_40"]
    # Capital flow proxy: sustained volume + momentum
    flow = _z(volume / vol_ma * np.sign(mom.fillna(0)), 120)
    raw = flow * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s444_housing_lead(ind):
    """444 | Housing Market Leading Indicators"""
    close = ind["close"]
    # Housing proxy: long-cycle momentum changes
    long_mom = close.pct_change(120)
    mom_change = long_mom.diff(60)
    z = _z(mom_change, 252)
    raw = z * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s445_sector_rs_rotation(ind):
    """445 | Cross-Sector Relative Strength Rotation"""
    mom_10 = ind["mom_10"]
    mom_20 = ind["mom_20"]
    mom_40 = ind["mom_40"]
    vol = ind["vol_20"]
    # RS: acceleration of multi-horizon momentum
    accel = _z(mom_10 - mom_40 / 4, 60)
    vol_adj = ind["vol_dampener"]
    raw = accel * vol_adj * 55
    return _clip_signal(raw)


def s446_monetary_divergence(ind):
    """446 | Monetary Policy Divergence FX Strategy"""
    vol = ind["vol_20"]
    vol_60 = ind["vol_60"]
    mom = ind["mom_40"]
    # Policy divergence proxy: vol term structure + trend
    divergence = _z(vol_60 - vol, 120) + _z(mom, 120)
    raw = divergence / 2 * 55
    return _clip_signal(raw)


def s447_cat_bond_spread(ind):
    """447 | Catastrophe Bond Spread Signal"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Cat bond: tail risk premium
    tail_risk = (ret < ret.rolling(60).quantile(0.05)).astype(float).rolling(20).mean()
    tail_z = _z(tail_risk, 120)
    # Buy after tail risk events (premium capture)
    raw = tail_z * 30 + ind["trend_score"] * 30
    return _clip_signal(raw)


def s448_earnings_revision(ind):
    """448 | Global Earnings Revision Breadth"""
    close = ind["close"]
    mom = ind["mom_20"]
    # Earnings revision proxy: momentum breadth
    positive_mom = (mom > 0).astype(float).rolling(20).mean()
    breadth = _z(positive_mom, 120)
    raw = breadth * 55
    return _clip_signal(raw)


def s449_geopolitical_premium(ind):
    """449 | Geopolitical Risk Premium Extraction"""
    vol = ind["vol_20"]
    ret = ind["ret_1"]
    # Geopolitical proxy: vol spike + large negative return
    shock = ((ret < -ret.rolling(60).std() * 2) &
             (vol > vol.rolling(60).quantile(0.8))).astype(float)
    post_shock = shock.rolling(30).sum()
    # Premium capture: buy after geopolitical shock
    raw = post_shock.clip(0, 3) / 3 * 40 + ind["trend_score"] * 20
    return _clip_signal(raw)


def s450_climate_transition(ind):
    """450 | Climate Transition Risk Factor"""
    close = ind["close"]
    vol = ind["vol_20"]
    eff = ind["efficiency"]
    # Transition risk proxy: quality factor + low vol
    quality = _z(eff, 60)
    low_vol = -_z(vol, 120)
    raw = (quality * 0.5 + low_vol * 0.3 + ind["trend_score"] * 0.2) * 60
    return _clip_signal(raw)


# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    401: ("Global Macro Regime", s401_global_macro_regime),
    402: ("Dollar Smile", s402_dollar_smile),
    403: ("Carry Universal", s403_carry_universal),
    404: ("Global Liquidity", s404_global_liquidity),
    405: ("Copper/Gold Growth", s405_copper_gold_ratio),
    406: ("EM Contagion", s406_em_contagion),
    407: ("PMI Momentum", s407_pmi_momentum),
    408: ("TED Spread Risk", s408_ted_spread),
    409: ("Risk Appetite", s409_risk_appetite),
    410: ("Fiscal Impulse", s410_fiscal_impulse),
    411: ("ISM New Orders", s411_ism_new_orders),
    412: ("Baltic Dry", s412_baltic_dry),
    413: ("TIPS Breakeven", s413_tips_breakeven),
    414: ("Term Premium", s414_term_premium),
    415: ("Real Rate Diff", s415_real_rate_diff),
    416: ("Fed Funds Implied", s416_fed_funds_implied),
    417: ("Vol Transmission", s417_vol_transmission),
    418: ("SOFR Spread", s418_sofr_spread),
    419: ("Earnings Yield Spread", s419_earnings_yield_spread),
    420: ("Oil-Equity Corr", s420_oil_equity_corr),
    421: ("Yen Risk-Off", s421_yen_risk_off),
    422: ("Gold/Silver Ratio", s422_gold_silver_ratio),
    423: ("Credit Spread Mom", s423_credit_spread_mom),
    424: ("VIX-Equity Regime", s424_vix_equity_regime),
    425: ("Intermarket Quad", s425_intermarket_quad),
    426: ("Commodity Supercycle", s426_commodity_supercycle),
    427: ("Yen Carry Monitor", s427_yen_carry),
    428: ("PPP Valuation", s428_ppp_valuation),
    429: ("Current Account", s429_current_account),
    430: ("Semi Cycle Lead", s430_semi_cycle),
    431: ("China Credit", s431_china_credit),
    432: ("Fund Flow Mom", s432_fund_flow),
    433: ("Sovereign CDS", s433_sovereign_cds),
    434: ("LEI Momentum", s434_lei_momentum),
    435: ("Commodity Roll", s435_commodity_roll),
    436: ("US-China Decouple", s436_us_china_decouple),
    437: ("Multi-Horizon Mom", s437_multi_horizon_mom),
    438: ("Overnight/Intraday", s438_overnight_intraday_decomp),
    439: ("CB Sentiment", s439_central_bank_sentiment),
    440: ("Equity-Credit Basis", s440_equity_credit_basis),
    441: ("Dividend Futures", s441_dividend_futures),
    442: ("Tail Hedge Put", s442_tail_hedge_put),
    443: ("Capital Flow", s443_capital_flow),
    444: ("Housing Lead", s444_housing_lead),
    445: ("Sector RS Rotation", s445_sector_rs_rotation),
    446: ("Monetary Divergence", s446_monetary_divergence),
    447: ("Cat Bond Spread", s447_cat_bond_spread),
    448: ("Earnings Revision", s448_earnings_revision),
    449: ("Geopolitical Premium", s449_geopolitical_premium),
    450: ("Climate Transition", s450_climate_transition),
}
