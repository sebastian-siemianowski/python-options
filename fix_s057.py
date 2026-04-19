"""Fix S057 Candlestick Ensemble - 0 trades due to too-strict thresholds."""
import re

path = "src/indicators/families/mean_reversion.py"
with open(path, "r") as f:
    content = f.read()

OLD = '''    # --- Signal composition ---
    # Event: strong pattern in right zone
    long_event = (bull_score > 0.3) & (oversold_zone > 0)
    short_event = (bear_score > 0.3) & (overbought_zone > 0)
    
    # Score intensity
    long_intensity = bull_score * vol_confirm * (0.5 + oversold_zone * 0.5)
    short_intensity = bear_score * vol_confirm * (0.5 + overbought_zone * 0.5)
    
    # Persistence: patterns effective for 3 bars
    long_persist = long_intensity.rolling(3, min_periods=1).max()
    short_persist = short_intensity.rolling(3, min_periods=1).max()
    
    # --- Continuous baseline from pattern flow ---
    pattern_flow = bull_score.ewm(span=10).mean() - bear_score.ewm(span=10).mean()
    continuous = pattern_flow * 25
    
    # --- Compose ---
    event_sig = long_persist * 55 - short_persist * 55
    raw = event_sig + continuous + ind["trend_score"] * 10'''

NEW = '''    # --- Signal composition ---
    # Three tiers: strong (pattern + zone), moderate (pattern alone), continuous
    
    # Tier 1: Strong event - pattern in matching RSI zone (full strength)
    strong_long = bull_score.where(oversold_zone > 0, 0.0) * vol_confirm
    strong_short = bear_score.where(overbought_zone > 0, 0.0) * vol_confirm
    
    # Tier 2: Moderate event - any pattern with minimum score 0.10 (60% strength)
    mod_long = bull_score.where(bull_score >= 0.10, 0.0) * vol_confirm * 0.6
    mod_short = bear_score.where(bear_score >= 0.10, 0.0) * vol_confirm * 0.6
    
    # Combine tiers (strong overrides moderate where present)
    long_intensity = strong_long.where(strong_long > 0, mod_long)
    short_intensity = strong_short.where(strong_short > 0, mod_short)
    
    # Persistence: patterns effective for 5 bars (candlestick reversals need time)
    long_persist = long_intensity.rolling(5, min_periods=1).max()
    short_persist = short_intensity.rolling(5, min_periods=1).max()
    
    # --- Continuous baseline from pattern flow ---
    pattern_flow = bull_score.ewm(span=8).mean() - bear_score.ewm(span=8).mean()
    
    # Also add body momentum (large bullish bodies = positive pressure)
    body_mom = body.ewm(span=10).mean() / atr.ewm(span=20).mean()
    body_mom = body_mom.clip(-1, 1)
    
    continuous = _norm(pattern_flow, span=40) * 18 + body_mom * 12
    
    # --- Compose ---
    event_sig = long_persist * 85 - short_persist * 85
    raw = event_sig + continuous + ind["trend_score"] * 8'''

assert OLD in content, "S057 old text not found!"
content = content.replace(OLD, NEW)
with open(path, "w") as f:
    f.write(content)
print("S057 fixed OK")
