"""Fix S054 - regime filter too strict, producing 0 trades."""
filepath = "src/indicators/families/mean_reversion.py"
with open(filepath, "r") as f:
    content = f.read()

old_054 = '''    # --- Regime gate: AC < -0.05 AND Hurst < 0.45 ---
    mr_regime = ((autocorr < -0.05) & (hurst < 0.45)).astype(float)
    
    # Softer version: partially trade if one condition met
    soft_regime = ((autocorr < 0.0) | (hurst < 0.50)).astype(float) * 0.4
    regime_strength = (mr_regime + soft_regime * (1 - mr_regime)).clip(0, 1)
    
    # --- Event signals ---
    long_event = ((z_spread < -2.0) & (regime_strength > 0.3)).astype(float)
    short_event = ((z_spread > 2.0) & (regime_strength > 0.3)).astype(float)
    emergency = (z_spread.abs() > 4.0).astype(float)
    
    # Persistence
    long_persist = long_event.rolling(8, min_periods=1).max()
    short_persist = short_event.rolling(8, min_periods=1).max()
    
    # --- Continuous baseline ---
    continuous = -z_spread * regime_strength * 18
    
    # --- Compose ---
    event_sig = long_persist * 40 - short_persist * 40
    raw = (event_sig + continuous) * regime_strength
    
    # Emergency flatten
    raw = raw * (1 - emergency * 0.9)'''

new_054 = '''    # --- Regime gate ---
    # Strong MR regime: AC < -0.05 AND Hurst < 0.45 -> full signal
    strong_mr = ((autocorr < -0.05) & (hurst < 0.45)).astype(float)
    # Moderate MR: either condition alone -> 60% signal
    moderate_mr = ((autocorr < 0.05) | (hurst < 0.50)).astype(float) * 0.6
    # Weak MR fallback: always-on base for any non-trending asset -> 30%
    weak_mr = (hurst < 0.55).astype(float) * 0.3
    regime_strength = (strong_mr + moderate_mr * (1 - strong_mr) + weak_mr * (1 - strong_mr) * (1 - moderate_mr.clip(0,1))).clip(0, 1)
    
    # --- Event signals: Z extreme + any MR regime ---
    long_event = ((z_spread < -1.8) & (regime_strength > 0.2)).astype(float)
    short_event = ((z_spread > 1.8) & (regime_strength > 0.2)).astype(float)
    emergency = (z_spread.abs() > 4.0).astype(float)
    
    # Deeper extremes get extra intensity
    long_depth = (-z_spread - 1.8).clip(0, 2) / 2  # 0 at -1.8, 1 at -3.8
    short_depth = (z_spread - 1.8).clip(0, 2) / 2
    
    # Persistence: hold event for up to 8 bars
    long_persist = long_event.rolling(8, min_periods=1).max()
    short_persist = short_event.rolling(8, min_periods=1).max()
    long_depth_p = long_depth.rolling(8, min_periods=1).max()
    short_depth_p = short_depth.rolling(8, min_periods=1).max()
    
    # --- Continuous baseline: proportional to z, always on ---
    continuous = -z_spread.clip(-3, 3) / 3 * 25 * regime_strength
    
    # --- Compose ---
    event_sig = long_persist * (35 + long_depth_p * 15) - short_persist * (35 + short_depth_p * 15)
    raw = event_sig * regime_strength + continuous
    
    # Emergency flatten
    raw = raw * (1 - emergency * 0.9)'''

assert old_054 in content, "S054 old text not found!"
content = content.replace(old_054, new_054, 1)
print("S054 fixed OK")

with open(filepath, "w") as f:
    f.write(content)
