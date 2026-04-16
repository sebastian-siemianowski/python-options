#!/usr/bin/env python3
"""Diagnostic: trace why forecasts are flat (+0.0% for short horizons)."""
import sys, os, json, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

TUNE_DIR = os.path.join("src", "data", "tune")

# Pick a few representative assets
test_assets = ["AAPL", "NVDA", "TSLA", "SPY", "GOOGL", "MSFT"]

print("=" * 80)
print("DIAGNOSTIC: Tracing flat forecast root causes")
print("=" * 80)

for sym in test_assets:
    cache_path = os.path.join(TUNE_DIR, f"{sym}.json")
    if not os.path.exists(cache_path):
        print(f"\n{sym}: NO CACHE FILE")
        continue
    
    with open(cache_path) as f:
        data = json.load(f)
    
    has_bma = data.get("has_bma", False)
    global_data = data.get("global", {})
    regime_data = data.get("regime", {})
    meta = data.get("meta", {})
    
    print(f"\n{'=' * 60}")
    print(f"ASSET: {sym}")
    print(f"{'=' * 60}")
    print(f"  has_bma: {has_bma}")
    print(f"  model_selection_method: {meta.get('model_selection_method', 'MISSING')}")
    
    # Check global model posterior
    gmp = global_data.get("model_posterior", {})
    print(f"  global model_posterior: {len(gmp)} models")
    if gmp:
        # Show top 3 models by weight
        sorted_models = sorted(gmp.items(), key=lambda x: x[1], reverse=True)
        for name, w in sorted_models[:5]:
            print(f"    {name}: {w:.4f}")
    
    # Check models and their parameters
    g_models = global_data.get("models", {})
    print(f"  global models: {len(g_models)} models")
    for mname, mdata in list(g_models.items())[:3]:
        if isinstance(mdata, dict):
            q = mdata.get("q", "?")
            c = mdata.get("c", "?")
            phi = mdata.get("phi", "?")
            bic = mdata.get("bic", "?")
            crps = mdata.get("crps", "?")
            print(f"    {mname}: q={q}, c={c}, phi={phi}, bic={bic}, crps={crps}")
    
    # Check regime data
    for rid in ["0", "1", "2", "3", "4"]:
        rd = regime_data.get(rid, {})
        if not rd:
            continue
        rmp = rd.get("model_posterior", {})
        rmeta = rd.get("regime_meta", {})
        n_obs = rmeta.get("n_obs", "?")
        fallback = rmeta.get("fallback", "?")
        borrowed = rmeta.get("borrowed_from_global", "?")
        print(f"  regime {rid}: {len(rmp)} models, n_obs={n_obs}, fallback={fallback}, borrowed={borrowed}")
    
    # Check Hansen skew-t
    hansen = global_data.get("hansen_skew_t", {})
    if hansen:
        print(f"  hansen_skew_t: lambda={hansen.get('lambda', '?')}, nu={hansen.get('nu', '?')}")
    
    # Check EMOS calibration
    sig_cal = data.get("signals_calibration", {})
    if sig_cal:
        print(f"  signals_calibration: {len(sig_cal)} horizons")
        for h_key, h_data in list(sig_cal.items())[:3]:
            if isinstance(h_data, dict):
                emos = h_data.get("emos", {})
                if emos:
                    a = emos.get("a", "?")
                    b = emos.get("b", "?")
                    print(f"    H={h_key}: EMOS a={a}, b={b}")
                by_regime = h_data.get("by_regime", {})
                if by_regime:
                    for reg_name, reg_cal in list(by_regime.items())[:2]:
                        reg_emos = reg_cal.get("emos", {}) if isinstance(reg_cal, dict) else {}
                        if reg_emos:
                            a = reg_emos.get("a", "?")
                            b = reg_emos.get("b", "?")
                            print(f"    H={h_key} regime={reg_name}: EMOS a={a}, b={b}")
    else:
        print(f"  signals_calibration: NONE")
    
    # Check recalibration
    recal = data.get("recalibration", {})
    if recal:
        print(f"  recalibration: applied={data.get('recalibration_applied', False)}")
    
    # Check isotonic calibration
    cal_params = data.get("calibration_params", {})
    if cal_params:
        iso_x = cal_params.get("isotonic_x_knots")
        if iso_x:
            print(f"  isotonic_x_knots: {len(iso_x)} knots")
    
    print()

# Now check what mu_kf values look like from compute_features
print("\n" + "=" * 80)
print("CHECKING KALMAN FILTER OUTPUT (mu_post, var_kf)")
print("=" * 80)

from ingestion.data_utils import get_price_series, enable_cache_only_mode
enable_cache_only_mode()

# Import the compute_features function
from decision.signals import compute_features

for sym in test_assets[:3]:
    try:
        px = get_price_series(sym, period="2y")
        if px is None or len(px) < 100:
            print(f"\n{sym}: insufficient price data")
            continue
        
        cache_path = os.path.join(TUNE_DIR, f"{sym}.json")
        tuned = None
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                tuned = json.load(f)
        
        feats = compute_features(px, tuned_params=tuned, asset_key=sym)
        
        # Extract key values
        mu_post = feats.get("mu_post")
        mu_kf = feats.get("mu_kf")
        var_kf = feats.get("var_kf")
        var_kf_sm = feats.get("var_kf_smoothed")
        vol = feats.get("vol")
        p_crisis = feats.get("p_crisis")
        
        print(f"\n{sym}:")
        
        if isinstance(mu_post, (pd.Series if 'pd' in dir() else type(None),)):
            import pandas as pd
            if isinstance(mu_post, pd.Series) and len(mu_post) > 0:
                print(f"  mu_post (last 5): {mu_post.tail(5).values}")
                print(f"  mu_post final: {float(mu_post.iloc[-1]):.8f}")
            else:
                print(f"  mu_post: EMPTY or None")
        else:
            print(f"  mu_post: {mu_post}")
        
        import pandas as pd
        if isinstance(mu_kf, pd.Series) and len(mu_kf) > 0:
            print(f"  mu_kf final: {float(mu_kf.iloc[-1]):.8f}")
        else:
            print(f"  mu_kf: EMPTY or None")
        
        if isinstance(var_kf, pd.Series) and len(var_kf) > 0:
            print(f"  var_kf final: {float(var_kf.iloc[-1]):.8f}")
        elif isinstance(var_kf_sm, pd.Series) and len(var_kf_sm) > 0:
            print(f"  var_kf_smoothed final: {float(var_kf_sm.iloc[-1]):.8f}")
        else:
            print(f"  var_kf: EMPTY or None")
        
        if isinstance(vol, pd.Series) and len(vol) > 0:
            print(f"  vol final: {float(vol.iloc[-1]):.6f}")
        
        # Check p_crisis
        if p_crisis is not None:
            if isinstance(p_crisis, (float, int)):
                print(f"  p_crisis: {p_crisis:.4f}")
            elif isinstance(p_crisis, pd.Series) and len(p_crisis) > 0:
                print(f"  p_crisis: {float(p_crisis.iloc[-1]):.4f}")
        
        # Check kalman_metadata
        km = feats.get("kalman_metadata", {})
        if km:
            print(f"  kalman phi: {km.get('phi_used', km.get('kalman_phi', '?'))}")
            print(f"  kalman q: {km.get('process_noise_var', '?')}")
            print(f"  kalman c: {km.get('kalman_c_optimal', '?')}")
            print(f"  noise_model: {km.get('kalman_noise_model', '?')}")
        
    except Exception as e:
        print(f"\n{sym}: ERROR - {e}")
        import traceback
        traceback.print_exc()
