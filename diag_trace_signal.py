#!/usr/bin/env python3
"""Run signal generation for ONE asset and trace intermediate values."""
import sys, os, json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

os.environ['TUNING_QUIET'] = '1'

from decision.signals import (
    compute_features, bayesian_model_average_mc, _load_tuned_kalman_params,
    _download_prices, fetch_px_asset,
)

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'

# Load prices from cached CSV
print(f"Loading prices for {SYMBOL}...")
px, title = fetch_px_asset(SYMBOL, None, None)
print(f"  Loaded {len(px)} rows, title: {title}")

# Load OHLC
print(f"Loading OHLC for {SYMBOL}...")
ohlc_df = _download_prices(SYMBOL, None, None)
print(f"  Loaded OHLC: {len(ohlc_df)} rows")

# Load tuned params
print(f"\nLoading tuned params...")
tuned = _load_tuned_kalman_params(SYMBOL)
if tuned:
    print(f"  has_bma={tuned.get('has_bma')}")
    gl = tuned.get('global', {})
    print(f"  global phi={gl.get('phi'):.4f}, q={gl.get('q'):.2e}")
else:
    print("  NO tuned params!")
    sys.exit(1)

# Compute features
print(f"\nComputing features...")
feats = compute_features(px, asset_symbol=SYMBOL, ohlc_df=ohlc_df)

# Extract the key MC inputs
mu_post = feats.get("mu_post", feats.get("mu_kf"))
var_kf = feats.get("var_kf")
vol = feats.get("vol")
garch_params = feats.get("garch_params", {})

mu_t_mc = float(mu_post.iloc[-1]) if hasattr(mu_post, 'iloc') and len(mu_post) > 0 else 0.0
P_t_mc = float(var_kf.iloc[-1]) if hasattr(var_kf, 'iloc') and len(var_kf) > 0 else 0.0
sigma_now = float(vol.iloc[-1]) if hasattr(vol, 'iloc') and len(vol) > 0 else 0.01
sigma2_step_mc = sigma_now ** 2

print(f"\n{'='*60}")
print(f"MC INPUTS for {SYMBOL}:")
print(f"  mu_t_mc (daily drift) = {mu_t_mc:.8f} ({mu_t_mc*100:.4f}%/day)")
print(f"  P_t_mc (drift variance) = {P_t_mc:.10f}")
print(f"  sigma_now (daily vol)  = {sigma_now:.6f} ({sigma_now*100:.2f}%/day)")
print(f"  sigma2_step_mc         = {sigma2_step_mc:.8f}")
print(f"  GARCH params           = {garch_params}")
print(f"  phi_used               = {feats.get('phi_used')}")
print(f"  kalman_available       = {feats.get('kalman_available')}")

km = feats.get("kalman_metadata", {})
print(f"\n  Kalman metadata:")
print(f"    phi_used = {km.get('phi_used')}")
print(f"    process_noise_var = {km.get('process_noise_var')}")
print(f"    kalman_c_optimal = {km.get('kalman_c_optimal')}")
print(f"    kalman_noise_model = {km.get('kalman_noise_model')}")

# Run BMA MC for each horizon
print(f"\n{'='*60}")
print(f"Running BMA MC for each horizon...")
for H in [1, 3, 7, 21, 63, 126, 252]:
    r_samples, vol_samples, regime_probs, bma_meta = bayesian_model_average_mc(
        feats=feats,
        regime_params={},
        mu_t=mu_t_mc,
        P_t=P_t_mc,
        sigma2_step=sigma2_step_mc,
        H=H,
        n_paths=10000,
        seed=42,
        tuned_params=tuned,
        asset_symbol=SYMBOL,
    )
    
    r = np.asarray(r_samples, dtype=float)
    r = r[np.isfinite(r)]
    
    if len(r) > 0:
        med = float(np.median(r))
        mean = float(np.mean(r))
        p05 = float(np.percentile(r, 5))
        p95 = float(np.percentile(r, 95))
        p_up = float(np.mean(r > 0))
        
        # Check for extremes
        n_extreme = int(np.sum(np.abs(r) > 1.0))
        
        print(f"  H={H:>3}d: median={med:>+.4f} ({med*100:>+.2f}%), mean={mean:>+.4f}, p05={p05:>+.4f}, p95={p95:>+.4f}, p_up={p_up:.3f}, n_extreme(>|1|)={n_extreme}/{len(r)}")
    else:
        print(f"  H={H:>3}d: NO valid samples!")
    
    # Print model details from metadata
    if H == 63:
        print(f"\n    BMA method: {bma_meta.get('method')}")
        print(f"    Current regime: {bma_meta.get('current_regime')} ({bma_meta.get('regime_name')})")
        print(f"    N models: {bma_meta.get('n_models_contributing', '?')}")
        md = bma_meta.get('model_details', {})
        for mname, minfo in md.items():
            w = minfo.get('weight', 0)
            phi_m = minfo.get('phi')
            q_m = minfo.get('q')
            nu_m = minfo.get('nu')
            c_m = minfo.get('c')
            go = minfo.get('garch_omega')
            ga = minfo.get('garch_alpha')
            gb = minfo.get('garch_beta')
            n_s = minfo.get('n_samples', 0)
            use_garch = go and float(go) > 1e-12 and ga and float(ga) > 1e-6
            print(f"    {mname}: w={w:.3f}, phi={phi_m}, q={q_m:.2e}, nu={nu_m}, c={c_m:.3f}, garch={use_garch}, samples={n_s}")
