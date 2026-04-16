#!/usr/bin/env python3
"""Diagnose GARCH variance explosion in MC simulation."""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

TUNE_DIR = os.path.join(os.path.dirname(__file__), 'src', 'data', 'tune')

def load_cache(symbol):
    path = os.path.join(TUNE_DIR, f"{symbol}.json")
    with open(path) as f:
        return json.load(f)

def simulate_garch_path(h0, omega, alpha, beta, phi, mu0, q, H=252, n_paths=5000):
    """Minimal GARCH MC to diagnose variance explosion."""
    rng = np.random.default_rng(42)
    h_t = np.full(n_paths, h0)
    mu_t = np.full(n_paths, mu0)
    cum = np.zeros(n_paths)
    
    results = {}
    for t in range(H):
        sigma_t = np.sqrt(h_t)
        e_t = sigma_t * rng.standard_normal(n_paths)
        r_t = mu_t + e_t
        cum += r_t
        
        # GARCH evolution
        h_t = omega + alpha * (e_t ** 2) + beta * h_t
        h_t = np.clip(h_t, 1e-12, 1e4)  # Current cap
        
        # Drift evolution  
        eta = rng.normal(0, np.sqrt(q), n_paths) if q > 0 else 0
        mu_t = phi * mu_t + eta
        
        if (t+1) in [1, 3, 7, 21, 63, 126, 252]:
            results[t+1] = {
                'h_median': float(np.median(h_t)),
                'h_p95': float(np.percentile(h_t, 95)),
                'h_max': float(np.max(h_t)),
                'sigma_median': float(np.sqrt(np.median(h_t))),
                'cum_median': float(np.median(cum)),
                'cum_p05': float(np.percentile(cum, 5)),
                'cum_p95': float(np.percentile(cum, 95)),
                'exp_ret_pct': float((np.exp(np.median(cum)) - 1) * 100),
            }
    return results


for sym in ['SPY', 'NVDA', 'TSLA', 'GOOGL', 'MSFT', 'AAPL']:
    try:
        cache = load_cache(sym)
    except FileNotFoundError:
        print(f"\n{sym}: No cache file")
        continue
    
    gl = cache.get('global', {})
    models = gl.get('models', {})
    mp = gl.get('model_posterior', {})
    
    # Find top model
    top_model = max(mp, key=mp.get) if mp else None
    if not top_model:
        print(f"\n{sym}: No model posterior")
        continue
    
    params = models.get(top_model, {})
    phi = params.get('phi', 0.99)
    q = params.get('q', 1e-6)
    c = params.get('c', 1.0)
    
    # Get GARCH params (try model-specific, then look for global garch)
    g_omega = params.get('garch_omega', 0)
    g_alpha = params.get('garch_alpha', 0)
    g_beta = params.get('garch_beta', 0)
    
    # Check if GARCH params are in global section
    if g_omega == 0:
        for key in ['garch_params', 'garch']:
            gp = gl.get(key, {})
            if isinstance(gp, dict) and 'omega' in gp:
                g_omega = gp.get('omega', 0)
                g_alpha = gp.get('alpha', 0)
                g_beta = gp.get('beta', 0)
                break
    
    # Get sigma2
    sigma2 = gl.get('sigma2_step', params.get('sigma2_step', 0.0002))
    mu_post = gl.get('mu_post', params.get('mu_post', 0.0001))
    
    print(f"\n{'='*60}")
    print(f"{sym}: top_model={top_model} (w={mp[top_model]:.3f})")
    print(f"  phi={phi:.4f}, q={q:.6f}, c={c:.4f}")
    print(f"  GARCH: omega={g_omega:.8f}, alpha={g_alpha:.4f}, beta={g_beta:.4f}")
    print(f"  alpha+beta={g_alpha + g_beta:.4f}")
    if g_alpha + g_beta < 1:
        h_long = g_omega / max(1 - g_alpha - g_beta, 0.001)
        print(f"  h_long_run={h_long:.6f} (sigma={np.sqrt(h_long)*100:.2f}%/day)")
    else:
        print(f"  alpha+beta >= 1: VARIANCE EXPLOSION!")
    print(f"  sigma2_step={sigma2:.6f} (sigma={np.sqrt(sigma2)*100:.2f}%/day)")
    print(f"  mu_post={mu_post:.6f} ({mu_post*100:.3f}%/day)")
    
    use_garch = g_omega > 0 and g_alpha > 0 and g_beta > 0
    if use_garch:
        results = simulate_garch_path(
            h0=sigma2 * c, omega=g_omega, alpha=g_alpha, beta=g_beta,
            phi=phi, mu0=mu_post, q=q, H=252, n_paths=5000
        )
        print(f"\n  Horizon │ h_median  │ h_p95     │ sigma_med │ cum_med   │ exp_ret%")
        print(f"  {'─'*7}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*10}")
        for H in [1, 3, 7, 21, 63, 126, 252]:
            r = results[H]
            print(f"  {H:>5}d │ {r['h_median']:.6f}  │ {r['h_p95']:.6f}  │ {r['sigma_median']*100:>6.2f}%  │ {r['cum_median']:>+.5f} │ {r['exp_ret_pct']:>+.2f}%")
    else:
        print(f"  No GARCH params found - constant vol MC")
        # Run without GARCH
        results = simulate_garch_path(
            h0=sigma2 * c, omega=0, alpha=0, beta=0,
            phi=phi, mu0=mu_post, q=q, H=252, n_paths=5000
        )
        print(f"\n  Horizon │ cum_median  │ exp_ret%")
        print(f"  {'─'*7}┼{'─'*13}┼{'─'*10}")
        for H in [1, 3, 7, 21, 63, 126, 252]:
            r = results[H]
            print(f"  {H:>5}d │ {r['cum_median']:>+.7f} │ {r['exp_ret_pct']:>+.2f}%")
