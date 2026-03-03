"""Profile the tuning pipeline for a single asset to identify hotspots."""
import sys, os, warnings, time, cProfile, pstats, io
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ['TUNING_QUIET'] = '1'
os.environ['OFFLINE_MODE'] = '1'

# Warm up Numba first
print("Warming up Numba kernels...")
t0 = time.time()
try:
    from models.numba_wrappers import is_numba_available
    if is_numba_available():
        import numpy as np
        from models.numba_wrappers import run_phi_student_t_enhanced_filter
        # Trigger compilation with dummy data
        dummy_r = np.random.randn(100).astype(np.float64)
        dummy_v = np.abs(np.random.randn(100)).astype(np.float64) * 0.01 + 0.01
        dummy_vov = np.abs(np.random.randn(100)).astype(np.float64) * 0.1
        try:
            run_phi_student_t_enhanced_filter(
                dummy_r, dummy_v, 1e-6, 1.0, 0.5, 8.0,
                robust_wt=True, online_scale_adapt=True,
                gamma_vov=0.3, vov_rolling=dummy_vov,
            )
        except Exception:
            pass
        # Also warm up CV test fold
        try:
            from models.numba_wrappers import run_phi_student_t_cv_test_fold
            run_phi_student_t_cv_test_fold(
                dummy_r, dummy_v**2, 1e-6, 1.0, 0.5,
                0.75, -1.0, -4.5, 0.125,
                0.0, 1e-4, 50, 100,
            )
        except Exception:
            pass
        print(f"  Numba warmup: {time.time()-t0:.1f}s")
    else:
        print("  Numba NOT available")
except ImportError:
    print("  Numba import failed")

print("\nProfiling tune_asset_q('MSTR')...")
print("=" * 70)

from tuning.tune import tune_asset_q

# Profile with cProfile  
pr = cProfile.Profile()
pr.enable()
t_start = time.time()
result = tune_asset_q('MSTR')
t_end = time.time()
pr.disable()

print(f"\nTotal wall time: {t_end - t_start:.2f}s")
print()

# Print top 40 by cumulative time
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats('cumulative')
ps.print_stats(40)
print("=== TOP 40 BY CUMULATIVE TIME ===")
print(s.getvalue())

# Print top 40 by total (self) time
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2)
ps2.sort_stats('tottime')
ps2.print_stats(40)
print("=== TOP 40 BY SELF TIME ===")
print(s2.getvalue())

# Count filter calls
s3 = io.StringIO()
ps3 = pstats.Stats(pr, stream=s3)
ps3.sort_stats('calls')
ps3.print_stats('filter|optimize|pit|crps|garch|ks')
print("=== FILTER/OPTIMIZER CALL COUNTS ===")
print(s3.getvalue())

# Model summary
if result:
    models = result.get('models', {})
    print(f"\nModels fitted: {len(models)}")
    for name in sorted(models.keys()):
        m = models[name]
        success = m.get('fit_success', False)
        crps = m.get('crps', 'N/A')
        pit = m.get('pit_ks_pvalue', 'N/A')
        w = m.get('model_weight_entropy', 0)
        print(f"  {name}: success={success}, CRPS={crps}, PIT={pit}, w={w:.4f}")
