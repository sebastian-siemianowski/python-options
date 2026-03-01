#!/usr/bin/env python3
"""Diagnose PIT variance characteristics for failing assets."""
import sys, os, math, warnings
import numpy as np

warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or '.'
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))
os.environ['TUNING_QUIET'] = '1'
os.environ['OFFLINE_MODE'] = '1'

from models.phi_student_t import PhiStudentTDriftModel
from tuning.tune import _download_prices
from scipy.stats import kstest


ASSETS = [
    'GC=F', 'SGLP', 'HII', 'ADBE', 'HWM', 'PSIX', 'ABTC', 'GPUS',
    'NVTS', 'UPS', 'ASTS', 'BNZI', 'MSTR', 'PYPL', 'CRS',
    'ASML', 'SMCI', 'AIRI', 'OPXS',
    # Good assets for comparison
    'BWXT', 'MMM', 'ACN', 'MRK', 'ASTC', 'ON',
]

TARGET_VAR = 1.0 / 12.0  # 0.08333

print(f"{'Asset':<12} {'nu':>3} {'KS_p':>7} {'PIT_m':>6} {'PIT_v':>7} "
      f"{'V_rat':>6} {'tail%':>6} {'<0.1%':>6} {'>0.9%':>6} {'n_test':>6}")
print("-" * 80)

for sym in ASSETS:
    try:
        df = _download_prices(sym, '2015-01-01', None)
        if df is None or len(df) < 120:
            print(f"{sym:<12} --- not enough data")
            continue
        close = df['Close'].values.astype(float)
        rets = np.diff(np.log(close))
        n = len(rets)
        n_train = int(n * 0.7)
        n_test = n - n_train
        if n_test < 30:
            print(f"{sym:<12} --- too few test ({n_test})")
            continue

        best_p = -1.0
        best_line = ""
        for nu in [4, 8, 20]:
            model = PhiStudentTDriftModel(nu=nu)
            result = model.fit(rets)
            if result is None:
                continue
            pit = result.get('pit_values')
            if pit is None or len(pit) < 20:
                continue
            ks_stat, ks_p = kstest(pit, 'uniform')
            pit_mean = float(np.mean(pit))
            pit_var = float(np.var(pit))
            var_ratio = pit_var / TARGET_VAR
            tail_lo = float(np.mean(pit < 0.1))
            tail_hi = float(np.mean(pit > 0.9))
            tail_frac = tail_lo + tail_hi
            line = (f"{sym:<12} {nu:>3} {ks_p:>7.4f} {pit_mean:>6.3f} "
                    f"{pit_var:>7.4f} {var_ratio:>6.3f} {tail_frac:>6.3f} "
                    f"{tail_lo:>6.3f} {tail_hi:>6.3f} {n_test:>6}")
            if ks_p > best_p:
                best_p = ks_p
                best_line = line
        if best_line:
            print(best_line)
        else:
            print(f"{sym:<12} --- all models failed")
    except Exception as e:
        print(f"{sym:<12} --- ERROR: {e}")
