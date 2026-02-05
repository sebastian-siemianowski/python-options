#!/usr/bin/env python3
"""Test elite tuning v2.0 imports."""

import sys
import os
import traceback

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("Testing Elite Tuning v2.0 imports...")

try:
    from src.tuning.elite_tuning import (
        EliteTuningConfig,
        EliteOptimizer,
        EliteTuningDiagnostics,
        create_elite_tuning_config,
        compute_directional_curvature_penalty,
        compute_asymmetric_coherence_penalty,
        evaluate_connected_plateau,
        compute_fragility_index,
        COUPLING_DANGER_WEIGHTS,
    )
    print("OK: elite_tuning v2.0 imports successful")
    print(f"  Coupling danger weights: {len(COUPLING_DANGER_WEIGHTS)} entries")
except Exception as e:
    print(f"FAIL: elite_tuning import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.models.phi_student_t import (
        PhiStudentTDriftModel, 
        ELITE_TUNING_ENABLED,
        _compute_curvature_penalty,
        _compute_fragility_index,
    )
    print("OK: phi_student_t imports successful")
    print(f"  Elite tuning enabled: {ELITE_TUNING_ENABLED}")
except Exception as e:
    print(f"FAIL: phi_student_t import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.models.phi_gaussian import PhiGaussianDriftModel
    print("OK: phi_gaussian imports successful")
except Exception as e:
    print(f"FAIL: phi_gaussian import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.tuning.tune_ux import render_elite_tuning_summary
    print("OK: tune_ux imports successful")
except Exception as e:
    print(f"FAIL: tune_ux import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test config presets
print("\nTesting config presets...")
for preset in ['balanced', 'conservative', 'aggressive', 'diagnostic', 'legacy']:
    config = create_elite_tuning_config(preset)
    print(f"  {preset}: ridge_detection={config.enable_ridge_detection}, drift_penalty={config.enable_drift_penalty}")

print("\nAll elite tuning v2.0 components imported successfully!")
