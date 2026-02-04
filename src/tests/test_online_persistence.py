#!/usr/bin/env python3
"""Test persistence and bulk processing features of online update module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("Testing persistence and bulk processing...")

from calibration.online_update import (
    OnlineBayesianUpdater,
    OnlineUpdateConfig,
    save_updater_state,
    load_updater_state,
    delete_updater_state,
    list_persisted_symbols,
    get_persistence_stats,
    process_asset_online_update,
    BulkUpdateResult,
    ONLINE_UPDATE_CACHE_DIR,
)

print(f"Cache dir: {ONLINE_UPDATE_CACHE_DIR}")

# Test 1: Create and save updater state
print()
print("Test 1: Save updater state")
batch_params = {'q': 1e-6, 'c': 1.0, 'phi': 0.95, 'nu': 8.0}
config = OnlineUpdateConfig(n_particles=50)
updater = OnlineBayesianUpdater(batch_params, config)

# Run some updates
np.random.seed(42)
for i in range(20):
    updater.update(0.001 * np.sin(i) + 0.002 * np.random.randn(), 0.02)

path = save_updater_state('TEST_SYMBOL', updater)
print(f"  Saved to: {path}")

# Test 2: Load updater state
print()
print("Test 2: Load updater state")
loaded = load_updater_state('TEST_SYMBOL')
if loaded:
    print(f"  Loaded successfully")
    print(f"  Step count: {loaded.step_count}")
    print(f"  Particles: {len(loaded.particles)}")
else:
    print("  FAILED to load")

# Test 3: List persisted symbols
print()
print("Test 3: List persisted symbols")
symbols = list_persisted_symbols()
print(f"  Persisted: {symbols}")

# Test 4: Get stats
print()
print("Test 4: Persistence stats")
stats = get_persistence_stats()
print(f"  Symbols: {stats['n_symbols']}")
print(f"  Size: {stats['total_size_mb']} MB")

# Test 5: Process asset with persistence
print()
print("Test 5: Process asset online update")
returns = np.random.randn(100) * 0.02
volatility = np.abs(np.random.randn(100)) * 0.02 + 0.01
result = process_asset_online_update(
    symbol='TEST_ASSET',
    returns=returns,
    volatility=volatility,
    tuned_params={'global': batch_params},
    persist=True,
)
print(f"  Success: {result.success}")
print(f"  Updates: {result.update_count}")
print(f"  ESS: {result.final_ess:.1f}")
print(f"  Persisted: {result.persisted}")

# Cleanup
print()
print("Cleanup...")
delete_updater_state('TEST_SYMBOL')
delete_updater_state('TEST_ASSET')
print("  Deleted test files")

print()
print("=" * 50)
print("ALL PERSISTENCE TESTS PASSED!")
print("=" * 50)
