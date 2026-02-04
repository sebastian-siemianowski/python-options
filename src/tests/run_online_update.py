#!/usr/bin/env python3
"""
Run bulk online parameter updates for all tuned assets.

This script:
1. Loads price data from cache (src/data/prices/)
2. Loads tuned parameters from cache (src/data/tune/)
3. Runs online Bayesian parameter updates using SMC
4. Persists state to disk (src/data/online_update/)

Usage:
    python src/tests/run_online_update.py
    python src/tests/run_online_update.py --symbols AAPL,MSFT,GOOGL
    python src/tests/run_online_update.py --workers 12
"""
import sys
import os
import argparse
import multiprocessing as mp

# Ensure we're running from project root for correct relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(src_dir)

# Change to project root so relative paths work
os.chdir(project_root)
sys.path.insert(0, src_dir)

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

from calibration.online_update import (
    OnlineUpdateConfig,
    OnlineBayesianUpdater,
    save_updater_state,
    load_updater_state,
    get_persistence_stats,
    BulkUpdateResult,
)


def _process_symbol_worker(args: Tuple[str, int, bool]) -> BulkUpdateResult:
    """
    Worker function for multiprocessing.
    
    Args:
        args: Tuple of (symbol, n_particles, persist)
        
    Returns:
        BulkUpdateResult
    """
    symbol, n_particles, persist = args
    
    # Import inside worker to avoid pickling issues
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    os.chdir(project_root)
    sys.path.insert(0, src_dir)
    
    import numpy as np
    import pandas as pd
    
    from tuning.kalman_cache import load_tuned_params
    from ingestion.data_utils import _load_disk_prices
    from calibration.online_update import (
        OnlineUpdateConfig,
        OnlineBayesianUpdater,
        save_updater_state,
        load_updater_state,
        BulkUpdateResult,
    )
    
    try:
        # Load tuned parameters first - skip if not tuned
        tuned_params = load_tuned_params(symbol)
        if tuned_params is None:
            return BulkUpdateResult(
                symbol=symbol,
                success=False,
                error="No tuned parameters",
            )
        
        # Load price data
        price_df = _load_disk_prices(symbol)
        if price_df is None or price_df.empty:
            return BulkUpdateResult(
                symbol=symbol,
                success=False,
                error="No price data",
            )
        
        # Extract close prices
        if "Close" in price_df.columns:
            close = price_df["Close"]
        elif "Adj Close" in price_df.columns:
            close = price_df["Adj Close"]
        else:
            return BulkUpdateResult(
                symbol=symbol,
                success=False,
                error="No Close column",
            )
        
        close = pd.to_numeric(close, errors='coerce').dropna()
        if len(close) < 50:
            return BulkUpdateResult(
                symbol=symbol,
                success=False,
                error=f"Insufficient data ({len(close)} rows)",
            )
        
        # Compute returns and volatility
        returns = np.log(close / close.shift(1)).dropna().values
        vol = pd.Series(returns).ewm(span=21).std().values
        
        # Take last 200 observations for online update
        n = min(200, len(returns))
        returns = returns[-n:]
        vol = vol[-n:]
        
        # Try to load existing state
        config = OnlineUpdateConfig(n_particles=n_particles)
        updater = load_updater_state(symbol)
        
        if updater is None:
            updater = OnlineBayesianUpdater.from_batch_params(tuned_params, config)
        
        # Process observations
        update_count = 0
        last_result = None
        
        for i in range(len(returns)):
            y = float(returns[i])
            sigma = float(vol[i])
            
            if np.isfinite(y) and np.isfinite(sigma) and sigma > 0:
                last_result = updater.update(y, sigma)
                update_count += 1
        
        # Get final parameters
        online_params = updater.get_current_params()
        final_ess = last_result.effective_sample_size if last_result else 0.0
        
        # Persist state
        persisted = False
        if persist:
            try:
                save_updater_state(symbol, updater)
                persisted = True
            except Exception as e:
                pass  # Silently fail persistence
        
        return BulkUpdateResult(
            symbol=symbol,
            success=True,
            online_params=online_params,
            update_count=update_count,
            final_ess=final_ess,
            persisted=persisted,
        )
        
    except Exception as e:
        return BulkUpdateResult(
            symbol=symbol,
            success=False,
            error=str(e),
        )


def main():
    parser = argparse.ArgumentParser(description="Run bulk online parameter updates")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--no-persist", action="store_true", help="Don't persist state to disk")
    parser.add_argument("--particles", type=int, default=100, help="Number of particles per updater")
    args = parser.parse_args()
    
    # Default workers to CPU count
    if args.workers is None:
        args.workers = mp.cpu_count()
    
    # Get symbols to process
    from tuning.kalman_cache import list_cached_symbols
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # Use tuned symbols (intersection of tuned and available price data)
        symbols = list_cached_symbols()
    
    print(f"Running Online Bayesian Parameter Updates...")
    print(f"  Workers: {args.workers} (processes)")
    print(f"  Particles: {args.particles}")
    print(f"  Persist: {not args.no_persist}")
    print(f"  Symbols: {len(symbols)}")
    print()
    
    # Prepare work items
    work_items = [(sym, args.particles, not args.no_persist) for sym in symbols]
    
    # Process symbols using multiprocessing
    results: Dict[str, BulkUpdateResult] = {}
    completed = 0
    total = len(symbols)
    
    # Use multiprocessing Pool
    with mp.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_symbol_worker, work_items):
            results[result.symbol] = result
            completed += 1
            
            if completed % 20 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({100*completed//total}%)")
    
    # Summary
    success_count = sum(1 for r in results.values() if r.success)
    persisted_count = sum(1 for r in results.values() if r.persisted)
    failed = [r for r in results.values() if not r.success]
    
    print()
    print(f"Completed: {success_count}/{len(results)} successful, {persisted_count} persisted")
    
    # Calculate average ESS for successful updates
    ess_values = [r.final_ess for r in results.values() if r.success and r.final_ess > 0]
    if ess_values:
        print(f"Average ESS: {np.mean(ess_values):.1f}")
    
    # Report failures by category
    if failed:
        error_counts: Dict[str, int] = {}
        for f in failed:
            error = f.error or "Unknown"
            # Simplify error message
            if "tuned" in error.lower():
                key = "No tuned parameters"
            elif "price" in error.lower() or "data" in error.lower():
                key = "No price data"
            elif "insufficient" in error.lower():
                key = "Insufficient history"
            else:
                key = error[:50]
            error_counts[key] = error_counts.get(key, 0) + 1
        
        print()
        print(f"Failures by category:")
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  - {error}: {count}")
    
    # Persistence stats
    stats = get_persistence_stats()
    print()
    print(f"📊 Persistence stats: {stats['n_symbols']} symbols, {stats['total_size_mb']} MB")


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    main()
