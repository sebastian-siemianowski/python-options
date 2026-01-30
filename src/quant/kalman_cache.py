"""
Per-asset Kalman parameter cache.

Stores tuned parameters in individual JSON files under src/quant/cache/tune/
to enable git-friendly storage and parallel-safe tuning.

Architecture:
    - Each asset gets its own JSON file: {SYMBOL}.json
    - Symbol normalization: BTC-USD -> BTC_USD, PLN=X -> PLN_X
    - Backward compatibility with legacy single-file cache during migration

Author: Quantitative Systems Team
Date: 2026-01-29
"""

import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# Cache directory path (relative to this file's location)
TUNE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "tune")

# Legacy single-file cache path (for backward compatibility)
LEGACY_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "kalman_q_cache.json")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol for filesystem-safe filenames.
    
    Handles common cases:
        BTC-USD -> BTC_USD
        PLN=X -> PLN_X
        AAPL.OQ -> AAPL_OQ
        ^SPX -> _SPX
        
    Args:
        symbol: Raw asset symbol
        
    Returns:
        Normalized symbol safe for filesystem use
    """
    normalized = symbol.strip().upper()
    # Replace common special characters with underscore
    for char in ["-", "=", "/", ".", "^", " ", ":"]:
        normalized = normalized.replace(char, "_")
    # Remove any remaining non-alphanumeric except underscore
    normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
    # Ensure it doesn't start with a number (prepend underscore)
    if normalized and normalized[0].isdigit():
        normalized = "_" + normalized
    return normalized


def _get_cache_path(symbol: str) -> str:
    """
    Get the cache file path for a symbol.
    
    Args:
        symbol: Asset symbol (raw or normalized)
        
    Returns:
        Full path to the per-asset cache file
    """
    normalized = _normalize_symbol(symbol)
    return os.path.join(TUNE_CACHE_DIR, f"{normalized}.json")


def load_tuned_params(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Load tuned Kalman parameters for a single asset.
    
    Tries per-asset cache first, falls back to legacy single-file cache
    for backward compatibility during migration period.
    
    Args:
        symbol: Asset symbol (e.g., 'AAPL', 'BTC-USD', 'PLN=X')
        
    Returns:
        Dict with tuned parameters or None if not cached
        
    Cache structure (BMA format from tune_q_mle.py):
        {
            "symbol": "AAPL",
            "normalized_symbol": "AAPL",
            "global": {
                "model_posterior": {...},
                "models": {...},
                ...
            },
            "regime": {
                "0": {...},
                "1": {...},
                ...
            },
            "meta": {...}
        }
    """
    # Try per-asset cache first
    cache_path = _get_cache_path(symbol)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load cache for {symbol}: {e}")
            pass
    
    # Fallback: legacy single-file cache (for migration period)
    if os.path.exists(LEGACY_CACHE_PATH):
        try:
            with open(LEGACY_CACHE_PATH, "r") as f:
                legacy_cache = json.load(f)
            
            # Try exact match first
            key = symbol.strip().upper()
            if key in legacy_cache:
                return legacy_cache[key]
            
            # Try normalized variations
            normalized = _normalize_symbol(symbol)
            for variant in [normalized, key.replace("-", "_"), key.replace("=", "_")]:
                if variant in legacy_cache:
                    return legacy_cache[variant]
        except Exception:
            pass
    
    return None


def save_tuned_params(symbol: str, params: Dict[str, Any]) -> str:
    """
    Save tuned Kalman parameters for a single asset.
    
    Args:
        symbol: Asset symbol
        params: Dict with tuned parameters (q, c, nu, regime params, BMA, etc.)
        
    Returns:
        Path to the saved cache file
        
    Note:
        Uses atomic write (temp file + rename) for safety during parallel tuning.
    """
    os.makedirs(TUNE_CACHE_DIR, exist_ok=True)
    cache_path = _get_cache_path(symbol)
    
    # Add metadata
    params_with_meta = {
        "symbol": symbol,
        "normalized_symbol": _normalize_symbol(symbol),
        "cache_version": "2.0",  # Per-asset cache format
        "saved_at": datetime.now().isoformat(),
        **params
    }
    
    # Atomic write using temp file + rename
    temp_path = cache_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(params_with_meta, f, indent=2, cls=NumpyEncoder)
        os.replace(temp_path, cache_path)
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
    
    return cache_path


def delete_tuned_params(symbol: str) -> bool:
    """
    Delete cached parameters for a symbol.
    
    Args:
        symbol: Asset symbol
        
    Returns:
        True if file was deleted, False if not found
    """
    cache_path = _get_cache_path(symbol)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        return True
    return False


def list_cached_symbols() -> List[str]:
    """
    List all symbols with cached tuning parameters.
    
    Returns:
        Sorted list of normalized symbol names (without .json extension)
    """
    if not os.path.exists(TUNE_CACHE_DIR):
        return []
    
    symbols = []
    for fname in os.listdir(TUNE_CACHE_DIR):
        if fname.endswith(".json") and fname not in [".keep", "kalman_q_cache.json"]:
            symbols.append(fname[:-5])  # Remove .json extension
    return sorted(symbols)


def load_full_cache() -> Dict[str, Dict[str, Any]]:
    """
    Load all cached parameters into a single dictionary.
    
    This provides backward compatibility with code that expects the old
    single-file cache format.
    
    Returns:
        Dict mapping symbol -> params for all cached assets
    """
    cache = {}
    
    # Load per-asset files
    for symbol in list_cached_symbols():
        params = load_tuned_params(symbol)
        if params:
            # Use the original symbol if available, else normalized
            original_symbol = params.get("symbol", symbol)
            cache[original_symbol] = params
    
    return cache


def save_full_cache(cache: Dict[str, Dict[str, Any]]) -> int:
    """
    Save a full cache dictionary to per-asset files.
    
    This is used for migration from legacy single-file cache.
    
    Args:
        cache: Dict mapping symbol -> params
        
    Returns:
        Number of assets saved
    """
    count = 0
    for symbol, params in cache.items():
        try:
            save_tuned_params(symbol, params)
            count += 1
        except Exception as e:
            print(f"Warning: Failed to save {symbol}: {e}")
    return count


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the tuning cache.
    
    Returns:
        Dict with cache statistics:
        - n_assets: Number of cached assets
        - total_size_kb: Total size in kilobytes
        - avg_size_kb: Average file size in kilobytes
        - cache_dir: Path to cache directory
    """
    if not os.path.exists(TUNE_CACHE_DIR):
        return {
            "n_assets": 0,
            "total_size_kb": 0,
            "avg_size_kb": 0,
            "cache_dir": TUNE_CACHE_DIR,
        }
    
    files = [f for f in os.listdir(TUNE_CACHE_DIR) 
             if f.endswith(".json") and f not in [".keep", "kalman_q_cache.json"]]
    
    total_size = sum(
        os.path.getsize(os.path.join(TUNE_CACHE_DIR, f)) 
        for f in files
    )
    
    return {
        "n_assets": len(files),
        "total_size_kb": round(total_size / 1024, 2),
        "avg_size_kb": round((total_size / len(files) / 1024), 2) if files else 0,
        "cache_dir": TUNE_CACHE_DIR,
    }


def migrate_legacy_cache(delete_legacy: bool = False) -> Dict[str, Any]:
    """
    Migrate legacy single-file cache to per-asset files.
    
    Args:
        delete_legacy: If True, delete the legacy file after successful migration
        
    Returns:
        Dict with migration statistics:
        - migrated: Number of assets migrated
        - failed: Number of assets that failed to migrate
        - skipped: Number of assets skipped (already exist)
        - legacy_deleted: Whether legacy file was deleted
    """
    stats = {
        "migrated": 0,
        "failed": 0,
        "skipped": 0,
        "legacy_deleted": False,
        "errors": [],
    }
    
    if not os.path.exists(LEGACY_CACHE_PATH):
        print("No legacy cache found.")
        return stats
    
    try:
        with open(LEGACY_CACHE_PATH, "r") as f:
            legacy_cache = json.load(f)
    except Exception as e:
        stats["errors"].append(f"Failed to load legacy cache: {e}")
        return stats
    
    print(f"Found {len(legacy_cache)} assets in legacy cache.")
    
    for symbol, params in legacy_cache.items():
        cache_path = _get_cache_path(symbol)
        
        # Skip if already exists
        if os.path.exists(cache_path):
            stats["skipped"] += 1
            continue
        
        try:
            save_tuned_params(symbol, params)
            stats["migrated"] += 1
            print(f"  ✓ Migrated: {symbol}")
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"{symbol}: {e}")
            print(f"  ✗ Failed: {symbol} - {e}")
    
    # Optionally delete legacy file
    if delete_legacy and stats["failed"] == 0:
        try:
            os.remove(LEGACY_CACHE_PATH)
            stats["legacy_deleted"] = True
            print(f"\n✓ Deleted legacy cache: {LEGACY_CACHE_PATH}")
        except Exception as e:
            stats["errors"].append(f"Failed to delete legacy: {e}")
    
    print(f"\nMigration complete:")
    print(f"  Migrated: {stats['migrated']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Failed:   {stats['failed']}")
    
    return stats


# Command-line interface for migration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kalman cache management utilities")
    parser.add_argument("--migrate", action="store_true", help="Migrate legacy cache to per-asset files")
    parser.add_argument("--delete-legacy", action="store_true", help="Delete legacy file after migration")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--list", action="store_true", help="List all cached symbols")
    
    args = parser.parse_args()
    
    if args.migrate:
        migrate_legacy_cache(delete_legacy=args.delete_legacy)
    elif args.stats:
        stats = get_cache_stats()
        print(f"Cache Statistics:")
        print(f"  Assets:     {stats['n_assets']}")
        print(f"  Total Size: {stats['total_size_kb']:.1f} KB")
        print(f"  Avg Size:   {stats['avg_size_kb']:.1f} KB")
        print(f"  Directory:  {stats['cache_dir']}")
    elif args.list:
        symbols = list_cached_symbols()
        print(f"Cached symbols ({len(symbols)}):")
        for s in symbols:
            print(f"  {s}")
    else:
        parser.print_help()
