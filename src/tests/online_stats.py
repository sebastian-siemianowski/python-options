#!/usr/bin/env python3
"""Show online update cache statistics."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calibration.online_update import get_persistence_stats, list_persisted_symbols

stats = get_persistence_stats()
symbols = list_persisted_symbols()

print(f"  Symbols:    {stats['n_symbols']}")
print(f"  Total Size: {stats['total_size_mb']} MB")
print(f"  Directory:  {stats['cache_dir']}")

if symbols:
    first_10 = ', '.join(symbols[:10])
    suffix = '...' if len(symbols) > 10 else ''
    print(f"  First 10:   {first_10}{suffix}")
