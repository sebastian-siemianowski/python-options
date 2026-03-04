"""Pytest configuration for test discovery and parallel execution."""

# Exclude script-style test files that don't use pytest functions/classes.
# These are standalone scripts with top-level code that would fail during collection.
collect_ignore = [
    "test_online_persistence.py",
    "test_gaussian_pit_failures.py",
    "test_pit_penalty.py",
    "test_metals_temp.py",
    "test_momentum_integration.py",
    "test_numba_quick.py",
    "test_v6_smoke.py",
]
