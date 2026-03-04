"""Pytest configuration for test discovery and parallel execution."""

# Exclude script-style test files that don't use pytest functions/classes
collect_ignore = ["test_online_persistence.py"]
