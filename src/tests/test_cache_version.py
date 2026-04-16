"""
Test Story 3.8: Cache Versioning and Migration.

Validates:
  1. v1 detected correctly
  2. v1->v2 migration adds defaults
  3. Migration is idempotent
  4. Backup created
  5. Already v2 untouched
  6. load_and_migrate end-to-end
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from calibration.cache_version import (
    detect_version,
    migrate_v1_to_v2,
    run_migrations,
    load_and_migrate,
    stamp_version,
    CURRENT_CACHE_VERSION,
    GARCH_DEFAULTS,
    OU_DEFAULTS,
)


class TestCacheVersioning(unittest.TestCase):
    """Tests for cache versioning and migration."""

    def test_detect_v1(self):
        """Unversioned cache detected as v1."""
        data = {"q": 1e-5, "c": 1.0, "phi": 0.98}
        self.assertEqual(detect_version(data), 1)

    def test_detect_v2(self):
        """Versioned cache detected correctly."""
        data = {"cache_version": 2, "q": 1e-5}
        self.assertEqual(detect_version(data), 2)

    def test_v1_to_v2_adds_defaults(self):
        """v1 -> v2 adds garch_params and ou_params."""
        data = {"q": 1e-5, "c": 1.0}
        result = migrate_v1_to_v2(data)
        self.assertEqual(result["cache_version"], 2)
        self.assertIn("garch_params", result)
        self.assertIn("ou_params", result)
        self.assertEqual(result["garch_params"]["alpha"], GARCH_DEFAULTS["alpha"])
        self.assertEqual(result["ou_params"]["kappa"], OU_DEFAULTS["kappa"])

    def test_migration_idempotent(self):
        """Running migrations on v2 does nothing."""
        data = {"cache_version": 2, "garch_params": {"alpha": 0.1}, "ou_params": {"kappa": 0.03}}
        result = run_migrations(data)
        self.assertEqual(result["garch_params"]["alpha"], 0.1)  # Not overwritten
        self.assertEqual(result["ou_params"]["kappa"], 0.03)

    def test_already_v2_no_change(self):
        """v2 cache passes through unchanged."""
        data = {"cache_version": 2, "q": 1e-5}
        result = run_migrations(data)
        self.assertEqual(detect_version(result), 2)

    def test_load_and_migrate_creates_backup(self):
        """load_and_migrate creates .bak file for v1."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"q": 1e-5, "c": 1.0}, f)
            path = f.name
        
        try:
            result = load_and_migrate(path, backup=True)
            self.assertIsNotNone(result)
            self.assertEqual(result["cache_version"], 2)
            self.assertTrue(os.path.exists(path + ".bak"))
            
            # Original file also updated
            with open(path) as f:
                saved = json.load(f)
            self.assertEqual(saved["cache_version"], 2)
        finally:
            os.unlink(path)
            if os.path.exists(path + ".bak"):
                os.unlink(path + ".bak")

    def test_load_nonexistent_returns_none(self):
        """Missing file returns None."""
        result = load_and_migrate("/tmp/does_not_exist_xyz.json")
        self.assertIsNone(result)

    def test_stamp_version(self):
        """stamp_version sets current version."""
        data = {"q": 1e-5}
        result = stamp_version(data)
        self.assertEqual(result["cache_version"], CURRENT_CACHE_VERSION)

    def test_v2_no_backup_created(self):
        """Already-v2 file should NOT create backup."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"cache_version": 2, "q": 1e-5}, f)
            path = f.name
        
        try:
            load_and_migrate(path, backup=True)
            self.assertFalse(os.path.exists(path + ".bak"))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
