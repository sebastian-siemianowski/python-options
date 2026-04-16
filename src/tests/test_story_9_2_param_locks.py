"""
Story 9.2: Hierarchical Parameter Optimization
================================================
Parameter locks across optimization stages to prevent error propagation.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.phi_student_t_unified import HierarchicalParameterLock


class TestHierarchicalParameterLock:
    """Acceptance criteria for Story 9.2."""

    def test_stage1_params_locked_in_stage3(self):
        """AC1: Stage 1 params (q, c, phi) locked during Stage 3."""
        lock = HierarchicalParameterLock()
        assert lock.is_locked('q', 3)
        assert lock.is_locked('c', 3)
        assert lock.is_locked('phi', 3)

    def test_beta_free_in_stage5(self):
        """AC2: Stage 2 (beta) allowed to update in Stage 5."""
        lock = HierarchicalParameterLock()
        free = lock.get_free_params(5)
        assert 'beta' in free

    def test_garch_bounds_from_stage1(self):
        """AC3: GARCH parameters bounded by 2x Stage 1 variance scale."""
        lock = HierarchicalParameterLock()
        lock.lock_after_stage(1, {'q': 1e-5, 'c': 0.01, 'phi': 0.8},
                              stage1_variance_scale=0.001)
        bounds = lock.get_bounds('garch_omega')
        assert bounds is not None
        assert bounds[1] == pytest.approx(0.002, rel=1e-6)
        persist_bounds = lock.get_bounds('garch_persistence')
        assert persist_bounds[1] == 0.999
        leverage_bounds = lock.get_bounds('garch_leverage')
        assert leverage_bounds[1] == 0.5

    def test_locked_values_stored(self):
        """AC4: Locked values are stored and retrievable."""
        lock = HierarchicalParameterLock()
        lock.lock_after_stage(1, {'q': 1e-5, 'c': 0.01, 'phi': 0.85})
        assert lock.get_locked_value('q') == 1e-5
        assert lock.get_locked_value('phi') == 0.85

    def test_stage1_nothing_locked(self):
        """AC5: Nothing locked during Stage 1 (first stage)."""
        lock = HierarchicalParameterLock()
        assert not lock.is_locked('q', 1)
        assert not lock.is_locked('c', 1)
        assert not lock.is_locked('phi', 1)

    def test_diagnostics(self):
        """AC6: Diagnostics show lock state."""
        lock = HierarchicalParameterLock()
        lock.lock_after_stage(1, {'q': 1e-5, 'c': 0.01})
        diag = lock.get_diagnostics()
        assert diag['n_locked'] == 2
        assert 'q' in diag['locked_params']

    def test_garch_locked_in_stage4(self):
        """AC7: GARCH params locked in stages 4 and 5."""
        lock = HierarchicalParameterLock()
        assert lock.is_locked('garch_omega', 4)
        assert lock.is_locked('garch_alpha', 5)
