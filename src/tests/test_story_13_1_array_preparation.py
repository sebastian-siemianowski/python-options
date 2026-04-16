"""
Story 13.1 – Array Preparation Correctness Audit
==================================================
Verify prepare_arrays() preserves float64 precision, C-contiguity,
handles type promotion, and works with edge cases.
"""

import os, sys, time
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import prepare_arrays, validate_prepare_arrays


class TestFloat64Preservation:
    """Prepared arrays are always float64."""

    def test_float64_passthrough(self):
        """float64 arrays pass through unchanged."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert result.dtype == np.float64
        assert result is a  # no copy

    def test_float32_promoted(self):
        """float32 promoted to float64."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = prepare_arrays(a)
        assert result.dtype == np.float64

    def test_int32_promoted(self):
        """int32 promoted to float64."""
        a = np.array([1, 2, 3], dtype=np.int32)
        (result,) = prepare_arrays(a)
        assert result.dtype == np.float64

    def test_int64_promoted(self):
        """int64 promoted to float64."""
        a = np.array([1, 2, 3], dtype=np.int64)
        (result,) = prepare_arrays(a)
        assert result.dtype == np.float64

    def test_precision_preserved(self):
        """Small value differences preserved after preparation."""
        a = np.array([1.0, 1.0 + 1e-15, 1.0 + 2e-15], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert result[1] - result[0] == a[1] - a[0]


class TestCContiguity:
    """All prepared arrays are C-contiguous."""

    def test_contiguous_passthrough(self):
        """Already contiguous arrays pass through."""
        a = np.ascontiguousarray([1.0, 2.0, 3.0], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert result.flags["C_CONTIGUOUS"]

    def test_fortran_order_fixed(self):
        """Fortran-order arrays become C-contiguous."""
        a = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        (result,) = prepare_arrays(a)
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == np.float64

    def test_slice_fixed(self):
        """Non-contiguous slices become contiguous."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        sliced = a[::2]  # non-contiguous
        (result,) = prepare_arrays(sliced)
        assert result.flags["C_CONTIGUOUS"]


class TestTwoArrayFastPath:
    """Fast path for 2 arrays (most common case)."""

    def test_two_arrays_passthrough(self):
        """Two good arrays pass through without copy."""
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        r_a, r_b = prepare_arrays(a, b)
        assert r_a is a
        assert r_b is b

    def test_two_arrays_mixed(self):
        """One float64, one float32 -> both float64."""
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float32)
        r_a, r_b = prepare_arrays(a, b)
        assert r_a.dtype == np.float64
        assert r_b.dtype == np.float64


class TestEdgeCases:
    """Edge cases: empty, single element, NaN, Inf."""

    def test_empty_array(self):
        """Empty array handled."""
        a = np.array([], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert len(result) == 0
        assert result.dtype == np.float64

    def test_single_element(self):
        """Single element array."""
        a = np.array([42.0], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert result[0] == 42.0

    def test_nan_preserved(self):
        """NaN values are preserved (not silently dropped)."""
        a = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert np.isnan(result[1])

    def test_inf_preserved(self):
        """Inf values are preserved."""
        a = np.array([1.0, np.inf, -np.inf], dtype=np.float64)
        (result,) = prepare_arrays(a)
        assert np.isinf(result[1])
        assert np.isinf(result[2])


class TestValidationDiagnostic:
    """validate_prepare_arrays() returns correct diagnostics."""

    def test_all_valid(self):
        """Two good float64 arrays -> all_valid = True."""
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        diag = validate_prepare_arrays(a, b)
        assert diag["all_valid"]
        assert diag["n_arrays"] == 2

    def test_mixed_types_valid_after_prep(self):
        """Mixed types -> all_valid after preparation."""
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4.0, 5.0], dtype=np.float32)
        diag = validate_prepare_arrays(a, b)
        assert diag["all_valid"]
        for arr_diag in diag["arrays"]:
            assert arr_diag["is_float64"]
            assert arr_diag["is_c_contiguous"]

    def test_nan_detection(self):
        """NaN detected in diagnostics."""
        a = np.array([1.0, np.nan, 3.0])
        diag = validate_prepare_arrays(a)
        assert diag["arrays"][0]["has_nan"]
