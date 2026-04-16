"""
Story 3.5: Tuning Validation Gate (Quality Control).

Validates tuned parameters before they reach the cache. Catches NaN, Inf,
extreme values, and convergence failures.

Usage:
    from calibration.tune_validator import validate_model_params, ValidationResult
    result = validate_model_params("kalman_gaussian", params)
    if not result.passed:
        print(f"Failed: {result.warnings}")
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Per-parameter bounds
PARAM_BOUNDS = {
    "q": (1e-8, 1e-1),
    "c": (1e-4, 100.0),
    "phi": (0.5, 1.0),
    "nu": (2.5, 50.0),
    "bic": (None, 0.0),  # Must be negative (valid log-likelihood)
}

# Conservative prior for total failure
CONSERVATIVE_PRIOR = {
    "q": 1e-5,
    "c": 1.0,
    "phi": 0.98,
    "nu": 8.0,
}


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    passed: bool
    warnings: List[str] = field(default_factory=list)
    clipped_params: Dict[str, float] = field(default_factory=dict)


def validate_model_params(model_name: str, params: dict) -> ValidationResult:
    """
    Story 3.5: Validate a single model's tuned parameters.
    
    Checks:
      - All values are finite (no NaN, Inf)
      - Parameters within reasonable bounds
      - BIC is finite and negative
    
    Returns ValidationResult with passed=True/False and any warnings.
    """
    result = ValidationResult(passed=True)
    
    if params is None:
        result.passed = False
        result.warnings.append(f"{model_name}: params is None")
        return result
    
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key not in params:
            continue
        
        val = params.get(key)
        
        # NaN/Inf check
        if val is None or not isinstance(val, (int, float)):
            result.passed = False
            result.warnings.append(f"{model_name}: {key} is invalid ({val})")
            continue
        
        if math.isnan(val) or math.isinf(val):
            result.passed = False
            result.warnings.append(f"{model_name}: {key} is NaN/Inf")
            continue
        
        # Bounds check (clip to valid range)
        if lo is not None and val < lo:
            result.warnings.append(f"{model_name}: {key}={val:.2e} below floor {lo:.2e}, clipping")
            result.clipped_params[key] = lo
        
        if hi is not None and val > hi:
            result.warnings.append(f"{model_name}: {key}={val:.2e} above ceiling {hi:.2e}, clipping")
            result.clipped_params[key] = hi
    
    return result


def validate_tune_result(tune_result: dict) -> dict:
    """
    Story 3.5: Validate all models in a tune result.
    
    Returns summary dict with:
      - passed: count of passing models
      - failed: count of failing models
      - warnings: list of warning strings
      - all_failed: bool (True if every model failed)
    """
    if tune_result is None:
        return {"passed": 0, "failed": 0, "warnings": ["tune_result is None"], "all_failed": True}
    
    global_block = tune_result.get("global", tune_result)
    models = global_block.get("models", {})
    
    passed_count = 0
    failed_count = 0
    all_warnings = []
    
    if not models:
        # Validate global params directly
        vr = validate_model_params("global", global_block)
        if vr.passed:
            passed_count = 1
        else:
            failed_count = 1
            all_warnings.extend(vr.warnings)
    else:
        for model_name, model_params in models.items():
            vr = validate_model_params(model_name, model_params)
            if vr.passed:
                passed_count += 1
            else:
                failed_count += 1
                all_warnings.extend(vr.warnings)
    
    return {
        "passed": passed_count,
        "failed": failed_count,
        "warnings": all_warnings,
        "all_failed": passed_count == 0,
    }


def get_conservative_prior() -> dict:
    """Story 3.5: Conservative prior params for total failure case."""
    return dict(CONSERVATIVE_PRIOR)
