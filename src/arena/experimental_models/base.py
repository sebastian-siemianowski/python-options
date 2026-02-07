"""
===============================================================================
BASE — Experimental Model Base Classes and Specifications
===============================================================================

Common base classes, enums, and specifications for experimental models.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any
from enum import Enum


class ExperimentalModelFamily(Enum):
    """Experimental model families."""
    STUDENT_T_V2 = "student_t_v2"
    REGIME_COUPLED = "regime_coupled"
    SKEW_ADAPTIVE = "skew_adaptive"
    TAIL_SWITCHING = "tail_switching"


@dataclass
class ExperimentalModelSpec:
    """
    Specification for an experimental model.
    
    Attributes:
        name: Unique model identifier
        family: Model family
        n_params: Number of free parameters
        param_names: Parameter names
        default_params: Default parameter values
        description: Human-readable description
    """
    name: str
    family: ExperimentalModelFamily
    n_params: int
    param_names: Tuple[str, ...]
    default_params: Dict[str, float]
    description: str
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Check if provided params contain all required parameter names."""
        return all(p in params for p in self.param_names)


class BaseExperimentalModel:
    """
    Base class for experimental models.
    
    All experimental models should inherit from this class and implement:
        - fit(returns, vol, init_params) -> Dict with fitted parameters
        - filter(returns, vol, **params) -> (mu, P, log_likelihood)
    """
    
    def fit(self, returns, vol, init_params=None):
        """Fit model parameters via MLE."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def filter(self, returns, vol, **params):
        """Run Kalman filter with given parameters."""
        raise NotImplementedError("Subclasses must implement filter()")
