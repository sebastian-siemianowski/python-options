"""
Arena Experimental Models Registry - Auto-Discovery
====================================================
Automatically discovers and registers all experimental models.
Just add a .py file with a get_*_models() function - no init.py changes needed!
"""
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Type, List
from dataclasses import dataclass
from enum import Enum
class ExperimentalModelFamily(Enum):
    GEOMETRIC_SCORE = "geometric_score"
    STUDENT_T_V2 = "student_t_v2"
    REGIME_COUPLED = "regime_coupled"
    SKEW_ADAPTIVE = "skew_adaptive"
    TAIL_SWITCHING = "tail_switching"
    ELITE_HYBRID = "elite_hybrid"
    DTCWT = "dtcwt"
    CUSTOM = "custom"
@dataclass
class ExperimentalModelSpec:
    name: str
    family: ExperimentalModelFamily
    n_params: int
    param_names: tuple
    description: str
    model_class: Type
EXPERIMENTAL_MODELS: Dict[str, Type] = {}
EXPERIMENTAL_MODEL_SPECS: Dict[str, ExperimentalModelSpec] = {}
EXPERIMENTAL_MODEL_KWARGS: Dict[str, Dict] = {}
_DISCOVERED_MODULES: Dict[str, Any] = {}
def _get_family_enum(family_str: str) -> ExperimentalModelFamily:
    try:
        return ExperimentalModelFamily[family_str.upper().replace("-", "_")]
    except KeyError:
        return ExperimentalModelFamily.CUSTOM
def _register_model(name, cls, kwargs, family, description):
    if name in EXPERIMENTAL_MODELS:
        return
    EXPERIMENTAL_MODELS[name] = cls
    EXPERIMENTAL_MODEL_KWARGS[name] = kwargs or {}
    EXPERIMENTAL_MODEL_SPECS[name] = ExperimentalModelSpec(
        name=name, family=_get_family_enum(family), n_params=3,
        param_names=("q", "c", "phi"), description=description or name,
        model_class=cls
    )
def _discover_models_from_module(module) -> int:
    count = 0
    for attr_name in dir(module):
        if attr_name.startswith("get_") and attr_name.endswith("_models"):
            try:
                getter = getattr(module, attr_name)
                if callable(getter):
                    models = getter()
                    if isinstance(models, list):
                        for m in models:
                            if isinstance(m, dict) and "name" in m and "class" in m:
                                _register_model(m["name"], m["class"],
                                    m.get("kwargs", {}), m.get("family", "custom"),
                                    m.get("description", ""))
                                count += 1
            except Exception:
                pass
    return count
def _discover_all_models():
    current_dir = Path(__file__).parent
    for py_file in sorted(current_dir.glob("*.py")):
        filename = py_file.stem
        if filename in ("__init__", "base", "__pycache__"):
            continue
        if filename in _DISCOVERED_MODULES:
            continue
        try:
            module_name = f"arena.experimental_models.{filename}"
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    continue
            _discover_models_from_module(module)
            _DISCOVERED_MODULES[filename] = module
        except Exception as e:
            print(f"Warning: Could not import {filename}: {e}")
def get_experimental_model(name: str) -> Type:
    if name not in EXPERIMENTAL_MODELS:
        _discover_all_models()
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return EXPERIMENTAL_MODELS[name]
def get_experimental_model_specs() -> List[ExperimentalModelSpec]:
    return list(EXPERIMENTAL_MODEL_SPECS.values())
def create_experimental_model(name: str, **kwargs) -> Any:
    if name not in EXPERIMENTAL_MODELS:
        _discover_all_models()
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    defaults = EXPERIMENTAL_MODEL_KWARGS.get(name, {})
    return EXPERIMENTAL_MODELS[name](**{**defaults, **kwargs})
def list_experimental_models() -> List[str]:
    return list(EXPERIMENTAL_MODELS.keys())
def get_model_count() -> int:
    return len(EXPERIMENTAL_MODELS)
def refresh_models():
    global _DISCOVERED_MODULES
    _DISCOVERED_MODULES = {}
    EXPERIMENTAL_MODELS.clear()
    EXPERIMENTAL_MODEL_SPECS.clear()
    EXPERIMENTAL_MODEL_KWARGS.clear()
    _discover_all_models()
_discover_all_models()
__all__ = ["EXPERIMENTAL_MODELS", "EXPERIMENTAL_MODEL_SPECS", "EXPERIMENTAL_MODEL_KWARGS",
    "ExperimentalModelSpec", "ExperimentalModelFamily", "get_experimental_model",
    "get_experimental_model_specs", "create_experimental_model", "list_experimental_models",
    "get_model_count", "refresh_models"]
