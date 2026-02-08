"""
Arena Experimental Models Registry

All experimental models have been promoted to safe storage.
No active experimental models.
"""

from typing import Dict, Any, Type, List
from dataclasses import dataclass
from enum import Enum


class ExperimentalModelFamily(Enum):
    ELITE_HYBRID = "elite_hybrid"


@dataclass
class ExperimentalModelSpec:
    name: str
    family: ExperimentalModelFamily
    n_params: int
    param_names: tuple
    description: str
    model_class: Type


# No experimental models - all promoted to safe storage
EXPERIMENTAL_MODELS: Dict[str, Type] = {}


EXPERIMENTAL_MODEL_SPECS: Dict[str, ExperimentalModelSpec] = {}


def get_experimental_model(name):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return EXPERIMENTAL_MODELS[name]


def get_experimental_model_specs():
    return list(EXPERIMENTAL_MODEL_SPECS.values())


def create_experimental_model(name, **kwargs):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return EXPERIMENTAL_MODELS[name](**kwargs)


def list_experimental_models():
    return list(EXPERIMENTAL_MODELS.keys())


def get_model_count():
    return len(EXPERIMENTAL_MODELS)


__all__ = [
    "EXPERIMENTAL_MODELS", "EXPERIMENTAL_MODEL_SPECS",
    "ExperimentalModelSpec", "ExperimentalModelFamily",
    "get_experimental_model", "get_experimental_model_specs",
    "create_experimental_model", "list_experimental_models", "get_model_count",
]