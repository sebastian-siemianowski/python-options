"""Arena Experimental Models Registry Gen17: 60 Models"""

from typing import Dict, Any, Type, List
from dataclasses import dataclass
from enum import Enum


class ExperimentalModelFamily(Enum):
    OPTIMAL_HYV = "optimal_hyv"
    CSS_NEGHYV = "css_neghyv"
    ULTIMATE = "ultimate"
    MERGED_ELITE = "merged_elite"
    EXACT_CLONE = "exact_clone"
    HYV_AWARE = "hyv_aware"


@dataclass
class ExperimentalModelSpec:
    name: str
    family: ExperimentalModelFamily
    n_params: int
    param_names: tuple
    description: str
    model_class: Type


from .gen13_optimal_hyv_batch1 import BATCH1_MODELS as GEN13_BATCH1
from .gen13_css_neghyv_batch2 import BATCH2_MODELS as GEN13_BATCH2
from .gen14_ultimate_batch1 import BATCH1_MODELS as GEN14_BATCH1
from .gen15_merged_elite_batch1 import BATCH1_MODELS as GEN15_BATCH1
from .gen16_exact_clone_batch1 import BATCH1_MODELS as GEN16_BATCH1
from .gen17_hyv_aware_batch1 import BATCH1_MODELS as GEN17_BATCH1

EXPERIMENTAL_MODELS: Dict[str, Type] = {**GEN13_BATCH1, **GEN13_BATCH2, **GEN14_BATCH1, **GEN15_BATCH1, **GEN16_BATCH1, **GEN17_BATCH1}


def _create_spec(name, family, model_class):
    return ExperimentalModelSpec(
        name=name, family=family, n_params=4,
        param_names=("q", "c", "phi", "complex_weight"),
        description=f"{family.value} model: {name}",
        model_class=model_class
    )


EXPERIMENTAL_MODEL_SPECS: Dict[str, ExperimentalModelSpec] = {}
for name, cls in GEN13_BATCH1.items():
    EXPERIMENTAL_MODEL_SPECS[name] = _create_spec(name, ExperimentalModelFamily.OPTIMAL_HYV, cls)
for name, cls in GEN13_BATCH2.items():
    EXPERIMENTAL_MODEL_SPECS[name] = _create_spec(name, ExperimentalModelFamily.CSS_NEGHYV, cls)
for name, cls in GEN14_BATCH1.items():
    EXPERIMENTAL_MODEL_SPECS[name] = _create_spec(name, ExperimentalModelFamily.ULTIMATE, cls)
for name, cls in GEN15_BATCH1.items():
    EXPERIMENTAL_MODEL_SPECS[name] = _create_spec(name, ExperimentalModelFamily.MERGED_ELITE, cls)
for name, cls in GEN16_BATCH1.items():
    EXPERIMENTAL_MODEL_SPECS[name] = _create_spec(name, ExperimentalModelFamily.EXACT_CLONE, cls)
for name, cls in GEN17_BATCH1.items():
    EXPERIMENTAL_MODEL_SPECS[name] = _create_spec(name, ExperimentalModelFamily.HYV_AWARE, cls)


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
