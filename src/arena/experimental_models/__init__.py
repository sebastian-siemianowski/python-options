"""Arena Experimental Models Registry - Empty"""
from typing import Dict,Any,Type
from dataclasses import dataclass
from enum import Enum
class ExperimentalModelFamily(Enum):
    ELITE_HYBRID="elite_hybrid"
@dataclass
class ExperimentalModelSpec:
    name:str
    family:ExperimentalModelFamily
    n_params:int
    param_names:tuple
    description:str
    model_class:Type
EXPERIMENTAL_MODELS:Dict[str,Type]={}
EXPERIMENTAL_MODEL_SPECS:Dict[str,ExperimentalModelSpec]={}
EXPERIMENTAL_MODEL_KWARGS:Dict[str,Dict]={}
def get_experimental_model(n):return EXPERIMENTAL_MODELS.get(n)
def get_experimental_model_specs():return list(EXPERIMENTAL_MODEL_SPECS.values())
def create_experimental_model(n,**kw):return EXPERIMENTAL_MODELS.get(n)
def list_experimental_models():return list(EXPERIMENTAL_MODELS.keys())
def get_model_count():return len(EXPERIMENTAL_MODELS)
