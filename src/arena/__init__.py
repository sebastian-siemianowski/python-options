"""
===============================================================================
ARENA — Experimental Model Competition Framework
===============================================================================

A laboratory for testing experimental distributional models against the
production-grade momentum-augmented ensemble before promoting to main tuning.

Architecture: 
    - Isolated sandbox for model development
    - Standardized benchmark suite (small/mid/large cap + index)
    - Head-to-head scoring against standard momentum models
    - Promotion gate: experimental models must beat standard to graduate

STANDARD MODELS (Baselines to beat):
    - momentum_gaussian
    - momentum_phi_gaussian  
    - momentum_phi_student_t_nu_{4,6,8,12,20}

GOVERNANCE:
    - Arena tests run ONLY via `make arena-*` commands
    - Standard `make tune` NEVER touches experimental models
    - Experimental models live in arena_models/ (not models/)
    - Promotion requires panel review after beating baselines

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from .arena_config import (
    ArenaConfig,
    ARENA_BENCHMARK_SYMBOLS,
    SMALL_CAP_SYMBOLS,
    MID_CAP_SYMBOLS,
    LARGE_CAP_SYMBOLS,
    INDEX_SYMBOLS,
    DEFAULT_ARENA_CONFIG,
)

from .arena_data import (
    download_arena_data,
    load_arena_data,
    ArenaDataset,
)

from .arena_tune import (
    run_arena_competition,
    ArenaResult,
    ModelScore,
)

from .arena_models import (
    EXPERIMENTAL_MODELS,
    get_experimental_model_specs,
    get_standard_model_specs,
)

__all__ = [
    # Config
    'ArenaConfig',
    'ARENA_BENCHMARK_SYMBOLS',
    'SMALL_CAP_SYMBOLS',
    'MID_CAP_SYMBOLS',
    'LARGE_CAP_SYMBOLS',
    'INDEX_SYMBOLS',
    'DEFAULT_ARENA_CONFIG',
    # Data
    'download_arena_data',
    'load_arena_data',
    'ArenaDataset',
    # Tune
    'run_arena_competition',
    'ArenaResult',
    'ModelScore',
    # Models
    'EXPERIMENTAL_MODELS',
    'get_experimental_model_specs',
    'get_standard_model_specs',
]
