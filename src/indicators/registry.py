"""
Strategy registry: maps strategy IDs to (name, family, compute_fn).
Auto-discovers all strategy families and builds master registry.
"""

from indicators.families import (
    trend_momentum,
    mean_reversion,
    volatility,
    volume_micro,
    pattern_fractal,
    oscillator_cycle,
    multifactor_regime,
    derivatives,
    cross_asset,
    hybrid_ensemble,
)

# Each family module exposes STRATEGIES: dict[int, tuple[str, Callable]]
_FAMILY_MODULES = [
    ("Trend & Momentum", trend_momentum),
    ("Mean Reversion", mean_reversion),
    ("Volatility", volatility),
    ("Volume & Microstructure", volume_micro),
    ("Pattern & Fractal", pattern_fractal),
    ("Oscillator & Cycle", oscillator_cycle),
    ("Multi-Factor & Regime", multifactor_regime),
    ("Derivatives-Informed", derivatives),
    ("Cross-Asset Macro", cross_asset),
    ("Hybrid Ensemble", hybrid_ensemble),
]

# Master registry: {id: {"name": str, "family": str, "fn": Callable}}
REGISTRY: dict = {}


def _build_registry():
    """Build the master registry from all family modules."""
    global REGISTRY
    REGISTRY.clear()
    for family_name, module in _FAMILY_MODULES:
        strategies = getattr(module, "STRATEGIES", {})
        for sid, (name, fn) in strategies.items():
            REGISTRY[sid] = {
                "name": name,
                "family": family_name,
                "fn": fn,
            }


def get_strategy(sid: int) -> dict | None:
    """Get strategy by ID. Returns {"name", "family", "fn"} or None."""
    if not REGISTRY:
        _build_registry()
    return REGISTRY.get(sid)


def get_all_strategies() -> dict:
    """Get full registry."""
    if not REGISTRY:
        _build_registry()
    return REGISTRY


def get_family_strategies(family: str) -> dict:
    """Get strategies for a specific family."""
    if not REGISTRY:
        _build_registry()
    return {sid: v for sid, v in REGISTRY.items() if v["family"] == family}


def get_strategy_ids() -> list[int]:
    """Get all strategy IDs sorted."""
    if not REGISTRY:
        _build_registry()
    return sorted(REGISTRY.keys())


def get_families() -> list[str]:
    """Get list of family names."""
    return [name for name, _ in _FAMILY_MODULES]
