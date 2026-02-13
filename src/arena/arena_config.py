"""
===============================================================================
ARENA CONFIGURATION — Benchmark Universe & Competition Settings
===============================================================================

Defines the standardized benchmark universe for model competition:
    - 3 Small Cap stocks (high volatility, illiquidity challenges)
    - 3 Mid Cap stocks (moderate dynamics)
    - 3 Large Cap stocks (institutional-grade liquidity)
    - 3 Index ETFs (diversified, regime-sensitive)

Selection Criteria (Chinese Staff Professor methodology):
    - Small Cap: Market cap < $2B, ADV > 500K, options available
    - Mid Cap: Market cap $2B-$10B, ADV > 1M
    - Large Cap: Market cap > $100B, highly liquid options
    - Index: Broad market exposure, highest liquidity

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class CapCategory(Enum):
    """Market capitalization category."""
    SMALL_CAP = "small_cap"
    MID_CAP = "mid_cap"
    LARGE_CAP = "large_cap"
    INDEX = "index"


# =============================================================================
# BENCHMARK UNIVERSE — Carefully Selected Representative Symbols
# =============================================================================

# Small Cap (high vol, liquidity challenges, options available)
SMALL_CAP_SYMBOLS = [
    "UPST",   # Upstart Holdings - AI lending, extreme vol
    "AFRM",   # Affirm - BNPL fintech, high beta
    "IONQ",   # IonQ - Quantum computing, speculative
]

# Mid Cap (moderate dynamics, good option liquidity)
MID_CAP_SYMBOLS = [
    "CRWD",   # CrowdStrike - Cybersecurity leader
    "DKNG",   # DraftKings - Sports betting
    "SNAP",   # Snap Inc - Social media, vol clustering
]

# Large Cap (institutional liquidity, stable dynamics)
LARGE_CAP_SYMBOLS = [
    "AAPL",   # Apple - Mega cap, benchmark quality
    "NVDA",   # NVIDIA - AI semiconductor leader
    "TSLA",   # Tesla - High vol mega cap
]

# Index ETFs (regime-sensitive, highest liquidity)
INDEX_SYMBOLS = [
    "SPY",    # S&P 500 - Ultimate benchmark
    "QQQ",    # Nasdaq-100 - Tech-heavy
    "IWM",    # Russell 2000 - Small cap exposure
]

# Complete benchmark universe
ARENA_BENCHMARK_SYMBOLS = (
    SMALL_CAP_SYMBOLS + 
    MID_CAP_SYMBOLS + 
    LARGE_CAP_SYMBOLS + 
    INDEX_SYMBOLS
)

# Symbol to category mapping
SYMBOL_CATEGORIES: Dict[str, CapCategory] = {}
for sym in SMALL_CAP_SYMBOLS:
    SYMBOL_CATEGORIES[sym] = CapCategory.SMALL_CAP
for sym in MID_CAP_SYMBOLS:
    SYMBOL_CATEGORIES[sym] = CapCategory.MID_CAP
for sym in LARGE_CAP_SYMBOLS:
    SYMBOL_CATEGORIES[sym] = CapCategory.LARGE_CAP
for sym in INDEX_SYMBOLS:
    SYMBOL_CATEGORIES[sym] = CapCategory.INDEX


# =============================================================================
# ARENA CONFIGURATION
# =============================================================================

@dataclass
class ArenaConfig:
    """
    Configuration for arena model competition.
    
    Attributes:
        symbols: List of symbols to test (default: full benchmark)
        lookback_years: Years of historical data for tuning
        min_observations: Minimum data points required
        test_experimental: Whether to test experimental models
        test_standard: Whether to test standard models
        scoring_method: 'bic', 'hyvarinen', 'combined', or 'pit'
        pit_threshold: PIT p-value threshold for calibration
        promotion_threshold: Score improvement required for promotion
        n_regimes: Number of market regimes
        temporal_alpha: Temporal smoothing parameter
        parallel_workers: Number of parallel workers
        verbose: Verbose output
    """
    symbols: List[str] = field(default_factory=lambda: list(ARENA_BENCHMARK_SYMBOLS))
    lookback_years: int = 5
    min_observations: int = 252  # 1 year minimum
    test_experimental: bool = True
    test_standard: bool = True
    scoring_method: str = "combined"  # 'bic', 'hyvarinen', 'combined', 'pit'
    pit_threshold: float = 0.05  # p-value threshold
    promotion_threshold: float = 0.05  # 5% improvement required
    n_regimes: int = 5
    temporal_alpha: float = 0.7
    parallel_workers: int = 4
    verbose: bool = True
    
    # Output paths
    arena_data_dir: str = "src/arena/data"
    arena_results_dir: str = "src/arena/data/results"
    
    def get_symbols_by_category(self, category: CapCategory) -> List[str]:
        """Get symbols for a specific market cap category."""
        return [s for s in self.symbols if SYMBOL_CATEGORIES.get(s) == category]
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.symbols:
            raise ValueError("No symbols configured")
        if self.lookback_years < 1:
            raise ValueError("lookback_years must be >= 1")
        if self.scoring_method not in ('bic', 'hyvarinen', 'combined', 'pit'):
            raise ValueError(f"Invalid scoring_method: {self.scoring_method}")
        return True


# Default configuration
DEFAULT_ARENA_CONFIG = ArenaConfig()


# =============================================================================
# SCORING WEIGHTS BY CATEGORY
# =============================================================================
# Different market cap categories have different statistical properties.
# Weight scoring to emphasize relevant characteristics.
# =============================================================================

CATEGORY_SCORING_WEIGHTS: Dict[CapCategory, Dict[str, float]] = {
    CapCategory.SMALL_CAP: {
        "bic": 0.3,       # Less weight on BIC (sparse data)
        "hyvarinen": 0.3, # Robustness matters
        "pit": 0.4,       # Calibration critical for high vol
    },
    CapCategory.MID_CAP: {
        "bic": 0.4,
        "hyvarinen": 0.3,
        "pit": 0.3,
    },
    CapCategory.LARGE_CAP: {
        "bic": 0.5,       # More data → trust BIC
        "hyvarinen": 0.25,
        "pit": 0.25,
    },
    CapCategory.INDEX: {
        "bic": 0.4,
        "hyvarinen": 0.2,
        "pit": 0.4,       # Regime calibration crucial
    },
}


def get_category_weights(symbol: str) -> Dict[str, float]:
    """Get scoring weights for a symbol based on its category."""
    category = SYMBOL_CATEGORIES.get(symbol, CapCategory.MID_CAP)
    return CATEGORY_SCORING_WEIGHTS[category]


# =============================================================================
# DISABLED MODELS — Models that have failed against standard baselines
# =============================================================================

import json
from pathlib import Path
from datetime import datetime

DISABLED_MODELS_FILE = "src/arena/disabled/disabled_models.json"


def load_disabled_models() -> Dict[str, Dict]:
    """
    Load list of disabled experimental models.
    
    Returns:
        Dict mapping model_name to disable info:
        {
            "model_name": {
                "disabled_at": "2026-02-07T10:30:00",
                "reason": "Failed against kalman_gaussian_momentum by -45.2%",
                "best_std_model": "kalman_gaussian_momentum",
                "score_gap": -0.452
            }
        }
    """
    filepath = Path(DISABLED_MODELS_FILE)
    if not filepath.exists():
        return {}
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_disabled_models(disabled: Dict[str, Dict]) -> None:
    """Save disabled models to file."""
    filepath = Path(DISABLED_MODELS_FILE)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(disabled, f, indent=2)


def disable_model(
    model_name: str,
    best_std_model: str,
    score_gap: float,
    reason: str = None,
) -> None:
    """
    Disable an experimental model that failed against standards.
    
    Args:
        model_name: Name of the model to disable
        best_std_model: Name of the best standard model it failed against
        score_gap: Score gap (negative means failed)
        reason: Optional reason string
    """
    disabled = load_disabled_models()
    
    if reason is None:
        reason = f"Failed against {best_std_model} by {score_gap*100:.1f}%"
    
    disabled[model_name] = {
        "disabled_at": datetime.now().isoformat(),
        "reason": reason,
        "best_std_model": best_std_model,
        "score_gap": score_gap,
    }
    
    save_disabled_models(disabled)


def enable_model(model_name: str) -> bool:
    """
    Re-enable a previously disabled model.
    
    Args:
        model_name: Name of the model to re-enable
        
    Returns:
        True if model was re-enabled, False if it wasn't disabled
    """
    disabled = load_disabled_models()
    
    if model_name in disabled:
        del disabled[model_name]
        save_disabled_models(disabled)
        return True
    
    return False


def is_model_disabled(model_name: str) -> bool:
    """Check if a model is disabled."""
    disabled = load_disabled_models()
    return model_name in disabled


def get_enabled_experimental_models() -> list:
    """Get list of experimental models that are not disabled."""
    from .experimental_models import EXPERIMENTAL_MODELS
    
    disabled = load_disabled_models()
    return [name for name in EXPERIMENTAL_MODELS.keys() if name not in disabled]


def clear_disabled_models() -> int:
    """
    Clear all disabled models (re-enable all).
    
    Returns:
        Number of models that were re-enabled
    """
    disabled = load_disabled_models()
    count = len(disabled)
    
    if count > 0:
        save_disabled_models({})
    
    return count
