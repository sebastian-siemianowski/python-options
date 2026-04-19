"""
signal_modules -- Modular decomposition of signals.py.

This package contains the extracted configuration, utilities, and computational
modules from the monolithic signals.py file. All public symbols are re-exported
here for backward compatibility.

Usage:
    from decision.signal_modules.config import *  # Feature flags and imports
    from decision.signal_modules.volatility_imports import *  # Volatility framework
    from decision.signal_modules.regime_classification import *  # Regime classification
    from decision.signal_modules.data_fetching import *  # Price data + GARCH + Student-t
"""
from decision.signal_modules.config import *  # noqa: F401,F403
from decision.signal_modules.volatility_imports import *  # noqa: F401,F403
from decision.signal_modules.regime_classification import *  # noqa: F401,F403
from decision.signal_modules.data_fetching import *  # noqa: F401,F403
from decision.signal_modules.momentum_features import *  # noqa: F401,F403
from decision.signal_modules.kalman_diagnostics import *  # noqa: F401,F403
from decision.signal_modules.parameter_loading import *  # noqa: F401,F403
from decision.signal_modules.kalman_filtering import *  # noqa: F401,F403
from decision.signal_modules.hmm_regimes import *  # noqa: F401,F403
from decision.signal_modules.feature_pipeline import *  # noqa: F401,F403
from decision.signal_modules.monte_carlo import *  # noqa: F401,F403
from decision.signal_modules.bma_engine import *  # noqa: F401,F403
from decision.signal_modules.walk_forward import *  # noqa: F401,F403
from decision.signal_modules.signal_dataclass import *  # noqa: F401,F403
from decision.signal_modules.threshold_calibration import *  # noqa: F401,F403
from decision.signal_modules.probability_mapping import *  # noqa: F401,F403
from decision.signal_modules.signal_generation import *  # noqa: F401,F403
from decision.signal_modules.signal_state import *  # noqa: F401,F403
from decision.signal_modules.pnl_attribution import *  # noqa: F401,F403
from decision.signal_modules.comprehensive_diagnostics import *  # noqa: F401,F403
from decision.signal_modules.asset_processing import *  # noqa: F401,F403
from decision.signal_modules.cli import *  # noqa: F401,F403
from decision.signal_modules.cli import _process_assets_with_retries  # noqa: F401
