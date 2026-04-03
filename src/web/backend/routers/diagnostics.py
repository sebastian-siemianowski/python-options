"""
Diagnostics router — PIT diagnostics, model comparison, regime distribution.

Equivalent to `make diag` / `make diag-pit` but via API.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/pit-summary")
async def pit_summary():
    """PIT calibration summary for all tuned assets with per-model metrics."""
    from web.backend.services.diagnostics_service import get_pit_summary
    return get_pit_summary()


@router.get("/calibration-failures")
async def calibration_failures():
    """Get assets that failed calibration (from calibration_failures.json)."""
    from web.backend.services.diagnostics_service import get_calibration_failures
    return get_calibration_failures()


@router.get("/model-comparison")
async def model_comparison():
    """Model comparison across all assets — win rates, avg weights."""
    from web.backend.services.diagnostics_service import get_model_comparison
    return get_model_comparison()


@router.get("/regime-distribution")
async def regime_distribution():
    """Regime classification distribution across all assets."""
    from web.backend.services.diagnostics_service import get_regime_distribution
    return get_regime_distribution()


@router.get("/cross-asset-summary")
async def cross_asset_summary():
    """Cross-asset summary matrices — assets × models with PIT/CRPS/BIC scores."""
    from web.backend.services.diagnostics_service import get_cross_asset_summary
    return get_cross_asset_summary()


@router.get("/profitability")
async def profitability_metrics():
    """Story 8.3: Time-series of profitability metrics for monitoring dashboard."""
    import json
    import os

    history_path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
        "data", "calibration", "profitability_history.json",
    )
    history_path = os.path.abspath(history_path)

    # Load from validation_config
    targets = {
        "hit_rate_7d": 0.55,
        "hit_rate_21d": 0.53,
        "signal_rate": 0.15,
        "sharpe_7d": 0.50,
        "ece_max": 0.03,
        "max_drawdown_single": 0.30,
    }

    if os.path.isfile(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        # Return empty structure with targets
        history = {
            "timestamps": [],
            "hit_rates": {"7d": [], "21d": []},
            "signal_rates": [],
            "sharpe": {"7d": [], "21d": []},
            "crps": [],
            "ece": [],
        }

    history["targets"] = targets
    return history
