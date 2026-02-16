#!/usr/bin/env python3
"""
===============================================================================
MARKET CONDITIONING LAYER — Cross-Sectional & VIX-Based Model Enhancement
===============================================================================

February 2026 - Expert Panel Recommendations

This module provides market-wide conditioning for asset-specific models:

1. CROSS-SECTIONAL VOLATILITY COUPLING:
   σ²_composite = σ²_asset + β² × σ²_market
   
   Assets don't live in isolation - systemic risk affects all.

2. VIX-CONDITIONAL TAIL THICKNESS:
   ν_t = ν_base - κ × VIX_normalized
   
   Tails thicken when VIX spikes (market-wide fear).

3. MARKET BETA INTEGRATION:
   β_asset = Cov(r_asset, r_SPY) / Var(r_SPY)
   
   High-beta assets get more market vol contribution.

WHY THIS MATTERS:
   - Asset-local models miss systemic risk buildup
   - VIX leading indicator for tail events
   - Cross-asset calibration improves CRPS/CSS scores

REFERENCES:
   Engle, R. (2002). "Dynamic Conditional Correlation"
   Adrian, T. & Brunnermeier, M. (2016). "CoVaR"

===============================================================================
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Market proxy tickers
MARKET_PROXY_TICKER = "SPY"
VIX_TICKER = "^VIX"

# Cache settings
MARKET_DATA_CACHE_HOURS = 4  # Refresh market data every 4 hours
MARKET_DATA_CACHE_DIR = Path(__file__).parent.parent / "data" / "market_cache"

# VIX conditioning parameters
VIX_KAPPA_DEFAULT = 0.15  # Sensitivity of ν to VIX (ν reduction per VIX unit above median)
VIX_MEDIAN_DEFAULT = 18.0  # Historical VIX median
VIX_IQR_DEFAULT = 8.0  # Historical VIX IQR for normalization
NU_MIN_FLOOR = 3.0  # Minimum ν even in extreme stress

# Beta computation
BETA_LOOKBACK_DAYS = 252  # 1 year for beta estimation
BETA_MIN_OBS = 60  # Minimum observations for beta

# Cross-sectional vol coupling
MARKET_VOL_COUPLING_DEFAULT = 0.3  # Default β² contribution


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketConditioningResult:
    """Result of market conditioning computation."""
    # Composite volatility
    vol_composite: np.ndarray
    vol_asset: np.ndarray
    vol_market: np.ndarray
    
    # Beta metrics
    beta: float
    beta_squared: float
    
    # VIX conditioning
    vix_current: Optional[float] = None
    vix_normalized: Optional[float] = None
    nu_adjustment: Optional[float] = None
    
    # Diagnostics
    market_data_available: bool = True
    vix_data_available: bool = True
    estimation_method: str = "full"
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "beta": self.beta,
            "beta_squared": self.beta_squared,
            "vix_current": self.vix_current,
            "vix_normalized": self.vix_normalized,
            "nu_adjustment": self.nu_adjustment,
            "market_data_available": self.market_data_available,
            "vix_data_available": self.vix_data_available,
            "estimation_method": self.estimation_method,
        }


@dataclass
class VIXConditioningResult:
    """Result of VIX-based ν conditioning."""
    nu_original: float
    nu_adjusted: float
    vix_value: float
    vix_normalized: float
    kappa: float
    adjustment_applied: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "nu_original": self.nu_original,
            "nu_adjusted": self.nu_adjusted,
            "vix_value": self.vix_value,
            "vix_normalized": self.vix_normalized,
            "kappa": self.kappa,
            "adjustment_applied": self.adjustment_applied,
        }


# =============================================================================
# MARKET DATA FETCHING (with caching)
# =============================================================================

class MarketDataCache:
    """
    Cached market data provider for SPY/VIX.
    
    Avoids repeated downloads within the same session.
    """
    _instance = None
    _spy_data: Optional[pd.DataFrame] = None
    _vix_data: Optional[pd.DataFrame] = None
    _last_fetch: Optional[datetime] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _needs_refresh(self) -> bool:
        """Check if cache needs refresh."""
        if self._last_fetch is None:
            return True
        age_hours = (datetime.now() - self._last_fetch).total_seconds() / 3600
        return age_hours > MARKET_DATA_CACHE_HOURS
    
    def get_spy_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached SPY data or fetch if needed."""
        if self._needs_refresh() or self._spy_data is None:
            self._fetch_market_data(start_date, end_date)
        return self._spy_data
    
    def get_vix_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached VIX data or fetch if needed."""
        if self._needs_refresh() or self._vix_data is None:
            self._fetch_market_data(start_date, end_date)
        return self._vix_data
    
    def _fetch_market_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> None:
        """Fetch SPY and VIX data from Yahoo Finance."""
        try:
            import yfinance as yf
            
            # Default to last 3 years
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
            
            # Fetch SPY
            try:
                spy = yf.download(MARKET_PROXY_TICKER, start=start_date, end=end_date, progress=False)
                if not spy.empty:
                    self._spy_data = spy
                    logger.debug(f"Fetched {len(spy)} days of SPY data")
            except Exception as e:
                logger.warning(f"Failed to fetch SPY: {e}")
                self._spy_data = None
            
            # Fetch VIX
            try:
                vix = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False)
                if not vix.empty:
                    self._vix_data = vix
                    logger.debug(f"Fetched {len(vix)} days of VIX data")
            except Exception as e:
                logger.warning(f"Failed to fetch VIX: {e}")
                self._vix_data = None
            
            self._last_fetch = datetime.now()
            
        except ImportError:
            logger.warning("yfinance not available for market data")
            self._spy_data = None
            self._vix_data = None


# Global cache instance
_market_cache = MarketDataCache()


def get_market_volatility(
    end_date: Optional[str] = None,
    lookback_days: int = BETA_LOOKBACK_DAYS,
) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """
    Get SPY volatility time series.
    
    Returns:
        Tuple of (volatility array, date index) or (None, None) if unavailable
    """
    spy_data = _market_cache.get_spy_data()
    if spy_data is None or spy_data.empty:
        return None, None
    
    # Handle multi-level columns from yfinance
    if isinstance(spy_data.columns, pd.MultiIndex):
        close = spy_data['Close'][MARKET_PROXY_TICKER] if MARKET_PROXY_TICKER in spy_data['Close'].columns else spy_data['Close'].iloc[:, 0]
    else:
        close = spy_data['Close']
    
    # Compute returns and volatility
    returns = np.log(close / close.shift(1)).dropna()
    vol = returns.ewm(span=21, adjust=False).std()
    
    return vol.values, vol.index


def get_current_vix() -> Optional[float]:
    """Get current (most recent) VIX value."""
    vix_data = _market_cache.get_vix_data()
    if vix_data is None or vix_data.empty:
        return None
    
    # Handle multi-level columns
    if isinstance(vix_data.columns, pd.MultiIndex):
        close = vix_data['Close'][VIX_TICKER] if VIX_TICKER in vix_data['Close'].columns else vix_data['Close'].iloc[:, 0]
    else:
        close = vix_data['Close']
    
    return float(close.iloc[-1])


def get_vix_series() -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """Get VIX time series."""
    vix_data = _market_cache.get_vix_data()
    if vix_data is None or vix_data.empty:
        return None, None
    
    # Handle multi-level columns
    if isinstance(vix_data.columns, pd.MultiIndex):
        close = vix_data['Close'][VIX_TICKER] if VIX_TICKER in vix_data['Close'].columns else vix_data['Close'].iloc[:, 0]
    else:
        close = vix_data['Close']
    
    return close.values, close.index


# =============================================================================
# BETA COMPUTATION
# =============================================================================

def compute_rolling_beta(
    asset_returns: np.ndarray,
    asset_dates: pd.DatetimeIndex,
    lookback: int = BETA_LOOKBACK_DAYS,
) -> Tuple[float, Dict]:
    """
    Compute beta of asset vs SPY market.
    
    β = Cov(r_asset, r_market) / Var(r_market)
    
    Args:
        asset_returns: Log returns of the asset
        asset_dates: Dates corresponding to returns
        lookback: Number of days for beta computation
        
    Returns:
        Tuple of (beta value, diagnostics dict)
    """
    spy_data = _market_cache.get_spy_data()
    if spy_data is None or spy_data.empty:
        return 1.0, {"estimation_method": "default", "error": "no_spy_data"}
    
    # Handle multi-level columns
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_close = spy_data['Close'][MARKET_PROXY_TICKER] if MARKET_PROXY_TICKER in spy_data['Close'].columns else spy_data['Close'].iloc[:, 0]
    else:
        spy_close = spy_data['Close']
    
    spy_returns = np.log(spy_close / spy_close.shift(1)).dropna()
    
    # Align dates
    common_dates = asset_dates.intersection(spy_returns.index)
    if len(common_dates) < BETA_MIN_OBS:
        return 1.0, {"estimation_method": "default", "error": "insufficient_overlap"}
    
    # Get aligned returns
    asset_df = pd.Series(asset_returns, index=asset_dates)
    asset_aligned = asset_df.loc[common_dates].values[-lookback:]
    spy_aligned = spy_returns.loc[common_dates].values[-lookback:]
    
    # Remove NaNs
    valid = np.isfinite(asset_aligned) & np.isfinite(spy_aligned)
    if np.sum(valid) < BETA_MIN_OBS:
        return 1.0, {"estimation_method": "default", "error": "insufficient_valid"}
    
    asset_clean = asset_aligned[valid]
    spy_clean = spy_aligned[valid]
    
    # Compute beta
    cov = np.cov(asset_clean, spy_clean)[0, 1]
    var_spy = np.var(spy_clean, ddof=1)
    
    if var_spy < 1e-12:
        return 1.0, {"estimation_method": "default", "error": "zero_variance"}
    
    beta = cov / var_spy
    
    # Clip to reasonable range
    beta = np.clip(beta, -3.0, 3.0)
    
    diagnostics = {
        "estimation_method": "ols",
        "n_obs": int(np.sum(valid)),
        "correlation": float(np.corrcoef(asset_clean, spy_clean)[0, 1]),
        "r_squared": float(np.corrcoef(asset_clean, spy_clean)[0, 1] ** 2),
    }
    
    return float(beta), diagnostics


# =============================================================================
# VIX-CONDITIONAL ν ADJUSTMENT
# =============================================================================

def compute_vix_nu_adjustment(
    nu_base: float,
    vix_value: Optional[float] = None,
    kappa: float = VIX_KAPPA_DEFAULT,
    vix_median: float = VIX_MEDIAN_DEFAULT,
    vix_iqr: float = VIX_IQR_DEFAULT,
) -> VIXConditioningResult:
    """
    Compute VIX-conditional tail thickness adjustment.
    
    Formula:
        ν_t = max(ν_min, ν_base - κ × VIX_normalized)
        
    Where:
        VIX_normalized = (VIX - median) / IQR
        
    Higher VIX → lower ν → heavier tails
    
    Args:
        nu_base: Base degrees of freedom from model
        vix_value: Current VIX (fetched if None)
        kappa: Sensitivity parameter
        vix_median: Historical VIX median
        vix_iqr: Historical VIX IQR
        
    Returns:
        VIXConditioningResult with adjusted ν
    """
    if vix_value is None:
        vix_value = get_current_vix()
    
    if vix_value is None:
        # No VIX data - return unadjusted
        return VIXConditioningResult(
            nu_original=nu_base,
            nu_adjusted=nu_base,
            vix_value=float('nan'),
            vix_normalized=0.0,
            kappa=kappa,
            adjustment_applied=False,
        )
    
    # Normalize VIX
    vix_normalized = (vix_value - vix_median) / vix_iqr
    
    # Compute adjustment
    nu_reduction = kappa * vix_normalized * nu_base
    nu_adjusted = nu_base - nu_reduction
    
    # Floor to minimum
    nu_adjusted = max(NU_MIN_FLOOR, nu_adjusted)
    
    return VIXConditioningResult(
        nu_original=nu_base,
        nu_adjusted=nu_adjusted,
        vix_value=vix_value,
        vix_normalized=vix_normalized,
        kappa=kappa,
        adjustment_applied=True,
    )


# =============================================================================
# COMPOSITE VOLATILITY (CROSS-SECTIONAL COUPLING)
# =============================================================================

def compute_composite_volatility(
    vol_asset: np.ndarray,
    asset_dates: pd.DatetimeIndex,
    asset_returns: Optional[np.ndarray] = None,
    beta: Optional[float] = None,
    coupling_strength: float = MARKET_VOL_COUPLING_DEFAULT,
) -> MarketConditioningResult:
    """
    Compute composite volatility with market beta coupling.
    
    Formula:
        σ²_composite = σ²_asset + (coupling × β²) × σ²_market
        
    High-beta assets get more market volatility contribution.
    
    Args:
        vol_asset: Asset-specific volatility
        asset_dates: Dates for the asset data
        asset_returns: Asset returns (for beta computation if not provided)
        beta: Pre-computed beta (computed if None)
        coupling_strength: How much market vol to include (0-1)
        
    Returns:
        MarketConditioningResult with composite volatility
    """
    n = len(vol_asset)
    
    # Get market volatility
    vol_market, market_dates = get_market_volatility()
    
    if vol_market is None or market_dates is None:
        # No market data - return asset vol unchanged
        return MarketConditioningResult(
            vol_composite=vol_asset,
            vol_asset=vol_asset,
            vol_market=np.full(n, np.nan),
            beta=1.0,
            beta_squared=1.0,
            market_data_available=False,
            estimation_method="asset_only",
        )
    
    # Compute or use provided beta
    if beta is None and asset_returns is not None:
        beta, beta_diag = compute_rolling_beta(asset_returns, asset_dates)
    elif beta is None:
        beta = 1.0
    
    beta_squared = beta ** 2
    
    # Align market vol with asset dates
    vol_market_series = pd.Series(vol_market, index=market_dates)
    vol_market_aligned = vol_market_series.reindex(asset_dates, method='ffill').values
    
    # Handle NaNs
    vol_market_aligned = np.nan_to_num(vol_market_aligned, nan=np.nanmean(vol_asset))
    
    # Compute composite volatility
    # σ²_composite = σ²_asset + (coupling × β²) × σ²_market
    var_asset = vol_asset ** 2
    var_market = vol_market_aligned ** 2
    var_composite = var_asset + coupling_strength * beta_squared * var_market
    vol_composite = np.sqrt(var_composite)
    
    # Get current VIX for diagnostics
    vix_current = get_current_vix()
    
    return MarketConditioningResult(
        vol_composite=vol_composite,
        vol_asset=vol_asset,
        vol_market=vol_market_aligned,
        beta=beta,
        beta_squared=beta_squared,
        vix_current=vix_current,
        market_data_available=True,
        estimation_method="full",
    )


# =============================================================================
# HIGH-LEVEL API FOR MODEL INTEGRATION
# =============================================================================

def condition_model_on_market(
    vol_asset: np.ndarray,
    asset_returns: np.ndarray,
    asset_dates: pd.DatetimeIndex,
    nu_base: float,
    use_beta_coupling: bool = True,
    use_vix_nu: bool = True,
    coupling_strength: float = MARKET_VOL_COUPLING_DEFAULT,
    vix_kappa: float = VIX_KAPPA_DEFAULT,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Apply full market conditioning to asset model.
    
    This is the main entry point for integrating market awareness
    into Student-t models.
    
    Args:
        vol_asset: Asset-specific volatility series
        asset_returns: Asset log returns
        asset_dates: Dates for asset data
        nu_base: Base degrees of freedom
        use_beta_coupling: Whether to add market vol component
        use_vix_nu: Whether to adjust ν based on VIX
        coupling_strength: Market vol contribution (0-1)
        vix_kappa: VIX sensitivity for ν adjustment
        
    Returns:
        Tuple of (conditioned_vol, adjusted_nu, diagnostics_dict)
    """
    diagnostics = {}
    
    # Step 1: Compute composite volatility with beta coupling
    if use_beta_coupling:
        vol_result = compute_composite_volatility(
            vol_asset=vol_asset,
            asset_dates=asset_dates,
            asset_returns=asset_returns,
            coupling_strength=coupling_strength,
        )
        vol_conditioned = vol_result.vol_composite
        diagnostics["beta"] = vol_result.beta
        diagnostics["beta_squared"] = vol_result.beta_squared
        diagnostics["market_data_available"] = vol_result.market_data_available
    else:
        vol_conditioned = vol_asset
        diagnostics["beta_coupling"] = "disabled"
    
    # Step 2: Adjust ν based on VIX
    if use_vix_nu:
        vix_result = compute_vix_nu_adjustment(
            nu_base=nu_base,
            kappa=vix_kappa,
        )
        nu_adjusted = vix_result.nu_adjusted
        diagnostics["vix_value"] = vix_result.vix_value
        diagnostics["vix_normalized"] = vix_result.vix_normalized
        diagnostics["nu_original"] = vix_result.nu_original
        diagnostics["nu_adjusted"] = vix_result.nu_adjusted
        diagnostics["vix_adjustment_applied"] = vix_result.adjustment_applied
    else:
        nu_adjusted = nu_base
        diagnostics["vix_conditioning"] = "disabled"
    
    return vol_conditioned, nu_adjusted, diagnostics


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MarketConditioningResult",
    "VIXConditioningResult",
    "MarketDataCache",
    "get_market_volatility",
    "get_current_vix",
    "get_vix_series",
    "compute_rolling_beta",
    "compute_vix_nu_adjustment",
    "compute_composite_volatility",
    "condition_model_on_market",
    # Constants
    "MARKET_PROXY_TICKER",
    "VIX_TICKER",
    "VIX_KAPPA_DEFAULT",
    "MARKET_VOL_COUPLING_DEFAULT",
]
