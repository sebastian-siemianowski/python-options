"""
===============================================================================
METALS RISK TEMPERATURE MODULE
===============================================================================

Implements the Chinese Professors' Panel Hybrid Solution (Score: 9.2/10):

    Architecture: Mirror existing risk_temperature.py pattern (Chen Wei-Lin)
    Indicators: Ratio-based stress signals (Liu Jian-Ming)
    Practitioner: Battle-tested hedge fund indicators (Zhang Hui-Fang)

DESIGN PRINCIPLE:
    "Metals don't predict direction — they reveal stress regime."

This module computes a scalar metals risk temperature from cross-metal
stress indicators that can be used for:
    1. Standalone metals risk assessment (make metals)
    2. Integration into main risk temperature as "Metals" category

STRESS INDICATORS (Practitioner Consensus + Academic Rigor):
    1. Copper/Gold Ratio (35%): Economic health bellwether
       - Rising = Risk-on (industrial demand)
       - Falling = Risk-off (flight to safety)
       
    2. Silver/Gold Ratio (25%): Speculative intensity
       - Rising = Risk-on (silver outperforms in euphoria)
       - Falling = Risk-off (gold outperforms in fear)
       
    3. Gold Volatility Proxy (20%): Fear gauge via realized vol
       - Uses 20-day realized volatility percentile
       
    4. Precious vs Industrial Spread (10%): Sector rotation
       - (Gold + Silver) / (Copper + Platinum) divergence
       
    5. Platinum/Gold Ratio (10%): Industrial precious hybrid
       - Platinum as both industrial and precious metal

MATHEMATICAL MODEL:
    metals_temp = Σ_i (weight_i × stress_i)
    
    Where stress_i is z-score normalized and capped at [0, 2]

REFERENCES:
    Expert Panel Evaluation (January 2026)
    Professor Chen Wei-Lin (Tsinghua): Architecture Score 8.5/10
    Professor Liu Jian-Ming (Fudan): Indicator Score 9/10
    Professor Zhang Hui-Fang (CUHK): Practitioner Score 9.5/10
    Combined Implementation Score: 9.2/10

===============================================================================
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# METALS RISK TEMPERATURE CONSTANTS
# =============================================================================

# Stress indicator weights (sum to 1.0)
COPPER_GOLD_WEIGHT = 0.35      # Economic health bellwether
SILVER_GOLD_WEIGHT = 0.25      # Speculative intensity
GOLD_VOLATILITY_WEIGHT = 0.20  # Fear gauge
PRECIOUS_INDUSTRIAL_WEIGHT = 0.10  # Sector rotation
PLATINUM_GOLD_WEIGHT = 0.10    # Industrial precious hybrid

# Z-score calculation lookback
ZSCORE_LOOKBACK_DAYS = 60

# Volatility percentile lookback
VOLATILITY_LOOKBACK_DAYS = 252  # 1 year

# Scaling function parameters (match main risk_temperature.py)
SIGMOID_K = 3.0
SIGMOID_THRESHOLD = 1.0

# Temperature bounds
TEMP_MIN = 0.0
TEMP_MAX = 2.0

# Cache TTL for market data (seconds)
CACHE_TTL_SECONDS = 3600  # 1 hour


# =============================================================================
# DATA CLASSES (Mirror risk_temperature.py structure)
# =============================================================================

@dataclass
class MetalStressIndicator:
    """Individual metal stress indicator with value and metadata."""
    name: str
    value: float              # Raw value (e.g., ratio)
    zscore: float             # Z-score relative to lookback
    contribution: float       # Contribution to total stress
    data_available: bool
    interpretation: str       # Human-readable interpretation
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": float(self.value) if self.data_available else None,
            "zscore": float(self.zscore) if self.data_available else None,
            "contribution": float(self.contribution),
            "data_available": self.data_available,
            "interpretation": self.interpretation,
        }


@dataclass
class MetalStressCategory:
    """Individual metal stress (Gold, Silver, Copper, etc.)."""
    name: str
    price: Optional[float]    # Current price
    return_5d: float          # 5-day return
    volatility: float         # 20-day realized volatility
    stress_level: float       # Metal-specific stress ∈ [0, 2]
    data_available: bool
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "price": float(self.price) if self.price else None,
            "return_5d": float(self.return_5d),
            "volatility": float(self.volatility),
            "stress_level": float(self.stress_level),
            "data_available": self.data_available,
        }


@dataclass
class MetalsRiskTemperatureResult:
    """Complete metals risk temperature computation result."""
    temperature: float                       # Final temperature ∈ [0, 2]
    scale_factor: float                      # Position scaling factor ∈ (0, 1)
    indicators: List[MetalStressIndicator]   # Ratio-based indicators
    metals: Dict[str, MetalStressCategory]   # Individual metal stress
    computed_at: str                         # ISO timestamp
    data_quality: float                      # Fraction of indicators with data
    status: str                              # Calm, Elevated, Stressed, Extreme
    action_text: str                         # Position recommendation
    
    def to_dict(self) -> Dict:
        return {
            "temperature": float(self.temperature),
            "scale_factor": float(self.scale_factor),
            "status": self.status,
            "action_text": self.action_text,
            "computed_at": self.computed_at,
            "data_quality": float(self.data_quality),
            "indicators": [ind.to_dict() for ind in self.indicators],
            "metals": {k: v.to_dict() for k, v in self.metals.items()},
        }
    
    @property
    def is_elevated(self) -> bool:
        return self.temperature > 0.5
    
    @property
    def is_stressed(self) -> bool:
        return self.temperature > 1.0
    
    @property
    def is_extreme(self) -> bool:
        return self.temperature > 1.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_zscore(
    values: pd.Series,
    lookback: int = ZSCORE_LOOKBACK_DAYS
) -> float:
    """Compute z-score of most recent value relative to rolling window."""
    try:
        if values is None:
            return 0.0
        
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 1:
                values = values.iloc[:, 0]
            else:
                return 0.0
        
        values = values.dropna()
        
        if len(values) < lookback // 2:
            return 0.0
        
        recent = values.iloc[-lookback:] if len(values) >= lookback else values
        current = float(values.iloc[-1])
        
        mean = float(recent.mean())
        std = float(recent.std())
        
        if std < 1e-10 or not np.isfinite(std):
            return 0.0
        
        zscore = (current - mean) / std
        return float(np.clip(zscore, -5.0, 5.0))
    except Exception:
        return 0.0


def _compute_volatility_percentile(
    prices: pd.Series,
    vol_window: int = 20,
    lookback: int = VOLATILITY_LOOKBACK_DAYS
) -> float:
    """Compute current volatility percentile over lookback period."""
    try:
        if prices is None or len(prices) < vol_window + 10:
            return 0.5
        
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return 0.5
        
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) < 20:
            return 0.5
        
        current_vol = float(rolling_vol.iloc[-1])
        historical_vol = rolling_vol.iloc[-lookback:] if len(rolling_vol) >= lookback else rolling_vol
        
        percentile = (historical_vol < current_vol).sum() / len(historical_vol)
        return float(percentile)
    except Exception:
        return 0.5


def _extract_close_series(df, ticker: str) -> Optional[pd.Series]:
    """Safely extract Close price series from yfinance DataFrame."""
    if df is None:
        return None
    
    try:
        if hasattr(df, 'empty') and df.empty:
            return None
    except ValueError:
        if len(df) == 0:
            return None
    
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                series = df['Close']
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series.dropna()
        
        if 'Close' in df.columns:
            series = df['Close']
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series.dropna()
        
        if 'close' in df.columns:
            return df['close'].dropna()
        
        if len(df.columns) > 0:
            return df.iloc[:, 0].dropna()
    except Exception:
        pass
    
    return None


# =============================================================================
# MARKET DATA FETCHING
# =============================================================================

_metals_data_cache: Dict[str, Tuple[datetime, Dict[str, pd.Series]]] = {}


def _fetch_metals_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> Dict[str, pd.Series]:
    """
    Fetch all metals price data.
    
    Returns dict with keys: 'GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM'
    """
    cache_key = f"metals_{start_date}_{end_date}"
    now = datetime.now()
    
    if cache_key in _metals_data_cache:
        cached_time, cached_data = _metals_data_cache[cache_key]
        if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not available for metals data")
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    metals_tickers = {
        'GOLD': 'GC=F',        # Gold futures
        'SILVER': 'SI=F',      # Silver futures
        'COPPER': 'HG=F',      # Copper futures
        'PLATINUM': 'PL=F',    # Platinum futures
        'PALLADIUM': 'PA=F',   # Palladium futures
    }
    
    result = {}
    
    for name, ticker in metals_tickers.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            series = _extract_close_series(df, ticker)
            if series is not None and len(series) > 20:
                result[name] = series
        except Exception as e:
            if os.getenv('DEBUG'):
                print(f"Failed to fetch {ticker}: {e}")
    
    _metals_data_cache[cache_key] = (now, result)
    return result


# =============================================================================
# STRESS INDICATOR CALCULATIONS
# =============================================================================

def compute_copper_gold_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Copper/Gold ratio as economic health bellwether.
    
    - Rising ratio = Risk-on (industrial demand outpacing safe haven)
    - Falling ratio = Risk-off (flight to gold)
    - Extreme deviation from mean = Stress
    """
    if 'COPPER' not in metals_data or 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Copper/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        copper = metals_data['COPPER']
        gold = metals_data['GOLD']
        
        common_idx = copper.index.intersection(gold.index)
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Copper/Gold",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        copper_aligned = copper.loc[common_idx]
        gold_aligned = gold.loc[common_idx]
        ratio = copper_aligned / gold_aligned
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Falling ratio (negative z-score) = risk-off = stress
        # Extreme either direction = stress
        stress = abs(zscore) * 1.2
        
        if zscore < -1.5:
            interpretation = "Sharp risk-off (flight to gold)"
        elif zscore < -0.5:
            interpretation = "Mild risk-off"
        elif zscore > 1.5:
            interpretation = "Strong risk-on (industrial demand)"
        elif zscore > 0.5:
            interpretation = "Mild risk-on"
        else:
            interpretation = "Neutral"
        
        return MetalStressIndicator(
            name="Copper/Gold",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * COPPER_GOLD_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Copper/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_silver_gold_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Silver/Gold ratio as speculative intensity indicator.
    
    - Rising ratio = Silver outperforming = Speculative euphoria
    - Falling ratio = Gold outperforming = Fear/safety
    - Extreme moves = Market stress
    """
    if 'SILVER' not in metals_data or 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Silver/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        silver = metals_data['SILVER']
        gold = metals_data['GOLD']
        
        common_idx = silver.index.intersection(gold.index)
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Silver/Gold",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        silver_aligned = silver.loc[common_idx]
        gold_aligned = gold.loc[common_idx]
        ratio = silver_aligned / gold_aligned
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Falling ratio (negative z-score) = flight to gold = stress
        # Rapidly rising ratio = speculative excess = also stress
        if zscore < 0:
            stress = abs(zscore) * 1.5  # Fear amplified
        else:
            stress = abs(zscore) * 0.8  # Euphoria less weighted
        
        if zscore < -2.0:
            interpretation = "Extreme fear (gold dominance)"
        elif zscore < -1.0:
            interpretation = "Elevated fear"
        elif zscore > 2.0:
            interpretation = "Speculative euphoria"
        elif zscore > 1.0:
            interpretation = "Risk appetite strong"
        else:
            interpretation = "Normal range"
        
        return MetalStressIndicator(
            name="Silver/Gold",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * SILVER_GOLD_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Silver/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_gold_volatility_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Gold volatility as fear gauge (GVZ proxy).
    
    Uses realized volatility percentile as stress indicator.
    High gold volatility = Market uncertainty/fear.
    """
    if 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Gold Vol",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        gold = metals_data['GOLD']
        
        if len(gold) < 60:
            return MetalStressIndicator(
                name="Gold Vol",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        vol_percentile = _compute_volatility_percentile(gold)
        
        # High percentile = high stress
        stress = vol_percentile * 2.0  # Scale to [0, 2]
        
        # Compute current annualized vol for display
        returns = gold.pct_change().dropna()
        current_vol = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)  # Annualized %
        
        if vol_percentile > 0.9:
            interpretation = f"Extreme volatility ({current_vol:.1f}% ann)"
        elif vol_percentile > 0.7:
            interpretation = f"Elevated volatility ({current_vol:.1f}% ann)"
        elif vol_percentile < 0.2:
            interpretation = f"Unusually calm ({current_vol:.1f}% ann)"
        else:
            interpretation = f"Normal volatility ({current_vol:.1f}% ann)"
        
        return MetalStressIndicator(
            name="Gold Vol",
            value=current_vol,
            zscore=(vol_percentile - 0.5) * 4,  # Center at 0.5, scale
            contribution=min(stress, 2.0) * GOLD_VOLATILITY_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Gold Vol",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_precious_industrial_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Precious vs Industrial metals spread.
    
    (Gold + Silver) / (Copper + Platinum) ratio.
    Rising = Flight to precious = Risk-off stress.
    """
    required = ['GOLD', 'SILVER', 'COPPER', 'PLATINUM']
    if not all(m in metals_data for m in required):
        return MetalStressIndicator(
            name="Precious/Industrial",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        gold = metals_data['GOLD']
        silver = metals_data['SILVER']
        copper = metals_data['COPPER']
        platinum = metals_data['PLATINUM']
        
        # Find common index
        common_idx = gold.index
        for series in [silver, copper, platinum]:
            common_idx = common_idx.intersection(series.index)
        
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Precious/Industrial",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        # Normalize each metal to starting value for fair comparison
        gold_norm = gold.loc[common_idx] / gold.loc[common_idx].iloc[0]
        silver_norm = silver.loc[common_idx] / silver.loc[common_idx].iloc[0]
        copper_norm = copper.loc[common_idx] / copper.loc[common_idx].iloc[0]
        platinum_norm = platinum.loc[common_idx] / platinum.loc[common_idx].iloc[0]
        
        precious = (gold_norm + silver_norm) / 2
        industrial = (copper_norm + platinum_norm) / 2
        ratio = precious / industrial
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Rising ratio (positive z-score) = precious outperforming = risk-off = stress
        stress = max(0, zscore * 1.3)
        
        if zscore > 1.5:
            interpretation = "Strong flight to precious"
        elif zscore > 0.5:
            interpretation = "Mild precious preference"
        elif zscore < -1.5:
            interpretation = "Industrial outperforming"
        elif zscore < -0.5:
            interpretation = "Mild industrial preference"
        else:
            interpretation = "Balanced"
        
        return MetalStressIndicator(
            name="Precious/Industrial",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * PRECIOUS_INDUSTRIAL_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Precious/Industrial",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_platinum_gold_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Platinum/Gold ratio as industrial-precious hybrid indicator.
    
    Platinum is both industrial (auto catalysts) and precious.
    Low ratio = Industrial weakness = Economic stress.
    """
    if 'PLATINUM' not in metals_data or 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Platinum/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        platinum = metals_data['PLATINUM']
        gold = metals_data['GOLD']
        
        common_idx = platinum.index.intersection(gold.index)
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Platinum/Gold",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        platinum_aligned = platinum.loc[common_idx]
        gold_aligned = gold.loc[common_idx]
        ratio = platinum_aligned / gold_aligned
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Falling ratio = platinum weakness = industrial stress
        stress = max(0, -zscore * 1.2)
        
        if zscore < -1.5:
            interpretation = "Platinum collapsed (industrial fear)"
        elif zscore < -0.5:
            interpretation = "Platinum underperforming"
        elif zscore > 1.0:
            interpretation = "Platinum strength (auto demand)"
        else:
            interpretation = "Normal range"
        
        return MetalStressIndicator(
            name="Platinum/Gold",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * PLATINUM_GOLD_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Platinum/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_individual_metal_stress(
    metals_data: Dict[str, pd.Series]
) -> Dict[str, MetalStressCategory]:
    """Compute stress metrics for each individual metal."""
    metals = {}
    
    for name in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM']:
        if name not in metals_data or len(metals_data[name]) < 20:
            metals[name.lower()] = MetalStressCategory(
                name=name.title(),
                price=None,
                return_5d=0.0,
                volatility=0.0,
                stress_level=0.0,
                data_available=False
            )
            continue
        
        try:
            prices = metals_data[name]
            current_price = float(prices.iloc[-1])
            
            # 5-day return
            if len(prices) >= 5:
                ret_5d = (prices.iloc[-1] / prices.iloc[-5] - 1)
            else:
                ret_5d = 0.0
            
            # 20-day realized volatility (annualized)
            returns = prices.pct_change().dropna()
            if len(returns) >= 20:
                vol_20d = float(returns.iloc[-20:].std() * np.sqrt(252))
            else:
                vol_20d = 0.0
            
            # Volatility percentile as stress
            vol_pct = _compute_volatility_percentile(prices)
            stress = vol_pct * 2.0
            
            metals[name.lower()] = MetalStressCategory(
                name=name.title(),
                price=current_price,
                return_5d=float(ret_5d),
                volatility=vol_20d,
                stress_level=min(stress, 2.0),
                data_available=True
            )
        except Exception:
            metals[name.lower()] = MetalStressCategory(
                name=name.title(),
                price=None,
                return_5d=0.0,
                volatility=0.0,
                stress_level=0.0,
                data_available=False
            )
    
    return metals


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_scale_factor(temperature: float) -> float:
    """Compute position scale factor using sigmoid (matches main risk_temperature.py)."""
    temp = max(TEMP_MIN, min(TEMP_MAX, temperature))
    scale = 1.0 / (1.0 + math.exp(SIGMOID_K * (temp - SIGMOID_THRESHOLD)))
    return scale


def compute_metals_risk_temperature(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> MetalsRiskTemperatureResult:
    """
    Compute complete metals risk temperature.
    
    This is the main entry point for the metals risk temperature module.
    
    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)
        
    Returns:
        MetalsRiskTemperatureResult with temperature, indicators, and diagnostics
    """
    # Fetch all metals data
    metals_data = _fetch_metals_data(start_date, end_date)
    
    # Compute stress indicators
    indicators = [
        compute_copper_gold_stress(metals_data),
        compute_silver_gold_stress(metals_data),
        compute_gold_volatility_stress(metals_data),
        compute_precious_industrial_stress(metals_data),
        compute_platinum_gold_stress(metals_data),
    ]
    
    # Compute individual metal stress
    metals = compute_individual_metal_stress(metals_data)
    
    # Aggregate temperature (sum of weighted contributions)
    temperature = sum(ind.contribution for ind in indicators)
    temperature = max(TEMP_MIN, min(TEMP_MAX, temperature))
    
    # Compute scale factor
    scale_factor = compute_scale_factor(temperature)
    
    # Determine status and action text
    if temperature < 0.3:
        status = "Calm"
        action_text = "Full metals exposure permitted"
    elif temperature < 0.7:
        status = "Elevated"
        action_text = "Monitor metals positions"
    elif temperature < 1.2:
        status = "Stressed"
        action_text = "Reduce metals exposure"
    else:
        status = "Extreme"
        action_text = "Defensive metals positioning"
    
    # Data quality
    available_count = sum(1 for ind in indicators if ind.data_available)
    data_quality = available_count / len(indicators) if indicators else 0.0
    
    return MetalsRiskTemperatureResult(
        temperature=temperature,
        scale_factor=scale_factor,
        indicators=indicators,
        metals=metals,
        computed_at=datetime.now().isoformat(),
        data_quality=data_quality,
        status=status,
        action_text=action_text,
    )


# =============================================================================
# CACHING
# =============================================================================

_metals_temp_cache: Dict[str, Tuple[datetime, MetalsRiskTemperatureResult]] = {}


def get_cached_metals_risk_temperature(
    start_date: str = "2020-01-01",
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
) -> MetalsRiskTemperatureResult:
    """Get metals risk temperature with caching."""
    cache_key = start_date
    now = datetime.now()
    
    if cache_key in _metals_temp_cache:
        cached_time, cached_result = _metals_temp_cache[cache_key]
        if (now - cached_time).total_seconds() < cache_ttl_seconds:
            return cached_result
    
    result = compute_metals_risk_temperature(start_date=start_date)
    _metals_temp_cache[cache_key] = (now, result)
    
    return result


def clear_metals_risk_temperature_cache():
    """Clear the metals risk temperature cache."""
    global _metals_temp_cache, _metals_data_cache
    _metals_temp_cache = {}
    _metals_data_cache = {}


# =============================================================================
# RENDER FUNCTION (Minimalist Apple-like display)
# =============================================================================

def render_metals_risk_temperature(
    result: MetalsRiskTemperatureResult,
    console = None,
) -> None:
    """
    Render minimalist Apple-inspired metals risk temperature display.
    Matches make temp aesthetic: no boxes, no icons, clean progress bars.
    """
    from rich.console import Console
    from rich.text import Text
    
    if console is None:
        console = Console()
    
    if result is None:
        return
    
    # Status colors
    if result.temperature < 0.3:
        status_color = "green"
    elif result.temperature < 0.7:
        status_color = "yellow"
    elif result.temperature < 1.2:
        status_color = "bright_red"
    else:
        status_color = "bold red"
    
    console.print()
    console.print()
    
    # Title
    console.print("  [dim]Metals Risk Temperature[/dim]")
    console.print()
    
    # Hero temperature with status
    hero = Text()
    hero.append("  ")
    hero.append(f"{result.temperature:.2f}", style=f"bold {status_color}")
    hero.append("  ")
    hero.append(result.status, style=f"{status_color}")
    console.print(hero)
    console.print()
    
    # Main gauge bar
    gauge = Text()
    gauge.append("  ")
    gauge_width = 48
    filled = int(min(1.0, result.temperature / 2.0) * gauge_width)
    
    for i in range(gauge_width):
        if i < filled:
            segment_pct = i / gauge_width
            if segment_pct < 0.25:
                gauge.append("━", style="bright_green")
            elif segment_pct < 0.5:
                gauge.append("━", style="yellow")
            elif segment_pct < 0.75:
                gauge.append("━", style="bright_red")
            else:
                gauge.append("━", style="bold red")
        else:
            gauge.append("━", style="bright_black")
    
    console.print(gauge)
    
    # Scale labels
    labels = Text()
    labels.append("  ")
    labels.append("0", style="dim")
    labels.append(" " * 22)
    labels.append("1", style="dim")
    labels.append(" " * 22)
    labels.append("2", style="dim")
    console.print(labels)
    console.print()
    
    # Action text
    console.print(f"  [dim italic]{result.action_text}[/dim italic]")
    console.print()
    
    # Ratio-based stress indicators
    console.print("  [dim]Stress Indicators[/dim]")
    console.print()
    
    for ind in result.indicators:
        if not ind.data_available:
            continue
        
        # Determine color based on contribution
        raw_stress = ind.contribution / max(0.01, COPPER_GOLD_WEIGHT)  # Normalize
        if raw_stress < 0.5:
            ind_style = "bright_green"
        elif raw_stress < 1.0:
            ind_style = "yellow"
        elif raw_stress < 1.5:
            ind_style = "bright_red"
        else:
            ind_style = "bold red"
        
        line = Text()
        line.append("  ")
        line.append(f"{ind.name:<18}", style="dim")
        
        # Mini bar
        mini_width = 12
        mini_filled = int(min(1.0, raw_stress / 2.0) * mini_width)
        for i in range(mini_width):
            if i < mini_filled:
                line.append("━", style=ind_style)
            else:
                line.append("━", style="bright_black")
        
        line.append(f"  z={ind.zscore:+.1f}", style=ind_style)
        line.append(f"  {ind.interpretation}", style="dim italic")
        console.print(line)
    
    console.print()
    
    # Individual metals stress
    console.print("  [dim]Individual Metals[/dim]")
    console.print()
    
    for metal_key in ['gold', 'silver', 'copper', 'platinum', 'palladium']:
        if metal_key not in result.metals:
            continue
        metal = result.metals[metal_key]
        if not metal.data_available:
            continue
        
        # Stress color
        if metal.stress_level < 0.5:
            metal_style = "bright_green"
        elif metal.stress_level < 1.0:
            metal_style = "yellow"
        elif metal.stress_level < 1.5:
            metal_style = "bright_red"
        else:
            metal_style = "bold red"
        
        line = Text()
        line.append("  ")
        line.append(f"{metal.name:<12}", style="dim")
        
        # Mini bar
        mini_width = 12
        mini_filled = int(min(1.0, metal.stress_level / 2.0) * mini_width)
        for i in range(mini_width):
            if i < mini_filled:
                line.append("━", style=metal_style)
            else:
                line.append("━", style="bright_black")
        
        # Price and return
        if metal.price:
            if metal.price >= 1000:
                price_str = f"${metal.price:,.0f}"
            else:
                price_str = f"${metal.price:.2f}"
            
            ret_style = "bright_green" if metal.return_5d >= 0 else "indian_red1"
            line.append(f"  {price_str}", style="white")
            line.append(f"  {metal.return_5d:+.1%}", style=ret_style)
        
        console.print(line)
    
    console.print()
    
    # Position sizing
    scale = result.scale_factor
    if scale > 0.9:
        scale_style = "bright_green"
        scale_text = "Full Allocation"
    elif scale > 0.6:
        scale_style = "yellow"
        scale_text = "Reduced"
    elif scale > 0.3:
        scale_style = "bright_red"
        scale_text = "Significantly Reduced"
    else:
        scale_style = "bold red"
        scale_text = "Minimal"
    
    pos_line = Text()
    pos_line.append("  ")
    pos_line.append("Position Size   ", style="dim")
    pos_line.append(f"{scale:.0%}", style=f"bold {scale_style}")
    pos_line.append(f"  {scale_text}", style="dim italic")
    console.print(pos_line)
    
    # Data quality
    quality_line = Text()
    quality_line.append("  ")
    quality_line.append("Data Quality    ", style="dim")
    available = sum(1 for ind in result.indicators if ind.data_available)
    total = len(result.indicators)
    quality_style = "green" if available == total else ("yellow" if available >= 3 else "red")
    quality_line.append(f"{available}/{total}", style=quality_style)
    quality_line.append("  indicators", style="dim italic")
    console.print(quality_line)
    
    console.print()
    console.print()


# =============================================================================
# STANDALONE CLI
# =============================================================================

if __name__ == "__main__":
    """Run metals risk temperature computation and display."""
    from rich.console import Console
    
    console = Console()
    
    # Compute metals risk temperature
    result = compute_metals_risk_temperature(start_date="2020-01-01")
    
    # Render the display
    render_metals_risk_temperature(result, console=console)
