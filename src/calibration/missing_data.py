"""
Epic 29: Missing Data, Halts, and Market Closures

Handles data gaps in the Kalman filter pipeline:
1. Gap-aware prediction step that advances state by k timesteps
2. Holiday calendar for NYSE, NASDAQ, and crypto markets
3. Graceful degradation on extreme missing data

References:
- Harvey (1989): Forecasting, Structural Time Series Models (irregular spacing)
- Durbin & Koopman (2012): Time Series Analysis by State Space Methods
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import date, timedelta

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 29.1: Gap-aware prediction
MAX_GAP_DAYS = 365  # Cap gap at 1 year

# Story 29.3: Data quality thresholds
QUALITY_NORMAL = 0.95           # > 95%: normal operation
QUALITY_REDUCED = 0.80          # 80-95%: reduced confidence
QUALITY_LOW = 0.50              # 50-80%: low quality
INTERVAL_MULTIPLIER_REDUCED = 1.3  # Multiply intervals by 1.3 for reduced
INTERVAL_MULTIPLIER_LOW = 2.0      # For low quality


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GapAwarePredictResult:
    """Result of gap-aware Kalman prediction."""
    mu_predicted: float    # Predicted state after k steps
    P_predicted: float     # Predicted covariance after k steps
    gap_days: int          # Number of gap days
    phi_k: float           # phi^k (decay factor)
    P_growth: float        # How much P grew due to gap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu_predicted": float(self.mu_predicted),
            "P_predicted": float(self.P_predicted),
            "gap_days": self.gap_days,
            "phi_k": float(self.phi_k),
            "P_growth": float(self.P_growth),
        }


@dataclass
class DataQualityResult:
    """Result of data quality assessment."""
    quality_score: float     # Fraction of available data [0, 1]
    quality_flag: str        # "NORMAL", "REDUCED_CONFIDENCE", "LOW_QUALITY", "UNUSABLE"
    interval_multiplier: float  # Multiply prediction intervals by this
    suppress_direction: bool    # True if directional signals should be suppressed
    n_available: int
    n_expected: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": float(self.quality_score),
            "quality_flag": self.quality_flag,
            "interval_multiplier": float(self.interval_multiplier),
            "suppress_direction": self.suppress_direction,
            "n_available": self.n_available,
            "n_expected": self.n_expected,
        }


# ---------------------------------------------------------------------------
# Story 29.1: Gap-Aware Kalman Prediction Step
# ---------------------------------------------------------------------------

def gap_aware_predict(
    mu: float,
    P: float,
    phi: float,
    q: float,
    gap_days: int,
) -> GapAwarePredictResult:
    """
    Gap-aware Kalman prediction step.

    Advances state by k timesteps when k days elapse between observations:
      mu_{t+k} = phi^k * mu_t
      P_{t+k} = phi^{2k} * P_t + q * (1 - phi^{2k}) / (1 - phi^2)

    For phi = 1.0 (random walk): P_{t+k} = P_t + k * q.

    Parameters
    ----------
    mu : float
        Current state estimate.
    P : float
        Current state covariance.
    phi : float
        AR(1) coefficient (typically 0.99-1.0).
    q : float
        Process noise variance.
    gap_days : int
        Number of elapsed days (k >= 1).

    Returns
    -------
    GapAwarePredictResult
        Predicted state and covariance after gap.
    """
    mu = float(mu)
    P = float(P)
    phi = float(phi)
    q = float(q)
    gap_days = max(1, min(int(gap_days), MAX_GAP_DAYS))

    k = gap_days
    phi_k = phi ** k

    # State prediction: mu decays toward 0
    mu_predicted = phi_k * mu

    # Covariance prediction
    phi_sq = phi ** 2

    if abs(phi_sq - 1.0) < 1e-12:
        # Random walk case: geometric series simplifies to k * q
        P_growth = k * q
    else:
        # AR(1) case: geometric series sum
        phi_2k = phi_sq ** k
        P_growth = q * (1.0 - phi_2k) / (1.0 - phi_sq)

    P_predicted = (phi_sq ** k) * P + P_growth

    # Ensure P is positive
    P_predicted = max(P_predicted, 1e-15)

    return GapAwarePredictResult(
        mu_predicted=mu_predicted,
        P_predicted=P_predicted,
        gap_days=k,
        phi_k=phi_k,
        P_growth=P_growth,
    )


# ---------------------------------------------------------------------------
# Story 29.2: Holiday Calendar Integration
# ---------------------------------------------------------------------------

def _easter_date(year: int) -> date:
    """Compute Easter Sunday for a given year (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _good_friday(year: int) -> date:
    """Good Friday = Easter Sunday - 2."""
    return _easter_date(year) - timedelta(days=2)


def _observed_holiday(d: date) -> date:
    """
    If holiday falls on Saturday, observed Friday.
    If Sunday, observed Monday.
    """
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    elif d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> Set[date]:
    """
    NYSE holidays for a given year.

    Includes: New Year, MLK, Presidents, Good Friday, Memorial,
    Juneteenth (2022+), July 4, Labor, Thanksgiving, Christmas.
    """
    holidays = set()

    # New Year's Day
    holidays.add(_observed_holiday(date(year, 1, 1)))

    # MLK Day: 3rd Monday in January
    d = date(year, 1, 1)
    mondays = 0
    while mondays < 3:
        if d.weekday() == 0:
            mondays += 1
        if mondays < 3:
            d += timedelta(days=1)
    holidays.add(d)

    # Presidents' Day: 3rd Monday in February
    d = date(year, 2, 1)
    mondays = 0
    while mondays < 3:
        if d.weekday() == 0:
            mondays += 1
        if mondays < 3:
            d += timedelta(days=1)
    holidays.add(d)

    # Good Friday
    holidays.add(_good_friday(year))

    # Memorial Day: last Monday in May
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    holidays.add(d)

    # Juneteenth (observed since 2022)
    if year >= 2022:
        holidays.add(_observed_holiday(date(year, 6, 19)))

    # Independence Day
    holidays.add(_observed_holiday(date(year, 7, 4)))

    # Labor Day: 1st Monday in September
    d = date(year, 9, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    holidays.add(d)

    # Thanksgiving: 4th Thursday in November
    d = date(year, 11, 1)
    thursdays = 0
    while thursdays < 4:
        if d.weekday() == 3:
            thursdays += 1
        if thursdays < 4:
            d += timedelta(days=1)
    holidays.add(d)

    # Christmas
    holidays.add(_observed_holiday(date(year, 12, 25)))

    return holidays


def market_gap_days(
    dates: np.ndarray,
    market: str = "nyse",
) -> np.ndarray:
    """
    Compute gap lengths between consecutive observations.

    For NYSE: accounts for weekends and holidays.
    For crypto: gap_days is always 1 (24/7 trading).

    Parameters
    ----------
    dates : array-like
        Array of dates (datetime64, date objects, or strings).
    market : str
        Market identifier: "nyse", "nasdaq", "crypto".

    Returns
    -------
    np.ndarray
        Array of gap days (length T-1 for T dates). gap_days[i] = days between
        dates[i] and dates[i+1] in trading days.
    """
    # Convert to date objects
    if hasattr(dates, 'dtype') and np.issubdtype(dates.dtype, np.datetime64):
        date_list = [d.astype('datetime64[D]').astype(date) for d in dates]
    elif isinstance(dates, np.ndarray):
        date_list = [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in dates]
    else:
        date_list = list(dates)

    if len(date_list) < 2:
        return np.array([], dtype=np.int64)

    market = market.lower()

    if market == "crypto":
        # Crypto: 24/7, every calendar day counts
        gaps = np.array([
            (date_list[i + 1] - date_list[i]).days
            for i in range(len(date_list) - 1)
        ], dtype=np.int64)
        return np.maximum(gaps, 1)

    # NYSE/NASDAQ: count trading days between consecutive observations
    # A trading day is a weekday that is not a holiday
    # Collect all needed years' holidays
    years = set()
    for d in date_list:
        years.add(d.year)
    all_holidays = set()
    for y in years:
        all_holidays.update(nyse_holidays(y))

    gaps = np.zeros(len(date_list) - 1, dtype=np.int64)
    for i in range(len(date_list) - 1):
        d_start = date_list[i]
        d_end = date_list[i + 1]

        # Count trading days between d_start (exclusive) and d_end (inclusive)
        trading_days = 0
        d = d_start + timedelta(days=1)
        while d <= d_end:
            if d.weekday() < 5 and d not in all_holidays:
                trading_days += 1
            d += timedelta(days=1)

        gaps[i] = max(trading_days, 1)

    return gaps


# ---------------------------------------------------------------------------
# Story 29.3: Graceful Degradation on Extreme Missing Data
# ---------------------------------------------------------------------------

def data_quality_score(
    returns: np.ndarray,
    expected_obs: int,
) -> DataQualityResult:
    """
    Assess data quality and recommend appropriate degradation.

    Parameters
    ----------
    returns : array-like
        Available return observations (may contain NaN for missing).
    expected_obs : int
        Expected number of observations for the period.

    Returns
    -------
    DataQualityResult
        Quality assessment with recommended actions.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    expected_obs = max(int(expected_obs), 1)

    # Count valid (non-NaN, finite) observations
    n_available = int(np.sum(np.isfinite(r)))
    quality = n_available / expected_obs
    quality = min(quality, 1.0)  # Cap at 1.0

    # Determine quality flag and actions
    if quality >= QUALITY_NORMAL:
        flag = "NORMAL"
        multiplier = 1.0
        suppress = False
    elif quality >= QUALITY_REDUCED:
        flag = "REDUCED_CONFIDENCE"
        multiplier = INTERVAL_MULTIPLIER_REDUCED
        suppress = False
    elif quality >= QUALITY_LOW:
        flag = "LOW_QUALITY"
        multiplier = INTERVAL_MULTIPLIER_LOW
        suppress = True  # Suppress directional signals
    else:
        flag = "UNUSABLE"
        multiplier = float('inf')
        suppress = True

    return DataQualityResult(
        quality_score=quality,
        quality_flag=flag,
        interval_multiplier=multiplier,
        suppress_direction=suppress,
        n_available=n_available,
        n_expected=expected_obs,
    )
