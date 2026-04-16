"""
Story 8.4: Pair Trading Cointegration Engine.

Engle-Granger cointegration testing, OU half-life estimation, and spread signals.

Usage:
    from decision.pair_trading import (
        test_cointegration,
        estimate_ou_halflife,
        generate_pair_signal,
    )
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Configuration
COINT_PVALUE_THRESHOLD = 0.05   # p-value for cointegration
OU_HALFLIFE_MAX = 60            # max half-life (days) to consider tradeable
ENTRY_ZSCORE = 2.0              # entry threshold (std devs)
EXIT_ZSCORE = 0.5               # exit threshold
TOP_PAIRS = 20                  # max pairs to select


@dataclass
class PairResult:
    """Cointegration test result for a pair."""
    asset_a: str
    asset_b: str
    adf_stat: float
    pvalue: float
    is_cointegrated: bool
    hedge_ratio: float
    halflife: float
    spread_zscore: float
    signal: str   # "CONVERGE_LONG", "CONVERGE_SHORT", "NEUTRAL"


def test_cointegration(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Engle-Granger two-step cointegration test.
    
    Step 1: OLS regression of log(A) on log(B) -> hedge ratio
    Step 2: ADF test on residuals (spread)
    
    Uses simplified ADF (AR(1) t-test on residual changes).
    
    Args:
        prices_a: Price series A.
        prices_b: Price series B.
    
    Returns:
        (adf_statistic, p_value_approx, hedge_ratio)
    """
    log_a = np.log(prices_a)
    log_b = np.log(prices_b)
    
    # Step 1: OLS hedge ratio
    # log_a = alpha + beta * log_b + epsilon
    x = np.column_stack([np.ones(len(log_b)), log_b])
    beta = np.linalg.lstsq(x, log_a, rcond=None)[0]
    hedge_ratio = beta[1]
    
    # Spread (residuals)
    spread = log_a - beta[0] - hedge_ratio * log_b
    
    # Step 2: ADF test on spread
    # Delta(spread) = rho * spread_{t-1} + error
    dy = np.diff(spread)
    y_lag = spread[:-1]
    
    x_adf = y_lag.reshape(-1, 1)
    rho = np.linalg.lstsq(x_adf, dy, rcond=None)[0][0]
    
    residuals = dy - rho * y_lag
    se_rho = float(np.std(residuals) / np.sqrt(np.sum(y_lag ** 2)))
    
    if se_rho < 1e-15:
        adf_stat = 0.0
    else:
        adf_stat = float(rho / se_rho)
    
    # Approximate p-value using MacKinnon critical values for N=2
    # -2.58 (10%), -2.88 (5%), -3.43 (1%)
    if adf_stat < -3.43:
        pvalue = 0.01
    elif adf_stat < -2.88:
        pvalue = 0.05
    elif adf_stat < -2.58:
        pvalue = 0.10
    else:
        pvalue = 0.50
    
    return adf_stat, pvalue, float(hedge_ratio)


def estimate_ou_halflife(spread: np.ndarray) -> float:
    """
    Estimate Ornstein-Uhlenbeck half-life from spread series.
    
    OU: dX = theta * (mu - X) dt + sigma dW
    Half-life = ln(2) / theta
    
    Estimated via AR(1): X_t = phi * X_{t-1} + c
    theta = -ln(phi), half-life = ln(2) / theta
    
    Args:
        spread: Spread time series.
    
    Returns:
        Half-life in days. Capped at OU_HALFLIFE_MAX.
    """
    if len(spread) < 10:
        return float(OU_HALFLIFE_MAX)
    
    y = spread[1:]
    x = spread[:-1].reshape(-1, 1)
    
    phi = float(np.linalg.lstsq(x, y, rcond=None)[0][0])
    
    if phi >= 1.0 or phi <= 0.0:
        return float(OU_HALFLIFE_MAX)
    
    theta = -np.log(phi)
    halflife = np.log(2) / theta
    
    return min(float(halflife), float(OU_HALFLIFE_MAX))


def compute_spread(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio: float,
) -> np.ndarray:
    """Compute log-price spread."""
    log_a = np.log(prices_a)
    log_b = np.log(prices_b)
    return log_a - hedge_ratio * log_b


def compute_zscore(spread: np.ndarray, window: int = 60) -> float:
    """Current z-score of spread relative to rolling stats."""
    if len(spread) < window:
        window = max(len(spread), 2)
    
    recent = spread[-window:]
    mu = np.mean(recent)
    sigma = np.std(recent)
    
    if sigma < 1e-10:
        return 0.0
    
    return float((spread[-1] - mu) / sigma)


def generate_pair_signal(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    asset_a: str,
    asset_b: str,
) -> PairResult:
    """
    Full pair trading analysis: cointegration + OU + signal.
    
    Args:
        prices_a: Price series A.
        prices_b: Price series B.
        asset_a: Symbol A.
        asset_b: Symbol B.
    
    Returns:
        PairResult with cointegration test, half-life, and signal.
    """
    adf_stat, pvalue, hedge_ratio = test_cointegration(prices_a, prices_b)
    
    is_coint = pvalue <= COINT_PVALUE_THRESHOLD
    
    spread = compute_spread(prices_a, prices_b, hedge_ratio)
    halflife = estimate_ou_halflife(spread)
    zscore = compute_zscore(spread)
    
    # Signal
    if not is_coint or halflife >= OU_HALFLIFE_MAX:
        signal = "NEUTRAL"
    elif zscore > ENTRY_ZSCORE:
        signal = "CONVERGE_SHORT"  # Spread too high -> short A, long B
    elif zscore < -ENTRY_ZSCORE:
        signal = "CONVERGE_LONG"   # Spread too low -> long A, short B
    elif abs(zscore) < EXIT_ZSCORE:
        signal = "NEUTRAL"
    else:
        signal = "NEUTRAL"
    
    return PairResult(
        asset_a=asset_a,
        asset_b=asset_b,
        adf_stat=adf_stat,
        pvalue=pvalue,
        is_cointegrated=is_coint,
        hedge_ratio=hedge_ratio,
        halflife=halflife,
        spread_zscore=zscore,
        signal=signal,
    )


def screen_pairs(
    price_dict: dict,
    symbols: List[str],
    top_n: int = TOP_PAIRS,
) -> List[PairResult]:
    """
    Screen all pairs for cointegration, return top N.
    
    Args:
        price_dict: {symbol: price_array}.
        symbols: List of symbols to test.
        top_n: Max pairs to return.
    
    Returns:
        Top pairs sorted by ADF statistic (most negative = strongest).
    """
    results = []
    
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = symbols[i], symbols[j]
            if a not in price_dict or b not in price_dict:
                continue
            
            pa = price_dict[a]
            pb = price_dict[b]
            
            min_len = min(len(pa), len(pb))
            if min_len < 60:
                continue
            
            result = generate_pair_signal(pa[-min_len:], pb[-min_len:], a, b)
            if result.is_cointegrated:
                results.append(result)
    
    # Sort by ADF (most negative = strongest cointegration)
    results.sort(key=lambda r: r.adf_stat)
    return results[:top_n]
