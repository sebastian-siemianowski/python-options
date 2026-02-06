#!/usr/bin/env python3
"""
===============================================================================
HIGH CONVICTION SIGNAL STORAGE WITH OPTIONS DATA
===============================================================================

Stores high conviction BUY and SELL signals to separate directories with
comprehensive data including:
- Signal metadata (ticker, sector, horizon, probability, expected return)
- Historical price data (30 days)
- Options chain data (calls for buys, puts for sells)

Directory structure:
    src/data/high_conviction/
    ├── buy/
    │   ├── AAPL_1d.json
    │   ├── AAPL_3d.json
    │   └── manifest.json
    └── sell/
        ├── TSLA_7d.json
        └── manifest.json

Each run completely regenerates directories (no incremental updates).

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import json
import os
import re
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log, sqrt, exp

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds matching signals_ux.py render_strong_signals_summary
BUY_THRESHOLD = 0.62      # P(r>0) >= 62% for strong buy
SELL_THRESHOLD = 0.38     # P(r>0) <= 38% for strong sell
MIN_EXPECTED_MOVE = 0.02  # Minimum 2% expected move

# Options data configuration
OPTIONS_EXPIRY_DAYS = 30  # Fetch options expiring within 30 days
PRICE_HISTORY_DAYS = 30   # Historical price data lookback

# Risk-free rate assumption
RISK_FREE_RATE = 0.045  # 4.5% as of early 2026

# Parallel fetching
MAX_WORKERS = 8


# =============================================================================
# BLACK-SCHOLES GREEKS CALCULATION
# =============================================================================

def _calculate_greeks(
    S: float,           # Current stock price
    K: float,           # Strike price
    T: float,           # Time to expiration (years)
    r: float,           # Risk-free rate
    sigma: float,       # Implied volatility (decimal, e.g., 0.30 for 30%)
    option_type: str,   # "call" or "put"
) -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks for an option.
    
    Returns dict with: delta, gamma, theta, vega, rho
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "delta": None,
            "gamma": None,
            "theta": None,
            "vega": None,
            "rho": None,
        }
    
    try:
        # Calculate d1 and d2
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        # Standard normal PDF and CDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        n_d1 = norm.pdf(d1)
        
        # Calculate Greeks
        if option_type == "call":
            delta = N_d1
            theta = (-(S * n_d1 * sigma) / (2 * sqrt(T)) 
                     - r * K * exp(-r * T) * N_d2) / 365  # Per day
            rho = K * T * exp(-r * T) * N_d2 / 100  # Per 1% rate change
        else:  # put
            delta = N_d1 - 1  # Equivalent to -N(-d1)
            theta = (-(S * n_d1 * sigma) / (2 * sqrt(T)) 
                     + r * K * exp(-r * T) * N_neg_d2) / 365  # Per day
            rho = -K * T * exp(-r * T) * N_neg_d2 / 100  # Per 1% rate change
        
        # Gamma and Vega are same for calls and puts
        gamma = n_d1 / (S * sigma * sqrt(T))
        vega = S * n_d1 * sqrt(T) / 100  # Per 1% vol change
        
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4),
        }
    except Exception:
        return {
            "delta": None,
            "gamma": None,
            "theta": None,
            "vega": None,
            "rho": None,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_ticker(asset_label: str) -> str:
    """
    Extract ticker symbol from asset label.
    
    Formats handled:
    - "Apple Inc. (AAPL)" → "AAPL"
    - "AAPL" → "AAPL"
    - "Gold (GC=F)" → "GC=F"
    """
    match = re.search(r'\(([A-Z0-9.=^-]+)\)', asset_label)
    if match:
        return match.group(1)
    # Fallback: first word if no parentheses
    return asset_label.split()[0] if asset_label else "UNKNOWN"


def _sanitize_filename(ticker: str) -> str:
    """Convert ticker to safe filename (replace special chars)."""
    return ticker.replace("^", "_").replace("=", "_").replace(".", "_")


def _fetch_historical_prices(ticker: str, days: int = 30) -> Optional[Dict]:
    """
    Fetch historical price data for a ticker.
    
    Returns dict with:
    - prices: list of {date, open, high, low, close, volume}
    - current_price: latest close
    - price_change_pct: percent change over period
    """
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 5)  # Buffer for weekends
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                            end=end_date.strftime("%Y-%m-%d"))
        
        if hist.empty:
            return None
        
        prices = []
        for idx, row in hist.iterrows():
            prices.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        
        if len(prices) < 2:
            return None
            
        current_price = prices[-1]["close"]
        first_price = prices[0]["close"]
        price_change_pct = ((current_price - first_price) / first_price * 100) if first_price > 0 else 0.0
        
        return {
            "prices": prices[-days:],  # Limit to requested days
            "current_price": current_price,
            "price_change_pct": round(price_change_pct, 2),
            "data_start": prices[0]["date"],
            "data_end": prices[-1]["date"],
        }
    except Exception as e:
        return {"error": str(e)}


def _fetch_options_chain(ticker: str, option_type: str = "calls", days_ahead: int = 30) -> Optional[Dict]:
    """
    Fetch options chain data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        option_type: "calls" for strong buys, "puts" for strong sells
        days_ahead: Maximum days until expiration to include
        
    Returns dict with:
    - expiration_dates: available expiration dates
    - options: list of option contracts with greeks
    - nearest_expiry: closest expiration date options
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        try:
            expirations = stock.options
        except Exception:
            return {"error": "No options available", "options": []}
        
        if not expirations:
            return {"error": "No options available", "options": []}
        
        # Filter to expirations within days_ahead
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        valid_expirations = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                if exp_date <= cutoff_date:
                    valid_expirations.append(exp_str)
            except ValueError:
                continue
        
        if not valid_expirations:
            # Use nearest expiration if none within range
            valid_expirations = [expirations[0]] if expirations else []
        
        all_options = []
        
        for exp_date in valid_expirations[:3]:  # Limit to 3 nearest expirations
            try:
                opt_chain = stock.option_chain(exp_date)
                options_df = opt_chain.calls if option_type == "calls" else opt_chain.puts
                
                if options_df.empty:
                    continue
                
                # Get current stock price for moneyness calculation
                current_price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose", 0)
                
                for _, opt in options_df.iterrows():
                    strike = float(opt.get("strike", 0))
                    
                    # Calculate moneyness
                    if current_price > 0:
                        moneyness = (strike - current_price) / current_price * 100
                    else:
                        moneyness = 0
                    
                    # Filter to reasonable strikes (within 20% of current price)
                    if abs(moneyness) > 20:
                        continue
                    
                    # Get implied volatility (handle both decimal and percentage formats)
                    iv_raw = float(opt.get("impliedVolatility", 0))
                    iv_decimal = iv_raw if iv_raw < 1 else iv_raw / 100  # Normalize to decimal
                    iv_pct = iv_decimal * 100  # For display
                    
                    option_data = {
                        "expiration": exp_date,
                        "strike": strike,
                        "type": option_type[:-1],  # "call" or "put"
                        "last_price": round(float(opt.get("lastPrice", 0)), 2),
                        "bid": round(float(opt.get("bid", 0)), 2),
                        "ask": round(float(opt.get("ask", 0)), 2),
                        "volume": int(opt.get("volume", 0)) if pd.notna(opt.get("volume")) else 0,
                        "open_interest": int(opt.get("openInterest", 0)) if pd.notna(opt.get("openInterest")) else 0,
                        "implied_volatility_pct": round(iv_pct, 2),
                        "moneyness_pct": round(moneyness, 2),
                        "in_the_money": bool(opt.get("inTheMoney", False)),
                        "contract_symbol": str(opt.get("contractSymbol", "")),
                    }
                    
                    # Calculate time to expiration in years
                    try:
                        exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
                        days_to_exp = (exp_datetime - datetime.now()).days
                        T = max(days_to_exp / 365.0, 1/365.0)  # At least 1 day
                        option_data["days_to_expiration"] = max(days_to_exp, 1)
                    except Exception:
                        T = 30 / 365.0  # Default to 30 days
                        option_data["days_to_expiration"] = 30
                    
                    # Calculate Greeks using Black-Scholes
                    if iv_decimal > 0 and current_price > 0:
                        greeks = _calculate_greeks(
                            S=current_price,
                            K=strike,
                            T=T,
                            r=RISK_FREE_RATE,
                            sigma=iv_decimal,
                            option_type=option_type[:-1],  # "call" or "put"
                        )
                        option_data.update(greeks)
                    else:
                        option_data.update({
                            "delta": None,
                            "gamma": None,
                            "theta": None,
                            "vega": None,
                            "rho": None,
                        })
                    
                    all_options.append(option_data)
                    
            except Exception as e:
                continue
        
        # Sort by expiration then strike
        all_options.sort(key=lambda x: (x["expiration"], x["strike"]))
        
        # Identify ATM options (closest to current price)
        atm_options = []
        if all_options:
            for exp in valid_expirations[:1]:  # Nearest expiry ATM
                exp_opts = [o for o in all_options if o["expiration"] == exp]
                if exp_opts:
                    # Find closest to ATM
                    atm = min(exp_opts, key=lambda x: abs(x["moneyness_pct"]))
                    atm_options.append(atm)
        
        return {
            "underlying_price": current_price,
            "expiration_dates": valid_expirations,
            "total_contracts": len(all_options),
            "options": all_options[:50],  # Limit to 50 contracts
            "atm_options": atm_options,
            "option_type": option_type,
            "greeks_metadata": {
                "model": "Black-Scholes",
                "risk_free_rate_pct": RISK_FREE_RATE * 100,
                "note": "Delta: position sensitivity, Gamma: delta sensitivity, Theta: time decay/day, Vega: vol sensitivity/1%, Rho: rate sensitivity/1%"
            },
            # Enhanced IV surface metrics for options tuning
            "iv_surface_metrics": _compute_iv_surface_metrics(all_options, current_price),
        }
        
    except Exception as e:
        return {"error": str(e), "options": []}


def _compute_iv_surface_metrics(options: List[Dict], current_price: float) -> Dict:
    """
    Compute IV surface metrics for options tuning.
    
    Metrics computed:
    - ATM IV: Implied volatility at the money
    - IV Skew: Difference between OTM put and call IVs
    - Term structure: IV by expiration
    - Put/Call IV ratio at various moneyness levels
    
    These metrics are used by option_tune.py for volatility modeling.
    """
    if not options or current_price <= 0:
        return {"error": "insufficient_data"}
    
    try:
        import numpy as np
        
        # Group by expiration
        by_expiry = {}
        for opt in options:
            exp = opt.get("expiration", "unknown")
            if exp not in by_expiry:
                by_expiry[exp] = []
            by_expiry[exp].append(opt)
        
        # Compute per-expiry metrics
        expiry_metrics = {}
        all_ivs = []
        
        for exp, exp_options in by_expiry.items():
            ivs = [o.get("implied_volatility_pct", 0) for o in exp_options if o.get("implied_volatility_pct", 0) > 0]
            if not ivs:
                continue
            
            all_ivs.extend(ivs)
            
            # Find ATM option (closest to current price)
            atm_opt = min(exp_options, key=lambda x: abs(x.get("strike", 0) - current_price))
            atm_iv = atm_opt.get("implied_volatility_pct", 0) / 100 if atm_opt else 0
            
            # Compute skew (25-delta risk reversal approximation)
            # Lower strike IV - higher strike IV for puts (negative = normal skew)
            sorted_by_strike = sorted(exp_options, key=lambda x: x.get("strike", 0))
            if len(sorted_by_strike) >= 3:
                low_strike_iv = sorted_by_strike[0].get("implied_volatility_pct", 0)
                high_strike_iv = sorted_by_strike[-1].get("implied_volatility_pct", 0)
                skew = (low_strike_iv - high_strike_iv) / 100 if low_strike_iv > 0 and high_strike_iv > 0 else 0
            else:
                skew = 0
            
            # Average IV for this expiry
            avg_iv = np.mean(ivs) / 100
            
            expiry_metrics[exp] = {
                "atm_iv": round(atm_iv, 4),
                "avg_iv": round(avg_iv, 4),
                "skew": round(skew, 4),
                "n_contracts": len(exp_options),
                "days_to_expiry": exp_options[0].get("days_to_expiration", 0) if exp_options else 0,
            }
        
        # Overall metrics
        if all_ivs:
            overall_avg_iv = np.mean(all_ivs) / 100
            overall_std_iv = np.std(all_ivs) / 100
            iv_range = (min(all_ivs) / 100, max(all_ivs) / 100)
        else:
            overall_avg_iv = 0
            overall_std_iv = 0
            iv_range = (0, 0)
        
        # Term structure (slope)
        term_structure_slope = 0
        if len(expiry_metrics) >= 2:
            exps = sorted(expiry_metrics.keys())
            first_exp = expiry_metrics[exps[0]]
            last_exp = expiry_metrics[exps[-1]]
            if first_exp.get("days_to_expiry", 0) > 0 and last_exp.get("days_to_expiry", 0) > first_exp.get("days_to_expiry", 0):
                day_diff = last_exp["days_to_expiry"] - first_exp["days_to_expiry"]
                iv_diff = last_exp["avg_iv"] - first_exp["avg_iv"]
                term_structure_slope = iv_diff / day_diff * 30  # Per 30 days
        
        return {
            "overall_avg_iv": round(overall_avg_iv, 4),
            "overall_std_iv": round(overall_std_iv, 4),
            "iv_range": (round(iv_range[0], 4), round(iv_range[1], 4)),
            "term_structure_slope_per_30d": round(term_structure_slope, 4),
            "by_expiry": expiry_metrics,
            "n_expirations": len(expiry_metrics),
        }
        
    except Exception as e:
        return {"error": str(e)}


def _fetch_signal_data(signal: Dict, is_buy: bool) -> Dict:
    """
    Fetch all supplementary data for a signal.
    
    Args:
        signal: Signal dict with ticker, sector, etc.
        is_buy: True for buy signals (fetch calls), False for sell (fetch puts)
        
    Returns:
        Signal dict enriched with price history and options data
    """
    ticker = signal.get("ticker", "")
    
    # Skip non-optionable assets (indices, futures, forex)
    skip_options = ticker.startswith("^") or "=" in ticker or ticker.endswith(".X")
    
    # Fetch historical prices
    price_data = _fetch_historical_prices(ticker, PRICE_HISTORY_DAYS)
    signal["price_history"] = price_data if price_data else {"error": "Failed to fetch prices"}
    
    # Fetch options chain (calls for buys, puts for sells)
    if not skip_options:
        option_type = "calls" if is_buy else "puts"
        options_data = _fetch_options_chain(ticker, option_type, OPTIONS_EXPIRY_DAYS)
        signal["options_chain"] = options_data if options_data else {"error": "Failed to fetch options"}
    else:
        signal["options_chain"] = {"skipped": True, "reason": "Non-optionable asset (index/future/forex)"}
    
    return signal


# =============================================================================
# MAIN STORAGE FUNCTION
# =============================================================================

def save_high_conviction_signals(
    summary_rows: List[Dict],
    horizons: List[int] = None,
    fetch_options: bool = True,
    fetch_prices: bool = True,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, int]:
    """
    Save high conviction BUY and SELL signals to separate directories.
    
    Files are saved to:
    - src/data/high_conviction/buy/{ticker}_{horizon}d.json
    - src/data/high_conviction/sell/{ticker}_{horizon}d.json
    
    Each run completely regenerates the directories, replacing previous versions.
    
    Args:
        summary_rows: List of summary row dicts from signal generation
        horizons: List of horizons to check (default: [1, 3, 7])
        fetch_options: Whether to fetch options chain data
        fetch_prices: Whether to fetch historical price data
        max_workers: Number of parallel workers for data fetching
        
    Returns:
        Dict with counts: {"buy": N, "sell": M, "errors": E}
    """
    if not summary_rows:
        return {"buy": 0, "sell": 0, "errors": 0}
    
    if horizons is None:
        horizons = [1, 3, 7]
    
    # Define paths relative to this file's location
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "high_conviction")
    buy_dir = os.path.join(base_dir, "buy")
    sell_dir = os.path.join(base_dir, "sell")
    
    # Clear and recreate directories
    for dir_path in [buy_dir, sell_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # Collect high conviction signals
    buy_signals = []
    sell_signals = []
    
    for row in summary_rows:
        asset_label = row.get("asset_label", "Unknown")
        horizon_signals = row.get("horizon_signals", {})
        sector = row.get("sector", "Other")
        ticker = _extract_ticker(asset_label)
        
        for horizon in horizons:
            signal_data = horizon_signals.get(horizon) or horizon_signals.get(str(horizon)) or {}
            p_up = signal_data.get("p_up", 0.5)
            exp_ret = signal_data.get("exp_ret", 0.0)
            profit_pln = signal_data.get("profit_pln", 0.0)
            label = signal_data.get("label", "HOLD")
            
            # Skip EXIT signals
            if label.upper() == "EXIT":
                continue
            
            # Calculate strength
            distance_from_neutral = abs(p_up - 0.5)
            strength = distance_from_neutral + abs(exp_ret) * 0.5
            
            # Build base signal record
            signal_record = {
                "ticker": ticker,
                "asset_label": asset_label,
                "sector": sector,
                "horizon_days": horizon,
                "signal_type": None,  # Set below
                "probability_up": round(p_up, 4),
                "probability_down": round(1 - p_up, 4),
                "expected_return_pct": round(exp_ret * 100, 2),
                "expected_profit_pln": round(profit_pln, 2),
                "signal_strength": round(strength, 4),
                "generated_at": timestamp,
            }
            
            # Classify as strong buy or sell
            if p_up >= BUY_THRESHOLD and exp_ret >= MIN_EXPECTED_MOVE:
                signal_record["signal_type"] = "STRONG_BUY"
                signal_record["conviction_probability"] = round(p_up, 4)
                buy_signals.append(signal_record)
            elif p_up <= SELL_THRESHOLD and exp_ret <= -MIN_EXPECTED_MOVE:
                signal_record["signal_type"] = "STRONG_SELL"
                signal_record["conviction_probability"] = round(1 - p_up, 4)
                sell_signals.append(signal_record)
    
    # Fetch supplementary data in parallel
    error_count = 0
    
    if fetch_options or fetch_prices:
        # Combine all signals with their buy/sell flag
        all_signals = [(s, True) for s in buy_signals] + [(s, False) for s in sell_signals]
        
        if all_signals:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_fetch_signal_data, sig, is_buy): (sig, is_buy)
                    for sig, is_buy in all_signals
                }
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        error_count += 1
    
    # Sort signals
    buy_signals.sort(key=lambda x: (-x["expected_return_pct"], -x["probability_up"]))
    sell_signals.sort(key=lambda x: (x["expected_return_pct"], x["probability_up"]))
    
    # Write individual signal files
    buy_files = []
    for sig in buy_signals:
        ticker_safe = _sanitize_filename(sig["ticker"])
        filename = f"{ticker_safe}_{sig['horizon_days']}d.json"
        filepath = os.path.join(buy_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(sig, f, indent=2, default=str)
        buy_files.append(filename)
    
    sell_files = []
    for sig in sell_signals:
        ticker_safe = _sanitize_filename(sig["ticker"])
        filename = f"{ticker_safe}_{sig['horizon_days']}d.json"
        filepath = os.path.join(sell_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(sig, f, indent=2, default=str)
        sell_files.append(filename)
    
    # Write manifests
    for dir_path, files, signal_type in [
        (buy_dir, buy_files, "STRONG_BUY"),
        (sell_dir, sell_files, "STRONG_SELL"),
    ]:
        # Summary statistics
        if signal_type == "STRONG_BUY":
            signals_list = buy_signals
        else:
            signals_list = sell_signals
        
        avg_probability = np.mean([s["conviction_probability"] for s in signals_list]) if signals_list else 0
        avg_return = np.mean([s["expected_return_pct"] for s in signals_list]) if signals_list else 0
        total_profit = sum(s["expected_profit_pln"] for s in signals_list)
        
        # Sector breakdown
        sector_counts = {}
        for s in signals_list:
            sec = s.get("sector", "Other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        
        manifest = {
            "signal_type": signal_type,
            "count": len(files),
            "generated_at": timestamp,
            "thresholds": {
                "buy_probability": BUY_THRESHOLD,
                "sell_probability": SELL_THRESHOLD,
                "min_expected_move_pct": MIN_EXPECTED_MOVE * 100,
            },
            "summary": {
                "avg_conviction_probability": round(avg_probability, 4),
                "avg_expected_return_pct": round(avg_return, 2),
                "total_expected_profit_pln": round(total_profit, 2),
            },
            "sector_breakdown": sector_counts,
            "horizons_included": horizons,
            "options_included": fetch_options,
            "price_history_included": fetch_prices,
            "files": sorted(files),
        }
        
        with open(os.path.join(dir_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    
    return {
        "buy": len(buy_signals),
        "sell": len(sell_signals),
        "errors": error_count,
        "buy_dir": buy_dir,
        "sell_dir": sell_dir,
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    from rich.console import Console
    
    parser = argparse.ArgumentParser(description="Export high conviction signals with options data")
    parser.add_argument("--no-options", action="store_true", help="Skip options chain fetching")
    parser.add_argument("--no-prices", action="store_true", help="Skip price history fetching")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()
    
    console = Console()
    console.print("[bold]High Conviction Signal Storage[/bold]")
    console.print("This module is typically called from signals.py after signal generation.")
    console.print("\nTo generate signals, run: [cyan]make stocks[/cyan]")
