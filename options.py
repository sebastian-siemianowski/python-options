"""
options_screener_0_3_7dte.py

Scans a universe of liquid tickers for CALL options with 0, 3 and 7 DTE that have the *highest probability*
of producing >=1000% (10x) return by expiry, filters by liquidity (volume & open interest),
plots price charts with simple support/resistance (pivot-based) and buy/sell markers,
and runs a conservative backtest using historical underlying data and option-pricing (BSM) to approximate
how often the 10x return would have occurred historically.

NOTES / DISCLAIMER:
- This is research & educational code only. Options are risky. This script DOES download real data
  from Yahoo Finance via yfinance for underlying historicals and live option chains.
- Historical option-level tick-by-tick data is generally not available via Yahoo; the backtest here
  *approximates* option prices via Black-Scholes using historical realized vol or available implied vol
  and is therefore an approximation (not "made up" prices, but modelled prices).
- You must `pip install yfinance numpy pandas scipy matplotlib tqdm` before running.

Usage example:
    python options_screener_0_3_7dte.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
 - screener_results.csv  (ranked by probability of 1000%+ return)
 - backtest_report.csv
 - plots/<TICKER>_support_resistance.png (chart files)

"""

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- Black-Scholes helpers --------------------

def bsm_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def bsm_implied_vol(price, S, K, T, r):
    # invert BSM for calls using Brent
    intrinsic = max(S - K, 0.0)
    price = float(max(price, intrinsic + 1e-8))
    def f(sig):
        return bsm_call_price(S, K, T, r, sig) - price
    try:
        return brentq(f, 1e-6, 5.0, maxiter=300)
    except Exception:
        return np.nan


def lognormal_prob_geq(S0, mu_ln, sigma_ln, threshold):
    # Probability that lognormal(X; mu_ln, sigma_ln) >= threshold
    # X ~ lognormal with parameters mu_ln, sigma_ln where ln(X) ~ N(mu_ln, sigma_ln^2)
    if threshold <= 0:
        return 1.0
    z = (np.log(threshold) - mu_ln) / sigma_ln
    return 1.0 - norm.cdf(z)

# -------------------- Utility functions --------------------

def days_to_expiry_from_date(expiry_date, ref_date=None):
    if ref_date is None:
        ref_date = datetime.utcnow()
    return max(0, (pd.to_datetime(expiry_date) - pd.to_datetime(ref_date)).days)


def get_closest_expiry_dates(option_dates, target_days):
    # option_dates: list of strings like '2025-09-15'
    # target_days: integer (0,3,7)
    target_days = int(target_days)
    dates = [pd.to_datetime(d) for d in option_dates]
    today = pd.to_datetime(datetime.utcnow().date())
    best = None
    best_diff = 999
    for d in dates:
        diff = abs((d - today).days - target_days)
        if diff < best_diff:
            best_diff = diff
            best = d
    return best

# -------------------- Screener core --------------------

def analyze_ticker_for_dtes(ticker, dte_targets=(0,3,7), min_oi=100, min_volume=20, r=0.01, hist_years=1):
    tk = yf.Ticker(ticker)
    # load underlying history (N years daily) to use in backtest & volatility estimates
    period_str = f"{int(max(1, hist_years))}y"
    hist = tk.history(period=period_str, interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No historical data for {ticker}")
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist[['Date','Open','High','Low','Close','Volume']].copy()

    # compute realized vol (rolling 21-day daily vol annualized)
    hist['ret'] = hist['Close'].pct_change()
    hist['rv21'] = hist['ret'].rolling(21).std() * np.sqrt(252)
    hist['rv21'] = hist['rv21'].fillna(method='bfill').fillna(hist['rv21'].median())

    opportunities = []

    try:
        option_dates = tk.options
    except Exception:
        option_dates = []
    if not option_dates:
        # no option chain (eg. some ETFs or delisted) -> return empty
        return pd.DataFrame(), hist

    processed_expiries = set()
    for target in dte_targets:
        expiry = get_closest_expiry_dates(option_dates, target)
        if expiry is None:
            continue
        expiry_str = expiry.strftime('%Y-%m-%d')
        # Avoid processing the same expiry multiple times when multiple target DTEs map to the same date
        if expiry_str in processed_expiries:
            continue
        processed_expiries.add(expiry_str)
        try:
            chain = tk.option_chain(expiry_str)
            calls = chain.calls.copy()
        except Exception:
            continue

        # Compute mid price and filter liquidity
        if 'bid' in calls.columns and 'ask' in calls.columns:
            calls['mid'] = (calls['bid'].fillna(0) + calls['ask'].fillna(0)) / 2.0
        else:
            calls['mid'] = calls['lastPrice'].fillna(0.0)
        # ensure OI/volume exist
        calls['openInterest'] = calls.get('openInterest', np.nan)
        calls['volume'] = calls.get('volume', np.nan)
        calls = calls.assign(strike=lambda df: df['strike'].astype(float))

        # Underlying spot
        spot = float(hist['Close'].iloc[-1])
        # Time to expiry in fraction of year
        days_to_expiry = max(0, (pd.to_datetime(expiry_str).date() - datetime.utcnow().date()).days)
        # If target is 0 DTE but we selected expiry today but yahoo may not include 0DTE as same-day; treat T as small
        T_years = max(1/252.0, days_to_expiry/252.0)

        # For each call compute probability of 10x return
        for _, row in calls.iterrows():
            strike = float(row['strike'])
            mid = float(row['mid'])
            oi = float(row['openInterest']) if not pd.isna(row['openInterest']) else 0.0
            volm = float(row['volume']) if not pd.isna(row['volume']) else 0.0
            if oi < min_oi and volm < min_volume:
                continue
            if mid <= 0.01:
                # extremely cheap option; skip absurd pennies due to noise
                continue

            # Required underlying at expiry to yield 10x option price (1000% return)
            payoff_needed = mid * 10.0
            S_thresh = strike + payoff_needed  # must be >= this for payoff >= 10x

            # Derive implied vol for this option if available or approximate with historical rv21
            implied = np.nan
            if 'impliedVolatility' in row.index and not pd.isna(row['impliedVolatility']):
                # Yahoo stores as decimal (e.g., 0.35)
                implied = float(row['impliedVolatility'])
            else:
                # try invert BSM using mid and current spot
                implied = bsm_implied_vol(mid, spot, strike, T_years, r)
            if not np.isfinite(implied) or implied <= 0:
                # fallback to recent realized vol
                implied = float(hist['rv21'].iloc[-1])

            # Under risk-neutral lognormal dynamics: ln(S_T) ~ N(ln(S0) + (r - 0.5*sigma^2)*T, sigma^2*T)
            mu_ln = np.log(max(spot,1e-8)) + (r - 0.5*implied*implied) * T_years
            sigma_ln = np.sqrt(max(1e-12, implied*implied * T_years))

            prob_10x = float(lognormal_prob_geq(spot, mu_ln, sigma_ln, S_thresh))

            # approximate expected return conditional on achieving 10x (simplistic)
            expected_return_if_hit = 10.0

            opportunities.append({
                'ticker': ticker,
                'expiry': expiry_str,
                'dte': days_to_expiry,
                'strike': strike,
                'mid': mid,
                'openInterest': oi,
                'volume': volm,
                'impliedVol': implied,
                'S0': spot,
                'S_thresh_for_10x': S_thresh,
                'prob_10x': prob_10x,
                'estimated_return_if_hit_x': expected_return_if_hit
            })

    df_ops = pd.DataFrame(opportunities)
    if not df_ops.empty:
        # Deduplicate in case multiple target DTEs map to same expiry
        df_ops = df_ops.drop_duplicates(subset=['ticker','expiry','strike','dte'], keep='first')
        # sort and return top candidates
        df_ops = df_ops.sort_values(['prob_10x','openInterest','volume'], ascending=[False,False,False])
    return df_ops, hist

# -------------------- Backtest approximation --------------------

def approximate_backtest_option_10x(ticker, candidate_row, hist, r=0.01):
    # candidate_row: one row from opportunities with expiry chosen.
    # We'll perform a rolling backtest: for each historical business day in hist where we could have bought
    # a call with same DTE relative to that day, compute whether the 10x payoff would have happened.
    # This is an approximation because historical option chains differ; we model option prices via BSM

    strike = float(candidate_row['strike'])
    dte = int(candidate_row['dte'])
    mid_price = float(candidate_row['mid'])

    results = []
    dates = hist['Date'].values
    for i in range(252, len(dates)-dte):
        buy_date = dates[i]
        S_buy = float(hist['Close'].iloc[i])
        # use realized vol at index i
        sigma = float(hist['rv21'].iloc[i])
        T_buy = max(1/252.0, dte/252.0)
        # approximate mid price at buy_date using BSM
        price_model = bsm_call_price(S_buy, strike, T_buy, r, sigma)
        if price_model <= 0:
            continue
        # required S at expiry for 10x from that price_model
        thresh = strike + 10.0 * price_model

        # look up actual close at expiry index
        expiry_idx = i + dte
        S_exp = float(hist['Close'].iloc[expiry_idx])
        hit = 1 if S_exp >= thresh else 0
        payoff = max(S_exp - strike, 0.0)
        ret_x = payoff / price_model if price_model>0 else 0.0
        results.append({'buy_date': buy_date, 'S_buy': S_buy, 'price_model': price_model, 'S_exp': S_exp, 'hit_10x': hit, 'ret_x': ret_x})

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return pd.DataFrame(), {}

    hits = df_res['hit_10x'].sum()
    tries = len(df_res)
    hit_rate = hits / tries if tries>0 else 0.0
    avg_return_x = df_res['ret_x'].mean()
    metrics = {'tries':tries, 'hits':int(hits), 'hit_rate':float(hit_rate), 'avg_return_x':float(avg_return_x)}
    return df_res, metrics

# -------------------- Strategy backtest (multi-year, SL/TP) --------------------

def backtest_breakout_option_strategy(hist, dte=7, moneyness=0.05, r=0.01, tp_x=None, sl_x=None):
    """Simulate a simple strategy: on breakout BUY signal, buy an OTM call (strike = S*(1+moneyness)),
    price via BSM, hold until TP/SL or expiry. Returns equity curve and trade log + metrics.

    Parameters:
        hist: DataFrame with Date, Close, Volume, rv21 columns
        dte: days to expiry for each trade
        moneyness: e.g., 0.05 means 5% OTM strike K = S*(1+0.05)
        r: risk-free rate
        tp_x: take-profit multiple of entry premium (e.g., 3.0 = +200%)
        sl_x: stop-loss multiple of entry premium (e.g., 0.5 = -50%)
    """
    df = hist.copy()
    if 'rv21' not in df.columns:
        df['ret'] = df['Close'].pct_change()
        df['rv21'] = df['ret'].rolling(21).std() * np.sqrt(252)
        df['rv21'] = df['rv21'].fillna(method='bfill').fillna(df['rv21'].median())
    signals = generate_breakout_signals(df)
    signal_idx = set(signals.index.tolist())

    dates = df['Date'].reset_index(drop=True)
    closes = df['Close'].reset_index(drop=True)
    vols = df['rv21'].reset_index(drop=True)

    equity = 1.0
    equity_curve = []
    trades = []

    i = 0
    n = len(df)
    while i < n:
        # Not enough time remaining for a full trade
        if i >= n - 2:
            equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
            i += 1
            continue

        # If buy signal today and enough room for trade horizon
        if i in signal_idx and i + dte < n:
            S0 = float(closes.iloc[i])
            sigma0 = float(vols.iloc[i]) if np.isfinite(vols.iloc[i]) else float(np.nanmean(vols[:i+1]))
            K = S0 * (1.0 + float(moneyness))
            price0 = bsm_call_price(S0, K, max(1/252.0, dte/252.0), r, max(1e-6, sigma0))
            if price0 <= 0:
                # skip unpriceable
                equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                i += 1
                continue
            exit_idx = i + dte
            exit_price = None
            reason = 'expiry'

            # simulate daily and check SL/TP
            for j in range(i+1, i + dte + 1):
                t_remaining = max(1/252.0, (i + dte - j)/252.0)
                S_t = float(closes.iloc[j])
                sigma_t = float(vols.iloc[j]) if np.isfinite(vols.iloc[j]) else sigma0
                model_price_t = bsm_call_price(S_t, K, t_remaining, r, max(1e-6, sigma_t)) if t_remaining>0 else max(S_t - K, 0.0)
                if tp_x is not None and model_price_t >= price0 * float(tp_x):
                    exit_idx = j
                    exit_price = model_price_t
                    reason = 'tp'
                    break
                if sl_x is not None and model_price_t <= price0 * float(sl_x):
                    exit_idx = j
                    exit_price = model_price_t
                    reason = 'sl'
                    break
            if exit_price is None:
                # expiry payoff
                S_T = float(closes.iloc[min(i + dte, n-1)])
                exit_price = max(S_T - K, 0.0)

            ret_x = (exit_price / price0) if price0 > 0 else 0.0
            equity *= max(0.0, ret_x)

            trades.append({
                'entry_date': dates.iloc[i],
                'exit_date': dates.iloc[exit_idx],
                'entry_price': price0,
                'exit_price': exit_price,
                'ret_x': ret_x,
                'days_held': int(exit_idx - i),
                'reason': reason,
                'K': K,
                'S_entry': S0,
                'S_exit': float(closes.iloc[exit_idx])
            })
            # Fill equity curve from i to exit_idx
            for k in range(i, exit_idx+1):
                equity_curve.append({'Date': dates.iloc[k], 'equity': equity})
            i = exit_idx + 1
            continue

        # No trade today
        equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
        i += 1

    eq_df = pd.DataFrame(equity_curve).drop_duplicates(subset=['Date'], keep='last')
    trades_df = pd.DataFrame(trades)

    # Metrics
    if not eq_df.empty:
        eq_df = eq_df.sort_values('Date').reset_index(drop=True)
        eq_df['ret'] = eq_df['equity'].pct_change().fillna(0.0)
        # Max drawdown
        rolling_max = eq_df['equity'].cummax()
        drawdown = (eq_df['equity'] / rolling_max) - 1.0
        max_dd = float(drawdown.min()) if len(drawdown)>0 else 0.0
        total_ret = float(eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0] - 1.0)
        days = max(1, (eq_df['Date'].iloc[-1] - eq_df['Date'].iloc[0]).days)
        cagr = float((eq_df['equity'].iloc[-1] / max(1e-9, eq_df['equity'].iloc[0])) ** (365.0/days) - 1.0) if days>0 else 0.0
        # Daily Sharpe (no risk-free subtraction for simplicity)
        sh = float(np.sqrt(252) * (eq_df['ret'].mean() / (eq_df['ret'].std() + 1e-9))) if len(eq_df)>2 else 0.0
    else:
        max_dd = 0.0
        total_ret = 0.0
        cagr = 0.0
        sh = 0.0

    win_rate = float((trades_df['ret_x'] > 1.0).mean()) if not trades_df.empty else 0.0
    avg_trade_ret_x = float(trades_df['ret_x'].mean()) if not trades_df.empty else 0.0

    metrics = {
        'total_trades': int(len(trades_df)),
        'win_rate': win_rate,
        'avg_trade_ret_x': avg_trade_ret_x,
        'total_return': total_ret,
        'CAGR': cagr,
        'Sharpe': sh,
        'max_drawdown': max_dd
    }

    return eq_df, trades_df, metrics

# -------------------- Support/Resistance & plotting --------------------

def compute_pivots_levels(price_series, window=20):
    # simple pivot levels: local rolling min/max as supports/resistances
    highs = price_series.rolling(window).max()
    lows = price_series.rolling(window).min()
    # last level
    support = float(lows.iloc[-1]) if not pd.isna(lows.iloc[-1]) else float(price_series.iloc[-1])
    resistance = float(highs.iloc[-1]) if not pd.isna(highs.iloc[-1]) else float(price_series.iloc[-1])
    return support, resistance


def plot_support_resistance_with_signals(ticker, hist, signals=None, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    dates = hist['Date']
    prices = hist['Close']
    support, resistance = compute_pivots_levels(prices, window=20)

    plt.figure(figsize=(12,6))
    plt.plot(dates, prices)
    # Per python_user_visible/charting rules we won't hardcode colors here in code comments; matplotlib will use defaults.
    plt.axhline(support, linestyle='--', label='Support')
    plt.axhline(resistance, linestyle='--', label='Resistance')

    if signals is not None and not signals.empty:
        buys = signals[signals['signal']=='BUY']
        sells = signals[signals['signal']=='SELL']
        if not buys.empty:
            plt.scatter(buys['Date'], buys['Price'], marker='^', s=80, label='BUY')
        if not sells.empty:
            plt.scatter(sells['Date'], sells['Price'], marker='v', s=80, label='SELL')

    plt.title(f"{ticker} price with support/resistance and signals")
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ticker}_support_resistance.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

# -------------------- Simple price-breakout signal generator --------------------

def generate_breakout_signals(hist, window=20, lookback=5):
    # Buy when price closes above recent resistance and volume > recent avg
    df = hist.copy()
    df['resistance'] = df['Close'].rolling(window).max().shift(1)
    df['support'] = df['Close'].rolling(window).min().shift(1)
    df['vol_avg'] = df['Volume'].rolling(window).mean().shift(1)
    df['signal'] = None
    df['Price'] = df['Close']
    for i in range(window, len(df)):
        if df['Close'].iloc[i] > df['resistance'].iloc[i] and df['Volume'].iloc[i] > 1.2 * max(1e-6, df['vol_avg'].iloc[i]):
            df.at[i,'signal'] = 'BUY'
        elif df['Close'].iloc[i] < df['support'].iloc[i]:
            df.at[i,'signal'] = 'SELL'
    return df[df['signal'].notnull()][['Date','Price','signal']]

# -------------------- Main runner --------------------

def run_screener(tickers, min_oi=200, min_vol=30, out_prefix='screener_results', bt_years=3, bt_dte=7, bt_moneyness=0.05, bt_tp_x=None, bt_sl_x=None):
    all_candidates = []
    option_bt_rows = []
    strat_rows = []
    for t in tqdm(tickers, desc='Tickers'):
        try:
            df_ops, hist = analyze_ticker_for_dtes(t, dte_targets=(0,3,7), min_oi=min_oi, min_volume=min_vol, hist_years=bt_years)
            if df_ops.empty:
                # even if no options found, still try to run strategy backtest on price data
                pass
            else:
                # select top N per ticker
                topn = df_ops.head(10)
                for _, r in topn.iterrows():
                    all_candidates.append(r.to_dict())
                    # approximate backtest of 10x condition for context
                    df_bt, metrics = approximate_backtest_option_10x(t, r, hist)
                    metrics_row = {**{'ticker':t, 'expiry':r['expiry'],'strike':r['strike'],'dte':r['dte']}, **metrics}
                    option_bt_rows.append(metrics_row)

            # Strategy backtest on extended history
            eq_df, trades_df, strat_metrics = backtest_breakout_option_strategy(
                hist, dte=bt_dte, moneyness=bt_moneyness, r=0.01, tp_x=bt_tp_x, sl_x=bt_sl_x
            )
            # save equity curve per ticker
            try:
                os.makedirs('backtests', exist_ok=True)
                eq_out = os.path.join('backtests', f"{t}_equity.csv")
                eq_df.assign(ticker=t).to_csv(eq_out, index=False)
            except Exception:
                pass

            # collect summary metrics row for strategy (per-ticker)
            strat_row = {'ticker': t, 'strategy_total_trades': strat_metrics.get('total_trades',0),
                         'strategy_win_rate': strat_metrics.get('win_rate',0.0),
                         'strategy_avg_trade_ret_x': strat_metrics.get('avg_trade_ret_x',0.0),
                         'strategy_total_return': strat_metrics.get('total_return',0.0),
                         'strategy_CAGR': strat_metrics.get('CAGR',0.0),
                         'strategy_Sharpe': strat_metrics.get('Sharpe',0.0),
                         'strategy_max_drawdown': strat_metrics.get('max_drawdown',0.0),
                         'bt_years': bt_years, 'bt_dte': bt_dte, 'bt_moneyness': bt_moneyness,
                         'bt_tp_x': bt_tp_x if bt_tp_x is not None else '',
                         'bt_sl_x': bt_sl_x if bt_sl_x is not None else ''}
            strat_rows.append(strat_row)

            # generate chart with signals
            signals = generate_breakout_signals(hist)
            plot_path = plot_support_resistance_with_signals(t, hist, signals=signals)
        except Exception as e:
            print(f"Ticker {t} error: {e}")
            continue

    df_all = pd.DataFrame(all_candidates)
    # Per-option backtest results
    df_bt_options = pd.DataFrame(option_bt_rows)
    # Per-ticker strategy metrics
    df_strat = pd.DataFrame(strat_rows).drop_duplicates(subset=['ticker'], keep='last')

    # Merge strategy metrics onto option rows (by ticker)
    if not df_bt_options.empty and not df_strat.empty:
        df_bt_report = df_bt_options.merge(df_strat, on='ticker', how='left')
    elif not df_bt_options.empty:
        df_bt_report = df_bt_options.copy()
    elif not df_strat.empty:
        df_bt_report = df_strat.copy()
    else:
        df_bt_report = pd.DataFrame()

    # If there are tickers with only strategy rows (no options), ensure they are present
    if not df_strat.empty and not df_bt_options.empty:
        tickers_with_options = set(df_bt_options['ticker'].unique())
        only_strat = df_strat[~df_strat['ticker'].isin(tickers_with_options)]
        if not only_strat.empty:
            df_bt_report = pd.concat([df_bt_report, only_strat], ignore_index=True, sort=False)

    # Save outputs
    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=['ticker','expiry','strike','dte'], keep='first')
        df_all.to_csv(f"{out_prefix}.csv", index=False)
    if not df_bt_report.empty:
        df_bt_report.to_csv(f"{out_prefix}_backtest.csv", index=False)
    return df_all, df_bt_report

# -------------------- CLI --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers_csv', type=str, default='tickers.csv',
                        help='Path to a CSV file containing tickers. If present, this takes precedence over --tickers.')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Optional: comma-separated tickers to screen (fallback if CSV not provided/found).')
    parser.add_argument('--min_oi', type=int, default=200, help='Minimum open interest to consider')
    parser.add_argument('--min_vol', type=int, default=30, help='Minimum option volume to consider')
    # Backtest parameters
    parser.add_argument('--bt_years', type=int, default=3, help='Backtest lookback period in years for underlying history')
    parser.add_argument('--bt_dte', type=int, default=7, help='DTE (days to expiry) for simulated trades')
    parser.add_argument('--bt_moneyness', type=float, default=0.05, help='Relative OTM for strike: K = S * (1 + moneyness)')
    parser.add_argument('--bt_tp_x', type=float, default=None, help='Take-profit multiple of premium (e.g., 3.0 for +200%). Leave empty for none.')
    parser.add_argument('--bt_sl_x', type=float, default=None, help='Stop-loss multiple of premium (e.g., 0.5 for -50%). Leave empty for none.')
    args = parser.parse_args()

    def load_tickers_from_csv(path):
        import os, csv, re
        if not os.path.isfile(path):
            return []
        # Accept tickers separated by commas/semicolons/whitespace. Ignore headers like 'ticker'/'symbol'.
        header_tokens = {"TICKER", "SYMBOL", "TICKERS", "SYMBOLS"}
        valid_re = re.compile(r"^[A-Z0-9.\-^]{1,15}$")
        tickers_list = []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    cell = cell.strip()
                    if not cell:
                        continue
                    for tok in re.split(r'[;,\s]+', cell):
                        if not tok:
                            continue
                        u = tok.upper()
                        if u in header_tokens:
                            continue
                        if not valid_re.match(u):
                            continue
                        if u not in tickers_list:
                            tickers_list.append(u)
        return tickers_list

    tickers = []
    if args.tickers_csv:
        tickers = load_tickers_from_csv(args.tickers_csv)
    if not tickers and args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(',') if x.strip()]
    if not tickers:
        # final fallback default list
        tickers = ['SPY','QQQ','AAPL','MSFT','NVDA','AMZN','TSLA','AMD','INTC','META','GOOG','MS','JNJ']

    # Final sanitize: remove header-like tokens and obviously invalid symbols
    import re as _re
    _header_tokens = {"TICKER", "SYMBOL", "TICKERS", "SYMBOLS"}
    _valid_re = _re.compile(r"^[A-Z0-9.\-^]{1,15}$")
    tickers = [t for t in tickers if t and t not in _header_tokens and _valid_re.match(t)]

    print('Running screener on:', tickers)
    df_res, df_bt = run_screener(
        tickers,
        min_oi=args.min_oi,
        min_vol=args.min_vol,
        out_prefix='screener_results',
        bt_years=args.bt_years,
        bt_dte=args.bt_dte,
        bt_moneyness=args.bt_moneyness,
        bt_tp_x=args.bt_tp_x,
        bt_sl_x=args.bt_sl_x,
    )

    print('\nScreener finished.')
    if not df_res.empty:
        print('Top results saved to screener_results.csv')
        print(df_res[['ticker','expiry','dte','strike','mid','openInterest','volume','impliedVol','prob_10x']].head(20).to_string(index=False))
    if not df_bt.empty:
        print('\nBacktest summary saved to screener_results_backtest.csv')
        print(df_bt.head(20).to_string(index=False))

    print('\nPlots saved to plots/ (one per ticker)')
    print('Per-ticker equity curves saved to backtests/<TICKER>_equity.csv')
