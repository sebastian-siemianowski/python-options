"""
Signal Charts — 12-month candlestick charts with SMA overlays.

Two chart categories:
  1. SIGNAL CHARTS — Strong buy/sell signals from high-conviction JSONs
     → src/data/plots/signals/{TICKER}_signal.png
  2. SMA CHARTS — Stocks trading below their 50-day SMA
     → src/data/plots/sma/{TICKER}_sma.png

Both directories are cleared at the start of each run to keep charts fresh.
Called automatically after save_high_conviction_signals() in the signal flow.
"""

import json
import os
import pathlib
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ─── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ─── Directories ─────────────────────────────────────────────────────────────
HC_BUY_DIR = os.path.join(REPO_ROOT, "src", "data", "high_conviction", "buy")
HC_SELL_DIR = os.path.join(REPO_ROOT, "src", "data", "high_conviction", "sell")
PRICE_DIR = os.path.join(REPO_ROOT, "src", "data", "prices")
CHARTS_DIR = os.path.join(REPO_ROOT, "src", "data", "plots", "signals")
SMA_CHARTS_DIR = os.path.join(REPO_ROOT, "src", "data", "plots", "sma")
INDEX_CHARTS_DIR = os.path.join(REPO_ROOT, "src", "data", "plots", "index")
TUNE_DIR = os.path.join(REPO_ROOT, "src", "data", "tune")

# ─── Index fund / ETF universe ────────────────────────────────────────────────
INDEX_FUND_TICKERS = [
    # Broad market
    ("SPY", "S&P 500 (SPY)"),
    ("QQQ", "Nasdaq 100 (QQQ)"),
    ("IWM", "Russell 2000 (IWM)"),
    ("DIA", "Dow Jones (DIA)"),
    ("VOO", "Vanguard S&P 500 (VOO)"),
    # Sector ETFs
    ("XLK", "Technology (XLK)"),
    ("XLF", "Financials (XLF)"),
    ("XLE", "Energy (XLE)"),
    ("XLV", "Healthcare (XLV)"),
    ("XLI", "Industrials (XLI)"),
    ("XLY", "Consumer Disc. (XLY)"),
    ("XLP", "Consumer Staples (XLP)"),
    ("XLU", "Utilities (XLU)"),
    ("XLB", "Materials (XLB)"),
    ("XLC", "Communication (XLC)"),
    ("XLRE", "Real Estate (XLRE)"),
    # Commodities / other
    ("GLD", "Gold (GLD)"),
    ("SLV", "Silver (SLV)"),
    ("SMH", "Semiconductors (SMH)"),
]

# ─── Chart config ─────────────────────────────────────────────────────────────
CHART_LOOKBACK_MONTHS = 12
SMA_PERIODS = (10, 20, 50, 200)
SMA_COLORS = {
    10: "#FFD700",   # Gold/Yellow
    20: "#FF8C00",   # Dark Orange
    50: "#00CED1",   # Cyan
    200: "#DA70D6",  # Orchid/Magenta
}
FIGURE_SIZE = (14, 8)
DPI = 150

# ─── Drift forecast overlay config ───────────────────────────────────────────
FORECAST_HORIZON_DAYS = 60          # Trading days forward (~3 months)
FORECAST_CI_LEVELS = (0.50, 0.80, 0.95)  # Confidence intervals (inner→outer)
FORECAST_CI_ALPHAS = (0.28, 0.16, 0.08)  # Fill opacity (inner→outer)
FORECAST_COLOR = "#00E676"          # Green — matching reversal/buy theme
FORECAST_COLOR_SELL = "#FF5252"     # Red — for sell signals
FORECAST_ENABLED = True             # Master switch

# ─── Bollinger Bands config ──────────────────────────────────────────────────
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
BOLLINGER_COLOR = "#B388FF"         # Light purple
BOLLINGER_FILL_ALPHA = 0.06
BOLLINGER_ENABLED = True

# ─── RSI config ──────────────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_COLOR = "#FFD740"               # Amber
RSI_ENABLED = True

# ─── ATR stop/take-profit config ─────────────────────────────────────────────
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0                 # 2x ATR below close for stop-loss
ATR_TARGET_MULT = 3.0               # 3x ATR above close for take-profit
ATR_STOP_COLOR = "#FF5252"          # Red dashed
ATR_TARGET_COLOR = "#69F0AE"        # Green dashed
ATR_ENABLED = True

# ─── Options max OI config ───────────────────────────────────────────────────
OPTIONS_MAX_OI_TOP_N = 3            # Show top N strikes by open interest
OPTIONS_CALL_COLOR = "#64FFDA"      # Teal for call max OI
OPTIONS_PUT_COLOR = "#FF8A80"       # Pink for put max OI
OPTIONS_OI_ENABLED = True

# ─── Multi-horizon forecast markers ──────────────────────────────────────────
FORECAST_MARKER_HORIZONS = (7, 30, 90)  # Days: 1W, 1M, 3M
FORECAST_MARKER_COLOR = "#FFAB40"   # Amber markers
FORECAST_MARKERS_ENABLED = True

# ─── SMA50 filter config ─────────────────────────────────────────────────────
SMA50_PERIOD = 50
# Skip non-equity symbols (FX, indices, futures, crypto)
SMA50_SKIP_PATTERNS = re.compile(
    r"(.*_X$|^\^|.*_F$|.*-USD$|.*\.KS$|.*\.T$|.*\.L$|.*\.PA$|.*\.DE$)",
    re.IGNORECASE,
)

# ─── Lazy imports for mplfinance (avoid import cost when not charting) ────────
_mpf = None
_plt = None


def _ensure_mpf():
    """Lazy-load mplfinance and matplotlib."""
    global _mpf, _plt
    if _mpf is None:
        try:
            import mplfinance as mpf
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend for file saving
            import matplotlib.pyplot as plt
            _mpf = mpf
            _plt = plt
        except ImportError:
            raise ImportError(
                "mplfinance is required for signal charts. "
                "Install with: pip install mplfinance"
            )


def _price_csv_path(ticker: str) -> str:
    """
    Map ticker → price CSV path, matching data_utils._price_cache_path() sanitization.
    Replaces / = : with underscore, uppercases.
    """
    safe = ticker.replace("/", "_").replace("=", "_").replace(":", "_")
    return os.path.join(PRICE_DIR, f"{safe.upper()}.csv")


# Extra months to load for SMA200 warmup (200 trading days ≈ 10 months)
SMA_WARMUP_MONTHS = 12


def _load_price_data(
    ticker: str,
    months: int = CHART_LOOKBACK_MONTHS,
    warmup_months: int = SMA_WARMUP_MONTHS,
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV price data from CSV for the last N months + warmup.

    Loads (months + warmup_months) of data so SMA200 can be computed
    on the full history. The caller trims to the display window after
    SMA computation.

    Returns a DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
    suitable for mplfinance, or None if file not found / parsing fails.
    """
    csv_path = _price_csv_path(ticker)
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date")
        df = df.sort_index()

        # Load extra data for SMA warmup
        total_months = months + warmup_months
        cutoff = datetime.now() - timedelta(days=total_months * 30)
        df = df[df.index >= cutoff]

        if len(df) < 10:
            return None

        # Ensure correct column names for mplfinance
        rename_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "open":
                rename_map[col] = "Open"
            elif lower == "high":
                rename_map[col] = "High"
            elif lower == "low":
                rename_map[col] = "Low"
            elif lower in ("close", "adj close"):
                if "Close" not in rename_map.values():
                    rename_map[col] = "Close"
            elif lower == "volume":
                rename_map[col] = "Volume"

        df = df.rename(columns=rename_map)

        # Keep only what mplfinance needs
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None

        df = df[required].copy()

        # Drop rows with NaN in OHLC (Volume NaN is OK, fill with 0)
        df["Volume"] = df["Volume"].fillna(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        if len(df) < 10:
            return None

        return df

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Kalman Drift Forecast — BMA-weighted forward projection with Student-t bands
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_drift_forecast(
    ticker: str,
    price_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> Optional[Dict]:
    """
    Compute BMA-weighted Kalman drift forecast with Student-t uncertainty bands.

    Uses tuned parameters from the tune cache to:
      1. Run each model's Kalman filter on historical returns → get terminal (mu_T, P_T)
      2. Project forward via AR(1) decay: cumulative drift at day h
      3. Accumulate forward variance with observation noise
      4. Convert to price levels with Student-t confidence intervals
      5. BMA-blend across all models with their posterior weights

    Returns dict with keys:
      - future_x: array of integer x-positions (relative to last candle)
      - median: array of median price path
      - bands: list of (ci_level, low_array, high_array) for each CI level
      - drift_annualized: float, annualized drift from best model
      - best_nu: float, effective tail thickness

    Returns None if tune cache missing or computation fails.
    """
    import numpy as np

    # ── Load tune cache ───────────────────────────────────────────────
    safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").upper()
    tune_path = os.path.join(TUNE_DIR, f"{safe_ticker}.json")
    if not os.path.exists(tune_path):
        return None

    try:
        with open(tune_path, "r") as f:
            tune_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    global_params = tune_data.get("global", {})
    model_weights = global_params.get("model_weights", {})
    models_params = global_params.get("models", {})

    if not model_weights or not models_params:
        return None

    # ── Prepare returns and volatility from price data ────────────────
    close = price_df["Close"].values.astype(np.float64)
    if len(close) < 60:
        return None

    returns = np.diff(np.log(close))
    last_price = float(close[-1])

    # Compute volatility: Garman-Klass if OHLC available, else EWMA
    try:
        open_ = price_df["Open"].values.astype(np.float64)
        high = price_df["High"].values.astype(np.float64)
        low = price_df["Low"].values.astype(np.float64)

        from calibration.realized_volatility import compute_hybrid_volatility_har
        vol_arr, _ = compute_hybrid_volatility_har(
            open_=open_[1:], high=high[1:], low=low[1:], close=close[1:],
            span=21, annualize=False, use_har=True,
        )
        # Floor volatility
        vol_floor = max(np.nanmedian(vol_arr) * 0.1, 1e-6)
        vol_arr = np.maximum(vol_arr, vol_floor)
    except Exception:
        # Fallback: EWMA volatility from returns
        vol_arr = np.full(len(returns), np.std(returns))
        ewm_var = np.zeros(len(returns))
        ewm_var[0] = returns[0] ** 2
        alpha = 2.0 / 22.0  # ~21-day EWMA
        for i in range(1, len(returns)):
            ewm_var[i] = alpha * returns[i] ** 2 + (1 - alpha) * ewm_var[i - 1]
        vol_arr = np.sqrt(np.maximum(ewm_var, 1e-12))

    last_vol = float(vol_arr[-1])

    # ── Import Kalman filters ─────────────────────────────────────────
    try:
        from models.gaussian import GaussianDriftModel
        from models.phi_student_t import PhiStudentTDriftModel
    except ImportError:
        return None

    # ── Run each model's filter and project forward ───────────────────
    # Collect per-model: (weight, mu_T, P_T, q, c, phi, nu)
    model_forecasts = []
    total_weight = 0.0

    for model_name, weight in model_weights.items():
        if weight < 1e-4:
            continue  # Skip negligible weights

        params = models_params.get(model_name, {})
        q = params.get("q")
        c = params.get("c")
        phi = params.get("phi")

        if q is None or c is None or phi is None:
            continue

        # Determine if Student-t or Gaussian
        nu = params.get("nu", None)
        is_student_t = "student_t" in model_name and nu is not None and nu > 2.0

        try:
            if is_student_t:
                mu_filt, P_filt, ll = PhiStudentTDriftModel.filter_phi(
                    returns, vol_arr, q, c, phi, nu,
                )
            else:
                mu_filt, P_filt, ll = GaussianDriftModel.filter_phi(
                    returns, vol_arr, q, c, phi,
                )
                nu = 30.0  # Gaussian approximation — wide tails for safety

            # Terminal state
            mu_T = float(mu_filt[-1])
            P_T = float(P_filt[-1])

            if not (np.isfinite(mu_T) and np.isfinite(P_T) and P_T > 0):
                continue

            model_forecasts.append({
                "weight": weight,
                "mu_T": mu_T,
                "P_T": P_T,
                "q": q,
                "c": c,
                "phi": phi,
                "nu": nu,
            })
            total_weight += weight

        except Exception:
            continue

    if not model_forecasts or total_weight < 0.01:
        return None

    # Normalize weights
    for mf in model_forecasts:
        mf["weight"] /= total_weight

    # ── BMA-blend forward projection ─────────────────────────────────
    # For each horizon h = 1..60:
    #   Cumulative drift: sum(phi^k * mu_T, k=1..h)
    #   Cumulative variance: sum(phi^{2k} * P_T + q * (1-phi^{2k})/(1-phi^2) + R, k=1..h)
    # BMA: weight-average the median and blend the variances

    from scipy.stats import t as t_dist

    H = horizon
    bma_cum_drift = np.zeros(H)
    bma_cum_var = np.zeros(H)
    bma_nu_eff = 0.0  # Weighted harmonic mean of nu for tail blending

    R_last = last_vol ** 2  # Base observation noise (without c — c is per-model)

    for mf in model_forecasts:
        w = mf["weight"]
        mu_T = mf["mu_T"]
        P_T = mf["P_T"]
        q = mf["q"]
        phi = mf["phi"]
        c = mf["c"]
        nu = mf["nu"]

        phi2 = phi * phi
        R = c * R_last  # Model-specific observation noise

        cum_drift = np.zeros(H)
        cum_var = np.zeros(H)

        drift_sum = 0.0
        var_sum = 0.0

        for h in range(H):
            k = h + 1  # 1-indexed horizon
            phi_k = phi ** k
            phi_2k = phi2 ** k

            # Drift at step k: phi^k * mu_T
            drift_sum += phi_k * mu_T

            # Variance at step k: phi^{2k}*P_T + process noise accumulation + obs noise
            if abs(1 - phi2) > 1e-10:
                P_k = phi_2k * P_T + q * (1 - phi_2k) / (1 - phi2)
            else:
                P_k = P_T + q * k
            step_var = P_k + R

            var_sum += step_var

            cum_drift[h] = drift_sum
            cum_var[h] = var_sum

        bma_cum_drift += w * cum_drift
        bma_cum_var += w * cum_var
        bma_nu_eff += w / nu  # For harmonic mean

    # Effective nu (harmonic mean — conservative: lower nu dominates)
    if bma_nu_eff > 0:
        nu_eff = min(1.0 / bma_nu_eff, 50.0)
    else:
        nu_eff = 30.0

    # ── Convert to price levels ───────────────────────────────────────
    median_prices = last_price * np.exp(bma_cum_drift)

    bands = []
    for ci in FORECAST_CI_LEVELS:
        alpha_tail = (1.0 - ci) / 2.0
        # Student-t quantile (heavier tails than Gaussian)
        z = t_dist.ppf(1.0 - alpha_tail, df=max(nu_eff, 2.1))
        sigma_cum = np.sqrt(np.maximum(bma_cum_var, 1e-20))

        lo_prices = last_price * np.exp(bma_cum_drift - z * sigma_cum)
        hi_prices = last_price * np.exp(bma_cum_drift + z * sigma_cum)

        bands.append((ci, lo_prices, hi_prices))

    # ── Annualized drift for display ──────────────────────────────────
    best_mu = model_forecasts[0]["mu_T"]
    for mf in model_forecasts:
        if mf["weight"] > model_forecasts[0]["weight"]:
            best_mu = mf["mu_T"]
    drift_ann = best_mu * 252  # daily → annualized

    # x-positions: integers starting from len(price_df) (right after last candle)
    n_candles = len(price_df)
    future_x = np.arange(n_candles, n_candles + H)

    return {
        "future_x": future_x,
        "median": median_prices,
        "bands": bands,
        "drift_annualized": drift_ann,
        "best_nu": nu_eff,
    }


def _get_signal_tickers() -> List[Dict]:
    """
    Read high-conviction signal JSONs — one entry per unique ticker.

    When a ticker has signals at multiple horizons (1d, 3d, 7d), keeps the
    longest horizon (most informative). When a ticker appears in both buy
    and sell, the one with the higher absolute expected return wins.

    Returns list of dicts sorted by ticker:
      {"ticker": str, "asset_label": str, "signal_type": "STRONG_BUY"|"STRONG_SELL",
       "horizon_days": int, "exp_ret": float, "probability": float,
       "sector": str, "signal_strength": float, "options_chain": dict|None}
    """
    best: Dict[str, Dict] = {}

    for directory, default_type in [(HC_BUY_DIR, "STRONG_BUY"), (HC_SELL_DIR, "STRONG_SELL")]:
        if not os.path.isdir(directory):
            continue

        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".json") or fname == "manifest.json":
                continue

            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r") as f:
                    sig = json.load(f)

                ticker = sig.get("ticker", "")
                if not ticker:
                    continue

                asset_label = sig.get("asset_label", ticker)
                signal_type = sig.get("signal_type", default_type)
                horizon = sig.get("horizon_days", 0)
                exp_ret = abs(sig.get("expected_return_pct", 0.0))
                prob = sig.get("probability_up", 0.5) if "BUY" in signal_type else sig.get("probability_down", 0.5)
                sector = sig.get("sector", "")
                signal_strength = sig.get("signal_strength", 0.0)
                options_chain = sig.get("options_chain", None)

                entry = {
                    "ticker": ticker,
                    "asset_label": asset_label,
                    "signal_type": signal_type,
                    "horizon_days": horizon,
                    "exp_ret": exp_ret,
                    "probability": prob,
                    "sector": sector,
                    "signal_strength": signal_strength,
                    "options_chain": options_chain,
                }

                # Keep longest horizon per ticker; break ties by exp_ret
                if ticker in best:
                    prev = best[ticker]
                    if horizon > prev["horizon_days"] or (
                        horizon == prev["horizon_days"] and exp_ret > prev["exp_ret"]
                    ):
                        best[ticker] = entry
                else:
                    best[ticker] = entry

            except (json.JSONDecodeError, OSError):
                continue

    result = list(best.values())
    result.sort(key=lambda s: s["ticker"])
    return result


def _render_chart(
    ticker: str,
    asset_label: str,
    signal_type: str,
    price_df: pd.DataFrame,
    output_path: str,
    sector: str = "",
    probability: float = 0.0,
    exp_ret: float = 0.0,
    signal_strength: float = 0.0,
    options_chain: Optional[Dict] = None,
) -> bool:
    """
    Render a candlestick chart with SMA overlays, Bollinger Bands, RSI panel,
    ATR stop/take-profit zones, drift forecast, options max OI, and save as PNG.

    Returns True on success, False on failure.
    """
    _ensure_mpf()
    import numpy as np

    try:
        is_buy = "BUY" in signal_type
        is_sma_warning = "BELOW_SMA" in signal_type
        is_reversal = signal_type == "REVERSAL_UP"
        is_index = signal_type == "INDEX_FUND"

        if is_index:
            signal_color = "#42A5F5"   # Blue — index/ETF
            signal_label = "INDEX"
            signal_arrow = "◆"
        elif is_reversal:
            signal_color = "#00E676"   # Green — uptrend reversal
            signal_label = "↑ REVERSAL"
            signal_arrow = "↑"
        elif is_sma_warning:
            signal_color = "#FF6D00" if signal_type == "BELOW_SMA50" else "#FF1744"
            signal_label = "BELOW SMA50 + SMA200" if "200" in signal_type else "BELOW SMA50"
            signal_arrow = "▼▼" if "200" in signal_type else "▼"
        elif is_buy:
            signal_color = "#00E676"
            signal_label = "STRONG BUY"
            signal_arrow = "▲▲"
        else:
            signal_color = "#FF1744"
            signal_label = "STRONG SELL"
            signal_arrow = "▼▼"

        # ── Custom style ─────────────────────────────────────────────
        mc = _mpf.make_marketcolors(
            up="#26A69A",        # Teal green for up candles
            down="#EF5350",      # Red for down candles
            edge="inherit",
            wick="inherit",
            volume={"up": "#26A69A", "down": "#EF5350"},
            ohlc="inherit",
        )
        style = _mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#1a1a2e",
            edgecolor="#16213e",
            figcolor="#0f0f23",
            gridcolor="#1e2d4a",
            gridstyle="--",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#b0b0b0",
                "xtick.color": "#808080",
                "ytick.color": "#808080",
            },
        )

        # ── SMA overlays ─────────────────────────────────────────────
        sma_lines = {}
        for p in SMA_PERIODS:
            if len(price_df) >= p:
                sma_lines[p] = price_df["Close"].rolling(window=p).mean()

        # Trim to display window (last CHART_LOOKBACK_MONTHS months)
        display_cutoff = datetime.now() - timedelta(days=CHART_LOOKBACK_MONTHS * 30)
        display_df = price_df[price_df.index >= display_cutoff].copy()
        if len(display_df) < 10:
            display_df = price_df  # fallback

        # Trim SMA series to match display window
        display_smas = {}
        for p, sma_series in sma_lines.items():
            trimmed = sma_series.reindex(display_df.index)
            if trimmed.notna().any():
                display_smas[p] = trimmed

        # Build mplfinance addplot list for pre-computed SMAs
        addplots = []
        mav_periods_used = []
        for p in SMA_PERIODS:
            if p in display_smas:
                addplots.append(
                    _mpf.make_addplot(
                        display_smas[p],
                        color=SMA_COLORS[p],
                        width=1.5,
                        panel=0,
                    )
                )
                mav_periods_used.append(p)

        # ── Bollinger Bands (20-day, 2σ) ─────────────────────────────
        bb_upper = None
        bb_lower = None
        if BOLLINGER_ENABLED and len(price_df) >= BOLLINGER_PERIOD:
            bb_mid = price_df["Close"].rolling(window=BOLLINGER_PERIOD).mean()
            bb_std = price_df["Close"].rolling(window=BOLLINGER_PERIOD).std()
            bb_upper_full = bb_mid + BOLLINGER_STD * bb_std
            bb_lower_full = bb_mid - BOLLINGER_STD * bb_std
            bb_upper = bb_upper_full.reindex(display_df.index)
            bb_lower = bb_lower_full.reindex(display_df.index)
            if bb_upper.notna().any():
                addplots.append(_mpf.make_addplot(
                    bb_upper, color=BOLLINGER_COLOR, width=0.8,
                    linestyle="--", panel=0, alpha=0.5,
                ))
                addplots.append(_mpf.make_addplot(
                    bb_lower, color=BOLLINGER_COLOR, width=0.8,
                    linestyle="--", panel=0, alpha=0.5,
                ))

        # ── RSI(14) ──────────────────────────────────────────────────
        rsi_series = None
        use_rsi_panel = RSI_ENABLED and len(price_df) > RSI_PERIOD + 1
        if use_rsi_panel:
            delta = price_df["Close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=RSI_PERIOD).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_full = 100.0 - (100.0 / (1.0 + rs))
            rsi_series = rsi_full.reindex(display_df.index)
            if rsi_series.notna().any():
                addplots.append(_mpf.make_addplot(
                    rsi_series, color=RSI_COLOR, width=1.2,
                    panel=2, ylabel="RSI",
                ))
                # Overbought/oversold reference lines
                rsi_ob = pd.Series(RSI_OVERBOUGHT, index=display_df.index, dtype=float)
                rsi_os = pd.Series(RSI_OVERSOLD, index=display_df.index, dtype=float)
                addplots.append(_mpf.make_addplot(
                    rsi_ob, color="#FF5252", width=0.6,
                    linestyle="--", panel=2, alpha=0.5,
                ))
                addplots.append(_mpf.make_addplot(
                    rsi_os, color="#69F0AE", width=0.6,
                    linestyle="--", panel=2, alpha=0.5,
                ))

        # ── Title with sector subtitle ────────────────────────────────
        title_main = f"{asset_label}  {signal_arrow} {signal_label}"
        if sector:
            title_main += f"\n{sector}"

        # ── Panel ratios: price, volume, RSI (if enabled) ─────────────
        if use_rsi_panel and rsi_series is not None and rsi_series.notna().any():
            panel_ratios = (5, 1, 1.5)
        else:
            panel_ratios = (4, 1)

        fig, axes = _mpf.plot(
            display_df,
            type="candle",
            style=style,
            title=title_main,
            volume=True,
            addplot=addplots if addplots else None,
            figsize=FIGURE_SIZE,
            returnfig=True,
            panel_ratios=panel_ratios,
            tight_layout=True,
            warn_too_much_data=9999,
        )

        # ── Style the title ──────────────────────────────────────────
        ax_main = axes[0]
        ax_main.set_title(title_main, fontsize=14, fontweight="bold", color=signal_color, pad=15)

        # ── Bollinger Band fill ───────────────────────────────────────
        if BOLLINGER_ENABLED and bb_upper is not None and bb_lower is not None:
            x_idx = np.arange(len(display_df))
            bb_up_vals = bb_upper.values
            bb_lo_vals = bb_lower.values
            valid = ~(np.isnan(bb_up_vals) | np.isnan(bb_lo_vals))
            if valid.any():
                ax_main.fill_between(
                    x_idx, bb_lo_vals, bb_up_vals,
                    where=valid, alpha=BOLLINGER_FILL_ALPHA,
                    color=BOLLINGER_COLOR, linewidth=0, zorder=0,
                )

        # ── Current price horizontal line ─────────────────────────────
        last_close = float(display_df["Close"].iloc[-1])
        ax_main.axhline(
            y=last_close, color="#FFFFFF", linewidth=0.6,
            linestyle=":", alpha=0.4, zorder=1,
        )
        ax_main.text(
            len(display_df) + 1, last_close,
            f"${last_close:,.2f}",
            fontsize=7, color="#FFFFFF", alpha=0.6,
            va="center", ha="left",
        )

        # ── ATR stop-loss / take-profit zones ─────────────────────────
        atr_value = None
        if ATR_ENABLED and len(display_df) >= ATR_PERIOD + 1:
            high = display_df["High"].values
            low = display_df["Low"].values
            close = display_df["Close"].values
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                ),
            )
            if len(tr) >= ATR_PERIOD:
                atr_value = float(np.mean(tr[-ATR_PERIOD:]))
                stop_level = last_close - ATR_STOP_MULT * atr_value
                target_level = last_close + ATR_TARGET_MULT * atr_value

                # For sell signals, flip stop/target
                if not is_buy and not is_index and not is_reversal:
                    stop_level = last_close + ATR_STOP_MULT * atr_value
                    target_level = last_close - ATR_TARGET_MULT * atr_value

                ax_main.axhline(
                    y=stop_level, color=ATR_STOP_COLOR, linewidth=0.8,
                    linestyle="--", alpha=0.5, zorder=1,
                )
                ax_main.axhline(
                    y=target_level, color=ATR_TARGET_COLOR, linewidth=0.8,
                    linestyle="--", alpha=0.5, zorder=1,
                )
                # Labels on right edge
                ax_main.text(
                    len(display_df) + 1, stop_level,
                    f"Stop ${stop_level:,.0f}",
                    fontsize=6, color=ATR_STOP_COLOR, alpha=0.7,
                    va="center", ha="left",
                )
                ax_main.text(
                    len(display_df) + 1, target_level,
                    f"Target ${target_level:,.0f}",
                    fontsize=6, color=ATR_TARGET_COLOR, alpha=0.7,
                    va="center", ha="left",
                )

        # ── Options max OI strike levels ──────────────────────────────
        if OPTIONS_OI_ENABLED and options_chain is not None:
            try:
                options_list = options_chain.get("options", [])
                if options_list:
                    # Separate calls and puts
                    calls = [o for o in options_list if o.get("type") == "call"]
                    puts = [o for o in options_list if o.get("type") == "put"]

                    # Sort by open interest descending
                    calls.sort(key=lambda x: x.get("open_interest", 0), reverse=True)
                    puts.sort(key=lambda x: x.get("open_interest", 0), reverse=True)

                    # Draw top N call strikes
                    for opt in calls[:OPTIONS_MAX_OI_TOP_N]:
                        oi = opt.get("open_interest", 0)
                        strike = opt.get("strike", 0)
                        if oi > 0 and strike > 0:
                            ax_main.axhline(
                                y=strike, color=OPTIONS_CALL_COLOR,
                                linewidth=0.5, linestyle=":", alpha=0.4, zorder=0,
                            )
                            ax_main.text(
                                0, strike, f"C{strike:.0f} OI:{oi:,}",
                                fontsize=5, color=OPTIONS_CALL_COLOR, alpha=0.6,
                                va="bottom", ha="left",
                            )

                    # Draw top N put strikes
                    for opt in puts[:OPTIONS_MAX_OI_TOP_N]:
                        oi = opt.get("open_interest", 0)
                        strike = opt.get("strike", 0)
                        if oi > 0 and strike > 0:
                            ax_main.axhline(
                                y=strike, color=OPTIONS_PUT_COLOR,
                                linewidth=0.5, linestyle=":", alpha=0.4, zorder=0,
                            )
                            ax_main.text(
                                0, strike, f"P{strike:.0f} OI:{oi:,}",
                                fontsize=5, color=OPTIONS_PUT_COLOR, alpha=0.6,
                                va="bottom", ha="left",
                            )
            except Exception:
                pass  # Options overlay is best-effort

        # ── RSI overbought/oversold zone shading ──────────────────────
        if use_rsi_panel and rsi_series is not None and rsi_series.notna().any():
            # Find the RSI axis (panel 2)
            try:
                ax_rsi = axes[4] if len(axes) > 4 else None
                if ax_rsi is not None:
                    ax_rsi.axhspan(RSI_OVERBOUGHT, 100, alpha=0.05, color="#FF5252")
                    ax_rsi.axhspan(0, RSI_OVERSOLD, alpha=0.05, color="#69F0AE")
                    ax_rsi.set_ylim(0, 100)
            except (IndexError, Exception):
                pass

        # ── Volume relative coloring ──────────────────────────────────
        try:
            ax_vol = axes[2] if len(axes) > 2 else None
            if ax_vol is not None:
                vol = display_df["Volume"].values
                avg_vol = np.mean(vol[vol > 0]) if np.any(vol > 0) else 1
                for i, bar in enumerate(ax_vol.patches):
                    if i < len(vol):
                        ratio = vol[i] / avg_vol if avg_vol > 0 else 1
                        if ratio > 2.0:
                            bar.set_alpha(1.0)
                            bar.set_edgecolor("#FFFFFF")
                            bar.set_linewidth(0.5)
                        elif ratio > 1.5:
                            bar.set_alpha(0.85)
                        else:
                            bar.set_alpha(0.5)
        except (IndexError, Exception):
            pass

        # ── SMA Legend ────────────────────────────────────────────────
        from matplotlib.lines import Line2D
        legend_handles = []
        if mav_periods_used:
            legend_handles = [
                Line2D([0], [0], color=SMA_COLORS[p], linewidth=1.5, label=f"SMA {p}")
                for p in mav_periods_used
            ]
        if BOLLINGER_ENABLED and bb_upper is not None:
            legend_handles.append(
                Line2D([0], [0], color=BOLLINGER_COLOR, linewidth=0.8,
                       linestyle="--", alpha=0.5,
                       label=f"BB({BOLLINGER_PERIOD}, {BOLLINGER_STD:.0f}σ)")
            )
        if ATR_ENABLED and atr_value is not None:
            legend_handles.append(
                Line2D([0], [0], color=ATR_STOP_COLOR, linewidth=0.8,
                       linestyle="--", label=f"ATR Stop ({ATR_STOP_MULT:.0f}x)")
            )
            legend_handles.append(
                Line2D([0], [0], color=ATR_TARGET_COLOR, linewidth=0.8,
                       linestyle="--", label=f"ATR Target ({ATR_TARGET_MULT:.0f}x)")
            )

        # ── Drift forecast overlay (ALL signal types) ─────────────────
        forecast_info = None
        fc_color = FORECAST_COLOR if is_buy or is_reversal or is_index else FORECAST_COLOR_SELL
        if FORECAST_ENABLED:
            try:
                fc = _compute_drift_forecast(ticker, display_df)
                if fc is not None:
                    forecast_info = fc
                    future_x = fc["future_x"]
                    median_line = fc["median"]

                    # Draw confidence bands (outer → inner so inner overlays)
                    for ci_level, ci_alpha, (ci, lo, hi) in zip(
                        reversed(FORECAST_CI_LEVELS),
                        reversed(FORECAST_CI_ALPHAS),
                        reversed(fc["bands"]),
                    ):
                        ax_main.fill_between(
                            future_x, lo, hi,
                            alpha=ci_alpha,
                            color=fc_color,
                            linewidth=0,
                            zorder=1,
                        )

                    # Draw median drift line
                    ax_main.plot(
                        future_x, median_line,
                        color=fc_color,
                        linewidth=1.8,
                        linestyle="--",
                        alpha=0.9,
                        zorder=2,
                    )

                    # Connect last candle to forecast start
                    last_x = len(display_df) - 1
                    ax_main.plot(
                        [last_x, future_x[0]],
                        [last_close, median_line[0]],
                        color=fc_color,
                        linewidth=1.2,
                        linestyle="--",
                        alpha=0.6,
                        zorder=2,
                    )

                    # ── Multi-horizon forecast markers (7D/30D/90D) ───────
                    if FORECAST_MARKERS_ENABLED:
                        for h_days in FORECAST_MARKER_HORIZONS:
                            if h_days <= len(median_line):
                                h_idx = h_days - 1
                                marker_x = future_x[h_idx]
                                marker_y = float(median_line[h_idx])
                                pct_chg = ((marker_y / last_close) - 1.0) * 100.0
                                sign = "+" if pct_chg >= 0 else ""

                                ax_main.plot(
                                    marker_x, marker_y, "D",
                                    color=FORECAST_MARKER_COLOR,
                                    markersize=5, zorder=3,
                                )
                                ax_main.annotate(
                                    f"{h_days}D: {sign}{pct_chg:.1f}%",
                                    xy=(marker_x, marker_y),
                                    xytext=(5, 8),
                                    textcoords="offset points",
                                    fontsize=6, color=FORECAST_MARKER_COLOR,
                                    fontweight="bold", alpha=0.9,
                                )

                    # Extend x-axis to show forecast area
                    ax_main.set_xlim(right=future_x[-1] + 5)

                    # Add forecast to legend
                    drift_pct = fc["drift_annualized"] * 100
                    drift_sign = "+" if drift_pct >= 0 else ""
                    legend_handles.append(
                        Line2D([0], [0], color=fc_color, linewidth=1.5,
                               linestyle="--",
                               label=f"60d Kalman Drift ({drift_sign}{drift_pct:.0f}%/yr)")
                    )
                    legend_handles.append(
                        Line2D([0], [0], color=fc_color, linewidth=6,
                               alpha=0.25, label="50/80/95% CI (Student-t)")
                    )

            except Exception:
                pass  # Forecast failed — chart renders without overlay

        # Show legend
        if legend_handles:
            ax_main.legend(
                handles=legend_handles,
                loc="upper left",
                fontsize=7,
                framealpha=0.7,
                facecolor="#1a1a2e",
                edgecolor="#333",
                labelcolor="#b0b0b0",
                ncol=2 if len(legend_handles) > 6 else 1,
            )

        # ── Signal badge (top-right) ──────────────────────────────────
        badge_lines = [signal_label]
        if probability > 0.01:
            badge_lines.append(f"P={probability:.0%}")
        if exp_ret > 0.01:
            sign = "+" if is_buy else "-"
            badge_lines.append(f"E[r]={sign}{exp_ret:.1f}%")
        if signal_strength > 0.01:
            badge_lines.append(f"Str={signal_strength:.2f}")
        if forecast_info is not None:
            drift_pct = forecast_info["drift_annualized"] * 100
            drift_sign = "+" if drift_pct >= 0 else ""
            badge_lines.append(f"{drift_sign}{drift_pct:.0f}%/yr drift")

        badge_text = "\n".join(badge_lines)

        ax_main.text(
            0.99, 0.97, badge_text,
            transform=ax_main.transAxes,
            fontsize=9, fontweight="bold",
            color=signal_color,
            ha="right", va="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#0f0f23",
                edgecolor=signal_color,
                alpha=0.9,
            ),
        )

        # ── PIT calibration grade badge (top-left, below legend) ──────
        try:
            safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").upper()
            tune_path = os.path.join(TUNE_DIR, f"{safe_ticker}.json")
            if os.path.exists(tune_path):
                with open(tune_path, "r") as f:
                    tune_data = json.load(f)
                g = tune_data.get("global", {})
                pit_grade = g.get("pit_calibration_grade", None)
                best_model = g.get("best_model", "")
                regime_label = ""

                # Derive regime from volatility + trend structure
                if best_model:
                    regime_label = best_model.replace("_momentum", "").replace("phi_", "φ-")

                if pit_grade or regime_label:
                    info_lines = []
                    if pit_grade:
                        grade_colors = {"A": "#69F0AE", "B": "#FFD740", "C": "#FF8A65", "D": "#FF5252", "F": "#FF1744"}
                        grade_color = grade_colors.get(str(pit_grade)[:1], "#808080")
                        info_lines.append(f"PIT: {pit_grade}")
                    if regime_label:
                        info_lines.append(f"Model: {regime_label}")

                    info_text = "\n".join(info_lines)
                    ax_main.text(
                        0.01, 0.02, info_text,
                        transform=ax_main.transAxes,
                        fontsize=7,
                        color=grade_color if pit_grade else "#808080",
                        ha="left", va="bottom",
                        alpha=0.7,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="#0f0f23",
                            edgecolor="#333",
                            alpha=0.6,
                        ),
                    )
        except Exception:
            pass  # PIT grade is best-effort

        # ── Save ──────────────────────────────────────────────────────
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="#0f0f23")
        _plt.close(fig)
        return True

    except Exception:
        return False


def _clear_chart_dirs():
    """Clear signal, SMA, and index chart directories for fresh output."""
    for chart_dir in [CHARTS_DIR, SMA_CHARTS_DIR, INDEX_CHARTS_DIR]:
        os.makedirs(chart_dir, exist_ok=True)
        for old_file in pathlib.Path(chart_dir).glob("*.png"):
            try:
                old_file.unlink()
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Multiprocessing — chart workers run in separate processes for speed
# ═══════════════════════════════════════════════════════════════════════════════

def _pool_initializer():
    """Initialize matplotlib Agg backend in each worker process."""
    import matplotlib
    matplotlib.use("Agg")


def _chart_worker(task: Dict) -> Dict:
    """
    Worker function for parallel chart generation.

    Runs in a subprocess: loads price data, renders chart, returns result.
    All arguments and return values must be picklable (no DataFrames).

    Args: dict with ticker, asset_label, signal_type, output_path (+ optional display fields)
    Returns: same dict with added "status" key ("generated"|"skipped"|"error")
    """
    ticker = task["ticker"]
    asset_label = task["asset_label"]
    signal_type = task["signal_type"]
    output_path = task["output_path"]
    sector = task.get("sector", "")
    probability = task.get("probability", 0.0)
    exp_ret = task.get("exp_ret", 0.0)
    signal_strength = task.get("signal_strength", 0.0)
    options_chain = task.get("options_chain", None)

    price_df = _load_price_data(ticker)
    if price_df is None:
        return {**task, "status": "skipped"}

    try:
        success = _render_chart(
            ticker, asset_label, signal_type, price_df, output_path,
            sector=sector,
            probability=probability,
            exp_ret=exp_ret,
            signal_strength=signal_strength,
            options_chain=options_chain,
        )
        return {**task, "status": "generated" if success else "error"}
    except Exception:
        return {**task, "status": "error"}


def generate_signal_charts(quiet: bool = False) -> Dict:
    """
    Generate candlestick charts for all strong buy/sell signals.

    Reads from src/data/high_conviction/{buy,sell}/ JSON files,
    loads 12 months of price data from src/data/prices/,
    and saves charts to src/data/plots/signals/.

    Both plots/signals/ and plots/sma/ are cleared at the start.

    Returns: {"generated": int, "skipped": int, "errors": int}
    """
    result = {"generated": 0, "skipped": 0, "errors": 0}

    # Always clear both chart directories for fresh output
    _clear_chart_dirs()

    # Collect signal tickers
    signals = _get_signal_tickers()
    if not signals:
        return result

    # Lazy-load mplfinance (verify availability before spawning workers)
    try:
        _ensure_mpf()
    except ImportError:
        if not quiet:
            try:
                from rich.console import Console
                Console().print("[yellow]⚠ mplfinance not installed — skipping signal charts[/yellow]")
            except Exception:
                pass
        result["skipped"] = len(signals)
        return result

    # Progress display
    console = None
    if not quiet:
        try:
            from rich.console import Console
            from rich.text import Text
            console = Console()

            header = Text()
            header.append("\n    ")
            header.append("📊 ", style="")
            header.append("Signal Charts", style="bold cyan")
            header.append(f"  ({len(signals)} charts)", style="dim")
            console.print(header)
        except Exception:
            pass

    # Build task list — one chart per unique ticker
    tasks = []
    for sig in signals:
        ticker = sig["ticker"]
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").replace("^", "_")
        tasks.append({
            "ticker": ticker,
            "asset_label": sig["asset_label"],
            "signal_type": sig["signal_type"],
            "output_path": os.path.join(CHARTS_DIR, f"{safe_ticker}_signal.png"),
            "sector": sig.get("sector", ""),
            "probability": sig.get("probability", 0.0),
            "exp_ret": sig.get("exp_ret", 0.0),
            "signal_strength": sig.get("signal_strength", 0.0),
            "options_chain": sig.get("options_chain", None),
        })

    # Generate charts in parallel using multiple processes
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(os.cpu_count() or 4, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_initializer) as pool:
        futures = {pool.submit(_chart_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception:
                result["errors"] += 1
                continue

            if res["status"] == "generated":
                result["generated"] += 1
                if console:
                    signal_icon = "▲" if "BUY" in res["signal_type"] else "▼"
                    color = "green" if "BUY" in res["signal_type"] else "red"
                    console.print(f"    [dim]  ✓[/dim] [{color}]{signal_icon}[/{color}] [dim]{res['ticker']}[/dim]")
            elif res["status"] == "skipped":
                result["skipped"] += 1
                if console:
                    console.print(f"    [dim]  skip {res['ticker']} (no price data)[/dim]")
            else:
                result["errors"] += 1
                if console:
                    console.print(f"    [dim]  ✗ {res['ticker']} (chart error)[/dim]")

    # Summary
    if console and result["generated"] > 0:
        from rich.text import Text
        summary = Text()
        summary.append("    ")
        summary.append(f"→ {result['generated']} charts", style="bold green")
        summary.append(f" saved to src/data/plots/signals/", style="dim")
        console.print(summary)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BELOW SMA50 DETECTION — Scan price universe for stocks under their 50-day SMA
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_ticker_from_label(asset_label: str) -> str:
    """Extract ticker from asset label like 'Apple Inc. (AAPL)' → 'AAPL'."""
    match = re.search(r'\(([A-Z0-9.=^-]+)\)', asset_label)
    return match.group(1) if match else asset_label.split()[0] if asset_label else ""


def find_below_sma50_stocks(
    summary_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Find stocks trading below their 50-day SMA.

    If summary_rows is provided, only checks tickers from those rows.
    Otherwise, scans all equity CSVs in src/data/prices/.

    Returns list of dicts sorted by deviation (most below SMA50 first):
      {"ticker": str, "asset_label": str, "sector": str,
       "close": float, "sma50": float, "deviation_pct": float,
       "sma200": float|None, "below_sma200": bool}
    """
    results = []

    if summary_rows:
        # Use tickers from the signal run
        ticker_info = {}
        for row in summary_rows:
            asset_label = row.get("asset_label", "")
            sector = row.get("sector", "Other")
            ticker = _extract_ticker_from_label(asset_label)
            if ticker and ticker not in ticker_info:
                ticker_info[ticker] = {"asset_label": asset_label, "sector": sector}

        candidates = list(ticker_info.items())
    else:
        # Scan all price CSVs
        candidates = []
        if not os.path.isdir(PRICE_DIR):
            return results
        for fname in sorted(os.listdir(PRICE_DIR)):
            if not fname.endswith(".csv"):
                continue
            ticker = fname.replace(".csv", "")
            # Skip non-equity symbols
            if SMA50_SKIP_PATTERNS.match(ticker):
                continue
            candidates.append((ticker, {"asset_label": ticker, "sector": ""}))

    for ticker, info in candidates:
        # Skip non-equity symbols when from summary_rows too
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_")
        if SMA50_SKIP_PATTERNS.match(safe_ticker):
            continue

        csv_path = _price_csv_path(ticker)
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.sort_values("Date")

            # Need column names
            close_col = None
            for col in df.columns:
                if col.lower().strip() in ("close", "adj close"):
                    close_col = col
                    break
            if close_col is None:
                continue

            closes = df[close_col].dropna()
            if len(closes) < SMA50_PERIOD:
                continue

            current_close = closes.iloc[-1]
            sma50 = closes.iloc[-SMA50_PERIOD:].mean()

            if current_close >= sma50:
                continue  # Not below SMA50

            deviation_pct = ((current_close - sma50) / sma50) * 100.0

            # Also compute SMA200 if available
            sma200 = None
            below_sma200 = False
            if len(closes) >= 200:
                sma200 = closes.iloc[-200:].mean()
                below_sma200 = current_close < sma200

            # ── Uptrend reversal detection ────────────────────────────
            # Stock is below SMA50 but showing signs of turning up:
            #   1. Close > SMA10 (short-term uptrend)
            #   2. 5-day return is positive
            #   3. SMA10 slope is positive (10-day MA rising)
            reversing_up = False
            sma10 = None
            ret_5d = None
            if len(closes) >= 10:
                sma10 = closes.iloc[-10:].mean()
                sma10_prev = closes.iloc[-11:-1].mean() if len(closes) >= 11 else sma10
                sma10_rising = sma10 > sma10_prev
                above_sma10 = current_close > sma10
                ret_5d = ((current_close / closes.iloc[-6]) - 1.0) * 100.0 if len(closes) >= 6 else 0.0
                reversing_up = above_sma10 and sma10_rising and ret_5d > 0

            results.append({
                "ticker": ticker,
                "asset_label": info["asset_label"],
                "sector": info.get("sector", ""),
                "close": round(current_close, 2),
                "sma50": round(sma50, 2),
                "sma10": round(sma10, 2) if sma10 is not None else None,
                "deviation_pct": round(deviation_pct, 2),
                "sma200": round(sma200, 2) if sma200 is not None else None,
                "below_sma200": below_sma200,
                "reversing_up": reversing_up,
                "ret_5d": round(ret_5d, 2) if ret_5d is not None else None,
            })

        except Exception:
            continue

    # Sort by deviation (most below SMA50 first)
    results.sort(key=lambda x: x["deviation_pct"])
    return results


def render_below_sma50_table(
    below_sma50: List[Dict],
    quiet: bool = False,
) -> None:
    """
    Render a Rich table of stocks trading below their 50-day SMA.
    """
    if not below_sma50 or quiet:
        return

    # Filter to only reversing-up stocks for display
    reversing_stocks = [s for s in below_sma50 if s.get("reversing_up")]
    if not reversing_stocks:
        return

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich import box
    except ImportError:
        return

    console = Console()

    # Header
    console.print()
    header = Text()
    header.append("  ↑ SMA50 REVERSALS", style="bold green")
    header.append(f"  ({len(reversing_stocks)} of {len(below_sma50)} below SMA50)", style="dim")
    console.print(Panel(header, box=box.HEAVY, border_style="green", expand=False))

    # Table
    table = Table(
        box=box.ROUNDED,
        border_style="green",
        header_style="bold white on dark_green",
        row_styles=["", "on grey7"],
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Asset", width=50, no_wrap=True)
    table.add_column("Sector", width=28, no_wrap=True)
    table.add_column("Close", justify="right", width=10)
    table.add_column("SMA50", justify="right", width=10)
    table.add_column("Dev %", justify="right", width=10)
    table.add_column("SMA200", justify="right", width=10)
    table.add_column("Status", width=14)
    table.add_column("5d Return", width=12)

    for stock in reversing_stocks[:40]:  # Cap at 40
        # Deviation color: deeper red for more deviation
        dev = stock["deviation_pct"]
        if dev <= -20:
            dev_style = "bold red"
        elif dev <= -10:
            dev_style = "red"
        elif dev <= -5:
            dev_style = "bright_red"
        else:
            dev_style = "yellow"

        # Status badge
        if stock["below_sma200"]:
            status = Text("▼▼ WEAK", style="bold red")
        else:
            status = Text("▼ CAUTION", style="yellow")

        # 5-day return
        ret_5d = stock.get("ret_5d", 0)
        ret_text = Text(f"↑ +{ret_5d:.1f}%", style="bold green")

        sma200_str = f"${stock['sma200']:,.2f}" if stock["sma200"] is not None else "—"

        table.add_row(
            stock["asset_label"],
            stock["sector"],
            f"${stock['close']:,.2f}",
            f"${stock['sma50']:,.2f}",
            Text(f"{dev:+.1f}%", style=dev_style),
            sma200_str,
            status,
            ret_text,
        )

    console.print(table)

    # Footer
    below_both = sum(1 for s in reversing_stocks if s["below_sma200"])
    footer = Text()
    footer.append("    ")
    footer.append(f"Reversing: {len(reversing_stocks)}", style="bold green")
    footer.append("  ·  ", style="dim")
    footer.append(f"Also below SMA200: {below_both}", style="red")
    footer.append("  ·  ", style="dim")
    footer.append(f"Total below SMA50: {len(below_sma50)}", style="dim")
    console.print(footer)


def generate_sma_charts(
    below_sma50: List[Dict],
    quiet: bool = False,
) -> Dict:
    """
    Generate candlestick charts for stocks below SMA50 that show uptrend reversal.

    Only charts stocks where:
      - Close > SMA10 (short-term recovery)
      - SMA10 is rising (momentum turning)
      - 5-day return is positive

    Saves to src/data/plots/sma/{TICKER}_sma.png

    Returns: {"generated": int, "skipped": int, "errors": int, "reversing": int}
    """
    result = {"generated": 0, "skipped": 0, "errors": 0, "reversing": 0}

    if not below_sma50:
        return result

    # Filter to only reversing-up stocks for charting
    reversing_stocks = [s for s in below_sma50 if s.get("reversing_up")]
    result["reversing"] = len(reversing_stocks)

    if not reversing_stocks:
        return result

    # Clear and recreate SMA chart directory for fresh output
    os.makedirs(SMA_CHARTS_DIR, exist_ok=True)
    for old_file in pathlib.Path(SMA_CHARTS_DIR).glob("*.png"):
        try:
            old_file.unlink()
        except OSError:
            pass

    # Lazy-load mplfinance (verify availability before spawning workers)
    try:
        _ensure_mpf()
    except ImportError:
        if not quiet:
            try:
                from rich.console import Console
                Console().print("[yellow]⚠ mplfinance not installed — skipping SMA charts[/yellow]")
            except Exception:
                pass
        result["skipped"] = len(reversing_stocks)
        return result

    # Progress display
    console = None
    if not quiet:
        try:
            from rich.console import Console
            from rich.text import Text
            console = Console()

            header = Text()
            header.append("\n    ")
            header.append("📈 ", style="")
            header.append("SMA Reversal Charts", style="bold green")
            header.append(f"  ({len(reversing_stocks)} reversing up)", style="dim")
            console.print(header)
        except Exception:
            pass

    # Build task list with display metadata
    tasks = []
    for stock in reversing_stocks:
        ticker = stock["ticker"]
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").replace("^", "_")
        tasks.append({
            "ticker": ticker,
            "asset_label": stock["asset_label"],
            "signal_type": "REVERSAL_UP",
            "output_path": os.path.join(SMA_CHARTS_DIR, f"{safe_ticker}_sma.png"),
            "deviation_pct": stock["deviation_pct"],
            "ret_5d": stock.get("ret_5d", 0),
        })

    # Generate charts in parallel using multiple processes
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(os.cpu_count() or 4, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_initializer) as pool:
        futures = {pool.submit(_chart_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception:
                result["errors"] += 1
                continue

            if res["status"] == "generated":
                result["generated"] += 1
                if console:
                    dev = res["deviation_pct"]
                    ret_5d = res["ret_5d"]
                    console.print(f"    [dim]  ✓[/dim] [green]↑[/green] [dim]{res['ticker']} (SMA50: {dev:+.1f}%, 5d: +{ret_5d:.1f}%)[/dim]")
            elif res["status"] == "skipped":
                result["skipped"] += 1
            else:
                result["errors"] += 1

    # Summary
    if console and result["generated"] > 0:
        from rich.text import Text
        summary = Text()
        summary.append("    ")
        summary.append(f"→ {result['generated']} charts", style="bold green")
        summary.append(f" saved to src/data/plots/sma/", style="dim")
        console.print(summary)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX FUND CHARTS — Broad market + sector ETF dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def generate_index_charts(quiet: bool = False) -> Dict:
    """
    Generate candlestick charts with SMA overlays + drift forecast for index funds/ETFs.

    Charts all tickers in INDEX_FUND_TICKERS that have price data.
    Saves to src/data/plots/index/{TICKER}_index.png

    Returns: {"generated": int, "skipped": int, "errors": int}
    """
    result = {"generated": 0, "skipped": 0, "errors": 0}

    # Clear and recreate index chart directory
    os.makedirs(INDEX_CHARTS_DIR, exist_ok=True)
    for old_file in pathlib.Path(INDEX_CHARTS_DIR).glob("*.png"):
        try:
            old_file.unlink()
        except OSError:
            pass

    # Filter to tickers with available price data
    available = []
    for ticker, label in INDEX_FUND_TICKERS:
        csv_path = _price_csv_path(ticker)
        if os.path.exists(csv_path):
            available.append((ticker, label))

    if not available:
        return result

    # Lazy-load mplfinance
    try:
        _ensure_mpf()
    except ImportError:
        if not quiet:
            try:
                from rich.console import Console
                Console().print("[yellow]⚠ mplfinance not installed — skipping index charts[/yellow]")
            except Exception:
                pass
        result["skipped"] = len(available)
        return result

    # Progress display
    console = None
    if not quiet:
        try:
            from rich.console import Console
            from rich.text import Text
            console = Console()

            header = Text()
            header.append("\n    ")
            header.append("🏦 ", style="")
            header.append("Index & ETF Charts", style="bold blue")
            header.append(f"  ({len(available)} funds)", style="dim")
            console.print(header)
        except Exception:
            pass

    # Build task list
    tasks = []
    for ticker, label in available:
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").replace("^", "_")
        tasks.append({
            "ticker": ticker,
            "asset_label": label,
            "signal_type": "INDEX_FUND",
            "output_path": os.path.join(INDEX_CHARTS_DIR, f"{safe_ticker}_index.png"),
        })

    # Generate charts in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(os.cpu_count() or 4, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_initializer) as pool:
        futures = {pool.submit(_chart_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception:
                result["errors"] += 1
                continue

            if res["status"] == "generated":
                result["generated"] += 1
                if console:
                    console.print(f"    [dim]  ✓[/dim] [blue]🏦[/blue] [dim]{res['asset_label']}[/dim]")
            elif res["status"] == "skipped":
                result["skipped"] += 1
                if console:
                    console.print(f"    [dim]  skip {res['ticker']} (no price data)[/dim]")
            else:
                result["errors"] += 1
                if console:
                    console.print(f"    [dim]  ✗ {res['ticker']} (chart error)[/dim]")

    # Summary
    if console and result["generated"] > 0:
        from rich.text import Text
        summary = Text()
        summary.append("    ")
        summary.append(f"→ {result['generated']} charts", style="bold blue")
        summary.append(f" saved to src/data/plots/index/", style="dim")
        console.print(summary)

    return result


# ─── CLI entry point for standalone usage ─────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate signal charts from high-conviction signals")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--sma-only", action="store_true", help="Only generate SMA50 charts")
    parser.add_argument("--signals-only", action="store_true", help="Only generate signal charts")
    parser.add_argument("--index-only", action="store_true", help="Only generate index fund charts")
    args = parser.parse_args()

    run_all = not (args.sma_only or args.signals_only or args.index_only)

    if run_all or args.signals_only:
        result = generate_signal_charts(quiet=args.quiet)
        if not args.quiet:
            print(f"\nSignal Charts: {result['generated']} generated, "
                  f"{result['skipped']} skipped, {result['errors']} errors")

    if run_all or args.sma_only:
        below = find_below_sma50_stocks()
        if not args.quiet:
            render_below_sma50_table(below)
        sma_result = generate_sma_charts(below, quiet=args.quiet)
        if not args.quiet:
            print(f"\nSMA Charts: {sma_result['generated']} generated, "
                  f"{sma_result['skipped']} skipped, {sma_result['errors']} errors")

    if run_all or args.index_only:
        idx_result = generate_index_charts(quiet=args.quiet)
        if not args.quiet:
            print(f"\nIndex Charts: {idx_result['generated']} generated, "
                  f"{idx_result['skipped']} skipped, {idx_result['errors']} errors")
