from __future__ import annotations
"""
Signal modules configuration: imports, feature flags, conditional dependencies.

Extracted from signals.py (Story 5.1). Contains all standard library imports,
conditional try/except import guards, presentation layer imports, data utility
imports, and cache-only mode enforcement.
"""

# NOTE: No __all__ defined. All public symbols (not starting with _) are
# automatically exported via `from config import *`. Private symbols
# (_to_float, _download_prices, etc.) are imported explicitly by signals.py.

import argparse
import json
import math
import os
import sys

# Ensure src directory is in path for imports
# __file__ is src/decision/signal_modules/config.py, need to get to src/
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))  # signal_modules/
_DECISION_DIR = os.path.dirname(_CONFIG_DIR)              # decision/
_SRC_DIR = os.path.dirname(_DECISION_DIR)                 # src/
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
# NOTE: yfinance is NOT imported here - all data access goes through data_utils
# which respects OFFLINE_MODE. Signal generation should NEVER call Yahoo Finance directly.
from scipy.stats import t as student_t, norm, skew as scipy_stats_skew
from scipy.special import gammaln
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.align import Align
from rich import box
import logging
import os

# HMM regime detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Isotonic Recalibration — Probability Transport Operator
# This is the CORE calibration layer for aligning model beliefs with reality
try:
    from calibration.isotonic_recalibration import (
        TransportMapResult,
        IsotonicRecalibrator,
        apply_recalibration,
        compute_raw_pit_gaussian,
        compute_raw_pit_student_t,
    )
    ISOTONIC_RECALIBRATION_AVAILABLE = True
except ImportError:
    ISOTONIC_RECALIBRATION_AVAILABLE = False

# Calibrated Trust Authority — Single Point of Trust Decision
# ARCHITECTURAL LAW: Trust = Calibration Authority − Governed, Bounded Regime Penalty
# This is the CANONICAL authority for trust decisions.
# All downstream decisions (position sizing, drift weight) flow from here.
try:
    from calibration.calibrated_trust import (
        CalibratedTrust,
        TrustConfig,
        compute_calibrated_trust,
        compute_drift_weight,
        MAX_REGIME_PENALTY,
        MAX_MODEL_PENALTY,
        DEFAULT_REGIME_PENALTY_SCHEDULE,
        DEFAULT_MODEL_PENALTY_SCHEDULE,
        REGIME_NAMES,
    )
    CALIBRATED_TRUST_AVAILABLE = True
except ImportError:
    CALIBRATED_TRUST_AVAILABLE = False

# Control Policy — Authority Boundary Layer (Counter-Proposal v1.0)
# ARCHITECTURAL LAW: Diagnostics RECOMMEND, Policy DECIDES, Models OBEY
# This ensures explicit, auditable escalation decisions.
try:
    from calibration.control_policy import (
        EscalationDecision,
        CalibrationDiagnostics,
        ControlPolicy,
        DECISION_NAMES,
        DEFAULT_CONTROL_POLICY,
        create_diagnostics_from_result,
    )
    CONTROL_POLICY_AVAILABLE = True
except ImportError:
    CONTROL_POLICY_AVAILABLE = False

# PIT Violation Penalty — Asymmetric Calibration Governance (February 2026)
# CORE DESIGN CONSTRAINT: PIT must only act as a PENALTY, never a reward.
# When belief cannot be trusted, the only correct signal is EXIT.
try:
    from calibration.pit_penalty import (
        check_exit_signal_required,
        compute_model_pit_penalty,
        PITViolationResult,
        PITPenaltyReport,
        PIT_EXIT_THRESHOLD,
        PIT_CRITICAL_THRESHOLDS,
    )
    PIT_PENALTY_AVAILABLE = True
except ImportError:
    PIT_PENALTY_AVAILABLE = False

# Filter Result Cache — Deterministic Kalman Filter Reuse (February 2026)
# Enables caching of filter results during signal generation to avoid
# redundant computations when same parameters are used across horizons.
try:
    from models.filter_cache import (
        get_filter_cache,
        get_cache_stats,
        reset_cache_stats,
        clear_filter_cache,
        FILTER_CACHE_ENABLED,
    )
    FILTER_CACHE_AVAILABLE = True
except ImportError:
    FILTER_CACHE_AVAILABLE = False

# Context manager to suppress noisy HMM convergence messages
import contextlib
import io
import warnings

@contextlib.contextmanager  
def suppress_stdout():
    """Temporarily suppress stdout/stderr to hide noisy library messages like HMM convergence warnings."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stderr = old_stderr

# Import presentation layer for display logic
from decision.signals_ux import (
    render_detailed_signal_table,
    render_simplified_signal_table,
    render_multi_asset_summary_table,
    render_sector_summary_tables,
    render_strong_signals_summary,
    render_augmentation_layers_summary,
    build_asset_display_label,
    extract_symbol_from_title,
    format_horizon_label,
    DETAILED_COLUMN_DESCRIPTIONS,
    SIMPLIFIED_COLUMN_DESCRIPTIONS,
)

# Import high conviction signal storage with options data
# February 2026: Saves strong buy/sell signals to src/data/high_conviction/
try:
    from decision.high_conviction_storage import save_high_conviction_signals
    HIGH_CONVICTION_STORAGE_AVAILABLE = True
except ImportError:
    HIGH_CONVICTION_STORAGE_AVAILABLE = False
    save_high_conviction_signals = None

# Import signal chart generation
# March 2026: Candlestick charts with SMA overlays for strong signals + below-SMA50 analysis
try:
    from decision.signal_charts import (
        generate_signal_charts,
        find_below_sma50_stocks,
        render_below_sma50_table,
        generate_sma_charts,
        generate_index_charts,
    )
    SIGNAL_CHARTS_AVAILABLE = True
except ImportError:
    SIGNAL_CHARTS_AVAILABLE = False
    generate_signal_charts = None
    find_below_sma50_stocks = None
    render_below_sma50_table = None
    generate_sma_charts = None
    generate_index_charts = None

# Import render_risk_temperature_summary from risk_temperature module
# (Temperature modules own their own rendering - no duplication in signals_ux)
from decision.risk_temperature import render_risk_temperature_summary

# Import unified risk dashboard (replaces fragmented risk temperature summary)
# February 2026: Single dashboard combining cross-asset, metals, and equity risk
# Uses parallel processing with maximum CPU cores for speed
try:
    from decision.risk_dashboard import compute_and_render_unified_risk
    UNIFIED_RISK_DASHBOARD_AVAILABLE = True
except ImportError:
    UNIFIED_RISK_DASHBOARD_AVAILABLE = False
    compute_and_render_unified_risk = None  # stub to avoid NameError

# Import data utilities and helper functions
from ingestion.data_utils import (
    norm_cdf,
    _to_float,
    safe_last,
    winsorize,
    _download_prices,
    _resolve_display_name,
    _fetch_px_symbol,
    _fetch_with_fallback,
    fetch_px,
    fetch_usd_to_pln_exchange_rate,
    detect_quote_currency,
    _as_series,
    _ensure_float_series,
    _align_fx_asof,
    convert_currency_to_pln,
    convert_price_series_to_pln,
    _resolve_symbol_candidates,
    DEFAULT_ASSET_UNIVERSE,
    get_default_asset_universe,
    COMPANY_NAMES,
    get_company_name,
    SECTOR_MAP,
    get_sector,
    download_prices_bulk,
    save_failed_assets,
    get_price_series,
    STANDARD_PRICE_COLUMNS,
    print_symbol_tables,
    enable_cache_only_mode,
)

# =============================================================================
# SIGNAL GENERATION: CACHE-ONLY MODE
# =============================================================================
# Signal generation should NEVER make Yahoo Finance API calls.
# All price data must come from cache populated during 'make data' or 'make refresh'.
#
# This ensures:
# 1. Fast signal generation (no network latency)
# 2. No rate limiting issues with Yahoo Finance
# 3. Reproducible results (same cache = same signals)
# 4. System works offline once data is cached
#
# If you see "Failed download" errors, run 'make data' first to populate the cache.
# =============================================================================
enable_cache_only_mode()

# Suppress noisy yfinance download warnings (e.g., "1 Failed download: ...")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
