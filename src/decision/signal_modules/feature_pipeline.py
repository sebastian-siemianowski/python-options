from __future__ import annotations
"""
Feature pipeline: comprehensive feature engineering from price series.

Extracted from signals.py (Story 6.5). Contains compute_features() which
performs data validation, EWMA drift/vol, HAR-GK volatility estimation,
vol flooring, Kalman filtering, post-filter features, nu estimation,
and HMM regime fitting.
"""
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -- path setup ---------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Pull in all public symbols from signal_modules
from decision.signal_modules.config import *  # noqa: F403
from decision.signal_modules.volatility_imports import *  # noqa: F403
from decision.signal_modules.hmm_regimes import *  # noqa: F403

# Explicit private-name imports
from ingestion.data_utils import _ensure_float_series
from decision.signal_modules.data_fetching import _fit_student_nu_mle
from decision.signal_modules.kalman_filtering import _kalman_filter_drift
from decision.signal_modules.parameter_loading import (
    _load_tuned_kalman_params,
    is_student_t_family_model_name,
)


def compute_features(
    px: pd.Series, 
    asset_symbol: Optional[str] = None,
    ohlc_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.Series]:
    """
    Compute features from price series for signal generation.

    Args:
        px: Price series (Close prices)
        asset_symbol: Asset symbol (e.g., "PLNJPY=X") for loading tuned Kalman parameters
        ohlc_df: Optional DataFrame with OHLC columns for Garman-Klass volatility

    Returns:
        Dictionary of computed features
        
    VOLATILITY ESTIMATION PRIORITY (February 2026):
        1. Garman-Klass (if OHLC available) - 7.4x more efficient than close-to-close
        2. GARCH(1,1) via MLE - captures volatility clustering
        3. EWMA blend (fallback) - robust baseline
    """
    # Protect log conversion from garbage ticks and non-positive prices
    px = _ensure_float_series(px)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]

    log_px = np.log(px)
    ret = log_px.diff().dropna()
    ret = winsorize(ret, p=0.01)
    ret.name = "ret"

    # Multi-speed EWMA for drift and vol
    mu_fast = ret.ewm(span=21, adjust=False).mean()
    mu_slow = ret.ewm(span=126, adjust=False).mean()

    vol_fast = ret.ewm(span=21, adjust=False).std()
    vol_slow = ret.ewm(span=126, adjust=False).std()

    # =========================================================================
    # VOLATILITY ESTIMATION - ENFORCE HAR-GK ONLY (February 2026)
    # =========================================================================
    # HAR-GK provides multi-horizon memory for crash detection
    # Combined with Garman-Klass (7.4x more efficient than EWMA)
    # =========================================================================
    vol_source = "ewma_fallback"
    garch_params = {}
    
    # ENFORCE HAR-GK ONLY - require OHLC data
    if GK_VOLATILITY_AVAILABLE and ohlc_df is not None and not ohlc_df.empty:
        try:
            cols = {c.lower(): c for c in ohlc_df.columns}
            if all(c in cols for c in ['open', 'high', 'low', 'close']):
                open_ = ohlc_df[cols['open']].values
                high = ohlc_df[cols['high']].values
                low = ohlc_df[cols['low']].values
                close = ohlc_df[cols['close']].values
                
                # ENFORCE HAR-GK ONLY (February 2026)
                vol_gk, vol_estimator = compute_hybrid_volatility_har(
                    open_=open_, high=high, low=low, close=close,
                    span=21, annualize=False, use_har=True
                )
                
                # Convert to Series with proper index
                vol_gk_series = pd.Series(vol_gk, index=ohlc_df.index)
                # Align to ret index
                vol = vol_gk_series.reindex(ret.index).rename("vol")
                vol_source = f"gk_{vol_estimator.lower()}"
            else:
                # OHLC not available - raise error as HAR-GK is required
                raise ValueError(f"OHLC data required for HAR-GK volatility estimation")
        except Exception as e:
            # Log error but don't silently fall back to inferior estimator
            raise ValueError(f"HAR-GK volatility estimation required but failed: {e}")
    else:
        # GK/HAR module not available - this should not happen in production
        raise ImportError("HAR-GK volatility module required but not available")
            
    # Robust global volatility floor to avoid feedback loops when vol collapses recently:
    # - Use a lagged expanding 10th percentile over the entire history (no look-ahead)
    # - Add a relative floor vs long-run median and a small absolute epsilon
    # - Provide an early-history fallback to ensure continuity
    MIN_HIST = 252
    LAG_DAYS = 21  # ~1 trading month lag to avoid immediate reaction to shocks
    abs_floor = 1e-6
    try:
        vol_lag = vol.shift(LAG_DAYS)
        # Expanding quantile and median computed on information up to t-LAG
        global_floor_series = vol_lag.expanding(MIN_HIST).quantile(0.10)
        long_med = vol_lag.expanding(MIN_HIST).median()
        rel_floor = 0.10 * long_med
        # Combine available floors at each timestamp
        floor_candidates = pd.concat([
            global_floor_series.rename("gf"),
            rel_floor.rename("rf")
        ], axis=1)
        floor_t = floor_candidates.max(axis=1)
        # Early history fallback (before MIN_HIST+LAG_DAYS)
        early_med = vol.rolling(63, min_periods=20).median()
        early_floor = np.maximum(0.10 * early_med, abs_floor)
        floor_t = floor_t.combine_first(early_floor)
        # Ensure absolute epsilon
        floor_t = np.maximum(floor_t, abs_floor)
        # Apply the floor index-wise
        vol = np.maximum(vol, floor_t)
    except Exception:
        # Fallback to a simple median-based floor if expanding quantile not available
        fallback_floor = np.maximum(vol.rolling(252, min_periods=63).median() * 0.10, abs_floor)
        vol = np.maximum(vol, fallback_floor)

    # Vol regime (relative to 1y median) — kept for diagnostics, not for shrinkage
    vol_med = vol.rolling(252).median()
    vol_regime = vol / vol_med

    # ========================================
    # Pillar 1: Model-Based Drift Estimation
    # ========================================
    # Use best model selected by BIC from tune.py model comparison:
    # - zero_drift: μ = 0 (no predictable drift)
    # - constant_drift: μ = constant (fixed drift)
    # - ewma_drift: μ = EWMA of returns (adaptive)
    # - kalman_drift: μ from Kalman filter (state-space model)

    # Load tuned parameters and model selection results
    tuned_params = None
    best_model = 'kalman_gaussian'
    # Valid Kalman model names (primary grid + adaptive/refined/unified families)
    kalman_keys = {
        'kalman_gaussian',
        'kalman_phi_gaussian',
        'kalman_gaussian_unified',
        'kalman_phi_gaussian_unified',
    }
    kalman_keys.update({f'phi_student_t_nu_{nu}' for nu in [3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 25]})
    kalman_keys.update({f'phi_student_t_improved_nu_{nu}' for nu in [3, 4, 8, 20]})
    kalman_keys.update({f'phi_student_t_unified_nu_{nu}' for nu in [3, 4, 8, 20]})
    kalman_keys.update({f'phi_student_t_unified_improved_nu_{nu}' for nu in [3, 4, 8, 20]})
    kalman_keys.update({'phi_student_t_nu_mle', 'phi_student_t_improved_nu_mle'})
    tuned_noise_model = 'gaussian'
    tuned_nu = None
    if asset_symbol is not None:
        tuned_params = _load_tuned_kalman_params(asset_symbol)
        if tuned_params:
            best_model = tuned_params.get('best_model', 'kalman_gaussian')
            tuned_noise_model = tuned_params.get('noise_model', 'gaussian')
            # Get nu for any Student-t family model.
            if is_student_t_family_model_name(tuned_noise_model):
                tuned_nu = tuned_params.get('nu')

    # Print BMA model information
    if asset_symbol and tuned_params and tuned_params.get('has_bma'):
        model_posterior = tuned_params.get('model_posterior', {})
        global_data = tuned_params.get('global') or {}
        global_models = global_data.get('models', {})

        # Get model selection method from cache metadata
        model_selection_method = tuned_params.get('model_selection_method', 'combined')
        bic_weight = tuned_params.get('bic_weight', 0.5)

        # Use Rich for world-class presentation
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.columns import Columns
        from rich.console import Group

        console = Console(force_terminal=True)

        # Get company name and sector
        company_name = get_company_name(asset_symbol) or asset_symbol
        sector = get_sector(asset_symbol) or ""

        # Model short names and descriptions - comprehensive mapping
        # Includes: Base models, Momentum variants, Student-t ν grid, and adaptive refinements
        model_info = {
            # ═══════════════════════════════════════════════════════════════════
            # BASE GAUSSIAN MODELS (disabled in BMA, kept for compatibility)
            # ═══════════════════════════════════════════════════════════════════
            'kalman_gaussian': {'short': 'Gaussian', 'desc': 'Random walk drift', 'family': 'gaussian'},
            'kalman_phi_gaussian': {'short': 'φ-Gaussian', 'desc': 'AR(1) mean-reverting drift', 'family': 'gaussian'},
            
            # ═══════════════════════════════════════════════════════════════════
            # UNIFIED GAUSSIAN MODELS (February 2026 — replaces legacy momentum/GAS-Q)
            # ═══════════════════════════════════════════════════════════════════
            'kalman_gaussian_unified': {'short': 'Gaussian-Uni [U]', 'desc': 'Unified Gaussian with internal momentum + GAS-Q', 'family': 'gaussian_unified'},
            'kalman_phi_gaussian_unified': {'short': 'φ-Gaussian-Uni [U]', 'desc': 'Unified φ-Gaussian with internal momentum + GAS-Q', 'family': 'gaussian_unified'},

            # ═══════════════════════════════════════════════════════════════════
            # LEGACY MOMENTUM-AUGMENTED GAUSSIAN (deprecated — kept for cached compatibility)
            # ═══════════════════════════════════════════════════════════════════
            'kalman_gaussian_momentum': {'short': 'Gaussian+Momentum', 'desc': 'Random walk with momentum (legacy)', 'family': 'momentum'},
            'kalman_phi_gaussian_momentum': {'short': 'φ-Gaussian+Momentum', 'desc': 'AR(1) with momentum (legacy)', 'family': 'momentum'},
            
            # ═══════════════════════════════════════════════════════════════════
            # STUDENT-T MODELS (Discrete ν grid: 4, 8, 20)
            # Momentum augmentation is internal (activated if CRPS improves)
            # ═══════════════════════════════════════════════════════════════════
            'phi_student_t_nu_4': {'short': 'φ-T(ν=4)', 'desc': 'Very heavy tails', 'family': 'student_t'},
            'phi_student_t_nu_8': {'short': 'φ-T(ν=8)', 'desc': 'Moderate-heavy tails', 'family': 'student_t'},
            'phi_student_t_nu_20': {'short': 'φ-T(ν=20)', 'desc': 'Light tails', 'family': 'student_t'},
            
            # ═══════════════════════════════════════════════════════════════════
            # LEGACY MOMENTUM-AUGMENTED STUDENT-T (kept for cached compatibility)
            # ═══════════════════════════════════════════════════════════════════
            'phi_student_t_nu_4_momentum': {'short': 'φ-T(ν=4)+Mom', 'desc': 'Very heavy tails with momentum (legacy)', 'family': 'student_t'},
            'phi_student_t_nu_8_momentum': {'short': 'φ-T(ν=8)+Mom', 'desc': 'Moderate-heavy tails with momentum (legacy)', 'family': 'student_t'},
            'phi_student_t_nu_20_momentum': {'short': 'φ-T(ν=20)+Mom', 'desc': 'Light tails with momentum (legacy)', 'family': 'student_t'},
            
            # ═══════════════════════════════════════════════════════════════════
            # ADAPTIVE ν REFINEMENT / LEGACY CANDIDATES (intermediate values)
            # ═══════════════════════════════════════════════════════════════════
            'phi_student_t_nu_3': {'short': 'φ-T(ν=3)', 'desc': 'Extreme tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_5': {'short': 'φ-T(ν=5)', 'desc': 'Heavy tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_6': {'short': 'φ-T(ν=6)', 'desc': 'Heavy tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_7': {'short': 'φ-T(ν=7)', 'desc': 'Heavy tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_10': {'short': 'φ-T(ν=10)', 'desc': 'Moderate tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_12': {'short': 'φ-T(ν=12)', 'desc': 'Moderate tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_14': {'short': 'φ-T(ν=14)', 'desc': 'Light tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_15': {'short': 'φ-T(ν=15)', 'desc': 'Light tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_16': {'short': 'φ-T(ν=16)', 'desc': 'Light tails (refined)', 'family': 'student_t'},
            'phi_student_t_nu_25': {'short': 'φ-T(ν=25)', 'desc': 'Near-Gaussian (refined)', 'family': 'student_t'},
            # Momentum variants for refined/legacy ν values (legacy cached compatibility)
            'phi_student_t_nu_6_momentum': {'short': 'φ-T(ν=6)+Mom', 'desc': 'Heavy tails with momentum (legacy)', 'family': 'student_t'},
            'phi_student_t_nu_12_momentum': {'short': 'φ-T(ν=12)+Mom', 'desc': 'Moderate tails with momentum (legacy)', 'family': 'student_t'},
            'phi_student_t_nu_15_momentum': {'short': 'φ-T(ν=15)+Mom', 'desc': 'Light tails with momentum (legacy)', 'family': 'student_t'},
            
            # ═══════════════════════════════════════════════════════════════════
            # ENHANCED STUDENT-T MODELS (February 2026)
            # ═══════════════════════════════════════════════════════════════════
            # Vol-of-Vol enhanced (dynamic observation noise)
            'phi_student_t_vov': {'short': 'φ-T(VoV)', 'desc': 'Vol-of-Vol enhanced', 'family': 'enhanced'},
            # Two-Piece asymmetric (different νL vs νR)
            'phi_student_t_two_piece': {'short': 'φ-T(2P)', 'desc': 'Two-Piece asymmetric tails', 'family': 'enhanced'},
            # Two-Component mixture (calm/stress body)
            'phi_student_t_mixture': {'short': 'φ-T(Mix)', 'desc': 'Two-Component mixture', 'family': 'enhanced'},
        }

        # Dynamic fallback: generate model info for any model name
        def get_model_info(model_name: str) -> dict:
            """
            Get display info for any model name.
            Handles: base models, momentum suffixes, Student-t ν values, and unknown models.
            """
            # Direct lookup first
            if model_name in model_info:
                return model_info[model_name]
            
            # Check for momentum suffix
            is_momentum = '_momentum' in model_name
            base_name = model_name.replace('_momentum', '')
            
            # Handle enhanced Student-t variants (February 2026)
            # Vol-of-Vol enhanced: phi_student_t_nu_{nu}_vov_g{gamma}_momentum
            if 'vov' in base_name.lower() and 'student_t' in base_name.lower():
                # Try to extract gamma value from name (e.g., phi_student_t_nu_6_vov_g0.5_momentum)
                gamma_str = ""
                nu_str = ""
                import re
                # Extract gamma: look for g followed by number
                gamma_match = re.search(r'_g(\d+\.?\d*)', base_name)
                if gamma_match:
                    gamma_str = gamma_match.group(1)
                # Extract nu
                nu_match = re.search(r'nu_(\d+)', base_name)
                if nu_match:
                    nu_str = nu_match.group(1)
                
                if gamma_str and nu_str:
                    short = f'φ-T(ν={nu_str},γ={gamma_str})'
                elif gamma_str:
                    short = f'φ-T(VoV,γ={gamma_str})'
                elif nu_str:
                    short = f'φ-T(ν={nu_str},VoV)'
                else:
                    short = 'φ-T(VoV)'
                
                if is_momentum:
                    short += '+Momentum'
                family = 'enhanced'
                desc = 'Vol-of-Vol enhanced Student-t'
                if is_momentum:
                    desc += ' with momentum'
                return {'short': short, 'desc': desc, 'family': family}
            
            # Two-Piece asymmetric: phi_student_t_nuL{L}_nuR{R}_momentum
            if 'nul' in base_name.lower() and 'nur' in base_name.lower():
                # Extract nu_left and nu_right
                import re
                nul_match = re.search(r'nul(\d+)', base_name.lower())
                nur_match = re.search(r'nur(\d+)', base_name.lower())
                if nul_match and nur_match:
                    short = f'φ-T(νL={nul_match.group(1)},νR={nur_match.group(1)})'
                else:
                    short = 'φ-T(2P)'
                if is_momentum:
                    short += '+Momentum'
                family = 'enhanced'
                desc = 'Two-Piece asymmetric Student-t'
                if is_momentum:
                    desc += ' with momentum'
                return {'short': short, 'desc': desc, 'family': family}
            
            # Two-Component mixture: phi_student_t_mix_{calm}_{stress}_momentum
            if 'mix' in base_name.lower() and 'student_t' in base_name.lower():
                # Extract calm and stress nu values
                import re
                mix_match = re.search(r'mix_(\d+)_(\d+)', base_name)
                if mix_match:
                    short = f'φ-T(Mix:{mix_match.group(1)}/{mix_match.group(2)})'
                else:
                    short = 'φ-T(Mix)'
                if is_momentum:
                    short += '+Momentum'
                family = 'enhanced'
                desc = 'Two-Component mixture Student-t'
                if is_momentum:
                    desc += ' with momentum'
                return {'short': short, 'desc': desc, 'family': family}
            
            # Handle phi_student_t_nu_* with any ν value
            if is_student_t_family_model_name(base_name):
                try:
                    nu_match = re.search(r'nu_(mle|\d+)', base_name)
                    nu_val = nu_match.group(1) if nu_match else "?"
                    prefix = 'φ-T'
                    if 'unified_improved' in base_name:
                        prefix = 'φ-T-Uni-Imp'
                    elif 'unified' in base_name:
                        prefix = 'φ-T-Uni'
                    elif 'improved' in base_name:
                        prefix = 'φ-T-Imp'
                    short = f'{prefix}(ν={nu_val})'
                    if is_momentum:
                        short += '+Mom'  # Legacy cache compatibility
                    family = 'student_t'
                    desc = f'Student-t family model with ν={nu_val}'
                    if is_momentum:
                        desc += ' (legacy momentum)'
                    return {'short': short, 'desc': desc, 'family': family}
                except ValueError:
                    pass
            
            # Handle other Gaussian variants with momentum
            if is_momentum:
                if 'phi_gaussian' in base_name or 'kalman_phi_gaussian' in base_name:
                    return {'short': 'φ-Gaussian+Momentum', 'desc': 'AR(1) drift with momentum', 'family': 'momentum'}
                elif 'gaussian' in base_name or 'kalman_gaussian' in base_name:
                    return {'short': 'Gaussian+Mom', 'desc': 'Random walk with momentum', 'family': 'momentum'}
            
            # Handle phi_gaussian without momentum
            if 'phi_gaussian' in model_name:
                return {'short': 'φ-Gaussian', 'desc': 'AR(1) mean-reverting drift', 'family': 'gaussian'}
            
            # Handle plain gaussian
            if 'gaussian' in model_name.lower():
                return {'short': 'Gaussian', 'desc': 'Random walk drift', 'family': 'gaussian'}
            
            # Final fallback - clean up the name
            # Remove common prefixes and format nicely
            clean_name = model_name
            for prefix in ['kalman_', 'phi_']:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            # Capitalize and truncate
            clean_name = clean_name.replace('_', ' ').title()
            if len(clean_name) > 18:
                clean_name = clean_name[:16] + '…'
            return {'short': clean_name, 'desc': model_name, 'family': 'other'}

        # Model selection method description
        selection_method_info = {
            'bic': ('BIC-only', 'Traditional Bayesian Information Criterion'),
            'hyvarinen': ('Hyvärinen-only', 'Robust scoring under misspecification'),
            'combined': (f'Combined (α={bic_weight:.1f})', 'BIC + Hyvärinen geometric mean'),
        }
        method_short, method_desc = selection_method_info.get(
            model_selection_method,
            ('Unknown', 'Model selection method')
        )

        # Helper functions to describe parameters in human terms
        def describe_drift_speed(q_val):
            if q_val is None or not np.isfinite(q_val):
                return ("unknown", "white")
            if q_val < 1e-9:
                return ("frozen", "blue")
            elif q_val < 1e-8:
                return ("slow", "cyan")
            elif q_val < 1e-7:
                return ("moderate", "green")
            elif q_val < 1e-6:
                return ("fast", "yellow")
            else:
                return ("rapid", "red")

        def describe_vol_scale(c_val):
            if c_val is None or not np.isfinite(c_val):
                return ("normal", "white")
            if c_val < 0.7:
                return ("muted", "blue")
            elif c_val < 0.9:
                return ("reduced", "cyan")
            elif c_val < 1.1:
                return ("normal", "green")
            elif c_val < 1.3:
                return ("elevated", "yellow")
            else:
                return ("amplified", "red")

        def describe_persistence(phi_val):
            if phi_val is None or not np.isfinite(phi_val):
                return ("n/a", "dim")
            if phi_val < 0.5:
                return ("weak", "red")
            elif phi_val < 0.8:
                return ("moderate", "yellow")
            elif phi_val < 0.95:
                return ("strong", "green")
            elif phi_val < 0.99:
                return ("very strong", "cyan")
            else:
                return ("near-unit", "blue")

        def describe_tail_weight(nu_val):
            if nu_val is None or not np.isfinite(nu_val):
                return ("normal", "white")
            if nu_val < 5:
                return ("very heavy", "red")
            elif nu_val < 10:
                return ("heavy", "yellow")
            elif nu_val < 30:
                return ("moderate", "green")
            else:
                return ("light", "cyan")

        # ═══════════════════════════════════════════════════════════════════════════════
        # EXTRAORDINARY APPLE-QUALITY MODEL PANEL
        # Design: Clean, premium, scannable, beautiful
        # ═══════════════════════════════════════════════════════════════════════════════

        from rich.rule import Rule
        from rich.align import Align

        # Get all models from posterior, sorted by weight descending
        all_models = sorted(model_posterior.keys(), key=lambda m: model_posterior.get(m, 0), reverse=True)

        # Get global-level aggregate scores
        global_hyv_max = tuned_params.get('hyvarinen_max')
        global_bic_min = tuned_params.get('bic_min')

        console.print()
        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # ASSET HEADER - Cinematic, clean, CENTERED
        # ─────────────────────────────────────────────────────────────────────────────
        header_content = Text(justify="center")
        header_content.append("\n", style="")
        header_content.append(asset_symbol, style="bold bright_white")
        header_content.append("\n", style="")
        header_content.append(company_name, style="dim")
        if sector:
            header_content.append(f"  ·  {sector}", style="dim italic")
        header_content.append("\n", style="")

        header_panel = Panel(
            Align.center(header_content),
            box=box.ROUNDED,
            border_style="bright_cyan",
            padding=(0, 4),
        )
        console.print(Align.center(header_panel, width=55))
        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # WINNING MODEL - Hero section
        # ─────────────────────────────────────────────────────────────────────────────
        best_info = get_model_info(best_model)
        best_params = global_models.get(best_model, {})
        best_weight = model_posterior.get(best_model, 0.0)

        # Get BIC/Hyvärinen from best model params (more reliable)
        best_bic = best_params.get('bic')
        best_hyv = best_params.get('hyvarinen_score')

        winner_grid = Table.grid(padding=(0, 4))
        winner_grid.add_column(justify="center")
        winner_grid.add_column(justify="center")
        winner_grid.add_column(justify="center")
        winner_grid.add_column(justify="center")

        def metric_text(value: str, label: str, color: str = "white") -> Text:
            t = Text()
            t.append(f"{value}\n", style=f"bold {color}")
            t.append(label, style="dim")
            return t

        bic_str = f"{best_bic:.0f}" if best_bic and np.isfinite(best_bic) else "—"
        hyv_str = f"{best_hyv:.0f}" if best_hyv and np.isfinite(best_hyv) else "—"

        winner_grid.add_row(
            metric_text(best_info['short'], "Model", "bright_green"),
            metric_text(f"{best_weight:.0%}", "Weight", "bright_cyan"),
            metric_text(bic_str, "BIC", "white"),
            metric_text(hyv_str, "Hyv", "white"),
        )
        console.print(Align.center(winner_grid))
        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # MODEL COMPARISON - Apple-quality, clean, no icons, organized by family
        # ─────────────────────────────────────────────────────────────────────────────
        console.print(Rule(style="dim", characters="─"))
        console.print()
        
        # Section header
        comp_header = Text()
        comp_header.append("    Model Weights", style="bold white")
        console.print(comp_header)
        console.print()

        # Only show models with weight > 1% to reduce clutter (show significant models only)
        visible_models = [m for m in all_models if model_posterior.get(m, 0) >= 0.01]
        
        # Organize models by family for cleaner display
        model_families = {'momentum': [], 'gaussian': [], 'student_t': [], 'enhanced': [], 'other': []}
        for model_name in visible_models:
            info = get_model_info(model_name)
            family = info.get('family', 'other')
            # Handle unknown families by putting them in 'other'
            if family not in model_families:
                family = 'other'
            model_families[family].append(model_name)
        
        # Sort within each family by weight descending
        for family in model_families:
            model_families[family].sort(key=lambda m: model_posterior.get(m, 0), reverse=True)
        
        # Display order: momentum first (most relevant), then enhanced, gaussian, student_t
        display_order = ['momentum', 'enhanced', 'gaussian', 'student_t', 'other']
        
        for family in display_order:
            family_models = model_families[family]
            if not family_models:
                continue
                
            # Family sub-header (only show if multiple families present)
            total_families = sum(1 for f in display_order if model_families[f])
            if total_families > 1 and len(family_models) > 0:
                family_names = {
                    'momentum': 'Momentum-Augmented',
                    'enhanced': 'Enhanced Student-t',
                    'gaussian': 'Gaussian',
                    'student_t': 'Student-t',
                    'other': 'Other'
                }
                fam_header = Text()
                fam_header.append(f"      {family_names.get(family, family)}", style="dim italic")
                console.print(fam_header)
            
            for model_name in family_models:
                p = model_posterior.get(model_name, 0.0)
                m_params = global_models.get(model_name, {})
                info = get_model_info(model_name)
                is_best = model_name == best_model
                is_significant = p >= 0.02  # 2% threshold for significant contribution

                # Visual weight bar - clean, Apple-style
                bar_width = 20
                filled = int(p * bar_width)

                # Build row - no icons, clean typography
                row = Text()
                row.append("    ", style="")

                if is_best:
                    # Best model: emphasized, bright
                    row.append(f"  {info['short']:<18}", style="bold bright_green")
                    row.append(f"{p:>6.1%}  ", style="bold bright_green")
                    row.append("━" * filled, style="bright_green")
                    row.append("─" * (bar_width - filled), style="dim")
                elif is_significant:
                    # Significant model (>=2%): visible
                    row.append(f"  {info['short']:<18}", style="white")
                    row.append(f"{p:>6.1%}  ", style="white")
                    row.append("━" * filled, style="green")
                    row.append("─" * (bar_width - filled), style="dim")
                else:
                    # Minor model (<2%): subdued
                    row.append(f"  {info['short']:<18}", style="dim")
                    row.append(f"{p:>6.1%}  ", style="dim")
                    row.append("─" * bar_width, style="dim")

                console.print(row)
            
            # Small spacing between families
            if family != display_order[-1] and family_models:
                console.print()

        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # PARAMETER ESTIMATES TABLE - Clean, scannable, Apple-quality
        # ─────────────────────────────────────────────────────────────────────────────
        params_header = Text()
        params_header.append("    Parameter Summary", style="bold white")
        console.print(params_header)
        console.print()

        params_table = Table(
            show_header=True,
            header_style="bold dim",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
        params_table.add_column("Model", style="white", width=18)
        params_table.add_column("Drift (q)", justify="center", width=10)
        params_table.add_column("Vol (c)", justify="center", width=10)
        params_table.add_column("Persist (φ)", justify="center", width=10)
        params_table.add_column("Tails (ν)", justify="center", width=10)
        params_table.add_column("Skew/Mix", justify="center", width=12)

        # Helper to describe skewness for various model families
        def describe_skewness(model_name: str, params: dict) -> tuple:
            """Return (description, color) for skewness/mixture parameters."""
            # Hansen Skew-t: lambda parameter
            if 'hansen_skew_t' in model_name or params.get('lambda') is not None:
                lam = params.get('lambda')
                if lam is None:
                    return ("—", "dim")
                if lam < -0.1:
                    return (f"λ={lam:+.2f}", "red")  # Left-skewed, crash risk
                elif lam > 0.1:
                    return (f"λ={lam:+.2f}", "cyan")  # Right-skewed
                return (f"λ={lam:+.2f}", "green")  # Symmetric
            
            # Contaminated Student-t: epsilon (crisis probability)
            if 'cst' in model_name or params.get('epsilon') is not None:
                eps = params.get('epsilon')
                if eps is None:
                    return ("—", "dim")
                if eps > 0.15:
                    return (f"ε={eps:.0%}", "red")  # High crisis prob
                elif eps > 0.08:
                    return (f"ε={eps:.0%}", "yellow")
                return (f"ε={eps:.0%}", "green")
            
            # Phi-Skew-t: gamma parameter
            if 'skew_t' in model_name or params.get('gamma') is not None:
                gamma = params.get('gamma')
                if gamma is None:
                    return ("—", "dim")
                if gamma < 0.9:
                    return (f"γ={gamma:.2f}", "red")  # Left-skewed
                elif gamma > 1.1:
                    return (f"γ={gamma:.2f}", "cyan")  # Right-skewed
                return (f"γ={gamma:.2f}", "green")
            
            return ("—", "dim")

        for model_name in visible_models:
            m_params = global_models.get(model_name, {})
            info = get_model_info(model_name)
            is_best = model_name == best_model

            if m_params.get('fit_success', False):
                q = m_params.get('q', float('nan'))
                c = m_params.get('c', float('nan'))
                phi = m_params.get('phi')
                nu = m_params.get('nu')

                drift_desc, drift_color = describe_drift_speed(q)
                vol_desc, vol_color = describe_vol_scale(c)
                persist_desc, persist_color = describe_persistence(phi) if phi else ("—", "dim")
                tail_desc, tail_color = describe_tail_weight(nu) if nu else ("—", "dim")
                skew_desc, skew_color = describe_skewness(model_name, m_params)

                if is_best:
                    params_table.add_row(
                        f"[bold bright_green]{info['short']}[/bold bright_green]",
                        f"[bold {drift_color}]{drift_desc}[/bold {drift_color}]",
                        f"[bold {vol_color}]{vol_desc}[/bold {vol_color}]",
                        f"[bold {persist_color}]{persist_desc}[/bold {persist_color}]",
                        f"[bold {tail_color}]{tail_desc}[/bold {tail_color}]",
                        f"[bold {skew_color}]{skew_desc}[/bold {skew_color}]",
                    )
                else:
                    params_table.add_row(
                        f"[dim]{info['short']}[/dim]",
                        f"[dim]{drift_desc}[/dim]",
                        f"[dim]{vol_desc}[/dim]",
                        f"[dim]{persist_desc}[/dim]",
                        f"[dim]{tail_desc}[/dim]",
                        f"[dim]{skew_desc}[/dim]",
                    )
            else:
                params_table.add_row(
                    f"[dim]{info['short']}[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                )

        console.print(Padding(params_table, (0, 0, 0, 4)))

        # ─────────────────────────────────────────────────────────────────────────────
        # MODEL QUALITY - Combined calibration status and augmentation layers
        # Clean, compact Apple-quality design
        # ─────────────────────────────────────────────────────────────────────────────
        console.print()

        quality_header = Text()
        quality_header.append("    Model Quality", style="bold white")
        console.print(quality_header)
        console.print()

        # Get calibration data from tuned params
        calibrated_trust_data = tuned_params.get('calibrated_trust', {})
        effective_trust = tuned_params.get('effective_trust')
        calibration_trust = tuned_params.get('calibration_trust')
        regime_penalty = tuned_params.get('regime_penalty')
        calibration_warning = tuned_params.get('calibration_warning', False)
        pit_ks_pvalue = global_data.get('pit_ks_pvalue')
        pit_ks_pvalue_calibrated = global_data.get('pit_ks_pvalue_calibrated')
        recalibration_applied = tuned_params.get('recalibration_applied', False)
        nu_refinement = tuned_params.get('nu_refinement', {})
        gh_selected = tuned_params.get('gh_selected', False)
        gh_model = tuned_params.get('gh_model', {})

        # Extract augmentation layer data
        hansen_data = global_data.get('hansen_skew_t', {})
        cst_data = global_data.get('contaminated_student_t', {})
        skew_t_data = global_data.get('phi_skew_t', {})
        vol_estimator = global_data.get('volatility_estimator', 'EWMA')

        # Build compact quality table
        quality_table = Table(
            show_header=False,
            border_style="dim",
            box=box.SIMPLE,
            padding=(0, 2),
            expand=False,
        )
        quality_table.add_column("Label", style="dim", width=24)
        quality_table.add_column("Value", width=56)

        # ═══════════════════════════════════════════════════════════════════════════
        # PIT CALIBRATION - Model fit quality check
        # Tests if predictions match actual market behavior (higher p = better fit)
        # ═══════════════════════════════════════════════════════════════════════════
        pit_status_parts = []
        pit_explanation = ""
        
        if pit_ks_pvalue is not None:
            pit_color = "red" if pit_ks_pvalue < 0.01 else "yellow" if pit_ks_pvalue < 0.05 else "green"
            pit_status_parts.append(f"[{pit_color}]p={pit_ks_pvalue:.4f}[/{pit_color}]")
            
            # User-friendly explanation
            if pit_ks_pvalue >= 0.05:
                pit_explanation = "✓ Good fit"
            elif pit_ks_pvalue >= 0.01:
                pit_explanation = "⚠ Marginal fit"
            else:
                pit_explanation = "✗ Poor fit"
        
        # ν refinement (tail thickness adjusted)
        if nu_refinement:
            nu_improved = nu_refinement.get('improvement_achieved', False)
            nu_original = nu_refinement.get('nu_original')
            nu_final = nu_refinement.get('nu_final')
            if nu_improved and nu_original != nu_final:
                pit_status_parts.append(f"[green]ν {nu_original}→{nu_final}[/green]")
        
        # Recalibration applied
        if recalibration_applied:
            pit_status_parts.append("[green]recalibrated[/green]")
        
        if pit_status_parts:
            if calibration_warning:
                cal_label = "[yellow]PIT Calibration[/yellow]"
            else:
                cal_label = "[green]PIT Calibration[/green]"
            status_str = "  ".join(pit_status_parts)
            if pit_explanation:
                status_str += f"  [dim]{pit_explanation}[/dim]"
            quality_table.add_row(cal_label, status_str)

        # ═══════════════════════════════════════════════════════════════════════════
        # TRUST AUTHORITY - Overall confidence in model predictions
        # Higher = more reliable signals, lower = treat with caution
        # ═══════════════════════════════════════════════════════════════════════════
        if effective_trust is not None:
            eff_trust_color = "green" if effective_trust > 0.7 else "yellow" if effective_trust > 0.4 else "red"
            trust_desc = "High confidence" if effective_trust > 0.7 else "Moderate" if effective_trust > 0.4 else "Low confidence"
            
            trust_str = f"[bold {eff_trust_color}]{effective_trust:.0%}[/bold {eff_trust_color}]  [dim]{trust_desc}[/dim]"
            
            # Add regime penalty context if significant
            if regime_penalty is not None and regime_penalty >= 0.05:
                regime_context = calibrated_trust_data.get('regime_context', 'stressed')
                trust_str += f"  [dim](-{regime_penalty:.0%} {regime_context} market)[/dim]"
            
            quality_table.add_row("[bold]Trust[/bold]", trust_str)

        # ═══════════════════════════════════════════════════════════════════════════
        # VOLATILITY ESTIMATOR - Method used to estimate price volatility
        # ═══════════════════════════════════════════════════════════════════════════
        if vol_estimator and vol_estimator.upper() == "GK":
            quality_table.add_row(
                "[green]Volatility[/green]",
                "[green]Garman-Klass[/green]  [dim]Uses OHLC data, 7.4× more accurate[/dim]"
            )
        elif vol_estimator and "gk" in vol_estimator.lower():
            quality_table.add_row("[green]Volatility[/green]", f"[green]{vol_estimator}[/green]  [dim]OHLC range-based[/dim]")
        else:
            quality_table.add_row("[dim]Volatility[/dim]", f"[dim]{vol_estimator or 'EWMA'}  Close-to-close estimate[/dim]")

        # ═══════════════════════════════════════════════════════════════════════════
        # HANSEN SKEW-T - Captures asymmetric crash/rally behavior
        # λ < 0 = left-skewed (crash prone), λ > 0 = right-skewed (rally prone)
        # ═══════════════════════════════════════════════════════════════════════════
        hansen_lambda = hansen_data.get('lambda') if hansen_data else None
        hansen_nu = hansen_data.get('nu') if hansen_data else None
        hansen_enabled = hansen_lambda is not None and abs(hansen_lambda) > 0.01
        
        if hansen_enabled:
            # Direction indicator and explanation
            if hansen_lambda < -0.05:
                skew_symbol = "←"
                skew_desc = "crash prone"
            elif hansen_lambda > 0.05:
                skew_symbol = "→"
                skew_desc = "rally prone"
            else:
                skew_symbol = "○"
                skew_desc = "balanced"
            
            hansen_str = f"[green]λ={hansen_lambda:+.2f}[/green] {skew_symbol}  [dim]{skew_desc}[/dim]"
            if hansen_nu:
                tail_desc = "extreme tails" if hansen_nu <= 4 else "heavy tails" if hansen_nu <= 8 else "moderate tails"
                hansen_str += f"  [dim]ν={hansen_nu:.0f} ({tail_desc})[/dim]"
            quality_table.add_row("[cyan]Hansen Skew-T[/cyan]", hansen_str)
        else:
            quality_table.add_row("[dim]Hansen Skew-T[/dim]", "[dim]Not fitted  (Asymmetric tail model)[/dim]")

        # ═══════════════════════════════════════════════════════════════════════════
        # CONTAMINATED STUDENT-T - Regime-switching model for crisis detection
        # Models market as mixture of "normal" and "crisis" states
        # ε = probability of being in crisis mode
        # ═══════════════════════════════════════════════════════════════════════════
        cst_nu_normal = cst_data.get('nu_normal') if cst_data else None
        cst_nu_crisis = cst_data.get('nu_crisis') if cst_data else None
        cst_epsilon = cst_data.get('epsilon') if cst_data else None
        cst_enabled = cst_nu_normal is not None and cst_epsilon is not None and cst_epsilon > 0.001
        
        if cst_enabled:
            # Explain crisis probability
            if cst_epsilon >= 0.25:
                crisis_desc = "high stress"
            elif cst_epsilon >= 0.15:
                crisis_desc = "elevated"
            else:
                crisis_desc = "normal"
            
            cst_str = f"[green]ε={cst_epsilon:.0%}[/green] {crisis_desc}  [dim]ν: {cst_nu_normal:.0f}→{cst_nu_crisis:.0f} in crisis[/dim]"
            quality_table.add_row("[magenta]Contaminated-T[/magenta]", cst_str)
        else:
            quality_table.add_row("[dim]Contaminated-T[/dim]", "[dim]Not fitted  (Crisis regime model)[/dim]")

        # ═══════════════════════════════════════════════════════════════════════════
        # SKEW-T (Fernández-Steel) - Another asymmetric distribution model
        # γ < 1 = left-skewed, γ > 1 = right-skewed
        # ═══════════════════════════════════════════════════════════════════════════
        skew_t_gamma = skew_t_data.get('gamma') if skew_t_data else None
        skew_t_nu = skew_t_data.get('nu') if skew_t_data else None
        skew_t_enabled = skew_t_gamma is not None
        
        if skew_t_enabled:
            if skew_t_gamma < 0.95:
                gamma_desc = "crash risk higher"
            elif skew_t_gamma > 1.05:
                gamma_desc = "rally potential"
            else:
                gamma_desc = "balanced"
            
            skew_str = f"[green]γ={skew_t_gamma:.2f}[/green]  [dim]{gamma_desc}[/dim]"
            if skew_t_nu:
                skew_str += f"  [dim]ν={skew_t_nu:.0f}[/dim]"
            quality_table.add_row("[purple]Skew-T (F-S)[/purple]", skew_str)
        else:
            quality_table.add_row("[dim]Skew-T (F-S)[/dim]", "[dim]Not fitted[/dim]")

        console.print(Padding(quality_table, (0, 0, 0, 4)))

        console.print()
        console.print(Rule(style="dim", characters="─"))
        console.print()
    elif asset_symbol and tuned_params:
        # Old cache format warning
        print(f"\n\033[93m⚠️  {asset_symbol}: Old cache format — run tune.py\033[0m\n")

    # Apply drift estimation based on best model selection
    # NOTE: In BMA architecture, "best_model" is used for Kalman filter params,
    # but actual predictions use weighted mixture over all models
    if best_model in kalman_keys:
        kf_result = _kalman_filter_drift(ret, vol, q=None, asset_symbol=asset_symbol)

        # Extract Kalman-filtered drift estimates
        if kf_result and "mu_kf_smoothed" in kf_result:
            # Use backward-smoothed estimates (uses all data, statistically optimal)
            mu_kf = kf_result["mu_kf_smoothed"]
            var_kf = kf_result["var_kf_smoothed"]
            kalman_available = True
            kalman_metadata = {
                "log_likelihood": kf_result.get("log_likelihood", float("nan")),
                "process_noise_var": kf_result.get("process_noise_var", float("nan")),
                "n_obs": kf_result.get("n_obs", 0),
                # Refinement 1: q optimization metadata
                "q_optimal": kf_result.get("q_optimal", float("nan")),
                "q_heuristic": kf_result.get("q_heuristic", float("nan")),
                "q_optimization_attempted": kf_result.get("q_optimization_attempted", False),
                # Refinement 2: Kalman gain statistics (situational awareness)
                "kalman_gain_mean": kf_result.get("kalman_gain_mean", float("nan")),
                "kalman_gain_recent": kf_result.get("kalman_gain_recent", float("nan")),
                # Refinement 3: Innovation whiteness test (model adequacy)
                "innovation_whiteness": kf_result.get("innovation_whiteness", {}),
                # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
                "kalman_heteroskedastic_mode": kf_result.get("heteroskedastic_mode", False),
                "kalman_c_optimal": kf_result.get("c_optimal"),
                "kalman_q_t_mean": kf_result.get("q_t_mean"),
                "kalman_q_t_std": kf_result.get("q_t_std"),
                "kalman_q_t_min": kf_result.get("q_t_min"),
                "kalman_q_t_max": kf_result.get("q_t_max"),
                # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
                "kalman_robust_t_mode": kf_result.get("robust_t_mode", False),
                "kalman_nu_robust": kf_result.get("nu_robust"),
                # Level-7+ Refinement: Regime-dependent drift priors
                "kalman_regime_prior_used": kf_result.get("regime_prior_used", False),
                "kalman_regime_info": kf_result.get("regime_prior_info", {}),
                # φ persistence (from tuned cache or filter)
                "kalman_phi": tuned_params.get("phi") if tuned_params else kf_result.get("phi_used"),
                "phi_used": kf_result.get("phi_used"),
                # Regime-conditional parameters from tune.py hierarchical cache
                "regime_params": tuned_params.get("regime", {}) if tuned_params else {},
                "has_regime_params": bool(tuned_params.get("regime")) if tuned_params else False,
                # Noise model for Student-t support
                "kalman_noise_model": tuned_params.get("noise_model", "gaussian") if tuned_params else "gaussian",
                "kalman_nu": tuned_params.get("nu") if tuned_params else None,
                # BMA diagnostics from best model (for drift quality assessment)
                "pit_ks_pvalue": tuned_params.get("pit_ks_pvalue") if tuned_params else None,
                "ks_statistic": tuned_params.get("ks_statistic") if tuned_params else None,
                "bic": tuned_params.get("bic") if tuned_params else None,
                "hyvarinen_score": tuned_params.get("hyvarinen_score") if tuned_params else None,
                "combined_score": tuned_params.get("combined_score") if tuned_params else None,
                # Global-level aggregates for model selection
                "hyvarinen_max": tuned_params.get("hyvarinen_max") if tuned_params else None,
                "combined_score_min": tuned_params.get("combined_score_min") if tuned_params else None,
                "bic_min": tuned_params.get("bic_min") if tuned_params else None,
                "model_selection_method": tuned_params.get("model_selection_method", "combined") if tuned_params else "combined",
                "bic_weight": tuned_params.get("bic_weight", 0.5) if tuned_params else 0.5,
                "entropy_lambda": tuned_params.get("entropy_lambda", 0.05) if tuned_params else 0.05,
                "model_posterior": tuned_params.get("model_posterior", {}) if tuned_params else {},
                "best_model": tuned_params.get("best_model") if tuned_params else best_model,
                # Elite Tuning diagnostics (v2.0 - February 2026)
                # Stability-aware model selection synced with BIC/Hyvarinen
                "elite_tuning_enabled": tuned_params.get("elite_tuning_enabled", False) if tuned_params else False,
                "elite_fragility_index": tuned_params.get("elite_fragility_index") if tuned_params else None,
                "elite_is_ridge": tuned_params.get("elite_is_ridge", False) if tuned_params else False,
                "elite_basin_score": tuned_params.get("elite_basin_score") if tuned_params else None,
                "elite_fragility_penalty": tuned_params.get("elite_fragility_penalty", 0.0) if tuned_params else 0.0,
                # VIX-based ν adjustment diagnostics (February 2026)
                "vix_nu_adjustment_applied": kf_result.get("vix_nu_adjustment_applied", False),
                "nu_original": kf_result.get("nu_original"),
                "nu_adjusted": kf_result.get("nu_adjusted"),
            }
        else:
            # Fallback: use EWMA blend if Kalman fails
            mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
            mu_kf = mu_blend
            var_kf = pd.Series(0.0, index=mu_kf.index)  # no uncertainty quantified
            kalman_available = False
            kalman_metadata = {
                "model_selected": str(best_model),
                "reason": "Unrecognized model key; defaulting to EWMA blend"
            }
    else:
        # Unknown model key: fallback to EWMA blend
        mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
        mu_kf = mu_blend
        var_kf = pd.Series(0.0, index=mu_kf.index)
        kalman_available = False
        kalman_metadata = {
            "model_selected": str(best_model),
            "reason": "Unrecognized model key; defaulting to EWMA blend"
        }

    # Trend filter (200D z-distance) - kept for diagnostics
    sma200 = px.rolling(200).mean()
    trend_z = (px - sma200) / px.rolling(200).std()

    # HMM regime detection (for regime-aware adjustments, not drift estimation)
    # Fit HMM to get regime posteriors
    hmm_result_prelim = fit_hmm_regimes(
        {"ret": ret, "vol": vol},
        n_states=3,
        random_seed=42
    )

    # Apply light regime-aware shrinkage to Kalman drift in extreme regimes
    # (Kalman already handles uncertainty; this adds regime-specific conservatism)
    if hmm_result_prelim is not None and "posterior_probs" in hmm_result_prelim:
        try:
            posterior_probs = hmm_result_prelim["posterior_probs"]

            # Align posteriors with mu_kf index
            posterior_aligned = posterior_probs.reindex(mu_kf.index).ffill().fillna(0.333)

            # Extract regime probabilities
            regime_names = hmm_result_prelim["regime_names"]
            calm_idx = [k for k, v in regime_names.items() if v == "calm"]
            crisis_idx = [k for k, v in regime_names.items() if v == "crisis"]

            p_calm = posterior_aligned.iloc[:, calm_idx[0]].values if calm_idx else np.zeros(len(mu_kf))
            p_crisis = posterior_aligned.iloc[:, crisis_idx[0]].values if crisis_idx else np.zeros(len(mu_kf))

            # Light shrinkage in crisis regimes (Kalman handles most uncertainty)
            # Shrink toward zero in extreme crisis to be conservative
            shrinkage = 0.3 * p_crisis  # 0-30% shrinkage based on crisis probability
            shrinkage = np.clip(shrinkage, 0.0, 0.5)

            # Final drift: Kalman estimate with regime-aware shrinkage
            mu_final = pd.Series(
                (1.0 - shrinkage) * mu_kf.values,  # shrink toward zero in crisis
                index=mu_kf.index,
                name="mu_final"
            )

        except Exception:
            # Fallback: use Kalman estimate without regime adjustment
            mu_final = mu_kf.copy()
    else:
        # HMM not available: use pure Kalman estimate
        mu_final = mu_kf.copy()

    # Robust fallback for NaNs
    mu_final = mu_final.fillna(0.0)

    # Legacy aliases for backward compatibility
    mu_blend = 0.5 * mu_fast + 0.5 * mu_slow  # kept for diagnostics
    mu_post = mu_final  # primary drift estimate
    mu = mu_final  # shorthand

    # Short-term mean-reversion z (5d move over 1m vol)
    r5 = (log_px - log_px.shift(5))
    rv_1m = ret.rolling(21).std() * math.sqrt(5)
    z5 = r5 / rv_1m

    # Rolling skewness (directional asymmetry) and excess kurtosis (Fisher)
    skew = ret.rolling(252, min_periods=63).skew()
    # Optional stabilization: smooth skew to avoid warm-up swings when it first becomes defined
    try:
        skew_s = skew.ewm(span=30, adjust=False).mean()
    except Exception:
        skew_s = skew
    ex_kurt = ret.rolling(252, min_periods=63).kurt()  # normal ~ 0
    # Convert excess kurtosis to t degrees of freedom via: excess = 6/(nu-4) => nu = 4 + 6/excess
    # Handle near-zero/negative excess by mapping to large nu (approx normal)
    eps = 1e-6
    nu = 4.0 + 6.0 / ex_kurt.where(ex_kurt > eps, np.nan)
    nu = nu.fillna(1e6)  # ~normal
    # Clip degrees of freedom to a stable range to prevent extreme tail chaos in flash crashes
    nu = nu.clip(lower=4.5, upper=500.0)

    # Tail parameter: prefer tuned ν from cache for Student-t world; otherwise keep legacy estimate
    is_student_t_world = is_student_t_family_model_name(tuned_noise_model)
    if is_student_t_world and tuned_nu is not None and np.isfinite(tuned_nu):
        # Level-7 rule: ν is fixed from tuning cache in Student-t world
        nu_hat = float(tuned_nu)
        nu_info = {"nu_hat": nu_hat, "source": "tuned_cache"}
    else:
        # Non-Student-t worlds may estimate ν diagnostically; Student-t world never refits
        try:
            mu_post_aligned = pd.Series(mu_post, index=ret.index).astype(float)
            vol_aligned = pd.Series(vol, index=ret.index).astype(float)
            resid = (ret - mu_post_aligned).replace([np.inf, -np.inf], np.nan)
            z_std = resid / vol_aligned.replace(0.0, np.nan)
            z_std = z_std.replace([np.inf, -np.inf], np.nan).dropna()
            nu_info = _fit_student_nu_mle(z_std, min_n=200, bounds=(4.5, 500.0))
            nu_hat = float(nu_info.get("nu_hat", 50.0))
        except Exception:
            nu_info = {"nu_hat": 50.0, "ll": float("nan"), "n": 0, "converged": False}
            nu_hat = 50.0

    # t-stat style momentum: cum return / realized vol over window
    def mom_t(days: int) -> pd.Series:
        cum = (log_px - log_px.shift(days))
        rv = ret.rolling(days).std() * math.sqrt(days)
        return cum / rv

    mom21 = mom_t(21)
    mom63 = mom_t(63)
    mom126 = mom_t(126)
    mom252 = mom_t(252)

    # Reuse HMM result from drift estimation (avoid duplicate fitting)
    hmm_result = hmm_result_prelim

    return {
        "px": px,
        "ret": ret,
        "mu": mu,
        "mu_post": mu_post,
        "mu_blend": mu_blend,
        "vol": vol,
        "vol_regime": vol_regime,
        "trend_z": trend_z,
        "z5": z5,
        "nu": nu,               # rolling, for diagnostics only
        "nu_hat": pd.Series([nu_hat], index=[ret.index[-1]]) if len(ret.index)>0 else pd.Series([nu_hat]),
        "nu_info": nu_info,     # dict metadata
        "skew": skew,
        "skew_s": skew_s,
        "mom21": mom21,
        "mom63": mom63,
        "mom126": mom126,
        "mom252": mom252,
        # meta (not series)
        "vol_source": vol_source,
        "garch_params": garch_params,
        # HMM regime detection
        "hmm_result": hmm_result,
        # Pillar 1: Kalman filter drift estimation
        "mu_kf": mu_kf if kalman_available else mu_blend,  # Kalman-filtered drift
        "var_kf": var_kf if kalman_available else pd.Series(0.0, index=ret.index),  # drift variance
        "mu_final": mu_final,  # shorthand
        "kalman_available": kalman_available,  # flag for diagnostics
        "kalman_metadata": kalman_metadata,  # log-likelihood, process noise, etc.
        "phi_used": kalman_metadata.get("phi_used", tuned_params.get("phi") if tuned_params else None),
        # Calibrated Trust Authority
        # ARCHITECTURAL LAW: Trust = Calibration Authority − Bounded Regime Penalty
        "calibrated_trust": tuned_params.get("calibrated_trust") if tuned_params else None,
        "recalibration": tuned_params.get("recalibration") if tuned_params else None,
        "recalibration_applied": tuned_params.get("recalibration_applied", False) if tuned_params else False,
    }
