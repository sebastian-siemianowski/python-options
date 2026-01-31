#!/usr/bin/env python3
"""
tune_ux.py

World-class UX wrapper for tune.py using the Rich presentation layer.
Provides beautiful, informative output while delegating to the core tuning logic.

Usage:
    python src/tuning/tune_ux.py --dry-run --max-assets 5
    python src/tuning/tune_ux.py --force
    python src/tuning/tune_ux.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime

# Suppress verbose tuning output by default to prevent mixing with Rich Live display
# Errors are still captured and displayed prominently
# Use --verbose to see all tuning messages
os.environ['TUNING_QUIET'] = '1'

# Add paths for imports
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import core tuning functionality
from tuning.tune import (
    load_asset_list,
    load_cache,
    save_cache_json,
    _tune_worker,
    _extract_previous_posteriors,
    REGIME_LABELS,
)

# Import PIT-Driven Distribution Escalation
try:
    from calibration.pit_driven_escalation import (
        get_escalation_summary_from_cache,
        extract_escalation_from_result,
        EscalationLevel,
        LEVEL_NAMES,
    )
    PDDE_AVAILABLE = True
except ImportError:
    PDDE_AVAILABLE = False

# Import presentation layer
from decision.signals_ux import (
    create_tuning_console,
    render_tuning_header,
    render_tuning_progress_start,
    render_tuning_summary,
    render_parameter_table,
    render_pdde_escalation_summary,
    render_failed_assets,
    render_dry_run_preview,
    render_cache_status,
    render_cache_update,
    render_end_of_run_summary,
    TuningProgressTracker,
)

from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    parser = argparse.ArgumentParser(
        description="Kalman MLE Tuning with World-Class UX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --force                          # Re-estimate all assets
  %(prog)s --max-assets 10 --dry-run        # Preview first 10 assets
  %(prog)s --prior-lambda 2.0 --prior-mean -5.5  # Custom regularization
  %(prog)s --debug                          # Enable debug output
        """
    )
    parser.add_argument('--assets', type=str, help='Comma-separated list of asset symbols')
    parser.add_argument('--assets-file', type=str, help='Path to file with asset list (one per line)')
    parser.add_argument('--cache-json', type=str, default='src/data/tune',
                       help='Path to cache directory (per-asset) or legacy JSON file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-estimation even if cached values exist')
    parser.add_argument('--force-escalation', action='store_true',
                       help='Force re-estimation only for assets that failed calibration without escalation')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date for data fetching')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for data fetching (default: today)')
    parser.add_argument('--max-assets', type=int, default=None,
                       help='Maximum number of assets to process (useful for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be done without actually processing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (stack traces on errors)')
    parser.add_argument('--prior-mean', type=float, default=-6.0,
                       help='Prior mean for log10(q) (default: -6.0)')
    parser.add_argument('--prior-lambda', type=float, default=1.0,
                       help='Regularization strength (default: 1.0, set to 0 to disable)')
    parser.add_argument('--lambda-regime', type=float, default=0.05,
                       help='Hierarchical shrinkage toward global (default: 0.05, set to 0 for original behavior)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all tuning output including per-model details (may clutter display)')

    args = parser.parse_args()

    # Enable debug mode
    if args.debug:
        os.environ['DEBUG'] = '1'
    
    # Enable verbose mode - unset TUNING_QUIET to show all messages
    if args.verbose:
        os.environ.pop('TUNING_QUIET', None)
        os.environ['TUNING_VERBOSE'] = '1'

    # Create console for rich output
    console = create_tuning_console()

    # Render beautiful header
    render_tuning_header(
        prior_mean=args.prior_mean,
        prior_lambda=args.prior_lambda,
        lambda_regime=args.lambda_regime,
        console=console,
    )

    # Load asset list
    assets = load_asset_list(args.assets, args.assets_file)

    # Apply max-assets limit
    if args.max_assets:
        assets = assets[:args.max_assets]

    # Dry-run mode
    if args.dry_run:
        render_dry_run_preview(assets, console=console)
        return

    # Load existing cache
    cache = load_cache(args.cache_json)

    # Process counters
    new_estimates = 0
    reused_cached = 0
    failed = 0
    calibration_warnings = 0
    student_t_count = 0
    gaussian_count = 0
    regime_tuning_count = 0
    # K=2 mixture removed (empirically falsified: 206 attempts, 0 selections)
    # Counters kept for backward compatibility with cached results
    mixture_attempted_count = 0
    mixture_selected_count = 0
    nu_refinement_attempted_count = 0
    nu_refinement_improved_count = 0
    gh_attempted_count = 0
    gh_selected_count = 0
    regime_tuning_count = 0
    
    # New model counters for comprehensive MODEL SELECTION display
    phi_gaussian_count = 0
    phi_student_t_count = 0
    phi_skew_t_count = 0
    phi_nig_count = 0
    gmm_fitted_count = 0
    hansen_fitted_count = 0
    hansen_left_skew_count = 0
    hansen_right_skew_count = 0
    evt_fitted_count = 0
    evt_heavy_tail_count = 0
    evt_moderate_tail_count = 0
    evt_light_tail_count = 0
    contaminated_t_count = 0
    tvvm_attempted_count = 0
    tvvm_selected_count = 0
    
    # Calibrated Trust Authority statistics
    recalibration_applied_count = 0
    calibrated_trust_count = 0
    trust_effective_values = []  # For computing average trust

    assets_to_process: List[str] = []
    failure_reasons: Dict[str, str] = {}
    
    # Comprehensive data collection for end-of-run summary
    processed_assets: Dict[str, Dict] = {}  # Full results per asset
    regime_distributions: Dict[str, Dict[int, int]] = {}  # Per-asset regime counts
    model_comparisons: Dict[str, Dict] = {}  # Per-asset model comparison results
    processing_log: List[str] = []  # Log of what was processed

    # Helper function to check if asset needs escalation re-tuning
    def needs_escalation_retune(data: Dict) -> bool:
        """Check if asset failed calibration without proper escalation attempt."""
        global_data = data.get('global', data)
        pit_p = global_data.get('pit_ks_pvalue', 1.0)
        calibration_warning = global_data.get('calibration_warning', False)
        mixture_attempted = global_data.get('mixture_attempted', False)
        nu_ref = global_data.get('nu_refinement', {})
        nu_refinement_attempted = nu_ref.get('refinement_attempted', False)
        
        # Asset needs re-tuning if:
        # 1. Has calibration warning (PIT < 0.05)
        # 2. AND neither mixture nor ν-refinement was attempted
        if calibration_warning or pit_p < 0.05:
            if not mixture_attempted and not nu_refinement_attempted:
                return True
        return False

    # Check cache for each asset
    for asset in assets:
        if args.force:
            # Force mode: re-tune all
            assets_to_process.append(asset)
        elif args.force_escalation and asset in cache:
            # Force-escalation mode: only re-tune if escalation was skipped
            if needs_escalation_retune(cache[asset]):
                assets_to_process.append(asset)
            else:
                reused_cached += 1
        elif asset in cache:
            # Normal mode: use cached value
            reused_cached += 1
        else:
            # Asset not in cache: always process
            assets_to_process.append(asset)

    if assets_to_process:
        # Parallel processing
        import multiprocessing
        n_workers = multiprocessing.cpu_count()
        
        render_tuning_progress_start(
            len(assets_to_process), 
            n_workers, 
            reused_cached,
            len(cache),
            args.cache_json,
            console=console
        )
        
        # Create progress tracker
        tracker = TuningProgressTracker(len(assets_to_process), console=console)

        # Prepare arguments for workers
        worker_args = []
        for asset in assets_to_process:
            prev_posteriors = _extract_previous_posteriors(cache.get(asset))
            worker_args.append(
                (asset, args.start, args.end, args.prior_mean, args.prior_lambda, args.lambda_regime, prev_posteriors)
            )

        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_tune_worker, arg): arg[0] for arg in worker_args}

            # Track in-flight assets for display
            in_flight = list(futures.values())
            
            # Show initial assets being processed
            # Use all assets since they're all submitted at once
            tracker.set_in_progress(in_flight)

            for future in as_completed(futures):
                asset = futures[future]
                if asset in in_flight:
                    in_flight.remove(asset)
                
                # Update in-progress list as jobs complete
                tracker.set_in_progress(in_flight)
                
                try:
                    asset_name, result, error, traceback_str = future.result()

                    if result:
                        cache[asset_name] = result
                        new_estimates += 1
                        regime_tuning_count += 1
                        
                        # Store full result for end-of-run summary
                        processed_assets[asset_name] = result

                        global_result = result.get('global', result)
                        noise_model = global_result.get('noise_model', '')
                        
                        # Count base distribution models
                        if noise_model.startswith('phi_nig_'):
                            phi_nig_count += 1
                            student_t_count += 1  # Also count as heavy-tailed
                        elif noise_model.startswith('phi_skew_t_nu_'):
                            phi_skew_t_count += 1
                            student_t_count += 1  # Also count as heavy-tailed
                        elif noise_model.startswith('phi_student_t_nu_'):
                            phi_student_t_count += 1
                            student_t_count += 1  # Count all heavy-tailed models together
                        elif noise_model == 'phi_gaussian' or 'phi' in noise_model.lower():
                            phi_gaussian_count += 1
                            gaussian_count += 1
                        else:
                            gaussian_count += 1

                        if global_result.get('calibration_warning'):
                            calibration_warnings += 1
                        
                        # Track K=2 mixture model attempts and selections
                        if global_result.get('mixture_attempted'):
                            mixture_attempted_count += 1
                        if global_result.get('mixture_selected'):
                            mixture_selected_count += 1
                        
                        # Track adaptive ν refinement attempts and improvements
                        nu_refinement = global_result.get('nu_refinement', {})
                        if nu_refinement.get('refinement_attempted'):
                            nu_refinement_attempted_count += 1
                        if nu_refinement.get('improvement_achieved'):
                            nu_refinement_improved_count += 1
                        
                        # Track GH distribution attempts and selections
                        if global_result.get('gh_attempted'):
                            gh_attempted_count += 1
                        if global_result.get('gh_selected'):
                            gh_selected_count += 1
                        
                        # Collect regime distribution
                        if result.get('regime_counts'):
                            regime_distributions[asset_name] = result['regime_counts']
                        
                        # Collect model comparison data
                        if global_result.get('model_comparison'):
                            model_comparisons[asset_name] = {
                                'model_comparison': global_result['model_comparison'],
                                'selected_model': global_result.get('noise_model', 'unknown'),
                                'best_model_by_bic': global_result.get('best_model_by_bic', 'unknown'),
                                'q': global_result.get('q'),
                                'c': global_result.get('c'),
                                'phi': global_result.get('phi'),
                                'nu': global_result.get('nu'),
                                'bic': global_result.get('bic'),
                                'aic': global_result.get('aic'),
                                'log_likelihood': global_result.get('log_likelihood'),
                                'n_obs': global_result.get('n_obs'),
                            }

                        # Update progress tracker with rich model info
                        q_val = global_result.get('q', float('nan'))
                        c_val = global_result.get('c', 1.0)
                        phi_val = global_result.get('phi')
                        nu_val = global_result.get('nu')
                        bic_val = global_result.get('bic', float('nan'))
                        model_type = global_result.get('noise_model', 'gaussian')
                        nu_was_refined = nu_refinement.get('improvement_achieved', False)
                        
                        # Build comprehensive details string for UX display
                        # Format: model|q|c|phi|nu|bic|trust
                        if global_result.get('gh_selected'):
                            gh_model = global_result.get('gh_model', {})
                            gh_params = gh_model.get('parameters', {})
                            beta = gh_params.get('beta', 0)
                            skew = gh_model.get('skewness_direction', 'sym')[:1].upper()
                            model_str = f"GH(β={beta:.1f},{skew})"
                        elif global_result.get('mixture_selected'):
                            mixture_model = global_result.get('mixture_model', {})
                            sigma_ratio = mixture_model.get('sigma_ratio', 0)
                            weight = mixture_model.get('weight', 0)
                            model_str = f"K2-Mix(σ={sigma_ratio:.1f})"
                        elif model_type.startswith('phi_nig_'):
                            # φ-NIG model with alpha/beta parameters
                            nig_alpha = global_result.get('nig_alpha')
                            nig_beta = global_result.get('nig_beta')
                            if nig_beta is not None and abs(nig_beta) > 0.01:
                                skew_dir = "L" if nig_beta < 0 else "R"
                                model_str = f"NIG({skew_dir})"
                            else:
                                model_str = "NIG"
                        elif model_type.startswith('phi_skew_t_nu_'):
                            # φ-Skew-t model with gamma parameter
                            gamma_val = global_result.get('gamma')
                            if gamma_val is not None and abs(gamma_val - 1.0) > 0.01:
                                skew_dir = "L" if gamma_val < 1.0 else "R"
                                model_str = f"Skew-t({skew_dir})"
                            else:
                                model_str = "Skew-t"
                        elif model_type.startswith('phi_student_t_nu_') and nu_val is not None:
                            model_str = "Student-t"
                        elif phi_val is not None:
                            model_str = "φ-Gaussian"
                        else:
                            model_str = "Gaussian"
                        
                        # Check for GMM availability
                        gmm_data = global_result.get('gmm')
                        has_gmm = (gmm_data is not None and 
                                   isinstance(gmm_data, dict) and 
                                   not gmm_data.get('is_degenerate', False))
                        if has_gmm:
                            model_str += "+GMM"
                            gmm_fitted_count += 1
                        
                        # Check for Hansen Skew-t availability
                        hansen_data = global_result.get('hansen_skew_t')
                        has_hansen = (hansen_data is not None and 
                                      isinstance(hansen_data, dict) and
                                      hansen_data.get('lambda') is not None and
                                      abs(hansen_data.get('lambda', 0)) > 0.01)
                        if has_hansen:
                            hansen_lambda = hansen_data.get('lambda', 0)
                            hansen_dir = "←" if hansen_lambda < 0 else "→"
                            model_str += f"+Hλ{hansen_dir}"
                            hansen_fitted_count += 1
                            if hansen_lambda < 0:
                                hansen_left_skew_count += 1
                            else:
                                hansen_right_skew_count += 1
                        
                        # Check for EVT availability
                        evt_data = global_result.get('evt')
                        has_evt = (evt_data is not None and 
                                   isinstance(evt_data, dict) and
                                   evt_data.get('fit_success', False))
                        if has_evt:
                            evt_xi = evt_data.get('xi', 0)
                            tail_type = "H" if evt_xi > 0.2 else ("M" if evt_xi > 0.05 else "L")
                            model_str += f"+EVT{tail_type}"
                            evt_fitted_count += 1
                            if evt_xi > 0.2:
                                evt_heavy_tail_count += 1
                            elif evt_xi > 0.05:
                                evt_moderate_tail_count += 1
                            else:
                                evt_light_tail_count += 1
                        
                        # Check for Contaminated Student-t availability
                        cst_data = global_result.get('contaminated_student_t')
                        has_cst = (cst_data is not None and 
                                   isinstance(cst_data, dict) and
                                   cst_data.get('nu_normal') is not None and
                                   cst_data.get('nu_crisis') is not None)
                        if has_cst:
                            cst_epsilon = cst_data.get('epsilon', 0.05)
                            model_str += f"+CST{int(cst_epsilon*100)}%"
                            contaminated_t_count += 1
                        
                        import math
                        details = f"{model_str}|q={q_val:.2e}|c={c_val:.3f}"
                        if phi_val is not None:
                            details += f"|φ={phi_val:+.2f}"
                        if nu_val is not None:
                            nu_indicator = f"ν={int(nu_val)}" + ("*" if nu_was_refined else "")
                            details += f"|{nu_indicator}"
                        # Add gamma for skew-t models
                        gamma_val = global_result.get('gamma')
                        if gamma_val is not None and abs(gamma_val - 1.0) > 0.01:
                            details += f"|γ={gamma_val:.2f}"
                        # Add Hansen lambda for asymmetric tails
                        if has_hansen:
                            details += f"|λ={hansen_lambda:+.2f}"
                        # Add EVT xi for tail heaviness
                        if has_evt:
                            details += f"|ξ={evt_xi:.2f}"
                        # Add Contaminated Student-t crisis ν
                        if has_cst:
                            cst_nu_normal = cst_data.get('nu_normal', 12)
                            cst_nu_crisis = cst_data.get('nu_crisis', 4)
                            details += f"|ν_c={int(cst_nu_crisis)}"
                        if math.isfinite(bic_val):
                            details += f"|bic={bic_val:.0f}"
                        
                        # Add trust indicator if available
                        effective_trust = global_result.get('effective_trust')
                        if effective_trust is not None:
                            trust_pct = effective_trust * 100
                            if trust_pct >= 70:
                                trust_indicator = f"T={trust_pct:.0f}%✓"
                            elif trust_pct < 30:
                                trust_indicator = f"T={trust_pct:.0f}%⚠"
                            else:
                                trust_indicator = f"T={trust_pct:.0f}%"
                            details += f"|{trust_indicator}"
                        
                        # Log this processing for end-of-run
                        processing_log.append(f"✓ {asset_name}: {details}")
                        
                        tracker.update(asset_name, 'success', details)
                    else:
                        failed += 1
                        error_msg = error or "tuning returned None"
                        # Build full error with traceback
                        full_error = error_msg
                        if traceback_str:
                            full_error = f"{error_msg}\n{traceback_str}"
                        failure_reasons[asset_name] = full_error
                        processing_log.append(f"❌ {asset_name}: {error_msg}")
                        # Pass full error to tracker so it's displayed
                        tracker.update(asset_name, 'failed', full_error)

                except Exception as e:
                    import traceback
                    failed += 1
                    tb_str = traceback.format_exc()
                    full_error = f"{str(e)}\n{tb_str}"
                    failure_reasons[asset] = full_error
                    processing_log.append(f"❌ {asset}: {str(e)}")
                    # Pass full error to tracker so it's displayed
                    tracker.update(asset, 'failed', full_error)
        
        tracker.finish()
    else:
        # All cached - show minimal info
        from rich.align import Align
        from rich.text import Text
        console.print()
        info = Text()
        info.append("○", style="dim cyan")
        info.append(f"  All {len(assets)} assets cached", style="dim")
        info.append(f"  ·  {len(cache):,} total in cache", style="dim")
        console.print(Align.center(info))
        console.print()

    # ========================================================================
    # MERGE NEWLY TUNED ASSETS INTO CACHE BEFORE SAVING
    # ========================================================================
    # CRITICAL: processed_assets contains all newly tuned results.
    # These must be merged into cache before saving, otherwise new results
    # are lost and only old cached values persist.
    # ========================================================================
    if processed_assets:
        cache.update(processed_assets)

    # Save updated cache
    if new_estimates > 0:
        save_cache_json(cache, args.cache_json)

    # Count regime statistics from cache
    regime_fit_counts = {r: 0 for r in range(5)}
    regime_shrunk_counts = {r: 0 for r in range(5)}
    collapse_warnings = 0
    
    # Detailed model breakdown: {regime_id: {'gaussian': count, 'phi_gaussian': count, 'student_t_4': count, ...}}
    regime_model_breakdown = {r: {} for r in range(5)}
    
    for asset, data in cache.items():
        # Get the global noise model for this asset
        global_data = data.get('global', data)
        noise_model = global_data.get('noise_model', 'gaussian')
        nu_val = global_data.get('nu')
        phi_val = global_data.get('phi')
        gamma_val = global_data.get('gamma')
        nig_alpha = global_data.get('nig_alpha')
        nig_beta = global_data.get('nig_beta')
        mixture_selected = global_data.get('mixture_selected', False)
        mixture_model = global_data.get('mixture_model', {})
        
        # Get augmentation layer data
        gmm_data = global_data.get('gmm')
        hansen_data = global_data.get('hansen_skew_t')
        evt_data = global_data.get('evt')
        cst_data = global_data.get('contaminated_student_t')
        
        # Determine base model category
        if mixture_selected and mixture_model:
            sigma_ratio = mixture_model.get('sigma_ratio', 0)
            model_key = f"K2-Mix(σ={sigma_ratio:.1f})"
        elif noise_model.startswith('phi_nig_'):
            # φ-NIG model
            if nig_beta is not None and abs(nig_beta) > 0.01:
                skew_dir = "L" if nig_beta < 0 else "R"
                model_key = f"φ-NIG({skew_dir})"
            else:
                model_key = "φ-NIG"
        elif noise_model.startswith('phi_skew_t_nu_'):
            # φ-Skew-t model
            if gamma_val is not None and abs(gamma_val - 1.0) > 0.01:
                skew_dir = "L" if gamma_val < 1.0 else "R"
                model_key = f"φ-Skew-t({skew_dir})"
            else:
                model_key = "φ-Skew-t"
        elif noise_model.startswith('phi_student_t_nu_') and nu_val is not None:
            model_key = f"φ-t(ν={int(nu_val)})"
        elif noise_model == 'kalman_phi_gaussian' or phi_val is not None:
            model_key = "φ-Gaussian"
        else:
            model_key = "Gaussian"
        
        # Create augmentation suffix for tracking
        aug_suffix = ""
        if gmm_data is not None and isinstance(gmm_data, dict) and not gmm_data.get('is_degenerate', False):
            aug_suffix += "+GMM"
        if hansen_data is not None and isinstance(hansen_data, dict):
            hansen_lambda = hansen_data.get('lambda')
            if hansen_lambda is not None and abs(hansen_lambda) > 0.01:
                aug_suffix += "+Hλ"
        if evt_data is not None and isinstance(evt_data, dict) and evt_data.get('fit_success', False):
            aug_suffix += "+EVT"
        if cst_data is not None and isinstance(cst_data, dict) and cst_data.get('nu_normal') is not None:
            aug_suffix += "+CST"
        
        # Store both base model and augmented model for breakdown
        regime_data = data.get('regime')
        if regime_data is not None and isinstance(regime_data, dict):
            for r, params in regime_data.items():
                if isinstance(params, dict):
                    is_fallback = params.get('fallback', False) or params.get('regime_meta', {}).get('fallback', False)
                    if not is_fallback:
                        r_int = int(r)
                        regime_fit_counts[r_int] += 1
                        
                        # Count model breakdown per regime
                        if model_key not in regime_model_breakdown[r_int]:
                            regime_model_breakdown[r_int][model_key] = 0
                        regime_model_breakdown[r_int][model_key] += 1
                        
                        is_shrunk = params.get('shrinkage_applied', False) or params.get('regime_meta', {}).get('shrinkage_applied', False)
                        if is_shrunk:
                            regime_shrunk_counts[r_int] += 1
        if 'hierarchical_tuning' in data:
            if data['hierarchical_tuning'].get('collapse_warning', False):
                collapse_warnings += 1

    # Compute escalation statistics from cache (for both fresh and cached runs)
    # These need to be computed from the full cache to show accurate totals
    mixture_attempted_count = 0
    mixture_selected_count = 0
    nu_refinement_attempted_count = 0
    nu_refinement_improved_count = 0
    gh_attempted_count = 0
    gh_selected_count = 0
    tvvm_attempted_count = 0
    tvvm_selected_count = 0
    calibration_warnings = 0
    gaussian_count = 0
    student_t_count = 0
    # Reset new model counters for full cache computation
    phi_gaussian_count = 0
    phi_student_t_count = 0
    phi_skew_t_count = 0
    phi_nig_count = 0
    gmm_fitted_count = 0
    hansen_fitted_count = 0
    hansen_left_skew_count = 0
    hansen_right_skew_count = 0
    evt_fitted_count = 0
    evt_heavy_tail_count = 0
    evt_moderate_tail_count = 0
    evt_light_tail_count = 0
    contaminated_t_count = 0
    # Reset trust statistics for full cache computation
    recalibration_applied_count = 0
    calibrated_trust_count = 0
    trust_effective_values = []
    
    for asset, data in cache.items():
        global_data = data.get('global', data)
        
        # Count model types
        noise_model = global_data.get('noise_model', '')
        if noise_model.startswith('phi_nig_'):
            phi_nig_count += 1
            student_t_count += 1
        elif noise_model.startswith('phi_skew_t_nu_'):
            phi_skew_t_count += 1
            student_t_count += 1
        elif noise_model.startswith('phi_student_t_nu_'):
            phi_student_t_count += 1
            student_t_count += 1
        elif noise_model == 'phi_gaussian' or noise_model == 'kalman_phi_gaussian':
            phi_gaussian_count += 1
            gaussian_count += 1
        elif noise_model == 'generalized_hyperbolic':
            pass  # GH is separate
        elif 'gaussian' in noise_model.lower():
            gaussian_count += 1
        
        # Count augmentation layers from cache
        gmm_data = global_data.get('gmm')
        if gmm_data is not None and isinstance(gmm_data, dict) and not gmm_data.get('is_degenerate', False):
            gmm_fitted_count += 1
        
        hansen_data = global_data.get('hansen_skew_t')
        if hansen_data is not None and isinstance(hansen_data, dict):
            hansen_lambda = hansen_data.get('lambda')
            if hansen_lambda is not None and abs(hansen_lambda) > 0.01:
                hansen_fitted_count += 1
                if hansen_lambda < 0:
                    hansen_left_skew_count += 1
                else:
                    hansen_right_skew_count += 1
        
        evt_data = global_data.get('evt')
        if evt_data is not None and isinstance(evt_data, dict) and evt_data.get('fit_success', False):
            evt_fitted_count += 1
            evt_xi = evt_data.get('xi', 0)
            if evt_xi > 0.2:
                evt_heavy_tail_count += 1
            elif evt_xi > 0.05:
                evt_moderate_tail_count += 1
            else:
                evt_light_tail_count += 1
        
        cst_data = global_data.get('contaminated_student_t')
        if cst_data is not None and isinstance(cst_data, dict):
            if cst_data.get('nu_normal') is not None and cst_data.get('nu_crisis') is not None:
                contaminated_t_count += 1
        
        # Count calibration warnings
        if global_data.get('calibration_warning'):
            calibration_warnings += 1
        
        # Count mixture attempts and selections
        if global_data.get('mixture_attempted'):
            mixture_attempted_count += 1
        if global_data.get('mixture_selected'):
            mixture_selected_count += 1
        
        # Count ν refinement attempts and improvements
        nu_refinement = global_data.get('nu_refinement', {})
        if nu_refinement.get('refinement_attempted'):
            nu_refinement_attempted_count += 1
        if nu_refinement.get('improvement_achieved'):
            nu_refinement_improved_count += 1
        
        # Count GH attempts and selections
        if global_data.get('gh_attempted'):
            gh_attempted_count += 1
        if global_data.get('gh_selected'):
            gh_selected_count += 1
        
        # Count TVVM attempts and selections
        if global_data.get('tvvm_attempted'):
            tvvm_attempted_count += 1
        if global_data.get('tvvm_selected'):
            tvvm_selected_count += 1
        
        # Count Calibrated Trust Authority statistics
        if global_data.get('recalibration_applied'):
            recalibration_applied_count += 1
        if global_data.get('calibrated_trust'):
            calibrated_trust_count += 1
            effective_trust = global_data.get('effective_trust')
            if effective_trust is not None:
                trust_effective_values.append(effective_trust)

    # Compute trust statistics
    avg_effective_trust = sum(trust_effective_values) / len(trust_effective_values) if trust_effective_values else 0.0
    low_trust_count = sum(1 for t in trust_effective_values if t < 0.3)
    high_trust_count = sum(1 for t in trust_effective_values if t >= 0.7)

    # Render beautiful summary
    render_tuning_summary(
        total_assets=len(assets),
        new_estimates=new_estimates,
        reused_cached=reused_cached,
        failed=failed,
        calibration_warnings=calibration_warnings,
        gaussian_count=gaussian_count,
        student_t_count=student_t_count,
        regime_tuning_count=regime_tuning_count,
        lambda_regime=args.lambda_regime,
        regime_fit_counts=regime_fit_counts,
        regime_shrunk_counts=regime_shrunk_counts,
        collapse_warnings=collapse_warnings,
        cache_path=args.cache_json,
        regime_model_breakdown=regime_model_breakdown,
        mixture_attempted_count=mixture_attempted_count,
        mixture_selected_count=mixture_selected_count,
        nu_refinement_attempted_count=nu_refinement_attempted_count,
        nu_refinement_improved_count=nu_refinement_improved_count,
        gh_attempted_count=gh_attempted_count,
        gh_selected_count=gh_selected_count,
        tvvm_attempted_count=tvvm_attempted_count,
        tvvm_selected_count=tvvm_selected_count,
        # New model counters
        phi_gaussian_count=phi_gaussian_count,
        phi_student_t_count=phi_student_t_count,
        phi_skew_t_count=phi_skew_t_count,
        phi_nig_count=phi_nig_count,
        gmm_fitted_count=gmm_fitted_count,
        hansen_fitted_count=hansen_fitted_count,
        hansen_left_skew_count=hansen_left_skew_count,
        hansen_right_skew_count=hansen_right_skew_count,
        evt_fitted_count=evt_fitted_count,
        evt_heavy_tail_count=evt_heavy_tail_count,
        evt_moderate_tail_count=evt_moderate_tail_count,
        evt_light_tail_count=evt_light_tail_count,
        contaminated_t_count=contaminated_t_count,
        # Calibrated Trust Authority statistics
        recalibration_applied_count=recalibration_applied_count,
        calibrated_trust_count=calibrated_trust_count,
        avg_effective_trust=avg_effective_trust,
        low_trust_count=low_trust_count,
        high_trust_count=high_trust_count,
        console=console,
    )

    # Render PDDE escalation summary if available
    if PDDE_AVAILABLE and cache:
        try:
            escalation_summary = get_escalation_summary_from_cache(cache)
            if escalation_summary.get('total', 0) > 0:
                render_pdde_escalation_summary(escalation_summary, console=console)
        except Exception:
            pass  # Silently skip if PDDE summary fails

    # Render parameter table
    if cache:
        render_parameter_table(cache, console=console)

    # Render comprehensive end-of-run summary with all collected data
    render_end_of_run_summary(
        processed_assets=processed_assets,
        regime_distributions=regime_distributions,
        model_comparisons=model_comparisons,
        failure_reasons=failure_reasons,
        processing_log=processing_log,
        console=console,
        cache=cache,
    )


if __name__ == '__main__':
    main()
