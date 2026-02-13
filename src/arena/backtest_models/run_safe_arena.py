#!/usr/bin/env python3
"""
Arena Safe Storage Runner
=========================
Tests all models in safe_storage using the main arena infrastructure
and saves results to data/safe_storage_results.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()

# Safe storage directory
SAFE_STORAGE_DIR = Path(__file__).parent
DATA_DIR = SAFE_STORAGE_DIR / "data"
RESULTS_FILE = DATA_DIR / "safe_storage_results.json"

# Model registry
SAFE_STORAGE_MODEL_REGISTRY = {
    "elite_hybrid_omega2": {"file": "elite_hybrid_omega2", "class": "EliteHybridOmega2Model"},
    "optimal_hyv_iota": {"file": "optimal_hyv_iota", "class": "OptimalHyvIotaModel"},
    "dtcwt_qshift": {"file": "dtcwt_qshift", "class": "DTCWTQShiftModel"},
    "dtcwt_magnitude_threshold": {"file": "dtcwt_magnitude_threshold", "class": "DTCWTMagnitudeThresholdModel"},
    "dualtree_complex_wavelet": {"file": "dualtree_complex_wavelet", "class": "DualTreeComplexWaveletKalmanModel"},
    "elite_hybrid_eta": {"file": "elite_hybrid_eta", "class": "EliteHybridEtaModel"},
    "dtcwt_adaptive_levels": {"file": "dtcwt_adaptive_levels", "class": "DTCWTAdaptiveLevelsModel"},
    "css_omicron": {"file": "css_omicron", "class": "CssOmicronModel"},
    "css_variant_v5": {"file": "css_variant_v5", "class": "CssVariantV5Model"},
    "malliavin_score_weight": {"file": "malliavin_score_weight", "class": "MalliavinScoreWeightModel"},
    "matrix_log_barrier": {"file": "matrix_log_barrier", "class": "MatrixLogBarrierModel"},
    "resolvent_regularized": {"file": "resolvent_regularized", "class": "ResolventRegularizedModel"},
    "free_probability": {"file": "free_probability", "class": "FreeProbabilityModel"},
    "martingale_stopping": {"file": "martingale_stopping", "class": "MartingaleStoppingModel"},
    "fisher_rao_metric": {"file": "fisher_rao_metric", "class": "FisherRaoMetricModel"},
    "adjoint_duality": {"file": "adjoint_duality", "class": "AdjointDualityModel"},
    "rg_fixed_point": {"file": "rg_fixed_point", "class": "RGFixedPointModel"},
}


def run_safe_storage_arena():
    """Run arena competition for all safe_storage models using main arena infrastructure."""
    import importlib
    from arena.arena_tune import run_arena_competition
    from arena.arena_config import (
        ARENA_BENCHMARK_SYMBOLS,
        load_disabled_models,
        save_disabled_models,
    )
    from arena.experimental_models import (
        EXPERIMENTAL_MODELS, 
        EXPERIMENTAL_MODEL_KWARGS,
        EXPERIMENTAL_MODEL_SPECS,
        ExperimentalModelSpec,
        ExperimentalModelFamily,
    )
    
    console.print("\n[bold cyan]SAFE STORAGE ARENA[/bold cyan]")
    console.print("─" * 60)
    
    # Save and clear disabled models FIRST (we want to test ALL safe storage models)
    original_disabled = load_disabled_models()
    save_disabled_models({})
    console.print(f"  [dim]Temporarily cleared {len(original_disabled)} disabled models[/dim]")
    
    # Build model list for arena - import each safe_storage model class
    safe_storage_classes = {}
    safe_storage_kwargs = {}
    for name, info in SAFE_STORAGE_MODEL_REGISTRY.items():
        try:
            module = importlib.import_module(f"arena.safe_storage.{info['file']}")
            model_class = getattr(module, info['class'])
            safe_storage_classes[name] = model_class
            safe_storage_kwargs[name] = {'n_levels': 4}
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to load {name}: {e}[/yellow]")
    
    console.print(f"  Loaded {len(safe_storage_classes)} safe_storage models")
    console.print(f"  Testing on {len(ARENA_BENCHMARK_SYMBOLS)} symbols")
    console.print()
    
    # Save original experimental models
    original_models = EXPERIMENTAL_MODELS.copy()
    original_kwargs = EXPERIMENTAL_MODEL_KWARGS.copy()
    original_specs = EXPERIMENTAL_MODEL_SPECS.copy()
    
    # Clear existing experimental and add safe_storage models
    EXPERIMENTAL_MODELS.clear()
    EXPERIMENTAL_MODEL_KWARGS.clear()
    EXPERIMENTAL_MODEL_SPECS.clear()
    
    # Add safe_storage models as experimental
    for name, cls in safe_storage_classes.items():
        EXPERIMENTAL_MODELS[name] = cls
        EXPERIMENTAL_MODEL_KWARGS[name] = safe_storage_kwargs.get(name, {})
        EXPERIMENTAL_MODEL_SPECS[name] = ExperimentalModelSpec(
            name=name,
            family=ExperimentalModelFamily.CUSTOM,
            n_params=3,
            param_names=("q", "c", "phi"),
            description=f"Safe storage model: {name}",
            model_class=cls,
        )
    
    try:
        # Run competition - skip_disable=True to prevent disabling safe storage models
        result = run_arena_competition(
            symbols=list(ARENA_BENCHMARK_SYMBOLS),
            parallel=True,
            n_workers=8,
            skip_disable=True,
        )
        
        # Extract results for safe_storage models
        aggregated = {}
        for name in safe_storage_classes.keys():
            model_scores = [s for s in result.scores if s.model_name == name]
            if not model_scores:
                continue
            
            # Aggregate
            finals = [s.final_score for s in model_scores if s.final_score]
            bics = [s.bic for s in model_scores if s.bic]
            crps_vals = [s.crps for s in model_scores if s.crps]
            hyv_vals = [s.hyvarinen_score for s in model_scores if s.hyvarinen_score is not None]
            css_vals = [s.css_score for s in model_scores if s.css_score is not None]
            fec_vals = [s.fec_score for s in model_scores if s.fec_score is not None]
            times = [s.fit_time_ms for s in model_scores if s.fit_time_ms]
            pit_pass = sum(1 for s in model_scores if s.pit_calibrated)
            pit_total = len(model_scores)
            
            pit_rate = pit_pass / pit_total if pit_total > 0 else 0
            
            aggregated[name] = {
                'final': round(sum(finals) / len(finals), 2) if finals else 0.0,
                'bic': round(sum(bics) / len(bics), 0) if bics else 0,
                'crps': round(sum(crps_vals) / len(crps_vals), 4) if crps_vals else 0.0,
                'hyv': round(sum(hyv_vals) / len(hyv_vals), 1) if hyv_vals else 0.0,
                'pit': 'PASS' if pit_rate >= 0.75 else f'{pit_rate*100:.0f}%',
                'pit_rate': round(pit_rate, 2),
                'css': round(sum(css_vals) / len(css_vals), 2) if css_vals else 0.0,
                'fec': round(sum(fec_vals) / len(fec_vals), 2) if fec_vals else 0.0,
                'time_ms': round(sum(times) / len(times), 0) if times else 0,
                'n_tests': len(model_scores),
            }
        
        # Save results
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        output = {
            'last_updated': datetime.now().isoformat(),
            'n_symbols': len(ARENA_BENCHMARK_SYMBOLS),
            'n_models': len(safe_storage_classes),
            'n_successful_tests': sum(m['n_tests'] for m in aggregated.values()),
            'models': aggregated,
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        
        console.print()
        console.print(f"[green]✓[/green] Results saved to: {RESULTS_FILE}")
        
        # Display results table
        display_results_table(aggregated)
        
        return aggregated
        
    finally:
        # Restore original experimental models
        EXPERIMENTAL_MODELS.clear()
        EXPERIMENTAL_MODELS.update(original_models)
        EXPERIMENTAL_MODEL_KWARGS.clear()
        EXPERIMENTAL_MODEL_KWARGS.update(original_kwargs)
        EXPERIMENTAL_MODEL_SPECS.clear()
        EXPERIMENTAL_MODEL_SPECS.update(original_specs)
        # Restore disabled models
        save_disabled_models(original_disabled)


def display_results_table(aggregated: dict):
    """Display results in a formatted table."""
    sorted_models = sorted(aggregated.items(), key=lambda x: x[1]['final'], reverse=True)
    
    table = Table(title="Safe Storage Model Results", show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", style="green", width=30)
    table.add_column("Score", justify="right", width=7)
    table.add_column("BIC", justify="right", width=9)
    table.add_column("CRPS", justify="right", width=7)
    table.add_column("Hyv", justify="right", width=8)
    table.add_column("PIT", justify="center", width=6)
    table.add_column("CSS", justify="right", width=5)
    table.add_column("FEC", justify="right", width=5)
    table.add_column("Time", justify="right", width=6)
    
    for i, (name, m) in enumerate(sorted_models, 1):
        time_sec = m['time_ms'] / 1000 if m['time_ms'] else 0
        table.add_row(
            str(i),
            name,
            f"{m['final']:.1f}",
            f"{m['bic']:.0f}",
            f"{m['crps']:.4f}",
            f"{m['hyv']:.1f}",
            m['pit'],
            f"{m['css']:.2f}",
            f"{m['fec']:.2f}",
            f"{time_sec:.1f}s",
        )
    
    console.print(table)


if __name__ == "__main__":
    run_safe_storage_arena()