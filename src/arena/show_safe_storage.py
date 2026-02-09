#!/usr/bin/env python3
"""Show models in safe storage with their arena scores and metrics."""

import os
import json
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Paths
SAFE_STORAGE_DIR = Path(__file__).parent / "safe_storage"
DATA_DIR = SAFE_STORAGE_DIR / "data"
RESULTS_FILE = DATA_DIR / "safe_storage_results.json"


# Model registry - metadata only, no scores
MODEL_REGISTRY = {
    "elite_hybrid_omega2": {
        "file": "elite_hybrid_omega2.py",
        "class": "EliteHybridOmega2Model",
        "description": "BEST Gen18 - Q-shift filters, memory deflation, hierarchical stress",
    },
    "optimal_hyv_iota": {
        "file": "optimal_hyv_iota.py",
        "class": "OptimalHyvIotaModel",
        "description": "BEST CSS/FEC (0.84/0.88) - Memory-smoothed deflation regime",
    },
    "dtcwt_qshift": {
        "file": "dtcwt_qshift.py",
        "class": "DTCWTQShiftModel",
        "description": "Q-shift filters for improved frequency selectivity",
    },
    "dtcwt_magnitude_threshold": {
        "file": "dtcwt_magnitude_threshold.py",
        "class": "DTCWTMagnitudeThresholdModel",
        "description": "Magnitude-based thresholding for noise reduction",
    },
    "dualtree_complex_wavelet": {
        "file": "dualtree_complex_wavelet.py",
        "class": "DualTreeComplexWaveletKalmanModel",
        "description": "Core DTCWT with phase-aware Kalman filtering",
    },
    "elite_hybrid_eta": {
        "file": "elite_hybrid_eta.py",
        "class": "EliteHybridEtaModel",
        "description": "Full ensemble combination with adaptive calibration",
    },
    "dtcwt_adaptive_levels": {
        "file": "dtcwt_adaptive_levels.py",
        "class": "DTCWTAdaptiveLevelsModel",
        "description": "Adaptive decomposition levels based on signal length",
    },
    "css_omicron": {
        "file": "css_omicron.py",
        "class": "CssOmicronModel",
        "description": "Gen26 CSS-focused - Q-shift filters, memory deflation, high LL boost",
    },
    "css_variant_v5": {
        "file": "css_variant_v5.py",
        "class": "CssVariantV5Model",
        "description": "Best CSS variant - Aggressive deflation with high LL boost",
    },
    "malliavin_score_weight": {
        "file": "malliavin_score_weight.py",
        "class": "MalliavinScoreWeightModel",
        "description": "Malliavin derivative sensitivity weighting for likelihood",
    },
    "matrix_log_barrier": {
        "file": "matrix_log_barrier.py",
        "class": "MatrixLogBarrierModel",
        "description": "Log-det barrier for positive definiteness of covariance",
    },
    "resolvent_regularized": {
        "file": "resolvent_regularized.py",
        "class": "ResolventRegularizedModel",
        "description": "Resolvent operator Tikhonov regularization",
    },
    "free_probability": {
        "file": "free_probability.py",
        "class": "FreeProbabilityModel",
        "description": "Free convolution eigenvalue distribution of random matrices",
    },
    "martingale_stopping": {
        "file": "martingale_stopping.py",
        "class": "MartingaleStoppingModel",
        "description": "Doob maximal inequality martingale control",
    },
    "fisher_rao_metric": {
        "file": "fisher_rao_metric.py",
        "class": "FisherRaoMetricModel",
        "description": "Fisher-Rao information metric adaptive step sizing",
    },
    "adjoint_duality": {
        "file": "adjoint_duality.py",
        "class": "AdjointDualityModel",
        "description": "Adjoint functor pair free-forgetful on vol space",
    },
    "rg_fixed_point": {
        "file": "rg_fixed_point.py",
        "class": "RGFixedPointModel",
        "description": "RG fixed-point iteration for optimal deflation",
    },
}


def load_results() -> dict:
    """Load results from data file."""
    if not RESULTS_FILE.exists():
        return {}
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        return data.get('models', {})
    except Exception:
        return {}


def get_safe_storage_models() -> dict:
    """Get SAFE_STORAGE_MODELS dict with scores from data file."""
    results = load_results()
    
    models = {}
    for name, meta in MODEL_REGISTRY.items():
        scores = results.get(name, {})
        models[name] = {
            "file": meta["file"],
            "class": meta["class"],
            "description": meta["description"],
            "final": scores.get("final", 0.0),
            "bic": scores.get("bic", 0),
            "crps": scores.get("crps", 0.0),
            "hyv": scores.get("hyv", 0.0),
            "pit": scores.get("pit", "N/A"),
            "css": scores.get("css", 0.0),
            "fec": scores.get("fec", 0.0),
            "time_ms": scores.get("time_ms", 0),
            "vs_std": f"+{scores.get('final', 0) - 55:.1f}" if scores.get('final', 0) > 0 else "N/A",
        }
    
    return models


# For backward compatibility - computed dynamically
SAFE_STORAGE_MODELS = get_safe_storage_models()


# Archived models - no longer candidates for promotion
ARCHIVED_MODELS = {
    "hyv_aware_eta": {
        "file": "hyv_aware_eta.py",
        "class": "HyvAwareEtaModel",
        "final": 62.94,
        "bic": -23703,
        "crps": 0.0209,
        "hyv": 294.2,
        "pit": "67%",
        "css": 0.62,
        "fec": 0.78,
        "time_ms": 8104,
        "vs_std": "+4.6",
        "description": "BEST Hyvärinen (294) - Entropy-preserving inflation",
    },
    "dtcwt_vol_regime": {
        "file": "dtcwt_vol_regime.py",
        "class": "DTCWTVolRegimeModel",
        "final": 61.44,
        "bic": -24799,
        "crps": 0.0205,
        "hyv": -2392.3,
        "pit": "83%",
        "css": 0.66,
        "fec": 0.80,
        "time_ms": 7926,
        "vs_std": "+4.6",
        "description": "Volatility regime conditioning - PASSES ALL HARD GATES",
    },
    "stress_adaptive_inflation": {
        "file": "stress_adaptive_inflation.py",
        "class": "StressAdaptiveInflationModel",
        "final": 59.78,
        "bic": -22912,
        "crps": 0.0198,
        "hyv": 902.1,
        "pit": "PASS",
        "css": 0.51,
        "fec": 0.80,
        "time_ms": 8177,
        "vs_std": "+3.0",
        "description": "Adaptive inflation based on calibration feedback",
    },
}


def show_safe_storage_table():
    """Display safe storage models in a formatted table."""
    # Reload to get fresh data
    models = get_safe_storage_models()
    
    if RICH_AVAILABLE:
        console = Console()
        
        # Check if data exists
        results = load_results()
        last_updated = "Never"
        if RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE) as f:
                    data = json.load(f)
                last_updated = data.get('last_updated', 'Unknown')
            except:
                pass
        
        # Summary panel
        console.print(Panel(
            f"Models: {len(models)}  |  Location: src/arena/safe_storage/  |  Last tested: {last_updated}",
            title="Safe Storage Summary",
            border_style="blue"
        ))
        
        if not results:
            console.print("\n[yellow]⚠ No test results found. Run 'make arena-safe' to test models.[/yellow]\n")
        
        # Sort by final score descending
        sorted_models = sorted(models.items(), key=lambda x: x[1]['final'], reverse=True)
        
        # Main table
        table = Table(title="SAFE STORAGE MODELS (Promotion Candidates)", 
                     show_header=True, header_style="bold cyan")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Model", style="green")
        table.add_column("FINAL", justify="right")
        table.add_column("BIC", justify="right")
        table.add_column("CRPS", justify="right")
        table.add_column("Hyv", justify="right")
        table.add_column("PIT", justify="center")
        table.add_column("CSS", justify="right")
        table.add_column("FEC", justify="right")
        table.add_column("vs STD", justify="right", style="yellow")
        
        for i, (name, m) in enumerate(sorted_models, 1):
            final_str = f"{m['final']:.2f}" if m['final'] > 0 else "-"
            bic_str = f"{m['bic']:.0f}" if m['bic'] != 0 else "-"
            crps_str = f"{m['crps']:.4f}" if m['crps'] > 0 else "-"
            hyv_str = f"{m['hyv']:.1f}" if m['hyv'] != 0 else "-"
            css_str = f"{m['css']:.2f}" if m['css'] > 0 else "-"
            fec_str = f"{m['fec']:.2f}" if m['fec'] > 0 else "-"
            
            table.add_row(
                f"#{i}",
                name,
                final_str,
                bic_str,
                crps_str,
                hyv_str,
                m['pit'],
                css_str,
                fec_str,
                m['vs_std'],
            )
        
        console.print(table)
        console.print()
        
        # File details table
        table2 = Table(title="Model Files", show_header=True, header_style="bold magenta")
        table2.add_column("File", style="dim")
        table2.add_column("Class Name", style="yellow")
        table2.add_column("Description")
        
        for name, m in sorted_models:
            table2.add_row(m['file'], m['class'], m['description'])
        
        console.print(table2)
        console.print()
        console.print("[dim]To test models: make arena-safe[/dim]")
        console.print("[dim]To use a model: from arena.safe_storage.{file} import {Class}[/dim]")
        
    else:
        # Plain text output
        models = get_safe_storage_models()
        sorted_models = sorted(models.items(), key=lambda x: x[1]['final'], reverse=True)
        
        print("=" * 100)
        print("SAFE STORAGE MODELS")
        print("=" * 100)
        print(f"Models: {len(models)} | Location: src/arena/safe_storage/")
        print("-" * 100)
        print(f"{'Rank':<6} {'Model':<30} {'FINAL':>8} {'BIC':>10} {'CRPS':>8} {'Hyv':>10} {'PIT':>6} {'CSS':>6} {'FEC':>6} {'vs STD':>8}")
        print("-" * 100)
        
        for i, (name, m) in enumerate(sorted_models, 1):
            print(f"#{i:<5} {name:<30} {m['final']:>8.2f} {m['bic']:>10.0f} {m['crps']:>8.4f} {m['hyv']:>10.1f} {m['pit']:>6} {m['css']:>6.2f} {m['fec']:>6.2f} {m['vs_std']:>8}")


def get_safe_storage_table_for_arena():
    """Return table data for arena output."""
    return get_safe_storage_models()


if __name__ == "__main__":
    show_safe_storage_table()
