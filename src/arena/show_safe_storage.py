#!/usr/bin/env python3
"""Show models in safe storage with their arena scores and metrics."""

import os
import re
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Model metadata from arena competition
SAFE_STORAGE_MODELS = {
    "elite_hybrid_omega2": {
        "file": "elite_hybrid_omega2.py",
        "class": "EliteHybridOmega2Model",
        "final": 68.82,
        "bic": -43270,
        "crps": 0.0184,
        "hyv": 4535.8,
        "pit": "PASS",
        "css": 0.74,
        "fec": 0.87,
        "vs_std": "+10.7",
        "description": "BEST Gen18 - Q-shift filters, memory deflation, hierarchical stress",
    },
    "optimal_hyv_iota": {
        "file": "optimal_hyv_iota.py",
        "class": "OptimalHyvIotaModel",
        "final": 72.27,
        "bic": -29273,
        "crps": 0.0183,
        "hyv": 4645.5,
        "pit": "PASS",
        "css": 0.84,
        "fec": 0.88,
        "vs_std": "+13.9",
        "description": "BEST CSS/FEC (0.84/0.88) - Memory-smoothed deflation regime",
    },
    "dtcwt_qshift": {
        "file": "dtcwt_qshift.py",
        "class": "DTCWTQShiftModel",
        "final": 63.98,
        "bic": -33700,
        "crps": 0.0204,
        "hyv": 3729.1,
        "pit": "PASS",
        "css": 0.44,
        "fec": 0.79,
        "vs_std": "+7.2",
        "description": "Q-shift filters for improved frequency selectivity",
    },
    "dtcwt_magnitude_threshold": {
        "file": "dtcwt_magnitude_threshold.py",
        "class": "DTCWTMagnitudeThresholdModel",
        "final": 63.94,
        "bic": -24312,
        "crps": 0.0184,
        "hyv": 4299.4,
        "pit": "PASS",
        "css": 0.73,
        "fec": 0.85,
        "vs_std": "+7.1",
        "description": "Magnitude-based thresholding for noise reduction",
    },
    "dualtree_complex_wavelet": {
        "file": "dualtree_complex_wavelet.py",
        "class": "DualTreeComplexWaveletKalmanModel",
        "final": 63.90,
        "bic": -26003,
        "crps": 0.0207,
        "hyv": 3629.2,
        "pit": "75%",
        "css": 0.77,
        "fec": 0.81,
        "vs_std": "+7.1",
        "description": "Core DTCWT with phase-aware Kalman filtering",
    },
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
        "vs_std": "+4.6",
        "description": "BEST Hyvärinen (294) - Entropy-preserving inflation",
    },
    "elite_hybrid_eta": {
        "file": "elite_hybrid_eta.py",
        "class": "EliteHybridEtaModel",
        "final": 62.29,
        "bic": -21911,
        "crps": 0.0185,
        "hyv": 4227.6,
        "pit": "PASS",
        "css": 0.69,
        "fec": 0.84,
        "vs_std": "+5.5",
        "description": "Full ensemble combination with adaptive calibration",
    },
    "dtcwt_adaptive_levels": {
        "file": "dtcwt_adaptive_levels.py",
        "class": "DTCWTAdaptiveLevelsModel",
        "final": 62.24,
        "bic": -25112,
        "crps": 0.0192,
        "hyv": 4421.2,
        "pit": "PASS",
        "css": 0.56,
        "fec": 0.82,
        "vs_std": "+5.4",
        "description": "Adaptive decomposition levels based on signal length",
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
        "vs_std": "+3.0",
        "description": "Adaptive inflation based on calibration feedback",
    },
}


def show_safe_storage_table():
    """Display safe storage models in a formatted table."""
    if RICH_AVAILABLE:
        console = Console()
        
        # Summary panel
        console.print(Panel(
            f"Models: {len(SAFE_STORAGE_MODELS)}  |  Location: src/arena/safe_storage/",
            title="Safe Storage Summary",
            border_style="blue"
        ))
        
        # Main table
        table = Table(title="SAFE STORAGE MODELS (Archived Competition Winners)", 
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
        
        for i, (name, m) in enumerate(SAFE_STORAGE_MODELS.items(), 1):
            table.add_row(
                f"#{i}",
                name,
                f"{m['final']:.2f}",
                f"{m['bic']:.0f}",
                f"{m['crps']:.4f}",
                f"{m['hyv']:.1f}",
                m['pit'],
                f"{m['css']:.2f}",
                f"{m['fec']:.2f}",
                m['vs_std'],
            )
        
        console.print(table)
        console.print()
        
        # File details table
        table2 = Table(title="Model Files", show_header=True, header_style="bold magenta")
        table2.add_column("File", style="dim")
        table2.add_column("Class Name", style="yellow")
        table2.add_column("Description")
        
        for name, m in SAFE_STORAGE_MODELS.items():
            table2.add_row(m['file'], m['class'], m['description'])
        
        console.print(table2)
        console.print()
        console.print("[dim]To use a model: from src.arena.safe_storage.{file} import {Class}[/dim]")
        
    else:
        # Plain text output
        print("=" * 100)
        print("SAFE STORAGE MODELS")
        print("=" * 100)
        print(f"Models: {len(SAFE_STORAGE_MODELS)} | Location: src/arena/safe_storage/")
        print("-" * 100)
        print(f"{'Rank':<6} {'Model':<30} {'FINAL':>8} {'BIC':>10} {'CRPS':>8} {'Hyv':>10} {'PIT':>6} {'CSS':>6} {'FEC':>6} {'vs STD':>8}")
        print("-" * 100)
        
        for i, (name, m) in enumerate(SAFE_STORAGE_MODELS.items(), 1):
            print(f"#{i:<5} {name:<30} {m['final']:>8.2f} {m['bic']:>10.0f} {m['crps']:>8.4f} {m['hyv']:>10.1f} {m['pit']:>6} {m['css']:>6.2f} {m['fec']:>6.2f} {m['vs_std']:>8}")


def get_safe_storage_table_for_arena():
    """Return table data for arena output."""
    return SAFE_STORAGE_MODELS


if __name__ == "__main__":
    show_safe_storage_table()
