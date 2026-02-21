"""
===============================================================================
REPORTING — CLI Output and Calibration Tables
===============================================================================

This module provides rich CLI reporting for the tuning pipeline:

    - render_calibration_issues_table(): Apple-quality calibration report

All reporting functions use the Rich library for beautiful terminal output,
with graceful fallback to plain text if Rich is unavailable.
"""

from __future__ import annotations

import sys
from typing import Dict

import numpy as np


def render_calibration_issues_table(cache: Dict, failure_reasons: Dict) -> None:
    """
    Render a comprehensive, Apple-quality calibration issues table.
    
    Shows all assets with calibration problems:
    - PIT p-value < 0.05 (model predictions not well-calibrated)
    - High kurtosis (fat tails not captured)
    - Failed tuning
    - Regime collapse warnings
    
    Design: Clean, scannable, actionable.
    
    Args:
        cache: Dictionary mapping asset symbol to tuned parameters
        failure_reasons: Dictionary mapping failed asset symbol to error reason
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.rule import Rule
        from rich import box
    except ImportError:
        print("\n[Calibration issues table requires 'rich' library]")
        sys.stdout.flush()
        return
    
    try:
        console = Console(force_terminal=True, width=140)
        
        # Collect calibration issues
        issues = []
        
        # 1. Failed assets
        for asset, reason in failure_reasons.items():
            issues.append({
                'asset': asset,
                'issue_type': 'FAILED',
                'severity': 'critical',
                'pit_p': None,
                'ks_stat': None,
                'kurtosis': None,
                'model': '-',
                'q': None,
                'phi': None,
                'nu': None,
                'details': reason[:50] + '...' if len(reason) > 50 else reason
            })
        
        # 2. Calibration warnings from cache
        for asset, raw_data in cache.items():
            if 'global' in raw_data:
                data = raw_data['global']
            else:
                data = raw_data
            
            pit_p = data.get('pit_ks_pvalue')
            ks_stat = data.get('ks_statistic')
            kurtosis = data.get('std_residual_kurtosis') or data.get('excess_kurtosis')
            calibration_warning = data.get('calibration_warning', False)
            noise_model = data.get('noise_model', '')
            q_val = data.get('q')
            phi_val = data.get('phi')
            nu_val = data.get('nu')
            
            collapse_warning = raw_data.get('hierarchical_tuning', {}).get('collapse_warning', False)
            
            has_issue = False
            issue_type = []
            severity = 'ok'
            
            if calibration_warning or (pit_p is not None and pit_p < 0.05):
                has_issue = True
                issue_type.append('PIT < 0.05')
                severity = 'warning'
            
            if pit_p is not None and pit_p < 0.01:
                severity = 'critical'
            
            if kurtosis is not None and kurtosis > 6:
                has_issue = True
                issue_type.append('High Kurt')
                if severity != 'critical':
                    severity = 'warning'
            
            if collapse_warning:
                has_issue = True
                issue_type.append('Regime Collapse')
            
            if has_issue:
                # Determine model display name
                if 'unified' in noise_model.lower():
                    # Unified Student-t model
                    model_str = f"φ-T-Uni(ν={int(nu_val)})" if nu_val else "φ-T-Unified"
                elif 'student_t' in noise_model:
                    model_str = f"φ-T(ν={int(nu_val)})" if nu_val else "Student-t"
                elif 'gaussian' in noise_model and 'momentum' in noise_model.lower():
                    model_str = "φ-G+Mom"
                elif 'phi' in noise_model:
                    model_str = "φ-Gaussian"
                elif 'gaussian' in noise_model:
                    model_str = "Gaussian"
                elif 'momentum' in noise_model.lower():
                    model_str = "φ-G+Mom"
                else:
                    model_str = noise_model[:12] if noise_model else '-'
                
                issues.append({
                    'asset': asset,
                    'issue_type': ', '.join(issue_type),
                    'severity': severity,
                    'pit_p': pit_p,
                    'ks_stat': ks_stat,
                    'kurtosis': kurtosis,
                    'model': model_str,
                    'q': q_val,
                    'phi': phi_val,
                    'nu': nu_val,
                    'details': ''
                })
        
        # Sort by severity (critical first), then by PIT p-value
        severity_order = {'critical': 0, 'warning': 1, 'ok': 2}
        issues.sort(key=lambda x: (severity_order.get(x['severity'], 2), x.get('pit_p') or 1.0))
        
        # SECTION HEADER - Always show
        console.print()
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        
        section_header = Text()
        section_header.append("  📊  ", style="bold bright_cyan")
        section_header.append("CALIBRATION REPORT", style="bold bright_white")
        console.print(section_header)
        console.print()
        
        # Show success or issues
        if not issues:
            console.print()
            success_text = Text()
            success_text.append("  ✓ ", style="bold bright_green")
            success_text.append("All ", style="white")
            success_text.append(f"{len(cache)}", style="bold bright_cyan")
            success_text.append(" assets passed calibration checks", style="white")
            console.print(success_text)
            console.print()
            
            stats_text = Text()
            stats_text.append("    PIT p-value ≥ 0.05 for all models  ·  ", style="dim")
            stats_text.append("No regime collapse detected", style="dim")
            console.print(stats_text)
            console.print()
            return
        
        # ISSUES HEADER
        issues_header = Text()
        issues_header.append("  ⚠️  ", style="bold yellow")
        issues_header.append(f"{len(issues)} assets with calibration issues", style="bold yellow")
        console.print(issues_header)
        console.print()
        
        # SUMMARY STATS
        critical_count = sum(1 for i in issues if i['severity'] == 'critical')
        warning_count = sum(1 for i in issues if i['severity'] == 'warning')
        failed_count = sum(1 for i in issues if i['issue_type'] == 'FAILED')
        
        summary = Text()
        summary.append("    ", style="")
        if critical_count > 0:
            summary.append(f"{critical_count}", style="bold indian_red1")
            summary.append(" critical", style="dim")
            summary.append("   ·   ", style="dim")
        if warning_count > 0:
            summary.append(f"{warning_count}", style="bold yellow")
            summary.append(" warnings", style="dim")
            summary.append("   ·   ", style="dim")
        if failed_count > 0:
            summary.append(f"{failed_count}", style="bold red")
            summary.append(" failed", style="dim")
            summary.append("   ·   ", style="dim")
        summary.append(f"{len(cache)}", style="white")
        summary.append(" total assets", style="dim")
        
        console.print(summary)
        console.print()
        
        # ISSUES TABLE
        table = Table(
            show_header=True,
            header_style="bold white",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        
        table.add_column("Asset", justify="left", width=30, no_wrap=True)
        table.add_column("Issue", justify="left", width=18)
        table.add_column("PIT p", justify="right", width=8)
        table.add_column("KS", justify="right", width=6)
        table.add_column("Kurt", justify="right", width=6)
        table.add_column("Model", justify="left", width=12)
        table.add_column("log₁₀(q)", justify="right", width=9)
        table.add_column("φ", justify="right", width=6)
        table.add_column("Details", justify="left", width=25, no_wrap=True)
        
        for issue in issues:
            if issue['severity'] == 'critical':
                severity_style = "bold indian_red1"
                asset_style = "indian_red1"
            elif issue['severity'] == 'warning':
                severity_style = "yellow"
                asset_style = "yellow"
            else:
                severity_style = "dim"
                asset_style = "white"
            
            pit_str = f"{issue['pit_p']:.4f}" if issue['pit_p'] is not None else "-"
            ks_str = f"{issue['ks_stat']:.3f}" if issue['ks_stat'] is not None else "-"
            kurt_str = f"{issue['kurtosis']:.1f}" if issue['kurtosis'] is not None else "-"
            
            if issue['q'] is not None and issue['q'] > 0:
                log_q_str = f"{np.log10(issue['q']):.2f}"
            else:
                log_q_str = "-"
            
            phi_str = f"{issue['phi']:.3f}" if issue['phi'] is not None else "-"
            
            if issue['pit_p'] is not None:
                if issue['pit_p'] < 0.01:
                    pit_styled = f"[bold indian_red1]{pit_str}[/]"
                elif issue['pit_p'] < 0.05:
                    pit_styled = f"[yellow]{pit_str}[/]"
                else:
                    pit_styled = f"[dim]{pit_str}[/]"
            else:
                pit_styled = "[dim]-[/]"
            
            if issue['kurtosis'] is not None:
                if issue['kurtosis'] > 10:
                    kurt_styled = f"[bold indian_red1]{kurt_str}[/]"
                elif issue['kurtosis'] > 6:
                    kurt_styled = f"[yellow]{kurt_str}[/]"
                else:
                    kurt_styled = f"[dim]{kurt_str}[/]"
            else:
                kurt_styled = "[dim]-[/]"
            
            table.add_row(
                f"[{asset_style}]{issue['asset']}[/]",
                f"[{severity_style}]{issue['issue_type']}[/]",
                pit_styled,
                f"[dim]{ks_str}[/]",
                kurt_styled,
                f"[dim]{issue['model']}[/]",
                f"[dim]{log_q_str}[/]",
                f"[dim]{phi_str}[/]",
                f"[dim]{issue['details']}[/]",
            )
        
        console.print(table)
        console.print()
        
        # LEGEND
        legend = Text()
        legend.append("    ", style="")
        legend.append("PIT p < 0.05", style="yellow")
        legend.append(" = model may be miscalibrated   ·   ", style="dim")
        legend.append("Kurt > 6", style="yellow")
        legend.append(" = heavy tails not fully captured", style="dim")
        
        console.print(legend)
        console.print()
        
        # Action recommendation
        if critical_count > 0:
            action = Text()
            action.append("    → ", style="dim")
            action.append("Consider re-tuning critical assets with ", style="dim")
            action.append("make tune ARGS='--force --assets <TICKER>'", style="bold white")
            console.print(action)
            console.print()
    
    except Exception as e:
        # Fallback to simple print output if Rich fails
        print(f"\n[Calibration report error: {e}]")
        print("\n📊 CALIBRATION REPORT (text fallback)")
        print("-" * 60)
        
        issue_count = 0
        for asset, raw_data in cache.items():
            data = raw_data.get('global', raw_data)
            if data.get('calibration_warning') or (data.get('pit_ks_pvalue') or 1.0) < 0.05:
                issue_count += 1
                print(f"  ⚠️  {asset}: PIT p={data.get('pit_ks_pvalue', 'N/A')}")
        
        for asset in failure_reasons:
            print(f"  ❌ {asset}: FAILED")
        
        if issue_count == 0 and not failure_reasons:
            print(f"  ✓ All {len(cache)} assets passed calibration checks")
        
        sys.stdout.flush()
