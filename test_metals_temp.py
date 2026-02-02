#!/usr/bin/env python3
"""Test script for metals risk temperature diagnostic."""
import sys
sys.path.insert(0, 'src')

from decision.metals_risk_temperature import compute_anticipatory_metals_risk_temperature

result, alerts, quality = compute_anticipatory_metals_risk_temperature()

print('=== METALS RISK TEMPERATURE ===')
print(f'Temperature: {result.temperature:.2f}')
print(f'Status: {result.status}')
print(f'Regime State: {result.regime_state}')
print(f'Gap Risk: {result.gap_risk_estimate:.1%}')
print(f'Crash Risk %: {result.crash_risk_pct:.1%}')
print(f'Crash Risk Level: {result.crash_risk_level}')
print(f'Vol Inversion Count: {result.vol_inversion_count}')
print()
print('=== INDIVIDUAL METALS ===')
for name, metal in result.metals.items():
    if metal.data_available:
        print(f'{name}: 5d={metal.return_5d:+.1%}, 21d={metal.return_21d:+.1%}, vol={metal.volatility:.1%}')
print()
print('=== ALERTS ===')
for a in alerts:
    print(f'{a.get("severity")}: {a.get("message")}')
