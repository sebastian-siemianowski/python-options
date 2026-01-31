#!/usr/bin/env python3
"""
test_kelly_integration.py

Integration test: Build Kelly portfolio from real asset signals.
Uses signals.py to compute features and expected returns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from decision.signals import fetch_px_asset, compute_features
from portfolio_kelly import build_multi_asset_portfolio
from signals_ux import render_portfolio_allocation_table


def main():
    print("=" * 80)
    print("Kelly Portfolio Integration Test")
    print("=" * 80)
    
    # Define test assets
    test_assets = {
        'PLNJPY=X': 'PLNJPY=X',
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Bitcoin': 'BTC-USD',
    }
    
    # Fetch data and compute features
    print("\n📊 Fetching asset data and computing features...")
    asset_returns = {}
    asset_expected_returns = {}
    
    for name, symbol in test_assets.items():
        try:
            print(f"  Processing {name} ({symbol})...")
            px, title = fetch_px_asset(symbol, start='2020-01-01', end=None)
            feats = compute_features(px)
            
            # Extract returns series
            ret_series = feats.get('ret')
            if ret_series is not None and not ret_series.empty:
                asset_returns[name] = ret_series
                
                # Extract expected return (daily drift from Kalman filter or posterior)
                mu_post = feats.get('mu_post')
                if mu_post is not None and not mu_post.empty:
                    mu_daily = float(mu_post.iloc[-1])
                else:
                    mu_daily = 0.0
                
                asset_expected_returns[name] = mu_daily
                
                print(f"    ✓ {len(ret_series)} observations, μ_daily = {mu_daily:+.6f}")
            else:
                print(f"    ✗ No returns data")
                
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue
    
    print(f"\n✓ Successfully loaded {len(asset_returns)} assets")
    
    # Build Kelly portfolio
    if len(asset_returns) >= 2:
        print("\n" + "=" * 80)
        print("Building Kelly Portfolio (21-day horizon, half-Kelly)")
        print("=" * 80)
        
        try:
            portfolio = build_multi_asset_portfolio(
                asset_returns=asset_returns,
                asset_expected_returns=asset_expected_returns,
                horizon_days=21,
                ewma_lambda=0.94,
                risk_fraction=0.5,
            )
            
            print("\n✓ Portfolio optimization successful!")
            
            # Display portfolio allocation
            render_portfolio_allocation_table(
                portfolio_result=portfolio,
                horizon_days=21,
                notional_pln=1_000_000.0
            )
            
            print("\n" + "=" * 80)
            print("Integration Test PASSED ✓")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n✗ Portfolio construction failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    else:
        print(f"\n✗ Insufficient assets: need at least 2, got {len(asset_returns)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
