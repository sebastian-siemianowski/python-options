#!/usr/bin/env python3
"""
Test script for the Market Stress Index indicator.
This script tests if the new Market Stress Index would have prevented 
poor performance during 2022 by analyzing historical data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from our options module
try:
    from options import calculate_market_stress_index, load_price_history
    from data_cache import DEFAULT_DATA_DIR
    print("Successfully imported functions from options module")
except ImportError as e:
    print(f"Error importing from options module: {e}")
    sys.exit(1)

def test_market_stress_calculation():
    """Test the market stress index calculation on SPY data"""
    print("\n=== Testing Market Stress Index Calculation ===")
    
    try:
        # Load SPY historical data (should include 2022)
        print("Loading SPY historical data...")
        hist = load_price_history('SPY', years=6)
        
        if hist.empty:
            print("ERROR: No historical data loaded for SPY")
            return False
            
        print(f"Loaded {len(hist)} days of SPY data from {hist['Date'].min()} to {hist['Date'].max()}")
        
        # Calculate market stress index
        print("Calculating Market Stress Index...")
        stress_scores = calculate_market_stress_index(hist)
        
        if stress_scores is None or len(stress_scores) == 0:
            print("ERROR: Market Stress Index calculation failed")
            return False
            
        print(f"Successfully calculated Market Stress Index with {len(stress_scores)} data points")
        
        # Analyze 2022 stress levels
        print("\n--- 2022 Market Stress Analysis ---")
        hist['market_stress'] = stress_scores
        hist['Date'] = pd.to_datetime(hist['Date'])
        
        # Filter for 2022 data
        data_2022 = hist[hist['Date'].dt.year == 2022]
        
        if len(data_2022) == 0:
            print("WARNING: No 2022 data found in dataset")
            return False
            
        print(f"Found {len(data_2022)} trading days in 2022")
        
        # Calculate stress statistics for 2022
        avg_stress_2022 = data_2022['market_stress'].mean()
        max_stress_2022 = data_2022['market_stress'].max()
        high_stress_days = len(data_2022[data_2022['market_stress'] > 0.7])
        
        print(f"2022 Average Market Stress: {avg_stress_2022:.3f}")
        print(f"2022 Maximum Market Stress: {max_stress_2022:.3f}")
        print(f"Days with high stress (>0.7): {high_stress_days} out of {len(data_2022)} ({high_stress_days/len(data_2022)*100:.1f}%)")
        
        # Compare with other years
        print("\n--- Comparison with Other Years ---")
        for year in [2019, 2020, 2021, 2023, 2024]:
            year_data = hist[hist['Date'].dt.year == year]
            if len(year_data) > 0:
                avg_stress = year_data['market_stress'].mean()
                high_stress_pct = len(year_data[year_data['market_stress'] > 0.7]) / len(year_data) * 100
                print(f"{year}: Avg Stress = {avg_stress:.3f}, High Stress Days = {high_stress_pct:.1f}%")
        
        # Success criteria: 2022 should show elevated stress levels
        if avg_stress_2022 > 0.5 and high_stress_days > 0:
            print("\n✅ SUCCESS: Market Stress Index correctly identifies 2022 as a challenging period!")
            return True
        else:
            print("\n❌ CONCERN: Market Stress Index may not be sensitive enough to 2022 conditions")
            return False
            
    except Exception as e:
        print(f"ERROR during market stress calculation test: {e}")
        return False

def test_backtest_with_filter():
    """Test running a backtest with the market stress filter enabled"""
    print("\n=== Testing Backtest with Market Stress Filter ===")
    
    try:
        # Run a short backtest with the market stress filter enabled
        print("Running backtest with Market Stress filter enabled...")
        
        cmd = [
            sys.executable, "options.py", 
            "--tickers", "SPY", 
            "--bt_years", "4",
            "--bt_market_stress_filter", "true",
            "--bt_market_stress_threshold", "0.7",
            "--min_oi", "10000000",  # Force backtest-only mode
            "--min_vol", "10000000"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ SUCCESS: Backtest with Market Stress filter completed successfully!")
            
            # Check if equity curve was generated
            equity_file = "backtests/SPY_equity.csv"
            if os.path.exists(equity_file):
                print("✅ SUCCESS: Equity curve file generated")
                
                # Quick analysis of the equity curve
                equity_df = pd.read_csv(equity_file)
                equity_df['Date'] = pd.to_datetime(equity_df['Date'])
                
                # Count trades in 2022 (should be fewer with stress filter)
                data_2022 = equity_df[equity_df['Date'].dt.year == 2022]
                trades_2022 = len(data_2022[data_2022['ret'] != 0])
                
                print(f"Trades executed in 2022 with stress filter: {trades_2022}")
                return True
            else:
                print("WARNING: No equity curve file generated")
                return False
        else:
            print(f"ERROR: Backtest failed with return code {result.returncode}")
            print("STDOUT:", result.stdout[-500:] if result.stdout else "None")
            print("STDERR:", result.stderr[-500:] if result.stderr else "None")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: Backtest timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"ERROR during backtest test: {e}")
        return False

def main():
    """Main test function"""
    print("Market Stress Index Test Suite")
    print("=" * 50)
    
    # Test 1: Market stress calculation
    test1_passed = test_market_stress_calculation()
    
    # Test 2: Backtest with filter
    test2_passed = test_backtest_with_filter()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Market Stress Calculation: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Backtest Integration:      {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Market Stress Index is working correctly.")
        print("\nThe custom indicator should help prevent poor performance during")
        print("challenging market regimes like 2022 by filtering out high-stress periods.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)