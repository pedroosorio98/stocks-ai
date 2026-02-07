# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 21:50:25 2026

@author: Pedro
"""

import pandas as pd
from pathlib import Path

# Check what files exist
ticker_folder = Path("Data/NVDA")

print("=== Checking NVDA data files ===\n")

# Check Alpha Vantage
av_path = ticker_folder / "alpha_vantage" / "csv"
if av_path.exists():
    print(f"✅ Alpha Vantage folder exists: {av_path}")
    csv_files = list(av_path.glob("*.csv"))
    print(f"   Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    # Try to load the stock price file
    for f in csv_files:
        if 'TIME_SERIES' in f.name or 'DAILY' in f.name:
            print(f"\n📊 Reading {f.name}...")
            df = pd.read_csv(f)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            print(f"\n   First 5 rows:")
            print(df.head())
            
            # Check for close column
            close_cols = [col for col in df.columns if 'close' in col.lower()]
            print(f"\n   Close columns found: {close_cols}")
            
            if close_cols:
                close_col = close_cols[0]
                prices = pd.to_numeric(df[close_col], errors='coerce')
                print(f"\n   Price stats for '{close_col}':")
                print(f"   Min: ${prices.min():.2f}")
                print(f"   Max: ${prices.max():.2f}")
                print(f"   Mean: ${prices.mean():.2f}")
else:
    print(f"❌ Alpha Vantage folder NOT found: {av_path}")

# Check Yahoo Finance
yf_path = ticker_folder / "yahoo_finance" / "csv"
if yf_path.exists():
    print(f"\n✅ Yahoo Finance folder exists: {yf_path}")
    csv_files = list(yf_path.glob("*.csv"))
    print(f"   Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    # Try to load historical data
    for f in csv_files:
        if 'historical' in f.name.lower() or 'price' in f.name.lower():
            print(f"\n📊 Reading {f.name}...")
            df = pd.read_csv(f)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            print(f"\n   First 5 rows:")
            print(df.head())
else:
    print(f"❌ Yahoo Finance folder NOT found: {yf_path}")