#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising & visualization utilities for sensor CSV time series.

- Moving Average
- Median Filter
- Savitzky–Golay

Usage:
    python denoise_compare.py --csv /path/to/preprocessing.csv \
      --cols OxygenHighChamber OxygenLowChamber \
      --window 21 --poly 3 \
      --out /path/to/denoised.csv \
      --plotdir ./plots
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def moving_average(series: pd.Series, window: int, center: bool = True) -> pd.Series:
    return series.rolling(window=window, center=center, min_periods=max(1, window//2)).mean()

def median_filter(series: pd.Series, window: int, center: bool = True) -> pd.Series:
    return series.rolling(window=window, center=center, min_periods=max(1, window//2)).median()

def savgol(series: pd.Series, window: int, poly: int) -> pd.Series:
    if window % 2 == 0:
        window += 1
    if window <= poly:
        window = poly + 3 if poly % 2 == 0 else poly + 2
    s = series.copy()
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
    y = s.values.astype(float)
    try:
        filtered = savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    except ValueError:
        filtered = y
    return pd.Series(filtered, index=series.index)

def plot_comparison(df: pd.DataFrame, col: str, window: int, poly: int, plotdir: Path = None):
    series = df[col]
    ma = moving_average(series, window)
    med = median_filter(series, window)
    sg = savgol(series, window, poly)

    plt.figure(figsize=(14, 6))
    plt.plot(series.values, label="Original", color="black", linewidth=1,alpha=0.5)
    plt.plot(ma.values, label=f"Moving Average ({window})", color="red", linewidth=1, alpha=0.8)
    plt.plot(med.values, label=f"Median Filter ({window})", color="blue", linewidth=1, alpha=0.5)
    plt.plot(sg.values, label=f"Savitzky–Golay ({window},{poly})", color="green", linewidth=1, alpha=0.5)
    plt.legend()
    plt.title(f"Noise Reduction Comparison on '{col}' (len={len(series)})")
    plt.xlabel("Time Index")
    plt.ylabel(col)
    plt.tight_layout()

    if plotdir:
        plotdir.mkdir(parents=True, exist_ok=True)
        out_path = plotdir / f"{col}_comparison.png"
        plt.savefig(out_path)
        print(f"[ok] Saved plot: {out_path}")
    plt.close()

def apply_filters(df: pd.DataFrame, cols: list, window: int, poly: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}_ma{window}"] = moving_average(out[c], window)
        out[f"{c}_med{window}"] = median_filter(out[c], window)
        out[f"{c}_sg{window}_{poly}"] = savgol(out[c], window, poly)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--cols", type=str, nargs="+", required=False, help="Columns to denoise")
    parser.add_argument("--window", type=int, default=21, help="Window size for rolling/SG")
    parser.add_argument("--poly", type=int, default=3, help="Polynomial order for SG")
    parser.add_argument("--out", type=str, default="", help="Optional path to save denoised CSV")
    parser.add_argument("--plotdir", type=str, default="", help="Optional directory to save plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = args.cols

    plotdir = Path(args.plotdir) if args.plotdir else None
    for col in cols:
        if col not in df.columns:
            print(f"[warn] Column '{col}' not in CSV. Skipping.")
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"[warn] Column '{col}' is not numeric. Skipping.")
            continue
        plot_comparison(df, col, args.window, args.poly, plotdir=plotdir)

    denoised = apply_filters(df, cols, args.window, args.poly)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        denoised.to_csv(args.out, index=False)
        print(f"[ok] Denoised CSV saved to: {args.out}")

if __name__ == "__main__":
    main()
