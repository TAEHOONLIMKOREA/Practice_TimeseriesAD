#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT(푸리에 변환) 기반 노이즈 제거 + 플로팅 + CSV 저장 스크립트

모든 수치형 컬럼(타임컬럼 제외)에 대해 FFT 기반 노이즈 제거를 수행합니다.

사용 예시:
  python fft_denoise.py \
      --input preprocessing.csv \
      --output preprocessing_fft_denoised.csv \
      --energy-threshold 0.98
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_time_col(df: pd.DataFrame):
    candidates = ["time", "t", "timestamp", "date", "datetime", "Time", "Time(s)", "TimeSeconds"]
    for c in df.columns:
        if str(c).strip() in candidates:
            return c
    return None


def build_time_axis(df: pd.DataFrame, time_col: str | None) -> np.ndarray:
    if time_col is None:
        return np.arange(len(df), dtype=float)
    try:
        t_parsed = pd.to_datetime(df[time_col], errors="raise", infer_datetime_format=True)
        t_sec = (t_parsed - t_parsed.iloc[0]).dt.total_seconds().to_numpy()
    except Exception:
        t_sec = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        if np.isnan(t_sec).mean() > 0.5:
            t_sec = np.arange(len(df), dtype=float)
    return t_sec


def median_dt(t_sec: np.ndarray) -> float:
    if t_sec.size < 2:
        return 1.0
    diffs = np.diff(t_sec)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def fft_denoise(x: np.ndarray, dt: float, energy_threshold: float | None, cutoff_hz: float | None):
    N = len(x)
    if N < 4:
        return x, None, None, None

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)

    if cutoff_hz is not None:
        mask = freqs <= cutoff_hz
    else:
        power = np.abs(X) ** 2
        total = power.sum() if power.sum() > 0 else 1.0
        cum = np.cumsum(power) / total
        thr = 0.98 if energy_threshold is None else float(energy_threshold)
        mask = cum <= thr

    X_f = X * mask
    x_f = np.fft.irfft(X_f, n=N)
    return x_f, X, X_f, freqs


def main():
    ap = argparse.ArgumentParser(description="FFT 기반 노이즈 제거 스크립트 (모든 수치형 컬럼)")
    ap.add_argument("--input", "-i", type=str, required=True, help="입력 CSV 경로")
    ap.add_argument("--output", "-o", type=str, default="preprocessing_fft_denoised.csv", help="저장할 출력 CSV 경로")
    ap.add_argument("--energy-threshold", type=float, default=0.98, help="누적 에너지 보존 임계값(0~1), 기본 0.98")
    ap.add_argument("--cutoff-hz", type=float, default=None, help="고정 저역 컷오프 주파수(Hz)")
    ap.add_argument("--save-plots-prefix", type=str, default=None, help="플롯 PNG 저장 시 파일명 접두사 (예: out/plot_)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {in_path}")

    df = pd.read_csv(in_path)

    # 시간 컬럼 감지
    time_col = detect_time_col(df)
    t_sec = build_time_axis(df, time_col)
    dt = median_dt(t_sec)
    fs = 1.0 / dt if dt > 0 else 1.0

    # 타임컬럼 제외한 numeric 컬럼
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    out_df = pd.DataFrame({"time_s": t_sec})

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        out_df[f"{col}_original"] = x
        if np.isnan(x).any():
            x = pd.Series(x).ffill().bfill().fillna(0.0).to_numpy()

        denoised, X, X_f, freqs = fft_denoise(
            x, dt,
            energy_threshold=None if args.cutoff_hz is not None else args.energy_threshold,
            cutoff_hz=args.cutoff_hz
        )
        out_df[f"{col}_denoised"] = denoised

        # 플롯 저장
        if args.save_plots_prefix:
            # 시간영역
            fig1 = plt.figure()
            plt.plot(t_sec, x, label="original")
            plt.plot(t_sec, denoised, label="denoised")
            plt.xlabel("Time (s)")
            plt.ylabel(col)
            plt.legend()
            plt.title(f"{col} - Time Domain")
            plt.tight_layout()
            fig1.savefig(f"{args.save_plots_prefix}{col}_time.png", dpi=150)
            plt.close(fig1)

            # 스펙트럼
            if X is not None:
                fig2 = plt.figure()
                plt.plot(freqs, np.abs(X), label="original")
                plt.plot(freqs, np.abs(X_f), label="denoised")
                plt.xlabel("Frequency (Hz) [approx.]")
                plt.ylabel("|X(f)|")
                plt.legend()
                plt.title(f"{col} - Spectrum")
                plt.tight_layout()
                fig2.savefig(f"{args.save_plots_prefix}{col}_spectrum.png", dpi=150)
                plt.close(fig2)
    
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] 전처리 CSV 저장 완료: {out_path.resolve()}")
    print(f"[INFO] 처리된 컬럼: {numeric_cols}")
    print(f"[INFO] 추정 샘플링 주기 dt={dt:.6f}s, fs≈{fs:.6f}Hz, N={len(t_sec)}")


if __name__ == "__main__":
    main()
