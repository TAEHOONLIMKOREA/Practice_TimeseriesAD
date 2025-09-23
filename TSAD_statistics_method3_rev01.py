# -*- coding: utf-8 -*-
"""
전처리/원본 모두 이상탐지 수행하되,
시각화는 '전처리 칼럼에서 탐지한 이상치'만 '원본 시계열' 위에 오버레이.

입력: ./data/preprocessing_denoised.csv
출력:
  - method_summary.txt
  - plots/<base_column>__<method>.png   (전처리 기준 이상치만 원본에 overlay)
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============= 설정 =============
CSV_PATH = "./data/preprocessing2_denoised.csv"
TIME_COL = "Time"
SAVE_PLOTS = True
PLOT_DIR = Path("./anomaly_statistical_methods3_denoised2_rev03/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# IQR 점 이상치
IQR_K = 1.5

# 구간(컨텍스트) 이상치: 강건 롤링 z-score
ROLL_WINDOW = 301
Z_THRESH = 3.0

# 집단(연속) 이상치
COLLECTIVE_MIN_LEN = 30

# 큰 시간 갭을 집단 이상으로 간주
GAP_MULT = 8.0

# 추세 이상(Trend): 롤링 선형회귀 slope의 강건 z-score
TREND_WINDOW = 301
TREND_Z = 3.0
TREND_MIN_LEN = 30
TREND_LINE_WINDOW = 301

# ============= 유틸 =============
def ensure_datetime(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)

def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# suffix 규칙: _denoised, _maXX, _medXX, _sgXX_YY, _original
PROC_SUFFIX_RE = r'(_denoised|_ma\d+|_med\d+|_sg\d+_\d+)$'
ORIG_SUFFIX = '_original'

def get_base_name(col: str):
    # 먼저 _original 제거, 그다음 전처리 suffix 제거
    col = re.sub(fr'{ORIG_SUFFIX}$', '', col)
    col = re.sub(PROC_SUFFIX_RE, '', col)
    return col

def find_original_col(df: pd.DataFrame, base: str):
    """원본 컬럼 선택 우선순위: base(그대로) > base_original"""
    if base in df.columns:
        return base
    alt = f"{base}{ORIG_SUFFIX}"
    return alt if alt in df.columns else None

def iqr_bounds(s, k=IQR_K):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def detect_point_outliers_iqr(s):
    low, high = iqr_bounds(s)
    return (s < low) | (s > high)

def rolling_zscore(s, window=ROLL_WINDOW):
    med = s.rolling(window, center=True, min_periods=window//2).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window//2).median()
    mad = mad.replace(0, np.nan)
    return (s - med) / (1.4826 * mad)

def detect_contextual_outliers(s, z_thresh=Z_THRESH, window=ROLL_WINDOW):
    z = rolling_zscore(s, window=window)
    return z.abs() > z_thresh

def group_consecutive_true(mask, min_len=1):
    idx = np.where(mask.values)[0]
    if len(idx) == 0: return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            if prev - start + 1 >= min_len: runs.append((start, prev))
            start, prev = i, i
    if prev - start + 1 >= min_len: runs.append((start, prev))
    return runs

def detect_collective_outliers(s, t):
    """컨텍스트 연속 + 큰 시간 갭을 집단 이상으로."""
    ctx_mask = detect_contextual_outliers(s)
    runs = group_consecutive_true(ctx_mask, min_len=COLLECTIVE_MIN_LEN)
    collective = pd.Series(False, index=s.index)
    for st, ed in runs:
        collective.iloc[st:ed+1] = True

    # 큰 시간 갭
    t_s = pd.to_datetime(t)
    dt = t_s.diff().dt.total_seconds()
    dt_med = np.nanmedian(dt.values)
    gap_threshold = dt_med * GAP_MULT if np.isfinite(dt_med) and dt_med > 0 else None
    gap_spans = []
    if gap_threshold is not None:
        gap_idx = np.where(dt.values > gap_threshold)[0]
        for i in gap_idx:
            if 0 <= i-1 < len(t_s) and 0 <= i < len(t_s):
                gap_spans.append((t_s.iloc[i-1], t_s.iloc[i]))
                collective.iloc[max(i-2,0):min(i+1,len(collective)-1)+1] = True
    return collective, runs, gap_spans

# 추세 이상(Trend)
def _rolling_linear_slope(t_sec, y, window):
    n, half = len(y), window // 2
    slopes = np.full(n, np.nan)
    for i in range(n):
        L, R = max(0,i-half), min(n,i+half+1)
        if R-L >= max(10, window//2):
            xw, yw = t_sec[L:R], y[L:R]
            A = np.vstack([xw, np.ones_like(xw)]).T
            coeff, *_ = np.linalg.lstsq(A, yw, rcond=None)
            slopes[i] = coeff[0]
    return slopes

def _robust_z(series_like: pd.Series):
    med = np.nanmedian(series_like)
    mad = np.nanmedian(np.abs(series_like - med))
    if mad == 0 or not np.isfinite(mad):
        return pd.Series(np.nan, index=series_like.index)
    return (series_like - med) / (1.4826 * mad)

def detect_trend_anomaly(s, t):
    t_sec = pd.to_datetime(t).astype("int64") // 10**9
    slopes = _rolling_linear_slope(t_sec.values.astype(float), s.values.astype(float), TREND_WINDOW)
    z = _robust_z(pd.Series(slopes, index=s.index))
    trend_mask = (z.abs() > TREND_Z).fillna(False)
    runs = group_consecutive_true(trend_mask, min_len=TREND_MIN_LEN)
    trend_line = s.rolling(TREND_LINE_WINDOW, center=True, min_periods=TREND_LINE_WINDOW//2).median()
    return trend_mask, runs, trend_line

# ============= 데이터 로드 & 칼럼 분리 =============
df = pd.read_csv(CSV_PATH)
df = ensure_datetime(df, TIME_COL)
num_cols = numeric_columns(df)

# base → original 매핑
base_to_original = {}
for col in num_cols:
    base = get_base_name(col)
    orig = find_original_col(df, base)
    if orig and base not in base_to_original:
        base_to_original[base] = orig  # base가 여러 번 등장해도 최초(접미어 없는 것) 우선

# 전처리/원본 구분
processed_cols = []
original_cols = set(base_to_original.values())
for col in num_cols:
    if col in original_cols:
        continue
    # 전처리 패턴에 맞으면 processed로
    if re.search(PROC_SUFFIX_RE, col) or col.endswith(ORIG_SUFFIX) is False:
        # 안전하게: 원본이 아니고 suffix 매칭되면 processed로
        if re.search(PROC_SUFFIX_RE, col):
            processed_cols.append(col)

# ============= 탐지(원본+전처리 모두) & 통계 =============
method_stats = {m: 0 for m in ["point", "contextual", "collective", "trend"]}

def accumulate_stats(series, t):
    global method_stats
    method_stats["point"]       += int(detect_point_outliers_iqr(series).sum())
    method_stats["contextual"]  += int(detect_contextual_outliers(series).sum())
    method_stats["collective"]  += int(detect_collective_outliers(series, t)[0].sum())
    method_stats["trend"]       += int(detect_trend_anomaly(series, t)[0].sum())

# 1) 원본 칼럼에도 이상탐지 수행(통계만)
for base, orig in base_to_original.items():
    s = df[orig]
    accumulate_stats(s, df[TIME_COL])

# 2) 전처리 칼럼에도 이상탐지 수행(통계 + 이후 플롯 대상)
for pcol in processed_cols:
    accumulate_stats(df[pcol], df[TIME_COL])

# 탐지율(모든 탐지 대상 포인트 수로 나눔: 원본+전처리)
n_cols_for_stats = len(base_to_original) + len(processed_cols)
n_points_total = len(df) * max(1, n_cols_for_stats)
method_rates = {k: (v / n_points_total) for k, v in method_stats.items()}

with open("method_summary.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps({
        "n_rows": len(df),
        "n_cols_in_stats": n_cols_for_stats,
        "method_counts": method_stats,
        "method_rates": method_rates,
        "notes": {
            "collective": "컨텍스트 연속 이상 + 큰 시간 갭 포함",
            "trend": "롤링 선형회귀 slope의 강건 z-score 기반",
            "plots": "전처리 칼럼에서 탐지한 이상치를 원본 위에 overlay"
        }
    }, ensure_ascii=False, indent=2))

print("=== 방법별 카운트/탐지율 요약 ===")
print(json.dumps({"method_counts": method_stats, "method_rates": method_rates},
                 ensure_ascii=False, indent=2))

# ============= 시각화 (원본 + 전처리 모두 figure 생성) =============
def plot_series_with_method(t, y_base, y_mask_src, title, method, save_path):
    """
    y_base: 시각화에 그릴 원본 시계열
    y_mask_src: 이상치 탐지할 대상 시계열 (원본 혹은 전처리)
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y_base, linewidth=0.8, label="base(original)", color="black", alpha=0.6)

    if method == "point":
        mask = detect_point_outliers_iqr(y_mask_src)
        ax.scatter(t[mask], y_base[mask], s=20, c="red", label="point")

    elif method == "contextual":
        mask = detect_contextual_outliers(y_mask_src)
        ax.scatter(t[mask], y_base[mask], s=20, c="red", marker="x", label="contextual")

    elif method == "collective":
        _, runs, gaps = detect_collective_outliers(y_mask_src, t)
        for (st, ed) in runs:
            ax.axvspan(t.iloc[st], t.iloc[ed], alpha=0.3, color="red", label="collective")
        for (ts, te) in gaps:
            ax.axvspan(ts, te, alpha=0.2, color="orange", label="gap")

    elif method == "trend":
        _, runs, trend_line = detect_trend_anomaly(y_mask_src, t)
        ax.plot(t, trend_line, c="C0", lw=1.2, label="trend line")
        for (st, ed) in runs:
            ax.axvspan(t.iloc[st], t.iloc[ed], alpha=0.3, color="red", label="trend")

    ax.set_title(f"{title} | {method} (mask from {y_mask_src.name}, plotted on {y_base.name})")
    ax.set_xlabel("Time"); ax.set_ylabel(title)
    ax.legend(loc="best"); plt.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close(fig)

ALL_METHODS = ["point", "contextual", "collective", "trend"]

if SAVE_PLOTS:
    # 1) 원본 칼럼들: 자기 자신에 대해 plot
    for base, orig in base_to_original.items():
        for m in ALL_METHODS:
            out_path = PLOT_DIR / f"{orig}__{m}.png"
            plot_series_with_method(df[TIME_COL], df[orig], df[orig], base, m, out_path)

    # 2) 전처리 칼럼들: 원본 위에 overlay
    for pcol in processed_cols:
        base = get_base_name(pcol)
        orig = find_original_col(df, base)
        if orig is None: continue
        for m in ALL_METHODS:
            out_path = PLOT_DIR / f"{pcol}__{m}.png"
            plot_series_with_method(df[TIME_COL], df[orig], df[pcol], base, m, out_path)

    print(f"\n완료: figure 총 { (len(base_to_original)+len(processed_cols)) * len(ALL_METHODS) } 개 저장됨")
    print(f"저장 경로: {PLOT_DIR.resolve()}")
