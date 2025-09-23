# -*- coding: utf-8 -*-
"""
원본 시계열 CSV에서 4가지 방법(점/구간/집단/변화점)을
'각각' 원본 데이터 위에 시각화(타임별 매칭)하여 저장하는 코드.

입력: ./data/preprocessing.csv  (열: Time, Layer, 나머지 수치 컬럼들)
출력:
  - method_summary.txt         (방법별 카운트/탐지율 요약)
  - plots/<column>__<method>.png  (각 컬럼 x 방법별 시각화 PNG)
필요 라이브러리: pandas, numpy, matplotlib
"""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============= 설정 =============
CSV_PATH = "./data/preprocessing_denoised.csv"  # 필요 시 변경
TIME_COL = "Time"
SAVE_PLOTS = True
PLOT_DIR = Path("./anomaly_statistical_methods2_denoised_rev02/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# IQR 점 이상치 기준
IQR_K = 1.5

# 구간(컨텍스트) 이상치: rolling robust z-score
ROLL_WINDOW = 301   # 데이터 간격에 맞춰 조정
Z_THRESH = 3.0

# 집단(연속) 이상치 최소 길이
COLLECTIVE_MIN_LEN = 30

# 변화점(CUSUM) 파라미터
CUSUM_THRESHOLD = 5.0
CUSUM_DRIFT = 0.0


# ============= 유틸 함수 =============
def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return df


def numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def iqr_bounds(s: pd.Series, k: float = IQR_K):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return low, high


def detect_point_outliers_iqr(s: pd.Series) -> pd.Series:
    low, high = iqr_bounds(s)
    return (s < low) | (s > high)


def rolling_zscore(s: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    # 강건 롤링 기준 (median / MAD 근사)
    med = s.rolling(window, center=True, min_periods=window//2).median()
    abs_dev = (s - med).abs()
    mad_approx = abs_dev.rolling(window, center=True, min_periods=window//2).median()
    mad_approx = mad_approx.replace(0, np.nan)
    z = (s - med) / (1.4826 * mad_approx)  # MAD 정규화 계수
    return z


def detect_contextual_outliers(s: pd.Series, z_thresh: float = Z_THRESH, window: int = ROLL_WINDOW) -> pd.Series:
    z = rolling_zscore(s, window=window)
    return z.abs() > z_thresh


def group_consecutive_true(mask: pd.Series, min_len: int = 1):
    """True가 연속된 구간을 (start_idx, end_idx) 리스트로 반환 (end_idx 포함)"""
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            if (prev - start + 1) >= min_len:
                runs.append((start, prev))
            start = i
            prev = i
    if (prev - start + 1) >= min_len:
        runs.append((start, prev))
    return runs


def detect_collective_outliers(s: pd.Series) -> pd.Series:
    # 컨텍스트 이상치 마스크에서 연속 길이가 기준 이상인 구간만 집단 이상치로 본다
    ctx_mask = detect_contextual_outliers(s)
    runs = group_consecutive_true(ctx_mask, min_len=COLLECTIVE_MIN_LEN)
    collective = pd.Series(False, index=s.index)
    for st, ed in runs:
        collective.iloc[st:ed+1] = True
    return collective


def cusum_change_points(s: pd.Series, threshold=CUSUM_THRESHOLD, drift=CUSUM_DRIFT):
    """간단 CUSUM(mean-shift) 변화점 인덱스 리스트"""
    x = s.values.astype(float)
    mean = np.nanmean(x)
    pos, neg = 0.0, 0.0
    change_points = []
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        pos = max(0.0, pos + (v - mean - drift))
        neg = max(0.0, neg - (v - mean + drift))
        if pos > threshold or neg > threshold:
            change_points.append(i)
            pos, neg = 0.0, 0.0
    return change_points

def detect_changepoint_mask(s: pd.Series) -> pd.Series:
    # 변화점 자체(지점)만 표시하기 위한 마스크
    cps = cusum_change_points(s)
    mask = pd.Series(False, index=s.index)
    for i in cps:
        if 0 <= i < len(mask):
            mask.iloc[i] = True
    return mask, cps


# ============= 데이터 로드 =============
df = pd.read_csv(CSV_PATH)
df = ensure_datetime(df, TIME_COL)
num_cols = numeric_columns(df)

n_rows = len(df)
n_points_total = n_rows * max(1, len(num_cols))

# ============= 방법별 탐지 카운트/탐지율 계산 =============
method_stats = {
    "point": 0,
    "contextual": 0,
    "collective": 0,
    "changepoint": 0,
}

for col in num_cols:
    s = df[col]

    # point
    pm = detect_point_outliers_iqr(s)
    method_stats["point"] += int(pm.sum())

    # contextual
    cm = detect_contextual_outliers(s)
    method_stats["contextual"] += int(cm.sum())

    # collective
    coll = detect_collective_outliers(s)
    method_stats["collective"] += int(coll.sum())

    # changepoint (지점 수로 계산)
    cp_mask, cps = detect_changepoint_mask(s)
    method_stats["changepoint"] += int(cp_mask.sum())

# 탐지율(전체 데이터포인트 대비 비율)
method_rates = {k: v / n_points_total for k, v in method_stats.items()}

# 요약 저장
with open("method_summary.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps({
        "n_rows": n_rows,
        "n_numeric_cols": len(num_cols),
        "n_points_total": n_points_total,
        "method_counts": method_stats,
        "method_rates": method_rates
    }, ensure_ascii=False, indent=2))

print("=== 방법별 카운트/탐지율 요약 ===")
print(json.dumps({
    "method_counts": method_stats,
    "method_rates": method_rates
}, ensure_ascii=False, indent=2))


# ============= 시각화 (각 방법별로 모두 저장) =============
def plot_series_with_method(t, y, title, method: str, save_path: Path):
    """원본 시계열 위에 특정 이상치 방법만 오버레이하여 단일 그래프 저장"""
    fig, ax = plt.subplots(figsize=(14, 4))

    # 원본 시계열
    ax.plot(
        t, y, linewidth=0.8, label="series",
        color="black", alpha=0.5
    )

    if method == "point":
        mask = detect_point_outliers_iqr(y)
        idx = np.where(mask.values)[0]
        if len(idx):
            ax.scatter(
                t.iloc[idx], y.iloc[idx],
                s=20, label="point outlier",
                color="red", alpha=0.5
            )

    elif method == "contextual":
        mask = detect_contextual_outliers(y)
        idx = np.where(mask.values)[0]
        if len(idx):
            ax.scatter(
                t.iloc[idx], y.iloc[idx],
                s=20, label="contextual outlier",
                marker="x", color="red", alpha=0.7
            )

    elif method == "collective":
        ctx_mask = detect_contextual_outliers(y)
        runs = group_consecutive_true(ctx_mask, min_len=COLLECTIVE_MIN_LEN)
        for i, (st, ed) in enumerate(runs):
            ax.axvspan(
                t.iloc[st], t.iloc[ed],
                alpha=0.3, color="red",
                label="collective outlier" if i == 0 else None
            )

    elif method == "changepoint":
        _, cps = detect_changepoint_mask(y)
        if len(cps):
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)
            ax.vlines(
                t.iloc[cps], ymin=y_min, ymax=y_max,
                linestyles="dashed", color="red", alpha=0.7,
                label="changepoint"
            )

    # 제목/축/범례
    ax.set_title(f"{title} | Outlier method: {method}", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel(title)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

ALL_METHODS = ["point", "contextual", "collective", "changepoint"]

if SAVE_PLOTS:
    for col in num_cols:
        for m in ALL_METHODS:
            out_path = PLOT_DIR / f"{col}__{m}.png"
            plot_series_with_method(
                df[TIME_COL], df[col], col, m, out_path
            )
    print("\n각 컬럼별로 모든 방법의 시각화 PNG가 저장되었습니다.")
    print(f"저장 경로: {PLOT_DIR.resolve()}")
