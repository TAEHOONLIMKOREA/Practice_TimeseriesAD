# -*- coding: utf-8 -*-
"""
시계열 CSV에서 이상치를 탐지하고 유형(점/구간/집단/변화점)으로 분류하는 예시 코드.
- 입력: /mnt/data/preprocessing.csv  (열: Time, Layer, 6개 수치 컬럼)
- 출력:
  - anomalies_summary.json  (요약 통계/유형)
  - anomalies_long.csv      (이상치 레코드 롱테이블)
  - change_points.csv       (변화점 목록)
  - (선택) 시각화 PNG 파일들
필요 라이브러리: pandas, numpy, matplotlib (표준)
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============= 설정 =============
CSV_PATH = "./data/preprocessing.csv"  # 파일 경로 변경 가능
TIME_COL = "Time"
# 시각화 저장 여부
SAVE_PLOTS = True
PLOT_DIR = Path("./anomaly_statistical_methods2/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# IQR 점 이상치 기준
IQR_K = 1.5
# 구간(컨텍스트) 이상치: 굴리는 윈도우와 임계치(z-score)
ROLL_WINDOW = 301  # 홀수 권장 (약 5분/10분 등 데이터 간격에 맞춰 조정)
Z_THRESH = 3.0
# 집단(연속) 이상치: 연속 길이 기준
COLLECTIVE_MIN_LEN = 30
# 변화점(CUSUM) 파라미터
CUSUM_THRESHOLD = 5.0
CUSUM_DRIFT = 0.0  # 평균 이동이 작으면 0~0.5 정도로 소량 드리프트 추가 고려


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
    # 강건한 기준(중앙값/MAE 근사) 사용: Median/ MAD 대용으로 |x-med|의 rolling median
    med = s.rolling(window, center=True, min_periods=window//2).median()
    abs_dev = (s - med).abs()
    mad_approx = abs_dev.rolling(window, center=True, min_periods=window//2).median()
    # 분산 0 방지
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

def cusum_change_points(s: pd.Series, threshold=CUSUM_THRESHOLD, drift=CUSUM_DRIFT):
    """
    간단 CUSUM(mean-shift) 탐지. 반환: 변화점 인덱스 리스트
    참고: 파라미터는 데이터 스케일에 맞게 조정 필요.
    """
    x = s.values.astype(float)
    mean = np.nanmean(x)
    pos, neg = 0.0, 0.0
    change_points = []
    for i, v in enumerate(x):
        if math.isnan(v):
            continue
        pos = max(0.0, pos + (v - mean - drift))
        neg = max(0.0, neg - (v - mean + drift))
        if pos > threshold or neg > threshold:
            change_points.append(i)
            pos, neg = 0.0, 0.0  # 재시작
    return change_points


# ============= 메인 로직 =============
df = pd.read_csv(CSV_PATH)
df = ensure_datetime(df, TIME_COL)
num_cols = numeric_columns(df)

results_long = []           # 각 이상치 레코드 목록 (롱테이블)
change_points_long = []     # 변화점 목록
summary = {}                # 변수별 요약

for col in num_cols:
    s = df[col]

    # 1) 점 이상치(IQR)
    point_mask = detect_point_outliers_iqr(s)

    # 2) 구간(컨텍스트) 이상치(rolling robust z-score)
    ctx_mask = detect_contextual_outliers(s)

    # 3) 집단 이상치(연속 길이로 판단) - 컨텍스트 이상치 중 연속 run이 긴 구간
    runs = group_consecutive_true(ctx_mask, min_len=COLLECTIVE_MIN_LEN)
    collective_mask = pd.Series(False, index=s.index)
    for st, ed in runs:
        collective_mask.iloc[st:ed+1] = True

    # 4) 변화점(CUSUM)
    cps = cusum_change_points(s, threshold=CUSUM_THRESHOLD, drift=CUSUM_DRIFT)

    # 유형 태깅 (우선순위: 변화점/집단 > 구간 > 점)
    type_tags = np.array([""] * len(s), dtype=object)
    type_tags[point_mask.values] = "point"
    type_tags[ctx_mask.values] = "contextual"
    type_tags[collective_mask.values] = "collective"

    # 변화점은 별도로 기록(해당 시점 전후 구간의 분포 변화 신호)
    for cp in cps:
        change_points_long.append({
            "column": col,
            "index": int(cp),
            "time": df.loc[cp, TIME_COL],
            "value": float(s.iloc[cp]),
            "type": "changepoint"
        })

    # 요약 통계
    summary[col] = {
        "n": int(len(s)),
        "point_outliers": int(point_mask.sum()),
        "contextual_outliers": int(ctx_mask.sum()),
        "collective_runs": [{"start_idx": int(st), "end_idx": int(ed), "length": int(ed-st+1),
                             "start_time": str(df.loc[st, TIME_COL]),
                             "end_time": str(df.loc[ed, TIME_COL])} for st, ed in runs],
        "changepoints": [{"idx": int(i), "time": str(df.loc[i, TIME_COL]), "value": float(s.iloc[i])} for i in cps],
        "bounds_iqr": {
            "low": float(iqr_bounds(s)[0]),
            "high": float(iqr_bounds(s)[1]),
        },
        "min": float(s.min()) if pd.notna(s.min()) else None,
        "max": float(s.max()) if pd.notna(s.max()) else None,
    }

    # 롱테이블(레코드 단위)
    for i in range(len(s)):
        tag = type_tags[i]
        if tag:
            results_long.append({
                "time": df.loc[i, TIME_COL],
                "index": i,
                "column": col,
                "value": float(s.iloc[i]),
                "anomaly_type": tag
            })

# 저장
out_long = pd.DataFrame(results_long)
out_cps = pd.DataFrame(change_points_long)
with open("anomalies_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
out_long.to_csv("anomalies_long.csv", index=False)
out_cps.to_csv("change_points.csv", index=False)

print("=== 요약 ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))

print("\n파일 저장 완료:")
print("- anomalies_summary.json")
print("- anomalies_long.csv")
print("- change_points.csv")

# ============= (선택) 시각화 =============
def plot_series_with_marks(t, y, title, point_mask, ctx_mask, runs, cps, save_path: Path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y, linewidth=0.8, label="series", color="black", alpha=0.5)
    # 점 이상치
    pm_idx = np.where(point_mask.values)[0]
    if len(pm_idx):
        ax.scatter(t.iloc[pm_idx], y.iloc[pm_idx], 
        s=10, label="point", 
        marker="o", color="red", alpha=0.5)
    # 구간 이상치
    cm_idx = np.where(ctx_mask.values)[0]
    if len(cm_idx):
        ax.scatter(t.iloc[cm_idx], y.iloc[cm_idx], 
        s=10, label="contextual", marker="o",
        color="blue", alpha=0.5)
    # 집단 이상치 영역
    for st, ed in runs:
        ax.axvspan(t.iloc[st], t.iloc[ed], 
        alpha=0.15, color="green",
        label="collective" if st == runs[0][0] else None)
    # 변화점
    if len(cps):
        ax.vlines(t.iloc[cps], 
        ymin=np.nanmin(y), ymax=np.nanmax(y),
        color="orange", alpha=0.5,
        linestyles="dashed", label="changepoint")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(title)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

if SAVE_PLOTS:
    for col in num_cols:
        s = df[col]
        pm = detect_point_outliers_iqr(s)
        cm = detect_contextual_outliers(s)
        runs = group_consecutive_true(cm, min_len=COLLECTIVE_MIN_LEN)
        cps = cusum_change_points(s, threshold=CUSUM_THRESHOLD, drift=CUSUM_DRIFT)
        plot_series_with_marks(
            df[TIME_COL], s, col, pm, cm, runs, cps, PLOT_DIR / f"{col}.png"
        )
    print(f"\n시각화 PNG 저장 경로: {PLOT_DIR.resolve()}")
