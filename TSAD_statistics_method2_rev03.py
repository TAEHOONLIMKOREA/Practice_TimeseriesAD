# -*- coding: utf-8 -*-
"""
원본 시계열 CSV에서 4가지 방법(점/구간/집단/추세)을
'각각' 원본 데이터 위에 시각화(타임별 매칭)하여 저장하는 코드.

입력: ./data/preprocessing.csv  (열: Time, Layer, 나머지 수치 컬럼들)
출력:
  - method_summary.txt             (방법별 카운트/탐지율 요약)
  - plots/<column>__<method>.png   (각 컬럼 x 방법별 시각화 PNG)
필요 라이브러리: pandas, numpy, matplotlib
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============= 설정 =============
CSV_PATH = "./data/preprocessing2_denoised.csv"  # 필요 시 변경
TIME_COL = "Time"
SAVE_PLOTS = True
PLOT_DIR = Path("./anomaly_statistical_methods2_denoised2_rev03/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# IQR 점 이상치 기준
IQR_K = 1.5

# 구간(컨텍스트) 이상치: rolling robust z-score
ROLL_WINDOW = 301   # 데이터 간격에 맞춰 조정
Z_THRESH = 3.0

# 집단(연속) 이상치 최소 길이
COLLECTIVE_MIN_LEN = 30

# === (신규) 시간 갭을 집단 이상치로 판단하는 임계 ===
#   중앙 dt의 GAP_MULT배보다 큰 갭은 데이터 공백 구간으로 간주
GAP_MULT = 8.0

# === (신규) 추세 이상(Trend) 탐지 파라미터 ===
#   롤링 선형회귀의 기울기( slope )에 대한 강건 z-score 기준
TREND_WINDOW = 301               # 기울기 계산 윈도우 (데이터 간격 맞춰 조절)
TREND_Z = 3.0                    # 기울기 이상 탐지 z 임계
TREND_MIN_LEN = 30               # 이상 기울기 최소 연속 길이
TREND_LINE_WINDOW = 301          # 시각화용 추세선(롤링 중앙값) 윈도우


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

def detect_collective_outliers(s: pd.Series, t: pd.Series) -> (pd.Series, list, list):
    """
    집단(연속) 이상치 마스크.
    - 기본: 컨텍스트 이상치의 연속 길이가 기준 이상
    - 추가: '큰 시간 갭' 구간도 집단 이상치로 간주 (시각화용 spans 반환)
    반환:
        collective_mask, runs_idx[(st,ed), ...], gap_spans_time[(t_left, t_right), ...]
    """
    # 1) 컨텍스트 기반 연속 구간
    ctx_mask = detect_contextual_outliers(s)
    runs = group_consecutive_true(ctx_mask, min_len=COLLECTIVE_MIN_LEN)
    collective = pd.Series(False, index=s.index)
    for st, ed in runs:
        collective.iloc[st:ed+1] = True

    # 2) (신규) 시간 갭 기반 집단 이상치
    t_s = pd.to_datetime(t)
    dt = t_s.diff().dt.total_seconds()
    # 기준: 중앙값 * GAP_MULT
    dt_med = np.nanmedian(dt.values)
    gap_threshold = dt_med * GAP_MULT if np.isfinite(dt_med) and dt_med > 0 else None
    gap_spans = []
    if gap_threshold is not None:
        gap_idx = np.where(dt.values > gap_threshold)[0]  # i 위치는 (i-1)->i 사이의 갭
        for i in gap_idx:
            # 표시용 시간 구간 [t[i-1], t[i]]
            if 0 <= (i - 1) < len(t_s) and 0 <= i < len(t_s):
                gap_spans.append((t_s.iloc[i-1], t_s.iloc[i]))
                # 마스크는 구간 내부 인덱스가 없으니, 경계 몇 개 포인트만 True로 보조 표시
                collective.iloc[max(i-2, 0):min(i+1, len(collective)-1)+1] = True

    return collective, runs, gap_spans

# === (신규) 추세 이상 탐지 ===
def _rolling_linear_slope(t_sec: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    중심 정렬 윈도우로 각 지점의 로컬 선형회귀 기울기(slope)를 계산.
    t_sec, y는 동일 길이 1D 배열.
    """
    n = len(y)
    half = window // 2
    slopes = np.full(n, np.nan)
    for i in range(n):
        L = max(0, i - half)
        R = min(n, i + half + 1)
        if R - L >= max(10, window // 2):  # 최소 샘플 확보
            xw = t_sec[L:R]
            yw = y[L:R]
            # 상수항 포함 1차 회귀
            A = np.vstack([xw, np.ones_like(xw)]).T
            coeff, *_ = np.linalg.lstsq(A, yw, rcond=None)
            slopes[i] = coeff[0]
    return slopes

def _robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return pd.Series(np.full(len(x), np.nan), index=x.index)
    return (x - med) / (1.4826 * mad)

def detect_trend_anomaly(s: pd.Series, t: pd.Series,
                         win_slope: int = TREND_WINDOW, z_thresh: float = TREND_Z,
                         min_len: int = TREND_MIN_LEN):
    """
    롤링 선형회귀의 slope에 대해 강건 z-점수로 이상 추세 구간을 탐지.
    반환:
        trend_mask(pd.Series), runs_idx[(st,ed), ...], trend_line(pd.Series)
    """
    # 시간 -> 초 단위
    t_sec = pd.to_datetime(t).astype("int64") // 10**9
    t_sec = t_sec.astype(float).values
    y = s.values.astype(float)

    slopes = _rolling_linear_slope(t_sec, y, win_slope)
    slopes_s = pd.Series(slopes, index=s.index)
    z = _robust_z(slopes_s)
    trend_mask = (z.abs() > z_thresh)

    runs = group_consecutive_true(trend_mask.fillna(False), min_len=min_len)

    # 시각화용 추세선(롤링 중앙값)
    trend_line = s.rolling(TREND_LINE_WINDOW, center=True, min_periods=TREND_LINE_WINDOW//2).median()

    return trend_mask.fillna(False), runs, trend_line


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
    "trend": 0,          # (변경) changepoint → trend
}

for col in num_cols:
    s = df[col]

    # point
    pm = detect_point_outliers_iqr(s)
    method_stats["point"] += int(pm.sum())

    # contextual
    cm = detect_contextual_outliers(s)
    method_stats["contextual"] += int(cm.sum())

    # collective (컨텍스트 연속 + 시간 갭)
    coll_mask, _, _ = detect_collective_outliers(s, df[TIME_COL])
    method_stats["collective"] += int(coll_mask.sum())

    # trend (롤링 기울기 z-score)
    tr_mask, _, _ = detect_trend_anomaly(s, df[TIME_COL])
    method_stats["trend"] += int(tr_mask.sum())

# 탐지율(전체 데이터포인트 대비 비율)
method_rates = {k: v / n_points_total for k, v in method_stats.items()}

# 요약 저장
with open("method_summary.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps({
        "n_rows": n_rows,
        "n_numeric_cols": len(num_cols),
        "n_points_total": n_points_total,
        "method_counts": method_stats,
        "method_rates": method_rates,
        "notes": {
            "collective": "컨텍스트 연속 이상 + 큰 시간 갭 포함",
            "trend": "롤링 선형회귀 slope의 강건 z-score 기반"
        }
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
                color="red", alpha=0.6
            )

    elif method == "contextual":
        mask = detect_contextual_outliers(y)
        idx = np.where(mask.values)[0]
        if len(idx):
            ax.scatter(
                t.iloc[idx], y.iloc[idx],
                s=20, label="contextual outlier",
                marker="x", color="red", alpha=0.8
            )

    elif method == "collective":
        coll_mask, runs, gap_spans = detect_collective_outliers(y, t)
        # 연속 컨텍스트 구간 음영
        for i, (st, ed) in enumerate(runs):
            ax.axvspan(
                t.iloc[st], t.iloc[ed],
                alpha=0.3, color="red",
                label="collective (context runs)" if i == 0 else None
            )
        # (신규) 큰 시간 갭 구간 음영
        for j, (ts, te) in enumerate(gap_spans):
            ax.axvspan(
                ts, te, alpha=0.25, color="orange",
                label="gap (no data)" if j == 0 else None
            )

    elif method == "trend":
        tr_mask, runs, trend_line = detect_trend_anomaly(y, t)
        # 추세선
        ax.plot(t, trend_line, linewidth=1.2, alpha=0.9, color="C0", label="trend (rolling median)")
        # 기울기 이상 구간 음영
        for i, (st, ed) in enumerate(runs):
            ax.axvspan(
                t.iloc[st], t.iloc[ed],
                alpha=0.3, color="red",
                label="trend anomaly" if i == 0 else None
            )

    # 제목/축/범례
    ax.set_title(f"{title} | Outlier method: {method}", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel(title)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

ALL_METHODS = ["point", "contextual", "collective", "trend"]  # (변경) trend 포함

if SAVE_PLOTS:
    for col in num_cols:
        for m in ALL_METHODS:
            out_path = PLOT_DIR / f"{col}__{m}.png"
            plot_series_with_method(
                df[TIME_COL], df[col], col, m, out_path
            )
    print("\n각 컬럼별로 모든 방법의 시각화 PNG가 저장되었습니다.")
    print(f"저장 경로: {PLOT_DIR.resolve()}")
