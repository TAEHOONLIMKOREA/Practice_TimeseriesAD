# pip install autogluon.timeseries==1.4.0 pandas numpy matplotlib tqdm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# -----------------------------
# 하이퍼파라미터
# -----------------------------
CONTEXT_LEN = 256   # 예측 컨텍스트 길이
HORIZON     = 32    # 한 번에 예측할 길이
STRIDE      = 16    # 창 이동 간격(클수록 빠름)
THRESHOLD_Z = 4.0   # robust z-score 임계값
OUT_DIR     = "anomaly_outputs_bolt"
CSV_PATH    = "./data/preprocessing.csv"
TIME_COL_CANDIDATES = ["timestamp","time","date","datetime"]  # 자동 탐지 후보

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 유틸
# -----------------------------
def find_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """시간열 자동탐지(있으면 그 열 사용, 없으면 1분 간격 가짜 타임스탬프 생성)."""
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=False)
            if ts.notna().mean() > 0.9:  # 90% 이상 파싱 성공 시 채택
                return pd.DatetimeIndex(ts)
    # 시간열이 없으면 더미 인덱스 생성(분 단위; 필요시 'S','H','D'로 변경)
    return pd.date_range("2000-01-01", periods=len(df), freq="T")

def is_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=True) <= 1

def to_tsdf(series: np.ndarray, idx: pd.DatetimeIndex, item_id: str) -> TimeSeriesDataFrame:
    """(item_id, timestamp, target) 형식으로 TSDF 생성"""
    df = pd.DataFrame({"item_id": item_id, "timestamp": idx, "target": series})
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

def plot_and_save(series: np.ndarray, anoms: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(10,4))
    plt.plot(series, label="data")
    idxs = np.where(anoms)[0]
    if len(idxs):
        plt.scatter(
            idxs,
            series[anoms],
            marker="o",
            label="anomaly",
            color="red",   # ← 빨간색
            alpha=0.5      # ← 반투명
        )
    plt.title(title)
    plt.legend(); plt.grid()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

# -----------------------------
# 메인 로직
# -----------------------------
def main():
    # 데이터 로드 & 시간 인덱스 준비
    df = pd.read_csv(CSV_PATH)
    time_index = find_time_index(df)

    # 대상 칼럼(수치형 & 비상수 & 비-bool)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_like = df.select_dtypes(include=["bool"]).columns.tolist()
    target_cols = [c for c in numeric_cols if c not in bool_like and not is_constant(df[c])]

    print(f"[INFO] 대상 칼럼 수: {len(target_cols)} -> {target_cols}")

    # Zero-shot Chronos-Bolt(Base) 예측기 생성 및 경량 fit
    chronos_device = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_col = target_cols[0]
    warmup_tsdf = to_tsdf(df[warmup_col].to_numpy(), time_index, item_id=warmup_col)
    predictor = TimeSeriesPredictor(prediction_length=HORIZON)
    predictor = predictor.fit(
    warmup_tsdf,
    hyperparameters={
        "Chronos": {
            "model_path": "amazon/chronos-bolt-base",  # 또는 "autogluon/chronos-bolt-base"
            "device": chronos_device,                  # ← Chronos dict 안에!
            # "dtype": "float32",                      # (옵션) 메모리 절약
        },
        # (권장) 백업 모델도 함께 두면 Chronos 실패해도 예측 가능
        "Naive": {},
        "SeasonalNaive": {},
        "ETS": {},
    },
)

    summary = []

    for col in tqdm(target_cols, desc="Columns"):
        s_all = df[col].astype(float).to_numpy()
        # NaN 선형 보간(필요시 수정)
        if np.isnan(s_all).any():
            s_all = pd.Series(s_all).interpolate(limit_direction="both").to_numpy()

        n = len(s_all)
        if n < CONTEXT_LEN + HORIZON:
            summary.append({"column": col, "num_points": n, "num_anomalies": 0, "anomaly_ratio": 0.0, "plot_path": ""})
            continue

        residual_sum = np.zeros(n)
        residual_cnt = np.zeros(n)

        # 윈도우 시작점들
        starts = range(CONTEXT_LEN, n - HORIZON + 1, STRIDE)

        for st in tqdm(starts, desc="  - windows", leave=False):
            ctx_series = s_all[st - CONTEXT_LEN: st]
            ctx_index  = time_index[st - CONTEXT_LEN: st]
            ctx_tsdf   = to_tsdf(ctx_series, ctx_index, item_id=col)

            # 🔧 변경: predict에 model=best_model 명시 → 경고 메세지 사라짐
            preds = predictor.predict(ctx_tsdf)

            pdf   = preds.to_data_frame()  # columns: quantiles (예: '0.1','0.5','0.9') 또는 'mean'

            # 중앙값(0.5 분위수) 경로 선택
            if "0.5" in pdf.columns:
                yhat = pdf["0.5"].to_numpy()
            elif "mean" in pdf.columns:
                yhat = pdf["mean"].to_numpy()
            else:
                yhat = pdf.iloc[:, 0].to_numpy()  # fallback

            true_seg = s_all[st: st + HORIZON]
            res = true_seg - yhat

            residual_sum[st: st + HORIZON] += res
            residual_cnt[st: st + HORIZON] += 1

        mask = residual_cnt > 0
        residuals = np.zeros(n)
        residuals[mask] = residual_sum[mask] / residual_cnt[mask]

        # robust z-score로 이상치 마스크
        if np.any(mask):
            med = np.median(residuals[mask])
            mad = np.median(np.abs(residuals[mask] - med))
            denom = mad if mad != 0 else 1e-9
            rzs = 0.6745 * (residuals - med) / denom
        else:
            rzs = np.zeros(n)

        anoms = np.zeros(n, dtype=bool)
        anoms[mask] = np.abs(rzs[mask]) > THRESHOLD_Z

        # 저장
        out_png = os.path.join(OUT_DIR, f"anomalies_{col}.png")
        plot_and_save(s_all, anoms, f"Anomalies (bolt_base): {col}", out_png)

        num_anoms = int(anoms.sum())
        summary.append({
            "column": col,
            "num_points": n,
            "num_anomalies": num_anoms,
            "anomaly_ratio": (num_anoms / max(1, n)),
            "plot_path": out_png
        })

    # 요약 CSV
    summary_df = pd.DataFrame(summary).sort_values(["num_anomalies", "column"], ascending=[False, True])
    out_csv = os.path.join(OUT_DIR, "anomaly_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"[INFO] 요약 저장: {out_csv}")
    print(summary_df)

if __name__ == "__main__":
    main()
