import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from tqdm import tqdm

# ---------- 공통 하이퍼파라미터 ----------
CONTEXT_LEN = 256   # 과거 컨텍스트 길이
HORIZON     = 32   # 한 번에 예측할 길이
STRIDE      = 32    # 창 이동 간격
BATCH_SIZE  = 64    # 배치 크기
THRESHOLD_Z = 4.0   # robust z 임계값
OUT_DIR     = "anomaly_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 모델 로딩 ----------
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="auto",
    dtype="auto",
)
print(pipeline)

# ---------- 데이터 로딩 ----------
csv_path = "./3PDX_TimeseriesAD/data/MachineDataLog/preprocessing.csv"
df = pd.read_csv(csv_path)
print(df.head())

# chronosAD.py 상단에 임시로 추가
print("pandas/numpy:", __import__("pandas").__version__, __import__("numpy").__version__)
print("csv_path type:", type(csv_path))

def is_constant(series: pd.Series) -> bool:
    """모든 값이 동일하면 True"""
    return series.nunique(dropna=True) <= 1

def detect_anomalies_one_column(values: np.ndarray,
                                context_len=CONTEXT_LEN,
                                horizon=HORIZON,
                                stride=STRIDE,
                                batch_size=BATCH_SIZE,
                                thr=THRESHOLD_Z):
    """단일 1D 시계열에서 Chronos 배치 예측 기반 이상치 마스크/점수 계산"""
    s = values.astype(float)
    n = len(s)
    if n < context_len + horizon:
        return np.zeros(n, dtype=bool), np.zeros(n)  # 데이터가 너무 짧을 때

    starts = list(range(context_len, n - horizon + 1, stride))
    residual_sum = np.zeros(n)
    residual_cnt = np.zeros(n)

    # 배치 진행률 tqdm
    for b in tqdm(range(0, len(starts), batch_size),
                  desc="  - batches", leave=False):
        chunk = starts[b:b+batch_size]
        contexts = [torch.tensor(s[i - context_len:i], dtype=torch.float32) for i in chunk]

        with torch.no_grad():
            fc = pipeline.predict(contexts, prediction_length=horizon)  # [B, S, H]

        median = np.median(fc.numpy(), axis=1)  # [B, H]

        for idx, st in enumerate(chunk):
            pred_seg = median[idx]                  # 길이 H
            true_seg = s[st:st + horizon]
            res = true_seg - pred_seg
            residual_sum[st:st + horizon] += res
            residual_cnt[st:st + horizon] += 1

    mask = residual_cnt > 0
    residuals = np.zeros(n)
    residuals[mask] = residual_sum[mask] / residual_cnt[mask]

    # robust z-score
    if np.any(mask):
        res_med = np.median(residuals[mask])
        res_mad = np.median(np.abs(residuals[mask] - res_med))
        denom = res_mad if res_mad != 0 else 1e-9
        robust_z = 0.6745 * (residuals - res_med) / denom
    else:
        robust_z = np.zeros(n)

    anoms = np.zeros(n, dtype=bool)
    anoms[mask] = np.abs(robust_z[mask]) > thr
    return anoms, robust_z

def plot_and_save(series: np.ndarray, anoms: np.ndarray, title: str, out_path: str):
    """시각화/저장"""
    plt.figure(figsize=(10,4))
    plt.plot(series, label="data")
    idxs = np.where(anoms)[0]
    if len(idxs):
        plt.scatter(
            idxs,
            series[anoms],
            marker="o",
            label="anomaly",
            color="red",      # 빨간색
            alpha=0.5         # 투명도 (0=완전투명, 1=불투명)
        )
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # ---------- 여러 칼럼에 대해 실행 ----------
    # 1) 대상 칼럼 자동 선택: 수치형 & (비상수) & (비-bool)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_like = df.select_dtypes(include=["bool"]).columns.tolist()
    target_cols = [c for c in numeric_cols if c not in bool_like and not is_constant(df[c])]

    summary_rows = []

    print(f"[INFO] 대상 칼럼 수: {len(target_cols)} -> {target_cols}")

    for col in tqdm(target_cols, desc="Columns"):
        series = df[col].to_numpy()

        # NaN가 있으면 단순 선형 보간(필요 없으면 제거해도 됨)
        if np.isnan(series).any():
            s_pd = pd.Series(series).interpolate(limit_direction="both")
            series = s_pd.to_numpy()

        anoms, z = detect_anomalies_one_column(series)

        # 결과 저장
        out_png = os.path.join(OUT_DIR, f"anomalies_{col}.png")
        plot_and_save(series, anoms, f"Anomalies: {col}", out_png)

        num_anoms = int(anoms.sum())
        summary_rows.append({
            "column": col,
            "num_points": len(series),
            "num_anomalies": num_anoms,
            "anomaly_ratio": (num_anoms / max(1, len(series))),
            "plot_path": out_png
        })

    # 요약 CSV 저장
    summary_df = pd.DataFrame(summary_rows).sort_values(["num_anomalies", "column"], ascending=[False, True])
    summary_csv = os.path.join(OUT_DIR, "anomaly_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[INFO] 요약 저장: {summary_csv}")
    print(summary_df)

if __name__ == "__main__":
    main()

