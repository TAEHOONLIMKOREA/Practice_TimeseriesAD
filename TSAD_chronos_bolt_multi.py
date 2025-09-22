# pip install autogluon.timeseries==1.4.0 pandas numpy matplotlib tqdm
# pip install -U huggingface_hub hf_transfer

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from huggingface_hub import snapshot_download
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# -----------------------------
# 하이퍼파라미터
# -----------------------------
MEGA_BATCH_ITEMS = 4096  # 메가배치 크기: 여러 컬럼×윈도우를 한 번에. 2048~8192 사이 튜닝
# BATCH_WINDOWS    = 256   # (더 이상 필수 아님) per-column 배치 크기. 남겨두지만 미사용 가능

CONTEXT_LEN = 2048    # 예측 컨텍스트 길이
HORIZON     = 256     # 한 번에 예측할 길이
STRIDE      = 64     # 창 이동 간격(클수록 빠름)
THRESHOLD_Z = 4.0    # robust z-score 임계값

OUT_DIR     = "anomaly_outputs_bolt"
CSV_PATH    = "./data/preprocessing.csv"
TIME_COL_CANDIDATES = ["timestamp","time","date","datetime"]  # 자동 탐지 후보

# 모델/캐시 경로
MODEL_ID  = "amazon/chronos-bolt-base"           # 또는 "autogluon/chronos-bolt-base"
LOCAL_DIR = os.path.abspath("./models/chronos-bolt-base")
HF_CACHE  = os.path.abspath("./hf_cache")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

RESULTS_DIR = os.path.join(OUT_DIR, "results")
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# 유틸
# -----------------------------
def find_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """시간열 자동탐지(있으면 그 열 사용, 없으면 1분 간격 가짜 타임스탬프 생성)."""
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=False)
            if ts.notna().mean() > 0.9:
                return pd.DatetimeIndex(ts)
    return pd.date_range("2000-01-01", periods=len(df), freq="T")

def is_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=True) <= 1

def to_tsdf(series: np.ndarray, idx: pd.DatetimeIndex, item_id: str) -> TimeSeriesDataFrame:
    df = pd.DataFrame({"item_id": item_id, "timestamp": idx, "target": series})
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

def plot_and_save(series: np.ndarray, anoms: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(10,4))
    plt.plot(series, label="data", alpha=0.5)
    idxs = np.where(anoms)[0]
    if len(idxs):
        plt.scatter(idxs, series[anoms], marker="o", label="anomaly", color="red")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

# -----------------------------
# 모델 사전 다운로드 (429 방지)
# -----------------------------
def prefetch_model():
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", HF_CACHE)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )

# -----------------------------
# 최적 모델명 선택 유틸
# -----------------------------
def pick_best_model_name(predictor: TimeSeriesPredictor):
    # 1) leaderboard 기반
    try:
        lb = predictor.leaderboard(silent=True)
        if "score_val" in lb.columns and "model" in lb.columns:
            return lb.sort_values("score_val", ascending=True)["model"].iloc[0]
    except Exception:
        pass
    # 2) WeightedEnsemble 우선
    names = predictor.get_model_names()
    for n in names:
        if n.lower().startswith("weighteden"):
            return n
    # 3) Chronos 우선
    for n in names:
        if n.lower().startswith("chronos"):
            return n
    # 4) 첫 모델
    return names[0] if names else None

# -----------------------------
# 워커(프로세스) 함수
# -----------------------------
def worker_run(gpu_id: int,
               cols_chunk: list[str],
               df: pd.DataFrame,
               time_index: pd.DatetimeIndex,
               context_len: int,
               horizon: int,
               stride: int,
               threshold_z: float,
               warmup_col: str) -> list[dict]:

    # GPU 고정 & 캐시/스레드 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"

    # Predictor 준비 (warmup)
    predictor = TimeSeriesPredictor(prediction_length=horizon, verbosity=0)

    warmup_series = df[warmup_col].astype(float).to_numpy()
    warmup_series = pd.Series(warmup_series).interpolate(limit_direction="both").to_numpy()
    warmup_tsdf = to_tsdf(warmup_series, time_index, item_id=warmup_col)

    predictor = predictor.fit(
        warmup_tsdf,
        hyperparameters={
            "Chronos": {
                "model_path": LOCAL_DIR,
                "device": device_str,
                "dtype": "bfloat16",   # 5090 권장. 문제 시 주석 처리
            },
        },
    )

    best_model = pick_best_model_name(predictor)

    # ---------- (A) 완료된 컬럼 스킵 & 버퍼 준비 ----------
    residual_buf = {}  # col -> dict(series, n, sum, cnt, starts)
    for col in tqdm(cols_chunk, desc=f"[GPU{gpu_id}] Pre-alloc", leave=False, mininterval=0.5):
        result_path = os.path.join(RESULTS_DIR, f"{col}.json")
        if os.path.exists(result_path):
            continue  # 이미 완료

        s = df[col].astype(float).to_numpy()
        if np.isnan(s).any():
            s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
        n = len(s)

        if n < context_len + horizon:
            # 바로 체크포인트 저장 후 건너뜀
            row = {"column": col, "num_points": n, "num_anomalies": 0, "anomaly_ratio": 0.0, "plot_path": ""}
            tmp_path = result_path + ".tmp"
            with open(tmp_path, "w") as f: json.dump(row, f)
            os.replace(tmp_path, result_path)
            continue

        starts = list(range(context_len, n - horizon + 1, stride))
        residual_buf[col] = {
            "series": s,
            "n": n,
            "sum": np.zeros(n, dtype=np.float32),
            "cnt": np.zeros(n, dtype=np.int32),
            "starts": starts,
        }

    # (완료되거나 짧은 컬럼만 있었을 때)
    if not residual_buf:
        return []

    # ---------- (B) (col, st) 페어들을 평탄화 → 메가배치로 예측 ----------
    pairs = []
    for col, buf in residual_buf.items():
        for st in buf["starts"]:
            pairs.append((col, st))

    for mb_start in tqdm(range(0, len(pairs), MEGA_BATCH_ITEMS),
                         desc=f"[GPU{gpu_id}] Mega-batches", leave=False, mininterval=0.5):
        chunk = pairs[mb_start: mb_start + MEGA_BATCH_ITEMS]

        # 메가배치 TSDF 구성
        frames = []
        for col, st in chunk:
            s = residual_buf[col]["series"]
            ctx_series = s[st - context_len: st]
            ctx_index  = time_index[st - context_len: st]
            frames.append(pd.DataFrame({
                "item_id": f"{col}__{st}",
                "timestamp": ctx_index,
                "target": ctx_series
            }))
        batch_df  = pd.concat(frames, ignore_index=True)
        batch_tsdf = TimeSeriesDataFrame.from_data_frame(batch_df, id_column="item_id", timestamp_column="timestamp")

        # 예측
        preds = predictor.predict(batch_tsdf, model=best_model)
        pdf   = preds.to_data_frame()

        # item_id별 예측 추출 ('0.5' 우선, 없으면 'mean' → 첫 컬럼)
        yhat_map = {}
        for it, g in pdf.groupby(level=0):
            if "0.5" in g.columns:
                yhat_map[it] = g["0.5"].to_numpy()
            elif "mean" in g.columns:
                yhat_map[it] = g["mean"].to_numpy()
            else:
                yhat_map[it] = g.iloc[:, 0].to_numpy()

        # 잔차 누적
        for col, st in chunk:
            yhat = yhat_map[f"{col}__{st}"]
            s    = residual_buf[col]["series"]
            true_seg = s[st: st + horizon]
            res     = true_seg - yhat
            residual_buf[col]["sum"][st: st + horizon] += res
            residual_buf[col]["cnt"][st: st + horizon] += 1

    # ---------- (C) 컬럼별 마무리/저장 ----------
    local_summary = []
    for col, buf in tqdm(residual_buf.items(), desc=f"[GPU{gpu_id}] Finalize", leave=False, mininterval=0.5):
        n    = buf["n"]; s = buf["series"]
        rs   = buf["sum"]; rc = buf["cnt"]
        mask = rc > 0

        residuals = np.zeros(n, dtype=np.float32)
        residuals[mask] = rs[mask] / rc[mask]

        if np.any(mask):
            med = np.median(residuals[mask])
            mad = np.median(np.abs(residuals[mask] - med))
            denom = mad if mad != 0 else 1e-9
            rzs = 0.6745 * (residuals - med) / denom
        else:
            rzs = np.zeros(n, dtype=np.float32)

        anoms = np.zeros(n, dtype=bool)
        anoms[mask] = np.abs(rzs[mask]) > threshold_z

        out_png = os.path.join(PLOTS_DIR, f"anomalies_{col}.png")
        plot_and_save(s, anoms, f"Anomalies (bolt_base): {col}", out_png)

        row = {
            "column": col,
            "num_points": n,
            "num_anomalies": int(anoms.sum()),
            "anomaly_ratio": (float(anoms.sum()) / max(1, n)),
            "plot_path": out_png
        }
        result_path = os.path.join(RESULTS_DIR, f"{col}.json")
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f: json.dump(row, f)
        os.replace(tmp_path, result_path)

        local_summary.append(row)

    return local_summary

# -----------------------------
# 메인 로직
# -----------------------------
def main():
    # 모델 사전 다운로드 (429 방지)
    prefetch_model()

    # 데이터 로드 & 시간 인덱스 준비
    df = pd.read_csv(CSV_PATH)
    time_index = find_time_index(df)

    # 대상 칼럼(수치형 & 비상수 & 비-bool)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_like = df.select_dtypes(include=["bool"]).columns.tolist()
    target_cols = [c for c in numeric_cols if c not in bool_like and not is_constant(df[c])]

    print(f"[INFO] 대상 칼럼 수: {len(target_cols)} -> {target_cols}")

    if len(target_cols) == 0:
        print("[WARN] 대상 칼럼이 없습니다.")
        return

    # 사용 가능한 GPU 수
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[INFO] CUDA GPU가 없어 단일 프로세스(CPU/GPU 기본)로 실행합니다.")
        num_gpus = 1

    # 이미 완료된 컬럼 스킵
    done = set(os.path.splitext(f)[0] for f in os.listdir(RESULTS_DIR) if f.endswith(".json"))
    remaining_cols = [c for c in target_cols if c not in done]
    print(f"[INFO] 완료된 컬럼 수: {len(done)} / 남은 컬럼 수: {len(remaining_cols)}")

    # 컬럼을 GPU 수만큼 균등 분할 (남은 컬럼 기준)
    chunks: list[list[str]] = [[] for _ in range(num_gpus)]
    for i, col in enumerate(remaining_cols):
        chunks[i % num_gpus].append(col)

    # warmup 컬럼 지정
    WARMUP_COL_NAME = "OxygenLowChamber" if "OxygenLowChamber" in df.columns else (remaining_cols[0] if remaining_cols else target_cols[0])
    warmup_col = WARMUP_COL_NAME

    futures = []
    summary_all = []

    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context("spawn")) as ex:
        for gpu_id, cols_chunk in enumerate(chunks):
            if not cols_chunk:
                continue
            futures.append(
                ex.submit(
                    worker_run,
                    gpu_id,
                    cols_chunk,
                    df,
                    time_index,
                    CONTEXT_LEN,
                    HORIZON,
                    STRIDE,
                    THRESHOLD_Z,
                    warmup_col,
                )
            )

        for fut in as_completed(futures):
            summary_all.extend(fut.result())

    # 최종 요약: 체크포인트 JSON들로부터 재조립
    rows = []
    for name in os.listdir(RESULTS_DIR):
        if name.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, name), "r") as f:
                rows.append(json.load(f))

    summary_df = pd.DataFrame(rows).sort_values(["num_anomalies", "column"], ascending=[False, True])
    out_csv = os.path.join(OUT_DIR, "anomaly_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"[INFO] 요약 저장: {out_csv}")
    print(summary_df)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    print("Starting training. Start time is", pd.Timestamp.now())
    main()
