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
import re

from huggingface_hub import snapshot_download
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# -----------------------------
# 하이퍼파라미터
# -----------------------------
MEGA_BATCH_ITEMS = 4096  # 메가배치 크기: 여러 컬럼×윈도우를 한 번에. 2048~8192 사이 튜닝

CONTEXT_LEN = 2048    # 예측 컨텍스트 길이
HORIZON     = 256     # 한 번에 예측할 길이
STRIDE      = 64      # 창 이동 간격(클수록 빠름)
THRESHOLD_Z = 4.0     # robust z-score 임계값

OUT_DIR     = "anomaly_outputs_bolt_fft_denoised"
CSV_PATH    = "./TimeseriesAD/data/preprocessing_fft_denoised.csv"
TIME_COL_CANDIDATES = ["timestamp", "time", "date", "datetime"]  # 자동 탐지 후보

# 모델/캐시 경로
MODEL_ID  = "amazon/chronos-bolt-base"           # 또는 "autogluon/chronos-bolt-base"
LOCAL_DIR = os.path.abspath("./models/chronos-bolt-base")
HF_CACHE  = os.path.abspath("./hf_cache")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

RESULTS_DIR = os.path.join(OUT_DIR, "results")  # 전처리 컬럼별 중간 결과 및 원본 요약 저장
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")    # 원본 단위 플롯 저장
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# 유틸
# -----------------------------
PREP_SUFFIXES = [
    "_denoised", "_ma21", "_med21", "_sg21_3",
]
# 필요 시 정규식 패턴을 더 확장 가능: 예) _sg(\d+)_(\d+)
PREP_REGEXES = [re.compile(rf"{re.escape(s)}$") for s in PREP_SUFFIXES]

def is_preprocessed_col(col: str) -> bool:
    return any(r.search(col) for r in PREP_REGEXES)

def base_name_from_preprocessed(col: str) -> str:
    for r in PREP_REGEXES:
        if r.search(col):
            return r.sub("", col)
    return col  # fallback

def origin_col_for_base(df: pd.DataFrame, base: str) -> str:
    cand = f"{base}_original"
    if cand in df.columns:
        return cand
    # 접미사 없는 순수 베이스가 원본인 경우
    if base in df.columns:
        return base
    # 마지막 보루: 원본 후보 사전 정의(필요시 확장)
    return base  # 못 찾으면 그대로 반환

def find_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """시간열 자동탐지(있으면 그 열 사용, 없으면 1분 간격 가짜 타임스탬프 생성)."""
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=False)
            if ts.notna().mean() > 0.9:
                return pd.DatetimeIndex(ts)
    return pd.date_range("2000-01-01", periods=len(df), freq="T")

def to_tsdf(series: np.ndarray, idx: pd.DatetimeIndex, item_id: str) -> TimeSeriesDataFrame:
    df = pd.DataFrame({"item_id": item_id, "timestamp": idx, "target": series})
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

def plot_and_save(series: np.ndarray, anoms: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(series, label="origin", alpha=0.6)
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

    # Predictor 준비 (warmup) — 전처리 컬럼 중 하나로 피팅
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

    # ---------- (A) 전처리 컬럼만 준비 ----------
    residual_buf = {}  # col -> dict(series, n, sum, cnt, starts)
    for col in tqdm(cols_chunk, desc=f"[GPU{gpu_id}] Pre-alloc", leave=False, mininterval=0.5):
        result_path = os.path.join(RESULTS_DIR, f"{col}.json")
        if os.path.exists(result_path):
            continue  # 이미 완료(전처리 컬럼 단위)

        s = df[col].astype(float).to_numpy()
        if np.isnan(s).any():
            s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
        n = len(s)

        if n < context_len + horizon:
            # 바로 저장 후 건너뜀(빈 이상치)
            base = base_name_from_preprocessed(col)
            row = {
                "processed_column": col,
                "base": base,
                "num_points": n,
                "anomaly_indices": [],
                "num_anomalies": 0,  # 추가
            }
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

    if not residual_buf:
        return []

    # ---------- (B) (col, st) 페어 → 메가배치 ----------
    pairs = []
    for col, buf in residual_buf.items():
        for st in buf["starts"]:
            pairs.append((col, st))

    for mb_start in tqdm(range(0, len(pairs), MEGA_BATCH_ITEMS),
                         desc=f"[GPU{gpu_id}] Mega-batches", leave=False, mininterval=0.5):
        chunk = pairs[mb_start: mb_start + MEGA_BATCH_ITEMS]

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

        preds = predictor.predict(batch_tsdf, model=best_model)
        pdf   = preds.to_data_frame()

        # item_id별 예측 ('0.5' 우선)
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

    # ---------- (C) 전처리 컬럼별 이상치 인덱스 저장 ----------
    local_summary = []
    for col, buf in tqdm(residual_buf.items(), desc=f"[GPU{gpu_id}] Finalize", leave=False, mininterval=0.5):
        n    = buf["n"]
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

        base = base_name_from_preprocessed(col)
        anomaly_idx_list = np.where(anoms)[0].astype(int).tolist()
        row = {
            "processed_column": col,
            "base": base,
            "num_points": n,
            "anomaly_indices": anomaly_idx_list,
            "num_anomalies": len(anomaly_idx_list),  # 추가
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

    # 데이터 로드 & 시간 인덱스
    df = pd.read_csv(CSV_PATH)
    time_index = find_time_index(df)

    # ---- 대상 컬럼: 전처리된 수치형만 선택 ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 시간열 후보는 numeric에서 제외(환경에 따라 숫자로 파싱될 수 있음)
    numeric_cols = [c for c in numeric_cols if c not in TIME_COL_CANDIDATES]
    preprocessed_cols = [c for c in numeric_cols if is_preprocessed_col(c)]

    if len(preprocessed_cols) == 0:
        print("[WARN] 전처리된 대상 컬럼이 없습니다. (접미사: _denoised, _ma21, _med21, _sg21_3 등)")
        return

    print(f"[INFO] 전처리 대상 컬럼 수: {len(preprocessed_cols)}")

    # ---- GPU 수 ----
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[INFO] CUDA GPU가 없어 단일 프로세스(CPU/GPU 기본)로 실행합니다.")
        num_gpus = 1

    # 이미 완료된(전처리 컬럼 단위) 결과 스킵 — 원본 요약(__originl__*)은 제외
    done = set()
    for f in os.listdir(RESULTS_DIR):
        if not f.endswith(".json"):
            continue
        name = os.path.splitext(f)[0]
        if name.startswith("__original__"):
            continue
        done.add(name)

    remaining_cols = [c for c in preprocessed_cols if c not in done]
    print(f"[INFO] 완료된(전처리) 컬럼 수: {len(done)} / 남은(전처리) 컬럼 수: {len(remaining_cols)}")

    # 컬럼을 GPU 수만큼 균등 분할
    chunks: list[list[str]] = [[] for _ in range(num_gpus)]
    for i, col in enumerate(remaining_cols):
        chunks[i % num_gpus].append(col)

    # warmup 컬럼(전처리 컬럼 중 하나)
    WARMUP_COL_NAME = remaining_cols[0] if remaining_cols else preprocessed_cols[0]
    warmup_col = WARMUP_COL_NAME

    futures = []
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
        # 결과 수집(예외 전파)
        _ = [f.result() for f in as_completed(futures)]

    # ---------- (최종 단계) 전처리 컬럼 결과 → 원본 단위로 통합 ----------
    # 전처리 결과 수집 (processed_column 포맷만 취급)
    per_proc_rows = []
    for name in os.listdir(RESULTS_DIR):
        if name.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, name), "r") as f:
                row = json.load(f)
            if "processed_column" in row and "base" in row and "anomaly_indices" in row:
                per_proc_rows.append(row)

    if not per_proc_rows:
        print("[WARN] 전처리 컬럼 결과(JSON)가 없습니다.")
        return

    # 베이스(원본)별 이상치 인덱스 합집합
    base_to_indices = {}
    base_to_count   = {}
    for r in per_proc_rows:
        base = r["base"]
        idxs = set(r.get("anomaly_indices", []))
        base_to_indices.setdefault(base, set()).update(idxs)
        base_to_count.setdefault(base, 0)
        base_to_count[base] += len(idxs)

    # 원본 단위 플래그/플롯/요약 생성
    flags_df = pd.DataFrame({"timestamp": time_index})
    summary_rows = []

    for base, idx_set in base_to_indices.items():
        origin_col = origin_col_for_base(df, base)
        if origin_col not in df.columns:
            print(f"[WARN] 원본 컬럼을 찾지 못해 스킵: base={base}, origin={origin_col}")
            continue

        n = len(df[origin_col])
        anom = np.zeros(n, dtype=bool)
        good_idxs = [i for i in idx_set if 0 <= i < n]
        anom[good_idxs] = True

        # 플롯(원본 시계열 + 이상치)
        out_png = os.path.join(PLOTS_DIR, f"anomalies_original_{origin_col}.png")
        origin_series = pd.Series(df[origin_col].astype(float)).interpolate(limit_direction="both").to_numpy()
        plot_and_save(origin_series, anom, f"Anomalies on ORIGIN: {origin_col}", out_png)

        # 요약 JSON (원본 단위)
        row = {
            "origin_column": origin_col,
            "base": base,
            "num_points": int(n),
            "num_anomalies": int(anom.sum()),
            "anomaly_ratio": float(anom.mean()),
            "plot_path": out_png,
            "anomaly_indices": good_idxs,
        }
        # 원본 단위 파일명 고유화
        origin_json = os.path.join(RESULTS_DIR, f"__original__{origin_col}.json")
        with open(origin_json + ".tmp", "w") as f: json.dump(row, f)
        os.replace(origin_json + ".tmp", origin_json)

        summary_rows.append(row)

        # 표기: 원본 컬럼명에 _anomaly 플래그 추가 (0/1)
        flags_df[f"{origin_col}_anomaly"] = anom.astype(int)

    # 요약 CSV (가드 포함)
    if not summary_rows:
        print("[INFO] 원본 단위에서 집계된 이상치가 없습니다. CSV/플롯 생성을 건너뜁니다.")
    else:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["num_anomalies", "origin_column"], ascending=[False, True]
        )
        out_csv_summary = os.path.join(OUT_DIR, "anomaly_summary_original.csv")
        summary_df.to_csv(out_csv_summary, index=False)
        print(f"[INFO] 원본 단위 요약 저장: {out_csv_summary}")
        print(summary_df)

    # 플래그 CSV (timestamp + 각 원본_anomaly)
    out_flags_csv = os.path.join(OUT_DIR, "anomaly_flags.csv")
    flags_df.to_csv(out_flags_csv, index=False)
    print(f"[INFO] 원본 이상치 플래그 저장: {out_flags_csv}")

    # (선택) 원본 “표기”된 CSV: 원본 값은 그대로 두고 *_anomaly 컬럼만 추가
    marked = df.copy()
    for col in flags_df.columns:
        if col == "timestamp":  # timestamp가 실제 열과 매칭되면 교체, 아니면 추가
            if "timestamp" in marked.columns:
                # 덮어쓰지 않음
                pass
            else:
                marked["timestamp"] = flags_df["timestamp"]
        elif col.endswith("_anomaly"):
            marked[col] = flags_df[col].values
    marked_csv = os.path.join(OUT_DIR, "marked_original.csv")
    marked.to_csv(marked_csv, index=False)
    print(f"[INFO] 원본 표기 CSV 저장: {marked_csv}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    print("Starting training. Start time is", pd.Timestamp.now())
    main()
