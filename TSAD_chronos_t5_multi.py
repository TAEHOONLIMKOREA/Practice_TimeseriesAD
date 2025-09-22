import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 공통 하이퍼파라미터 ----------
CONTEXT_LEN = 512   # 과거 컨텍스트 길이
HORIZON     = 64    # 한 번에 예측할 길이
STRIDE      = 32    # 창 이동 간격
BATCH_SIZE  = 32    # 배치 크기
THRESHOLD_Z = 4.0   # robust z 임계값
OUT_DIR     = "anomaly_outputs"

# (선택) Chronos 샘플 수: 메모리/시간에 큰 영향
CHRONOS_NUM_SAMPLES = 30  # 20~50 권장. 더 빠르게 하려면 줄이기



def build_pipelines_multi_gpu(model_name="amazon/chronos-t5-base", dtype="auto"):
    """
    GPU가 여러 장이면 GPU마다 ChronosPipeline을 하나씩 로드하여 리스트로 반환.
    GPU가 1장 이하이면 빈 리스트([])를 반환해 싱글 경로로 빠지게 한다.
    """
    if not torch.cuda.is_available():
        return []
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return []

    pipes = []
    for i in range(num_gpus):
        pipe = ChronosPipeline.from_pretrained(
            model_name,
            device_map={"": f"cuda:{i}"},  # 각 인스턴스를 특정 GPU에 고정
            dtype=dtype,
        )
        pipes.append(pipe)
    return pipes


def _predict_worker(pipe, contexts, prediction_length, **predict_kwargs):
    """
    단일 GPU에서 여러 context에 대해 predict 수행.
    contexts: list[torch.Tensor], 텐서는 CPU여도 됨(파이프라인 쪽에서 이동)
    반환: 순서대로 예측 결과 리스트
    """
    with torch.inference_mode():
        fc = pipe.predict(contexts, prediction_length=prediction_length, **predict_kwargs)
        # fc: [B, S, H] (torch.Tensor 또는 numpy) 가정
    return fc

def predict_batch_multi_gpu(pipes, contexts, prediction_length, **predict_kwargs):
    """
    여러 GPU에 contexts를 분할해서 병렬 predict.
    pipes: GPU별 ChronosPipeline 리스트
    contexts: list[torch.Tensor] (길이 CONTEXT_LEN의 1D)
    반환: torch.Tensor [K, S, H] 또는 numpy.ndarray [K, S, H]
    """
    if len(contexts) == 0:
        return None
    num_gpus = len(pipes)
    if num_gpus == 0:
        # 멀티GPU가 아니면 여기 오지 않음
        raise RuntimeError("predict_batch_multi_gpu called with zero pipes.")

    # 인덱스 유지용
    indexed = list(enumerate(contexts))
    # 라운드로빈 분할
    chunks = [indexed[i::num_gpus] for i in range(num_gpus)]

    results = [None] * len(contexts)

    def _submit(pipe, chunk):
        if not chunk:
            return None
        # 순서/인덱스 보존
        idxs = [idx for idx, _ in chunk]
        local_contexts = [t for _, t in chunk]
        out = _predict_worker(pipe, local_contexts, prediction_length, **predict_kwargs)
        # out: [b, S, H]
        return idxs, out

    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        futs = []
        for pipe, chunk in zip(pipes, chunks):
            if chunk:
                futs.append(ex.submit(_submit, pipe, chunk))

        for fut in as_completed(futs):
            packet = fut.result()
            if packet is None:
                continue
            idxs, out = packet
            # out을 CPU numpy로 통일
            if isinstance(out, torch.Tensor):
                arr = out.detach().cpu().numpy()
            else:
                arr = np.asarray(out)
            for j, idx in enumerate(idxs):
                results[idx] = arr[j]

    # 스택 (numpy로 통일)
    return np.stack(results, axis=0)



def is_constant(series: pd.Series) -> bool:
    """모든 값이 동일하면 True"""
    return series.nunique(dropna=True) <= 1

def detect_anomalies_one_column(values: np.ndarray,
                                context_len=CONTEXT_LEN,
                                horizon=HORIZON,
                                stride=STRIDE,
                                batch_size=BATCH_SIZE,
                                thr=THRESHOLD_Z,
                                num_samples=CHRONOS_NUM_SAMPLES,
                                use_multi_gpu=None,
                                pipes=[],
                                pipeline=None):
    """단일 1D 시계열에서 Chronos 배치 예측 기반 이상치 마스크/점수 계산 (멀티 GPU 지원)"""
    s = values.astype(float)
    n = len(s)
    if n < context_len + horizon:
        return np.zeros(n, dtype=bool), np.zeros(n)  # 데이터가 너무 짧을 때

    starts = list(range(context_len, n - horizon + 1, stride))
    residual_sum = np.zeros(n)
    residual_cnt = np.zeros(n)

    # 배치 진행률 tqdm
    for b in tqdm(range(0, len(starts), batch_size), desc="  - batches", leave=False):
        chunk = starts[b:b+batch_size]
        contexts = [torch.tensor(s[i - context_len:i], dtype=torch.float32) for i in chunk]

        if use_multi_gpu:
            # 여러 GPU에 분산 추론
            fc_np = predict_batch_multi_gpu(
                pipes, contexts, prediction_length=horizon, num_samples=num_samples
            )  # [B, S, H] numpy
        else:
            # 싱글 경로
            with torch.no_grad():
                fc = pipeline.predict(contexts, prediction_length=horizon, num_samples=num_samples)  # [B, S, H]
            # 텐서/넘파이 상관없이 numpy로
            if isinstance(fc, torch.Tensor):
                fc_np = fc.detach().cpu().numpy()
            else:
                fc_np = np.asarray(fc)

        # 샘플 차원 S 기준 중앙값
        median = np.median(fc_np, axis=1)  # [B, H]

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
            color="red",
            alpha=0.5
        )
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():    
    os.makedirs(OUT_DIR, exist_ok=True)
    # ---------- 모델 로딩 ----------
    # 싱글 파이프라인(기본)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="auto",
        dtype="auto",
    )
    print(pipeline)

    # 멀티 GPU 파이프라인(여러 장일 때만)
    PIPES = build_pipelines_multi_gpu(model_name="amazon/chronos-t5-base", dtype="auto")
    USE_MULTI_GPU = len(PIPES) > 0
    if USE_MULTI_GPU:
        print(f"[INFO] Multi-GPU enabled: {len(PIPES)} GPUs")
    else:
        print("[INFO] Single GPU/CPU mode")

    # ---------- 데이터 로딩 ----------
    csv_path = "./3PDX_TimeseriesAD/data/MachineDataLog/preprocessing.csv"
    df = pd.read_csv(csv_path)
    print(df.head())

    # chronosAD.py 상단에 임시로 추가
    print("pandas/numpy:", __import__("pandas").__version__, __import__("numpy").__version__)
    print("csv_path type:", type(csv_path))

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

        anoms, z = detect_anomalies_one_column(series, pipeline=pipeline,
                                               context_len=CONTEXT_LEN,
                                               horizon=HORIZON,
                                               stride=STRIDE,
                                               batch_size=BATCH_SIZE,
                                               thr=THRESHOLD_Z,
                                               num_samples=CHRONOS_NUM_SAMPLES,
                                               use_multi_gpu=USE_MULTI_GPU,
                                               pipes=PIPES if USE_MULTI_GPU else [],)

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
    # (선택) CPU 스레드 과점 방지
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
