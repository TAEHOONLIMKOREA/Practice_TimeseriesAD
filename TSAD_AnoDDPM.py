#!/usr/bin/env python3
"""
Zero-shot anomaly detection for time series columns using a tiny diffusion-style score model.

- Trains a small 1D MLP score network per column on sliding windows (unsupervised).
- Uses denoising-based reconstruction to derive anomaly scores for each timestamp.
- Works even when anomalies are present, assuming they are a minority ("zero-shot").

Usage:
    python zeroshot_diffusion_anomaly.py --csv /path/to/data.csv --out anomalies.csv

Optional args:
    --window 64               # sliding window length
    --epochs 30               # training epochs per column (increase for better quality)
    --batch 256               # batch size
    --steps 1000              # diffusion steps (beta schedule length)
    --hidden 128              # hidden width for the MLP
    --lr 1e-3                 # learning rate
    --seed 42
    --timeavg 8               # number of random timesteps to average for scoring
    --quantile 0.99           # anomaly decision threshold quantile (per column)
    --cols COL1 COL2 ...      # optional subset of columns to process (defaults to all numeric)
    --standardize zscore|robust  # scaler
    --save_intermediate       # if set, saves per-column score vectors as CSVs

Output CSV columns for each input column `<col>`:
    `<col>_score`  : normalized anomaly score in [0, 1]
    `<col>_is_anom`: 0/1 anomaly flag using `quantile`
"""
import argparse
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Utilities
# -------------------------

def set_seed(seed:int=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def sinusoidal_time_embedding(t, dim: int):
    """t in [0,1] shape [B] -> [B, dim]"""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=device)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=-1)
    return emb

# -------------------------
# Data
# -------------------------

class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int):
        self.window = window
        self.x = []
        # build overlapping windows
        for i in range(len(series) - window + 1):
            self.x.append(series[i:i+window])
        self.x = np.stack(self.x, axis=0).astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]

# -------------------------
# Diffusion scaffolding
# -------------------------

class TinyScoreNet(nn.Module):
    def __init__(self, window: int, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(window + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, window),
        )

    def forward(self, x, t):  # x: [B, W], t: [B]
        temb = sinusoidal_time_embedding(t, self.time_dim)
        h = torch.cat([x, temb], dim=-1)
        return self.net(h)

class BetaSchedule:
    """Cosine schedule, returns arrays of betas, alphas, and alpha_bars for T steps."""
    def __init__(self, T: int = 1000, s: float = 0.008, device='cpu'):
        self.T = T
        self.device = device
        steps = T
        t = torch.linspace(0, T, steps + 1, device=device)
        f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        self.alpha_bar = alpha_bar[1:]  # length T
        self.alphas = self.alpha_bar.clone()
        self.alphas[1:] = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.betas = 1 - self.alphas
        # clamp for numerical stability
        self.betas = torch.clamp(self.betas, 1e-6, 0.999)
        self.alphas = 1.0 - self.betas

    def to(self, device):
        self.device = device
        self.alpha_bar = self.alpha_bar.to(device)
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        return self

# -------------------------
# Training & Scoring
# -------------------------

def train_score_model(series: np.ndarray, window:int=64, epochs:int=30, batch:int=256,
                      T:int=1000, hidden:int=128, lr:float=1e-3, device:str='cpu') -> Tuple[TinyScoreNet, BetaSchedule]:
    ds = WindowDataset(series, window)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    betas = BetaSchedule(T=T, device=device).to(device)
    model = TinyScoreNet(window=window, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for x0 in dl:
            x0 = x0.to(device)
            B = x0.shape[0]
            # choose random t in {1..T}
            t_idx = torch.randint(0, T, (B,), device=device)
            at_bar = betas.alpha_bar[t_idx]  # [B]
            noise = torch.randn_like(x0)
            # forward diffusion
            xt = torch.sqrt(at_bar)[:, None] * x0 + torch.sqrt(1 - at_bar)[:, None] * noise
            # time normalized to [0,1]
            t = (t_idx.float() + 1.0) / T
            # predict noise
            noise_pred = model(xt, t)
            loss = ((noise_pred - noise) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * B
            n += B
        if (ep+1) % max(1, epochs//5) == 0:
            print(f"[epoch {ep+1}/{epochs}] loss={total/n:.6f}")
    return model, betas

@torch.no_grad()
def score_series(model: TinyScoreNet, betas: BetaSchedule, series: np.ndarray, window:int=64,
                 timeavg:int=8, device:str='cpu') -> np.ndarray:
    """
    Returns a pointwise anomaly score for each timestamp (len(series)).
    We reconstruct windows via one-step denoising at multiple random ts and
    measure per-point reconstruction error aggregated over overlapping windows.
    """
    model.eval()
    W = window
    N = len(series)
    # build windows
    windows = []
    for i in range(N - W + 1):
        windows.append(series[i:i+W])
    X = torch.as_tensor(np.stack(windows, axis=0), dtype=torch.float32, device=device)  # [M, W]

    # multi-timestep averaging for robustness
    recons_accum = torch.zeros_like(X)
    for _ in range(timeavg):
        M = X.shape[0]
        t_idx = torch.randint(0, betas.T, (M,), device=device)
        at_bar = betas.alpha_bar[t_idx]  # [M]
        noise = torch.randn_like(X)
        xt = torch.sqrt(at_bar)[:, None] * X + torch.sqrt(1 - at_bar)[:, None] * noise
        t = (t_idx.float() + 1.0) / betas.T
        eps = model(xt, t)
        # x0 hat from DDPM posterior (one-step)
        x0_hat = (xt - torch.sqrt(1 - at_bar)[:, None] * eps) / torch.sqrt(at_bar)[:, None]
        recons_accum += x0_hat

    X_rec = (recons_accum / timeavg).cpu().numpy()  # [M, W]
    # Map window reconstruction error to pointwise score (use center emphasis)
    errs = (X_rec - X.cpu().numpy()) ** 2  # [M, W]
    # triangular weights to emphasize center of each window
    w = np.bartlett(W)
    w = w / (w.sum() + 1e-8)
    win_scores = (errs * w[None, :]).sum(axis=1)  # [M]

    # aggregate overlapping windows into point scores by averaging contributions
    point_scores = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)
    for i in range(N - W + 1):
        center = i + W // 2
        # assign window score primarily to the center point, but also nearby using a small kernel
        k = np.array([0.25, 0.5, 0.25])  # simple smoothing
        idxs = [max(0, center-1), center, min(N-1, center+1)]
        for j, idx in enumerate(idxs):
            point_scores[idx] += win_scores[i] * k[j]
            counts[idx] += k[j]
    counts[counts == 0] = 1.0
    point_scores = point_scores / counts
    return point_scores.astype(np.float32)

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    # robust scaling to [0,1] via percentile mapping
    lo = np.percentile(scores, 5.0)
    hi = np.percentile(scores, 99.5)
    s = (scores - lo) / (hi - lo + 1e-8)
    return np.clip(s, 0.0, 1.0)

def choose_threshold(scores: np.ndarray, q: float = 0.99) -> float:
    return float(np.quantile(scores, q))

def standardize_series(x: np.ndarray, mode: str = "zscore") -> Tuple[np.ndarray, Tuple[float, float]]:
    if mode == "robust":
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8
        return (x - med) / (1.4826 * mad), (med, mad, "robust")
    # default zscore
    mu = float(np.mean(x))
    sigma = float(np.std(x) + 1e-8)
    return (x - mu) / sigma, (mu, sigma, "zscore")

def destandardize_series(xn: np.ndarray, stats):
    if stats[-1] == "robust":
        med, mad, _ = stats
        return xn * (1.4826 * mad) + med
    else:
        mu, sigma, _ = stats
        return xn * sigma + mu

# -------------------------
# Main pipeline
# -------------------------

def process(csv_path: str, out_path: str,
            window:int=64, epochs:int=30, batch:int=256, steps:int=1000, hidden:int=128, lr:float=1e-3,
            seed:int=42, timeavg:int=8, quantile:float=0.99, cols:List[str]=None, scaler:str="zscore",
            save_intermediate:bool=False):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    # keep numeric columns only
    num_df = df.select_dtypes(include=[np.number]).copy()
    if cols:
        missing = [c for c in cols if c not in num_df.columns]
        if missing:
            raise ValueError(f"Requested columns not found or non-numeric: {missing}")
        num_df = num_df[cols]

    results = pd.DataFrame(index=df.index)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for col in num_df.columns:
        print(f"\n=== Column: {col} ===")
        x = num_df[col].to_numpy().astype(np.float32)
        # standardize for stable training
        xs, stats = standardize_series(x, mode=scaler)

        # train model
        model, betas = train_score_model(xs, window=window, epochs=epochs, batch=batch,
                                         T=steps, hidden=hidden, lr=lr, device=device)

        # scoring
        raw_scores = score_series(model, betas, xs, window=window, timeavg=timeavg, device=device)
        norm_scores = normalize_scores(raw_scores)
        thr = choose_threshold(norm_scores, q=quantile)
        is_anom = (norm_scores >= thr).astype(np.int32)

        results[f"{col}_score"] = norm_scores
        results[f"{col}_is_anom"] = is_anom

        if save_intermediate:
            # per-column detailed CSV
            pd.DataFrame({
                "raw_score": raw_scores,
                "norm_score": norm_scores,
                "is_anom": is_anom,
                col: x,
            }).to_csv(f"{os.path.splitext(out_path)[0]}__{col}_details.csv", index=False)

    # align to original length (score computation shortens edges if window > 1)
    # In this implementation we scored every timestamp via aggregation, so lengths already match.
    results.to_csv(out_path, index=False)
    print(f"\nSaved anomaly scores to: {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    p.add_argument("--out", type=str, default="anomalies.csv", help="Output CSV path")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000, help="Diffusion steps (beta schedule length)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeavg", type=int, default=8)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--cols", type=str, nargs="*", default=None)
    p.add_argument("--standardize", type=str, default="zscore", choices=["zscore", "robust"])
    p.add_argument("--save_intermediate", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(csv_path=args.csv, out_path=args.out, window=args.window, epochs=args.epochs,
            batch=args.batch, steps=args.steps, hidden=args.hidden, lr=args.lr,
            seed=args.seed, timeavg=args.timeavg, quantile=args.quantile, cols=args.cols,
            scaler=args.standardize, save_intermediate=args.save_intermediate)
