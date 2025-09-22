#!/usr/bin/env python3
# (short notice) This is the plotting-enabled multi-GPU script. See previous cell for full docstring.
import argparse, os, math, numpy as np, pandas as pd
from typing import List, Dict
import torch, torch.nn as nn, torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed:int=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark=True

def sinusoidal_time_embedding(t, dim:int):
    device=t.device
    half=dim//2
    freqs=torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=device))
    angles=t[:,None]*freqs[None,:]
    emb=torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim%2==1: emb=torch.cat([emb, torch.zeros((t.shape[0],1), device=device)], dim=-1)
    return emb

def resolve_device(dev_str:str)->str:
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        try:
            idx=int(dev_str.split(":")[1])
            if 0<=idx<torch.cuda.device_count(): return f"cuda:{idx}"
        except Exception: pass
    return "cpu"

def device_of(dstr:str): return torch.device(dstr if dstr.startswith("cuda") else "cpu")

class WindowDataset(Dataset):
    def __init__(self, series:np.ndarray, window:int):
        self.window=window
        self.x=[series[i:i+window] for i in range(len(series)-window+1)]
        self.x=np.stack(self.x, axis=0).astype(np.float32)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx]

class TinyScoreNet(nn.Module):
    def __init__(self, window:int, hidden:int=128, time_dim:int=64):
        super().__init__()
        self.time_dim=time_dim
        self.net=nn.Sequential(nn.Linear(window+time_dim, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, window))
    def forward(self, x, t):
        temb=sinusoidal_time_embedding(t, self.time_dim)
        h=torch.cat([x, temb], dim=-1)
        return self.net(h)

class BetaSchedule:
    def __init__(self, T:int=1000, s:float=0.008, device='cpu'):
        dev=resolve_device(device)
        self.T=T
        self.device=dev
        t=torch.linspace(0, T, T+1, device=device_of(dev))
        f=torch.cos(((t/T)+s)/(1+s)*math.pi/2)**2
        alpha_bar=f/f[0]
        self.alpha_bar=alpha_bar[1:]
        self.alphas=self.alpha_bar.clone()
        self.alphas[1:]=self.alpha_bar[1:]/self.alpha_bar[:-1]
        self.betas=1-self.alphas
        self.betas=torch.clamp(self.betas,1e-6,0.999)
        self.alphas=1.0-self.betas
    def to(self, device):
        dev=resolve_device(device)
        self.device=dev
        self.alpha_bar=self.alpha_bar.to(device_of(dev))
        self.alphas=self.alphas.to(device_of(dev))
        self.betas=self.betas.to(device_of(dev))
        return self

def standardize_series(x:np.ndarray, mode:str="zscore"):
    if mode=="robust":
        med=np.median(x)
        mad=np.median(np.abs(x-med))+1e-8
        return (x-med)/(1.4826*mad),(med,mad,"robust")
    mu=float(np.mean(x))
    sigma=float(np.std(x)+1e-8)
    return (x-mu)/sigma,(mu,sigma,"zscore")

def normalize_scores(scores:np.ndarray)->np.ndarray:
    lo=np.percentile(scores,5.0)
    hi=np.percentile(scores,99.5)
    s=(scores-lo)/(hi-lo+1e-8)
    return np.clip(s,0.0,1.0)

def choose_threshold(scores:np.ndarray,q:float=0.99)->float: return float(np.quantile(scores,q))

def train_score_model(series:np.ndarray, window:int=64, epochs:int=30, batch:int=256,
                      T:int=1000, hidden:int=128, lr:float=1e-3, device:str='cpu', amp:bool=False,
                      num_workers:int=0, pin_memory:bool=True):
    ds=WindowDataset(series,window)
    dl=DataLoader(ds,batch_size=batch,shuffle=True,drop_last=True,
                  num_workers=num_workers,pin_memory=pin_memory,persistent_workers=num_workers>0)
    betas=BetaSchedule(T=T, device=device).to(device)
    model=TinyScoreNet(window=window,hidden=hidden).to(device_of(device))
    opt=torch.optim.AdamW(model.parameters(), lr=lr)
    scaler=GradScaler(enabled=amp)
    for ep in range(epochs):
        model.train()
        total=0.0; n=0
        for x0 in dl:
            x0=torch.as_tensor(x0,dtype=torch.float32)
            if pin_memory: x0=x0.pin_memory()
            x0=x0.to(device_of(device), non_blocking=True)
            B=x0.shape[0]
            t_idx=torch.randint(0,T,(B,),device=device_of(device))
            at_bar=betas.alpha_bar[t_idx]
            noise=torch.randn_like(x0)
            xt=torch.sqrt(at_bar)[:,None]*x0+torch.sqrt(1-at_bar)[:,None]*noise
            t=(t_idx.float()+1.0)/T
            with autocast(enabled=amp):
                noise_pred=model(xt,t)
                loss=((noise_pred-noise)**2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total+=loss.item()*B
            n+=B
        if (ep+1)%max(1,epochs//5)==0: print(f"[epoch {ep+1}/{epochs}] loss={total/n:.6f}")
    return model, betas

@torch.no_grad()
def score_series(model:TinyScoreNet, betas:BetaSchedule, series:np.ndarray, window:int=64,
                 timeavg:int=8, device:str='cpu', amp:bool=False, num_workers:int=0, pin_memory:bool=True):
    model.eval()
    W=window
    N=len(series)
    windows=[series[i:i+W] for i in range(N-W+1)]
    X=torch.as_tensor(np.stack(windows,axis=0),dtype=torch.float32)
    if pin_memory: 
        X=X.pin_memory()
    X=X.to(device_of(device), non_blocking=True)
    recons_accum=torch.zeros_like(X, device=device_of(device))
    for _ in range(timeavg):
        M=X.shape[0]
        t_idx=torch.randint(0, betas.T,(M,),device=device_of(device))
        at_bar=betas.alpha_bar[t_idx]
        noise=torch.randn_like(X,device=device_of(device))
        xt=torch.sqrt(at_bar)[:,None]*X+torch.sqrt(1-at_bar)[:,None]*noise        
        t=(t_idx.float()+1.0)/betas.T
        with autocast(enabled=amp):
            eps=model(xt,t)
            x0_hat=(xt-torch.sqrt(1-at_bar)[:,None]*eps)/torch.sqrt(at_bar)[:,None]
        recons_accum+=x0_hat
    X_rec=(recons_accum/timeavg).float().cpu().numpy()
    errs=(X_rec - X.cpu().numpy())**2
    w=np.bartlett(W)
    w=w/(w.sum()+1e-8)
    win_scores=(errs*w[None,:]).sum(axis=1)
    point_scores=np.zeros(N,dtype=np.float64)
    counts=np.zeros(N,dtype=np.float64)
    for i in range(N-W+1):
        center=i+W//2
        k=np.array([0.25,0.5,0.25])
        idxs=[max(0,center-1), center, min(N-1,center+1)]
        for j,idx in enumerate(idxs): 
            point_scores[idx]+=win_scores[i]*k[j]
            counts[idx]+=k[j]
    counts[counts==0]=1.0
    return (point_scores/counts).astype(np.float32)

def plot_series_with_anoms(y: np.ndarray, anom_idx: np.ndarray, save_path: str, title: str, dpi: int = 140):
    plt.figure(figsize=(12, 3.2), dpi=dpi)
    plt.plot(np.arange(len(y)), y, label="series", alpha=0.5)
    if anom_idx.size > 0:
        plt.scatter(anom_idx, y[anom_idx], marker='o', c='red', label="anomaly")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scores_with_thr(scores: np.ndarray, thr: float, save_path: str, title: str, dpi: int = 140):
    plt.figure(figsize=(12, 2.8), dpi=dpi)
    plt.plot(np.arange(len(scores)), scores, label="score", alpha=0.5)
    plt.axhline(thr, color="red", linestyle="--", label=f"thr={thr:.3f}")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def worker(rank:int, device:str, col_names:List[str], data_map:Dict[str, np.ndarray],
           out_partial:str, args_dict:dict, plots_dir:str, dpi:int, plot_format:str):
    dev=resolve_device(device)
    print(f"[rank {rank}] using device={dev} columns={col_names}")
    if dev.startswith("cuda"): torch.cuda.set_device(int(dev.split(":")[1]))
    results=None
    for col in col_names:
        series=data_map[col].astype(np.float32)
        xs,_=standardize_series(series, mode=args_dict['standardize'])
        model,betas=train_score_model(xs, window=args_dict['window'], epochs=args_dict['epochs'], batch=args_dict['batch'],
                                      T=args_dict['steps'], hidden=args_dict['hidden'], lr=args_dict['lr'],
                                      device=dev, amp=args_dict['amp'], num_workers=args_dict['num_workers'], pin_memory=True)
        raw_scores=score_series(model, betas, xs, window=args_dict['window'], timeavg=args_dict['timeavg'],
                                device=dev, amp=args_dict['amp'], num_workers=args_dict['num_workers'], pin_memory=True)
        norm_scores=normalize_scores(raw_scores)
        thr=choose_threshold(norm_scores, q=args_dict['quantile'])
        is_anom=(norm_scores>=thr).astype(np.int32)
        anom_idx=np.where(is_anom==1)[0]
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            series_path=os.path.join(plots_dir, f"{col}__series.{plot_format}")
            score_path=os.path.join(plots_dir, f"{col}__score.{plot_format}")
            plot_series_with_anoms(series, anom_idx, series_path, f"{col} (series + anomalies)", dpi=dpi)
            plot_scores_with_thr(norm_scores, thr, score_path, f"{col} (score, thr={thr:.3f})", dpi=dpi)
        df_part=pd.DataFrame({f"{col}_score": norm_scores, f"{col}_is_anom": is_anom})
        results = df_part if results is None else pd.concat([results, df_part], axis=1)
    if results is None: results=pd.DataFrame()
    results.to_csv(out_partial, index=False)
    print(f"[rank {rank}] wrote {out_partial}")

def process(csv_path:str, out_path:str, window:int=64, epochs:int=30, batch:int=256,
            steps:int=1000, hidden:int=128, lr:float=1e-3, seed:int=42, timeavg:int=8,
            quantile:float=0.99, cols:List[str]=None, standardize:str="zscore",
            amp:bool=False, num_workers:int=0, plots_dir:str=None, dpi:int=140, plot_format:str="png"):
    
    set_seed(seed)
    devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df=pd.read_csv(csv_path)
    num_df=df.select_dtypes(include=[np.number]).copy()
    if cols:
        missing=[c for c in cols if c not in num_df.columns]
        if missing: 
            raise ValueError(f"Requested columns not found or non-numeric: {missing}")
        num_df=num_df[cols]
    col_list=list(num_df.columns)
    
    if len(col_list)==0: 
        raise ValueError("No numeric columns found to process.")
    data_map={c:num_df[c].to_numpy(copy=True) for c in col_list}
    n_ranks=max(1,len(devices)) 
    assignments=[[] for _ in range(n_ranks)]
    
    for i,c in enumerate(col_list): 
        assignments[i % n_ranks].append(c)
    if plots_dir is None or plots_dir.strip()=="": 
        plots_dir=os.path.splitext(out_path)[0]+"_plots"
    
    args_dict=dict(window=window, epochs=epochs, batch=batch, steps=steps, hidden=hidden, lr=lr,
                   seed=seed, timeavg=timeavg, quantile=quantile, standardize=standardize, amp=amp, num_workers=num_workers)
    partial_paths=[]
    ctx=mp.get_context("spawn")
    procs=[]
    for rank in range(n_ranks):
        dev=devices[rank] if rank < len(devices) else "cpu"
        out_partial=f"{os.path.splitext(out_path)[0]}__part{rank}.csv"
        partial_paths.append(out_partial)
        p=ctx.Process(target=worker, args=(rank, dev, assignments[rank], data_map, out_partial, args_dict, plots_dir, dpi, plot_format))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode!=0: raise RuntimeError("A worker process exited with a non-zero status. Check logs above.")
    merged=None
    for pp in partial_paths:
        if not os.path.exists(pp): continue
        dfp=pd.read_csv(pp)
        merged = dfp if merged is None else pd.concat([merged, dfp], axis=1)
    if merged is None: merged=pd.DataFrame(index=df.index)
    merged.to_csv(out_path, index=False)
    print(f"\nSaved anomaly scores to: {out_path}")
    for pp in partial_paths:
        if os.path.exists(pp):
            try: os.remove(pp)
            except Exception: pass

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, default="anomalies.csv")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeavg", type=int, default=8)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--cols", type=str, nargs="*", default=None)
    p.add_argument("--standardize", type=str, default="zscore", choices=["zscore","robust"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--plots_dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--plot_format", type=str, default="png", choices=["png","jpg","jpeg","pdf","svg"])
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    process(csv_path=args.csv, out_path=args.out, window=args.window, epochs=args.epochs,
            batch=args.batch, steps=args.steps, hidden=args.hidden, lr=args.lr, seed=args.seed,
            timeavg=args.timeavg, quantile=args.quantile, cols=args.cols, standardize=args.standardize,
            amp=args.amp, num_workers=args.num_workers, plots_dir=args.plots_dir, dpi=args.dpi, plot_format=args.plot_format)
