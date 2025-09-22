
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

INPUT_CSV = "./data/preprocessing.csv"
OUTDIR = "anomaly_outputs_statistics"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.sort_values("Time").reset_index(drop=True)

def save_csv(df_obj: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUTDIR, name)
    df_obj.to_csv(path, index=False)
    return path

# 1) Quick stats
ohc = df["OxygenHighChamber"].dropna()
desc = ohc.describe()
save_csv(desc.to_frame(name="OxygenHighChamber_stats").reset_index(), "oxygen_high_chamber_stats.csv")

# 2) IQR bounds + plots
Q1 = ohc.quantile(0.25)
Q3 = ohc.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

plt.figure(figsize=(10,6))
plt.hist(ohc, bins=100, alpha=0.7, label="OxygenHighChamber")
plt.axvline(lower_bound, linestyle="--", label=f"Lower Bound {lower_bound:.3f}")
plt.axvline(upper_bound, linestyle="--", label=f"Upper Bound {upper_bound:.3f}")
plt.title("Distribution of OxygenHighChamber with IQR Bounds")
plt.xlabel("OxygenHighChamber")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(os.path.join(OUTDIR, "ohc_hist_iqr_bounds.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,4))
plt.boxplot(ohc, vert=False)
plt.title("Boxplot of OxygenHighChamber")
plt.xlabel("OxygenHighChamber")
plt.savefig(os.path.join(OUTDIR, "ohc_boxplot.png"), bbox_inches="tight")
plt.close()

# 3) Intervals where OxygenHighChamber >= upper bound
ohc_outlier = df[df["OxygenHighChamber"] >= upper_bound][["Time", "OxygenHighChamber"]].copy()
if not ohc_outlier.empty:
    # ohc_outlier["gap"] = ohc_outlier["Time"].diff().dt.total_seconds().gt(60).cumsum()
    # 이전 행과의 시간 차이 계산 (Timedelta 형식)
    time_diff = ohc_outlier["Time"].diff()    
    # 초(second) 단위로 변환
    time_diff_seconds = time_diff.dt.total_seconds()
    # 시간 차이가 60초보다 큰지 여부 (True/False)
    is_new_group = time_diff_seconds > 60
    # True가 나오면 그룹 번호를 하나 증가 (누적합으로 그룹 ID 생성)
    group_id = is_new_group.cumsum()
    # 최종적으로 gap 컬럼에 저장
    ohc_outlier["gap"] = group_id
    
    intervals = ohc_outlier.groupby("gap").agg(
        Start=("Time", "first"), 
        End=("Time", "last"), 
        Count=("Time", "count"), 
        MaxValue=("OxygenHighChamber", "max") 
    ).reset_index(drop=True)
else:
    intervals = pd.DataFrame(columns=["Start","End","Count","MaxValue"])
intervals.to_csv(os.path.join(OUTDIR, "ohc_outlier_intervals_ge_upper.csv"), index=False)

# 4) FFT periodicity (1s)
series_1s = df.set_index("Time")["OxygenHighChamber"].resample("1S").mean().interpolate()
signal = series_1s.values - np.mean(series_1s.values)
fft_vals = np.fft.rfft(signal)
fft_freqs = np.fft.rfftfreq(len(signal), d=1.0)
if len(fft_vals) > 1:
    dom_idx = np.argmax(np.abs(fft_vals[1:])) + 1
    dominant_freq = float(fft_freqs[dom_idx])
    dominant_period = (1.0 / dominant_freq) if dominant_freq > 0 else None
else:
    dominant_freq, dominant_period = None, None

plt.figure(figsize=(10,5))
plt.plot(fft_freqs[1:], np.abs(fft_vals[1:]))
plt.title("FFT Spectrum (OxygenHighChamber, 1s resampled)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.savefig(os.path.join(OUTDIR, "ohc_fft_spectrum.png"), bbox_inches="tight")
plt.close()

# 5) Trend anomalies for all numeric columns
numeric_cols = df.drop(columns=["Time","Layer"]).columns.tolist() if "Layer" in df.columns else df.drop(columns=["Time"]).columns.tolist()
trend_counts = []
trend_points_all = []

for col in numeric_cols:
    series_1m = df.set_index("Time")[col].resample("1T").mean().interpolate()
    trend = series_1m.rolling(window=60, center=True, min_periods=30).mean()
    resid = series_1m - trend
    std = float(resid.std())
    anomalies = resid[np.abs(resid) > 3 * std]

    trend_counts.append({"Column": col, "AnomalyCount": int(anomalies.shape[0])})
    if not anomalies.empty:
        temp = pd.DataFrame({
            "Time": anomalies.index,
            "Column": col,
            "Value": series_1m.loc[anomalies.index].values,
            "Trend": trend.loc[anomalies.index].values,
            "Residual": anomalies.values,
            "ThresholdAbs": 3 * std
        })
        trend_points_all.append(temp)

    plt.figure(figsize=(12,5))
    plt.plot(series_1m.index, series_1m.values, label=col, alpha=0.7)
    plt.plot(trend.index, trend.values, label="Rolling Mean (Trend)")
    if not anomalies.empty:
        plt.scatter(anomalies.index, series_1m.loc[anomalies.index].values, label="Trend Anomaly", s=25)
    plt.title(f"Trend/Context Anomalies in {col}")
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"trend_anomalies_{col}.png"), bbox_inches="tight")
    plt.close()

pd.DataFrame(trend_counts).to_csv(os.path.join(OUTDIR, "trend_anomaly_counts.csv"), index=False)
if trend_points_all:
    pd.concat(trend_points_all, ignore_index=True).to_csv(os.path.join(OUTDIR, "trend_anomaly_points.csv"), index=False)
else:
    pd.DataFrame(columns=["Time","Column","Value","Trend","Residual","ThresholdAbs"]).to_csv(os.path.join(OUTDIR, "trend_anomaly_points.csv"), index=False)

# 6) Contextual anomalies via STL
def stl_context(series: pd.Series, period: int = 1440, sigma: float = 3.0):
    stl = sm.tsa.STL(series, period=period, robust=True)
    result = stl.fit()
    resid = result.resid
    std = float(resid.std())
    anomalies = resid[np.abs(resid) > sigma * std]
    return result, anomalies, std

context_cols = [c for c in ["OxygenHighChamber", "TemperatureChamber"] if c in df.columns]
context_counts = []
context_points_all = []

for col in context_cols:
    series_1m = df.set_index("Time")[col].resample("1T").mean().interpolate()
    result, anomalies, std = stl_context(series_1m, period=1440, sigma=3.0)

    fig = result.plot()
    fig.set_size_inches(10, 7)
    fig.suptitle(f"STL Decomposition of {col}")
    fig.savefig(os.path.join(OUTDIR, f"stl_decomposition_{col}.png"), bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(12,5))
    plt.plot(series_1m.index, series_1m.values, label=col, alpha=0.7)
    if not anomalies.empty:
        plt.scatter(anomalies.index, series_1m.loc[anomalies.index].values, label="Contextual Anomaly", s=25)
    plt.title(f"Contextual Anomalies in {col} (STL residual > 3σ)")
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"stl_contextual_anomalies_{col}.png"), bbox_inches="tight")
    plt.close()

    context_counts.append({"Column": col, "AnomalyCount": int(anomalies.shape[0])})
    if not anomalies.empty:
        temp = pd.DataFrame({
            "Time": anomalies.index,
            "Column": col,
            "Value": series_1m.loc[anomalies.index].values,
            "Residual": anomalies.values,
            "ThresholdAbs": 3 * std
        })
        context_points_all.append(temp)

pd.DataFrame(context_counts).to_csv(os.path.join(OUTDIR, "contextual_anomaly_counts.csv"), index=False)
if context_points_all:
    pd.concat(context_points_all, ignore_index=True).to_csv(os.path.join(OUTDIR, "contextual_anomaly_points.csv"), index=False)
else:
    pd.DataFrame(columns=["Time","Column","Value","Residual","ThresholdAbs"]).to_csv(os.path.join(OUTDIR, "contextual_anomaly_points.csv"), index=False)

# 7) IQR summary for all numeric columns
numeric_cols = df.drop(columns=["Time","Layer"]).columns.tolist() if "Layer" in df.columns else df.drop(columns=["Time"]).columns.tolist()
rows = []
for col in numeric_cols:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    ratio = ((s < lb) | (s > ub)).mean() * 100.0
    rows.append({"Column": col, "LowerBound": lb, "UpperBound": ub, "OutlierRatio(%)": ratio, "Min": s.min(), "Max": s.max()})
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "iqr_outlier_summary.csv"), index=False)

print("Done. Outputs saved to:", OUTDIR)
