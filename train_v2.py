import os
import json
import numpy as np
import pandas as pd

df = pd.read_csv("backend/data/rainfall_history.csv")
df["time"] = pd.to_datetime(df["time"])

# Features
month = df["time"].dt.month
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)
df["rain_yesterday"] = df["rain_sum"].shift(1)
df["rain_last3"] = df["rain_sum"].rolling(3).sum()
df["rain_last7"] = df["rain_sum"].rolling(7).sum()
df["pressure_change"] = df["surface_pressure_mean"].diff()
df["humidity_change"] = df["relative_humidity_2m_mean"].diff()
df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

# Target: will it rain tomorrow?
nxt = df["rain_sum"].shift(-1)
df["rain_tomorrow"] = (nxt > 1).astype(int)
df = df[nxt.notna()].dropna().reset_index(drop=True)

OLD_FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "rain_sum",
    "wind_speed_10m_max", "relative_humidity_2m_mean",
    "surface_pressure_mean", "cloud_cover_mean",
    "month_sin", "month_cos", "rain_yesterday",
]
NEW_FEATURES = OLD_FEATURES + [
    "rain_last3", "rain_last7", "pressure_change", "humidity_change", "temp_range",
]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run(features, weighted):
    X = df[features].values.astype(float)
    y = df["rain_tomorrow"].values

    split = int(len(df) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    # Give rainy days more importance if weighted
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1) if weighted else 1.0
    sample_w = np.where(ytr == 1, pos_w, 1.0)

    w = np.zeros(Xtr.shape[1])
    b = 0.0
    for _ in range(3000):
        p = sigmoid(Xtr @ w + b)
        err = (p - ytr) * sample_w
        w -= 0.1 * (Xtr.T @ err / len(ytr) + 0.001 * w)
        b -= 0.1 * err.mean()

    pred = (sigmoid(Xte @ w + b) >= 0.5).astype(int)
    tp = ((pred == 1) & (yte == 1)).sum()
    fp = ((pred == 1) & (yte == 0)).sum()
    fn = ((pred == 0) & (yte == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": float((pred == yte).mean()),
        "baseline": float(1 - yte.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "model": {
            "features": features,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": w.tolist(),
            "bias": float(b),
        },
    }


results = {
    "1. Old features": run(OLD_FEATURES, False),
    "2. New features": run(NEW_FEATURES, False),
    "3. New features + rainy-day weight": run(NEW_FEATURES, True),
}

first = next(iter(results.values()))
print("Rows used:", len(df), "| Baseline (always dry):", round(first["baseline"], 3))
print()
print(f"{'Version':38} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}")
for name, r in results.items():
    print(f"{name:38} {r['accuracy']:>9.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>7.3f}")

best_name = max(results, key=lambda k: results[k]["f1"])
print("\nBest (highest F1):", best_name)

os.makedirs("backend/ml", exist_ok=True)
with open("backend/ml/model_v2.json", "w") as f:
    json.dump(results[best_name]["model"], f)
print("Saved to backend/ml/model_v2.json")