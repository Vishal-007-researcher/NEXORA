import os
import json
import numpy as np
import pandas as pd

df = pd.read_csv("backend/data/rainfall_history.csv")
df["time"] = pd.to_datetime(df["time"])

# Features
df["month"] = df["time"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["rain_yesterday"] = df["rain_sum"].shift(1)

# Target: will it rain tomorrow? (1 = yes, 0 = no)
df["rain_tomorrow"] = (df["rain_sum"].shift(-1) > 1).astype(int)

# First and last rows have no yesterday/tomorrow
df = df.iloc[1:-1].reset_index(drop=True)

features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "month_sin",
    "month_cos",
    "rain_yesterday",
]

X = df[features].values.astype(float)
y = df["rain_tomorrow"].values

# Time-based split: first 80% train, last 20% test
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
Xtr = (X_train - mean) / std
Xte = (X_test - mean) / std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Train logistic regression (gradient descent)
w = np.zeros(Xtr.shape[1])
b = 0.0
lr = 0.1
for _ in range(3000):
    p = sigmoid(Xtr @ w + b)
    w -= lr * (Xtr.T @ (p - y_train) / len(y_train))
    b -= lr * (p - y_train).mean()

# Evaluate
pred = (sigmoid(Xte @ w + b) >= 0.5).astype(int)
accuracy = (pred == y_test).mean()
baseline = 1 - y_test.mean()

tp = ((pred == 1) & (y_test == 1)).sum()
fp = ((pred == 1) & (y_test == 0)).sum()
fn = ((pred == 0) & (y_test == 1)).sum()
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0

print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Model accuracy:", round(accuracy, 3))
print("Baseline (always predict dry):", round(baseline, 3))
print("Rain precision:", round(precision, 3))
print("Rain recall:", round(recall, 3))
print()
print("Feature weights (bigger = more influence):")
for name, weight in sorted(zip(features, w), key=lambda x: -abs(x[1])):
    print(f"  {name}: {weight:.3f}")

# Save model as a JSON file
os.makedirs("backend/ml", exist_ok=True)
model = {
    "features": features,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "weights": w.tolist(),
    "bias": float(b),
}
with open("backend/ml/model.json", "w") as f:
    json.dump(model, f)
print("\nModel saved to backend/ml/model.json")