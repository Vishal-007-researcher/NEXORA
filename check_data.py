import pandas as pd

df = pd.read_csv("backend/data/rainfall_history.csv")

print("Shape:", df.shape)
print()
print("Missing values:")
print(df.isnull().sum())
print()
print("Rainy days (rain_sum > 1 mm):", (df["rain_sum"] > 1).sum())
print("Dry days:", (df["rain_sum"] <= 1).sum())
print()
print(df.describe())

