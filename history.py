import os
import requests
import pandas as pd

LATITUDE = 10.7905
LONGITUDE = 79.1378

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2021-01-01",
    "end_date": "2026-09-30",
    "daily": "temperature_2m_max,temperature_2m_min,rain_sum,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean,surface_pressure_mean,cloud_cover_mean",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()["daily"]
    df = pd.DataFrame(data)

    os.makedirs("backend/data", exist_ok=True)
    df.to_csv("backend/data/rainfall_history.csv", index=False)

    print("Rows:", len(df))
    print(df.head())
    print(df.tail())
else:
    print("API Error:", response.status_code, response.text)