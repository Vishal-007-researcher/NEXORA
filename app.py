import json
import math
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime
from functools import wraps

import requests
import pandas as pd
from flask import Flask, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

LATITUDE = 10.7905
LONGITUDE = 79.1378

# Use the improved model if it exists, otherwise the old one
MODEL_PATH = "backend/ml/model_v2.json"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "backend/ml/model.json"

with open(MODEL_PATH) as f:
    model = json.load(f)

# ---------- Auth setup ----------
DB_PATH = "backend/data/nexora.db"
KEY_PATH = "backend/data/secret.key"
TOKEN_MAX_AGE = 8 * 60 * 60  # login valid for 8 hours

os.makedirs("backend/data", exist_ok=True)
if not os.path.exists(KEY_PATH):
    with open(KEY_PATH, "w") as f:
        f.write(secrets.token_hex(32))
with open(KEY_PATH) as f:
    SECRET_KEY = f.read().strip()

serializer = URLSafeTimedSerializer(SECRET_KEY, salt="nexora-auth")

FAILED = {}
MAX_FAILS = 5
LOCK_SECONDS = 300


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "username TEXT UNIQUE NOT NULL, "
        "password_hash TEXT NOT NULL, "
        "role TEXT NOT NULL DEFAULT 'user', "
        "created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()


init_db()

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return resp


def current_user():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    try:
        data = serializer.loads(header[7:], max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    conn = db()
    row = conn.execute("SELECT username, role FROM users WHERE username = ?", (data.get("u"),)).fetchone()
    conn.close()
    return dict(row) if row else None


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            return jsonify({"error": "Login required"}), 401
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Login required"}), 401
        if user["role"] != "admin":
            return jsonify({"error": "Admin only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def is_locked(username):
    entry = FAILED.get(username)
    if not entry:
        return False
    count, since = entry
    if time.time() - since > LOCK_SECONDS:
        FAILED.pop(username, None)
        return False
    return count >= MAX_FAILS


def record_fail(username):
    count, since = FAILED.get(username, (0, time.time()))
    FAILED[username] = (count + 1, since)


# ---------- Auth routes ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip().lower()
    password = str(data.get("password", ""))

    if not re.fullmatch(r"[a-z0-9_]{3,20}", username):
        return jsonify({"error": "Username 3 to 20 characters: letters, numbers, _ mattum."}), 400
    if len(password) < 8 or len(password) > 100:
        return jsonify({"error": "Password kurainjadhu 8 characters irukkanum."}), 400

    conn = db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'user', ?)",
            (username, generate_password_hash(password), datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Andha username already irukku."}), 409
    finally:
        conn.close()

    return jsonify({"message": "Account create aayiduchu. Ippo login pannunga."}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip().lower()
    password = str(data.get("password", ""))
    portal = data.get("portal", "user")

    if is_locked(username):
        return jsonify({"error": "Romba thappaana attempts. 5 nimidam aprom try pannunga."}), 429

    conn = db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    ok = row is not None and check_password_hash(row["password_hash"], password)
    if not ok or (portal == "admin" and row["role"] != "admin"):
        record_fail(username)
        return jsonify({"error": "Username illa-na password thappu."}), 401
    if portal != "admin" and row["role"] == "admin":
        return jsonify({"error": "Idhu admin account. Admin login tab use pannunga."}), 403

    FAILED.pop(username, None)
    token = serializer.dumps({"u": row["username"]})
    return jsonify({"token": token, "username": row["username"], "role": row["role"]})


@app.route("/me")
@login_required
def me():
    return jsonify(current_user())


# ---------- Admin routes ----------
@app.route("/admin/users")
@admin_required
def admin_users():
    conn = db()
    rows = conn.execute("SELECT username, role, created_at FROM users ORDER BY id").fetchall()
    conn.close()
    users = [dict(r) for r in rows]
    return jsonify({
        "users": users,
        "total_users": len(users),
        "model_used": MODEL_PATH,
    })


@app.route("/admin/users/<username>", methods=["DELETE"])
@admin_required
def admin_delete_user(username):
    username = username.strip().lower()
    conn = db()
    row = conn.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "User illa."}), 404
    if row["role"] == "admin":
        conn.close()
        return jsonify({"error": "Admin account-ai delete panna mudiyaadhu."}), 400
    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return jsonify({"message": username + " delete aayiduchu."})


# ---------- Weather helpers ----------
def get_coords():
    try:
        lat = float(request.args.get("lat", LATITUDE))
        lon = float(request.args.get("lon", LONGITUDE))
    except ValueError:
        lat, lon = LATITUDE, LONGITUDE
    return lat, lon


def get_forecast(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,rain,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,rain_sum,wind_speed_10m_max,relative_humidity_2m_mean,surface_pressure_mean,cloud_cover_mean",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_openmeteo_tomorrow(lat, lon):
    # Open-Meteo's own rain probability for tomorrow (None if unavailable)
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_probability_max",
            "forecast_days": 2,
            "timezone": "Asia/Kolkata",
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        r.raise_for_status()
        return r.json()["daily"]["precipitation_probability_max"][1]
    except Exception:
        return None


# ---------- Weather routes (login required) ----------
@app.route("/")
def home():
    return jsonify({"message": "Nexora API running"})


@app.route("/weather")
@login_required
def weather():
    lat, lon = get_coords()
    return jsonify(get_forecast(lat, lon)["current"])


@app.route("/forecast")
@login_required
def forecast():
    lat, lon = get_coords()
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    return jsonify(r.json()["daily"])


@app.route("/predict")
@login_required
def predict():
    lat, lon = get_coords()
    data = get_forecast(lat, lon)
    d = data["daily"]
    month = datetime.now().month

    def last(name, k):
        vals = d[name][-k:]
        if len(vals) < k or any(v is None for v in vals):
            raise ValueError("incomplete data")
        return vals

    try:
        rain7 = last("rain_sum", 7)
        press2 = last("surface_pressure_mean", 2)
        hum2 = last("relative_humidity_2m_mean", 2)
        tmax = last("temperature_2m_max", 1)[0]
        tmin = last("temperature_2m_min", 1)[0]
        wind = last("wind_speed_10m_max", 1)[0]
        cloud = last("cloud_cover_mean", 1)[0]
    except ValueError:
        return jsonify({"error": "Weather API returned incomplete data"}), 502

    values = {
        "temperature_2m_max": tmax,
        "temperature_2m_min": tmin,
        "rain_sum": rain7[-1],
        "wind_speed_10m_max": wind,
        "relative_humidity_2m_mean": hum2[-1],
        "surface_pressure_mean": press2[-1],
        "cloud_cover_mean": cloud,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "rain_yesterday": rain7[-2],
        "rain_last3": sum(rain7[-3:]),
        "rain_last7": sum(rain7),
        "pressure_change": press2[-1] - press2[-2],
        "humidity_change": hum2[-1] - hum2[-2],
        "temp_range": tmax - tmin,
    }

    z = model["bias"]
    for name, m, s, w in zip(model["features"], model["mean"], model["std"], model["weights"]):
        z += ((values[name] - m) / s) * w

    probability = 1 / (1 + math.exp(-z))

    return jsonify({
        "rain_probability_percent": round(probability * 100, 1),
        "rain_tomorrow": probability >= 0.5,
        "openmeteo_probability_percent": get_openmeteo_tomorrow(lat, lon),
        "model_used": MODEL_PATH,
        "inputs_used": values,
    })


@app.route("/history")
@login_required
def history():
    df = pd.read_csv("backend/data/rainfall_history.csv")
    df["time"] = pd.to_datetime(df["time"])

    df["ym"] = df["time"].dt.to_period("M").astype(str)
    monthly = df.groupby("ym")["rain_sum"].sum().round(1).tail(12)
    yearly = df.groupby(df["time"].dt.year)["rain_sum"].sum().round(1)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "rain_sum",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    d = r.json()["daily"]

    heaviest = df.loc[df["rain_sum"].idxmax()]

    return jsonify({
        "last7": {"dates": d["time"], "rain": d["rain_sum"]},
        "monthly": {"labels": list(monthly.index), "rain": monthly.tolist()},
        "yearly": {"labels": [str(y) for y in yearly.index.tolist()], "rain": yearly.tolist()},
        "summary": {
            "total_days": int(len(df)),
            "rainy_days": int((df["rain_sum"] > 1).sum()),
            "dry_days": int((df["rain_sum"] <= 1).sum()),
            "heaviest_mm": float(heaviest["rain_sum"]),
            "heaviest_date": heaviest["time"].strftime("%Y-%m-%d"),
            "from": df["time"].min().strftime("%Y-%m-%d"),
            "to": df["time"].max().strftime("%Y-%m-%d"),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)