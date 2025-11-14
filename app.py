import os
import re
import threading
import time
import requests
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

app = Flask(__name__)

# ====================== إعدادات ======================
WINDOW_SIZE = 10
EPOCHS = 7          # شغال سريع ودقيق
BATCH_SIZE = 16

# ====================== كشف الـ IP ======================
IPV4_PRIVATE = re.compile(r'^(127\.0\.0\.1|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)')

def is_private_ip(ip: str) -> bool:
    return bool(IPV4_PRIVATE.match(ip))

def get_user_ip() -> str:
    headers = [
        "CF-Connecting-IP", "True-Client-IP", "X-Real-IP",
        "X-Forwarded-For", "X-Client-IP", "Forwarded"
    ]
    for header in headers:
        value = request.headers.get(header)
        if value:
            ips = [i.strip() for i in value.replace('"', '').split(",")]
            for ip in ips:
                if ip and not is_private_ip(ip):
                    return ip
    return request.remote_addr or "127.0.0.1"

# ====================== وظائف مساعدة ======================
def suggest_outfit(temp, rain):
    if rain > 2.0: return "الجو ممطر - خُد شمسية وجاكيت خفيف"
    elif temp < 10: return "برد جدًا - البس جاكيت تقيل وبلوفر"
    elif temp < 18: return "جو بارد - البس جاكيت أو بلوفر خفيف"
    elif temp < 26: return "جو معتدل - تيشيرت وجينز كفاية"
    elif temp < 32: return "جو دافي - تيشيرت خفيف وبنطلون صيفي"
    else: return "حر جدًا - شورت وتيشيرت خفيف واشرب مياه كتير"

def get_location(ip):
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}?fields=city,lat,lon,timezone", timeout=8)
        r.raise_for_status()
        d = r.json()
        return {k: d.get(k) for k in ("city", "lat", "lon", "timezone")}
    except:
        return None

def fetch_weather(lat, lon, tz, start, end):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&start_date={start}&end_date={end}&timezone={tz}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def lstm_predict(data, days_ahead):
    df = pd.DataFrame(data["daily"])
    if len(df) < WINDOW_SIZE:
        raise ValueError(f"البيانات قليلة: {len(df)} يوم")

    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    df = df[["temp_mean", "precipitation_sum", "windspeed_10m_max"]]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(len(scaled) - WINDOW_SIZE):
        X.append(scaled[i:i + WINDOW_SIZE])
        y.append(scaled[i + WINDOW_SIZE, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([Input(shape=(WINDOW_SIZE, 3)), LSTM(64, return_sequences=True), LSTM(32), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    seq = scaled[-WINDOW_SIZE:].copy()
    preds = []
    for _ in range(days_ahead):
        pred = model.predict(np.expand_dims(seq, 0), verbose=0)[0, 0]
        inv = np.zeros((1, 3)); inv[0, 0] = pred
        temp = scaler.inverse_transform(inv)[0, 0]
        preds.append(temp)
        new_row = np.array([[pred, seq[-1, 1], seq[-1, 2]]])
        seq = np.vstack((seq[1:], new_row))
    return preds, df["precipitation_sum"].iloc[-1]

# ====================== Keep Warm ======================
def keep_warm():
    while True:
        time.sleep(300)
        try:
            requests.post("https://web-production-cdbe.up.railway.app/api/weather", json={"days": 1}, timeout=10)
        except:
            pass
threading.Thread(target=keep_warm, daemon=True).start()

# ====================== Routes ======================
@app.route("/")
def home():
    return jsonify({"message": "Weather API شغال 100%!", "endpoint": "/api/weather"})

@app.route("/api/weather", methods=["POST"])
def weather():
    try:
        user_ip = get_user_ip()
        days = int(request.json.get("days", 7))
        if not 1 <= days <= 16:
            return jsonify({"error": "الأيام من 1 إلى 16"}), 400

        loc = get_location(user_ip)
        if not loc:
            return jsonify({"error": "فشل تحديد الموقع"}), 400

        fetch_days = max(days, 10)
        start = date.today()
        end = start + timedelta(days=fetch_days)
        data = fetch_weather(loc["lat"], loc["lon"], loc["timezone"], start.isoformat(), end.isoformat())

        if len(data["daily"]["time"]) < WINDOW_SIZE:
            return jsonify({"error": f"البيانات قليلة: {len(data['daily']['time'])} يوم"}), 400

        temps, rain = lstm_predict(data, days)
        results = []
        for i, temp in enumerate(temps, 1):
            d = (start + timedelta(days=i)).strftime("%d-%m-%Y")
            results.append({"date": d, "temp": round(temp, 1), "outfit": suggest_outfit(temp, rain)})

        return jsonify({"city": loc["city"], "user_ip": user_ip, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
