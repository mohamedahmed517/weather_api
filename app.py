import os
import re
from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

app = Flask(__name__)

# ====================== إعدادات ======================
WINDOW_SIZE = 10
EPOCHS = 15          # زي ما طلبت
BATCH_SIZE = 16

# ====================== كشف الـ IP ======================
IPV4_PRIVATE = re.compile(r'^(127\.0\.0\.1|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)')

def is_private_ip(ip: str) -> bool:
    return bool(IPV4_PRIVATE.match(ip))

def get_user_ip() -> str:
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip and not is_private_ip(cf_ip):
        return cf_ip
    true_ip = request.headers.get("True-Client-IP")
    if true_ip and not is_private_ip(true_ip):
        return true_ip
    real_ip = request.headers.get("X-Real-IP")
    if real iron_ip and not is_private_ip(real_ip):
        return real_ip
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        for ip in [i.strip() for i in fwd.split(",")]:
            if ip and not is_private_ip(ip):
                return ip
    client_ip = request.headers.get("X-Client-IP")
    if client_ip and not is_private_ip(client_ip):
        return client_ip
    forwarded = request.headers.get("Forwarded")
    if forwarded:
        for part in forwarded.split(";"):
            if part.strip().lower().startswith("for="):
                ip = part.split("=", 1)[1].strip().strip('"[]')
                if ip and not is_private_ip(ip):
                    return ip
    remote = request.remote_addr
    if remote and not is_private_ip(remote):
        return remote
    return "127.0.0.1"

# ====================== وظائف مساعدة ======================
def suggest_outfit(temp, rain):
    if rain > 2.0:
        return "الجو ممطر - خُد شمسية وجاكيت خفيف"
    elif temp < 10:
        return "برد جدًا - البس جاكيت تقيل وبلوفر"
    elif temp < 18:
        return "جو بارد - البس جاكيت أو بلوفر خفيف"
    elif temp < 26:
        return "جو معتدل - تيشيرت وجينز كفاية"
    elif temp < 32:
        return "جو دافي - تيشيرت خفيف وبنطلون صيفي"
    else:
        return "حر جدًا - شورت وتيشيرت خفيف واشرب مياه كتير"

def get_location(ip):
    try:
        url = f"http://ip-api.com/json/{ip}?fields=city,lat,lon,timezone"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d = r.json()
        return {k: d.get(k) for k in ("city", "lat", "lon", "timezone")}
    except Exception as e:
        print(f"IP → Location error: {e}")
        return None

def fetch_weather(lat, lon, tz, start, end):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        f"&start_date={start}&end_date={end}&timezone={tz}"
    )
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()

# ====================== دالة التنبؤ ======================
def lstm_predict(data, days_ahead):
    if "daily" not in data or "time" not in data["daily"]:
        raise ValueError("البيانات من open-meteo غير كاملة")

    df = pd.DataFrame(data["daily"])
    if len(df) < WINDOW_SIZE:
        raise ValueError(f"البيانات قليلة جدًا: {len(df)} يوم فقط (يحتاج {WINDOW_SIZE} على الأقل)")

    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    df = df[["time", "temp_mean", "precipitation_sum", "windspeed_10m_max"]]

    features = df[["temp_mean", "precipitation_sum", "windspeed_10m_max"]].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    if len(features_scaled) <= WINDOW_SIZE:
        raise ValueError(f"البيانات غير كافية للتدريب: {len(features_scaled)} يوم")

    X, y = [], []
    for i in range(len(features_scaled) - WINDOW_SIZE):
        X.append(features_scaled[i:i + WINDOW_SIZE])
        y.append(features_scaled[i + WINDOW_SIZE, 0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        raise ValueError("لا توجد بيانات كافية لتدريب النموذج")

    model = Sequential([
        Input(shape=(WINDOW_SIZE, X.shape[2])),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    last_seq = features_scaled[-WINDOW_SIZE:].copy()
    predictions = []

    for _ in range(days_ahead):
        input_seq = np.expand_dims(last_seq, axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]

        inv = np.zeros((1, features.shape[1]))
        inv[0, 0] = pred_scaled
        predicted_temp = scaler.inverse_transform(inv)[0, 0]
        predictions.append(predicted_temp)

        new_row = np.array([[pred_scaled, last_seq[-1, 1], last_seq[-1, 2]]])
        last_seq = np.vstack((last_seq[1:], new_row))

    rain = df.iloc[-1]["precipitation_sum"]
    return predictions, rain

# ====================== الـ Routes ======================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Weather API شغال!",
        "endpoint": "/api/weather (POST)",
        "example": {"days": 7},
        "status": "OK"
    })

@app.route("/api/weather", methods=["POST"])
def weather():
    try:
        user_ip = get_user_ip()
        days = int(request.json.get("days", 7))
        if days < 1 or days > 16:
            return jsonify({"error": "عدد الأيام من 1 إلى 16"}), 400

        # === الحل: اطلب دايمًا 10 أيام على الأقل ===
        fetch_days = max(days, 10)

        loc = get_location(user_ip)
        if not loc:
            return jsonify({"error": "فشل تحديد الموقع"}), 400

        start = date.today()
        end = start + timedelta(days=fetch_days)
        weather_data = fetch_weather(loc["lat"], loc["lon"], loc["timezone"],
                                     start.isoformat(), end.isoformat())

        if "daily" not in weather_data or len(weather_data["daily"]["time"]) < WINDOW_SIZE:
            return jsonify({
                "error": f"البيانات غير كافية من open-meteo (حصلت على {len(weather_data['daily']['time'])} يوم فقط)"
            }), 400

        temps, rain = lstm_predict(weather_data, days)

        results = []
        for i, temp in enumerate(temps, 1):
            d = (start + timedelta(days=i)).strftime("%d-%m-%Y")
            outfit = suggest_outfit(temp, rain)
            results.append({"date": d, "temp": round(temp, 1), "outfit": outfit})

        return jsonify({
            "city": loc["city"],
            "user_ip": user_ip,
            "results": results
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"خطأ داخلي: {str(e)}"}), 500

# ====================== تشغيل السيرفر ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
