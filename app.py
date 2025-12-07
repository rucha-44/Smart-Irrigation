from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import json
import statistics
from werkzeug.security import generate_password_hash, check_password_hash
from services.weather_service import WeatherService
from translator import tr
from groq import Groq
from flask_cors import CORS
from scheduler_controller import scheduler_bp

app = Flask(__name__)
weather_service = WeatherService()

# --- Scheduler ---
app.register_blueprint(scheduler_bp, url_prefix='/scheduler')

# --- Groq Client ---
# Using the key you provided in previous context
# Using the key you provided
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app.secret_key = "supersecretkey123"
USERS_FILE = "users.json"

# Enable CORS
CORS(app)

# -------------------------------------------
# DATA: Soil Moisture Ranges (New Feature)
# -------------------------------------------
# Qualitative Ranges based on Volumetric Water Content (VWC %)
SOIL_MOISTURE_RANGES = {
    'Sandy': {
        'Dry': (0, 10),      # Wilting point is ~5-10%
        'Moist': (11, 20),   # Field Capacity is ~20%
        'Wet': (21, 100)     # Drains rapidly above 20%
    },
    'Loamy': {
        'Dry': (0, 15),
        'Moist': (16, 30),   # Ideal agricultural range
        'Wet': (31, 100)
    },
    'Clayey': {
        'Dry': (0, 25),      # Clay holds water tightly; plants wilt even at 20%
        'Moist': (26, 45),
        'Wet': (46, 100)     # Prone to waterlogging
    },
    'Silty': {
        'Dry': (0, 15),
        'Moist': (16, 35),   
        'Wet': (36, 100)
    },
    'Peaty': {
        'Dry': (0, 30),      
        'Moist': (31, 60),   
        'Wet': (61, 100)
    },
    'Chalky': {
        'Dry': (0, 10),      
        'Moist': (11, 25),
        'Wet': (26, 100)
    }
}

# -------------------------------------------
# USER AUTH HELPERS
# -------------------------------------------

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# -------------------------------------------
# AUTH ROUTES
# -------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        users = load_users()
        for u in users:
            if u["email"] == email and check_password_hash(u["password"], password):
                session["user"] = u["email"]
                return redirect(url_for("home"))
        error = "Invalid email or password!"
    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.form
        users = load_users()
        for u in users:
            if u["email"] == data["email"]:
                return render_template("signup.html", error="Email already registered!")
        new_user = {
            "email": data["email"],
            "username": data["username"],
            "password": generate_password_hash(data["password"]),
            "land_area": "",
            "crops": ""
        }
        users.append(new_user)
        save_users(users)
        session["user"] = data["email"]
        return redirect(url_for("profile"))
    return render_template("signup.html")

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form["email"]
        users = load_users()
        for u in users:
            if u["email"] == email:
                return "📩 Password reset link sent! (Demo)"
        return "❌ No account found!"
    return render_template("forgot.html")

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    users = load_users()
    current_user = None
    for u in users:
        if u["email"] == session["user"]:
            current_user = u
            break
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        land_area = request.form.get("land_area", "").strip()
        crops_text = request.form.get("crops", "").strip()
        crops_list = [c.strip() for c in crops_text.split("\n") if c.strip()]
        current_user["username"] = username
        current_user["land_area"] = land_area
        current_user["crops"] = crops_list
        save_users(users)
        return redirect(url_for("home"))
    return render_template("profile.html", user=current_user)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------------------------------------------------------
# MODEL + PREDICTION LOGIC
# -------------------------------------------------------

feature_scaler = None
target_scaler = None
feature_columns = None
dnn_classifier = None
dnn_regressor = None

MAIN_CROPS = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Tomato', 'Potato', 'Coffee']

CROP_FACTS = {
    'Rice': "Rice is semi-aquatic. It requires standing water or consistently high moisture (>70%).",
    'Wheat': "Wheat needs water during 'Crown Root Initiation'. Keep moisture > 45%.",
    'Maize': "Maize is sensitive during silking. Keep above 50%.",
    'Cotton': "Cotton needs dry periods to burst bolls. Irrigate only if critical (<35%).",
    'Sugarcane': "Sugarcane consumes high water. Needs consistent moisture (>65%).",
    'Tomato': "Tomatoes need moisture >60% to avoid blossom end rot.",
    'Potato': "Shallow roots need frequent moisture (>45%).",
    'Coffee': "Coffee needs a dry stress period (<30%) before flowering."
}

CROP_THRESHOLDS = {
    'Rice': 70, 'Sugarcane': 65, 'Tomato': 60, 'Maize': 50,
    'Wheat': 45, 'Potato': 45, 'Cotton': 35, 'Coffee': 30
}

def load_artifacts():
    global feature_scaler, target_scaler, feature_columns, dnn_classifier, dnn_regressor
    path = "models/saved/"
    try:
        if os.path.exists(path):
            feature_scaler = joblib.load(f"{path}feature_scaler.pkl")
            target_scaler = joblib.load(f"{path}target_scaler.pkl")
            feature_columns = joblib.load(f"{path}feature_columns.pkl")
            dnn_classifier = tf.keras.models.load_model(f"{path}dnn_classifier.h5", compile=False)
            dnn_regressor = tf.keras.models.load_model(f"{path}dnn_regressor.h5", compile=False)
            print("✅ Artifacts Loaded.")
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

load_artifacts()

def get_prediction(input_dict):
    try:
        df_input = pd.DataFrame(input_dict)
        df_input = pd.get_dummies(df_input, columns=["CROP_TYPE", "SOIL_TYPE", "REGION", "WEATHER_CONDITION"])
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        
        cols = ["TEMPERATURE", "HUMIDITY", "RAINFALL", "WIND_SPEED", "SOIL_MOISTURE"]
        df_input[cols] = feature_scaler.transform(df_input[cols])
        
        need_prob = float(dnn_classifier.predict(df_input, verbose=0)[0][0])
        water_scaled = dnn_regressor.predict(df_input, verbose=0)
        water_amount = float(target_scaler.inverse_transform(water_scaled)[0][0])
        water_amount = max(0.0, water_amount)
        
        return need_prob, water_amount
    except Exception as e:
        print(e)
        return 0.0, 0.0

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    options = {
        "CROP_TYPE": MAIN_CROPS,
        "SOIL_TYPE": ['Sandy', 'Loamy', 'Clayey', 'Silty', 'Peaty', 'Chalky'],
        "REGION": ['North', 'East', 'West', 'South', 'Central'],
        "WEATHER_CONDITION": ['Sunny', 'Rainy', 'Cloudy', 'Windy']
    }
    return render_template("index.html", options=options)

@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.json.get("city")
    data = weather_service.get_weather_data(city)
    return jsonify(data)

@app.route("/predict_forecast", methods=["POST"])
def predict_forecast():
    try:
        data = request.json
        city = data.get('city')
        base = data.get('base_inputs')
        forecasts = weather_service.get_forecast_data(city)
        if "error" in forecasts: 
            return jsonify(forecasts)
        
        preds = []
        crop = base['crop_type']
        threshold = CROP_THRESHOLDS.get(crop, 50)
        
        # Determine numeric moisture for forecast base
        base_moist = 45 # Default fallback
        
        # Check if we have a condition (Dry/Moist) OR a raw number
        if 'soil_condition' in base and base['soil_condition']:
             s_type = base.get('soil_type', 'Loamy')
             cond = base.get('soil_condition')
             # Get the range and calculate mean
             rng = SOIL_MOISTURE_RANGES.get(s_type, {}).get(cond, (40, 50))
             base_moist = statistics.mean(rng)
        elif 'soil_moisture' in base:
             base_moist = float(base['soil_moisture'])

        for day in forecasts:
            # Simulate moisture dropping slightly each day
            curr_moist = base_moist - 5 
            day_in = {
                "CROP_TYPE": [crop], "SOIL_TYPE": [base['soil_type']],
                "REGION": [base['region']],
                "TEMPERATURE": [day['temperature']],
                "HUMIDITY": [day['humidity']],
                "RAINFALL": [day['rainfall']],
                "WIND_SPEED": [day['wind_speed']],
                "WEATHER_CONDITION": [day['weather_condition']],
                "SOIL_MOISTURE": [curr_moist]
            }
            prob, amount = get_prediction(day_in)
            
            needs = True
            if day['rainfall'] > 10: 
                needs = False; amount = 0.0
            if curr_moist > threshold:
                needs = False; amount = 0.0
            
            preds.append({
                "day": day['day_name'],
                "date": day['date_short'],
                "condition": day['condition_desc'],
                "temp": day['temperature'],
                "needs_water": needs,
                "amount": round(amount, 1)
            })
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    global dnn_classifier
    if dnn_classifier is None:
        load_artifacts()
    try:
        data = request.json
        crop = data.get("crop_type")
        soil = data.get("soil_type")
        
        # --- NEW LOGIC: Check for Condition vs Number ---
        if "soil_condition" in data and data["soil_condition"]:
            # User selected "Dry", "Moist", "Wet"
            condition = data.get("soil_condition")
            # Lookup range tuple (min, max)
            moist_range = SOIL_MOISTURE_RANGES.get(soil, {}).get(condition, (40, 50))
            # Use average for prediction
            moist = statistics.mean(moist_range)
        else:
            # User entered manual number (Backward compatibility)
            moist = float(data.get("soil_moisture", 45))

        rain = float(data.get("rainfall", 0))
        temp = float(data.get("temperature", 0))
        
        input_data = {
            "CROP_TYPE": [crop], "SOIL_TYPE": [soil],
            "REGION": [data.get("region")],
            "TEMPERATURE": [temp],
            "HUMIDITY": [float(data.get("humidity", 0))],
            "RAINFALL": [rain],
            "WIND_SPEED": [float(data.get("wind_speed", 0))],
            "WEATHER_CONDITION": [data.get("weather_condition")],
            "SOIL_MOISTURE": [moist]
        }
        
        prob, water_amount = get_prediction(input_data)
        needs_water = prob > 0.5
        advice = "Conditions are optimal."
        threshold = CROP_THRESHOLDS.get(crop, 50)
        
        # 1. Critical Dryness
        if moist < threshold:
            needs_water = True
            if water_amount < 1.0:
                water_amount = 3.5
            advice = f"Moisture (~{int(moist)}%) is below {crop} limit ({threshold}%). Irrigation Required."
        # 2. Heavy Rain
        elif rain > 15:
            needs_water = False
            water_amount = 0.0
            advice = f"Rainfall detected ({rain}mm). Irrigation Skipped."
        # 3. Moisture Sufficient
        elif moist > threshold:
            needs_water = False
            water_amount = 0.0
            advice = f"Soil Moisture (~{int(moist)}%) is sufficient for {crop}."
            
        temp_impact = min(100, max(0, (temp - 15) * 4))
        moist_impact = min(100, max(0, (100 - moist)))
        rain_impact = min(100, max(0, rain * 5))
        
        return jsonify({
            "needs_water": bool(needs_water),
            "confidence": round(prob, 4),
            "water_amount": round(water_amount, 2),
            "advice": advice,
            "crop_fact": CROP_FACTS.get(crop, ""),
            "impacts": {
                "temp": int(temp_impact),
                "moisture": int(moist_impact),
                "rain": int(rain_impact)
            }
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/translate_texts", methods=["POST"])
def translate_texts():
    data = request.json
    lang = data.get("lang")
    texts = data.get("texts", {})
    translated = {key: tr(value, lang) for key, value in texts.items()}
    return jsonify(translated)

@app.route("/planner")
def planner_page():
    return render_template("scheduler.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"reply": "", "error": "Empty message"}), 400
    system_prompt = (
        "You are AgriAssist, a friendly agriculture advisor for Indian farmers. "
        "Use simple language, short sentences, and bullet points. "
        "Give practical farming steps. "
        "If needed, ask one clarifying question. "
        "Do NOT give dangerous chemical pesticide advice."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply, "error": None})
    except Exception as e:
        print("Groq chatbot error:", str(e))
        return jsonify({"reply": "", "error": str(e)}), 500

if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)