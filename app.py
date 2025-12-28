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
from scheduler_controller import scheduler_bp, get_user_tasks
from geopy.geocoders import Nominatim

# Initialize Geocoder (Add this after creating your 'app' variable)
geolocator = Nominatim(user_agent="smart_irrigation_app")

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
# DATA: Soil Moisture Ranges
# -------------------------------------------
SOIL_MOISTURE_RANGES = {
    'Sandy': {'Dry': (0, 10), 'Moist': (11, 20), 'Wet': (21, 100)},
    'Loamy': {'Dry': (0, 15), 'Moist': (16, 30), 'Wet': (31, 100)},
    'Clayey': {'Dry': (0, 25), 'Moist': (26, 45), 'Wet': (46, 100)},
    'Silty': {'Dry': (0, 15), 'Moist': (16, 35), 'Wet': (36, 100)},
    'Peaty': {'Dry': (0, 30), 'Moist': (31, 60), 'Wet': (61, 100)},
    'Chalky': {'Dry': (0, 10), 'Moist': (11, 25), 'Wet': (26, 100)}
}

# -------------------------------------------
# DATA: Unified Crop Knowledge Base
# -------------------------------------------
# --- UNIFIED CROP KNOWLEDGE BASE ---
CROP_DATA = {
    'Cotton': {
        'days': [30, 50, 60, 40], 'kc': [0.35, 1.20, 0.60],
        'max_days': 180,
        'ideal_regions': ['West', 'Central', 'North', 'South'], 
        'ideal_soil': ['Clayey', 'Loamy'], # Black Cotton Soil (Regur) is Clayey
        'ideal_condition': ['Dry', 'Moist'], # Hates wet feet
        'stages': [
            {'range': (0, 20), 'name': 'Germination', 'water_msg': 'Keep soil moist for sprouting.'},
            {'range': (21, 60), 'name': 'Vegetative Phase', 'water_msg': 'Moderate watering for canopy.'},
            {'range': (61, 130), 'name': 'Flowering & Boll Formation', 'water_msg': 'CRITICAL! Avoid stress.'},
            {'range': (131, 180), 'name': 'Maturation (Boll Bursting)', 'water_msg': 'STOP IRRIGATION! Dry phase needed.'}
        ]
    },
    'Coffee': {
        'days': [30, 90, 150, 95], 'kc': [0.90, 1.10, 0.95],
        'max_days': 365,
        'ideal_regions': ['South'],
        'ideal_soil': ['Loamy'], # Needs rich, well-drained soil
        'ideal_condition': ['Moist'], # Cannot tolerate waterlogging OR extreme drought
        'stages': [
            {'range': (0, 60), 'name': 'Vegetative', 'water_msg': 'Regular maintenance.'},
            {'range': (61, 150), 'name': 'Winter Stress', 'water_msg': 'STOP WATER! Stress induces flowering.'},
            {'range': (151, 165), 'name': 'Blossom Showers', 'water_msg': 'HEAVY IRRIGATION needed.'},
            {'range': (166, 365), 'name': 'Berry Development', 'water_msg': 'Keep moist for fruit.'}
        ]
    },
    'Sugarcane': {
        'days': [35, 60, 180, 90], 'kc': [0.40, 1.25, 0.75],
        'max_days': 365,
        'ideal_regions': ['North', 'West', 'South'],
        'ideal_soil': ['Loamy', 'Clayey'], # Needs water retention
        'ideal_condition': ['Moist', 'Wet'], # Thirsty crop
        'stages': [
            {'range': (0, 35), 'name': 'Germination', 'water_msg': 'Light frequent watering.'},
            {'range': (36, 120), 'name': 'Formative', 'water_msg': 'Increase water gradually.'},
            {'range': (121, 270), 'name': 'Grand Growth', 'water_msg': 'PEAK DEMAND! Maximize irrigation.'},
            {'range': (271, 365), 'name': 'Maturation', 'water_msg': 'Reduce water for sugar.'}
        ]
    },
    'Rice': {
        'days': [30, 30, 40, 20], 'kc': [1.05, 1.20, 0.90],
        'max_days': 120, 
        'ideal_regions': ['East', 'South', 'Central'],
        'ideal_soil': ['Clayey', 'Silty', 'Loamy'], # Needs to hold water
        'ideal_condition': ['Wet'], # Semi-aquatic
        'stages': [{'range': (0, 120), 'name': 'General Growth', 'water_msg': 'Maintain standing water.'}]
    },
    'Wheat': {
        'days': [20, 35, 40, 25], 'kc': [0.30, 1.15, 0.25],
        'max_days': 120, 'ideal_regions': ['North', 'Central'],
        'ideal_soil': ['Loamy', 'Clayey'],
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 120), 'name': 'General Growth', 'water_msg': 'Ensure moisture at crown root.'}]
    },
    'Maize': {
        'days': [25, 40, 50, 35], 'kc': [0.30, 1.20, 0.60],
        'max_days': 150, 'ideal_regions': ['North', 'South'],
        'ideal_soil': ['Loamy', 'Sandy'], # Needs drainage
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 150), 'name': 'General Growth', 'water_msg': 'Sensitive to stress.'}]
    },
    'Tomato': {
        'days': [15, 25, 30, 20], 'kc': [0.60, 1.15, 0.80],
        'max_days': 90, 'ideal_regions': ['All'],
        'ideal_soil': ['Loamy', 'Sandy'],
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 90), 'name': 'General Growth', 'water_msg': 'Consistent moisture prevents cracking.'}]
    },
    'Potato': {
        'days': [25, 30, 45, 20], 'kc': [0.50, 1.15, 0.75],
        'max_days': 120, 'ideal_regions': ['North', 'East'],
        'ideal_soil': ['Sandy', 'Loamy'], # Needs loose soil for tubers
        'ideal_condition': ['Moist'], # Rots in wet soil
        'stages': [{'range': (0, 120), 'name': 'General Growth', 'water_msg': 'Avoid waterlogging.'}]
    }
}

# Backward Compatibility
CROP_PARAMS = CROP_DATA
# Helper for limits
CROP_MAX_DAYS = {crop: data['max_days'] for crop, data in CROP_DATA.items()}

# -------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------

def get_region_by_coords(lat, lon):
    """Mathematically determines region based on Lat/Lon"""
    if lat > 24.0: return 'North'
    if lat < 20.0 and lon > 77.0: return 'South'
    if lat < 16.0: return 'South'
    if lon > 82.0: return 'East'
    if lon < 77.0: return 'West'
    return 'Central'

def calculate_crop_coefficient(crop_name, day):
    """Calculates dynamic Kc based on exact day and crop type"""
    params = CROP_PARAMS.get(crop_name, {'days': [30, 40, 50, 30], 'kc': [0.4, 1.0, 0.5]})
    l_ini, l_dev, l_mid, l_late = params['days']
    kc_ini, kc_mid, kc_end = params['kc']
    
    total_days = sum(params['days'])
    day = min(day, total_days)
    
    if day <= l_ini:
        return "Initial", kc_ini
    elif day <= (l_ini + l_dev):
        progress = (day - l_ini) / l_dev
        kc_current = kc_ini + progress * (kc_mid - kc_ini)
        return "Development", kc_current
    elif day <= (l_ini + l_dev + l_mid):
        return "Mid-Season", kc_mid
    else:
        days_into_late = day - (l_ini + l_dev + l_mid)
        progress = days_into_late / l_late
        kc_current = kc_mid - progress * (kc_mid - kc_end)
        return "Late/Harvest", max(kc_current, kc_end)

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f: json.dump([], f)
    with open(USERS_FILE, "r") as f: return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f: json.dump(users, f, indent=4)

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

# Load on startup
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

# -------------------------------------------
# ROUTES
# -------------------------------------------

@app.route("/planner")
def planner():
    if "user" not in session:
        return redirect(url_for("login"))
    
    # Fetch tasks for the logged-in user
    tasks = get_user_tasks(session["user"])
    
    return render_template("scheduler.html", tasks=tasks)


@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    
    options = {
        "CROP_TYPE": list(CROP_PARAMS.keys()),
        "SOIL_TYPE": ['Sandy', 'Loamy', 'Clayey', 'Silty', 'Peaty', 'Chalky'],
        "REGION": ['North', 'East', 'West', 'South', 'Central'],
        "WEATHER_CONDITION": ['Sunny', 'Rainy', 'Cloudy', 'Windy']
    }
    
    # ⚠️ THIS IS THE KEY CHANGE YOU ASKED FOR
    return render_template("index.html", options=options, crop_data=CROP_DATA)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        users = load_users()
        for u in users:
            if u["email"] == email and check_password_hash(u["password"], password):
                session.permanent = True
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
            "land_area": "", "crops": ""
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
                return render_template("login.html", error="✅ Password reset link sent!")
        return render_template("forgot.html", error="❌ No account found.")
    return render_template("forgot.html")

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session: return redirect(url_for("login"))
    users = load_users()
    current_user = next((u for u in users if u["email"] == session["user"]), None)
    
    if request.method == "POST":
        current_user["username"] = request.form.get("username", "").strip()
        current_user["land_area"] = request.form.get("land_area", "").strip()
        current_user["crops"] = [c.strip() for c in request.form.get("crops", "").split("\n") if c.strip()]
        save_users(users)
        return redirect(url_for("home"))
    return render_template("profile.html", user=current_user)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.json.get("city")
    data = weather_service.get_weather_data(city)
    try:
        location = geolocator.geocode(f"{city}, India")
        if location:
            detected_region = get_region_by_coords(location.latitude, location.longitude)
            data['detected_region'] = detected_region
            print(f"📍 City: {city} -> Region: {detected_region}")
    except Exception as e:
        print(f"Region detection error: {e}")
    return jsonify(data)

@app.route("/predict_forecast", methods=["POST"])
def predict_forecast():
    try:
        data = request.json
        city = data.get('city')
        base = data.get('base_inputs')
        forecasts = weather_service.get_forecast_data(city)
        if "error" in forecasts: return jsonify(forecasts)
        
        preds = []
        crop = base['crop_type']
        threshold = 50 # Default
        if 'soil_type' in base and 'soil_condition' in base:
             # Just a simple check, actual logic handled in loop
             pass

        # Determine moisture
        base_moist = 45
        if 'soil_condition' in base and base['soil_condition']:
             rng = SOIL_MOISTURE_RANGES.get(base.get('soil_type', 'Loamy'), {}).get(base.get('soil_condition'), (40, 50))
             base_moist = statistics.mean(rng)
        elif 'soil_moisture' in base:
             base_moist = float(base['soil_moisture'])

        for day in forecasts:
            curr_moist = base_moist - 5 
            day_in = {
                "CROP_TYPE": [crop], "SOIL_TYPE": [base['soil_type']],
                "REGION": [base['region']],
                "TEMPERATURE": [day['temperature']], "HUMIDITY": [day['humidity']],
                "RAINFALL": [day['rainfall']], "WIND_SPEED": [day['wind_speed']],
                "WEATHER_CONDITION": [day['weather_condition']], "SOIL_MOISTURE": [curr_moist]
            }
            prob, amount = get_prediction(day_in)
            needs = True
            if day['rainfall'] > 10 or curr_moist > threshold: needs = False; amount = 0.0
            
            preds.append({
                "day": day['day_name'], "date": day['date_short'],
                "condition": day['condition_desc'], "temp": day['temperature'],
                "needs_water": needs, "amount": round(amount, 1)
            })
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        crop = data.get("crop_type")
        soil = data.get("soil_type")
        
        # 1. Get Age
        raw_age = data.get("crop_age")
        if raw_age is None or raw_age == "":
            return jsonify({"error": "Please enter the Crop Age in days."})
        try:
            crop_age = int(raw_age)
        except:
            return jsonify({"error": "Crop Age must be a number."})

        # 2. Get Moisture
        if "soil_condition" in data and data["soil_condition"]:
            condition = data.get("soil_condition")
            moist_range = SOIL_MOISTURE_RANGES.get(soil, {}).get(condition, (40, 50))
            moist = statistics.mean(moist_range)
        else:
            moist = float(data.get("soil_moisture", 45))

        rain = float(data.get("rainfall", 0))
        temp = float(data.get("temperature", 0))
        
        input_data = {
            "CROP_TYPE": [crop], "SOIL_TYPE": [soil], "REGION": [data.get("region")],
            "TEMPERATURE": [temp], "HUMIDITY": [float(data.get("humidity", 0))],
            "RAINFALL": [rain], "WIND_SPEED": [float(data.get("wind_speed", 0))],
            "WEATHER_CONDITION": [data.get("weather_condition")], "SOIL_MOISTURE": [moist]
        }
        
        prob, raw_water_amount = get_prediction(input_data)
        
        # 3. Apply Age Multiplier
        stage_name, specific_kc = calculate_crop_coefficient(crop, crop_age)
        adjusted_water = raw_water_amount * specific_kc
        
        needs_water = prob > 0.5
        advice = "Conditions are optimal."
        threshold = 50 # Default threshold logic can be refined if needed
        stage_info = f" (Day {crop_age}: {stage_name} Phase)"

        if moist < threshold:
            needs_water = True
            if adjusted_water < 1.0: adjusted_water = 3.5 * specific_kc
            advice = f"Moisture (~{int(moist)}%) is below limit. Water Required."
        elif rain > 15:
            needs_water = False
            adjusted_water = 0.0
            advice = f"Rainfall detected ({rain}mm). Irrigation Skipped."
        elif moist > threshold:
            needs_water = False
            adjusted_water = 0.0
            advice = f"Soil Moisture (~{int(moist)}%) is sufficient."
            
        return jsonify({
            "needs_water": bool(needs_water),
            "confidence": round(prob, 4),
            "water_amount": round(adjusted_water, 2),
            "advice": advice + stage_info,
            "crop_fact": "", # You can fetch fact if needed
            "impacts": { "temp": 50, "moisture": 50, "rain": 50 } # Simplified for brevity
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

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_message: return jsonify({"reply": "", "error": "Empty message"}), 400
    
    system_prompt = "You are AgriAssist. Short, simple advice for farmers."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, max_tokens=300)
        return jsonify({"reply": completion.choices[0].message.content.strip(), "error": None})
    except Exception as e:
        return jsonify({"reply": "", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)