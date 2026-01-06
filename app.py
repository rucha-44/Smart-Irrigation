from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import json
import statistics
import random
from datetime import datetime 
from werkzeug.security import generate_password_hash, check_password_hash
from services.weather_service import WeatherService
from translator import tr
from groq import Groq
from flask_cors import CORS
from scheduler_controller import scheduler_bp, get_user_tasks
from geopy.geocoders import Nominatim
from services.notification_service import NotificationService
notifier = NotificationService()

# Initialize Geocoder
geolocator = Nominatim(user_agent="smart_irrigation_app")

app = Flask(__name__)
weather_service = WeatherService()

# --- Scheduler ---
app.register_blueprint(scheduler_bp, url_prefix='/scheduler')

# --- Groq Client ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app.secret_key = "supersecretkey123"
USERS_FILE = "users.json"
CROPS_FILE = "my_crops.json"  # NEW: Database for user crops

# Enable CORS
CORS(app)

# -------------------------------------------
# DATA: Soil & Crops
# -------------------------------------------

REGION_MAPPING = {
    # Non-Directional (New) -> Directional (Model-Friendly)
    'North East': 'East',   # Assam/Meghalaya share wet climate with East
    'North West': 'North',  # Punjab/Haryana share winter trends with North
    'South East': 'South',  # Coastal AP/TN share tropical climate with South
    'South West': 'South',  # Kerala/Karnataka/Goa are classic South
    
    # Standard Regions (Pass-through)
    'North': 'North',
    'South': 'South',
    'East': 'East',
    'West': 'West',
    'Central': 'Central'
}

SOIL_MOISTURE_RANGES = {
    'Sandy': {'Dry': (0, 10), 'Moist': (11, 20), 'Wet': (21, 100)},
    'Loamy': {'Dry': (0, 15), 'Moist': (16, 30), 'Wet': (31, 100)},
    'Clayey': {'Dry': (0, 25), 'Moist': (26, 45), 'Wet': (46, 100)},
    'Silty': {'Dry': (0, 15), 'Moist': (16, 35), 'Wet': (36, 100)},
    'Peaty': {'Dry': (0, 30), 'Moist': (31, 60), 'Wet': (61, 100)},
    'Chalky': {'Dry': (0, 10), 'Moist': (11, 25), 'Wet': (26, 100)}
}

CROP_DATA = {
    'Cotton': {
        'days': [30, 50, 60, 40], 'kc': [0.35, 1.20, 0.60],
        'max_days': 180,
        # 🟢 NEW: Suitability Logic
        'suitability': {
            'ideal': ['West', 'Central', 'South', 'North West', 'South West'],
            'marginal': ['North', 'South East'],
            'unsuitable': ['North East', 'East']
        },
        'ideal_regions': ['West', 'Central', 'North', 'South', 'North West', 'South West'],
        'ideal_soil': ['Clayey', 'Loamy'],
        'ideal_condition': ['Dry', 'Moist'],
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
        # 🟢 NEW: Suitability Logic
        'suitability': {
            'ideal': ['South West', 'South', 'South East', 'North East'],
            'marginal': [],
            'unsuitable': ['North', 'West', 'Central', 'East', 'North West']
        },
        'ideal_regions': ['South', 'South West', 'South East', 'North East'],
        'ideal_soil': ['Loamy'],
        'ideal_condition': ['Moist'],
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
        'suitability': {
            'ideal': ['North', 'West', 'South', 'Central', 'South West', 'South East'],
            'marginal': ['North West'],
            'unsuitable': ['North East', 'East']
        },
        'ideal_regions': ['North', 'West', 'South', 'Central', 'North West', 'South East', 'SouthWest'],
        'ideal_soil': ['Loamy', 'Clayey'],
        'ideal_condition': ['Moist', 'Wet'],
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
        'suitability': {
            'ideal': ['South', 'East', 'North East', 'South East', 'South West', 'Central'],
            'marginal': ['North', 'West'],
            'unsuitable': ['North West', 'Ladakh'] 
        },
        'ideal_regions': ['East', 'South', 'Central', 'North East', 'South East', 'South West'],
        'ideal_soil': ['Clayey', 'Silty', 'Loamy'],
        'ideal_condition': ['Wet'],
        'stages': [{'range': (0, 120), 'name': 'General Growth', 'water_msg': 'Maintain standing water.'}]
    },
    'Wheat': {
        'days': [20, 35, 40, 25], 'kc': [0.30, 1.15, 0.25],
        'max_days': 120, 
        'suitability': {
            'ideal': ['North', 'North West', 'Central', 'East'],
            'marginal': ['West'],
            'unsuitable': ['South', 'South West', 'South East', 'North East']
        },
        'ideal_regions': ['North', 'North West', 'Central', 'East'],
        'ideal_soil': ['Loamy', 'Clayey'],
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 120), 'name': 'General Growth', 'water_msg': 'Ensure moisture at crown root.'}]
    },
    'Maize': {
        'days': [25, 40, 50, 35], 'kc': [0.30, 1.20, 0.60],
        'max_days': 150, 
        'suitability': {
            'ideal': ['North', 'South', 'East', 'West', 'Central'],
            'marginal': ['North East', 'South West'],
            'unsuitable': []
        },
        'ideal_regions': ['North', 'South', 'Central', 'West', 'North West', 'East'],
        'ideal_soil': ['Loamy', 'Sandy'],
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 150), 'name': 'General Growth', 'water_msg': 'Sensitive to stress.'}]
    },
    'Tomato': {
        'days': [15, 25, 30, 20], 'kc': [0.60, 1.15, 0.80],
        'max_days': 90, 
        'suitability': {
            'ideal': ['North', 'South', 'East', 'West', 'Central', 'North West', 'South East'],
            'marginal': ['North East', 'South West'],
            'unsuitable': []
        },
        'ideal_regions': ['North', 'South', 'East', 'West', 'Central', 'South West', 'South East'],
        'ideal_soil': ['Loamy', 'Sandy'],
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 90), 'name': 'General Growth', 'water_msg': 'Consistent moisture prevents cracking.'}]
    },
    'Potato': {
        'days': [25, 30, 45, 20], 'kc': [0.50, 1.15, 0.75],
        'max_days': 120, 
        'suitability': {
            'ideal': ['North', 'East', 'North East', 'Central', 'North West'],
            'marginal': ['West'],
            'unsuitable': ['South', 'South East', 'South West']
        },
        'ideal_regions': ['North', 'East', 'North East', 'Central', 'North West'],
        'ideal_soil': ['Sandy', 'Loamy'],
        'ideal_condition': ['Moist'],
        'stages': [{'range': (0, 120), 'name': 'General Growth', 'water_msg': 'Avoid waterlogging.'}]
    }
}

CROP_PARAMS = CROP_DATA
CROP_MAX_DAYS = {crop: data['max_days'] for crop, data in CROP_DATA.items()}

# -------------------------------------------
# HELPER FUNCTIONS (File IO & Math)
# -------------------------------------------

def get_region_by_coords(lat, lon):
    """
    Splits India into 8 zones (4 Cardinal + 4 Intercardinal) + Central.
    """
    # 1. North East (The Seven Sisters + Sikkim)
    if lon > 88.0:
        return 'North East'

    # 2. Define Latitude Boundaries
    is_north = lat > 24.0
    is_south = lat < 20.0
    
    # 3. Define Longitude Boundaries
    is_west = lon < 77.0
    is_east = lon > 82.0

    # --- CORNER CHECKS ---
    
    if is_north and is_west:
        return 'North West'  # Punjab, Rajasthan borders
    
    if is_north: 
        return 'North'       # UP, HP, Uttarakhand
        
    if is_south and is_west:
        return 'South West'  # Kerala, Karnataka Coast, Goa
        
    if is_south and is_east:
        return 'South East'  # Tamil Nadu, Coastal AP
        
    if is_south:
        return 'South'       # Interior Karnataka/Telangana

    # --- SIDE CHECKS ---
    
    if is_west:
        return 'West'        # Gujarat, Maharashtra
        
    if is_east:
        return 'East'        # Odisha, Bengal, Jharkhand

    # --- DEFAULT ---
    return 'Central'         # MP, Chattisgarh

def calculate_crop_coefficient(crop_name, day):
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

# --- NEW: Crop Database Helpers ---
def load_my_crops():
    if not os.path.exists(CROPS_FILE): return []
    try:
        with open(CROPS_FILE, 'r') as f: return json.load(f)
    except: return []

def save_my_crops(crops):
    with open(CROPS_FILE, 'w') as f: json.dump(crops, f, indent=4)

# --- AI Model Loading ---
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

# ---------------------------------------------------------
# CORE BUSINESS LOGIC (REUSABLE)
# ---------------------------------------------------------
def calculate_irrigation_needs(crop, soil, region, age, weather_data):
    """
    Pure Python function: Takes inputs, returns water amount & advice.
    Used by both the Calculator and the Dashboard.
    """
    model_region = REGION_MAPPING.get(region, 'Central')
    # 1. Prepare Input for AI Model (Assume avg moisture if unknown)
    moist = 45.0 
    safe_rain = weather_data.get("rainfall", 0.0)
    safe_temp = weather_data.get("temperature", 25.0)

    input_data = {
        "CROP_TYPE": [crop], 
        "SOIL_TYPE": [soil], 
        "REGION": [model_region],
       "TEMPERATURE": [weather_data.get("temperature", 25.0)],
        "HUMIDITY": [weather_data.get("humidity", 50.0)],
        "RAINFALL": [weather_data.get("rainfall", 0.0)],
        "WIND_SPEED": [weather_data.get("wind_speed", 5.0)],
        "WEATHER_CONDITION": [weather_data.get("weather_condition", "Sunny")],
        "SOIL_MOISTURE": [moist]
    }
    
    # 2. Get AI Prediction
    prob, raw_water = get_prediction(input_data)
    
    # 3. Apply Crop Age Logic (Kc Factor)
    stage_name, kc = calculate_crop_coefficient(crop, age)
    final_water = raw_water * kc
    
    # 4. Decision Rules
    advice = "Stable"
    needs_water = False
    
    if prob > 0.5:
        needs_water = True
        advice = "Irrigation Needed"
        # Ensure minimum effective water
        if final_water < 1.0: final_water = 3.5 * kc
            
   # 🛡️ RAIN OVERRIDE (Fixed Line)
    # Replaced 'weather_data["rainfall"]' with 'safe_rain'
    if safe_rain > 10:
        needs_water = False
        final_water = 0.0
        advice = "Rain Detected (Skip)"

    return {
        "water_amount": round(final_water, 1),
        "advice": advice,
        "needs_water": needs_water,
        "stage": stage_name
    }

# -------------------------------------------
# ROUTES
# -------------------------------------------

@app.route("/")
def home():
    if "user" not in session: return redirect(url_for("login"))
    options = {
        "CROP_TYPE": list(CROP_PARAMS.keys()),
        "SOIL_TYPE": ['Sandy', 'Loamy', 'Clayey', 'Silty', 'Peaty', 'Chalky'],
        "REGION": ['North', 'East', 'West', 'South', 'Central'],
        "WEATHER_CONDITION": ['Sunny', 'Rainy', 'Cloudy', 'Windy']
    }
    return render_template("index.html", options=options, crop_data=CROP_DATA)

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    
    users = load_users()
    current_user = next((u for u in users if u["email"] == session["user"]), None)
    if not current_user:
        session.clear()
        return redirect(url_for("login"))

    all_crops = load_my_crops()
    my_crops = [c for c in all_crops if c["user_email"] == session["user"]]
    
    processed_crops = []
    
    for crop in my_crops:
        try:
            sowing = datetime.strptime(crop["sowing_date"], "%Y-%m-%d")
            age = max(0, (datetime.now() - sowing).days)
        except: age = 0
            
        crop_city = crop.get("city", "")
        weather_data = weather_service.get_weather_data(crop_city)
        
        # Safe fallback for region
        if "detected_region" not in weather_data:
            weather_data["detected_region"] = crop.get("region", "Central")

        # 🛡️ FIX: Safe Name Extraction (Prevents KeyError)
        # Tries "crop_name" first, falls back to "name"
        safe_name = crop.get("crop_name", crop.get("name", "Unknown Crop"))

        result = calculate_irrigation_needs(
            crop=safe_name,  # <--- Use safe_name here
            soil=crop["soil_type"],
            region=weather_data["detected_region"],
            age=age,
            weather_data=weather_data
        )
        
        status_color = "success"
        if result["needs_water"]: status_color = "danger"
        elif "Rain" in result["advice"]: status_color = "primary"

        processed_crops.append({
            "id": crop.get("id", random.randint(1000,9999)),
            "name": safe_name,       # <--- Use safe_name here
            "crop_name": safe_name,  # <--- Use safe_name here
            "city": crop_city,
            "soil_type": crop["soil_type"],
            "age": age,
            "stage": result["stage"],
            "water_amount": result["water_amount"],
            "advice": result["advice"],
            "needs_water": result["needs_water"],
            "color": status_color,
            "viability": crop.get("viability", "Ideal") 
        })

    return render_template("dashboard.html", 
                           user=current_user, 
                           crops=processed_crops, 
                           crop_data=CROP_DATA)

@app.route('/add_crop', methods=['POST'])
def add_crop():
    if 'user' not in session: return redirect(url_for('login'))
    
    user_email = session['user']
    
    # 1. Get Form Data
    crop_name = request.form.get('crop_name')
    city = request.form.get('city')
    region = request.form.get('region') # <--- Get the region from hidden input
    soil_type = request.form.get('soil_type')
    sowing_date = request.form.get('sowing_date')
    
    # 2. Calculate Age
    sowing_dt = datetime.strptime(sowing_date, '%Y-%m-%d')
    age = (datetime.now() - sowing_dt).days
    
    # 3. 🚩 CALCULATE VIABILITY (Server-Side)
    viability_status = "Ideal" # Default
    advice_text = "Conditions look good."
    color = "success"

    if crop_name in CROP_DATA:
        suitability = CROP_DATA[crop_name].get("suitability", {})
        
        # Normalize Region (Handle "Ladakh" edge case)
        check_region = region if region else "Central"
        
        # Specific check for Ladakh if API returns it directly
        if "ladakh" in city.lower():
             check_region = "North West" # Map Ladakh to North West internally if needed

        if check_region in suitability.get("unsuitable", []):
            viability_status = "Unsuitable"
            advice_text = f"Warning: {crop_name} is not suitable for {check_region}."
            color = "danger"
        elif check_region in suitability.get("marginal", []):
            viability_status = "Marginal"
            advice_text = f"Caution: {check_region} is marginal for {crop_name}."
            color = "warning"

    # 4. Create Object
    new_crop = {
        "user_email": user_email,
        "name": crop_name,
        "city": city,
        "region": region, # Save the region
        "soil_type": soil_type,
        "sowing_date": sowing_date,
        "age": age,
        "viability": viability_status, # <--- Save the calculated flag
        "advice": advice_text,
        "color": color
    }
    
    # 5. Save to List/File
    # (Assuming you use a list named 'user_crops' or load/save from json)
    users_crops = load_my_crops() # Use your existing loader
    users_crops.append(new_crop)
    save_my_crops(users_crops)      # Use your existing saver
    
    return redirect(url_for('dashboard'))

@app.route("/predict_saved", methods=["POST"])
def predict_saved():
    """
    The 'One-Click' Prediction Logic for the Dashboard
    """
    data = request.json
    crop_id = int(data.get("crop_id"))
    current_condition = data.get("soil_condition")
    
    # 1. Find the Saved Crop Data
    all_crops = load_my_crops()
    target_crop = next((c for c in all_crops if c["id"] == crop_id), None)
    
    if not target_crop:
        return jsonify({"error": "Crop not found"})

    # 2. Auto-Calculate Age
    sowing = datetime.strptime(target_crop["sowing_date"], "%Y-%m-%d")
    age_days = (datetime.now() - sowing).days
    if age_days < 0: age_days = 0

    # 3. Auto-Fetch Weather (Using saved City)
    weather = weather_service.get_weather_data(target_crop["city"])
    # Detect region if missing
    if "detected_region" not in weather:
        try:
            loc = geolocator.geocode(f"{target_crop['city']}, India")
            if loc: weather["detected_region"] = get_region_by_coords(loc.latitude, loc.longitude)
            else: weather["detected_region"] = "Central"
        except: weather["detected_region"] = "Central"
    
    # 4. Run the Logic
    result = calculate_irrigation_needs(
        crop=target_crop["crop_name"],
        soil=target_crop["soil_type"],
        region=weather["detected_region"],
        age=age_days,
        weather_data=weather
    )
    
    # Override logic based on manual Soil Condition
    advice = result["advice"]
    needs_water = result["needs_water"]
    
    if current_condition == "Wet":
        needs_water = False
        result["water_amount"] = 0.0
        advice = "Soil is Wet. No irrigation needed."
    elif current_condition == "Dry" and not needs_water:
        needs_water = True
        result["water_amount"] = max(result["water_amount"], 2.0)
        advice = "Soil is Dry. Water required."

    return jsonify({
        "needs_water": needs_water,
        "water_amount": result["water_amount"],
        "advice": advice,
        "age": age_days,
        "weather_summary": f"{weather['temperature']}°C in {target_crop['city']}"
    })

# --- EXISTING ROUTES (Auth, Planner, etc.) ---

@app.route("/planner")
def planner():
    if "user" not in session: return redirect(url_for("login"))
    tasks = get_user_tasks(session["user"])
    return render_template("scheduler.html", tasks=tasks)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        users = load_users()
        user_found = None
        
        # 1. Check Credentials
        for u in users:
            if u["email"] == email and check_password_hash(u["password"], password):
                user_found = u
                break
        
        if user_found:
            # 2. Check if they have a phone number for OTP
            contact = user_found.get("contact")
            if contact:
                # Generate 6-digit OTP
                otp_code = str(random.randint(100000, 999999))
                
                # Store in Session (Temporary)
                session["pending_user"] = email
                session["otp"] = otp_code
                
                # Send WhatsApp
                success = notifier.send_otp(contact, otp_code)
                
                if success:
                    return redirect(url_for("verify_otp"))
                else:
                    error = "Failed to send OTP. Check internet/Twilio settings."
            else:
                # Fallback: If no phone number, just log in (or force them to add one)
                session["user"] = email
                session.permanent = False
                return redirect(url_for("dashboard"))

        else:
            error = "Invalid email or password!"

    return render_template("login.html", error=error)

@app.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    if "pending_user" not in session or "otp" not in session:
        return redirect(url_for("login"))
    
    error = None
    
    if request.method == "POST":
        user_otp = request.form.get("otp")
        
        # 1. Verify Code
        if user_otp == session["otp"]:
            # Success! Log them in for real.
            session["user"] = session["pending_user"]
            session.permanent = False
            
            # Clear temporary session data
            session.pop("pending_user", None)
            session.pop("otp", None)
            
            return redirect(url_for("dashboard"))
        else:
            error = "❌ Invalid OTP. Please try again."
            
    return render_template("verify_otp.html", error=error)

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
            "land_area": "", "crops": []
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
        # Only save Name and Contact
        current_user["full_name"] = request.form.get("full_name", "").strip()
        current_user["contact"] = request.form.get("contact", "").strip()
        
        save_users(users)
        return redirect(url_for("dashboard"))
        
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
        
        # Determine moisture base
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
            # Use Helper Logic logic here implicitly via get_prediction
            prob, amount = get_prediction(day_in)
            needs = True
            if day['rainfall'] > 10: needs = False; amount = 0.0
            
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
    """
    Original Calculator Logic (Refactored to use Helper)
    """
    try:
        data = request.json
        crop = data.get("crop_type")
        soil = data.get("soil_type")
        region = data.get("region")
        
        # Validation
        raw_age = data.get("crop_age")
        if not raw_age: return jsonify({"error": "Please enter Crop Age"})
        crop_age = int(raw_age)

        # Weather Dict
        weather = {
            "temperature": float(data.get("temperature", 0)),
            "humidity": float(data.get("humidity", 0)),
            "rainfall": float(data.get("rainfall", 0)),
            "wind_speed": float(data.get("wind_speed", 0)),
            "weather_condition": data.get("weather_condition", "")
        }

        # Use the Helper Function
        result = calculate_irrigation_needs(crop, soil, region, crop_age, weather)
        
        # Handle specific condition override if provided (Calculator specific)
        if "soil_condition" in data and data["soil_condition"]:
            condition = data.get("soil_condition")
            if condition == "Wet":
                result["needs_water"] = False
                result["water_amount"] = 0.0
                result["advice"] = "Soil is Wet (Calculator Override)"

        return jsonify({
            "needs_water": result["needs_water"],
            "confidence": 0.9, # Placeholder
            "water_amount": result["water_amount"],
            "advice": f"{result['advice']} ({result['stage']} Phase)",
            "crop_fact": "",
            "impacts": { "temp": 50, "moisture": 50, "rain": 50 }
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