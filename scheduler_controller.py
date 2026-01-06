from flask import Blueprint, request, jsonify, session
import json
import os
from datetime import datetime
from services.notification_service import NotificationService

# 1. Define the Blueprint
scheduler_bp = Blueprint('scheduler', __name__)

notifier = NotificationService()

SCHEDULE_FILE = 'schedules.json'

# --- 🟢 DATA: Suitability Rules (Included to avoid Circular Import) ---
CROP_DATA = {
    'Cotton': { 'suitability': { 'ideal': ['West', 'Central', 'South', 'North West', 'South West'], 'marginal': ['North', 'South East'], 'unsuitable': ['North East', 'East'] } },
    'Coffee': { 'suitability': { 'ideal': ['South West', 'South', 'South East', 'North East'], 'marginal': [], 'unsuitable': ['North', 'West', 'Central', 'East', 'North West'] } },
    'Sugarcane': { 'suitability': { 'ideal': ['North', 'West', 'South', 'Central', 'South West', 'South East'], 'marginal': ['North West'], 'unsuitable': ['North East', 'East'] } },
    'Rice': { 'suitability': { 'ideal': ['South', 'East', 'North East', 'South East', 'South West', 'Central'], 'marginal': ['North', 'West'], 'unsuitable': ['North West', 'Ladakh'] } },
    'Wheat': { 'suitability': { 'ideal': ['North', 'North West', 'Central', 'East'], 'marginal': ['West'], 'unsuitable': ['South', 'South West', 'South East', 'North East'] } },
    'Maize': { 'suitability': { 'ideal': ['North', 'South', 'East', 'West', 'Central'], 'marginal': ['North East', 'South West'], 'unsuitable': [] } },
    'Tomato': { 'suitability': { 'ideal': ['North', 'South', 'East', 'West', 'Central', 'North West', 'South East'], 'marginal': ['North East', 'South West'], 'unsuitable': [] } },
    'Potato': { 'suitability': { 'ideal': ['North', 'East', 'North East', 'Central', 'North West'], 'marginal': ['West'], 'unsuitable': ['South', 'South East', 'South West'] } }
}

# --- FILE HELPERS ---
def load_tasks():
    if not os.path.exists(SCHEDULE_FILE):
        return []
    try:
        with open(SCHEDULE_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_tasks(tasks):
    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(tasks, f, indent=4)

def get_user_phone(user_email):
    """Finds the contact number for the logged-in user."""
    users_file_path = 'users.json'
    
    if not os.path.exists(users_file_path): 
        print("❌ Users file not found.")
        return None
        
    try:
        with open(users_file_path, 'r') as f:
            users = json.load(f)
            user = next((u for u in users if u["email"] == user_email), None)
            if user:
                return user.get("contact")
            return None
    except Exception as e:
        print(f"❌ Error reading users file: {e}")
        return None

# --- LOGIC FUNCTIONS (Exported for app.py) ---
def get_user_tasks(user_email):
    all_tasks = load_tasks()
    user_tasks = [t for t in all_tasks if t.get("user_email") == user_email]
    
    try:
        user_tasks.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=False)
    except:
        pass 
        
    return user_tasks

# --- API ROUTES ---

@scheduler_bp.route('/get_tasks', methods=['GET'])
def api_get_tasks():
    if 'user' not in session:
        return jsonify([])
    tasks = get_user_tasks(session['user'])
    return jsonify(tasks)

@scheduler_bp.route('/add_task', methods=['POST'])
def api_add_task():
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    data = request.json
    user_email = session['user']
    tasks = load_tasks()
    
    # 1. Get Inputs
    crop_name = data.get('crop')
    region = data.get('region', 'Central')
    
    # 2. 🚩 CALCULATE VIABILITY FLAG (Using local CROP_DATA)
    viability_status = "Ideal"
    
    if crop_name in CROP_DATA:
        suitability = CROP_DATA[crop_name].get("suitability", {})
        
        # Check Unsuitable
        if region in suitability.get("unsuitable", []):
            viability_status = "Unsuitable"
        # Check Marginal
        elif region in suitability.get("marginal", []):
            viability_status = "Marginal"

    # 3. Create Task
    task_id = int(datetime.now().timestamp() * 1000)
    
    new_task = {
        "id": task_id,
        "user_email": user_email,
        "date": data.get('date'),
        "time": data.get('time', '06:00'),
        "crop": crop_name,
        "amount": data.get('amount'),
        "status": "Pending",
        "region": region,
        "viability": viability_status, # <--- Saves the flag
        "warning": data.get('warning', '')
    }
    
    # Check for duplicates
    for t in tasks:
        if (t.get("user_email") == user_email and 
            t.get("date") == new_task["date"] and 
            t.get("crop") == new_task["crop"]):
            return jsonify({"status": "duplicate", "message": "Task already exists"})

    tasks.append(new_task)
    save_tasks(tasks)
    
    # 4. WhatsApp Notification
    try:
        user_phone = get_user_phone(user_email)
        if user_phone:
            notifier.send_irrigation_alert(
                to_number=user_phone,
                var_1=new_task['crop'],
                var_2=new_task['date']
            )
    except Exception as e:
        print(f"Notification Error: {e}")

    return jsonify({"status": "success", "message": "Task scheduled & WhatsApp sent"})

@scheduler_bp.route('/delete_task', methods=['POST'])
def api_delete_task():
    if 'user' not in session: return jsonify({"status": "error"})
    
    data = request.json
    try:
        task_id = int(data.get('id') or data.get('task_id'))
    except:
        return jsonify({"status": "error", "message": "Invalid ID"})

    user_email = session['user']
    tasks = load_tasks()
    
    updated_tasks = [t for t in tasks if not (t["id"] == task_id and t.get("user_email") == user_email)]
    
    save_tasks(updated_tasks)
    return jsonify({"status": "success"})

@scheduler_bp.route('/complete_task', methods=['POST'])
def api_complete_task():
    if 'user' not in session: return jsonify({"status": "error"})
        
    data = request.json
    try:
        task_id = int(data.get('id'))
    except:
        return jsonify({"status": "error"})

    user_email = session['user']
    tasks = load_tasks()
    
    found = False
    for t in tasks:
        if t["id"] == task_id and t.get("user_email") == user_email:
            t["status"] = "Completed"
            found = True
            break
            
    if found:
        save_tasks(tasks)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})