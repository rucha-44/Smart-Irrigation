from flask import Blueprint, request, jsonify, session
import json
import os
from datetime import datetime

# 1. Define the Blueprint
scheduler_bp = Blueprint('scheduler', __name__)

SCHEDULE_FILE = 'schedules.json'

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

# --- LOGIC FUNCTIONS (Exported for app.py) ---
def get_user_tasks(user_email):
    """
    Returns a list of tasks specifically for the logged-in user.
    """
    all_tasks = load_tasks()
    # ⚠️ FIXED: Changed "user" to "user_email" to match app.py
    user_tasks = [t for t in all_tasks if t.get("user_email") == user_email]
    
    # Sort by date (Oldest first)
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
    
    # Generate ID
    task_id = int(datetime.now().timestamp() * 1000)
    
    new_task = {
        "id": task_id,
        "user_email": user_email, # ⚠️ FIXED: Standardized key
        "date": data.get('date'),
        "time": data.get('time', '06:00'),
        "crop": data.get('crop'),
        "amount": data.get('amount'),
        "status": "Pending",
        "warning": data.get('warning', '')
    }
    
    # Check for duplicates
    for t in tasks:
        # ⚠️ FIXED: Check against "user_email"
        if (t.get("user_email") == user_email and 
            t.get("date") == new_task["date"] and 
            t.get("crop") == new_task["crop"]):
            return jsonify({"status": "duplicate", "message": "Task already exists"})

    tasks.append(new_task)
    save_tasks(tasks)
    return jsonify({"status": "success", "message": "Task scheduled"})

@scheduler_bp.route('/delete_task', methods=['POST'])
def api_delete_task():
    if 'user' not in session: return jsonify({"status": "error"})
    
    data = request.json
    # Handle both string/int ID formats safely
    try:
        task_id = int(data.get('id') or data.get('task_id'))
    except:
        return jsonify({"status": "error", "message": "Invalid ID"})

    user_email = session['user']
    tasks = load_tasks()
    
    # Keep tasks that DO NOT match the ID
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