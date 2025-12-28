import json
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, session

# 1. Define the Blueprint (This allows app.py to register it)
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
    # Filter tasks by email
    user_tasks = [t for t in all_tasks if t.get("user") == user_email]
    
    # Sort by date (Oldest first)
    try:
        user_tasks.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=False)
    except:
        pass # Handle potential date format errors gracefully
        
    return user_tasks

# --- API ROUTES (For JavaScript / Frontend) ---

@scheduler_bp.route('/get_tasks', methods=['GET'])
def api_get_tasks():
    """API Endpoint for scheduler.html to fetch tasks via AJAX"""
    if 'user' not in session:
        return jsonify([])
    tasks = get_user_tasks(session['user'])
    return jsonify(tasks)

@scheduler_bp.route('/add_task', methods=['POST'])
def api_add_task():
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"})
    
    data = request.json
    user_email = session['user']
    tasks = load_tasks()
    
    # Generate a unique ID (timestamp based)
    task_id = int(datetime.now().timestamp() * 1000)
    
    new_task = {
        "id": task_id,
        "user": user_email,
        "date": data['date'],
        "time": data.get('time', '06:00'),
        "crop": data['crop'],
        "amount": data['amount'],
        "status": "Pending",
        "warning": data.get('warning', '')
    }
    
    # Check for duplicates
    for t in tasks:
        if (t["user"] == user_email and 
            t["date"] == new_task["date"] and 
            t["crop"] == new_task["crop"] and 
            str(t["amount"]) == str(new_task["amount"])):
            return jsonify({"status": "duplicate", "message": "Task already exists"})

    tasks.append(new_task)
    save_tasks(tasks)
    return jsonify({"status": "success", "message": "Task scheduled"})

@scheduler_bp.route('/delete_task', methods=['POST'])
def api_delete_task():
    if 'user' not in session:
        return jsonify({"status": "error"})
    
    data = request.json
    task_id = str(data.get('id') or data.get('task_id')) # Handle both key names
    user_email = session['user']
    
    tasks = load_tasks()
    # Keep tasks that DO NOT match the ID/User combo
    updated_tasks = [t for t in tasks if not (str(t["id"]) == task_id and t["user"] == user_email)]
    
    if len(tasks) == len(updated_tasks):
        return jsonify({"status": "error", "message": "Task not found"})
        
    save_tasks(updated_tasks)
    return jsonify({"status": "success"})

@scheduler_bp.route('/complete_task', methods=['POST'])
def api_complete_task():
    if 'user' not in session:
        return jsonify({"status": "error"})
        
    data = request.json
    task_id = str(data.get('id'))
    user_email = session['user']
    
    tasks = load_tasks()
    found = False
    for t in tasks:
        if str(t["id"]) == task_id and t["user"] == user_email:
            t["status"] = "Completed"
            found = True
            break
            
    if found:
        save_tasks(tasks)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})