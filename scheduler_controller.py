import json
import os
from flask import Blueprint, request, jsonify
from datetime import datetime

scheduler_bp = Blueprint('scheduler', __name__)
DB_FILE = 'schedules.json'

# --- Helper Functions ---
def load_db():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Routes ---

@scheduler_bp.route('/get_tasks', methods=['GET'])
def get_tasks():
    tasks = load_db()
    # Sort by date (YYYY-MM-DD string sorting works correctly)
    tasks.sort(key=lambda x: x['date'])
    return jsonify(tasks)

@scheduler_bp.route('/add_task', methods=['POST'])
def add_task():
    try:
        data = request.json
        tasks = load_db()

        # 1. DUPLICATE CHECK: Don't add if same crop on same day exists
        for t in tasks:
            if ( t['date'] == data.get('date') and t['crop'] == data.get('crop') and str(t['amount']) == str(data.get('amount'))):
                print("Duplicate found! Stopping save.") # Optional: For debugging
                
                # CRITICAL: This return statement STOPS the function here.
                # If you remove this 'return', the code continues and saves a duplicate.
                return jsonify({
                    "message": "Task already scheduled for this day!", 
                    "status": "duplicate"
                })

        # 2. SMART CHECK: Evaporation Warning (11 AM - 3 PM)
        time_str = data.get('time', '06:00')
        warning = None
        try:
            hour = int(time_str.split(':')[0])
            if 11 <= hour <= 15:
                warning = "⚠️ High evaporation risk! (11 AM - 3 PM)"
        except:
            pass # Ignore time parsing errors

        # 3. Create Task
        new_task = {
            "id": int(datetime.now().timestamp() * 1000),
            "crop": data.get('crop'),
            "date": data.get('date'),
            "time": time_str,
            "amount": data.get('amount'),
            "status": "Scheduled",
            "warning": warning
        }
        tasks.append(new_task)
        save_db(tasks)

        return jsonify({
            "message": "✅ Added to Planner", 
            "warning": warning, 
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@scheduler_bp.route('/complete_task', methods=['POST'])
def complete_task():
    """Marks a task as Completed without deleting it"""
    task_id = request.json.get('id')
    tasks = load_db()
    
    for t in tasks:
        if t['id'] == task_id:
            t['status'] = 'Completed'
            break
            
    save_db(tasks)
    return jsonify({"message": "Task moved to History"})


@scheduler_bp.route('/delete_task', methods=['POST'])
def delete_task():
    task_id = request.json.get('id')
    tasks = load_db()
    tasks = [t for t in tasks if t['id'] != task_id]
    save_db(tasks)
    return jsonify({"message": "Deleted"})