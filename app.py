from flask import Flask, render_template, request, jsonify
import datetime

app = Flask(__name__)

# --- MOCK DATABASE ---
# In a real app, this would be SQLite or PostgreSQL
JOURNAL_DB = [
    {
        "id": 1,
        "date": "Oct 25, 2024",
        "time": "8:30 PM",
        "preview": "I messed up the presentation...",
        "content": "I messed up the presentation. My boss hates me. I'm going to get fired.",
        "mood": "Anxious"
    },
    {
        "id": 2,
        "date": "Oct 24, 2024",
        "time": "9:15 AM",
        "preview": "Feeling calm today after...",
        "content": "Had a good sleep. The meditation helped.",
        "mood": "Calm"
    }
]


# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/onboarding')
def onboarding():
    return render_template('onboarding.html')


@app.route('/home')
def patient_dashboard():
    return render_template('patient_dashboard.html')


@app.route('/editor')
def editor():
    return render_template('editor.html')


# Inside app.py

@app.route('/mood-tracker')
def mood_tracker():
    """Step 4: Mood Analytics Page"""

    # Calculate Mood Counts for the Chart
    mood_counts = {"Happy": 0, "Calm": 0, "Anxious": 0, "Low": 0}

    for entry in JOURNAL_DB:
        m = entry.get('mood', 'Neutral')
        if m in mood_counts:
            mood_counts[m] += 1

    # Calculate total for percentages
    total = sum(mood_counts.values())
    if total == 0: total = 1  # Avoid div by zero

    stats = {
        "happy_pct": int((mood_counts["Happy"] / total) * 100),
        "calm_pct": int((mood_counts["Calm"] / total) * 100),
        "anxious_pct": int((mood_counts["Anxious"] / total) * 100),
        "low_pct": int((mood_counts["Low"] / total) * 100),
        "total_entries": len(JOURNAL_DB)
    }

    return render_template('mood_tracker.html', stats=stats, history=JOURNAL_DB)

@app.route('/dashboard')
def dashboard():
    # ... (Keep existing dashboard logic) ...
    return render_template('dashboard.html',
                           data={"name": "Sarina", "id": "8842", "compliance_rate": 85, "heatmap": []})


# --- API: JOURNAL HISTORY ---

@app.route('/api/journal', methods=['GET'])
def get_journal_history():
    """Fetch all past entries"""
    return jsonify(JOURNAL_DB)


@app.route('/api/journal/save', methods=['POST'])
def save_entry():
    """Save a new entry"""
    data = request.json

    # Create new entry object
    new_entry = {
        "id": len(JOURNAL_DB) + 1,
        "date": datetime.datetime.now().strftime("%b %d, %Y"),
        "time": datetime.datetime.now().strftime("%I:%M %p"),
        "preview": data.get('content', '')[:30] + "...",
        "content": data.get('content', ''),
        "mood": data.get('mood', 'Neutral')
    }

    # Prepend to list (newest first)
    JOURNAL_DB.insert(0, new_entry)

    return jsonify({"status": "success", "entry": new_entry})


# --- MOCK AI AGENT LOGIC (Keep existing) ---
def agent_detective(text):
    # ... (Keep existing detective logic) ...
    distortions = []
    triggers = {"boss": "Mind Reading", "hates": "Mind Reading", "fired": "Fortune Telling", "mess": "All-or-Nothing"}
    lower_text = text.lower()
    for keyword, type_val in triggers.items():
        if keyword in lower_text:
            start = lower_text.find(keyword)
            distortions.append({
                "text": text[start:start + len(keyword)],
                "start": start,
                "end": start + len(keyword),
                "type": type_val
            })
    return distortions


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    return jsonify({"distortions": agent_detective(data.get('text', ''))})


if __name__ == '__main__':
    app.run(debug=True, port=5000)