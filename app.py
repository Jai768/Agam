from flask import Flask, render_template, request, jsonify
import datetime

app = Flask(__name__)

# --- MOCK DATABASE ---
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

# --- THE REFRAME ENGINE (Defined Locally) ---
REFRAME_MAP = {
    "catastrophizing": {
        "title": "Catastrophizing",
        "keywords": ["fired", "ruined", "disaster", "suicide", "end", "mess", "commit"],
        "message": "You are jumping to the absolute worst-case scenario.",
        "alternative": "It's not the end of the world. This is a temporary crisis, and your value is not defined by this single event."
    },
    "mind_reading": {
        "title": "Mind Reading",
        "keywords": ["hates", "thinks", "judging", "because of me", "disappointed", "boss"],
        "message": "You are assuming others have negative intentions without proof.",
        "alternative": "They might be tired or stressed. You cannot know their thoughts unless they tell you."
    },
    "all_or_nothing": {
        "title": "All-or-Nothing",
        "keywords": ["always", "never", "everyone", "nobody", "perfect", "failure"],
        "message": "You are seeing things in extremes (black or white).",
        "alternative": "Life happens in the grey areas. A partial success is not a total failure."
    }
}


def agent_detective(text):
    text_lower = text.lower()
    distortions = []

    for key, data in REFRAME_MAP.items():
        for word in data['keywords']:
            if word in text_lower:
                start_index = text_lower.find(word)
                end_index = start_index + len(word)

                distortions.append({
                    "id": key,
                    "word": word,
                    "type": data['title'],
                    "verdict": data['message'],
                    "alternative": data['alternative'],
                    "start": start_index,
                    "end": end_index,
                    "text": text[start_index:end_index]
                })
                break
    return distortions


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


@app.route('/breathe')
def breathe():
    return render_template('breathe.html')


@app.route('/exercises')
def exercises():
    return render_template('therapist_exercises.html')


@app.route('/mood-tracker')
def mood_tracker():
    mood_counts = {"Happy": 0, "Calm": 0, "Anxious": 0, "Low": 0}
    for entry in JOURNAL_DB:
        m = entry.get('mood', 'Neutral')
        if m in mood_counts: mood_counts[m] += 1
    total = sum(mood_counts.values()) or 1
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
    return render_template('dashboard.html', data={"name": "Sarina", "compliance_rate": 85})


# --- API ---

@app.route('/api/journal', methods=['GET'])
def get_journal_history():
    return jsonify(JOURNAL_DB)


@app.route('/api/journal/save', methods=['POST'])
def save_entry():
    data = request.json
    new_entry = {
        "id": len(JOURNAL_DB) + 1,
        "date": datetime.datetime.now().strftime("%b %d, %Y"),
        "time": datetime.datetime.now().strftime("%I:%M %p"),
        "preview": data.get('content', '')[:30] + "...",
        "content": data.get('content', ''),
        "mood": data.get('mood', 'Neutral')
    }
    JOURNAL_DB.insert(0, new_entry)
    return jsonify({"status": "success", "entry": new_entry})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    return jsonify({"distortions": agent_detective(data.get('text', ''))})


if __name__ == '__main__':
    app.run(debug=True, port=5000)