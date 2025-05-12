import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"

def log_entry(text, threshold, predictions):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "threshold": threshold,
        "predictions": predictions
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
