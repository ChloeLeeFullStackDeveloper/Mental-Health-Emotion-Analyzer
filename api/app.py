import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))           # for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))) # for predict.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))  # for logger.py
from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_emotions
from utils.logger import log_entry

app = Flask(__name__, static_folder="../static", static_url_path="/")
CORS(app)

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    threshold = float(data.get("threshold", 0.3))

    if not text:
        return jsonify({"error": "No input text provided"}), 400

    predictions = predict_emotions(text, threshold)
    log_entry(text, threshold, predictions)
    return jsonify(predictions)

@app.route("/labels", methods=["GET"])
def get_labels():
    from predict import mlb
    return jsonify({"labels": list(mlb.classes_)})

@app.route("/history", methods=["GET"])
def get_history():
    import json
    try:
        with open("history.json", "r") as f:
            history = json.load(f)
        return jsonify(history)
    except FileNotFoundError:
        return jsonify([])
    except json.JSONDecodeError:
        return jsonify([])
    
@app.route("/history", methods=["DELETE"])
def clear_history():
    with open("history.json", "w") as f:
        f.write("[]")
    return jsonify({"status": "cleared"})

@app.route("/email-report", methods=["POST"])
def email_report():
    import smtplib
    from email.message import EmailMessage
    import json
    import csv
    from io import StringIO

    data = request.get_json()
    recipient = data.get("email")

    if not recipient:
        return jsonify({"error": "No email provided"}), 400

    with open("history.json") as f:
        history = json.load(f)

    csv_data = StringIO()
    writer = csv.writer(csv_data)
    writer.writerow(["Timestamp", "Text", "Threshold", "Emotions"])
    for entry in history:
        emotions = ", ".join([f"{k} ({v*100:.1f}%)" for k, v in entry.get("predictions", {}).items()])
        writer.writerow([entry["timestamp"], entry["text"], entry["threshold"], emotions])

    msg = EmailMessage()
    msg["Subject"] = "Your Mental Health Journal Report"
    msg["From"] = "you@example.com"
    msg["To"] = recipient
    msg.set_content("Attached is your emotion analysis journal report.")
    msg.add_attachment(csv_data.getvalue(), filename="journal_report.csv", subtype="csv")

    # 👇 Use your SMTP details (Gmail shown here)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("you@example.com", "your_app_password")
        smtp.send_message(msg)

    return jsonify({"status": "sent"})
if __name__ == "__main__":
    app.run(debug=True)
