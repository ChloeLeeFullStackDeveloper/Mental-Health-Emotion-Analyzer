# 🧠 Mental Health Journal Analyzer

An AI-powered web application that helps users reflect on their mental health by analyzing journal entries and predicting emotional states using NLP models.

---

## ✨ Features

- 🔍 Emotion analysis from free-form journal text
- 📊 Emotion frequency summary (e.g., joy, sadness, anger)
- 📁 Export journal history to CSV
- 📅 Filter past entries by emotion and date
- 🧠 Hugging Face model: `cardiffnlp/twitter-roberta-base-emotion`
- ⚙️ Backend powered by Flask + Transformers + Scikit-learn

---

## 📁 Project Structure

```
Mental_Health_Journal_Analyzer/
├── api/                  # Flask API backend
│   └── app.py
├── models/               # Emotion classification models
│   └── predict.py
├── static/               # Frontend static files
│   └── index.html
├── data/                 # CSV and training/test data
│   └── goemotions.csv
├── notebooks/            # Jupyter exploration
│   └── data_exploration.ipynb
├── utils/                # Visualization and helper scripts
│   └── visualization.py
├── history.json          # Emotion prediction history
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md             # ← You're here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Mental_Health_Journal_Analyzer.git
cd Mental_Health_Journal_Analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask backend

```bash
python api/app.py
```

### 5. Open `static/index.html` in a browser

No local server needed for frontend – it reads from the backend API.

---

## 🧪 Dependencies

See `requirements.txt`. Key libraries include:
- Flask
- transformers
- torch
- scikit-learn
- pandas
- matplotlib

---

## 📌 Notes

- This app works entirely on localhost.
- User entries are saved locally to `history.json`.
- No external tracking or cloud storage is used (privacy-focused).

---

## 📅 Last Updated

2025-05-12

---

## 📜 License

MIT License. Feel free to adapt and build upon this project.
