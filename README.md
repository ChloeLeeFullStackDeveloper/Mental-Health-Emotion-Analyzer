# 🧠 Mental Health Emotion Analyzer

This project uses AI to analyze emotion patterns in journal entries. Built with Flask and scikit-learn, it predicts emotional states from text and provides a visual summary of emotional trends.

---

## 🔍 Features

- 📝 Journal entry submission
- 🤖 AI-powered emotion classification using logistic regression
- 📊 Bar chart visualization of detected emotions
- 🎯 Customizable confidence threshold (0–1)
- 🗂️ Emotion filtering (e.g., show only joy-related results)
- 🕘 History of journal predictions (with timestamps)

---

## 🧠 Model Overview

- **Vectorizer**: TF-IDF
- **Model**: Logistic Regression (multi-label)
- **Training Data**: Emotion-labeled dataset (e.g., [GoEmotions](https://github.com/google-research/goemotions))
- **Labels**: Multi-label output using `MultiLabelBinarizer`

---

## 🛠️ Tech Stack

- **Frontend**: HTML + JavaScript + Chart.js
- **Backend**: Python (Flask)
- **ML Tools**: scikit-learn, joblib

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/ChloeLeeFullStackDeveloper/Mental-Health-Emotion-Analyzer.git
cd mental-health-analyzer

2. Set up virtual environment
bash
Copy
Edit
python3 -m venv env
source env/bin/activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Start the server
bash
Copy
Edit
cd api
python app.py
Then visit http://127.0.0.1:5000/ in your browser.

🗂️ File Structure
```
mental_health_analyzer/
│
├── api/
│   └── app.py               # Flask backend
├── models/
│   ├── predict.py           # Emotion prediction logic
│   ├── logistic_model.pkl   # Trained model
│   ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
│   └── label_binarizer.pkl  # MultiLabelBinarizer
├── utils/
│   └── logger.py            # Optional logging
├── index.html               # Frontend UI
├── requirements.txt
└── README.md
```
📄 License
MIT License © 2025