import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# === CONFIG ===
DATA_PATH = os.path.join("data", "goemotions.csv")  # your dataset path
TEXT_COLUMN = "text"
LABEL_COLUMN = "labels"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.pkl")

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: x.split(","))  # multi-label

texts = df[TEXT_COLUMN].tolist()
labels = df[LABEL_COLUMN].tolist()

# === VECTORIZATION ===
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(texts)

# === MULTI-LABEL BINARIZATION ===
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# === MODEL TRAINING ===
base_model = LogisticRegression(solver="liblinear")
classifier = MultiOutputClassifier(base_model)
classifier.fit(X, Y)

# === SAVE MODELS ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(classifier, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(mlb, BINARIZER_PATH)

print("✅ Training complete. Model, vectorizer, and binarizer saved.")
