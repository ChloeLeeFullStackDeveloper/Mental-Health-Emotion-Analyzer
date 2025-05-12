import joblib
import os
import numpy as np

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
BINARIZER_PATH = os.path.join(BASE_DIR, "label_binarizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
binarizer = joblib.load(BINARIZER_PATH)

def predict_emotions(text, threshold=0.3):
    """Predict emotion probabilities above a given threshold."""
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)

    if isinstance(probs, list):  # in case it's a list of arrays
        probs = np.array(probs)

    if probs.ndim == 3:  # For multilabel classifier chains, average probs
        probs = np.mean(probs, axis=1)

    emotion_probs = probs[0]
    labels = binarizer.classes_

    result = {}
    for i, prob in enumerate(emotion_probs):
        if prob >= threshold:
            result[labels[i]] = float(prob)

    return result
