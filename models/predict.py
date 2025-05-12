from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load once and reuse
MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Hugging Face model: {e}")

EMOTIONS = ['anger', 'joy', 'optimism', 'sadness']  # Update if different

def predict_emotions(text, threshold=0.3):
    """Predict emotions using Hugging Face Transformers model."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()

    result = {}
    for label, prob in zip(EMOTIONS, probs):
        if prob >= threshold:
            result[label] = float(round(prob, 4))

    return result
