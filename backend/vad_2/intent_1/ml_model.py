# ml_model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class MLIntentClassifier:
    def __init__(self, model_path="intent-bert", threshold=0.70):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, model_path)

        if not os.path.isdir(MODEL_PATH):
            raise OSError(f"Local model folder not found: {MODEL_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        self.model.eval()
        self.threshold = threshold

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0].tolist()
        interrupt_conf = probs[1]

        if interrupt_conf >= self.threshold:
            label = "INTERRUPT"
        else:
            label = "IGNORE"

        return {
            "text": text,
            "label": label,
            "confidence": interrupt_conf,
            "threshold": self.threshold
        }