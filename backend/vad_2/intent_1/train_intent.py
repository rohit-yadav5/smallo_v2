# train_intent.py

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "intent-bert"

# Load dataset
with open("dataset.json") as f:
    data = json.load(f)

texts = []
labels = []

for t in data["interrupt"]:
    texts.append(t)
    labels.append(1)

for t in data["ignore"]:
    texts.append(t)
    labels.append(0)

dataset = Dataset.from_dict({"text": texts, "label": labels})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=32
    )

dataset = dataset.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    OUTPUT_DIR,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model saved to:", OUTPUT_DIR)
