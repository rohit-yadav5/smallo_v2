import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MicStream:
    def __init__(self, sr=16000, chunk_ms=500):
        self.sr = sr
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(sr * (chunk_ms / 1000))

    def listen(self):
        while True:
            audio = sd.rec(self.chunk_samples, samplerate=self.sr, channels=1, dtype='float32')
            sd.wait()
            yield audio.flatten()

class ASREngine:
    def __init__(self, model_size="tiny", device="cpu"):
        self.model = WhisperModel(model_size, device=device)

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, best_of=1)
        text = " ".join([seg.text.strip() for seg in segments])
        return text.strip()

class IntentEngine:
    def __init__(self, model_path="intent-bert", threshold=0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.threshold = threshold

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        conf = float(probs[1])
        label = "INTERRUPT" if conf >= self.threshold else "IGNORE"
        return label, conf

class InterruptManager:
    def handle(self, text, label, conf):
        if label == "INTERRUPT":
            print(f"🚨 INTERRUPT ({conf:.2f}) → {text}")
            return True
        else:
            print(f"ok IGNORE ({conf:.2f}) → {text}")
            return False

class LiveRunner:
    def __init__(self):
        self.mic = MicStream()
        self.asr = ASREngine()
        self.intent = IntentEngine()
        self.manager = InterruptManager()

    def start(self):
        print("🎤 Live listening started...\n")
        for chunk in self.mic.listen():
            start = time.time()
            text = self.asr.transcribe(chunk)
            if not text:
                continue
            label, conf = self.intent.classify(text)
            self.manager.handle(text, label, conf)
            latency = (time.time() - start) * 1000
            print(f"[{latency:.2f} ms] text='{text}' → {label}")