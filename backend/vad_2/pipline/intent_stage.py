# vad/pipline/intent_stage.py
from vad.intent_1.hybrid_classifier import HybridIntentClassifier

class IntentStage:
    def __init__(self):
        self.classifier = HybridIntentClassifier()

    def classify(self, text):
        return self.classifier.predict(text)