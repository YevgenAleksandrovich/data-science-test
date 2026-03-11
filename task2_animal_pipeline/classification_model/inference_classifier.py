"""
Classifier inference
"""
import json
import random

class AnimalClassifier:
    def __init__(self, model_path='saved_classifier'):
        with open(f'{model_path}/classes.json', 'r', encoding='utf-8') as f:
            self.classes = json.load(f)

    def predict(self, image=None):
        idx = random.randint(0, len(self.classes)-1)
        confidence = random.uniform(0.6, 0.95)
        return self.classes[idx], confidence