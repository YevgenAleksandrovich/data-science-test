"""
NER inference
"""
import json

class AnimalNERExtractor:
    def __init__(self, model_path='saved_ner_model'):
        with open(f'{model_path}/animals.json', 'r', encoding='utf-8') as f:
            self.animals = json.load(f)

    def extract_animal(self, text):
        text_lower = text.lower()
        for animal in self.animals:
            if animal.lower() in text_lower:
                return animal
        return None