"""
Main pipeline
"""
from ner_model.inference_ner import AnimalNERExtractor
from classification_model.inference_classifier import AnimalClassifier

class AnimalVerificationPipeline:
    def __init__(self):
        print("🚀 Initializing pipeline...")
        self.ner = AnimalNERExtractor()
        self.classifier = AnimalClassifier()
        print("✅ Pipeline ready")

    def verify(self, text, image=None):
        text_animal = self.ner.extract_animal(text)
        if not text_animal:
            return False

        img_animal, confidence = self.classifier.predict(image)
        return text_animal.lower() == img_animal.lower()