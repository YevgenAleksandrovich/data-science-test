"""
Train NER model
"""
import json
import os

def train_ner_model():
    animals = [
        'кіт', 'собака', 'корова', 'кінь', 'вівця', 'коза', 'свиня', 'курка',
        'cat', 'dog', 'cow', 'horse', 'sheep', 'goat', 'pig', 'chicken',
        'elephant', 'lion', 'tiger'
    ]

    os.makedirs('saved_ner_model', exist_ok=True)
    with open('saved_ner_model/animals.json', 'w', encoding='utf-8') as f:
        json.dump(animals, f, ensure_ascii=False)

    print(f"✅ NER model trained: {len(animals)} animals")
    return animals

if __name__ == "__main__":
    train_ner_model()