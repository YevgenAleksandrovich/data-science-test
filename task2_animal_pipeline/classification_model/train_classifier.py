"""
Train classifier
"""
import json
import os
import numpy as np

def train_classifier():
    classes = ['кіт', 'собака', 'корова', 'кінь', 'вівця', 'коза', 'свиня', 'курка']

    os.makedirs('saved_classifier', exist_ok=True)
    with open('saved_classifier/classes.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False)

    np.save('saved_classifier/class_weights.npy', np.ones(len(classes)))
    print(f"✅ Classifier trained: {len(classes)} classes")
    return classes

if __name__ == "__main__":
    train_classifier()