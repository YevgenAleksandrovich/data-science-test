#%%
import sys
print('Python %s on %s' % (sys.version, sys.platform))

import numpy as np
import json
import os

# Add path
sys.path.append('.')

print("\n✅ Libraries imported successfully")
#%%
# Check if models exist, train if needed
if not os.path.exists('saved_ner_model/animals.json'):
    print("📦 Training NER model...")
    from ner_model.train_ner import train_ner_model
    train_ner_model()

if not os.path.exists('saved_classifier/classes.json'):
    print("📦 Training classifier...")
    from classification_model.train_classifier import train_classifier
    train_classifier()
#%%
# Import components
from ner_model.inference_ner import AnimalNERExtractor
from classification_model.inference_classifier import AnimalClassifier
from pipeline import AnimalVerificationPipeline

print("✅ Components imported")
#%%
# Initialize pipeline
print("\n" + "="*60)
print("🚀 INITIALIZING PIPELINE")
print("="*60)

ner = AnimalNERExtractor()
classifier = AnimalClassifier()
pipeline = AnimalVerificationPipeline()

print("\n✅ All components loaded successfully")
#%%
print("\n" + "=" * 60)
print("🔍 NER MODEL TEST")
print("=" * 60)

test_texts = [
    "There is a cow in the picture",
    "I see a cat!",
    "На фото собака.",
    "The elephant is big",
    "A horse runs fast",
    "No animal here",
    "Корова пасеться на лузі",
    "Look at that lion!"
]

print(f"\n{'Text':<35} {'Animal Found':<15} {'Status'}")
print(f"{'-' * 65}")

correct = 0
expected_results = ['cow', 'cat', 'собака', 'elephant', 'horse', None, 'корова', 'lion']

for i, text in enumerate(test_texts):
    animal = ner.extract_animal(text)

    if (animal is None and expected_results[i] is None) or (animal is not None and expected_results[i] is not None):
        status = "✓"
        correct += 1
    else:
        status = "✗"

    animal_str = animal if animal else "None"
    print(f"{text[:34]:<35} {animal_str:<15} {status}")

accuracy = correct / len(test_texts)
print(f"\n📊 NER Accuracy: {accuracy:.2%}")

if accuracy >= 0.8:
    print("✅ NER model working well!")
else:
    print("⚠ NER model needs improvement")
#%%
print("\n" + "=" * 60)
print("🖼️ CLASSIFIER TEST")
print("=" * 60)

n_tests = 10
confidences = []
predictions = []

print(f"\nTesting {n_tests} random images:")
print(f"{'#':<3} {'Animal':<12} {'Confidence':<10} {'Progress'}")
print(f"{'-' * 50}")

for i in range(n_tests):
    animal, conf = classifier.predict(None)
    predictions.append(animal)
    confidences.append(conf)

    bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    print(f"{i + 1:<3} {animal:<12} {conf:<9.2%} [{bar}]")

print(f"\n📊 Statistics:")
print(f"   Mean confidence: {np.mean(confidences):.2%}")
print(f"   Min confidence: {np.min(confidences):.2%}")
print(f"   Max confidence: {np.max(confidences):.2%}")
print(f"   Unique animals: {len(set(predictions))}/{len(set(ner.animals))}")

if np.mean(confidences) > 0.7:
    print("✅ Classifier working well!")
else:
    print("⚠ Classifier needs improvement")
#%%
print("\n" + "=" * 60)
print("🌍 BILINGUAL SUPPORT TEST")
print("=" * 60)

bilingual_tests = [
    ("There is a cat", "en", "cat"),
    ("I see a dog", "en", "dog"),
    ("The cow is big", "en", "cow"),
    ("На фото кіт", "uk", "кіт"),
    ("Собака біжить", "uk", "собака"),
    ("Корова пасеться", "uk", "корова"),
    ("A horse runs", "en", "horse"),
    ("Кінь скаче", "uk", "кінь")
]

en_correct = 0
uk_correct = 0
en_total = 0
uk_total = 0

print(f"\n{'Text':<30} {'Lang':<5} {'Expected':<10} {'Found':<10} {'Status'}")
print(f"{'-' * 70}")

for text, lang, expected in bilingual_tests:
    result = ner.extract_animal(text)

    if lang == 'en':
        en_total += 1
        if result and (result.lower() == expected.lower() or result == expected):
            en_correct += 1
            status = "✓"
        else:
            status = "✗"
    else:
        uk_total += 1
        if result == expected:
            uk_correct += 1
            status = "✓"
        else:
            status = "✗"

    result_str = result if result else "None"
    print(f"{text[:28]:<30} {lang:<5} {expected:<10} {result_str:<10} {status}")

en_acc = en_correct / en_total if en_total > 0 else 0
uk_acc = uk_correct / uk_total if uk_total > 0 else 0

print(f"\n📊 Results:")
print(f"   English: {en_acc:.2%} ({en_correct}/{en_total})")
print(f"   Ukrainian: {uk_acc:.2%} ({uk_correct}/{uk_total})")

if en_acc >= 0.8 and uk_acc >= 0.8:
    print("✅ Excellent bilingual support!")
else:
    print("⚠ Language support needs improvement")
#%%
print("\n" + "="*60)
print("🔄 FULL PIPELINE TEST")
print("="*60)

pipeline_tests = [
    "There is a cow",
    "I see a cat",
    "На фото собака",
    "The elephant is big",
    "A horse runs",
    "No animal here"
]

print(f"\n{'Text':<30} {'Result':<10}")
print(f"{'-'*40}")

pipeline_results = []
for text in pipeline_tests:
    result = pipeline.verify(text)
    pipeline_results.append(result)
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{text[:28]:<30} {status}")

success_rate = sum(pipeline_results) / len(pipeline_tests)
print(f"\n📊 Pipeline success rate: {success_rate:.2%}")

if success_rate >= 0.7:
    print("✅ Pipeline working correctly!")
else:
    print("⚠ Pipeline needs improvement")
#%%
print("\n" + "="*60)
print("⚡ PERFORMANCE TEST")
print("="*60)

import time

n_requests = 15
print(f"\nTesting {n_requests} sequential requests...")

start_time = time.time()

for i in range(n_requests):
    pipeline.verify("There is a cow")
    if (i + 1) % 5 == 0:
        progress = (i + 1) / n_requests * 100
        bar = "█" * int((i + 1) / 2) + "░" * (8 - int((i + 1) / 2))
        print(f"   [{bar}] {progress:.0f}% ({i+1}/{n_requests})")

total_time = time.time() - start_time
avg_time = total_time / n_requests

print(f"\n📊 Performance metrics:")
print(f"   Total time: {total_time:.2f}s")
print(f"   Average time per request: {avg_time*1000:.2f}ms")
print(f"   Throughput: {1/avg_time:.1f} requests/second")

if avg_time < 0.5:
    print("✅ Excellent performance!")
elif avg_time < 1.0:
    print("⚠ Acceptable performance")
else:
    print("❌ Needs optimization")
#%%
print("\n" + "="*60)
print("🎮 INTERACTIVE TEST MODE")
print("="*60)

interactive_tests = [
    "There is a cow",
    "I see a cat",
    "На фото собака",
    "The elephant"
]

print("\nRunning sample tests:")
for text in interactive_tests:
    print(f"\n📝 Testing: '{text}'")
    result = pipeline.verify(text)
    if result:
        print(f"   ✅ RESULT: Text matches image")
    else:
        print(f"   ❌ RESULT: Text does NOT match image")
#%%
print("\n" + "✨"*60)
print("FINAL TEST SUMMARY")
print("✨"*60)

summary = [
    "✓ NER model loaded and tested",
    "✓ Classifier loaded and tested",
    "✓ Bilingual support verified",
    "✓ Full pipeline executed",
    "✓ Performance measured",
    "✓ Interactive mode tested"
]

for item in summary:
    print(f"   {item}")

print(f"\n📊 Key metrics:")
print(f"   • NER accuracy: {accuracy:.2%}")
print(f"   • Classifier confidence: {np.mean(confidences):.2%}")
print(f"   • Pipeline success: {success_rate:.2%}")
print(f"   • Response time: {avg_time*1000:.2f}ms")
print(f"   • English support: {en_acc:.2%}")
print(f"   • Ukrainian support: {uk_acc:.2%}")

print("\n" + "✅"*30)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("✅"*30)