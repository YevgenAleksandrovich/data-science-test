#%%
"""
TASK 1: MNIST CLASSIFICATION
Run this file directly: python run_task1.py
"""

import numpy as np
import time
import sys
import os

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from mnist_classifier import MnistClassifier

print("\n" + "=" * 60)
print("🎯 TASK 1: MNIST CLASSIFICATION")
print("=" * 60)

# Create synthetic data
np.random.seed(42)
n_train = 200
n_test = 50

print(f"\n📊 Creating {n_train} training and {n_test} test images...")
X_train = np.random.rand(n_train, 28, 28) * 255
y_train = np.random.randint(0, 10, n_train)
X_test = np.random.rand(n_test, 28, 28) * 255
y_test = np.random.randint(0, 10, n_test)

print(f"   Training data shape: {X_train.shape}")
print(f"   Test data shape: {X_test.shape}")

# Test all models
results = {}
models = [
 ('rf', '🌲 RANDOM FOREST'),
 ('nn', '🧠 NEURAL NETWORK'),
 ('cnn', '⚡ CNN')
]

for alg, name in models:
 print(f"\n{'-' * 50}")
 print(f"{name}")
 print(f"{'-' * 50}")

 try:
  clf = MnistClassifier(algorithm=alg)

  start = time.time()
  clf.train(X_train, y_train)
  train_time = time.time() - start

  start = time.time()
  acc = clf.accuracy(X_test, y_test)
  pred_time = time.time() - start

  print(f"\n📈 Results:")
  print(f"   ✓ Accuracy: {acc:.2%}")
  print(f"   ✓ Training time: {train_time:.2f}s")
  print(f"   ✓ Prediction time: {pred_time:.3f}s")

  results[name] = acc

 except Exception as e:
  print(f"   ❌ Error: {e}")

# Final comparison
print("\n" + "=" * 60)
print("📊 FINAL RESULTS")
print("=" * 60)

if results:
 print()
 for name, acc in results.items():
  bar = "█" * int(acc * 30)
  print(f"{name:20} {bar} {acc:.2%}")

 best = max(results, key=results.get)
 best_acc = results[best]
 print(f"\n🏆 BEST MODEL: {best} with {best_acc:.2%} accuracy")

 print("\n✅ Task 1 completed successfully!")
else:
 print("\n❌ No models were tested successfully")