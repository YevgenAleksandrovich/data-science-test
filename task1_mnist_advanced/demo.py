"""
Demo script for Task 1
"""
import numpy as np
from mnist_classifier import MnistClassifier
import time

def main():
    print("\n" + "="*60)
    print("TASK 1: MNIST CLASSIFICATION")
    print("="*60)

    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.rand(200, 28, 28) * 255
    y_train = np.random.randint(0, 10, 200)
    X_test = np.random.rand(50, 28, 28) * 255
    y_test = np.random.randint(0, 10, 50)

    print(f"\n📊 Data created:")
    print(f"   Training: {X_train.shape}")
    print(f"   Test: {X_test.shape}")

    results = {}

    for alg, name in [('rf', 'Random Forest'), ('nn', 'Neural Network'), ('cnn', 'CNN')]:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")

        clf = MnistClassifier(algorithm=alg)

        start = time.time()
        clf.train(X_train, y_train)
        train_time = time.time() - start

        acc = clf.accuracy(X_test, y_test)
        results[name] = acc

        print(f"\n📈 Results:")
        print(f"   Accuracy: {acc:.2%}")
        print(f"   Time: {train_time:.2f}s")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    for name, acc in results.items():
        bar = "█" * int(acc * 30)
        print(f"{name:15}: {bar} {acc:.2%}")

    best = max(results, key=results.get)
    print(f"\n✅ Best model: {best}")

if __name__ == "__main__":
    main()