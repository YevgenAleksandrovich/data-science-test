"""
Main classifier
"""
from .models.rf_model import RandomForestMnistModel
from .models.nn_model import FeedForwardMnistModel
from .models.cnn_model import CNNMnistModel
import numpy as np

class MnistClassifier:
    def __init__(self, algorithm='rf'):
        self.algorithm = algorithm
        self.model = None

        algorithms = {
            'rf': (RandomForestMnistModel, 'Random Forest'),
            'nn': (FeedForwardMnistModel, 'Neural Network'),
            'cnn': (CNNMnistModel, 'CNN')
        }

        model_class, self.name = algorithms[algorithm]
        self.model = model_class()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, X_test, y_test):
        pred = self.predict(X_test)
        return np.mean(pred == y_test)