import random
import numpy as np


class PerceptronClassifier:

    def __init__(self):
        self.learning_rate = 0
        self.theta = 0.5
        self.X = None
        self.y = None
        self.weights = None
        self.m = None
        self.n = None

    def fit(self, X, y, learning_rate, epochs=1000):
        self.X = X
        self.y = y
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = [np.random.rand() for _ in range(self.n)]

        for _ in range(epochs):
            shuffle = list(zip(self.X, self.y))
            random.shuffle(shuffle)
            self.X, self.y = zip(*shuffle)

            for i in range(self.m):
                self.weights = self.weights + (self.y[i] - self._predict_theta(self.X[i])) * self.learning_rate * self.X[i]
                self.theta = self.theta - (self.y[i] - self._predict_theta(self.X[i])) * self.learning_rate

        return self

    def predict(self, test_set):
        return np.array([self._predict(v) for v in test_set])

    def _predict(self, vector):
        return np.dot(vector, self.weights)

    def _predict_theta(self, vector):
        if np.dot(vector, self.weights) >= self.theta:
            return 1
        else:
            return 0
