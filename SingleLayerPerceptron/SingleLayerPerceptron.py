from Perceptron import PerceptronClassifier
import numpy as np


class SingleLayerPerceptron:

    def __init__(self):
        self.train = None
        self.perceptrons = []

    def fit(self, X, y, number_of_classes):
        self.train = (list(zip(X, y)))
        self.train = np.array([np.append(x[0], x[1]) for x in self.train])

        for k in range(number_of_classes):
            train_tmp = np.array(list(map(lambda row: binary_labels(row, k), self.train)))
            X = train_tmp[:, :-1]
            y = train_tmp[:, -1]
            model = PerceptronClassifier().fit(X, y, 0.01)
            self.perceptrons.append(model)
        return self

    def predict(self, test_set):
        result = []
        for vect in test_set:
            _result = []
            for model in self.perceptrons:
                _result.append(model.predict([vect])[0])
            result.append(_result)
        return result


def binary_labels(row, k):
    new_row = row.copy()
    if new_row[-1] == k:
        new_row[-1] = 1
        return new_row
    else:
        new_row[-1] = 0
        return new_row
