import math
import pandas as pd
import numpy as np

def euclidean(v1, v2):
    assert len(v1) == len(v2)

    v1, v2 = np.array(v1), np.array(v2)
    return sum((v2 - v1) ** 2)


class KNNClassifier:

    def __init__(self, k=0):

        self.k = k

    def fit(self, X, y):

        self.X = X.copy()
        self.y = y.copy()

        if self.k == 0:
            self.k = int(math.sqrt(len(X)))

        self.dim = len(X.columns)
        X.reset_index(inplace=True, drop=True)
        X['index'] = X.index

        self.X_np = X.to_numpy()
        self.y_np = y.to_numpy()

        return self

    def predict(self, X):

        X_pred = X.copy()
        X_pred = X_pred.to_numpy()

        distances = [sorted([(euclidean(x1[:self.dim], x2[:self.dim]), x2[self.dim]) for x2 in self.X_np])[:self.k] for
                     x1 in X_pred]

        class_dict = {}
        for name in np.unique(self.y):
            class_dict[name] = 0

        labels = []
        for vector_group in distances:
            temp_dict = class_dict.copy()
            for vect in vector_group:
                temp_dict[self.y_np[int(vect[1])]] += 1

            labels.append(max(temp_dict, key=temp_dict.get))

        return np.array(labels)


    def predict_vector(self, V):
        distances = sorted([(euclidean(V[:self.dim], x2[:self.dim]), x2[self.dim]) for x2 in self.X_np])[:self.k]

        class_dict = {}
        for name in np.unique(self.y):
            class_dict[name] = 0

        for vect in distances:
            class_dict[self.y_np[int(vect[1])][0]] += 1

        return max(class_dict, key=class_dict.get)

