import math
import numpy as np


class KNNClassifier:


    def fit(self, X, y):

        self.k = int(math.sqrt(len(X)))
        self.dim = len(X.columns)
        self.X = X.copy()
        self.y = y.copy()

        self.X.reset_index(inplace=True, drop=True)
        self.X['index'] = self.X.index

        self.X_np = self.X.to_numpy()
        self.y_np = self.y.to_numpy()

        return self

    def predict(self, X):

        X_pred = X.copy()
        X_pred = X_pred.to_numpy()

        def euclidean(v1, v2):
            assert len(v1) == len(v2)

            v1, v2 = np.array(v1), np.array(v2)
            return sum((v2 - v1) ** 2)

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
