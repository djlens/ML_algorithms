import numpy as np
import pandas as pd
from Perceptron import PerceptronClassifier


def iris_test():
    iris = pd.read_csv('../datasets/iris/Iris.csv')
    iris_setosa = iris[iris['Species'] == 'Iris-setosa']
    iris_virginica = iris[iris['Species'] == 'Iris-virginica']
    iris_versicolor = iris[iris['Species'] == 'Iris-versicolor']

    iris_train = pd.concat([iris_virginica[:35], iris_versicolor[:35]])
    X_train = iris_train.drop(['Species', 'Id'], axis=1).to_numpy()
    y_train = iris_train['Species'].to_numpy()

    iris_test = pd.concat([iris_virginica[35:], iris_versicolor[35:]])
    X_test = iris_test.drop(['Species', 'Id'], axis=1).to_numpy()
    y_test = iris_test['Species'].to_numpy()

    mapping = {
        'Iris-virginica': 0,
        'Iris-versicolor': 1
    }

    y_train = np.array(list(map(lambda x: mapping[x], y_train)))
    y_test = np.array(list(map(lambda x: mapping[x], y_test)))

    perceptron = PerceptronClassifier().fit(X_train, y_train, 0.01, epochs=1000)
    print(("Train accuracy: " + str(sum(perceptron.predict(X_train) == y_train) / len(y_train))))

    pred = perceptron.predict(X_test)
    print("Test accuracy: " + str(sum(pred == y_test) / len(y_test)))


if __name__ == '__main__':
    iris_test()
