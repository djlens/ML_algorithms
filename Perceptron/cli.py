import pandas as pd

from Perceptron import PerceptronClassifier


def cli():
    iris = pd.read_csv('../datasets/iris/Iris.csv')
    iris_setosa = iris[iris['Species'] == 'Iris-setosa']
    iris_virginica = iris[iris['Species'] == 'Iris-virginica']
    iris_versicolor = iris[iris['Species'] == 'Iris-versicolor']

    iris_train = pd.concat([iris_virginica, iris_versicolor])
    X = iris_train.drop(['Species', 'Id'], axis=1).to_numpy()
    y = iris_train['Species'].replace(['Iris-virginica', 'Iris-versicolor'], [0, 1]).to_numpy()

    mod = PerceptronClassifier().fit(X, y, 0.01)
    print(("Train accuracy: " + str(sum(mod.predict(X) == y) / len(y))))

    while True:
        vector = input("Enter vector coordinates: ")
        if vector == "quit":
            exit(0)
        vector = list(map(float, vector.split(' ')))
        if mod.predict([vector])[0] == 0:
            print('Virginica')
        else:
            print('Versicolor')


if __name__ == '__main__':
    cli()
