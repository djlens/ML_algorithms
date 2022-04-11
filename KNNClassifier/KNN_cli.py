import sys
import pandas as pd

from KNN import KNNClassifier


def main(argv):
    if len(argv) == 1:
        print("No data file given")
        exit(1)

    X = pd.read_csv(argv[1])
    y = pd.read_csv(argv[2])

    model = KNNClassifier(int(argv[3])).fit(X, y)

    while True:
        vector = input("Enter vector coordinates: ")
        if vector == "quit":
            exit(0)
        vector = list(map(float, vector.split(' ')))
        print(model.predict_vector(vector))


if __name__ == '__main__':
    main(sys.argv)
