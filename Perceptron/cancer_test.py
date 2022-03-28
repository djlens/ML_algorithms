import pandas as pd
import numpy as np
from Perceptron import PerceptronClassifier
from sklearn.model_selection import train_test_split


def cancer_test():
    cancer = pd.read_csv('../datasets/breast_cancer/data.csv')
    cancer_train = cancer.drop('id', axis=1)
    X = cancer_train.drop(['diagnosis', 'Unnamed: 32'], axis=1).to_numpy()
    y = cancer_train['diagnosis'].replace(['B', 'M'], [0, 1]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    mod = PerceptronClassifier().fit(X_train, y_train, 0.01)
    print("Training accuracy: " + str(sum(mod.predict(X_train) == np.array(y_train))/len(y_train)))
    print("Testing accuracy: " + str(sum(mod.predict(X_test) == np.array(y_test))/len(y_test)))


if __name__ == "__main__":
    cancer_test()
