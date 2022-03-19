import pandas as pd

iris = pd.read_csv('data/Iris.csv')
iris.drop('Id', axis=1, inplace=True)

X = iris.drop('Species', axis=1)
y = iris['Species']

X.to_csv('iris_X.csv')
y.to_csv('iris_y.csv')