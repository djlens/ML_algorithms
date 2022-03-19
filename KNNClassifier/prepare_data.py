import pandas as pd

iris = pd.read_csv('data/Iris.csv')
iris.drop('Id', axis=1, inplace=True)

X = iris.drop('Species', axis=1)
y = iris['Species']

X.to_csv('data/iris_X.csv', index=False)
y.to_csv('data/iris_y.csv', index=False)
