import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pylab as plt
from KNNClassifier import KNNClassifier

df = pd.read_csv('Iris.csv')
df.drop('Id', axis=1, inplace=True)


df_setosa = df[df['Species'] == 'Iris-setosa']
df_versicolor = df[df['Species'] == 'Iris-versicolor']
df_virginica = df[df['Species'] == 'Iris-virginica']


train = pd.concat([df_setosa.iloc[:40], (df_versicolor.iloc[:40]), (df_virginica.iloc[:40])])
test = pd.concat([df_setosa.iloc[40:], (df_versicolor.iloc[40:]), (df_virginica.iloc[40:])])

X_train = train.drop('Species', axis=1)
y_train = train['Species']

X_test = test.drop('Species', axis=1)
y_test = test['Species']

model = KNNClassifier().fit(X_train, y_train)
pred = model.predict(X_test)

print(pred == y_test)