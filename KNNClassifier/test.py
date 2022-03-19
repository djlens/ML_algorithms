import pandas as pd
from KNN import KNNClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = pd.read_csv('data/Iris.csv')
iris.drop('Id', axis=1, inplace=True)

iris_setosa = iris[iris['Species'] == 'Iris-setosa']
iris_versicolor = iris[iris['Species'] == 'Iris-versicolor']
iris_virginica = iris[iris['Species'] == 'Iris-virginica']

iris_train = pd.concat([iris_setosa.iloc[:35], iris_versicolor.iloc[:35], iris_virginica.iloc[:35]])
iris_test = pd.concat([iris_setosa.iloc[35:], iris_versicolor.iloc[35:], iris_virginica.iloc[35:]])

X_iris_train = iris_train.drop('Species', axis=1)
y_iris_train = iris_train['Species']

X_iris_test = iris_test.drop('Species', axis=1)
y_iris_test = iris_test['Species']

model = KNNClassifier(k=1).fit(X_iris_train, y_iris_train)
pred = model.predict(X_iris_test)

accuracy_on_k = [(sum(KNNClassifier(k=x).fit(X_iris_train, y_iris_train).predict(X_iris_test) == y_iris_test)
                  / len(y_iris_test))
                 for x in range(1, 120)]


cancer = pd.read_csv('data/cancer.csv')
cancer.drop('id', axis=1, inplace=True)

cancer.drop('Unnamed: 32', axis=1, inplace=True)
X_cancer = cancer.drop('diagnosis', axis=1)
y_cancer = cancer['diagnosis']

X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.33)

model_cancer = KNNClassifier().fit(X_cancer_train, y_cancer_train)
pred_cancer = model_cancer.predict(X_cancer_test)

print("Iris model accurracy: " + str(sum(pred == y_iris_test) / len(y_iris_test)))
print("Breast cancer model accuracy: " + str(sum(pred_cancer == y_cancer_test) / len(y_cancer_test)))
plt.plot(range(1, 120), accuracy_on_k)
plt.show()
