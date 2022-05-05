import numpy as np
from KMeans import KMeans
from sklearn.cluster import KMeans as KMeansSK


def test():
    wines = np.genfromtxt('./data/wine-clustering.csv', delimiter=',', skip_header=True)
    model = KMeans(n_clusters=5).fit(wines, no_of_iters=1)
    model2 = KMeansSK(n_clusters=5).fit(wines)
    print(model.wcss)
    print(model.y)
    print("------")
    print(model2.inertia_)
    print(model2.labels_)


if __name__ == "__main__":
    test()