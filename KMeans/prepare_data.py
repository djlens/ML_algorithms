import numpy as np
from KMeans import KMeans
from sklearn.cluster import KMeans as KMeansSK


def prepare_data():
    wines = np.genfromtxt('./data/wine-clustering.csv', delimiter=',', skip_header=True)
    model = KMeans(n_clusters=3).fit(wines)
    model2 = KMeansSK(n_clusters=3, random_state=5).fit(wines)
    print(model.wcss_)
    print((model.wcss))
    print("------")
    print(model2.inertia_)


if __name__ == "__main__":
    prepare_data()
