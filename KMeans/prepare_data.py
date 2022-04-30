import numpy as np
from KMeans import KMeans


def prepare_data():
    wines = np.genfromtxt('./data/wine-clustering.csv', delimiter=',', skip_header=True)
    model = KMeans(n_clusters=3).fit(wines)
    return model


if __name__ == "__main__":
    prepare_data()
