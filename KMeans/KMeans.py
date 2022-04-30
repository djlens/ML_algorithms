import numpy as np


class KMeans:

    def __init__(self, n_clusters):
        self.X = None
        self.y = None
        self.centroids = None
        self.n_clusters = n_clusters

    def fit(self, X):
        self.X = X
        self.y = np.zeros(self.X.shape[0])

        # initialise centroids from randomly choosen points
        rows_idx = np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
        self.centroids = self.X[rows_idx, :]

        for i in range(100):
            self._assign_clusters()
            self._compute_centroids()

        return self

    def _assign_clusters(self):
        for index, obs in enumerate(self.X):
            distances = [self._euclidean(obs, centroid) for centroid in self.centroids]
            cluster = np.argmin(distances)
            self.y[index] = cluster

    def _compute_centroids(self):
        obs_with_clusters = list(zip(self.X, self.y))
        loss = 0

        for index, centroid in enumerate(self.centroids):
            obs_with_clusters_copy = obs_with_clusters.copy()
            cluster = list(filter(lambda row: row[-1] == index, obs_with_clusters_copy))
            cluster, _ = list(zip(*cluster))
            new_centroid = np.mean(cluster, axis=0)
            self.centroids[index] = new_centroid

            loss += self.WCSS(cluster, new_centroid)

        print(loss)

    def WCSS(self, cluster, centroid):
        return sum([self._euclidean(obs, centroid) for obs in cluster])

    def _euclidean(self, p1, p2):
        # compute distance between two points using Euclidean distance
        assert len(p1) == len(p2)
        return np.sqrt(np.sum((p1 - p2) ** 2))
