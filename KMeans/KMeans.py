
import numpy as np


def euclidean(p1, p2):
    # compute distance between two points using Euclidean distance
    assert len(p1) == len(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calculate_distances_within_cluster(cluster, centroid):
    return sum([euclidean(obs, centroid) ** 2 for obs in cluster])


class KMeans:

    def __init__(self, n_clusters):
        self.X = None
        self.y = None
        self.y_ = None
        self.wcss = 0
        self.wcss_ = []
        self.centroids = None
        self.centroids_ = None
        self.n_clusters = n_clusters
        self._iter = None
        self.loss = np.inf
        self.prev_loss = np.inf

    def fit(self, X, no_of_iters=10):
        self.y_ = [[] for _ in range(no_of_iters)]
        self.centroids_ = [[] for _ in range(no_of_iters)]
        for x in range(no_of_iters):
            self._iter = x
            self.X = X
            self.y_[self._iter] = np.zeros(self.X.shape[0])
            self.wcss = 0

            # initialise centroids from randomly choosen points
            rows_idx = np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
            self.centroids_[self._iter] = self.X[rows_idx, :]

            self._assign_clusters()
            self._compute_centroids()

            while self.prev_loss > self.loss:
                self._assign_clusters()
                self._compute_centroids()

            self.wcss_.append(self.loss)
            # print(self.loss)
            # print(self.y_[self._iter])

        self.wcss = min(self.wcss_)
        self.y = self.y_[np.argmin(self.wcss_)]
        self.centroids = self.centroids_[np.argmin(self.wcss_)]

        return self

    def _assign_clusters(self):
        for index, obs in enumerate(self.X):
            distances = [euclidean(obs, centroid) for centroid in self.centroids_[self._iter]]
            cluster = np.argmin(distances)
            if cluster != self.y_[self._iter][index]:
                self.y_[self._iter][index] = cluster

    def _compute_centroids(self):
        obs_with_clusters = list(zip(self.X, self.y_[self._iter]))
        self.prev_loss = self.loss
        self.loss = 0

        for index, centroid in enumerate(self.centroids_[self._iter]):
            obs_with_clusters_copy = obs_with_clusters.copy()
            cluster = list(filter(lambda row: row[-1] == index, obs_with_clusters_copy))
            cluster, _ = list(zip(*cluster))
            new_centroid = np.mean(cluster, axis=0)
            self.centroids_[self._iter][index] = new_centroid
            self.loss += calculate_distances_within_cluster(cluster, new_centroid)
