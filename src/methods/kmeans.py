import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
        self.centers = None
        self.cluster_center_label = None

    def init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features.
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        idx = np.random.permutation(data.shape[0])[:self.K]
        return data[idx].copy()

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features.
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        # broadcasting: (N,1,D) - (1,K,D) -> (N,K,D), then norm over D
        diff = data[:, None, :] - centers[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        return np.argmin(distances, axis=1)

    def compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = np.zeros((self.K, data.shape[1]))
        for k in range(self.K):
            mask = cluster_assignments == k
            if mask.any():
                centers[k] = data[mask].mean(axis=0)
            elif self.centers is not None:
                centers[k] = self.centers[k]  # keep old center if cluster is empty
        return centers

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.

        Arguments:
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        centers = self.init_centers(data)
        self.centers = centers
        cluster_assignments = None
        for _ in range(max_iter):
            distances = self.compute_distance(data, centers)
            cluster_assignments = self.find_closest_cluster(distances)
            new_centers = self.compute_centers(data, cluster_assignments)
            if np.allclose(new_centers, centers):  # converged: centers stopped moving
                break
            centers = new_centers
            self.centers = centers
        return centers, cluster_assignments

    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        labels = np.zeros(self.K, dtype=true_labels.dtype)
        for k in range(self.K):
            mask = cluster_assignments == k
            if mask.any():
                values, counts = np.unique(true_labels[mask], return_counts=True)
            else:
                values, counts = np.unique(true_labels, return_counts=True)  # empty: vote on all
            labels[k] = values[np.argmax(counts)]
        return labels

    def predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns:
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        distances = self.compute_distance(data, centers)
        nearest = self.find_closest_cluster(distances)
        return cluster_center_label[nearest]

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        self.centers = centers
        self.cluster_center_label = self.assign_labels_to_centers(
            centers, cluster_assignments, training_labels)
        return self.predict_with_centers(training_data, self.centers, self.cluster_center_label)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return self.predict_with_centers(test_data, self.centers, self.cluster_center_label)
