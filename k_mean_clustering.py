import matplotlib.pyplot as plt
import numpy as np
import random


class KMeanClustering:
    def __init__(self, data, error_threshold=0.01, k_max=10, repetition=10):
        self.data = data
        self.error_threshold = error_threshold
        self.k_max = k_max
        self.repetition = repetition

        self.clustered_points = {}
        self.cluster_location = {}
        self.inertia_value = {}
        self.elbow_k = None

    def k_mean_clustering_once(self, k):
        def average(mean_before, n, a_n1):
            return (mean_before * n + a_n1) / (n + 1)

        data_range = [[min(self.data[:, i]), max(self.data[:, i])] for i in range(self.data.shape[1])]
        cluster_location = np.array([[random.uniform(*min_max) for min_max in data_range] for _ in range(k)])

        while True:
            clustered_point = [[] for _ in range(k)]
            mean_error = 0

            # Divide the points into clusters
            for point in self.data:
                norm_array = [[np.linalg.norm(point - cluster, ord=2), i] for i, cluster in enumerate(cluster_location)]
                nearest_cluster = sorted(norm_array, key=lambda x: x[0])[0][1]
                clustered_point[nearest_cluster].append(point)

            # Update the location of clusters
            for index in range(k):
                if clustered_point[index]:
                    new_location = np.array([np.mean(np.array(clustered_point[index])[:, 0]),
                                             np.mean(np.array(clustered_point[index])[:, 1])])
                else:
                    new_location = np.array([random.uniform(*min_max) for min_max in data_range])
                error = np.linalg.norm(new_location - cluster_location[index]) / np.linalg.norm(new_location)
                mean_error = average(mean_error, index, error)
                cluster_location[index] = new_location

            # Evaluate the location of clusters
            if mean_error < self.error_threshold:
                final_clustered_point = clustered_point
                break

        inertia_value = sum([sum([np.linalg.norm(point - cluster, ord=2)
                                  for point in final_clustered_point[i]]) for i, cluster in
                             enumerate(cluster_location)])

        return final_clustered_point, cluster_location, inertia_value

    def repeat_K_mean_clustering(self, k):
        clustered_points = None
        cluster_location = None
        inertia_value = float('inf')
        for _ in range(self.repetition):
            clustering_result = self.k_mean_clustering_once(k)
            if clustering_result[2] < inertia_value:
                clustered_points = clustering_result[0]
                cluster_location = clustering_result[1]
                inertia_value = clustering_result[2]
        return clustered_points, cluster_location, inertia_value

    def find_elbow_k(self):
        inertia_value_list = list(self.inertia_value.values())
        d_inertia_value = [x2 - x1 for x1, x2 in zip(inertia_value_list[:-1], inertia_value_list[1:])]
        elbow_gradient = (d_inertia_value[0] + d_inertia_value[-1]) / 2
        for i, gradient in enumerate(d_inertia_value):
            if gradient > elbow_gradient:
                return (i + 1) + 1

    def plot_clustered_result(self):
        plt.figure()
        plt.plot(self.inertia_value.keys(), self.inertia_value.values(), 'r-o')
        plt.plot(self.elbow_k, self.inertia_value[self.elbow_k], 'b-o')

        plt.figure()
        for i, points in enumerate(self.clustered_points[self.elbow_k]):
            plt.scatter(*np.array(points).T,
                        color=(1 / self.elbow_k * i, 1 - 1 / self.elbow_k * i, 1 / self.elbow_k * i))
        plt.scatter(*self.cluster_location[self.elbow_k].T, color='r')
        plt.show()

    def k_mean_clustering(self, plot=True):
        for k in range(1, self.k_max):
            clustering_result = self.repeat_K_mean_clustering(k)
            self.clustered_points[k] = clustering_result[0]
            self.cluster_location[k] = clustering_result[1]
            self.inertia_value[k] = clustering_result[2]

        self.elbow_k = self.find_elbow_k()

        if plot:
            self.plot_clustered_result()

        return self.clustered_points[self.elbow_k]


def main():
    data = np.array([[1, 1],
                     [1, 20],
                     [10, 2],
                     [20, 30],
                     [30, 30],
                     [14, 12],
                     [5, 7],
                     [1, 8],
                     [18, 30],
                     [20, 20],
                     [25, 25],
                     [24, 24],
                     [30, 10],
                     [10, 2],
                     [1, 10],
                     [11, 21],
                     [11, 30],
                     [30, 2],
                     [30, 10],
                     [30, 20],
                     [12, 14],
                     [7, 7],
                     [2, 8],
                     [17, 30],
                     [22, 20],
                     [15, 25],
                     [28, 24],
                     [10, 10],
                     [20, 2],
                     [7, 10],
                     [1, 9],
                     [1, 3],
                     [10, 9],
                     [20, 19],
                     [30, 19],
                     [14, 19],
                     [5, 9],
                     [1, 19],
                     [5, 5],
                     [5, 6],
                     [6, 5],
                     [6, 6],
                     [7, 7],
                     [5, 7],
                     [7, 5],
                     [7, 6],
                     [6, 7],
                     [30, 19],
                     [14, 19],
                     [5, 9],
                     [1, 19],
                     [18, 17],
                     [20, 8],
                     [25, 6],
                     [17, 5],
                     [17, 17],
                     [10, 4],
                     [30, 2],
                     [29, 4],
                     [30, 3],
                     [16, 1],
                     [28, 2],
                     [17, 14],
                     [28, 3],
                     [29, 30],
                     [10, 18],
                     [10, 21],
                     [11, 28],
                     [29, 21],
                     [30, 4],
                     [19, 11],
                     [28, 21],
                     [28, 11],
                     [28, 30],
                     [29, 30],
                     [30, 28],
                     [10, 28],
                     [11, 28],
                     [29, 2],
                     [30, 3],
                     [29, 11],
                     [28, 21],
                     [28, 11],
                     [18, 3],
                     [19, 13],
                     [1, 18],
                     [4, 1],
                     [4, 4]])

    k_mean_clustering = KMeanClustering(data, error_threshold=0.001, k_max=10, repetition=10)
    k_mean_clustering.k_mean_clustering(plot=True)


if __name__ == '__main__':
    main()
