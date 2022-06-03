import matplotlib.pyplot as plt
import numpy as np
import random


class KMeanClustering:
    def __init__(self, data, error_threshold=0.01, k_max=10, repetition=10):
        self.data = data
        self.data_dimension = data.shape[1]
        self.data_range = self.get_data_range()
        self.normalized_data = self.data_normalization()

        self.error_threshold = error_threshold
        self.k_max = k_max
        self.repetition = repetition

        self.clustered_point_index = {}
        self.cluster_location = {}
        self.inertia_value = {}
        self.elbow_k = None

    def get_data_range(self):
        return [[min(self.data[:, i]), max(self.data[:, i])] for i in range(self.data_dimension)]

    def data_normalization(self):
        normalized_data = [(x - self.data_range[i][0]) / (self.data_range[i][1] - self.data_range[i][0])
                           for i, x in enumerate(self.data.T)]
        return np.array(normalized_data).T

    def k_mean_clustering_once(self, k):
        def average(mean_before, n, a_n1):
            return (mean_before * n + a_n1) / (n + 1)

        cluster_location = np.array([[random.random() for _ in range(self.data_dimension)] for _ in range(k)])

        while True:
            clustered_point_index = [[] for _ in range(k)]
            mean_error = 0

            # Divide the points into clusters
            for index, point in enumerate(self.normalized_data):
                norm_array = [[np.linalg.norm(point - cluster, ord=2), i] for i, cluster in enumerate(cluster_location)]
                nearest_cluster_index = min(norm_array, key=lambda x: x[0])[1]
                clustered_point_index[nearest_cluster_index].append([index])

            # Update the location of clusters
            for index in range(k):
                if clustered_point_index[index]:
                    clustered_point = np.array([self.normalized_data[point_i]
                                                for point_i in clustered_point_index[index]])
                    new_location = np.array([np.mean(vector) for vector in clustered_point.T]).T

                else:
                    new_location = np.array([random.random() for _ in range(self.data_dimension)])
                error = np.linalg.norm(new_location - cluster_location[index], ord=2)
                mean_error = average(mean_error, index, error)
                cluster_location[index] = new_location

            # Evaluate the location of clusters
            if mean_error < self.error_threshold:
                final_clustered_point_index = clustered_point_index
                break

        inertia_value = sum([sum([np.linalg.norm(self.normalized_data[point_i] - cluster, ord=2)
                                  for point_i in final_clustered_point_index[i]]) for i, cluster in
                             enumerate(cluster_location)])

        self.clustered_point_index[k] = final_clustered_point_index
        self.cluster_location[k] = cluster_location
        self.inertia_value[k] = inertia_value
        return final_clustered_point_index, cluster_location, inertia_value

    def repeat_K_mean_clustering(self, k):
        clustered_point_index = None
        cluster_location = None
        inertia_value = float('inf')
        for _ in range(self.repetition):
            clustering_result = self.k_mean_clustering_once(k)
            if clustering_result[2] < inertia_value:
                clustered_point_index = clustering_result[0]
                cluster_location = clustering_result[1]
                inertia_value = clustering_result[2]

        self.clustered_point_index[k] = clustered_point_index
        self.cluster_location[k] = cluster_location
        self.inertia_value[k] = inertia_value
        return clustered_point_index, cluster_location, inertia_value

    def find_elbow_k(self):
        inertia_value_list = list(self.inertia_value.values())
        d_inertia_value = [x2 - x1 for x1, x2 in zip(inertia_value_list[:-1], inertia_value_list[1:])]
        elbow_gradient = (inertia_value_list[-1] - inertia_value_list[0]) / self.k_max
        for i, gradient in enumerate(d_inertia_value):
            if gradient > elbow_gradient:
                return i + 1

    def plot_clustered_result(self, k, inertia_value_plot=True):
        if inertia_value_plot:
            plt.figure()
            plt.plot(self.inertia_value.keys(), self.inertia_value.values(), 'r-o')
            plt.plot(self.elbow_k, self.inertia_value[self.elbow_k], 'b-o')

        plt.figure()
        for cluster_i, clustered_point in enumerate(self.clustered_point_index[k]):
            points = np.array([self.data[point_i] for point_i in clustered_point])
            plt.scatter(*np.array(points).T[:2, :],
                        color=(1 / k * cluster_i, 1 - 1 / k * cluster_i, 1 / k * cluster_i),
                        s=10)
        plt.gca().invert_yaxis()

    def k_mean_clustering(self, plot=True):
        for k in range(1, self.k_max + 1):
            self.repeat_K_mean_clustering(k)
            print("Clustering Data - (%d/%d)k" % (k, self.k_max))

        self.elbow_k = self.find_elbow_k()

        if plot:
            self.plot_clustered_result(self.elbow_k)

        return self.clustered_point_index[self.elbow_k]


def main():
    data = np.array([[1, 0],
                     [1, 1],
                     [1, 2],
                     [1, 3],
                     [1, 4],
                     [1, 5],
                     [1, 6],
                     [1, 7],
                     [1, 8],
                     [1, 9],
                     [1, 10],
                     [2, 0],
                     [2, 1],
                     [2, 2],
                     [2, 3],
                     [2, 4],
                     [2, 5],
                     [2, 6],
                     [2, 7],
                     [2, 8],
                     [2, 9],
                     [2, 10],
                     [11, 10],
                     [11, 11],
                     [11, 12],
                     [11, 13],
                     [11, 14],
                     [11, 15],
                     [11, 16],
                     [11, 17],
                     [11, 18],
                     [11, 19],
                     [11, 20],
                     [12, 10],
                     [12, 11],
                     [12, 12],
                     [12, 13],
                     [12, 14],
                     [12, 15],
                     [12, 16],
                     [12, 17],
                     [12, 18],
                     [12, 19],
                     [12, 20],
                     [20, 0],
                     [18, 0],
                     [19, 0],
                     [20, 1],
                     [18, 1],
                     [19, 1],
                     [20, 2],
                     [18, 2],
                     [19, 2],
                     [18, 4],
                     [19, 4],
                     [20, 4],
                     [18, 5],
                     [19, 5],
                     [20, 5],
                     [18, 6],
                     [19, 6],
                     [20, 6],
                     [18, 7],
                     [19, 7],
                     [20, 7],
                     [20, 10],
                     [20, 11],
                     [20, 12],
                     [20, 13],
                     [20, 14],
                     [20, 15],
                     [20, 16],
                     [20, 17],
                     [20, 18],
                     [20, 19],
                     [20, 20],
                     [20, 10],
                     [20, 11],
                     [20, 12],
                     [20, 13],
                     [20, 14],
                     [20, 15],
                     [20, 16],
                     [20, 17],
                     [20, 18],
                     [20, 19],
                     [20, 20]])

    k_mean_clustering = KMeanClustering(data, error_threshold=0.1, k_max=20, repetition=5)
    k_mean_clustering.k_mean_clustering(plot=True)
    plt.show()


if __name__ == '__main__':
    main()
