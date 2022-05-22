import matplotlib.pyplot as plt
import numpy as np
import random


def k_mean_clustering(data, k):
    data_range = [[min(data[:, 0]), max(data[:, 0])],
                  [min(data[:, 1]), max(data[:, 1])]]
    cluster_array = np.array([[random.uniform(data_range[0][0], data_range[0][1]),
                               random.uniform(data_range[1][0], data_range[1][1])] for _ in range(k)])

    for _ in range(100):
        clustered_point = [[] for _ in range(k)]

        # Divide the points into clusters
        for point in data:
            norm_array = [[np.linalg.norm(point - cluster, ord=2), i] for i, cluster in enumerate(cluster_array)]
            nearest_cluster = sorted(norm_array, key=lambda x: x[0])[0][1]
            clustered_point[nearest_cluster].append(point)

        # Update the location of clusters
        for index in range(k):
            cluster_array[index] = np.mean(clustered_point[index])

    return cluster_array


def main():
    data = np.array([[1, 1],
                     [1, 20],
                     [10, 2],
                     [20, 30],
                     [30, 30],
                     [14, 14],
                     [5, 7],
                     [1, 8],
                     [18, 30],
                     [20, 20],
                     [25, 25],
                     [24, 24],
                     [30, 10],
                     [10, 2],
                     [1, 10]])
    cluster = k_mean_clustering(data, 2)
    plt.plot(data[:, 0], data[:, 1], 'o')
    plt.plot(cluster[:, 0], cluster[:, 1], 'ro')
    plt.show()


if __name__ == '__main__':
    main()
