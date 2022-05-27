import matplotlib.pyplot as plt
import numpy as np
import random


def k_mean_clustering(data, k, error_threshold):
    def average(mean_before, n, a_n1):
        return (mean_before * n + a_n1) / (n + 1)

    data_range = [[min(data[:, i]), max(data[:, i])] for i in range(data.shape[1])]
    cluster_location = np.array([[random.uniform(*min_max) for min_max in data_range] for _ in range(k)])

    while True:
        clustered_point = [[] for _ in range(k)]
        mean_error = 0

        # Divide the points into clusters
        for point in data:
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
        if mean_error < error_threshold:
            final_clustered_point = clustered_point
            break

    inertia_value = sum([sum([np.linalg.norm(point - cluster, ord=2)
                              for point in final_clustered_point[i]]) for i, cluster in enumerate(cluster_location)])

    return final_clustered_point, cluster_location, inertia_value


def find_elbow_k(array):
    derivative_array = [x2 - x1 for x1, x2 in zip(array[:-1], array[1:])]
    elbow_gradient = (derivative_array[0] + derivative_array[-1]) / 2
    for i, gradient in enumerate(derivative_array):
        if gradient > elbow_gradient:
            return (i + 1) + 1


def main():
    error_threshold = 0.00001
    k_max = 20
    random_iterate = 10

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

    inertia_value_array = []

    for k in range(1, k_max):
        clustered_points = None
        cluster_location = None
        inertia_value = float('inf')
        for _ in range(random_iterate):
            clustering_result = k_mean_clustering(data, k, error_threshold)
            if clustering_result[2] < inertia_value:
                clustered_points = clustering_result[0]
                cluster_location = clustering_result[1]
                inertia_value = clustering_result[2]

        inertia_value_array.append(inertia_value)

        plt.figure(k)
        for i, points in enumerate(clustered_points):
            plt.scatter(*np.array(points).T, color=(1 / k * i, 1 - 1 / k * i, 1 / k * i))
        plt.scatter(*cluster_location.T, color='r')

    elbow_k = find_elbow_k(inertia_value_array)

    plt.figure(0)
    plt.plot(range(1, k_max), inertia_value_array, 'r-o')
    plt.plot(elbow_k, inertia_value_array[elbow_k - 1], 'b-o')

    plt.show()


if __name__ == '__main__':
    main()
