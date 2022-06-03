import matplotlib.pyplot as plt
import numpy as np


class MeanShiftClustering:
    def __init__(self, data, error_threshold=0.01, window_radius=0.5):
        self.data = data
        self.data_dimension = data.shape[1]
        self.data_range = self.get_data_range()
        self.normalized_data = self.data_normalization()

        self.error_threshold = error_threshold
        self.window_radius = window_radius

        self.shifted_mean = np.zeros(data.shape)
        self.clustered_index = []

    def get_data_range(self):
        return [[min(self.data[:, i]), max(self.data[:, i])] for i in range(self.data_dimension)]

    def data_normalization(self):
        normalized_data = [(x - self.data_range[i][0]) / (self.data_range[i][1] - self.data_range[i][0])
                           for i, x in enumerate(self.data.T)]
        return np.array(normalized_data).T

    def calculate_neighbor_mean(self, location):
        neighbor = [location]
        for point in self.normalized_data:
            if np.linalg.norm(location - point, ord=2) < self.window_radius:
                neighbor.append(point)
        return np.array([np.mean(vector) for vector in np.array(neighbor).T]).T

    def shift_mean(self, point):
        mean_vector = point
        while True:
            updated_mean_vector = self.calculate_neighbor_mean(mean_vector)
            error = np.linalg.norm(mean_vector - updated_mean_vector, ord=2)
            mean_vector = updated_mean_vector
            if error < self.error_threshold:
                return updated_mean_vector

    def plot_clustered_result(self):
        plt.figure()
        plt.gca().invert_yaxis()
        plt.scatter(*np.array(self.normalized_data).T[:2, :],
                    color='b',
                    s=10)
        plt.scatter(*np.array(self.shifted_mean).T[:2, :],
                    color='r',
                    s=5)

        plt.figure()
        plt.gca().invert_yaxis()
        k = len(self.clustered_index)
        print(k)
        for cluster_i, indexes in enumerate(self.clustered_index):
            points = np.array([self.data[index] for index in indexes])
            plt.scatter(*points.T[:2, :],
                        color=(1 / k * cluster_i, 1 - 1 / k * cluster_i, 1 / k * cluster_i),
                        s=10)

    def mean_shift_clustering(self):
        for point_i, point in enumerate(self.normalized_data):
            self.shifted_mean[point_i] = self.shift_mean(point)
            print("Shifting Mean - (%d/%d)data" % (point_i + 1, self.data.shape[0]))

        shifted_mean_with_index = [[i, mean_vector] for i, mean_vector in enumerate(self.shifted_mean)]
        while shifted_mean_with_index:
            indexes = []
            for index, mean_vector in shifted_mean_with_index:
                if np.linalg.norm(mean_vector - shifted_mean_with_index[-1][1], ord=2) < self.error_threshold:
                    indexes.append(index)
            shifted_mean_with_index = [x for x in shifted_mean_with_index if x[0] not in indexes]
            self.clustered_index.append(indexes)


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

    mean_shift_clustering = MeanShiftClustering(data, error_threshold=0.1, window_radius=0.3)
    mean_shift_clustering.mean_shift_clustering()
    mean_shift_clustering.plot_clustered_result()

    plt.show()


if __name__ == '__main__':
    main()
