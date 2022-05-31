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

        self.memoization_dict = {}
        self.shifted_mean = np.zeros(data.shape)

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
        # if neighbor in self.memoization_dict:
        #     return self.memoization_dict[neighbor]
        # else:
        #     self.memoization_dict[neighbor] = np.array([np.mean(vector) for vector in np.array(neighbor).T]).T
        #     return self.memoization_dict[neighbor]

        return np.array([np.mean(vector) for vector in np.array(neighbor).T]).T

    def shift_mean(self, point):
        mean_vector = point
        while True:
            updated_mean_vector = self.calculate_neighbor_mean(mean_vector)
            print(mean_vector, updated_mean_vector)
            error = np.linalg.norm(mean_vector - updated_mean_vector, ord=2)
            mean_vector = updated_mean_vector
            if error < self.error_threshold:
                return updated_mean_vector

    def plot_clustered_result(self):
        plt.figure()
        plt.scatter(*np.array(self.normalized_data).T[:2, :],
                    color='b',
                    s=50)
        plt.scatter(*np.array(self.shifted_mean).T[:2, :],
                    color='r',
                    s=5)
        plt.gca().invert_yaxis()

    def mean_shift_clustering(self):
        for point_i, point in enumerate(self.normalized_data):
            self.shifted_mean[point_i] = self.shift_mean(point)
            print("Shifting Mean - (%d/%d)data" % (point_i + 1, self.data.shape[0]))

        self.plot_clustered_result()
        plt.show()

        # criterion_dict = {}
        # for mean_vector in self.shifted_mean:
        #     for criterion in criterion_dict:
        #         if np.linalg.norm(mean_vector - criterion, ord=2) < self.error_threshold:
        #             criterion_dict[criterion]


def main():
    data = np.array([[x, y] for x, y in zip(list(range(10)), list(range(10)))])
    print(data)
    mean_shift_clustering = MeanShiftClustering(data, error_threshold=0.01, window_radius=0.1)
    mean_shift_clustering.mean_shift_clustering()


if __name__ == '__main__':
    main()
