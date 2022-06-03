from Optical_Flow import *
from k_mean_clustering import *
from mean_shift_clustering import *


def main():
    img_name_array = ['Paris/200Frame/Paris 180.jpg', 'Paris/200Frame/Paris 181.jpg']
    image_array = get_image_array(img_name_array, downscale=4)

    optical_flow = OpticalFlow(image_array,
                               neighbor_distance=2,
                               corner_neighbor_distance=1,
                               corner_threshold=1000)
    optical_flow.calculate_optical_flow()
    optical_flow.display_flow()

    data = optical_flow.flow[0]
    a = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            a.append([x, y,
                      0.299 * image_array[0][y][x][0] + 0.587 * image_array[0][y][x][1] + 0.114 * image_array[0][y][x][2],
                      *data[y][x]])

    k_mean_clustering = KMeanClustering(np.array(a), error_threshold=0.1, k_max=10, repetition=1)
    #
    # k_mean_clustering.k_mean_clustering(plot=True)
    k = 11
    k_mean_clustering.repeat_K_mean_clustering(k)
    k_mean_clustering.plot_clustered_result(k, inertia_value_plot=False)
    #
    plt.imshow(image_array[0])
    plt.show()

    # mean_shift_clustering = MeanShiftClustering(np.array(a), error_threshold=0.05, window_radius=0.3)
    # mean_shift_clustering.mean_shift_clustering()
    # mean_shift_clustering.plot_clustered_result()
    #
    # plt.imshow(image_array[0])
    # plt.show()


if __name__ == "__main__":
    main()
