from image_functions import *
from Optical_Flow import *
from k_mean_clustering import *


def main():
    img_name_array = ['Paris/400Frame/Paris 180.jpg', 'Paris/400Frame/Paris 181.jpg']
    image_array = get_image_array(img_name_array, downscale=2)

    optical_flow = OpticalFlow(image_array,
                               neighbor_distance=1,
                               corner_neighbor_distance=1,
                               corner_threshold=5000)
    optical_flow.calculate_optical_flow()
    optical_flow.display_flow()

    data = optical_flow.flow[0]
    a = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y][x][0] is not None:
                if np.linalg.norm(data[y][x], ord=2):
                    a.append([x, y, data[y][x][1], data[y][x][0]])

    k_mean_clustering = KMeanClustering(np.array(a), error_threshold=0.001, k_max=10, repetition=5)

    # k_mean_clustering.k_mean_clustering(plot=True)
    k = 15
    k_mean_clustering.repeat_K_mean_clustering(k)
    k_mean_clustering.plot_clustered_result(k, inertia_value_plot=False)

    plt.imshow(image_array[0])
    plt.show()


if __name__ == "__main__":
    main()
