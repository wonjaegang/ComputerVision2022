import matplotlib.pyplot as plt
import matplotlib.patches as patches
from image_functions import *


class OpticalFlow:
    def __init__(self, img_array, neighbor_distance=1,
                 corner_neighbor_distance=1, corner_threshold=5000):
        self.image_array = img_array
        self.neighbor_distance = neighbor_distance
        self.corner_neighbor_distance = corner_neighbor_distance
        self.corner_threshold = corner_threshold

        self.flow = np.full((len(self.image_array) - 1, self.image_array[0].shape[0], self.image_array[0].shape[1], 2),
                            None)

    def calculate_optical_flow(self):
        grayscale_image_next = get_grayscale(self.image_array[-1])

        for index, image in enumerate(self.image_array[-2::-1]):
            # Gray Scaling & Add Padding
            grayscale_image = get_grayscale(image)
            yx_derivative_image = get_derivative(grayscale_image)
            t_derivative_image = grayscale_image_next - grayscale_image

            # # For each pixel
            # for y in range(grayscale_image.shape[0]):
            #     for x in range(grayscale_image.shape[1]):

            # For each corner
            corner = corner_Harris(grayscale_image, self.corner_neighbor_distance, self.corner_threshold)
            for y, x in corner:
                # Window Slicing & Calculate Derivative
                window_yx_derivative = window_slice(yx_derivative_image, [y, x], self.neighbor_distance)
                window_t_derivative = window_slice(t_derivative_image, [y, x], self.neighbor_distance)

                # Approximating for Optimized result
                window_size = (2 * self.neighbor_distance + 1) ** 2
                vector_dy_dx = window_yx_derivative.reshape(window_size, 2)
                vector_dt = window_t_derivative.reshape(window_size, 1)
                optimized_v = least_squares_approximation(vector_dy_dx, -vector_dt)

                # Save optimized (v, u) at flow matrix
                self.flow[-index][y][x] = optimized_v.reshape(2)

            # Save current pixel values for next loop
            grayscale_image_next = grayscale_image
            print("Calculating Optical Flow - (%d/%d)Frame" % (index + 2, len(self.image_array)))

        return self.flow

    def display_flow(self):
        plt.style.use('default')
        fig, ax = plt.subplots()
        plt.axis([0, self.flow.shape[2], 0, self.flow.shape[1]])
        for frame_num in range(self.flow.shape[0]):
            for y in range(self.flow.shape[1]):
                for x in range(self.flow.shape[2]):
                    if self.flow[frame_num][y][x][0] is not None:
                        ax.add_patch(
                            patches.Arrow(
                                x, y,
                                self.flow[frame_num][y][x][1], self.flow[frame_num][y][x][0],
                                width=0.3,
                                edgecolor='deeppink',
                                facecolor='white'
                            ))
        return 0


def main():
    video_name = 'Paris'
    frame = 100
    # img_name_array = get_image_name_from_video(video_name, frame)
    img_name_array = ['Paris/200Frame/Paris 180.jpg', 'Paris/200Frame/Paris 181.jpg']
    image_array = get_image_array(img_name_array, downscale=4)

    optical_flow = OpticalFlow(image_array,
                               neighbor_distance=1,
                               corner_neighbor_distance=1,
                               corner_threshold=5000)
    optical_flow.calculate_optical_flow()
    optical_flow.display_flow()
    plt.imshow(get_grayscale(image_array[0]), cmap='gray', origin='upper')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
