import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def down_scale(img, r):
    return img.resize((int(img.width / r), int(img.height / r)))


def get_grayscale(img):
    gray_img = np.zeros((img.shape[0], img.shape[1]))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            gray_img[y][x] = img[y][x].mean()
    return gray_img


def add_padding(img, padding_size):
    shape = list(img.shape)
    shape[0] += padding_size * 2
    shape[1] += padding_size * 2
    pixel_padding = np.zeros(shape)
    pixel_padding[padding_size:-padding_size, padding_size:-padding_size] = img
    return pixel_padding


def get_derivative(img):
    padding_img = add_padding(img, 1)
    derivative = np.zeros((img.shape[0], img.shape[1], 2))

    for y in range(1, padding_img.shape[0] - 1):
        for x in range(1, padding_img.shape[1] - 1):
            derivative[y - 1][x - 1] = [padding_img[y + 1][x + 0] - padding_img[y - 1][x + 0],
                                        padding_img[y + 0][x + 1] - padding_img[y + 0][x - 1]]
    return derivative


def window_slice(img, location, neighbor_distance):
    padding_img = add_padding(img, neighbor_distance)

    window = padding_img[location[0]: location[0] + 2 * neighbor_distance + 1,
                         location[1]: location[1] + 2 * neighbor_distance + 1]
    return window


# AX = B
def least_squares_approximation(A, B):
    if np.linalg.det(A.T @ A):
        return np.linalg.inv(A.T @ A) @ A.T @ B
    else:
        return np.array([[0], [0]])


def corner_Harris(img, neighbor_distance, threshold):
    corner_location = []

    grayscale_image = get_grayscale(img)
    yx_derivative_image = get_derivative(grayscale_image)

    for y in range(grayscale_image.shape[0]):
        for x in range(grayscale_image.shape[1]):
            window_yx_derivative = window_slice(yx_derivative_image, [y, x], neighbor_distance)
            vector_dy_dx = window_yx_derivative.reshape((2 * neighbor_distance + 1) ** 2, 2)
            structure_tensor = sum([np.array([[ix * ix, ix * iy],
                                              [iy * ix, iy * iy]]) for iy, ix in vector_dy_dx])

            if min(np.linalg.eigvals(structure_tensor)) > threshold:
                corner_location.append([y, x])

    return corner_location


class OpticalFlow:
    def __init__(self, img_array,
                 downscale=4, neighbor_distance=1,
                 corner_neighbor_distance=1, corner_threshold=5000):
        self.image_array = list(map(lambda x: np.array(down_scale(x, downscale)), img_array))
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
                                x, self.flow.shape[1] - 1 - y,
                                self.flow[frame_num][y][x][1], -self.flow[frame_num][y][x][0],
                                width=0.3,
                                edgecolor='deeppink',
                                facecolor='white'
                            ))
        plt.show()
        return 0


def get_image_name_from_video(video_name, frame):
    return ['%s/%dFrame/%s %03d.jpg' % (video_name, frame, video_name, x + 1) for x in range(frame)]


def get_image_array(img_name_array):
    return [Image.open(x) for x in img_name_array]


def main():
    video_name = 'Paris'
    frame = 100
    # img_name_array = get_image_name_from_video(video_name, frame)
    img_name_array = ['Paris/200Frame/Paris 180.jpg', 'Paris/200Frame/Paris 181.jpg']
    image_array = get_image_array(img_name_array)

    optical_flow = OpticalFlow(image_array,
                               downscale=4,
                               neighbor_distance=1,
                               corner_neighbor_distance=1,
                               corner_threshold=5000)
    optical_flow.calculate_optical_flow()
    optical_flow.display_flow()


if __name__ == '__main__':
    main()
