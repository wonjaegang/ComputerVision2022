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
    # dx = 0 or dy = 0 일때를 따로 고려하자.
    else:
        return np.array([[0], [0]])


# AX = B
def particle_swarm_approximation(A, B):
    return 0


def optical_flow(img_array, neighbor_distance=1):
    flow = np.zeros((len(img_array) - 1, img_array[0].shape[0], img_array[0].shape[1], 2))

    grayscale_image_next = get_grayscale(img_array[-1])

    for index, image in enumerate(img_array[-2::-1]):
        # Gray Scaling & Add Padding
        grayscale_image = get_grayscale(image)
        xy_derivative_image = get_derivative(grayscale_image)
        t_derivative_image = grayscale_image_next - grayscale_image

        # For each pixel
        for y in range(grayscale_image.shape[0]):
            for x in range(grayscale_image.shape[1]):
                # Window Slicing & Calculate Derivative
                window_xy_derivative = window_slice(xy_derivative_image, [y, x], neighbor_distance)
                window_t_derivative = window_slice(t_derivative_image, [y, x], neighbor_distance)

                # Approximating for Optimized result
                vector_dy_dx = window_xy_derivative.reshape((2 * neighbor_distance + 1) ** 2, 2)
                vector_dt = window_t_derivative.reshape((2 * neighbor_distance + 1) ** 2, 1)
                optimized_v = least_squares_approximation(vector_dy_dx, -vector_dt)

                # Save optimized (v, u) at flow matrix
                flow[-index][y][x] = optimized_v.reshape(2)
                print("location y: %d, x: %d optical flow:" % (y, x),
                      round(optimized_v[0][0], 2),
                      round(optimized_v[1][0], 2))

        # Save current pixel values for next loop
        grayscale_image_next = grayscale_image

    return flow


if __name__ == '__main__':
    # 추후에 downscale 은 따로 빼자
    img_name_array = ['jaewon.jpg', 'jaewon_after.jpg']
    image_array = [np.array(down_scale(Image.open(x), 4)) for x in img_name_array]

    k = optical_flow(image_array, neighbor_distance=1)

    plt.style.use('default')

    fig, ax = plt.subplots()
    plt.axis([0, k.shape[2], 0, k.shape[1]])

    for y in range(k.shape[1]):
        for x in range(k.shape[2]):
            ax.add_patch(
                patches.Arrow(
                    x, k.shape[1] - 1 - y,
                    k[0][y][x][1], -k[0][y][x][0],
                    width=0.3,
                    edgecolor='deeppink',
                    facecolor='lightgray'
                ))
    plt.show()
