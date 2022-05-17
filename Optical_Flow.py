import matplotlib as plt
import numpy as np
from PIL import Image

img_name_array = ['jaewon.jpg', 'jaewon.jpg']
img_array = [np.array(Image.open(x)) for x in img_name_array]


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


def window_slice(img, location, half_scale):
    padding_img = add_padding(img, half_scale)

    window = padding_img[location[0]: location[0] + 2 * half_scale + 1,
                         location[1]: location[1] + 2 * half_scale + 1]
    return window


# AX = B
def least_squares_approximation(A, B):
    return np.linalg.inv(A.T @ A) @ A.T @ B


# AX = B
def particle_swarm_approximation(A, B):
    return 0


def main():
    window_scale = 3
    half_scale = int((window_scale - 1) / 2)

    grayscale_image_next = get_grayscale(img_array[-1])

    for image in img_array[-2::-1]:
        # Gray Scaling & Add Padding
        grayscale_image = get_grayscale(image)
        xy_derivative_image = get_derivative(grayscale_image)
        t_derivative_image = grayscale_image_next - grayscale_image

        for y in range(grayscale_image.shape[0]):
            for x in range(grayscale_image.shape[1]):
                # Window Slicing & Calculate Derivative
                window_xy_derivative = window_slice(xy_derivative_image, [y, x], half_scale)
                window_t_derivative = window_slice(t_derivative_image, [y, x], half_scale)

                # Approximating for Optimized result
                vector_dy_dx = window_xy_derivative.reshape(window_scale ** 2, 2)
                vector_dt = window_t_derivative.reshape(window_scale ** 2, 1)
                print(vector_dy_dx)
                print(vector_dt)

                optimized_v = least_squares_approximation(vector_dy_dx, -vector_dt)
                print("location y: %d, x: %d optical flow:" % (y, x), optimized_v[0][0], optimized_v[1][0])

        # Save current pixel values for next loop
        grayscale_image_next = grayscale_image


if __name__ == '__main__':
    main()
