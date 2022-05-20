import numpy as np
from PIL import Image


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


def corner_Harris(img, neighbor_distance=1, threshold=1000):
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


if __name__ == '__main__':
    img_name = 'squares.jpg'
    image = np.array(Image.open(img_name))

    corner_result = corner_Harris(image, neighbor_distance=1, threshold=1000)
    corner_image = np.zeros((image.shape[0], image.shape[1]))

    for corner_y, corner_x in corner_result:
        corner_image[corner_y][corner_x] = 255

    Image.fromarray(corner_image).show()
