import matplotlib as plt
import numpy as np
from PIL import Image

img_name_array = ['jaewon.jpg', 'jaewon.jpg']
img_array = [np.array(Image.open(x)) for x in img_name_array]


def get_grayscale(img):
    gray_pix = np.zeros((img.shape[0], img.shape[1]))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            gray_pix[y][x] = img[y][x].mean()
        print("Gray scaling %d(height)" % y)
    corner_detect_result = Image.fromarray(gray_pix)
    corner_detect_result.show()
    return gray_pix


def add_padding(img):
    pixel_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    pixel_padding[1:-1, 1:-1] = img
    return pixel_padding


def get_derivative(img):
    derivative = np.zeros((img.shape[0], img.shape[1], 2))
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            derivative[y][x] = [img[y + 1][x + 0] - img[y - 1][x + 0],
                                img[y + 0][x + 1] - img[y + 0][x - 1]]
        print("Calculating structure tensor of %d(height)" % y)
    corner_detect_result = Image.fromarray(derivative[:, :, 0])
    corner_detect_result.show()
    corner_detect_result = Image.fromarray(derivative[:, :, 1])
    corner_detect_result.show()
    return derivative


grayscale_image = get_grayscale(img_array[0])
padding_image = add_padding(grayscale_image)
derivative_image = get_derivative(padding_image)
