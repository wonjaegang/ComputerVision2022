import matplotlib as plt
import numpy as np
from PIL import Image

img_name = 'CheckerBoard.jpg'
img = Image.open(img_name)

# Make grayscale pixel array
pixel = np.array(img)
gray_pix = np.full((img.size[1], img.size[0]), 0)

for y in range(img.size[1]):
    for x in range(img.size[0]):
        gray_pix[y][x] = pixel[y][x].mean()
    print("Gray scaling %d(height)" % y)

# Make padding array
pixel_padding = np.full((img.size[1] + 2, img.size[0] + 2), 0)
pixel_padding[1:-1, 1:-1] = gray_pix

# Calculate gradient & structure tensor
H = np.full((img.size[1], img.size[0], 2, 2), 0)
for y in range(img.size[1]):
    for x in range(img.size[0]):
        gradient = [pixel_padding[y + 1][x + 2] - pixel_padding[y + 1][x + 0],
                    pixel_padding[y + 2][x + 1] - pixel_padding[y + 0][x + 1]]
        H[y][x] = [[gradient[0] * gradient[0], gradient[0] * gradient[1]],
                   [gradient[1] * gradient[0], gradient[1] * gradient[1]]]
    print("Calculating structure tensor of %d(height)" % y)

# Calculate eigen Value of mean - H
H_eigenValue = np.full((img.size[1] - 2, img.size[0] - 2, 2), 0)
for y in range(img.size[1] - 2):
    for x in range(img.size[0] - 2):
        H_mean = (H[y + 0][x + 0] + H[y + 0][x + 1] + H[y + 0][x + 2] +
                  H[y + 1][x + 0] + H[y + 1][x + 1] + H[y + 1][x + 2] +
                  H[y + 2][x + 0] + H[y + 2][x + 1] + H[y + 2][x + 2]) / 9

        H_eigenValue[y][x] = np.linalg.eig(np.array(H_mean))[0]
    print("Calculating eigen Value of %d(height)" % y)

# Detect edge
edge_result_pixel = np.full((img.size[1] - 2, img.size[0] - 2), 0)
for y in range(img.size[1] - 2):
    for x in range(img.size[0] - 2):
        if max(H_eigenValue[y][x]) > 1000:
            edge_result_pixel[y][x] = 255
    print("Detecting edge of %d(height)" % y)

edge_detect_result = Image.fromarray(edge_result_pixel)
edge_detect_result.show()

# Detect corner
corner_result_pixel = np.full((img.size[1] - 2, img.size[0] - 2), 0)
for y in range(img.size[1] - 2):
    for x in range(img.size[0] - 2):
        if min(H_eigenValue[y][x]) > 1000:
            corner_result_pixel[y][x] = 255
    print("Detecting corner of %d(height)" % y)

corner_detect_result = Image.fromarray(corner_result_pixel)
corner_detect_result.show()
