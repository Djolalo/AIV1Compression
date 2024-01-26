import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import cv2 as cv

BLOCK_SIZE = 8


def dct2d(image):
    centreur = np.ones(image.shape) * 128
    array = np.array(image) - centreur

    array = dct(dct(array, axis=1, n=8, norm='ortho'), axis=0, n=8, norm='ortho')

    return array


def jpeg(img_name, show=False):
    image = cv.imread(img_name)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    res = np.zeros(image.shape).astype(float)
    for i in range(0, len(image[0]), BLOCK_SIZE):
        for j in range(0, len(image), BLOCK_SIZE):
            res[i:(i + BLOCK_SIZE), j:(j + BLOCK_SIZE)] = dct2d(image[i:(i + BLOCK_SIZE), j:(j + BLOCK_SIZE)])
    cv.imwrite(os.path.splitext(os.path.basename(img_name))[0]+"dct2d.tiff", res)
    if show:
        plt.imshow(res, cmap='gray', label="dct de l'image")
        plt.show()


if __name__ == '__main__':
    jpeg("lena_8bits.png", True)
