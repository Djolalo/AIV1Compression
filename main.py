import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import cv2 as cv

BLOCK_SIZE = 8

QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]).astype(float)


def dct2d(image):
    centreur = np.ones(image.shape) * 128
    array = np.array(image) - centreur

    array = dct(dct(array, axis=1, n=8, norm='ortho'), axis=0, n=8, norm='ortho')

    return array


def idct2d(image):
    centreur = np.ones(image.shape) * 128
    # on multiplie les coefs par la matrice de quantification
    image = image * QY
    image = dct(dct(image, type=3, axis=1, n=8, norm='ortho'), type=3, axis=0, n=8, norm='ortho') + centreur
    return image


def jpeg(img_name, alpha: int = 97, show=False):
    global QY
    image = cv.imread(img_name)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    res = np.zeros(image.shape).astype(float)
    if alpha > 50 and alpha <= 99:
        alpha = 2-0.02*float(alpha)
        QY = QY * alpha
    elif alpha >= 1:
        alpha = 50.0/float(alpha)
        QY = QY * alpha
    else:
        return None
    for i in range(0, len(image[0]), BLOCK_SIZE):
        for j in range(0, len(image), BLOCK_SIZE):
            res[i:(i + BLOCK_SIZE), j:(j + BLOCK_SIZE)] = np.round(
                dct2d(image[i:(i + BLOCK_SIZE), j:(j + BLOCK_SIZE)]) / QY)
    cv.imwrite(os.path.splitext(os.path.basename(img_name))[0] + "dct2d.tiff", res)
    for i in range(0, len(image[0]), BLOCK_SIZE):
        for j in range(0, len(image), BLOCK_SIZE):
            res[i:(i + BLOCK_SIZE), j:(j + BLOCK_SIZE)] = idct2d(res[i:(i + BLOCK_SIZE), j:(j + BLOCK_SIZE)])
    if show:
        plt.imshow(res, cmap='gray', label="dct de l'image")
        plt.show()
    return res


def error_measures(I, Iest, toPrint: bool = False):
    mae = np.sum(np.divide(np.abs(np.subtract(I, Iest)), Iest.shape[0] * Iest.shape[1]))
    mse = np.sum(np.divide(np.square(np.subtract(I, Iest)), Iest.shape[0] * Iest.shape[1]))
    psnr = 10 * np.log10((255 ** 2) / mse)
    if toPrint:
        print("mse= ", mse, " mae= ", mae, " psnr= ", psnr)
    return mae, mse, psnr


if __name__ == '__main__':
    base_image = cv.imread("lena_8bits.png")
    base_image = cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)
    base_image = np.array(base_image)
    decompre = jpeg("lena_8bits.png",97)
    error_measures(base_image, decompre, True)
