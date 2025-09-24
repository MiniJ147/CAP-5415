# Programming Assignment 1
# Canny Edge Detection Implementation

# Tasks
# use 3 examples form Berkeley Segmentation Dataset
# show the result of the edge detection from 3 different sigma values

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_gaussian_values(x, sigma):
    eulers_number = np.exp((-0.5) * ((x**2)/(sigma**2))) # -x^2/(2[sigma]^2)
    fraction = 1 / (np.sqrt(2 * np.pi * (sigma**2)))

    return fraction * eulers_number

def get_gaussian_derivative_values(x, sigma):
    scalar = (-x)/(sigma**2)
    return scalar * get_gaussian_values(x,sigma)

def get_gaussian_kernel(size, sigma):
    kernel = []

    for i in range(-1 * (size//2), (size//2)+1):
        val = get_gaussian_values(x=i,sigma=sigma)
        kernel.append(val)

    return np.array(kernel)

def get_gaussian_1der_kernel(size, sigma):
    kernel = []

    for i in range(-1 * (size//2), (size//2)+1):
        val = get_gaussian_derivative_values(x=i,sigma=sigma)
        kernel.append(val)

    return np.array(kernel)

def convolution_1d(signal, kernel):
    result = []

    for i in range(len(signal) - len(kernel) + 1):
        val = sum(signal[i:i+len(kernel)] * kernel)
        result.append(val)

    return np.array(result)

def conv_x(img, kernel):
    result = []

    for i in range(img.shape[0]):
        result.append(convolution_1d(img[i, :],kernel))
    
    return np.array(result)

def main():
    img_path = "./tests/test-3.png"

    # #read image
    img = Image.open(img_path).convert("L")

    # convert into matrix
    I = np.array(img)
    # I.shape -> (height, width, channels)

    # G = get_gaussian_kernel(size=5,sigma=1)

    # I_cx = conv_x(I, G)
    # I_cx_uint8 = I_cx.astype(np.uint8)
    # img_cx = Image.fromarray(I_cx_uint8)
    # plt.imshow(img_cx, cmap="gray")
    # plt.show()


    G_1der = get_gaussian_1der_kernel(size=5,sigma=2)

    # I_dx = conv_x(I,G_1der)

    # print(I_dx)
    print(G_1der)
    I_cx = conv_x(I, G_1der)
    I_cx_uint8 = I_cx.astype(np.uint8)
    img_cx = Image.fromarray(I_cx_uint8)
    plt.imshow(img_cx)
    plt.show()



   


if __name__ == "__main__":
    main()