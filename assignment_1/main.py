# Programming Assignment 1
# Canny Edge Detection Implementation

# Tasks
# use 3 examples form Berkeley Segmentation Dataset
# show the result of the edge detection from 3 different sigma values

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy import ndimage


TEST_DIR = "./tests"
OUTPUT_DIR = "./figs"

# (size, sigma)
SIZE = 5
SIGMAS = [1,2,3]

def get_gaussian_values(x, sigma):
    eulers_number = np.exp((-0.5) * ((x**2)/(sigma**2))) # -x^2/(2[sigma]^2)
    fraction = 1 / (np.sqrt(2 * np.pi * (sigma**2)))

    return fraction * eulers_number

def get_gaussian_derivative_values(x, sigma):
    scalar = (-x)/(sigma**2)
    return scalar * get_gaussian_values(x,sigma)

def get_gaussian_kernel(size, sigma):
    """
    return the guassin kernel for our given size
    """
    kernel = []

    for i in range(-1 * (size//2), (size//2)+1):
        val = get_gaussian_values(x=i,sigma=sigma)
        kernel.append(val)

    return np.array(kernel)

def get_gaussian_1der_kernel(size, sigma):
    """
    return the first derrative guassian kernel
    """
    kernel = []

    for i in range(-1 * (size//2), (size//2)+1):
        val = get_gaussian_derivative_values(x=i,sigma=sigma)
        kernel.append(val)

    return np.array(kernel)


def convolution_1d(signal, kernel):
    """
    convolute on a 1d array (helper function)
    """
    result = []

    for i in range(len(signal) - len(kernel) + 1):
        val = sum(signal[i:i+len(kernel)] * kernel)
        result.append(val)

    return np.array(result)

def conv_x(img, kernel):
    """
    Apply convolution in the x direction for the given kernel
    """
    result = []

    for i in range(img.shape[0]):
        result.append(convolution_1d(img[i, :],kernel))
    
    return np.array(result)

def conv_y(img, kernel):
    """
    apply convolution in the y direction for a given kernel
    """
    img_t = img.T 
    conv_t = []

    for i in range(img_t.shape[0]):
        conv_t.append(convolution_1d(img_t[i, :], kernel))

    return np.array(conv_t).T

def gradient_magnitude(D_X, D_Y):
    """
    applies gradient magnitude for all points in image
    """
    # find smallest common shape
    min_rows = min(D_X.shape[0], D_Y.shape[0])
    min_cols = min(D_X.shape[1], D_Y.shape[1])
    
    D_X = D_X[:min_rows, :min_cols]
    D_Y = D_Y[:min_rows, :min_cols]

    return np.sqrt(D_X**2 + D_Y**2)

def calculate_normal_direction_angle(D_X,D_Y):
    """
    gets normal direction for all data points in image
    """
    min_rows = min(D_X.shape[0], D_Y.shape[0])
    min_cols = min(D_X.shape[1], D_Y.shape[1])
    
    D_X = D_X[:min_rows, :min_cols]
    D_Y = D_Y[:min_rows, :min_cols]

    return np.arctan2(D_Y,D_X) % 180

def non_maximum_suppression(M, angles): 
    """
    returns image with non maximum suppression
    """
    # returns [r,q] --> (r = (x,y), q = (x,y))
    # helper function to get directions via angles
    def get_direction(angle):
        if (0 <= angle < 22.5) or (157.5 <= angle < 180): direction = np.array([[-1,0],[1,0]])
        elif (22.5 <= angle < 67.5): direction = direction = np.array([[-1,-1],[1,1]])
        elif (67.5 <= angle < 112.5): direction = np.array([[0,-1],[0,1]])
        else: direction = np.array([[-1,1],[1,-1]])

        return direction

    H, W = M.shape
    result = np.zeros((H,W),dtype=np.float32)

    # iterate through pixles 
    for y in range(1,H-1):
        for x in range(1,W-1):
            q, r = get_direction(angle=angles[y][x])
            q = [x,y] - q
            r = [x,y] - r

            # get local max
            mx = max(M[q[1],q[0]], M[r[1],r[0]], M[y][x])        

            # if local max write to our result iamge
            if M[y][x] == mx:
                result[y][x] = M[y][x]
    return result

def show_img(I, size, img_name, sigma, caption):
    """
    Logic to save image to disk
    """
    I_uint8 = I.astype(np.uint8)
    img = Image.fromarray(I_uint8)
    caption = caption.replace(' ','-')

    dump_dir = f"{OUTPUT_DIR}/{img_name}/sigma-{sigma}/size-{size}"

    os.makedirs(dump_dir,exist_ok=True)
    plt.title(f"{caption} sigma = {sigma} | size = {size}")

    plt.imsave(f"{dump_dir}/{caption}.png",img,cmap="gray")

    plt.close()

def hysteresis_thresholding(grad_mag, low_thresh, high_thresh):
    # Strong edges: pixels above high threshold
    strong = grad_mag >= high_thresh

    # Weak edges: between thresholds
    weak = (grad_mag >= low_thresh) & (grad_mag < high_thresh)

    # Combine strong + weak into a candidate mask
    candidate_mask = strong | weak

    # Label connected components in the candidate mask
    labeled, _ = ndimage.label(candidate_mask, structure=np.ones((3, 3)))

    # Find labels that actually contain strong edges
    strong_labels = np.unique(labeled[strong])

    # Keep only components connected to strong edges
    out = np.isin(labeled, strong_labels)

    return out.astype(np.uint8)  # binary edge map

def cany_edge(img_path, size, sigma):
    img_name = img_path.split("/")[-1].split(".png")[0]

    # #read image
    img = Image.open(img_path).convert("L")

    # convert into matrix
    I = np.array(img)

    G = get_gaussian_kernel(size=size,sigma=sigma)

    # smoothing the image out with the Gaussian distribution 
    I_smooth = conv_x(conv_y(I,G),G)

    # Smoothed Image
    show_img(
        I=I_smooth,
        size=size,
        sigma=sigma,
        caption=f"Smoothed Image",
        img_name=img_name)

    # Gaussian Derative
    G_D = get_gaussian_1der_kernel(size=size,sigma=sigma)

    # Calculate both directions
    I_D_X = conv_x(I_smooth, G_D)
    I_D_Y = conv_y(I_smooth, G_D)

    # Get our Gradient Magntiude with our calcluated directions
    I_GRAD_MAG = gradient_magnitude(I_D_X, I_D_Y)

    # Show image
    show_img(
        I=I_GRAD_MAG,
        size=size,
        sigma=sigma,
        caption=f"Gradient",
        img_name=img_name)

    # get non maximum suppresion
    I_NMS = non_maximum_suppression(
        M = I_GRAD_MAG, 
        angles = calculate_normal_direction_angle(I_D_X, I_D_Y))

    # show image
    show_img(
        I=I_NMS,
        size=size,
        sigma=sigma,
        caption=f"Non_Maxium_Suppression",
        img_name=img_name)


    # final step - apply hysteresis thresholding
    high = np.percentile(I_NMS, 90)
    low = 0.9 * high
    org_shape = I_NMS.shape
    mask = hysteresis_thresholding(I_NMS,0,1e9)

    # show image
    show_img(
        I=I_NMS*mask,
        size=size,
        sigma=sigma,
        caption=f"hysteresis_thresholding",
        img_name=img_name)

def main():
    # grab all test images
    imgs = os.listdir(TEST_DIR)

    # apply cany_edge detection on all images and sigma combinations
    for img in imgs:
        path = f"{TEST_DIR}/{img}"
        
        for sigma in SIGMAS: 
            cany_edge(
                img_path=path,
                size=SIZE,
                sigma=sigma)

if __name__ == "__main__":
    main()