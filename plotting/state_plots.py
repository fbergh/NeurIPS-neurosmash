### IMPORTS ###

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import mxnet.ndarray as nd


### FUNCTIONS FOR PLOTTING STATE IMAGES ###

def plot_img(img):
    if type(img) == nd.ndarray.NDArray:
        img = img.asnumpy()
    plt.figure()
    plt.imshow(img)
    plt.show()

def plot_separate_channels(img, channels=[0, 1, 2]):
    if type(img) == nd.ndarray.NDArray:
        img = img.asnumpy()
    for c in channels:
        plot_img(img[:,:,c])

def plot_two_channels(img, remove_channel):
    """ Plot an RG, RB, or BG image """
    img_copy = img.copy()
    if type(img_copy) == nd.ndarray.NDArray:
        img_copy = img_copy.asnumpy()
    img_copy[:, :, remove_channel] = 0
    plot_img(img_copy)

def increase_contrast(img):
    """ Increase contrast by rescaling the min and max value of the image to 0 and 255, respectively """
    img_copy = img.copy()
    if type(img_copy) == nd.ndarray.NDArray:
        img_copy = img_copy.asnumpy()
    min_values = [img_copy[:, :, 0].min(), img_copy[:, :, 1].min(), img_copy[:, :, 2].min()]
    max_values = [img_copy[:, :, 0].max(), img_copy[:, :, 1].max(), img_copy[:, :, 2].max()]
    for channel, (min_val, max_val) in enumerate(zip(min_values, max_values)):
        img_copy[:, :, channel] = (img_copy[:, :, channel] - min_val) / (max_val - min_val) * 255
    return img_copy

def gaussian_noise(img, sigma=1):
    """ Plot image with Gaussian noise """
    img_copy = img.copy()
    if type(img_copy) == nd.ndarray.NDArray:
        img_copy = img_copy.asnumpy()
    img_copy = ndimage.filters.gaussian_filter(img_copy, sigma, mode='nearest')
    return img_copy