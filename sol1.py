import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave as imsave, imread as imread
from skimage.color import rgb2gray

GRAY_REPR = 1
RGB_REPR = 2
RGB_SHAPE = 3
NUM_GRAY_COLORS = 255
Y_CHANNEL = 0
NUM_BINS = 256
INITIAL_Z_MAX =256.0
TWO = 2

# The matrix which is used in rgb2yiq conversion
mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])

# The matrix which is used in yiq2rgb conversion
mat_t = np.linalg.inv(mat).T

def read_image(filename, representation):
    """
    read the image and convert it to gray scale if necessary
    :param filename: the image
    :param representation: gray scale or RGB format to which the image is to be converted
    :return: A image whose values are normalized and in the given representation
    """
    if( representation < GRAY_REPR or representation > RGB_REPR):
        return -1
    #read the image
    img = imread(filename)

    #convert the RGB image to gray scale image
    if len(img.shape) == RGB_SHAPE and representation == GRAY_REPR:
        return rgb2gray(img)
    return np.divide(img.astype(np.float64), NUM_GRAY_COLORS)


def display_image(filename, representation):
    """
    read and display a image
    :param filename: a string that contains the name of the image
    :param representation: in which format display the image (gray scale pr RGB)
    :return: if fails return -1 otherwise return 0
    """
    if representation < GRAY_REPR or representation > RGB_REPR:
        return -1
    #read and convert the format of the image according to the representation
    img = read_image(filename, representation)

    #display the image
    if representation == GRAY_REPR:
        plt.imshow(img, cmap = plt.cm.gray)
    else:
        plt.imshow(img)
    plt.show()
    return 0

def rgb2yiq(imRGB):
    '''
    convert the image format from RGB to YIQ
    :param imRGB: The image as 3-D matrix whose values are of RGB
    :return: An image as 3-D matrix whose values are of YIQ
    '''
    imYIQ = np.dot(imRGB, mat.T.copy())
    return imYIQ

def yiq2rgb(imYIQ):
    '''
    convert the image format from YIQ to RGB
    :param imRGB: The image as 3-D matrix whose values are of YIQ
    :return: An image as 3-D matrix whose values are of RGB
    '''
    imRGB = np.dot(imYIQ,mat_t.copy())
    return imRGB


def get_hist_orig(im_gray,flag):
    '''
    Get the histogram of the origin image
    :param im_gray: The gray channel or the y channel
    :param flag: true if the origin image is in YIQ format else it's in gray scale
    :return: return the histogram of the origin image
    '''
    im_gray = (im_gray * NUM_GRAY_COLORS).round()

    if flag:
        gray_channel = np.asarray(im_gray[:, :, Y_CHANNEL]).flatten()
    else:
        gray_channel = np.asarray(im_gray).flatten()
    hist_orig, bins = np.histogram(gray_channel, NUM_BINS,[0,NUM_BINS])
    return hist_orig, bins


def equalize_image(hist_orig, im_gray, bins):
    '''
    Equalize the histogram and produce the equalized image
    :param hist_orig: origin image's histogram
    :param im_gray: The gray channel/Y channel of the origin image
    :return: The equalized histogram and the equalized image
    '''
    # get the cumulative distribution function
    cumulative_hist = hist_orig.cumsum()
    # normalize
    cumulative_hist = ((cumulative_hist / cumulative_hist[NUM_GRAY_COLORS]) * NUM_GRAY_COLORS).astype(np.uint8)

    im_eq = np.interp(im_gray.flatten(),bins[:-1],cumulative_hist)
    im_eq = im_eq.reshape(im_gray.shape)
    hist_eq, bins = np.histogram(im_eq.flatten(),NUM_BINS,[0,NUM_BINS])
    return hist_eq, im_eq

def histogram_equalize_rgb(im_orig):
    '''
    Perform the histogram equalization algorithm for RGB image
    :param im_orig: The origin image
    :return: hist_orig,hist_eq_im_eq
    '''
    # transform to YIQ
    im_yiq = rgb2yiq(im_orig)
    im_gray = ((np.asarray(im_yiq[:, :, Y_CHANNEL]) * NUM_GRAY_COLORS).round()).astype(np.uint8)

    # get the histogram of the origin image
    hist_orig, bins = get_hist_orig(im_yiq, True)
    # get the equalize histogram and image
    hist_eq, im_eq = equalize_image(hist_orig, im_gray, bins)

    # normalize the gray channel
    im_yiq[:, :, Y_CHANNEL] = np.divide(im_eq, NUM_GRAY_COLORS)
    # transform to RGB
    im_eq = yiq2rgb(im_yiq)

    return im_eq, hist_orig, hist_eq


def histogram_equalize_gray(im_orig):
    '''
    Perform the histogram equalization algorithm for gray scale image
    :param im_orig: The origin image..
    :return: hist_orig,hist_eq_im_eq
    '''
    hist_orig, bins = get_hist_orig(im_orig, False)
    im_gray = ((np.asarray(im_orig) * NUM_GRAY_COLORS).round()).astype(np.uint8)

    # get the equalize histogram and image
    hist_eq, im_eq = equalize_image(hist_orig, im_gray, bins)
    # normalize the gray channel
    im_eq = np.divide(im_eq, NUM_GRAY_COLORS)
    return im_eq, hist_orig, hist_eq


def histogram_equalize(im_orig):
    '''

    :param im_orig: The origin image
    :return: Return the histogram of the origin image and of the equalized image,
            in addition return the equalized image
    '''
    if len(im_orig.shape) >= RGB_SHAPE:
        im_eq, hist_orig, hist_eq= histogram_equalize_rgb(im_orig[:, :,0:RGB_SHAPE])
    else:
        im_eq, hist_orig, hist_eq= histogram_equalize_gray(im_orig)
    return np.clip(im_eq, 0, 1), hist_orig, hist_eq


def restart_z( n_quant, hist_orig):
    '''
    set the z parameter to the initial value for the quantization algorithm
    :param n_quant: the number of color in the new image
    :param hist_orig: The histogram of the origin image
    :return: the initial z array
    '''

    cdf = hist_orig.cumsum()
    hist_seg= np.arange(NUM_BINS) * (cdf[-1] / n_quant)
    z = (np.argmin(np.abs(cdf[:, np.newaxis] - hist_seg), axis=0))[:n_quant + 1]
    z[n_quant] = INITIAL_Z_MAX
    return z

def calculate_q(n_quant, q, z, hist_orig, bins):
    '''
    calculate the q parameter array i.e the gray scale values
    :param n_quant: the number of color in the new image
    :param q: the gray scale values
    :param z: the segment of the partition of the histogram
    :param hist_orig: The histogram of the origin image
    :param bins: The gray scale palette
    :return: The q parameter array
    '''
    for segment in range(n_quant):
        #partition the array
        hist_seg = np.asarray(hist_orig[z[segment]: z[segment + 1]])
        bins_seg = np.asarray(bins[z[segment]: z[segment + 1]])
        #calculate the q parameter
        segment_cdf = hist_seg.cumsum()
        q[segment] = np.divide((bins_seg * hist_seg).sum(), (segment_cdf[-1]))
    return q


def error_calculation(n_quant, z, q, hist_orig, bins):
    '''
    calculate the error per each iteration
    :param n_quant: the number of color in the new image
    :param z: the segment of the partition of the histogram
    :param q: the gray scale values
    :param hist_orig: The histogram of the origin image
    :param bins: The gray scale palette
    :return: the error for this specific iteration
    '''
    sum_err = 0
    for index in range(n_quant):
        # partition the array
        hist_seg = np.asarray(hist_orig[z[index]: z[index + 1]])
        bins_seg = np.asarray(bins[z[index]: z[index + 1]])
        #calculate the error for each segment
        sum_err += (np.power((q[index] - bins_seg), TWO) * hist_seg).sum()
    return sum_err


def get_lookup_table(n_quant, z, q):
    '''
    form a lookup table
    :param n_quant: the number of color in the new image
    :param z: the segment of the partition of the histogram
    :param q: the gray scale values
    :return: A lookup table
    '''


    lut = np.arange(NUM_BINS)
    # form a lookup table
    for i in range(n_quant):
        lut[z[i]:z[i + 1]] = q[i]
    return lut

def get_gray_channel_and_histogram(im_orig):
    '''
    get the gray channel of the original image and also get the original histogram
    :param im_orig: The original image
    :return: im_yiq is the returned image , gray_channel is the y channel(from the yiq format)
    or the gray channel (is the original image was a gray image), hist_ orig is the original histogram,
    bins is an array the contain the x_axis of the histogram.
    '''
    im_yiq = im_orig
    if len(im_orig.shape) >= RGB_SHAPE:
        im_yiq = rgb2yiq(im_orig[:, :, 0:RGB_SHAPE])
        gray_channel = im_yiq[:, :, Y_CHANNEL]
        hist_orig, bins = get_hist_orig(im_yiq, True)
    else:
        gray_channel = im_orig
        hist_orig, bins = get_hist_orig(im_yiq, False)
    return im_yiq, gray_channel, hist_orig


def quantization_convergence(z, n_iter, n_quant, hist_orig):
    '''
    perform the quantization iteration , i.e calculate q,z and the error
    :param z: the segment of the partition of the histogram
    :param n_iter: number of iterations until convergence
    :param n_quant: the number of color in the new image
    :param hist_orig: the original histogram
    :return: return the error which contain the error of each iteration
    and return the q array which contain the chosen gray colors
    '''
    bins = np.arange(NUM_BINS)
    q = np.arange(n_quant).astype(np.float64)
    last_z = np.zeros(z.shape)
    error = []

    for it in range(n_iter):
        # q calculation
        q = calculate_q(n_quant, q, z, hist_orig, bins)

        #z calculation
        rounded_q = q.round()
        for interval in range(1, n_quant):
            z[interval] = np.divide((rounded_q[interval - 1] + rounded_q[interval]), TWO).round()

        #error calucation
        error.append(error_calculation(n_quant,z ,q.round() ,hist_orig, bins))

        #check if z has converged
        if np.array_equal(last_z,z):
            break

        #update last_z
        last_z = np.copy(z)
    return error, q

def quantize(im_orig, n_quant, n_iter):
    '''
    perform a quantize algorithm
    :param im_orig: the original image
    :param n_quant: number of the colors in the quantized image
    :param n_iter: number of iterations until convergence
    :return: the new image and a list of error for each iteration
    '''
    # get the gray channel and the original histogram
    im_yiq, gray_channel, hist_orig = get_gray_channel_and_histogram(im_orig)

    # restart z
    z = restart_z( n_quant, hist_orig)

    #perform the quantization iteration , i.e calculate q,z and the error
    error, q = quantization_convergence(z, n_iter, n_quant, hist_orig)

    #form a lookup table
    lut = get_lookup_table(n_quant, z, q.round())

    #update the image
    gray_channel = np.asarray(np.multiply(gray_channel, NUM_GRAY_COLORS)).astype(int)
    im_quant = lut[gray_channel]
    im_quant = np.divide(im_quant,NUM_GRAY_COLORS)

    if len(im_orig.shape) == RGB_SHAPE:
        im_yiq[:, :, Y_CHANNEL] = im_quant
        im_quant = yiq2rgb(im_yiq)

    return im_quant, error
