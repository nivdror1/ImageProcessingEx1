import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave as imsave, imread as imread
from skimage.color import rgb2gray

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
    if( representation < 1 or representation > 2):
        return -1

    #read the image
    img = imread(filename)

    #todo  check if the image is proper/ maybe check if the suffix is jpg,gif and etc
    #convert the RGB image to gray scale image
    if len(img.shape) == 3 and representation == 1:
        return rgb2gray(img)
    return np.divide(img.astype(np.float64), 255)


def display_image(filename, representation):
    """
    read and display a image
    :param filename: a string that contains the name of the image
    :param representation: in which format display the image (gray scale pr RGB)
    :return: if fails return -1 otherwise return 0
    """
    if representation < 1 or representation > 2:
        return -1
    #read and convert the format of the image according to the representation
    img = read_image(filename, representation)

    #display the image
    if representation == 1:
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
    im_gray = (im_gray * 255).round()

    if flag:
        gray_channel = np.asarray(im_gray[:, :, 0]).flatten()
    else:
        gray_channel = np.asarray(im_gray).flatten()
    hist_orig, bins = np.histogram(gray_channel, 256, [0, 256])
    return hist_orig, bins


def equalize_image(hist_orig, im_gray, bins):
    '''
    Equalize the histogram and produce the equalized image
    :param hist_orig: origin image's histogram
    :param im_gray: The gray channel/Y channel of the origin image
    :return: The equalized histogram and the equalized image
    '''
    # get the cumulative distribution function
    hist_eq = hist_orig.cumsum()
    # normalize
    hist_eq = ((hist_eq / hist_eq[255]) * 255).astype(np.uint8)


    im_eq = np.interp(im_gray.flatten(),bins[:-1],hist_eq)
    im_eq = im_eq.reshape(im_gray.shape)
    #im_eq = hist_eq[im_gray]
    h, b = np.histogram(im_eq,256,[0,256])
    cdf = h.cumsum()

    plt.plot(cdf)
    plt.show()



    return hist_eq, im_eq

def histogram_equalize_rgb(im_orig):
    '''
    Perform the histogram equalization algorithm for RGB image
    :param im_orig: The origin image
    :return: hist_orig,hist_eq_im_eq
    '''
    # transform to YIQ
    im_yiq = rgb2yiq(im_orig)
    im_gray = ((np.asarray(im_yiq[:, :, 0]) * 255).round()).astype(np.uint8)

    # get the histogram of the origin image
    hist_orig, bins = get_hist_orig(im_yiq, True)
    # get the equalize histogram and image
    hist_eq, im_eq = equalize_image(hist_orig, im_gray, bins)

    # normalize the gray channel
    im_yiq[:, :, 0] = np.divide(im_gray, 255)
    # transform to RGB
    im_eq = yiq2rgb(im_yiq)
    return hist_orig, hist_eq, im_eq


def histogram_equalize_gray(im_orig):
    '''
    Perform the histogram equalization algorithm for gray scale image
    :param im_orig: The origin image..
    :return: hist_orig,hist_eq_im_eq
    '''
    hist_orig, bins = get_hist_orig(im_orig, False)
    im_gray = ((np.asarray(im_orig) * 255).round()).astype(np.uint8)

    # get the equalize histogram and image
    hist_eq, im_eq = equalize_image(hist_orig, im_gray, bins)
    # normalize the gray channel
    im_eq = np.divide(im_eq, 255)
    return hist_orig, hist_eq, im_eq


def histogram_equalize(im_orig):
    '''

    :param im_orig: The origin image
    :return: Return the histogram of the origin image and of the equalized image,
            in addition return the equalized image
    '''
    if len(im_orig.shape) == 3:
        hist_orig, hist_eq, im_eq = histogram_equalize_rgb(im_orig)
    else:
        hist_orig, hist_eq, im_eq = histogram_equalize_gray(im_orig)
    #plt.imshow(im_eq, cmap= plt.cm.gray)

    return hist_orig, hist_eq, im_eq


def restart_z( n_quant, hist_orig):
    '''
    set the z parameter to the initial value for the quantization algorithm
    :param n_quant: the number of color in the new image
    :param hist_orig: The histogram of the origin image
    :return: the initial z array
    '''

    cdf = hist_orig.cumsum()
    hist_seg= np.arange(256) * (cdf[-1] / n_quant)
    z = (np.argmin(np.abs(cdf[:, np.newaxis] - hist_seg), axis=0))[:n_quant + 1]
    z[n_quant] = 256.0
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
        q[segment] = ((bins_seg * hist_seg).sum())/ (segment_cdf[-1])
    return q


def error_calculation(n_quant ,z, q, hist_orig, bins):
    '''
    calculate the error per each iteration
    :param n_quant: the number of color in the new image
    :param z: the segment of the partition of the histogram
    :param q: the gray scale values
    :param hist_orig: The histogram of the origin image
    :param bins: The gray scale palette
    :return:
    '''
    sum_err = 0
    for index in range(n_quant):
        # partition the array
        hist_seg = np.asarray(hist_orig[z[index]: z[index + 1]])
        bins_seg = np.asarray(bins[z[index]: z[index + 1]])
        #calculate the error for each segment
        sum_err += (np.power((q[index] - bins_seg), 2) * hist_seg).sum()
    return sum_err

def get_lookup_table(n_quant, z, q):
    '''
    form a lookup table
    :param n_quant: the number of color in the new image
    :param z: the segment of the partition of the histogram
    :param q: the gray scale values
    :return: A lookup table
    '''
    lut = np.arange(256)
    # form a lookup table
    for i in range(n_quant):
        lut[z[i]:z[i + 1]] = q[i]
    return lut


def quantize(im_orig, n_quant, n_iter):
    im_yiq = im_orig
    if len(im_orig.shape) == 3:
        im_yiq = rgb2yiq(im_orig)
        gray_channel = im_yiq[:, :, 0]
        hist_orig, bins = get_hist_orig(im_yiq, True)
    else:
        gray_channel = im_orig
        hist_orig, bins = get_hist_orig(im_yiq, False)

    # restart z
    z = restart_z( n_quant, hist_orig)

    bins = np.arange(256)
    q = np.arange(n_quant).astype(np.float64)
    last_z = np.zeros(z.shape)
    error = []

    for it in range(n_iter):
        # q calculation
        q = calculate_q(n_quant, q, z, hist_orig, bins)

        #z calculation
        for interval in range(1, n_quant):

            z[interval] = (q[interval - 1] + q[interval])/2

        #error calucation
        error.append(error_calculation(n_quant,z ,q.round() ,hist_orig, bins))

        #check if z has converged
        if np.array_equal(last_z,z):
            break

        #update last_z
        last_z = np.copy(z)
    #form a lookup table
    lut = get_lookup_table(n_quant, z, q.round())


    #update the image
    gray_channel = np.asarray(np.multiply(gray_channel, 255)).astype(int)
    im_quant = lut[gray_channel]
    im_quant = np.divide(im_quant,255)
    plt.plot(error)
    plt.show()
    if len(im_orig.shape) == 3:
        im_yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(im_yiq)

    return [im_quant, error]


def main():
    name1 = "gray_orig.png"
    name2 = "logoGray.jpg"

    #display_image(name1, 2)
    imgRGB = read_image(name1,2)
    #display_image(name1, 1)
    #imsave("LowGray.jpg",imgRGB)
    list = quantize(imgRGB,10, 20)
    plt.imshow(list[0].clip(0,1))
    #hist_orig, hist_eq, im_eq = histogram_equalize(imgRGB)
    #plt.imshow(im_eq)
    # plt.show()

if __name__ == "__main__":
    main()