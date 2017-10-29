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

def main():
    name1 = "logo.jpg"
    name2 = "logoGray.jpg"

    #display_image(name1, 2)
    imgRGB = read_image(name1,2)
    imgYIQ = rgb2yiq(imgRGB)
    im = yiq2rgb(imgYIQ)
    plt.imshow(im)
    plt.show()

if __name__ == "__main__":
    main()