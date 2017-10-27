import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave as imsave, imread as imread
from skimage.color import rgb2gray

def read_image(fileName, representation):
    """
    read the image and convert it to gray scale if necessary
    :param fileName: the image
    :param representation: gray scale or RGB format to which the image is to be converted
    :return: A image whose values are normalized and in the given representation
    """
    if( representation < 1 or representation > 2):
        return -1

    #read the image
    img = imread(fileName)

    #todo  check if the image is proper
    #convert the RGB image to gray scale image
    if len(img.shape) == 3 and representation == 1:
        return rgb2gray(img)
    return np.divide(img.astype(np.float64), 255)

def main():
    name1 = "logo.jpg"
    name2 = "logoGray.jpg"

    gray_img = read_image(name1, 1)
    plt.imshow(gray_img, cmap =plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    main()