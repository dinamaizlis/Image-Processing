from numpy import dtype

from ex2_utils import *
import matplotlib.pyplot as plt
import time
import cv2


def main():
    print("My ID: 319092201\n")
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


def conv1Demo():
    inSignal1 = np.array([1, 2, 3])
    kernel1D_1 = np.array([1,1,1])
    kernel1D_2 = np.array([1,1])
    kernel1D_3 = np.array([1, 2, 1])
    print(np.convolve(inSignal1, kernel1D_1, "full") ,"=", conv1D(inSignal1, kernel1D_1) )
    print(np.convolve(inSignal1, kernel1D_2, "full") ,"=",conv1D(inSignal1, kernel1D_2))
    print(np.convolve(inSignal1, kernel1D_3, "full"),"=",conv1D(inSignal1, kernel1D_3))

    inSignal2 = np.array([1, 2, 3,4,5])
    kernel1D_21 = np.array([1,2,3])
    kernel1D_22 = np.array([0,2,0])
    kernel1D_23 = np.array([0,1,0])
    kernel1D_24 = np.array([1/3,1/3,1/3])
    print(np.convolve(inSignal2, kernel1D_21, "full"),"=",conv1D(inSignal2, kernel1D_21))
    print(np.convolve(inSignal2, kernel1D_22, "full"),"=",conv1D(inSignal2, kernel1D_22))
    print(np.convolve(inSignal2, kernel1D_23, "full") ,"=",conv1D(inSignal2, kernel1D_23))

def conv2Demo():
    img = read_pic("beach.jpg")

    # example 1
    kernel1 = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")
    kernel1 = kernel1 / kernel1.sum()
    conv_pic = conv2D(img, kernel1)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")
    ax[1].imshow(conv_pic, cmap='gray')
    ax[1].set_title("my conv2D")
    ax[2].imshow(cv2.filter2D(img, -1, kernel1, borderType=cv2.BORDER_REPLICATE), cmap='gray')
    ax[2].set_title("opencv conv2D")
    plt.show()



def derivDemo():
    img = read_pic("beach.jpg")

    directions, magnitude, x_der, y_der = convDerivative(img)
    f, ax = plt.subplots(1, 4)
    ax[0].imshow(y_der, cmap='gray')
    ax[0].set_title("y_der")
    ax[1].imshow(x_der, cmap='gray')
    ax[1].set_title("x_der")
    ax[2].imshow(directions, cmap='gray')
    ax[2].set_title("directions")
    ax[3].imshow(magnitude, cmap='gray')
    ax[3].set_title("magnitude")
    plt.show()


def blurDemo():
    img = read_pic("beach.jpg")
    # example 1
    ker_size = np.ones((3, 3))
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image")
    ax[1].imshow(blurImage2(img, ker_size), cmap='gray')
    ax[1].set_title("cv-blur(3,3)k")
    plt.show()
    # example 2
    img = read_pic("beach.jpg")
    ker_size = np.ones((9, 9))
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image")
    ax[1].imshow(blurImage2(img, ker_size), cmap='gray')
    ax[1].set_title("cv2-blur(9,9)k")
    plt.show()


def edgeDemo():
    img =plt.imread("Lenna_(test_image).png")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel
    openCV, myAns = edgeDetectionSobel(img, 0.55)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image")

    ax[1].imshow(myAns, cmap='gray')
    ax[1].set_title("my code soble")
    ax[2].imshow(openCV, cmap='gray')
    ax[2].set_title("opencv soble")
    plt.show()

    img =plt.imread("Lenna_(test_image).png")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Zero crossing
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original img")
    ax[1].imshow(edgeDetectionZeroCrossingSimple(img), cmap='gray')
    ax[1].set_title("zero crossing")
    plt.show()

    # Canny
    img = cv2.imread('Lenna_(test_image).png')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int)
    openCV, myAns = edgeDetectionCanny(img,30,15)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image")
    ax[1].imshow(myAns, cmap='gray')
    ax[1].set_title("my code-Canny")
    ax[2].imshow(openCV, cmap='gray')
    ax[2].set_title("opencv-Canny")
    plt.show()


def houghDemo():
    img =plt.imread("HoughCircles.jpg")
    drow(img)



def read_pic(img):
    img = cv2.cvtColor(cv2.imread(img, -1), cv2.COLOR_BGR2GRAY)
    img = img.astype('float64')
    return img / 255.0


if __name__ == '__main__':
    main()
