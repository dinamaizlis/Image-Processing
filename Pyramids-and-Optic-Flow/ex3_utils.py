
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

#static
k_size = 5
sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
kernel = cv2.getGaussianKernel(5, sigma)
kernel = kernel.dot(kernel.T)


#TOOK FROM MATALA 2 - SHAI AHARON -SOLUTION
def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    img_h, img_w = in_image.shape[:2]
    kernel_shape = np.array([x for x in kernel.shape])
    mid_ker = kernel_shape // 2
    padded_signal = np.pad(in_image.astype(np.float32),
                           ((kernel_shape[0], kernel_shape[0]), (kernel_shape[1], kernel_shape[1])), 'edge')
    out_signal = np.zeros_like(in_image)

    for i in range(img_h):
        for j in range(img_w):
            st_x = j + mid_ker[1] + 1
            end_x = st_x + kernel_shape[1]
            st_y = i + mid_ker[0] + 1
            end_y = st_y + kernel_shape[0]
            out_signal[i, j] = (padded_signal[st_y:end_y, st_x:end_x] * kernel).sum()
    return out_signal


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=15, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    Original_points=[]
    uv = []
    new_im1 = im1  /255
    new_im2 = im2  /255

    kernel_x = np.array([[1, 0, -1]])
    kernel_y = np.array([[1],[ 0],[ -1]])
    fx=conv2D(new_im1,kernel_x)
    fy=conv2D(new_im1,kernel_y)
    ft =new_im1-new_im2
    h = int(win_size / 2)  # [-w, w]
    # over all the pix
    for i in range(h, im1.shape[0] - h, step_size):
        for j in range(h, im1.shape[1] - h, step_size):
            # find the derivative
            Ix = fx[i - h:i + h + 1, j - h:j + h + 1].flatten()
            Iy = fy[i - h:i + h + 1, j - h:j + h + 1].flatten()
            It = ft[i - h:i + h + 1, j - h:j + h + 1].flatten()
            #calculation the matrix-A and vector-d
            A00 = np.sum(np.matmul(Ix , Ix))
            A01 = np.sum(np.matmul(Ix , Iy))
            A10 = np.sum(np.matmul(Iy , Ix))
            A11 = np.sum(np.matmul(Iy , Iy))
            D00 = np.sum(np.matmul(Ix , It))
            D01 = np.sum(np.matmul(Iy , It))

            newA = np.linalg.pinv(np.array([[A00, A01], [A10, A11]]))  # revers matrix (a_t*a)^(-1)
            vec_b = np.array([[-D00], [-D01]])  # b size 2*1
            vect_u_v = np.matmul(newA , vec_b)  # [u,v] size (*,2)
            uv.append([vect_u_v[0][0], vect_u_v[1][0]])
            Original_points.append([j,i])
    uv= np.array(uv)
    Original_points=np.array(Original_points)
    return Original_points , uv


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    gaussianPyr_list = [img]
    kernel_G = cv2.getGaussianKernel(5, sigma)

    
    for i in range(1, levels):
        new_img = cv2.filter2D(gaussianPyr_list[i - 1], -1, kernel=kernel_G)
        new_img =new_img[::2, ::2]
        gaussianPyr_list.append(new_img)
    return gaussianPyr_list


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if len(img.shape) == 3:
        new_img = np.zeros((2*img.shape[0], 2*img.shape[1], 3))#padding:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(3):
                    new_img[2*i][2*j][k] = img[i][j][k]
        new_img = cv2.filter2D(new_img, -1, kernel=4*gs_k)# blur
    else:
        new_img = np.zeros((2*img.shape[0], 2*img.shape[1]))#padding:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[2*i][2*j] = img[i][j]
        new_img = cv2.filter2D(new_img, -1, kernel=4*gs_k)# blur
    return new_img


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaussianPyr_list= gaussianPyr(img, levels)
    laplaceianPyr_list = [gaussianPyr_list[levels-1]]
    for i in range(levels-1, 0, -1):
        if i is not 0:
            original= gaussianPyr_list[i-1]
            expand = gaussExpand(gaussianPyr_list[i], kernel)
            if original.shape[0] == expand .shape[0]-1:
                expand=expand[0:-1,0:-1]
            diff = original - expand
            laplaceianPyr_list.append(diff)
    laplaceianPyr_list.reverse()
    return laplaceianPyr_list


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    upper_img = lap_pyr[levels-1]
    for i in range(levels-2, -1, -1):
        upper_img = lap_pyr[i] + gaussExpand(upper_img, kernel)
    return upper_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    if len(img_1.shape)==3:
        naive_blend =blend_3D(img_1, img_2, mask)
        laplacian_img1 = laplaceianReduce(img_1, levels)
        laplacian_img2 = laplaceianReduce(img_2, levels)
        gaussian_maks = gaussianPyr(mask, levels)
        blended_image = blend_3D(laplacian_img1[levels-1], laplacian_img2[levels-1], gaussian_maks[levels-1])
        for i in range(levels-2, -1, -1):
            new_image =blend_3D(laplacian_img1[i], laplacian_img2[i], gaussian_maks[i])
            expand = gaussExpand(blended_image, kernel)
            if new_image.shape[0] == expand.shape[0]-1:
                expand = expand[0:-1, 0:-1]
            blended_image = new_image+expand
    else:
        naive_blend =blend_2D(img_1, img_2, mask)
        laplacian_img1 = laplaceianReduce(img_1, levels)
        laplacian_img2 = laplaceianReduce(img_2, levels)
        gaussian_maks = gaussianPyr(mask, levels)
        blended_image = blend_2D(laplacian_img1[levels-1], laplacian_img2[levels-1], gaussian_maks[levels-1])
        for i in range(levels-2, -1, -1):
            new_image =blend_2D(laplacian_img1[i], laplacian_img2[i], gaussian_maks[i])
            expand = gaussExpand(blended_image, kernel)
            if new_image.shape[0] == expand.shape[0]-1:
                expand = expand[0:-1, 0:-1]
            blended_image = new_image+expand
    return naive_blend, blended_image


def blend_2D(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray) -> (np.ndarray):
    blend = np.zeros_like(img_1)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y] == 1:
                blend[x][y] = img_1[x][y]
            else:
                blend[x][y] = img_2[x][y]
    return blend

def blend_3D(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray) -> (np.ndarray):
    blend = np.zeros_like(img_1)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y][1] == 1:
                blend[x][y][0] = img_1[x][y][0]
                blend[x][y][1] = img_1[x][y][1]
                blend[x][y][2] = img_1[x][y][2]
            else:
                blend[x][y][0] = img_2[x][y][0]
                blend[x][y][1] = img_2[x][y][1]
                blend[x][y][2] = img_2[x][y][2]
    return blend


