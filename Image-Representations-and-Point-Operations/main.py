from typing import List
import matplotlib.pyplot as plt
import cv2
import numpy as np

from ex1_utils import myID, imReadAndConvert, imDisplay, transformRGB2YIQ, transformYIQ2RGB

if __name__ == '__main__':

    imReadAndConvert('/Users/dinamaizlis/Desktop/ex/dark.jpg',0)##men
    #imReadAndConvert('/Users/dinamaizlis/Desktop/ex/bac_con.png',0)
    #imag=imReadAndConvert('/Users/dinamaizlis/Desktop/ex/beach.jpg',0)

    #imDisplay('/Users/dinamaizlis/Desktop/ex/beach.jpg',0)
    # image=imReadAndConvert('/Users/dinamaizlis/Desktop/ex/beach.jpg',0)
    # yiq_img=transformYIQ2RGB(image)
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[1].imshow(yiq_img)
    # plt.show()