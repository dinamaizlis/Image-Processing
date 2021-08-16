# ps2
import os
import numpy as np
from ex4_utils import *
import cv2

print("ID 319092201")

def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    if(method==disparitySSD):
        plt.title("disparitySSD")
    else:
        plt.title("disparityNC")
    plt.colorbar()
    plt.show()


def main():
    ## part 1
    # Display depth SSD
    L = cv2.imread('pair0-L.png',cv2.COLOR_BGR2RGB)
    R = cv2.imread('pair0-R.png',cv2.COLOR_BGR2RGB)
    print("disparitySSD")
    displayDepthImage(L, R, (0, 5), method=disparitySSD)
    # # Display depth NC
    # print("disparityNC")
    # displayDepthImage(L, R, (0, 5), method=disparityNC)
    #
    #
    # #patr 2
    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, error = computeHomography(src, dst)
    # print("Homography= ",h)
    # print("error = ",error)
    #
    # src = cv2.imread('car.jpg' ,cv2.COLOR_BGR2RGB)
    # dst = cv2.imread('billBoard.jpg' ,cv2.COLOR_BGR2RGB)
    # print(warpImag)
    # warpImag(src, dst)




if __name__ == '__main__':
    main()
