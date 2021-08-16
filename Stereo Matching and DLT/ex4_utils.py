import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    min_ran=disp_range[0]
    max_ran=disp_range[1]
    img_left_sizeX=img_l.shape[0]
    img_left_sizeY=img_l.shape[1]
    disparity_map = np.zeros_like(img_l)
    for row in range(k_size, img_left_sizeX-k_size):
        for col in range(k_size, img_left_sizeY-k_size):
            window = img_l[row-k_size: row+k_size+1, col-k_size: col+k_size+1]
            ssd_min = 999999
            for i in range(max(col - max_ran, k_size), min(col + max_ran, img_l.shape[1] - k_size)):
                temp=abs(col-i)
                if (temp >= min_ran):
                    win_compere = img_r[row-k_size: row+k_size+1, i-k_size: i+k_size+1]
                    ssd = ((window - win_compere)*(window - win_compere)).sum()
                    if ssd < ssd_min:
                        ssd_min = ssd
                        disparity_map[row, col] = temp
    disparity_map =disparity_map* (255//max_ran)
    return disparity_map



def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    min_ran=disp_range[0]
    max_ran=disp_range[1]
    img_left_sizeX=img_l.shape[0]
    img_left_sizeY=img_l.shape[1]
    disparity_map = np.zeros_like(img_l)
    for row in range(k_size, img_left_sizeX - k_size):
        for col in range(k_size, img_left_sizeY - k_size):
            dist_point = 0
            #left window
            win_left = img_l[row - k_size: row + k_size + 1, col - k_size: col + k_size + 1]
            nor_left = np.linalg.norm(win_left)
            calculation_win_left = np.sqrt(((win_left-nor_left)*(win_left-nor_left)).sum())
            ncc_max = -100
            for i in range(min_ran, max_ran):
                if (col + i + k_size + 1 < img_left_sizeY) and(col + i - k_size >= 0)  and \
                        (col - i + k_size + 1 < img_left_sizeY) and (col - i - k_size >= 0) :
                    #right:
                    window_right = img_r[row - k_size: row + k_size + 1, col + i - k_size: col + i + k_size + 1]
                    norm_right = np.linalg.norm(window_right)
                    calculation_window_right = np.sqrt(((window_right - norm_right) *(window_right - norm_right)).sum())
                    nnc_first = ((win_left-nor_left)*(window_right-norm_right)).sum()
                    nnc_sec=calculation_win_left*calculation_window_right
                    nnc=nnc_first/nnc_sec
                    if ncc_max < nnc:
                        ncc_max = nnc
                        dist_point = i
                # left:
                    window_left = img_r[row - k_size: row + k_size + 1, col - i - k_size: col - i + k_size + 1]
                    norm_left = np.linalg.norm(window_left)
                    calculation_window_left = np.sqrt(np.sum((window_left - norm_left) ** 2))
                    nnc_first =  np.sum((win_left - nor_left) * (window_left - norm_left))
                    nnc_sec=(calculation_win_left * calculation_window_left)
                    nnc=nnc_first/nnc_sec
                    if ncc_max < nnc:
                        ncc_max = nnc
                        dist_point = i
            disparity_map[row, col] = dist_point
    disparity_map =disparity_map * (255 // max_ran)
    return disparity_map

def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))
        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]
        return: (Homography matrix shape:[3,3],
                Homography error)
    """
    matrixIndex = 0
    A = np.zeros((8, 9))
    for i in range(0, len(src_pnt)):
        x = src_pnt[i][0]
        y = src_pnt[i][1]
        x_tag = dst_pnt[i][0]
        y_tag = dst_pnt[i][1]
        A[matrixIndex] = [x, y, 1, 0, 0, 0, -x_tag*x, -x_tag*y, -x_tag]
        A[matrixIndex + 1] = [0, 0, 0, x, y, 1, -y_tag*x, -y_tag*y, -y_tag]
        matrixIndex = matrixIndex + 2
    A = np.asarray(A)
    U, D, V_tag = np.linalg.svd(A)
    h = V_tag[-1,:] / V_tag[-1,-1]
    H = h.reshape(3, 3)
    g = cv2.findHomography(src_pnt, dst_pnt) #check in cv2


    new_column = [[1],[1],[1],[1]]
    h_src = np.append(src_pnt, new_column, axis=1) #h_src = Homogeneous(src)
    pred = H.dot(h_src.T).T

    pred1=np.zeros((4, 2))#pred = unHomogeneous(pred) ##???
    for i in range (4):
        for j in range (2):
            pred1[i][j]=pred[i][j]/ pred[i][2]
    error=np.sqrt(np.square(pred1-dst_pnt).mean())

    return H ,error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.
       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.
       output:
        None.
    """
    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    src_p = np.array([[0, 0], [0, src_img.shape[1]-1], [src_img.shape[0]-1, src_img.shape[1]-1],[src_img.shape[0]-1, 0]])
    h, error = computeHomography(dst_p, src_p)
    dest_img_sizeX=dst_img.shape[0]
    dest_img_sizeY=dst_img.shape[1]
    for x in range(dest_img_sizeX - 1):
        for y in range(dest_img_sizeY - 1):
            cal = h.dot(np.array([[x],
                                  [y],
                                  [1]]))
            norm = h[2, :].dot(np.array([[x],
                                         [y],
                                         [1]]))
            cal = cal / norm
            if(cal[0]>0 and cal[0]<src_img.shape[0]):
                if(cal[1]>0 and cal[1]<src_img.shape[1]):
                    dst_img[y][x][0]=src_img[int(cal[0]),int(cal[1]),0]
                    dst_img[y][x][1]=src_img[int(cal[0]),int(cal[1]),1]
                    dst_img[y][x][2]=src_img[int(cal[0]),int(cal[1]),2]
    plt.matshow(dst_img)
    plt.title("warpImag")
    plt.colorbar()
    plt.show()