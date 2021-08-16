from math import sqrt, pi, cos, sin
import cv2
import numpy as np
import matplotlib.pyplot as plt


MATRIX=np.array([[0.299, 0.587, 0.114],
                 [0.596, -0.275, -0.3212],
                 [0.212, -0.523, 0.311]])



def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    kernel1=np.flip(kernel1)
    for j in range (len(kernel1)-1):
        inSignal=np.insert(inSignal,len(inSignal),0)
        inSignal=np.insert(inSignal,0,0)
    l=[]
    y=len(inSignal)
    u=len(kernel1)-1
    for i in range (y-(u)):
        sum=0;
        for j in range (len(kernel1)):
            mal=inSignal[i+len(kernel1)-j-1]
            place=kernel1[u-j]
            sum=sum+(mal*place)
        l.append(sum)
    return l

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    """
    Convolve a 2-D array with a given kernel :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    inImage_row = inImage.shape[0]
    inImage_col = inImage.shape[1]
    kernel_row = len(kernel2)
    kernel_col = len(kernel2[0])
    kernel = np.flipud(np.fliplr(kernel2))
    new_img = np.zeros_like(inImage)
    padded_image = np.zeros((inImage_row + kernel_row-1, inImage_col + kernel_col-1))
    padded_image[kernel_row-2:-(kernel_row-2), kernel_col-2:-(kernel_col-2)] = inImage

    for x in range(inImage.shape[1]):
        for y in range(inImage.shape[0]):
            new_img[y, x] = (kernel * padded_image[y: y + (len(kernel[0])), x: x + (len(kernel[1]))]).sum()
    return new_img

def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    Derivative_x= cv2.filter2D(inImage,-1,(np.array([[0,0,0],[1,0,-1],[0,0,0]])), borderType=cv2.BORDER_REPLICATE)
    Derivative_y= cv2.filter2D(inImage,-1,(np.array([[0,1,0],[0,0,0],[0,-1,0]])), borderType=cv2.BORDER_REPLICATE)
    Mag=MagG(Derivative_x,Derivative_y)
    Direction=(DirectionG(Derivative_x,Derivative_y)).astype(int)

    return  Direction , Mag, Derivative_x, Derivative_y

def DirectionG(ix:np.ndarray ,iy:np.ndarray ):
    ix=ix+0.000001
    return np.arctan(np.divide(iy,ix))

def MagG (ix:np.ndarray ,iy:np.ndarray):
    return (((ix**2)+(iy**2))**0.5)





def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
   Blur an image using a Gaussian kernel
   :param inImage: Input image
   :param kernelSize: Kernel size
   :return: The Blurred image
   """




def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    in_image=in_image.astype(float)
    gaussian_kernel = cv2.getGaussianKernel(kernel_size.shape[0], sigma=0)
    img = cv2.filter2D(in_image, -1, gaussian_kernel)
    return img


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    img=img.astype(float)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    x_edges = cv2.filter2D(img, -1, gx, borderType=cv2.BORDER_REPLICATE)
    y_edges = cv2.filter2D(img, -1, gy, borderType=cv2.BORDER_REPLICATE)
    grad = np.sqrt(x_edges**2 +y_edges**2)
    (thresh, my_output) = cv2.threshold(grad , thresh, 1, cv2.THRESH_BINARY)



    x_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=3)
    y_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=3)
    grad = np.sqrt(x_sobel**2 +y_sobel**2)
    (thresh, opencv_output) = cv2.threshold(grad, thresh, 1, cv2.THRESH_BINARY)



    return opencv_output,  my_output

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    img=img.astype(float)
    kernal=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian = np.flip(kernal)
    fill_img = cv2.filter2D(img, -1, laplacian, borderType=cv2.BORDER_REPLICATE)
    new_img=np.zeros_like(fill_img)
    for indrow in range (1,fill_img.shape[0]-1):
        for indcol in range(1,fill_img.shape[1]-1):
            if(fill_img[indrow][indcol-1]>0 and fill_img[indrow][indcol+1]<0)  :
                if(fill_img[indrow][indcol-1]>fill_img[indrow][indcol]>fill_img[indrow][indcol+1]):#left>index>right
                    new_img[indrow][indcol]=1
    return new_img



def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    img=img.astype(float)
    img=cv2.GaussianBlur(img,(3,3),0)
    Direction , Mag, ix, iy=convDerivative(img)
    #Mag = (Mag/Mag.max())*255
    for i in range (Direction.shape[0]):
        for j in range (Direction.shape[1]):
            Direction[i][j]=np.mod(Direction[i][j],180)
            if(0<=Direction[i][j]<22.5 or 157.5<=Direction[i][j]<180):
                Direction[i][j]=0
            elif(22.5<=Direction[i][j]<67.5):
                Direction[i][j]=45
            elif(67.5<=Direction[i][j]<112.5):
                Direction[i][j]=90
            elif(112.5<=Direction[i][j]<157.5):
                Direction[i][j]=135
    new_img = np.zeros_like(Mag)

    for i in range (1,Direction.shape[0]-1):
        for j in range (1,Direction.shape[1]-1):
            if(Direction[i][j]==0):#check up down
                if(Mag[i+1][j]<Mag[i][j] and  Mag[i-1][j]< Mag[i][j]):
                    new_img[i][j]=Mag[i][j]
            elif(Direction[i][j]==45):
                if(Mag[i-1][j-1]<Mag[i][j] and  Mag[i+1][j+1]<Mag[i][j]):
                    new_img[i][j]=Mag[i][j]
            elif(Direction[i][j]==90): # left right
                if(Mag[i][j-1]<Mag[i][j] and Mag[i][j+1]<Mag[i][j]):
                    new_img[i][j]=Mag[i][j]
            elif(Direction[i][j]==135):
                if(Mag[i-1][j+1]< Mag[i][j] and  Mag[i+1][j-1]<Mag[i][j]):
                    new_img[i][j]=Mag[i][j]



    strong_edges = np.zeros_like(new_img)
    for x in range(new_img.shape[0]-1):
        for y in range(new_img.shape[1]-1):
            if Mag[x][y] > thrs_1:
                strong_edges[x][y] = 1
            if new_img[x][y] < thrs_2:
                strong_edges[x][y] = 0
            elif thrs_2 <new_img[x][y]<thrs_1 :
                if new_img[x - 1][y] >thrs_1 \
                        or new_img[x + 1][y]  > thrs_1 \
                        or new_img[x][y - 1]  > thrs_1 \
                        or new_img[x][y + 1] > thrs_1 \
                        or new_img[x - 1][y - 1]  > thrs_1 \
                        or new_img[x + 1][y + 1]  > thrs_1 \
                        or new_img[x + 1][y - 1]  > thrs_1 \
                        or new_img[x - 1][y + 1]  > thrs_1:
                    strong_edges[x][y] = 1

    img=np.uint8(img)
    canny=cv2.Canny(img,100,200)
    return canny ,strong_edges


def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    canny_cv = cv2.Canny(img, 100, 200)
    edges = []
    points = []
    circles = []
    temp = {}
    for x in range(canny_cv.shape[0]):
        for y in range(canny_cv.shape[1]):
            if canny_cv[x, y] == 255:
                edges.append((x, y))
    #points
    for radius in range(min_radius, max_radius + 1):
        for steps_tims in range(100):
            x = int(radius * cos(2 * pi * steps_tims / 100))
            y = int(radius * sin(2 * pi * steps_tims / 100))
            points.append((x, y, radius))

    for x1, y1 in edges:
        for dx, dy, radius in points:
            new_2 = x1 - dx
            new_1 = y1 - dy
            numofshow  = temp.get((new_1, new_2, radius))
            if numofshow is None:
                numofshow = 0
            temp[(new_1, new_2, radius)] = numofshow + 1
    #circles
    sorted_temp = sorted(temp.items(), key=lambda i: -i[1])
    for circle, counter in sorted_temp:
        x, y, radius = circle
        if (counter / 100 >= 0.4):
            if(all((x - c_x) ** 2 + (y - c_y) ** 2 > c_radius ** 2 for c_x, c_y, c_radius in circles)):
                circles.append((x, y, radius))
    return circles

def drow(img:np.ndarray):
    ans=houghCircle(img, 32,51)#18,20)
    fig, ax = plt.subplots()
    for x,y,radius in ans:
        circle = plt.Circle((x, y), radius, color='r',lw=2, fill=False)
        center = plt.Circle((x, y), 0.3, color='b', )
        ax.add_patch(circle)
        ax.add_patch(center)
    ax.imshow(img)
    ax.set_title("Hough Circles")
    plt.show()
    return ans





