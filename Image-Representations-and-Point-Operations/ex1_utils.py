"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
from typing import List
import numpy as np
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
MATRIX=np.array([[0.299, 0.587, 0.114],
                [0.596, -0.275, -0.3212],
                [0.212, -0.523, 0.311]])


def myID() -> np.int:
    return 319092201


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    src = cv2.imread(filename)
    if representation is 1 :
        #from bgr to gray
        image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)/255 #normalized to the range [0,1]
        return image
    else:
        #from bgr to rgb
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)/255 #normalized to the range [0,1]
        return image

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    window_name = 'Image'
    image=imReadAndConvert(filename,representation)
    if representation is 1:
        plt.imshow(image,cmap='gray')
        plt.show()
    else:
        plt.imshow(image)
        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    window_name = 'Image'
    shape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), MATRIX.transpose()).reshape(shape)

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    Matrix = np.linalg.inv(MATRIX.transpose())
    shape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), Matrix).reshape(shape)

#Creat Histogram
def calhist(img: np.ndarray) ->  np.ndarray:
    hist=np.zeros(256)
    for pix in range(256):
        hist[pix]=np.count_nonzero(img==pix)
    return hist


#Ccalculation calCumSum
def calCumSum(arr: np.array) -> np.ndarray:
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
    return cum_sum

#Ccalculation the hist
def histEq(img: np.ndarray) -> np.ndarray:
    hist=np.zeros(256)
    for pix in range(256):
        hist[pix]=np.count_nonzero(img==pix)
    cumsum = calCumSum(hist)
    cumsum_n = cumsum / cumsum.max()
    look_ut = np.zeros(256)
    for inten in range(len(cumsum_n)):
        new_color = int(np.floor(255 * cumsum_n[inten]))
        look_ut[inten] = new_color
    new_img = np.zeros_like(img, dtype=np.float)
    for old_color, new_color in enumerate(look_ut):
        new_img[img == old_color] = new_color
    return new_img/255

def is_rgb(img1: np.ndarray) -> int:
    if len(img1.shape) is 3 :
        return 1
    return 0

def hsitogramEqualize(img1: np.ndarray) -> np.ndarray:
    if is_rgb(img1):#for the RGB images
    #convert from RGB to YIQ
        YIQimage=transformRGB2YIQ(img1)
        levelY=(YIQimage[:,:,0]*255).astype(int)
    #Creat Histogram
        historg=np.zeros(256)
        for pix in range(256):
            historg[pix]=np.count_nonzero(levelY==pix)
    #Ccalculation the hist
        YIQimage[:,:,0]=histEq(levelY)
        YIQpix=YIQimage[:,:,0]*255
        histEQ=np.zeros(256)
    #Creat Histogram
        for pix in range(256):
            histEQ[pix]=np.count_nonzero(YIQpix==pix)
    #convert from YIQ to RGB
        imgEq=transformYIQ2RGB(YIQimage)
    else:

        #for grayscale images
        level1 = (img1 * 255).astype(np.int)
        #Creat Histogram
        historg=np.zeros(256)
        for pix in range(256):
            historg[pix]=np.count_nonzero(level1==pix)
        #Ccalculation the hist
        imgEq=(histEq(level1))*255
        #Creat Histogram
        histEQ=np.zeros(256)
        for pix in range(256):
            histEQ[pix]=np.count_nonzero(imgEq==pix)


    return imgEq, historg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    MSE=[] #A list of the MSE error in each iteration
    alllist=[] #A list of the quantized image in each iteration

    if is_rgb(imOrig): #for the RGB images
        YIQimage=transformRGB2YIQ(imOrig)
        imOrig1=(YIQimage[:,:,0]*255).astype(int)
        imOrig1=imOrig1.ravel() ## imOrig1.shape(**,)

        #Creat Histogram
        instancesOfPixel=np.zeros(256)
        for pix in range(256):
            instancesOfPixel[pix]=np.count_nonzero(imOrig1==pix)
        instancesOfPixel=instancesOfPixel.astype(int) #convert from float to int
        #camsum
        instancesOfPixelSum=(calCumSum((instancesOfPixel))).astype(int)
        limit=[0]
        numOfPix=instancesOfPixelSum[255]
        numOfPixInCell=int(numOfPix/nQuant)
        #Initial division of boundaries - by almost identical division of pixels in each cell
        i=1 # start from the first cell
        for ind in range (255) or i > nQuant:
            if numOfPixInCell*(i) < instancesOfPixelSum[ind]: #check the comsum comperd to the ability places in the cell
                limit.append(ind-1)#insert
                i+=1 #next cell
        limit.append(255)
        #find the mean color bettwin two limits
        centercolor=[]
        centercolor=FindingQ(limit,instancesOfPixel)
        #updet the new image by the new colors
        firstlist=updateColor(imOrig1,limit,centercolor)
        #find the MSE error
        MSE.append(mse(imOrig1,firstlist,numOfPix))
        #firstlist.shape=(**,) for convert to 3d
        a=imOrig.shape[0] #row
        b=imOrig.shape[1] #col
        YIQimage[:,:,0]=np.reshape(firstlist/255, (a, b)) #convert
        firstlist=transformYIQ2RGB(YIQimage) #from yiq to rgb
        alllist.append(firstlist) #insert to the list of quantized image

        if nIter >1: #the first time is up
            for time in range(nIter): #loop nIter times
                limit=FindingZ(centercolor) # find the new limit by the color
                centercolor =FindingQ(limit,instancesOfPixel) #find the new color in the new limit
                firstlist=updateColor(imOrig1,limit,centercolor)#update color
                MSE.append(mse(imOrig1,firstlist,numOfPix))
                YIQimage[:,:,0]=np.reshape(firstlist/255, (a, b))
                firstlist=transformYIQ2RGB(YIQimage)
                alllist.append(firstlist)
    #for grayscale images
    else:
        MSE=[]
        alllist=[]
        imOrig1=(imOrig*255).astype(int)
        imOrig1=imOrig1.ravel()#imOrig1.shape()=(***,)
        instancesOfPixel=np.zeros(256)
        for pix in range(256):
            instancesOfPixel[pix]=np.count_nonzero(imOrig1==pix)
        instancesOfPixel=instancesOfPixel.astype(int)
        instancesOfPixelSum=(calCumSum((instancesOfPixel))).astype(int)
        limit=[0]
        numOfPix=instancesOfPixelSum[255]
        numOfPixInCell=int(numOfPix/nQuant)
        i=1
        for ind in range (255) or i > nQuant:
            if numOfPixInCell*(i) < instancesOfPixelSum[ind]:
                limit.append(ind-1)
                i+=1
        limit.append(255)
        centercolor=FindingQ(limit,instancesOfPixel)
        firstlist=updateColor(imOrig1,limit,centercolor)
        MSE.append(mse(imOrig1,firstlist,numOfPix))
        d=imOrig.shape[0]
        c=imOrig.shape[1]
        firstlist.resize(d,c)
        alllist.append(firstlist)
        if nIter >1:
            for time in range(nIter):
                limit=FindingZ(centercolor)
                centercolor =FindingQ(limit,instancesOfPixel)
                firstlist=updateColor(imOrig1,limit,centercolor)
                MSE.append(mse(imOrig1,firstlist,numOfPix))
                d=imOrig.shape[0]
                c=imOrig.shape[1]
                firstlist.resize(d,c)
                alllist.append(firstlist)

    return alllist,MSE

#updeat color- get the origin image, the limit and the color-by creat new array
def updateColor(img:np.ndarray,limit:list,centercolor:list)->np.ndarray:
    updateimg=np.zeros(len(img))
    for ind in range (len(centercolor)):
        updateimg[limit[len(centercolor)-ind]>img]=centercolor[len(centercolor)-ind-1]
    return updateimg.astype(int)

#find the limits
def FindingZ (centercolor:list[int])->list[int]:
    limit=[0]
    for ind in range (len(centercolor)-1):
        limit.append(int((centercolor[ind]+centercolor[ind+1])/2))
    limit.append(255)
    return limit

#find the colors
def FindingQ (limit:list[int],instancesOfPixel:np.ndarray)->list[int]:
    mid=[]
    for ind in range (len(limit)-1):
        thepix=limit[ind]
        sumpixcolor=0
        sum=0
        while thepix<limit[ind+1]:
            num=instancesOfPixel[thepix]
            sumpixcolor+=num
            thepix+=1
            sum+=num*thepix
        mid.append(int(sum/sumpixcolor))
    return mid

#calculation MSE error
def mse(prev_img: np.ndarray, img: np.ndarray, numPixels: int) -> float:
    ans = (np.square(((pow(prev_img-img, 2)).astype(int)).sum()))/numPixels

    return ans


