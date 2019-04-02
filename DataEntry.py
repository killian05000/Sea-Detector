from os import listdir
import cv2
import numpy
from PIL import Image

########################################################

def getDataImageVec(dirPath):
    data=[]
    target=[]

    seaPath = dirPath+"/Mer/"
    otherPath = dirPath+"/Ailleurs/"

    seaFileList = listdir(seaPath)
    otherFileList = listdir(otherPath)
    for iSea in range(0,len(seaFileList)-1):
        img_path = ""+ seaPath + seaFileList[iSea]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        data.append(resized.flatten())
        target.append(1)

    for iOther in range(0,len(otherFileList)-1):
        img_path = ""+otherPath+otherFileList[iOther]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        data.append(resized.flatten())
        target.append(-1)


    data=numpy.asarray(data)
    target=numpy.asarray(target)
    #print(data.shape)
    #print(target.shape)
    return data,target

########################################################

def getDataSobel(dirPath):
    data=[]
    target=[]

    seaPath = dirPath+"/Mer/"
    otherPath = dirPath+"/Ailleurs/"

    seaFileList = listdir(seaPath)
    otherFileList = listdir(otherPath)
   # print(len(seaFileList))
    for iSea in range(0,len(seaFileList)-1):
        img_path = ""+ seaPath + seaFileList[iSea]
        img = cv2.imread(img_path,1)
        img = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        sobely = cv2.Sobel(img,cv2.cv2_64F,1,0,ksize=5)
        sobelx = cv2.Sobel(img,cv2.cv2_64F,1,0,ksize=5)
        sobel = numpy.concatenate((sobelx,sobely),axis = 1)
        d = numpy.concatenate((sobel,img),axis = 1)
        data.append(d.flatten())
        target.append(1)

    for iOther in range(0,len(otherFileList)-1):
        img_path = ""+otherPath+otherFileList[iOther]
        img = cv2.imread(img_path,1)
        img = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        sobely = cv2.Sobel(img,cv2.cv2_64F,1,0,ksize=5)
        sobelx = cv2.Sobel(img,cv2.cv2_64F,1,0,ksize=5)
        d = numpy.concatenate((sobel,img),axis = 1)
        data.append(d.flatten())
        target.append(-1)

    data=numpy.asarray(data)
    target=numpy.asarray(target)
    return data,target

########################################################

def getFastDataSobel(dirPath):
    data=[]
    target=[]

    seaPath = dirPath+"/Mer/"
    otherPath = dirPath+"/Ailleurs/"

    seaFileList = listdir(seaPath)
    otherFileList = listdir(otherPath)
   # print(len(seaFileList))
    for iSea in range(0,len(seaFileList)-1):
        img_path = ""+ seaPath + seaFileList[iSea]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        data.append(resized.flatten())
        target.append(1)

    for iOther in range(0,len(otherFileList)-1):
        img_path = ""+otherPath+otherFileList[iOther]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        data.append(resized.flatten())
        target.append(-1)

    data=numpy.asarray(data)
    target=numpy.asarray(target)
    return data,target

########################################################

def getDataHistogramme(dirPath):

    data=[]
    target=[]

    seaPath = dirPath+"/Mer/"
    otherPath = dirPath+"/Ailleurs/"

    seaFileList = listdir(seaPath)
    otherFileList = listdir(otherPath)

    for iSea in range(0,len(seaFileList)-1):
        img_path = ""+ seaPath + seaFileList[iSea]
        data.append(VectorHistogrammeC(img_path))
        target.append(1)

    for iOther in range(0,len(otherFileList)-1):
        img_path = ""+otherPath+otherFileList[iOther]
        data.append(VectorHistogrammeC(img_path))
        target.append(-1)

    data=numpy.asarray(data)
    target=numpy.asarray(target)

    return data,target


def VectorHistogrammeC(image_path):
    chans = cv2.split(cv2.imread(image_path))
    img = Image.open(image_path)
    x,y = img.size
    colors = ('b','g','r')
    features = []

    for (chan, color) in zip(chans, colors):
        	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        	features.extend(hist.flatten())
    return numpy.array(features).flatten()/(x*y)

########################################################
