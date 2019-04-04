from os import listdir
import cv2
import numpy
import math
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

########################################################
#######################TRAIN############################
########################################################

def getDataImageVec(dirPath):
    data=[]
    target=[]

    seaPath = dirPath+"/Mer/"
    otherPath = dirPath+"/Ailleurs/"

    seaFileList = listdir(seaPath)
    otherFileList = listdir(otherPath)
    for iSea in range(0,len(seaFileList)):
        img_path = ""+ seaPath + seaFileList[iSea]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        data.append(resized.flatten())
        target.append(1)

    for iOther in range(0,len(otherFileList)):
        img_path = ""+otherPath+otherFileList[iOther]
        img = cv2.imread(img_path,1)
        #print("size before:",len(img.flatten()))
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        #print("size after :",len(resized.flatten()))
        data.append(resized.flatten())
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

    for iSea in range(0,len(seaFileList)):
        img_path = ""+ seaPath + seaFileList[iSea]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        sobel = (shapeDetectionCV(resized,30))
        data.append(sobel.flatten())
        target.append(1)

    for iOther in range(0,len(otherFileList)):
        img_path = ""+otherPath+otherFileList[iOther]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        sobel = (shapeDetectionCV(resized,30))
        data.append(sobel.flatten())
        target.append(-1)

    data=numpy.asarray(data)
    target=numpy.asarray(target)
    return data,target

def shapeDetectionCV(img, treshold):
    column = img.shape[0]
    line = img.shape[1]
    #print(img.shape)
    for i in range(1,line-1):
        for j in range(1,column-1):
            p1 = img[j-1][i]
            p2 = img[j][i-1]
            p3 = img[j+1][i]
            p4 = img[j][i+1]
            nr = math.sqrt((p1[0]-p3[0])*(p1[0]-p3[0]) + (p2[0]-p4[0])*(p2[0]-p4[0]))
            ng = math.sqrt((p1[1]-p3[1])*(p1[1]-p3[1]) + (p2[1]-p4[1])*(p2[1]-p4[1]))
            nb = math.sqrt((p1[2]-p3[2])*(p1[2]-p3[2]) + (p2[2]-p4[2])*(p2[2]-p4[2]))
            n = (nr+ng+nb)/3
            if n < treshold:
                p = (255,255,255)
            else:
                p = (0,0,0)
            img[j-1][i-1] = p[0]
    return img;

########################################################

def getDataHistogramme(dirPath):

    data=[]
    target=[]

    seaPath = dirPath+"/Mer/"
    otherPath = dirPath+"/Ailleurs/"

    seaFileList = listdir(seaPath)
    otherFileList = listdir(otherPath)

    for iSea in range(0,len(seaFileList)):
        img_path = ""+ seaPath + seaFileList[iSea]
        data.append(VectorHistogrammeC(img_path))
        target.append(1)

    for iOther in range(0,len(otherFileList)):
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
######################PREDICT###########################
########################################################

def dataHistogrammePredict(dirPath):

    data=[]
    fileList=[]
    fileListDir=listdir(dirPath)

    for i in range(0,len(fileListDir)):
        img_path = ""+ dirPath +"/"+ fileListDir[i]
        data.append(VectorHistogrammeC(img_path))
        fileList.append(fileListDir[i])

    data=numpy.asarray(data)
    fileList=numpy.asarray(fileList)
    return data,fileList

########################################################

def dataSobelPredict (dirpath):
    data=[]
    fileName=[]
    fileList=listdir(dirpath)

    for i in range(0,len(fileList)):
        img_path = ""+ dirpath +"/"+ fileList[i]
        img = cv2.imread(img_path,1)
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        sobel = (shapeDetectionCV(resized,30))
        #print(len(sobel.flatten()))
        data.append(sobel.flatten())
        fileName.append(img_path)

    data=numpy.asarray(data)
    return data,fileName

########################################################

def dataImageVecPredict (dirpath):
    data=[]
    fileName=[]
    fileList=listdir(dirpath)
    for i in range(0,len(fileList)):
        img_path = ""+ dirpath +"/"+ fileList[i]
        img = cv2.imread(img_path,1)
        #print("size before:",len(img.flatten()))
        resized = cv2.resize(img, (126,90), interpolation = cv2.INTER_CUBIC)
        #print("size after :",len(resized.flatten()))
        data.append(resized.flatten())
        fileName.append(img_path)

    data=numpy.asarray(data)
    fileList=numpy.asarray(fileList)
    return data,fileName
