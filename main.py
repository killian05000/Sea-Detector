#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import Images.DataGetter as dg
import time
import OutputJson as jout
from sklearn.metrics import accuracy_score


import DataEntry
import Algorithm
import Moyenne
import Model
import OutputJson as Json

ap=argparse.ArgumentParser()
ap.add_argument("-f","--fit",required= False, help="Path to Data Folder")
ap.add_argument("-p","--predict",required=False, help="Path to Data Folder")
args=vars(ap.parse_args())

pathTrain=args["fit"]
pathPredict=args["predict"]


accuracy_bayes_Img = 0
accuracy_ada_Img = 0
accuracy_svm_Img = 0
accuracy_bayes_Histo = 0
accuracy_ada_Histo = 0
accuracy_svm_Histo = 0
accuracy_bayes_Sobel = 0
accuracy_ada_Sobel = 0
accuracy_svm_Sobel = 0

tmp_bayes_image = 0
tmp_ada_image = 0
tmp_svm_image = 0
tmp_bayes_histo = 0
tmp_ada_histo = 0
tmp_svm_histo = 0
tmp_bayes_sobel = 0
tmp_ada_sobel = 0
tmp_svm_sobel = 0

accuracy = 0

if(pathTrain is not None):
    print("Demarrage")
    timeStart=time.clock()
    nbIter=0

    for i in range(20):
        print("-----Start",i,"-----")
        time1=time.clock()

        listeAlgo = Algorithm.Algos()
        data_I, target_I = DataEntry.getDataImageVec(pathTrain)
        data_S, target_S = DataEntry.getFastDataSobel(pathTrain)
        data_H, target_H = DataEntry.getDataHistogramme(pathTrain)


        ########### Algos lancé sur un vecteur d'image
        listeAlgo.setDataTarget(data_I, target_I)
        tmp_bayes_image = listeAlgo.Bayes('Bayes_ImageVec.sav')
        tmp_ada_image = listeAlgo.Ada_boost('AdaBoost_ImageVec.sav')
        tmp_svm_image = listeAlgo.svmImgvec('Svm_ImageVec.sav')
        tmp_accuracy_bayes_i = accuracy_score(listeAlgo.target_test, tmp_bayes_image)
        tmp_accuracy_ada_i = accuracy_score(listeAlgo.target_test,tmp_ada_image)
        tmp_accuracy_svm_i = accuracy_score(listeAlgo.target_test, tmp_svm_image)


        ########### Algos lancé sur un vecteur de contour d'image
        listeAlgo.setDataTarget(data_S, target_S)
        tmp_bayes_sobel = listeAlgo.Bayes('Bayes_Sobel.sav')
        tmp_ada_sobel = listeAlgo.Ada_boost('AdaBoost_Sobel.sav')
        tmp_svm_sobel = listeAlgo.svmImgvec('Svm_Sobel.sav')
        tmp_accuracy_bayes_s = accuracy_score(listeAlgo.target_test, tmp_bayes_sobel)
        tmp_accuracy_ada_s = accuracy_score(listeAlgo.target_test, tmp_ada_sobel)
        tmp_accuracy_svm_s = accuracy_score(listeAlgo.target_test, tmp_svm_sobel)


        ########### Algos lancé sur un histogramme de couleur
        listeAlgo.setDataTarget(data_H, target_H)
        tmp_bayes_histo = listeAlgo.Bayes('Bayes_Histo.sav')
        tmp_ada_histo = listeAlgo.Ada_boost('AdaBoost_Histo.sav')
        tmp_svm_histo = listeAlgo.svmImgvec('Svm_Histo.sav')
        tmp_accuracy_bayes_h = accuracy_score(listeAlgo.target_test, tmp_bayes_histo)
        tmp_accuracy_ada_h = accuracy_score(listeAlgo.target_test, tmp_ada_histo)
        tmp_accuracy_svm_h = accuracy_score(listeAlgo.target_test, tmp_svm_histo)


        print("Bayes Image : ",tmp_accuracy_bayes_i)
        print("Ada Image :",tmp_accuracy_ada_i)
        print("Svm Image :",tmp_accuracy_svm_i)
        print("Bayes Histo :",tmp_accuracy_bayes_h)
        print("Ada Histo :",tmp_accuracy_ada_h)
        print("Svm Histo :",tmp_accuracy_svm_h)
        print("Bayes Sobel :",tmp_accuracy_bayes_s)
        print("Ada Sobel :",tmp_accuracy_ada_s)
        print("Svm Sobel :",tmp_accuracy_svm_s)

        accuracy_bayes_Img += tmp_accuracy_bayes_i
        accuracy_ada_Img += tmp_accuracy_ada_i
        accuracy_svm_Img += tmp_accuracy_svm_i
        accuracy_bayes_Histo += tmp_accuracy_bayes_h
        accuracy_ada_Histo += tmp_accuracy_ada_h
        accuracy_svm_Histo += tmp_accuracy_svm_h
        accuracy_bayes_Sobel += tmp_accuracy_bayes_s
        accuracy_ada_Sobel += tmp_accuracy_ada_s
        accuracy_svm_Sobel += tmp_accuracy_svm_s

        time2=time.clock()
        print("Execution Time ",i," : ",round((time2-time1),4),'sec\n')
        nbIter+=1

    print("-----Averages on ",nbIter," Samples-----")
    print("Bayses ImgVec = ", accuracy_bayes_Img/nbIter)
    print("AdaBoost ImgVec = ", accuracy_ada_Img/nbIter)
    print("Svm Image ImgVec = ", accuracy_svm_Img/nbIter)
    print("Bayes Histo = ", accuracy_bayes_Histo/nbIter)
    print("AdaBoost Histo = ", accuracy_ada_Histo/nbIter)
    print("Svm Histo = ", accuracy_svm_Histo/nbIter)
    print("Bayes Sobel = ", accuracy_bayes_Sobel/nbIter)
    print("AdaBoost Sobel = ", accuracy_svm_Sobel/nbIter)
    print("Svm Sobel = ", accuracy_svm_Sobel/nbIter)

    timeEnd=time.clock()
    print("Final Execution time : ",round((timeEnd-timeStart),4),'sec\n')

########################################################


if(pathPredict is not None):
    print("--Predict start--")

    dataHisto,fileNames = DataEntry.dataHistogrammePredict(pathPredict)
    resultBayesHisto=Model.load_Model('Bayes_Histo.sav', dataHisto)
    resultAdaHisto=Model.load_Model('AdaBoost_Histo.sav', dataHisto)
    resultSvmHisto=Model.load_Model('Svm_Histo.sav', dataHisto)
    print("----load Histo Models done----")


    dataImageVec,fileNames = DataEntry.dataImageVecPredict(pathPredict)
    resultBayesImageVec=Model.load_Model('Bayes_ImageVec.sav',dataImageVec)
    resultAdaImageVec=Model.load_Model('AdaBoost_ImageVec.sav',dataImageVec)
    resultSvmImageVec=Model.load_Model('Svm_ImageVec.sav',dataImageVec)
    print("----load Image Models done----")


    dataSobel,fileNames = DataEntry.dataSobelPredict(pathPredict)
    resultBayesSobel=Model.load_Model('Bayes_Sobel.sav',dataSobel)
    resultAdaSobel=Model.load_Model('AdaBoost_Sobel.sav',dataSobel)
    resultSvmSobel=Model.load_Model('Svm_Sobel.sav',dataSobel)
    print("----load Sobel Models done----")

    vector=[resultBayesImageVec,resultAdaImageVec, resultSvmImageVec,
            resultBayesHisto, resultAdaHisto, resultSvmHisto,
            resultBayesSobel, resultAdaSobel, resultSvmSobel]

    voteResult=Moyenne.Votes(vector)
    #for i in range(len(vector)):
        #print(vector[i],'\n')
    #print(voteResult)

    Json.outPutJson(fileNames, pathPredict, voteResult)
