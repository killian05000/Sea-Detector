#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import Reader as re
import Images.DataGetter as dg
import time
import OutputJson as jout
from sklearn.metrics import accuracy_score


import DataEntry
import Algorithm
import Moyenne

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

    for i in range(10):
        print("-----Start",i,"-----")
        time1=time.clock()

        listeAlgo = Algorithm.Algos()
        data_I, target_I = DataEntry.getDataImageVec(pathTrain)
        data_H, target_H = DataEntry.getDataHistogramme(pathTrain)
        data_S, target_S = DataEntry.getFastDataSobel(pathTrain)

        ########### Algos lancé sur un vecteur d'image
        listeAlgo.setDataTarget(data_I, target_I)
        tmp_bayes_image = listeAlgo.Bayes('Bayes_ImageVec.sav')
        tmp_ada_image = listeAlgo.Ada_boost('AdaBoost_ImageVec.sav')
        tmp_svm_image = listeAlgo.svmImgvec('Svm_Imagevec.sav')

        tmp_accuracy_bayes_i = accuracy_score(listeAlgo.target_test, tmp_bayes_image)
        #print(tmp_accuracy_bayes_i)
        tmp_accuracy_ada_i = accuracy_score(listeAlgo.target_test,tmp_ada_image)
        #print(tmp_accuracy_ada_i)
        tmp_accuracy_svm_i = accuracy_score(listeAlgo.target_test, tmp_svm_image)
        #print(tmp_accuracy_svm_i)



        ########### Algos lancé sur un histogramme de couleur
        listeAlgo.setDataTarget(data_H, target_H)
        tmp_bayes_histo = listeAlgo.Bayes('Bayes_Histo.sav')
        tmp_ada_histo = listeAlgo.Ada_boost('AdaBoost_Histo.sav')
        tmp_svm_histo = listeAlgo.svmImgvec('Svm_Histo.sav')

        tmp_accuracy_bayes_h = accuracy_score(listeAlgo.target_test, tmp_bayes_histo)
        #print(tmp_accuracy_bayes_h)
        tmp_accuracy_ada_h = accuracy_score(listeAlgo.target_test, tmp_ada_histo)
        #print(tmp_accuracy_ada_h)
        tmp_accuracy_svm_h = accuracy_score(listeAlgo.target_test, tmp_svm_histo)
        #print(tmp_accuracy_svm_h)



        ########### Algos lancé sur un vecteur de contour d'image
        listeAlgo.setDataTarget(data_S, target_S)
        tmp_bayes_sobel = listeAlgo.Bayes('Bayes_Sobel.sav')
        tmp_ada_sobel = listeAlgo.Ada_boost('AdaBoost_Sobel.sav')
        tmp_svm_sobel = listeAlgo.svmImgvec('Svm_Sobel.sav')

        tmp_accuracy_bayes_s = accuracy_score(listeAlgo.target_test, tmp_bayes_sobel)
        #print(tmp_accuracy_bayes_s)
        tmp_accuracy_ada_s = accuracy_score(listeAlgo.target_test, tmp_ada_sobel)
        #print(tmp_accuracy_ada_s)
        tmp_accuracy_svm_s = accuracy_score(listeAlgo.target_test, tmp_svm_sobel)
        #print(tmp_accuracy_svm_s)




        vector = [tmp_bayes_image*(tmp_accuracy_bayes_i),
                  tmp_ada_image*(tmp_accuracy_ada_i),
                  tmp_svm_image*(tmp_accuracy_svm_i),
                  tmp_bayes_histo*(tmp_accuracy_bayes_h),
                  tmp_ada_histo*(tmp_accuracy_ada_h),
                  tmp_svm_histo*(tmp_accuracy_svm_h),
                  tmp_bayes_sobel*(tmp_accuracy_bayes_s),
                  tmp_ada_sobel*(tmp_accuracy_ada_s),
                  tmp_svm_sobel*(tmp_accuracy_svm_s),]
        '''

        vector = [1*tmp_bayes_image,
                  1*tmp_ada_image,
                  1*tmp_svm_image,
                  1*tmp_bayes_histo,
                  1*tmp_ada_histo,
                  1*tmp_svm_histo,
                  1*tmp_bayes_sobel,
                  1*tmp_ada_sobel,
                  1*tmp_svm_sobel,]
        '''
        print('\n')
        #for i in range(len(vector)):
            #print(vector[i],'\n')
        voteResult=Moyenne.Votes(vector)
        accuracy+=accuracy_score(listeAlgo.target_test, voteResult)

        print("Bayes Image : ",tmp_accuracy_bayes_i)
        print("Ada Image :",tmp_accuracy_ada_i)
        print("Svm Image :",tmp_accuracy_svm_i)
        print("Bayes Histo :",tmp_accuracy_bayes_h)
        print("Ada Histo :",tmp_accuracy_ada_h)
        print("Svm Histo :",tmp_accuracy_svm_h)
        print("Bayes Sobel :",tmp_accuracy_bayes_s)
        print("Ada Sobel :",tmp_accuracy_ada_s)
        print("Svm Sobel :",tmp_accuracy_svm_s)

        '''
        print("Bayes Image : ",tmp_bayes_image)
        print("Ada Image :",tmp_ada_image)
        print("Svm Image :",tmp_svm_image)
        print("Bayes Histo :",tmp_bayes_histo)
        print("Ada Histo :",tmp_ada_histo)
        print("Svm Histo :",tmp_svm_histo)
        print("Bayes Sobel :",tmp_bayes_sobel)
        print("Ada Sobel :",tmp_ada_sobel)
        print("Svm Sobel :",tmp_svm_sobel)
        '''
        accuracy_bayes_Img += tmp_accuracy_bayes_i
        accuracy_ada_Img += tmp_accuracy_ada_i
        accuracy_svm_Img += tmp_accuracy_svm_i
        accuracy_bayes_Histo += tmp_accuracy_bayes_h
        accuracy_ada_Histo += tmp_accuracy_ada_h
        accuracy_svm_Histo += tmp_accuracy_svm_h
        accuracy_bayes_Sobel += tmp_accuracy_bayes_s
        accuracy_ada_Sobel += tmp_accuracy_ada_s
        accuracy_svm_Sobel += tmp_accuracy_svm_s
        '''
        accuracy_bayes_Img += tmp_bayes_image
        accuracy_ada_Img += tmp_ada_image
        accuracy_svm_Img += tmp_svm_image
        accuracy_bayes_Histo += tmp_bayes_histo
        accuracy_ada_Histo += tmp_ada_histo
        accuracy_svm_Histo += tmp_svm_histo
        accuracy_bayes_Sobel += tmp_bayes_sobel
        accuracy_ada_Sobel += tmp_ada_sobel
        accuracy_svm_Sobel += tmp_svm_sobel
        '''
        print("Resultat du vote ",nbIter,": ",accuracy_score(listeAlgo.target_test, voteResult),'\n')
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
    dataImageVec,fileNames = DataEntry.dataImagevecPredict(pathPredict)
    resultBayesImageVec=model.load_Model('Bayes_ImageVec.sav',dataImageVec)
    resultAdaImageVec=model.load_Model('AdaBoost_ImageVec.sav',dataImageVec)
    resultSvmImageVec=model.load_Model('Svm_ImageVec.sav',dataImageVec)

    dataHisto,fileNames = DataEntry.dataHistogrammePredict(pathPredict)
    resultBayesHisto=model.load_Model('Bayes_Histo.sav', dataHisto)
    resultAdaHisto=model.load_Model('AdaBoost_Histo', dataHisto)
    resultSvmHisto=model.load_Model('Svm_Histo', dataHisto)

    dataSobel,fileNames = DataEntry.dataSobelPredict(pathPredict)
    resultBayesSobel=model.load_Model('Bayes_Sobel.sav',dataSobel)
    resultAdaSobel=model.load_Model('AdaBoost_Sobel.sav',dataSobel)
    resultSvmSobel=model.load_Model('Svm_Sobel.sav',dataSobel)



    vector=[resultBayesImageVec,resultAdaImageVec, resultSvmImageVec,
            resultBayesHisto, resultAdaHisto, resultSvmHisto,
            resultBayesSobel, resultAdaSobel, resultSvmSobel]

    voteResult=Moyenne.Votes(vector)
    for i in range(len(vector)):
        print(vector[i],'\n')

    print(result)

    JsonOut.outPutJson(fileList, pathPredict, voteResult)
