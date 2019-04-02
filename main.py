#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import Reader as re
import Images.DataGetter as dg
import time
import OutputJson as jout


import DataEntry
import Algorithm

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

tmp_bayes_image = 0
tmp_ada_image = 0
tmp_svm_image = 0
tmp_bayes_histo = 0
tmp_ada_histo = 0
tmp_svm_histo = 0


if(pathTrain is not None):
    print("Demarrage")
    timeStart=time.clock()
    nbIter=0

    for i in range(20):
        print("-----Start",i,"-----")
        time1=time.clock()

        listeAlgo = Algorithm.Algos()
        data_I, target_I = DataEntry.getDataImageVec(pathTrain)
        data_H, target_H = DataEntry.getDataHistogramme(pathTrain)

        ########### Algos lancé sur un vecteur d'image
        listeAlgo.setDataTarget(data_I, target_I)
        tmp_bayes_image = listeAlgo.Bayes('Bayes_ImageVec.sav')
        tmp_ada_image = listeAlgo.Ada_boost('AdaBoost_ImageVec.sav')
        tmp_svm_image = listeAlgo.svmImgvec('Svm_Imagevec.sav')

        ########### Algos lancé sur un histogramme de couleur
        listeAlgo.setDataTarget(data_H, target_H)
        tmp_bayes_histo = listeAlgo.Bayes('Bayes_Histo.sav')
        tmp_ada_histo = listeAlgo.Ada_boost('AdaBoost_Histo.sav')
        tmp_svm_histo = listeAlgo.svmImgvec('Svm_histo.sav')



        print("Bayes Image : ",tmp_bayes_image)
        print("Ada Image :",tmp_ada_image)
        print("Svm Image :",tmp_svm_image)
        print("Bayes Histo :",tmp_bayes_histo)
        print("Ada Histo :",tmp_ada_histo)
        print("Svm Histo :",tmp_svm_histo)

        accuracy_bayes_Img += tmp_bayes_image
        accuracy_ada_Img += tmp_ada_image
        accuracy_svm_Img += tmp_svm_image
        accuracy_bayes_Histo += tmp_bayes_histo
        accuracy_ada_Histo += tmp_ada_histo
        accuracy_svm_Histo += tmp_ada_histo

        time2=time.clock()
        print("Time ",i," : ",round((time2-time1),4),'sec\n')
        nbIter+=1

    print("-----",nbIter," Samples-----")
    print("Bayses ImgVec = ", accuracy_bayes_Img/nbIter)
    print("AdaBoost ImgVec = ", accuracy_ada_Img/nbIter)
    print("Svm Image ImgVec = ", accuracy_svm_Img/nbIter)
    print("Bayes Histo = ", accuracy_bayes_Histo/nbIter)
    print("AdaBoost Histo = ", accuracy_ada_Histo/nbIter)
    print("Svm Histo = ", accuracy_svm_Histo/nbIter)

    timeEnd=time.clock()
    print("Final time : ",round((timeEnd-timeStart),4),'sec\n')

########################################################


if(pathPredict is not None):
    data,nameFiles = re.dataHistogrammePredict(pathPredict)
    result=model.load_Model('Bayes.sav', data)
    print(result)
    jout.outPutJson(nameFiles,result)
