#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:56:41 2019

@author: sumavenugopal
"""
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


#------------------------------------------------------------------------
#VECTOR QUANTIZATION - DATA PROCESSING
#------------------------------------------------------------------------
#function to read each file / signal      
def readOneSignalFile(filename): 
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
        data = np.array(data, dtype=np.int64)
    fp.close()
    return (data)

#split each signal using sample size&flatten; with or without overlap 
def singleSignalSplit(singleSignalArr,sampleSize,overlapPrcnt):
    ovrLapPos = int(sampleSize * overlapPrcnt)
    s = singleSignalArr
    n = int(len(s) / sampleSize )
    startPos = 0
    endPos = sampleSize
    flatSig = []
    for i in range(n):
        if i ==0:
            q = s[startPos:endPos][:]
        else:
            q = s[startPos-ovrLapPos:endPos-ovrLapPos][:]
        k = q.ravel(order='C')
        flatSig.append(k)
        startPos += sampleSize
        endPos += sampleSize
    flatSigArr = np.array(flatSig)
    return (flatSigArr)

#combine all flattened signls per activity
def allSignalsPerActvRszd (inpFileLst,sampleSize,overLapPrcnt):    
    perActSigArr = []
    for file in inpFileLst:
        singleSignalArr = readOneSignalFile(file)
        resizedSingSigArr = singleSignalSplit(singleSignalArr,sampleSize,overLapPrcnt)
        perActSigArr.append(resizedSingSigArr)
    allSignalPerActvRszdArr = np.concatenate( perActSigArr, axis=0 )
    return(allSignalPerActvRszdArr)

#combining all flattened signals for all activities
def allSignalsAllActvRszd (allSignalPerActvRszdArr):    
    finalSampleArrTemp = []
    finalSampleArrTemp.append(allSignalPerActvRszdArr)
    finalSampleArr = np.concatenate( finalSampleArrTemp, axis=0 )
    return(finalSampleArr)
    
#-------------------------------------------------------------------------
#VECTOR QUANTIZATION - DICTIONARY INPUT
#-------------------------------------------------------------------------
# returns the final flattened all files/signals array input to the dictionary model(fed to KMeans)
def allDataFilesLoad(sampleSize,inpPathList,overLapPrcnt):
    finalTemp = []
    for i in range(len(inpPathList)):
        inpPath = inpPathList[i]
        inpFileLst = []
        for root, dirs, files in os.walk(inpPath):
            for name in files:
                inpFileLst.append(os.path.join(root,name))        
        allSignalsRszdArr = allSignalsPerActvRszd (inpFileLst,sampleSize,overLapPrcnt)
        finalTemp.append(allSignalsRszdArr)
    return (finalTemp)

#---------------------------------------
#VECTOR QUANTIZATION - HISTOGRAM
#---------------------------------------
#returns histogram feature array for each class / activity
def returnFeatureHistArrPerClass(classInpPath,sampleSize,clsLabel,numClus,\
                                 dictModel,overLapPrcnt):
    for i in range(len(classInpPath)):
        classInpFileLst = []
        for root, dirs, files in os.walk(classInpPath):
            for name in files:
                classInpFileLst.append(os.path.join(root,name))
    clsFeatArr = []
    for file in range(len(classInpFileLst)):         
        filename = classInpFileLst[file]
        singleSignalArr = readOneSignalFile(filename)
        resizedSingSigArr = singleSignalSplit(singleSignalArr,sampleSize,overLapPrcnt)
        labls = dictModel.predict(np.array(resizedSingSigArr)) #fitting the KMeans dictionary
        b = np.arange(0,numClus+1)
        histFeatArr,binArr  = np.histogram(labls, bins=b, density=False)
        histFeatArrLbld = np.append(histFeatArr, np.array([clsLabel]), axis = 0)
        clsFeatArr.append(histFeatArrLbld)
        retClsFeatArr = np.array(clsFeatArr)
    return retClsFeatArr
#returns labelled histogram feature array for ALL  classes / activities
def returnLabldFtrMatrixAllClasses(inpPathList,sampleSize,numClus,dictModel,overLapPrcnt):
    labldFtrMatrix = []
    for i in range(len(inpPathList)):
        clsFeatMtx = returnFeatureHistArrPerClass(inpPathList[i],sampleSize,i,numClus,dictModel,overLapPrcnt)
        labldFtrMatrix.append(np.array(clsFeatMtx))
        labldFtrMatrixAllClasses = np.asarray(labldFtrMatrix)
    return np.array(labldFtrMatrixAllClasses)
#returns labelled mean histogram feature array for ALL  classes / activities
def classMeanHistogram(labldFtrMatrixAllClasses,actFoldList):
    allClassesMeanHistArr = []
    for i in range(len(labldFtrMatrixAllClasses)):
        classHistFeatArr = labldFtrMatrixAllClasses[i][:,:-1]
        classMeanHistArr = np.mean(classHistFeatArr,axis = 0)
        allClassesMeanHistArr.append(classMeanHistArr)
    return(allClassesMeanHistArr)
    

#----------------------------------------------------------------------
#FINAL CLASSIFICATION - Random Forest
#---------------------------------------------------------------------- 
#three fold data split - all classes       
def threeFoldDataSplitAllClasses(labldFtrMatrixAllClasses):
    threeFoldSplitDataArrAllClasses = []
    for i in range(14):
        classArr = labldFtrMatrixAllClasses[i]
        np.take(classArr,np.random.permutation(classArr.shape[0]),axis=0,out=classArr)        
        classThreeFoldDataArr = []
        oneFold, twoFold, threeFold = np.array_split(classArr,3)
        classThreeFoldDataArr.append(oneFold)
        classThreeFoldDataArr.append(twoFold)
        classThreeFoldDataArr.append(threeFold) 
        threeFoldSplitDataArrAllClasses.append(classThreeFoldDataArr)
    return (threeFoldSplitDataArrAllClasses)
#Data Test - Train Split - all classes    
def dataTestTrainSplit(threeFoldSplitDataAllClasses,foldNum):
    if foldNum == 0:
        f = [0,1,2]
    if foldNum == 1:
        f = [1,2,0]
    if foldNum == 2:
        f = [2,0,1]               
    finalTrain = []
    finalTest = []
    for i in range (14):
        chunkOne = threeFoldSplitDataAllClasses[i][f[0]]
        chunkTwo= threeFoldSplitDataAllClasses[i][f[1]]
        chunkThree= threeFoldSplitDataAllClasses[i][f[2]]
        Train = np.concatenate((chunkOne,chunkTwo), axis=0)
        Test = np.array(chunkThree)
        for i in range(len(Train)):
            finalTrain.append(Train[i])
        for x in range(len(Test)):
            finalTest.append(Test[x])    
    XTrain = np.array(finalTrain)[:,:-1]    
    YTrain = np.array(finalTrain)[:,-1]
    XTest = np.array(finalTest)[:,:-1]
    YTest = np.array(finalTest)[:,-1]   
    return XTrain,YTrain,XTest,YTest
#Random Forest Classifier  - Returns accuracy, confusion matrix       
def classificationNFoldRandomForest(n,labldFtrMatrixAllClasses) :
    scoreArr = []
    confMatArr = []
    threeFoldSplitDataArrAllClasses = threeFoldDataSplitAllClasses\
    (labldFtrMatrixAllClasses)
    for i in range(n):
        XTrain,YTrain,XTest,YTest = \
        dataTestTrainSplit(threeFoldSplitDataArrAllClasses,i)
        clf = RandomForestClassifier(n_estimators=50,max_depth=32,\
                                     max_features='auto', n_jobs=-1) 
        clf.fit(XTrain, YTrain)
        YPred = clf.predict(XTest)
        confMatrix = confusion_matrix(YTest, YPred)
        accuracyPercent = (clf.score(XTest, YTest))*100
        scoreArr.append(accuracyPercent)
        confMatArr.append(confMatrix)
        print(YPred)
    return scoreArr, confMatArr


#Plot Confusion Matrix
def plotconfusionMatrix(confMatrix,actFoldList):            
    dfConfMatx = pd.DataFrame(confMatrix, range(14),range(14))
    sns.set(font_scale=1)#for label size
    dfConfMatx.index.name = 'True Activity Class Labels'
    dfConfMatx.columns.name = 'Predicted Activity Class Labels'   
    sns.heatmap(dfConfMatx, xticklabels=actFoldList, yticklabels = actFoldList, annot=True,annot_kws={"size": 12},cmap='Greens')# font size


#Main function        
def main():
    
    
    inFolder = './HMP_Dataset/'
    actFoldList = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass',\
                  'Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair',\
                  'Standup_chair','Use_telephone','Walk']      
    inpPathList = []
    for i in range(len(actFoldList)):
        inpPathList.append(inFolder+actFoldList[i]+'/')
        

    #Data pre-processing for dictionary
    sampleSize = 32  #can be tuned
    overLapPrcnt = .90     #can be tuned
    allDataArr = allDataFilesLoad(sampleSize,inpPathList,overLapPrcnt)
    allDataArrForDict = np.concatenate( allDataArr, axis=0 )
 
    
    # Create a Dictionary of size numClus using KMeans
    numClus = 70 #can be tuned
    dictModel = KMeans(n_clusters=numClus)   
    dictModel.fit(allDataArrForDict )


    #final labelled feature matrix - list of 14 individual feature matrices
    labldFtrMatrixAllClasses = returnLabldFtrMatrixAllClasses(inpPathList,sampleSize,numClus,dictModel,overLapPrcnt)
    
    #Mean Histogram Plots
    allClassesMeanHistArr = classMeanHistogram(labldFtrMatrixAllClasses,actFoldList)
    
    for i in range(len(allClassesMeanHistArr)):
        fig = plt.figure()
        plt.bar(np.arange(len(allClassesMeanHistArr[i])), allClassesMeanHistArr[i])
        plt.xlabel('ClusterCenters', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title(actFoldList[i].strip('/'))
        plt.plot()
        plt.show()        
        fig.savefig(str(i))
        
    #Calling the RF classifier - 3 Fold
    nFold = 3
    nFoldAccuracyArray, confMatArr = classificationNFoldRandomForest\
    (nFold,labldFtrMatrixAllClasses)     
    print('Prediction Accuracies - ' , nFoldAccuracyArray)
    
    #Plotting confusion matrix for the best accuracy across 3 runs
    bestAccIndx = np.argmax(nFoldAccuracyArray, axis=0)
    bestAccConfMatrix = confMatArr[bestAccIndx]     
    plotconfusionMatrix(bestAccConfMatrix,actFoldList)
    
    
if __name__ == "__main__":
    main()

        
        

