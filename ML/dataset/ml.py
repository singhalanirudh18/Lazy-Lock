#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

"""
IMPORTS 
"""
from skimage import data, io
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import hog 
from skimage.transform import resize

import pickle
import numpy as np
import pandas as pd

import glob
import random
import csv
from os import listdir
from os.path import isfile, join

from sklearn.ensemble import  RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import json
import time
import gzip

#given a list of filenames return s a dictionary of images 
def getfiles(filenames):
    dir_files = {}
    for x in filenames:
        dir_files[x]=io.imread(x)
    return dir_files

#return hog of a particular image vector
def convertToGrayToHOG(imgVector):
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)

#takes returns cropped image 
def crop(img,x1,x2,y1,y2):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((128,128)))#resize
    return crp

#save classifier
def dumpclassifier(filename,model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)    

#load classifier
def loadClassifier(picklefile):
    fd = open(picklefile, 'r+')
    model = pickle.load(fd)
    fd.close()
    return model

"""
This function randomly generates bounding boxes 
Return: hog vector of those cropped bounding boxes along with label 
Label : 1 if hand ,0 otherwise 
"""
def buildhandnothand_lis(frame,imgset):
    poslis =[]
    neglis =[]

    for nameimg in frame.image:
        tupl = frame[frame['image']==nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        
        arg1 = [x_tl,y_tl,conf,side,side]
        poslis.append(convertToGrayToHOG(crop(imgset[nameimg],x_tl,x_tl+side,y_tl,y_tl+side)))
        while dic[0] <= 1 or dic[1] < 1:
            x = random.randint(0,320-side)
            y = random.randint(0,240-side) 
            crp = crop(imgset[nameimg],x,x+side,y,y+side)
            hogv = convertToGrayToHOG(crp)
            arg2 = [x,y, conf, side, side]
            
            z = overlapping_area(arg1,arg2)
            if dic[0] <= 1 and z <= 0.5:
                neglis.append(hogv)
                dic[0] += 1
            if dic[0]== 1:
                break
    label_1 = [1 for i in range(0,len(poslis)) ]
    label_0 = [0 for i in range(0,len(neglis))]
    label_1.extend(label_0)
    poslis.extend(neglis)
    return poslis,label_1

#returns imageset and bounding box for a list of users 
def train_binary(train_list, data_directory):
    frame = pd.DataFrame()
    list_ = []
    for user in train_list:
        list_.append(pd.read_csv(data_directory+user+'/'+user+'_loc.csv',index_col=None,header=0))
    frame = pd.concat(list_)
    frame['side']=frame['bottom_right_x']-frame['top_left_x']
    frame['hand']=1

    imageset = getfiles(frame.image.unique())

    #returns actual images and dataframe 
    return imageset,frame

#loads data for binary classification (hand/not-hand)
def load_binary_data(user_list, data_directory):
    data1,df  =train_binary(user_list, data_directory) # data 1 - actual images , df is actual bounding box
    
    # third return, i.e., z is a list of hog vecs, labels
    z = buildhandnothand_lis(df,data1)
    return data1,df,z[0],z[1]


#loads data for multiclass 
def get_data(user_list, img_dict, data_directory):
    X = []
    Y = []

    for user in user_list:
        user_images = glob.glob(data_directory+user+'/*.jpg')

        boundingbox_df = pd.read_csv(data_directory+user+'/'+user+'_loc.csv')
        
        for rows in boundingbox_df.iterrows():
            cropped_img = crop(img_dict[rows[1]['image']], rows[1]['top_left_x'], rows[1]['bottom_right_x'], rows[1]['top_left_y'], rows[1]['bottom_right_y'])
            hogvector = convertToGrayToHOG(cropped_img)
            X.append(hogvector.tolist())
            Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y


train_list=["user_5","user_4","user_3","user_7","user_9","user_10"]
predict_list=["user_6"]
data_directory='/home/anirudh18/Documents/itsp/ml/dataset/'
imageset,frame = train_binary(train_list,data_directory)
imageset1,frame=train_binary(predict_list,data_directory)
#print imageset
X,Y=get_data(train_list,imageset,data_directory)
X1,Y1=get_data(predict_list,imageset1,data_directory)
#print Y
#i=0
"""
for x in Y:
	i+=1
	print x,Y[i]
	if(i==5) :
		break
"""
#print len(X)
#print len(Y)
svcmodel = SVC(kernel='linear', C=0.9, probability=True)
svcmodel.fit(X, Y)
predicted=svcmodel.predict(X1)
print predicted
print Y1
print("Classification report for classifier %s:\n%s\n"
      % (svcmodel, metrics.classification_report(Y1, predicted)))