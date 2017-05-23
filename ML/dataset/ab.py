import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import acos,degrees
import math
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
    #print img
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((128,128)))#resize
    return crp

#save classifier
def angle(A,B,C) :
	a = math.sqrt((B[0]-C[0])*(B[0]-C[0]) + (B[1]-C[1])*(B[1]-C[1]))
	b = math.sqrt((A[0]-C[0])*(A[0]-C[0]) + (A[1]-C[1])*(A[1]-C[1]))
	c = math.sqrt((B[0]-A[0])*(B[0]-A[0]) + (B[1]-A[1])*(B[1]-A[1]))
	ang =  degrees(acos((b*b + c*c - a*a)/(2*b*c)))
	return ang

files ={'A.jpg'}
ab=getfiles(files)
cd=io.imread('A.jpg')
#print ab['A0.jpg']
#print cd
im = cv2.imread('A.jpg',0)
im1 = cv2.imread('A.jpg')
kernel = np.ones((10,10),np.uint8)
#erosion = cv2.erode(thresh,kernel,iterations =2)
blur = cv2.blur(im,(20,20))
ret,thresh= cv2.threshold(blur,142,255,0)
ret,thresh2= cv2.threshold(blur,142,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(im1,contours,-1,(0,255,0),10)
#print len(contours)
ci=0
max_area=0
for i in range(len(contours)):
	#image=im
	cnt1=contours[i]
	#cv2.drawContours(image,[cnt],0,(0,255,0),1)
	#plt.subplot(2,2,i+1),plt.imshow(im,cmap = 'gray')
	#plt.title(area), plt.xticks([]), plt.yticks([])
	#max_area=0
	area = cv2.contourArea(cnt1)
	#cv2.drawContours(image,[cnt],0,(0,255,0),1)
	#plt.subplot(2,2,i+1),plt.imshow(im,cmap = 'gray')
	#plt.title(area), plt.xticks([]), plt.yticks([])
	#print area,i
	if( area > max_area ):
		max_area=area
		ci=i
		#print ci

#print max_area,ci
#print ci
cnt=contours[ci]
x,y,w,h = cv2.boundingRect(cnt)
t_w= int(0.1*w)
t_h=int(0.1*h)
x=x-t_w
y=y-t_h
w=w+2*t_w
h=h+2*t_h
print w,h
x2=x+w
y2=y+h
#print x+w,y+h
w1,h1=im1.shape[:2]
"""
if x2 >= w1 :
	x2=w1
if y2 >= h1 :
	y2=h1
"""
if x<= 0:
	x=0
if y<=0 :
	y=0
	
#print x,x2,y,y2
#print w1,h1
cv2.line(im1,tuple([x,y]),tuple([x2,y]),[0 ,0,255],10)
cv2.line(im1,tuple([x2,y]),tuple([x2,y2]),[0 ,0,255],10)
cv2.line(im1,tuple([x2,y2]),tuple([x,y2]),[0 ,0,255],10)
cv2.line(im1,tuple([x,y2]),tuple([x,y]),[0 ,0,255],10)
lbr=2*(w+h)
#cv2.drawContours(im1,[cnt],0,(0,0,255),10 )
#print 1
crp=crop(cd,x,x2,y,y2)
#crp=cd[y:y+h,x:x+w]
#print crp
xyz=convertToGrayToHOG(crp)
#pythonprint xyz
with open('ml3.pkl', 'rb') as fid:
	classifier=pickle.load(fid)    
y=classifier.predict(xyz)
print y
cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()