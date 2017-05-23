import cv2
import os
from os import listdir
from os.path import isfile, join
import sklearn
import os.path
from skimage import data, io
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import hog 
from skimage.transform import resize
import pickle
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC

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

def getXandY(directory,testcases) :

	#directory = '/home/anirudh18/Documents/itsp/ml/bw_dataset'
	#im_1 = cv2.imread('/home/anirudh18/Documents/itsp/ml/bw_dataset/A/B.jpg')
	#cd=io.imread('/home/anirudh18/Documents/itsp/ml/bw_dataset/A/B.jpg')
	#print cd
	#dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
	#print dirs
	#testcases= ['A']
	X=[]
	Y=[]
	for subdir in testcases :
		mypath = directory + '/' + subdir
		#print mypath
		onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
		#print onlyfiles
		for file in onlyfiles:
			file_path = mypath +'/' + file
			#print file_path
			
			im1 = cv2.imread(file_path)
			im = cv2.imread(file_path,0)
			cd=io.imread(file_path)
			kernel = np.ones((10,10),np.uint8)
			#erosion = cv2.erode(thresh,kernel,iterations =2)
			
			blur = cv2.blur(im,(20,20))
			ret,thresh= cv2.threshold(blur,112,255,0)
			ret,thresh2= cv2.threshold(blur,112,255,0)
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
			#print w,h
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
			"""
			cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
			cv2.imshow('thresh',im1)
			cv2.waitKey(500)
			cv2.destroyAllWindows()
			"""
			#crp=crop(cd,x,x+w,y,y+h)
			#crp=cd[y:y+h,x:x+w]
			#print crp
			xyz=convertToGrayToHOG(crp)
			#pythonprint xyz
			X.append(xyz)
			Y.append(subdir[0] + subdir[1])
	return X,Y




#print Y
train_directory= '/home/anirudh18/Documents/itsp/ml/bw_dataset'
testcases= ['AA','BB','yo']
X,Y=getXandY(train_directory,testcases)
print Y
predictcases=['AApr','BBpr','yopr']
X1,Y1=getXandY(train_directory,predictcases)

svcmodel = SVC(kernel='linear', C=0.9, probability=True)
svcmodel.fit(X, Y)
y2= svcmodel.predict(X1)
print y2
print Y1
with open('bw01.pkl', 'wb') as fid:
	pickle.dump(svcmodel,fid)

"""
cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh',im_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""