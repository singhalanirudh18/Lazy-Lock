pyimport numpy as np
import cv2
from matplotlib import pyplot as plt
from math import acos,degrees
import math
def angle(A,B,C) :
	a = math.sqrt((B[0]-C[0])*(B[0]-C[0]) + (B[1]-C[1])*(B[1]-C[1]))
	b = math.sqrt((A[0]-C[0])*(A[0]-C[0]) + (A[1]-C[1])*(A[1]-C[1]))
	c = math.sqrt((B[0]-A[0])*(B[0]-A[0]) + (B[1]-A[1])*(B[1]-A[1]))
	ang =  degrees(acos((b*b + c*c - a*a)/(2*b*c)))
	return ang


#im = cv2.imread('test.jpg')
im = cv2.imread('4.jpg')
count = 0
#kernel = np.ones((2,2),np.uint8)
#dilation = cv2.dilate(im,kernel,iterations = 1)
#im = dilation
#blur = cv2.blur(im,(10,10))
#im = cv2.imread('2.jpg')
#kernel = np.ones((2,2),np.uint8)
#dilation = cv2.dilate(im,kernel,iterations = 1)
#im = dilation
#blur = cv2.blur(im,(10,10))
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 240, 255, 0)
kernel = np.ones((1.5,1.5),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 1)
blur = cv2.blur(erosion,(2,2))
ret,thresh2= cv2.threshold(blur,240,255,0)
row, col = thresh2.shape[:2]
#print row,col
cv2.line(thresh2,(0,0),(col,0),(255,255,255),5)
cv2.line(thresh2,(0,0),(0,row),(255,255,255),5)
cv2.line(thresh2,(0,row),(col,row),(255,255,255),5)
cv2.line(thresh2,(col,row),(col,0),(255,255,255),5)

#cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
#cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#image=im
#print len(contours)
#cnt1= contours[1]
#area = cv2.contourArea(cnt1)
#p#rint area
#cv2.drawContours(im,[cnt1],0,(0,0,255),1)

for i in range(len(contours)):
	#image=im
	cnt=contours[i]
	#cv2.drawContours(image,[cnt],0,(0,255,0),1)
	#plt.subplot(2,2,i+1),plt.imshow(im,cmap = 'gray')
	#plt.title(area), plt.xticks([]), plt.yticks([])
	max_area=0
	area = cv2.contourArea(cnt)
	#cv2.drawContours(image,[cnt],0,(0,255,0),1)
	#plt.subplot(2,2,i+1),plt.imshow(im,cmap = 'gray')
	#plt.title(area), plt.xticks([]), plt.yticks([])
	if(area>max_area):
		max_area=area
        ci=i
#plt.show()
cnt=contours[ci]
x,y,w,h = cv2.boundingRect(cnt)
#print w,h
lbr=2*(w+h)
#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

hull = cv2.convexHull(cnt)
cv2.drawContours(im,[hull],0,(0,0,255),1)
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
drawing = np.zeros(im.shape,np.uint8)
#cv2.drawContours(im,[cnt],0,(0,255,0),1)
#cv2.drawContours(im,[hull],0,(0,0,255),1)
#defects = cv2.convexityDefects(cnt,hull)
#print defects
mind=0
maxd=0
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    if((d >= 0.4*lbr) and (angle(far,start,end)<80)) :
    	cv2.line(im,start,end,[0,255,0],1)
    	cv2.circle(im,far,2,[0,0,255],-1)
    	count=count+1
    	#print i,d

numberoffinger = count +1
print numberoffinger
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cnt = contours[0]
#hull = cv2.convexHull(cnt)
#print hull
#cv2.line(im,(54,87),(1,87),(255,0,0),2)
#cv2.line(im,(1,87),(1,1),(255,0,0),2)
#cv2.line(im,(1,1),(54,1),(255,0,0),2)
#cv2.line(im,(0,0),(54,87),(255,0,0),1)
#cv2.line(im,(0,0),(1,87),(255,0,0),1)
#cv2.line(im,(0,0),(1,1),(255,0,0),1)
#cv2.line(im,(0,0),(54,1),(255,0,0),1)
#cv2.line(img,(0,0),(511,511),(255,0,0),5)
#cv2.drawContours(im, contours, -1, (0,255,0), 1)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()  