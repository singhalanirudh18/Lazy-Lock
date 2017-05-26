import numpy as np
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

im = cv2.imread('g1.jpg',0)
im1 = cv2.imread('g1.jpg')
kernel = np.ones((10,10),np.uint8)
#erosion = cv2.erode(thresh,kernel,iterations =2)
blur = cv2.blur(im,(20,20))
ret,thresh= cv2.threshold(blur,142,255,0)
ret,thresh2= cv2.threshold(blur,142,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print len(contours)
x=int((len(contours)/2))
#cnt = contours[7]
#area = cv2.contourArea(cnt)
#cv2.drawContours(im1,[cnt],0,(0,0,255),100 )
#print area
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
#print w,h
lbr=2*(w+h)
#cv2.drawContours(im1,[cnt1],0,(0,0,255),100 )
hull = cv2.convexHull(cnt)
cv2.drawContours(im1,[hull],0,(0,255,0),100)
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
drawing = np.zeros(im.shape,np.uint8)
#print defects
#cv2.drawContours(im,[cnt],0,(0,255,0),1)
#cv2.drawContours(im,[hull],0,(0,0,255),1)
#defects = cv2.convexityDefects(cnt,hull)
#print defects
mind=0
maxd=0
count=0

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    #cv2.line(im1,start,end,[0 ,0,255],100)
    #cv2.circle(im1,far,100,[0,0,255],-1)
    if((d >= 0.4*lbr) and (angle(far,start,end)<90)) :
    	#print 1
    	cv2.line(im1,start,end,[0 ,0,255],100)
    	cv2.circle(im1,far,100,[0,0,255],-1)
    	count=count+1
    	#print i,d

numberoffinger = count +1
print (numberoffinger)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

	#plt.subplot(x,x,i+1),plt.imshow(im),plt.title(area)
	#plt.xticks([]), plt.yticks([])



plt.show()
cv2.imwrite("r1.jpg",im1)
cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()
