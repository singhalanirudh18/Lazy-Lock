# import the necessary packages
import numpy as np
import argparse
import cv2
from math import acos,degrees
import math
def angle(A,B,C) :
	a = math.sqrt((B[0]-C[0])*(B[0]-C[0]) + (B[1]-C[1])*(B[1]-C[1]))
	b = math.sqrt((A[0]-C[0])*(A[0]-C[0]) + (A[1]-C[1])*(A[1]-C[1]))
	c = math.sqrt((B[0]-A[0])*(B[0]-A[0]) + (B[1]-A[1])*(B[1]-A[1]))
	ang =  degrees(acos((b*b + c*c - a*a)/(2*b*c)))
	return ang

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help = "path to the image")
#args = vars(ap.parse_args())
 
# load the image
image = cv2.imread("r2.jpg")
kernel = np.ones((10,10),np.uint8)
#erosion = cv2.erode(thresh,kernel,iterations =2)
blur = cv2.blur(image,(20,20))

# define the list of boundaries
boundaries = [
	([7, 15, 100], [80, 80, 255]),
	#([86, 31, 4], [220, 88, 50]),
	#([25, 146, 190], [62, 174, 250]),
	#([103, 86, 65], [145, 133, 128])
]
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(blur, lower, upper)
	output = cv2.bitwise_and(blur, blur, mask = mask)
 
#contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print len(contours)

imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 60, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,contours,-1,(0,255,0),10)
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
cv2.drawContours(image,[cnt],0,(0,255,0),10)
x,y,w,h = cv2.boundingRect(cnt)
#print w,h
lbr=2*(w+h)
#cv2.drawContours(im1,[cnt1],0,(0,0,255),100 )
hull = cv2.convexHull(cnt)
cv2.drawContours(image,[hull],0,(255,0,0),10)
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
drawing = np.zeros(image.shape,np.uint8)
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
    	cv2.line(image,start,end,[0 ,0,255],10)
    	cv2.circle(image,far,20,[0,0,255],-1)
    	count=count+1
    	#print i,d




	# show the images
print count+1
#cv2.imwrite("r5.jpg",output)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()  