import numpy as np
import cv2
#im = cv2.imread('test.jpg')
im = cv2.imread('1.jpg')
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
kernel = np.ones((2,2),np.uint8)
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

for i in range(len(contours)):
	cnt=contours[i]
	max_area=0
	area = cv2.contourArea(cnt)
	if(area>max_area):
		max_area=area
        ci=i
cnt=contours[ci]
hull = cv2.convexHull(cnt)
drawing = np.zeros(im.shape,np.uint8)
cv2.drawContours(drawing,[cnt],0,(0,255,0),1)
cv2.drawContours(im,[hull],0,(0,0,255),1)


#cnt = contours[0]
hull = cv2.convexHull(cnt)
print hull
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