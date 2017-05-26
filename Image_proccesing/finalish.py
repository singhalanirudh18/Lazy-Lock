import cv2
import time
import picamera
import numpy as np
from math import acos,degrees
import math
camera = picamera.PiCamera()
#print('yes')
camera.resolution = (160, 128)
camera.framerate = 10
#rawCapture = PiRGBArray(camera, size=(160, 128))
 
# allow the camera to warmup
time.sleep(0.001)

def angle(A,B,C) :
	a = math.sqrt((B[0]-C[0])*(B[0]-C[0]) + (B[1]-C[1])*(B[1]-C[1]))
	b = math.sqrt((A[0]-C[0])*(A[0]-C[0]) + (A[1]-C[1])*(A[1]-C[1]))
	c = math.sqrt((B[0]-A[0])*(B[0]-A[0]) + (B[1]-A[1])*(B[1]-A[1]))
	ang =  degrees(acos((b*b + c*c - a*a)/(2*b*c)))
	return ang
def segment_colour(frame):    #returns only the red colors in the frame
    hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv_roi, np.array([160, 160,10]), np.array([190,255,255]))
    ycr_roi=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    mask_2=cv2.inRange(ycr_roi, np.array((0.,165.,0.)), np.array((255.,255.,255.)))

    mask = mask_1 | mask_2
    kern_dilate = np.ones((8,8),np.uint8)
    kern_erode  = np.ones((3,3),np.uint8)
    mask= cv2.erode(mask,kern_erode)      #Eroding
    mask=cv2.dilate(mask,kern_dilate)     #Dilating
    #cv2.imshow('mask',mask)
    return mask

j=0

def num_fingers( image ) :
    #grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text
    im = cv2.imread(image)
    im1 = cv2.imread(image)
    #frame=cv2.flip(frame,1)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower_blue = np.array([100,50,50])
    #upper_blue = np.array([140,255,255])
    thresh = segment_colour(im)
    thresh2 = segment_colour(im)
    #cv2.imshow('frame',img)
    #cv2.imshow('mask',mask)
    kernel = np.ones((10,10),np.uint8)
    #erosion = cv2.erode(thresh,kernel,iterations =2)

    im2,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print (len(contours))
    if (len(contours) == 0) :
        return 0
    #x=int((len(contours)/2))
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
                    #print (ci)
    cnt=contours[ci]
    cv2.drawContours(im,[cnt],0,(0,0,255),5 )
    x,y,w,h = cv2.boundingRect(cnt)
    #print w,h
    lbr=2*(w+h)
    #cv2.drawContours(im1,[cnt1],0,(0,0,255),100 )
    hull = cv2.convexHull(cnt)
    cv2.drawContours(im1,[hull],0,(0,255,0),10)
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

    for i in range(1,defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        #cv2.line(im1,start,end,[0 ,0,255],100)
        #cv2.circle(im1,far,100,[0,0,255],-1)
        if((d >= 0.4*lbr) and (angle(far,start,end)<90)) :
            #print(1)
            cv2.line(im1,start,end,[0 ,0,255],10)
            cv2.circle(im1,far,5,[0,0,255],-1)
            count=count+1
            """
           # cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
            #cv2.imshow('thresh',im1)
            #cv2.waitKey(2000)
           # cv2.destroyAllWindows()

            """
            
            
            
            #print i,d

    numberoffinger = count +1
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('thresh',im1)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return numberoffinger
for image in camera.capture_continuous('rawCapture.jpg'):#, format="bgr", use_video_port=True):
    #grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text
    x = num_fingers( image )
    print (x)
    j+=1
    if(j == 10) :
        break 
"""
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('thresh',mask)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    j+=1
    
    print (1)
    if (j==10) :
        break
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('thresh',im)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    j+=1
    if (j==10) :
        break
"""
