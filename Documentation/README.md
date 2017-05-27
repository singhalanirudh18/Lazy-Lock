# BCBzzz



# Lazy Lock :Life Simplefied

## Abstract

Here we present you a lock opening system designed such that a door locked from inside can be opened by a particular hand gesture or a particular knock pattern performed on the other side of the door. Potential of this system ranges from locking personal belongings in vault or locking your room or even house door.This system can be very useful for providing security. The system can then be further implemented to use face recognisition to provide better security. We first present the issues surrounding its conceptual design and then describe in detail the component level implementation of the system, which we have designed and built. 

Our problem statement which precisely describes the problem which we are aiming to tackle goes as follows: </br>
**To design a system which opens a lock on performing a specific gesture or knock pattern with maximum precision**.

## 1.Introduction
</br>

### 1.1 Motivation
Locking and unlocking of a door is a crutial aspect as far as security is concerned. The application of this system helps in fulfilling the given objective. Many a times it happens that the key of a lock goes missing, or a person leaves and lockes a room and another person want to enter the room, or a person is sleeping and other person needs to enter the room. at such places it becomes important to eleminate the orthodox way of using a key to open the lock or the need of other people to open the lock. hence to fulfill this objective we designed the above system such that a limited no of people have the access to open a lock by knowing the correct gesture or knock pattern. 

### 1.2 Image processing
#### 1.2.1 Idea
Using Image Processing through OpenCV what we hope to accomplish is that to recognise various gestures and then send a command to the rpi depending upon the gesture to whether to open the door or not. For learning basics of OpenCV I reffered [openCV documentation](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) and to see basics of gesture recognition I referred [this blog](http://creat-tabu.blogspot.in/2013/08/opencv-python-hand-gesture-recognition.html).
#### 1.2.2 Components
#### 1.2.3 Working
The basic working of our algorithm is as follows. :-
1. First of all we need to binarize our image using appropiate thresholding techniques.
2. After getting binarised image we find all the contours in the image.
3. As there may be some unwanted contours in the images we select them by selecting contour of the greatest area.
4. After getting the appropiate contour we draw the convex hull of the contour which is basically the smallest polygon enclosing the contour.
5. After getting the convex hull we find the defects of contours which are basically the points on contours farthest from the sides of polygon.
6. Then we filter out defects by applying appropiate filters such as angle between defect and the points of the line should be less than 80 degrees.

Our basic IP algoritm is neatly represented by this flowchart :
![alt text](https://github.com/singhalanirudh18/itsp/blob/master/images/gesture_flowchart.png)

#### 1.2.4 Problems
The major problems that we faced with respect to IP were :
1. **Thresholding problems** 
2. **Inability to detect complex gestures**
3. **Conversion From OpenCV2 to OpenCV3**

#### 1.2.5 Solutions to problems faced
This were the solutions we were able to come with :-
1.  For thresholding pains we used a red glove which proved out to be very good solution. For thresholding of red colour I would recommend using HSV colour scheme instead of RGB. Here is the code for thresholding using HSV colour scheme:-

```python
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
```



### 1.3 Knocking Recognition
#### 1.3.1 Idea
#### 1.3.2 Components
#### 1.3.3 Working
#### 1.3.4 Problems


### 1.4 Bluetooth
#### 1.4.1 Idea
#### 1.4.2 Components
#### 1.4.3 Working
#### 1.4.4 Problems

## 2. Integration with Raspberry Pi


