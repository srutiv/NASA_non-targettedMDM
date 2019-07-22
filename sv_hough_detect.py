#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:11:15 2019

@author: cngo
"""
import cv2
import numpy as np
import csv

wandPoints = []; right_points = []
    #    '''uncomment below to see see different stages of image preparation'''
    #    print_img = np.hstack([cv2.resize(blur,(width/3,height/3)), \
    #                          cv2.resize(thresh_init,(width/3,height/3)), \
    #                          cv2.resize(thresh,(width/3,height/3))])
    #    winname = "image prep"
    #    cv2.namedWindow(winname)        # Create a named window
    #    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    #    cv2.imshow(winname, print_img)  
    
def track_circles(img):
    '''applying hough circle search
    dp = 2 #inverse ration of the accumulator resolution 
    param1 = 110 #gradient value used to handle edge detection in Yuen et al
    param2 = 100# accumulator threshhold valuefor the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
    minDist=100 #min distance between centers
    minRadius = 50
    maxRadius = 90'''

    blur = cv2.medianBlur(img,5) #first, blur images
    #cv2.imwrite('blur', blur)
    
    ret,thresh_init = cv2.threshold(blur,15,255 ,cv2.THRESH_BINARY) # apply thresh holding
    #print(ret, thresh_init)
    
    thresh_img=cv2.adaptiveThreshold(thresh_init,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2) #apply adaptive and binary thresh holding again
    #cv2.imwrite('thresh_img', thresh_img)
    
    circles = cv2.HoughCircles(thresh_img, cv2.HOUGH_GRADIENT,\
                               dp=2,param1=110,param2=110,minDist=110, minRadius=10,maxRadius=100)

    return [circles, thresh_img, blur, ret, thresh_init]

def draw_circles(img_circle,cam_circles):
    output = cv2.cvtColor(img_circle,cv2.COLOR_GRAY2RGB)
    if cam_circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        cam_circles = np.round(cam_circles[0, :]).astype("int")
            	# loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in cam_circles:
            	# draw the circle in the output image, then draw a rectangle
             # corresponding to the center of the circle
             cv2.circle(output, (x, y), r, (0, 255, 0),4)
             cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return output
         
    
for i in range(0,1):#range(0,9):
    
    #read images
    left_path = 'cam1_0000%d.tif' % (i)
    #right_path = 'cam2_0000%d.tif' % (i)
    img1 = cv2.imread(left_path, 0) 
    #img2 = cv2.imread(right_path, 0) 
    height, width = img1.shape #get size of image
    
    '''this is the part where each cirle is found'''

    [cam1_circles, thresh_img, blur1, ret1, thresh_init1] = track_circles(img1)  
    #[cam2_circles, thresh_img, blur2, ret2, thresh_init2] = track_circles(img2)  

    output1 = draw_circles(img1,cam1_circles)
    #output2 = draw_circles(img2,cam2_circles)
    
    imS1=cv2.resize(output1,(int(width/4),int(height/4)))
    #imS2=cv2.resize(output2,(int(width/4),int(height/4)))
    if cam1_circles is not None: #if cam1_circles are found
        #if cam2_circles is not None: #if cam2_circles are found
        print('found circles')

#        if np.shape(cam1_circles)[1]==2 & np.shape(cam2_circles)[1]==2: #if two points are found
#            wandPoints.append(np.hstack([cam1_circles[0][0][:2],cam1_circles[0][1][:2]\
#                                         ,cam2_circles[0][0][:2],cam2_circles[0][1][:2]]))
#            print('save data')
        	# show the output image
        #print i
        #cv2.imwrite('detect' + str(i) + '.jpg', np.hstack([imS1,imS2]))
        cv2.imshow('detect2',imS1)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()
#        else:
#            continue
    
    else:
        print('none found')
        cam1_circles=[]
        #cam2_circles=[]
        #cv2.imshow('detect',np.hstack([imS1,imS2]))
        cv2.imshow('detect',imS1)

        cv2.waitKey(27)
        cv2.destroyAllWindows()
        continue

#np.savetxt("wandPts.csv", wandPoints, delimiter=",")