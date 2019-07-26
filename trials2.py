# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:14:32 2019

@author: svutukur
"""
import numpy as np
import cv2
import argparse
import os
from time import time
           
           
#high_res = cv2.imread('C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_0011.tiff',0)
#color = cv2.imread('C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_0011.tiff')

#feature_params = dict(maxCorners = 100,
#                           qualityLevel = 0.3,
#                           minDistance = 7,
#                           blockSize = 7,
#                           gradientSize = 7,
#                           mask = None)  
#
#p0 = cv2.goodFeaturesToTrack(high_res, **feature_params)
#
#img = high_res
#
#for i in range(0,len(p0)):
#    img = cv2.circle(color,(p0[i][0][0],p0[i][0][1]), radius = 10, color = (0,0,255), thickness=-1)
#
#cv2.imwrite('sift_keypoints.jpg',img)
#
#    #maskimage = cv2.imread(list_names,0) #1-channel image
#    
#    roi = old_gray[500:4000, 500:4000]
#    #roi = None


# import the necessary packages

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

#if __name__ == "__main__":    

def big(im_name):
    
    refPt = [(0, 0)]
    cropping = False
    num_roi = 0
    
        
    def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            #global refPt, cropping, num_roi
            
            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being performed
            if event == cv2.EVENT_LBUTTONDOWN:
                refPt.append((x, y))
                cropping = True
                
            # check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))
                cropping = False
                
                # draw a rectangle around the region of interest
                #cv2.rectangle(image, p, q, (0, 255, 0), 2)
                cv2.rectangle(image, refPt[num_roi*2+1], refPt[num_roi*2+2], (0, 255, 0), 2)
                #each rectangle to be plotted follows the index pattern num_roi*2 + 1
                cv2.imshow("image", image)
            
    color = cv2.imread(im_name)
    size = np.shape(color)
    # load the image, clone it, and setup the mouse callback function
    #r = 1000.0 / size[1]
    #dim = (100, int(size[0] * r))
     
    # perform the actual resizing of the image and show it
    image = color; #image = cv2.resize(image, (960, 540)) #take color image and resize
    #image = color; image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #take color image and resize
    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    
    
    
    while num_roi < 2:
    
        cv2.setMouseCallback("image", click_and_crop)
        
        #keep pressing 'c' until the 4 rectangles are drawn
        while True:
            cv2.imshow("image", image)
        
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()
                # if the 'space' key is pressed, a rectangle was captured and you break from the loop
            elif key == ord("c"):
                break
        
        num_roi  = num_roi + 1
    
    cv2.destroyAllWindows()
    
    
    #create mask
    #replace "image" with "color" --> OG
    ROI = np.zeros(image.shape[:2], np.uint8)
    #x1 = refPt[1][0]; x2 = refPt[2][0]; y1 = refPt[1][1]; y2 = refPt[2][1] 
    
    refPt = refPt[1:] #ignore the zero initializing row
    
    masked_grays = [0] * num_roi
    boundboxes = [0]* num_roi
    
    for r in range(0,num_roi):
        
        #are these offset somehow???
        x1 = np.minimum(refPt[r][0],refPt[r+1][0]); x2 = np.maximum(refPt[r][0],refPt[r+1][0])
        y1 = np.minimum(refPt[r][1],refPt[r+1][1]); y2 = np.maximum(refPt[r][1],refPt[r+1][1])
        
        
        ROI[y1:y2,x1:x2] = 255 #the order of these are important; (0,0) at top left
        boundboxes[r] = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
        masked_img = cv2.bitwise_and(image,image,mask = ROI)
        masked_grays[r] = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('mask', masked_img)
        #key = cv2.waitKey(2000)
        #cv2.destroyAllWindows()

    return [refPt, num_roi, masked_grays,boundboxes]

#[refPt, masked_img] = big()

## if there are two reference points, then crop the region of interest
## from teh image and display it
#if len(refPt) == 2:
#    x1 = refPt[0][0]; x2 = refPt[1][0]; y1 = refPt[0][1]; y2 = refPt[1][1]
#    boundbox = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
#    im_wROIs = cv2.add(image,boundbox) #plot ROIs and OG image
#    cv2.imshow("User-cdefined ROIs",im_wROIs)
#    
#    key = cv2.waitKey(5000)#pauses for 3 seconds before fetching next image
#    cv2.destroyAllWindows()
