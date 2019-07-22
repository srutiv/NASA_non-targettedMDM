# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:39:00 2019

@author: svutukur
"""

# Standard imports
import cv2
import numpy as np;
import os
import inspect
import sys
import os

#def create_params():
     
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

#Change the colors
params.filterByColor = True;
params.blobColor = 0;
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

print('successfully created params for blob_finder')
#return params

#def main(old_gray):

#params = create_params()
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create()

# Read image
im = cv2.imread('A_Run143_Seq6_00001.tif', cv2.IMREAD_GRAYSCALE)
#im = old_gray
    
# Detect blobs.
keypoints = detector.detect(im)
print('number of keypoints found: ' + str(len(keypoints)))

x = []
y = []
s = []

for keyPoint in keypoints:
    x.append(keyPoint.pt[0])
    y.append(keyPoint.pt[1])
    s.append(keyPoint.size)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints

img = cv2.resize(im_with_keypoints, (960, 540))
cv2.imshow("Keypoints",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    #cv2.imwrite("Keypoints.jpg", im_with_keypoints)
    
#    return [x,y]


######################################################## MAIN ###########################################
#if __name__ == "__main__":
#    [x,y] = main(sys.argv,old_gray)