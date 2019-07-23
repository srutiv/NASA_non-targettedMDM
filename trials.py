# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:14:32 2019

@author: svutukur
"""
import numpy as np
import cv2
           
           
high_res = cv2.imread('C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_0011.tiff',0)
color = cv2.imread('C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_0011.tiff')

feature_params = dict(maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7,
                           gradientSize = 7,
                           mask = None)  

p0 = cv2.goodFeaturesToTrack(high_res, **feature_params)

img = high_res

for i in range(0,len(p0)):
    img = cv2.circle(color,(p0[i][0][0],p0[i][0][1]), radius = 10, color = (0,0,255), thickness=-1)

cv2.imwrite('sift_keypoints.jpg',img)

    #maskimage = cv2.imread(list_names,0) #1-channel image
    
    roi = old_gray[500:4000, 500:4000]
    #roi = None
    