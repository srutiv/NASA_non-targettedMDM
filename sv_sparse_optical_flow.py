# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:37:34 2019

@author: svutukur
"""

import numpy as np
import cv2

# Create list of names here from A_Run24_Seq4_00000.tif up to A_Run24_Seq4_00009.tif
list_names = ['./tbw1_data/A_Run143_Seq' + str(i) + '_00001.tif' for i in range(6,9)]
print(list_names)

index = 0

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7,
                       gradientSize = 7)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it

old_frame = cv2.imread(list_names[index])
#old_gray = old_frame.copy()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while index < (len(list_names)-1):
    frame = cv2.imread(list_names[index+1])
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    img = cv2.resize(img, (960, 540))   
    cv2.imshow('frame',img)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    index = index + 1
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()