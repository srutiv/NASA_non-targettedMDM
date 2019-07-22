# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:01:38 2019

@author: svutukur
"""
import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
import inspect
#import sv_blob_finder

# Create list of names here from A_Run24_Seq4_00000.tif up to A_Run24_Seq4_00009.tif
#list_names = ['./tbw1_data/A_Run143_Seq' + str(i) + '_00001.tif' for i in range(6,9)]
#list_names = ['./multi_track/cam1_' + '0000' + str(i) + '.tif' for i in range(1,8)]
list_names = ['./fancy_wand/cam1_' + '0000' + str(i) + '.tif' for i in range(1,6)]
print('number of MDM images to track: ' + str(len(list_names)))

#[pix_disps1, tot_disps1, diag_disps1] = corner_detector(list_names)
#[pix_disps2] = blobs_detector(list_names)
    
#def corner_detector(list_names):
#params for ShiTomasi corner detection
#not needed for blob detector

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7,
                       gradientSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 # Generate some random colors later used to display movement paths
#color = np.random.randint(0,255,(100,3))
color = np.ones((100,3))*100
print(np.shape(color))

index = 0

# Take first frame
old_frame = cv2.imread(list_names[index],0)

#goodFeaturesToTrack determines strong corners on an image
#can be used to initialize a point-based tracker (like the calcOpticalFlowPyrLK)
p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
pix_disps1 = []
prev_coord = p0

print('number of corners found: '+ str(len(p0)))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame) 

while index < (len(list_names)-1):
    print(index)
    frame = cv2.imread(list_names[index+1],0)
    
    # calculate optical flow
    # for a sparse feature set using the iterative LK method with pyramids
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
   
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    
    
    pix_disps1.append(abs(p1 - prev_coord)) ###problem here
    prev_coord = p1
    
    #plot1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RBG)
    #plot2 = cv2.cvtColor(frame, cv2.COLOR_GRAY2RBG)
    
    img = cv2.add(frame,mask)
    img = cv2.resize(img, (960, 540))  
    cv2.imshow('frame',img)
    k = cv2.waitKey(2000) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_frame = frame.copy()
    index = index + 1
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()

#for j in range(np.shape(pix_disps1)[1]):
#    sum = pix_disps1[0][j][0] + pix_disps1[1][j][0] + pix_disps1[2][j][0] #instead of 0,1,2 make it dependent on the length of MDM images to track -1
#    tot_disps1.append(sum) #all elements should be comparable to eachother because its rigid body motion?
#    #some of these are other objects in the frame moving

tot_disps1 = np.sum(pix_disps1, axis = 0)

diag_disps1 = []
for k in range(np.shape(tot_disps1)[0]):
    diag_disps1.append(np.sqrt(tot_disps1[k][0][0]**2 + tot_disps1[k][0][1]**2))
    
#return [pix_disps1, tot_disps1, diag_disps1]

#def blobs_detector(list_names):
#    # Parameters for lucas kanade optical flow
#    
#    lk_params = dict( winSize  = (15,15),
#                      maxLevel = 10,
#                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     # Generate some random colors later used to display movement paths
#    color = np.random.randint(0,255,(100,3))
#    
#    index = 0
#    
#    # Take first frame
#    old_frame = cv2.imread(list_names[index],0)
#    
#    #use outputs from blob_finder instead of goodFeaturesToTrack
#    [x,y] = sv_blob_finder.main(old_frame)
#    
#    #make p0 digestible for calcOpticalFlowPyrLK
#    x = np.asfarray(x, dtype='float32'); y = np.asfarray(y, dtype='float32')#calcOpticalFlowPyrLK only takes float32.sadness
#    p0 = np.stack((x,y)); p0 = np.transpose(p0)
#    
#    pix_disps2 = []
#    prev_coord = p0
#    
#    print('number of blobs found: '+ str(len(p0)))
#    
#    # Create a mask image for drawing purposes
#    mask = np.zeros_like(old_frame)
#    
#    while index < (len(list_names)-1):
#        frame = cv2.imread(list_names[index+1],0)
#        
#        # calculate optical flow
#        # for a sparse feature set using the iterative LK method with pyramids
#        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
#        print(np.shape(p1))
#        ##### WHY IS ST == 0 for blob-detector input?? #######
#       
#        # Select good points
#        good_new = p1
#        good_old = p0
#        
#        # draw the tracks
#        for i,(new,old) in enumerate(zip(good_new,good_old)):
#            a,b = new.ravel()
#            c,d = old.ravel()
#            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#        
#        pix_disps2.append(abs(p1 - prev_coord))    
#        print(np.shape(abs(p1 - prev_coord)))
#        
#        prev_coord = p1
#        
#        print('passed 2')
#        
#        img = cv2.add(frame,mask)
#        img = cv2.resize(img, (960, 540))  
#        cv2.imshow('frame',img)
#        k = cv2.waitKey(2000) & 0xff
#        if k == 27:
#            break
#        # Now update the previous frame and previous points
#        old_frame = frame.copy()
#        index = index + 1
#        p0 = good_new.reshape(-1,1,2)
#        
#        print('passed 3')
#        
#    cv2.destroyAllWindows()
#    
#    tot_disps2 = []
#    
##    for j in range(np.shape(pix_disps2)[1]):
##        sum = pix_disps2[0][j][0] + pix_disps2[1][j][0] + pix_disps2[2][j][0] #instead of 0,1,2 make it dependent on the length of MDM images to track -1
##        tot_disps2.append(sum) #all elements should be comparable to eachother because its rigid body motion?
##        #some of these are other objects in the frame moving
##    
##    diag_disps2 = []
##    for k in range(len(tot_disps2)):
##       diag_disps2.append(np.sqrt(tot_disps2[k][0]**2+tot_disps2[k][1]**2))
##    
#    return [pix_disps2]


######################################################## MAIN ###########################################
#if __name__ == "__main__":
#    
#    # Create list of names here from A_Run24_Seq4_00000.tif up to A_Run24_Seq4_00009.tif
#   # list_names = ['./tbw1_data/A_Run143_Seq' + str(i) + '_00001.tif' for i in range(6,9)]
#    #list_names = ['./multi_track/cam1_' + '0000' + str(i) + '.tif' for i in range(1,4)]
#    list_names = ['./fancy_wand/cam1_' + '0000' + str(i) + '.tif' for i in range(1,4)]
#    print('number of MDM images to track: ' + str(len(list_names)))
#    
#    [pix_disps1, tot_disps1, diag_disps1] = corner_detector(list_names)
#    [pix_disps2] = blobs_detector(list_names)