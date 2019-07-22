#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:22:05 2019

@author: Sruti
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import fnmatch
import os
import math
from numpy.linalg import inv


def load_images():
    #returns an array of images of interest
    images = [];
    files = os.listdir('.') 
    for filename in sorted(files):
      case = fnmatch.fnmatchcase(filename, 'A_Run24_Seq4_*.tif')
      if case == True:
          images.append(filename)
      else:
           continue
    num_images = len(images)
    return [images, num_images]

def find_matches(pic1, pic2):
    MIN_MATCH_COUNT = 10
    
    img1 = cv2.imread(pic1,0)
    print('img1',np.shape(img1))
    img2 = cv2.imread(pic2,0)
    print('img2',np.shape(img2))
    
    #Feature detection by adaptive ROI
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    print('sift', np.shape(sift))
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    print('kp1', np.shape(kp1))
    print('des1', np.shape(des1))
    kp2, des2 = sift.detectAndCompute(img2,None)
    print('kp2', np.shape(kp2))
    print('des2', np.shape(des2))
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    #create FLANN Matcher object
    #matches the targets between two images
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    print('flann',np.shape(flann))
    
    #Match descriptors
    matches = flann.knnMatch(des1,des2,k=2)
    print('matches',np.shape(matches))
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        #feature detections
        img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #matched pt locations in image 1
        print('img1_pts',np.shape(img1_pts))
        img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #matched pt locations in image 2
        print('img2_pts',np.shape(img2_pts))
        
        
        #calculate homography before or after displacements?
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0) #H is 3x3 homography matrix
        matchesMask = mask.ravel().tolist()
        print('H', np.shape(H))
        print('mask', np.shape(mask))
        print('matchesMask', np.shape(matchesMask))
        
        #feature image coord displacements
        displacements = []
        for i in range(len(img1_pts)):
            disp = math.sqrt((img1_pts[i][0][0]-img2_pts[i][0][0])**2 + (img1_pts[i][0][1]-img2_pts[i][0][1])**2) #distance formula
            displacements.append(disp)
        
        print('displacements',np.shape(displacements))
        
    
        height,width = img1.shape
        pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        print('pts', np.shape(pts))
        print('dst', np.shape(dst))
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = False,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**dr0000aw_params)
    
    #return homography transform matrix H and image coordinate displacement vector
    return [H, mask, good, img1_pts, displacements] #change name; what *type* of displacements are these

def get_phys_disps(H, displacements):
    I = displacements #image coord displacements
    print('I', np.shape(I))
    Hinv = inv(H) #homography transform matrix
    print('Hinv', np.shape(Hinv))
    #w = np.matmul(Hinv,I) #phys coord of feature
    return I, Hinv

    
#####################################      MAIN      ####################################################3
[images, num_images] = load_images()
print(num_images)
#[M, mask, good, img1_pts, displacements] = find_matches(images[1], pic2) #feature detection by adaptive ROI; compute homography matrix H
#[M, mask, good, img1_pts, displacements] = find_matches('A_Run24_Seq4_00001.tif', 'A_Run24_Seq4_00002.tif')
H_all = []
first = 0
w_list = [] #list of phys coord displacements from entire set of sequential images
while first < num_images-1:
    [H, mask, good, img1_pts, displacements] = find_matches(images[first], images[first+1])
    H_all.append(H)
    print(images[first], images[first+1], len(good))
    [I, Hinv] = get_phys_disps(H, displacements) #transform image coord displ into phys coord displ
    #w_list[first] = w;
    first = first + 1;

#w = sum(w_list) #find total physical displacement across entire set of images