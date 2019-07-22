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
    img2 = cv2.imread(pic2,0)
    
    #Feature detection by adaptive ROI
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    #create FLANN Matcher object
    #matches the targets between two images
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    #Match descriptors
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        #feature detections
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #matched pts in image 1
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #matched pts in image 2
        
        #feature image coord displacements
        displacements = []
        for i in range(len(src_pts)):
            disp = math.sqrt((src_pts[i][0][0]-dst_pts[i][0][0])**2 + (src_pts[i][0][1]-dst_pts[i][0][1])**2) #distance formula
            displacements.append(disp)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = False,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img3
    #plt.imsave(img3, 'gray.jpg')
    
    #return homography transform matrix M and image coordinate displacement vector
    return [M, mask, good, src_pts, displacements] #change name; what *type* of displacements are these

def get_phys_disps(M, displacements):
    #w = inv(H) * I; phys coord of feature = homography transform matrix inv * image coord of feature
    
    I = displacements #image coord displacements
    print(np.shape(I))
    Hinv = inv(M) #homography transform matrix
    print(np.shape(Hinv))
    #w = np.matmul(Hinv,I) #phys coord of feature
    return I, Hinv


def find_sequential_matches():
    first = 0
    w_list = [] #list of phys coord displacements from entire set of sequential images
    while first < num_images-1:
        [M, mask, good, src_pts, displacements, img3] = find_matches(images[first], images[first+1])
        print(images[first], images[first+1], len(good))
        w_list[first] = get_phys_disps(M, displacements) #transform image coord displ into phys coord displ
        first = first + 1;
    return w_list

    
    
#####################################      MAIN      ####################################################3
#initialization
[images, num_images] = load_images() #read first frame

sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread(images[1],0) 
kp1, des1 = sift.detectAndCompute(img1,None)
#[M, mask, good, src_pts, displacements] = find_matches(images[1], pic2) #feature detection by adaptive ROI; compute homography matrix H
#[M, mask, good, src_pts, displacements] = find_matches('A_Run24_Seq4_00001.tif', 'A_Run24_Seq4_00002.tif')
w_list = find_sequential_matches()
w = sum(w_list) #find total physical displacement across entire set of images

