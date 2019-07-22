#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
import fnmatch
import os
import math

def load_images():
    #returns an array of images of interest
    images = ['C:/Users/svutukur/Desktop/cv_mdm2019D/splatter/A_Run24_Seq4_0000' + str(i) + '.tif' for i in range(2,5)]
    num_images = len(images)
    return [images, num_images]

def find_matches(pic1, pic2):
    MIN_MATCH_COUNT = 10
    
    img1 = cv2.imread(pic1,0) 
    img2 = cv2.imread(pic2,0)
    
    n_kp = 100 #to reduce the number of matches; play around with this variable. yeet
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(n_kp)
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    #create FLANN Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    #Match descriptors
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #matched pts in image 1
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #matched pts in image 2
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
    
    cv2.imwrite('tracksnlimit' + '.jpg', img3)
    #plt.imshow(img3, 'gray'),plt.show()


def find_sequential_matches():
    first = 0
    while first < num_images-1:
        [good, src_pts, displacements] = find_matches(images[first], images[first+1])
        print(images[first], images[first+1], len(good))
        first = first + 1;


[images, num_images] = load_images()
find_matches(images[1], images[2])
#find_sequential_matches()

