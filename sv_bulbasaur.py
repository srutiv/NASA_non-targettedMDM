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
import sys
import subprocess
import math
from numpy.linalg import inv
import sv_checker_calib
import os


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
        img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #matched pt locations in image 1
        img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #matched pt locations in image 2
        
        #calculate homography before or after displacements?
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0) #H is 3x3 homography matrix
        matchesMask = mask.ravel().tolist()

    
        height,width = img1.shape
        pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    print(len(good))
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = False,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
    #return homography transform matrix H and image coordinate displacement vector
    return [H, img1_pts, img2_pts, img3] #change name; what *type* of displacements are these



####################################################### MAIN ###########################################
#if __name__ == "__main__":
[newcameramtx, total_error] = sv_checker_calib.main() #extract the camera distorion matrix
#tot_disp = calc_tot_disp(newcameramtx)

#def calc_tot_disp(newcameramtx):
[images, num_images] = load_images() 

disp3 = []
first = 0

while first < num_images-1:
    [H, img1_pts, img2_pts, img3] = find_matches(images[first], images[first+1])
    
    #reshaping pixel coord matrices img1_pts, img2_pts to allow for matrix multiplication
    img1_pts = img1_pts[:,0,:]
    img2_pts = img2_pts[:,0,:]
    
    img1_pts = np.append(img1_pts, np.ones((len(img1_pts),1)), 1)
    img2_pts = np.append(img2_pts, np.ones((len(img2_pts),1)), 1)
    
    #get physical coord matrices w1, w2 from homography transformation and camera distortion matrix
    w1 = []
    w2 = []
    
    for i in range(len(img1_pts)):
        #w1.append(np.matmul(inv(newcameramtx), inv(H), img1_pts[i])) #w1 physical coord
        #w2.append(np.matmul(inv(newcameramtx), inv(H), img2_pts[i])) #w2 physical coord
        a = np.matmul(inv(H), img1_pts[i])
        w1.append(np.matmul(inv(newcameramtx), a))
        b = np.matmul(inv(H), img2_pts[i])
        w2.append(np.matmul(inv(newcameramtx), b))
    
    #get physical displacements from w1, w2
    disp2 = []
    s = 1 #physical scaling factor
    
    for j in range(len(img1_pts)):
        disp1 = math.sqrt((w1[j][0]-w2[j][0])**2 + (w1[j][1]-w2[j][1])**2 + (w1[j][2]-w2[j][2])**2) #distance formula
        disp1 = disp1 * s 
        disp2.append(disp1)
    
    disp3 = np.hstack((disp3,disp2))
    
    cv2.imwrite('/Users/svutukur/Documents/GitHub/cv_mdm2019/' + 'Matches' + str(first) + '-' + str(first+1) +'.png', img3)
    print('completed' + ' ' + images[first], images[first+1])
    first = first + 1;
 

disp3 = np.vstack((disp3,np.ones((1,len(disp3))))) #replace 2nd input displacements from other sets of images
# the dimension of the matrices must be the same (same number of targets), otherwise it overwrites

tot_disp = np.sum(disp3, axis = 0) #total displacement across all sets of images
