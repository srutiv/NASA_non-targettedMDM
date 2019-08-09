#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:22:05 2019

@author: Sruti
"""

import numpy as np
import cv2
import math
from numpy.linalg import inv
from sv_checker_calib import checker_calib


def find_matches(kp1,des1,img2):

    kp2, des2 = sift.detectAndCompute(img2,None)
    
    #create FLANN Matcher object
    #matches the targets between two images
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    #Match descriptors
    matches = flann.knnMatch(des1,des2,k=2)
    
    return [kp2, des2, matches, flann]



####################################################### MAIN ###########################################
#if __name__ == "__main__":

checkers = ['C:/Users/svutukur/Documents/our_checker_calib/A_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]
[newcameramtx, total_error] = checker_calib(checkers) #extract the instrinsic camera parameter matrix if using 2D openCV camera calibration script

#sequence of images from 1 camera
list_names = ['C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]

print('number of MDM images to track: ' + str(len(list_names)))

disp3 = []
first = 0

#initialize first image features and use them to track in all images

MIN_MATCH_COUNT = 10
    
img1 = cv2.imread(list_names[0],0)

n_kp = 100; #increase for fewer matches


# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(n_kp)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

ground_match_num = 0

while first < len(list_names)-1:
    
    #initialize FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    if first == 0: #initializing ground_match_num
        img2 = cv2.imread(list_names[first+1],0) 
        [kp2, des2, matches, flann] = find_matches(kp1,des1,img2)
        print('OG matches ' + str(len(matches)))
       
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            #if m.distance < 0.7*n.distance: #Uncomment if user expects drastic changes in keypoint location
            good.append(m)
        
        ground_match_num = len(good)
        print('ground_match_num = ' + str(ground_match_num))
        
    else:
        img2 = cv2.imread(list_names[first+1],0) 
        [kp2, des2, matches, flann]= find_matches(kp1,des1,img2)
        
        print(len(matches))
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            #if m.distance < 0.7*n.distance:
            good.append(m)

        print(len(good))
        
        if len(good) != ground_match_num:
            print('didnt find the same number of matches')
            
            #can't find same number of matches --> iterate over search_params first
            for k in range(5,20,5):
                index_params['trees'] = k
                print('number of trees: '+ str(k)) #may not the best parameter to iterate over
                
                low_res = cv2.pyrDown(img2) #add gaussian pyramid to detect more keypoints (downsampling)
                
                [kp2, des2, matches, flann] = find_matches(kp1,des1,low_res)
                
                print(len(matches))
                # store all the good matches as per Lowe's ratio test.
                
                good = []
                for m,n in matches:
                    #if m.distance < 0.7*n.distance:
                    good.append(m)

                print(len(good))
                
                if len(good) != ground_match_num:
                    print('still bad. continue changing checks')
                    continue
                else:
                    print('found good criteria')
                    break
        else:
            print('found the same number of matches')
    
    #previous block of code ensures that number of good matches == ground_match number
        
    if len(good)>MIN_MATCH_COUNT:
        #feature detections
        img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #matched pt locations in image 1
        img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #matched pt locations in image 2
        
        #calculate homography matrix (this is specific to the 2D displacement calculator)
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0) #H is 3x3 homography matrix
        matchesMask = mask.ravel().tolist()

        #used for plotting
        height,width = img1.shape
        pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    print('good len ' + str(len(good)))
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = False,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    img3 = cv2.resize(img3, (960, 540))  


    ###################################### 2D pixel --> physical coordinate transformation and 2D displacement calculator #####################
    
    #reshaping pixel coord matrices img1_pts, img2_pts to allow for matrix multiplication
    img1_pts = img1_pts[:,0,:]
    img2_pts = img2_pts[:,0,:]
    
    img1_pts = np.append(img1_pts, np.ones((len(img1_pts),1)), 1)
    img2_pts = np.append(img2_pts, np.ones((len(img2_pts),1)), 1)
    
    #get physical coord matrices w1, w2 from homography transformation and camera distortion matrix
    w1 = []
    w2 = []
    s = 1 #physical scaling factor
    
    for i in range(len(img1_pts)):
        a = np.matmul(inv(H), img1_pts[i])*s
        w1.append(np.matmul(inv(newcameramtx), a))
        b = np.matmul(inv(H), img2_pts[i])*s
        w2.append(np.matmul(inv(newcameramtx), b))
    
    #get physical displacements from w1, w2
    disp2 = []

    
    for j in range(len(img1_pts)):
        disp1 = math.sqrt((w1[j][0]-w2[j][0])**2 + (w1[j][1]-w2[j][1])**2 + (w1[j][2]-w2[j][2])**2) #distance formula
        disp2.append(disp1)
    
    disp3 = np.hstack((disp3,disp2))
    
    cv2.imwrite('matches.jpg', img3)
    print('completed first' + str(first+1))
    
    first = first + 1;
 

disp3 = np.vstack((disp3,np.ones((1,len(disp3))))) #replace 2nd input displacements from other sets of images
# the dimension of the matrices must be the same (same number of targets), otherwise it overwrites

tot_disp = np.sum(disp3, axis = 0) #total displacement across all sets of images
