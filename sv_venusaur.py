#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:22:05 2019

@author: Sruti
"""

import numpy as np
import cv2
import sv_checker_calib
from sv_getROIs import get_ROIs
from sv_get_disps import get_3Ddisps

def find_matches(kp1,des1,img2):

    kp2, des2 = sift.detectAndCompute(img2,None)
    
    #create FLANN Matcher object
    #matches the targets between two images
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    #Match descriptors
    matches = flann.knnMatch(des1,des2,k=2)
    
    return [kp2, des2, matches, flann]

def SIFT_detectTrack(camA, camB, tag = True):
    
    # Generate some random colors later used to display movement tracks
    color3 = np.ones((100,1))*255; color2 = np.zeros((100,1)); color1 = np.zeros((100,1))
    color = np.hstack((color1,color2,color3))
    
    index = 0
    
    #Get ROIs from ROI module
    [masked_grays1, boundboxes1] = get_ROIs(camA[0])
    [masked_grays2, boundboxes2] = get_ROIs(camB[0])
    
    # Take first frame
    old_frame = cv2.imread(camA[0])
    img1 = cv2.imread(camA[0],0)
    
    slidesA = [[None] for x in range(len(camA)-1)] #images from camA used for plotting
    slidesB = [[None] for x in range(len(camB)-1)] #images from camB used for plotting
    
    disps_roi = {'ROI # ' + str(u): 0 for u in range(num_roi)}
                 
    for q in range(0, num_roi):
        
        #used to store pixel coordinates in a specific ROI
        #for camera A/B and before and after frames, H matrices used only if user wants 2D displacement data
        imgA_before = []; imgA_after = []; H_matA = []
        imgB_before = []; imgB_after = []; H_matB = []
        
        
        # Initiate SIFT detector
        MIN_MATCH_COUNT = 10
        n_kp = 100; #increase for fewer matches
        sift = cv2.xfeatures2d.SIFT_create(n_kp)
        
        # initialize first image keypoints and descriptors and use them to track in all images       
        kp1, des1 = sift.detectAndCompute(img1,None)
        prev_coord = kp1, des1
        
        print('number of features found: ' + str(len(kp1)))
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        
        ground_match_num = 0
        
        ##################################################################  for cam A  ###########################################################        
        
        while index < len(camA)-1:
            
            # set up parameters for FLANN feature matcher
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            
            if index == 0: #initializsing ground_match_num
                img2 = cv2.imread(camA[index+1],0) 
                [kp2, des2, matches, flann] = find_matches(kp1,des1,img2)
                print('OG matches ' + str(len(matches)))
                
                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    #if m.distance < 0.7*n.distance: #Uncomment if user expects drastic changes in keypoint location
                    good.append(m)
                
                ground_match_num = len(good)
                print('ground_match_num = ' + str(ground_match_num))
                
            else: # matching continues after first and second images in set are matched
                img2 = cv2.imread(camA[index+1],0) 
                [kp2, des2, matches, flann]= find_matches(kp1,des1,img2)
                
                print(len(matches))
                
                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    #if m.distance < 0.7*n.distance: #Uncomment if user expects drastic changes in keypoint location
                    good.append(m)
                
                if len(good) != ground_match_num:
                    print('didnt find the same number of matches')
                    #can't find same number of matches --> iterate over search_params first
                    
                    if len(good) < ground_match_num: #goal is to find more keypoints for matches
                        
                        low_res = cv2.pyrDown(img2) #add gaussian pyramid (downsampling)
                        
                        [kp2, des2, matches, flann] = find_matches(kp1,des1,low_res)
                        
                        # store all the good matches as per Lowe's ratio test.
                        good = []
                        for m,n in matches:
                            #if m.distance < 0.7*n.distance:
                            good.append(m)
                               
                        if len(good) != ground_match_num:
                            print('not able to find enough matches')
                            continue
                        else:
                            print('found good criteria')
                            break
                        
                    else: #if there are too many keypoints for matches
                        for k in range(5,20,5):
                            index_params['trees'] = k
                            print('number of trees: '+ str(k)) #may not be the best parameter  in index_params to iterate over
                        
                        
                            # store all the good matches as per Lowe's ratio test.
                            good = []
                            for m,n in matches:
                                #if m.distance < 0.7*n.distance:
                                good.append(m)
                            
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
        
                # used for plotting
                height,width = img1.shape
                pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,H)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                
                #record keypoints to initialized arrays
                imgA_before.append(img1_pts)
                imgA_after.append(img2_pts)
                H_matA.append(H)
                
                #update prev_coord with most recent detected keypoints
                prev_coord = kp2, des2
                
                #old_frame = frame.copy()
                index = index + 1 #update index so you can move onto next image in camA sequence
            
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
        
        index = 0; #reset image sequence index for camB
        
        print('finished tracking images from camA')
            
##################################################################  for camB  ###########################################################
        old_frame = cv2.imread(camB[0])
        img1 = cv2.imread(camB[0],0)
        
        while index < len(camB)-1:
            
            # set up parameters for FLANN feature matcher
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            
            if index == 0: #initializsing ground_match_num
                img2 = cv2.imread(camB[first+1],0) 
                [kp2, des2, matches, flann] = find_matches(kp1,des1,img2)        
                print('OG matches ' + str(len(matches)))
                
                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    #if m.distance < 0.7*n.distance: #Uncomment if user expects drastic changes in keypoint location
                    good.append(m)
                
                ground_match_num = len(good)
                print('ground_match_num = ' + str(ground_match_num))
                
            else: # matching continues after first and second images in set are matched
                img2 = cv2.imread(camB[index+1],0) 
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
                    
                    if len(good) < ground_match_num: #goal is to find more keypoints for matches
                        
                        low_res = cv2.pyrDown(img2) #add gaussian pyramid (downsampling)
                        
                        [kp2, des2, matches, flann] = find_matches(kp1,des1,low_res)
                        
                        # store all the good matches as per Lowe's ratio test.
                        good = []
                        for m,n in matches:
                            #if m.distance < 0.7*n.distance:
                            good.append(m)
                               
                        if len(good) != ground_match_num:
                            print('not able to find enough matches')
                            continue
                        else:
                            print('found good criteria')
                            break
                        
                    else: #if there are too many keypoints for matches
                        for k in range(5,20,5):
                            index_params['trees'] = k
                            print('number of trees: '+ str(k)) #may not be the best parameter  in index_params to iterate over
                        
                        
                            # store all the good matches as per Lowe's ratio test.
                            good = []
                            for m,n in matches:
                                #if m.distance < 0.7*n.distance:
                                good.append(m)
                            
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
                
                #calculate homography before or after displacements?
                H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0) #H is 3x3 homography matrix
                matchesMask = mask.ravel().tolist()
        
                # used for plotting
                height,width = img1.shape
                pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,H)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                
                #record keypoints to intialized arrays
                imgB_before.append(img1_pts)
                imgB_after.append(img2_pts)
                H_matB.append(H)
                
                #plotting tracks on just camB image sequence for now
                for i,(new,old) in enumerate(zip(kp2, kp1)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 12)
                    frame = cv2.circle(frame,(a,b),7,color[i].tolist(),-1, lineType = 10)
                
                img = cv2.add(frame,mask)
                img = cv2.add(img,boundboxes2[q]) #plot ROIs and OG image; boundbox from trials
                
                #add images from consective images to slides_frames
                slidesB[index] = img
            
                #update prev_coord with most recent detected keypoints
                prev_coord = kp2, des2
                
                #old_frame = frame.copy()
                index = index + 1 #update index so you can move onto next image in camB sequence
            
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
        
        
        print('finished tracking images from camB') 
        index = 0 #reset index for new ROI
        
        tot_disps = get_3Ddisps(imgA_before, imgA_after, imgB_before, imgB_after) #calculate displacement for 1 ROI using camA and camB sequences
        disps_roi['ROI # ' + str(q)] = tot_disps  

    #iterate over each consecutive image set, stack ROIs for a respective set together --> display --> move to next set
    for s in range(0,len(slidesB)): #for now, just displays displacement tracks on 2 ROIs for camB images
        plot_image = slidesB[s]
        plot_image = cv2.resize(plot_image, (1440, 810 ))  #do we need to resize
        cv2.imshow('all ROIs', plot_image)
       
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break    
    cv2.destroyAllWindows()
    
    
    return [tot_disps, disps_roi]



####################################################### MAIN ###########################################
#if __name__ == "__main__":

    #[newcameramtx, total_error] = sv_checker_calib.main() #extract the instrinsic camera parameter matrix if using 2D openCV camera calibration script
    
    #sequence of images from 2+ cameras; sequences must have same number of images
    camA = ['C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]
    camB = ['C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/B_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]
    
    print('number of MDM images to track: ' + str(len(camA)))
    
    num_roi = 2 #number of ROI for the user to select
    
    [tot_disps, disps_roi] = SIFT_detectTrack(camA, camB, tag = True)
