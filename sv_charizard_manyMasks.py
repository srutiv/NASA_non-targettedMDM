# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:44:37 2019

@author: svutukur
"""

import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
import inspect
import sv_checker_calib
from numpy.linalg import inv
import math
from trials2 import big

#import sv_blob_finder

    
def corner_detector(list_names):

    img1_pts = []; img2_pts = []; H_mat = []
    
    # Generate some random colors later used to display movement paths
    #color = np.random.randint(0,255,(100,3))
    color1 = np.ones((100,1))*255; color2 = np.ones((100,1))*246; color3 = np.zeros((100,1))
    color = np.hstack((color1,color2,color3))
    
    index = 0
    
    #trials #import script that contains get_ROI in refPt
    
    # Take first frame
    old_frame = cv2.imread(list_names[0])
    old_gray = cv2.imread(list_names[0],0)
    
#    old_frame = old_frame[]
#    old_gray = old_gray[]
    #old_gray = cv2.cvtColor(old_gray, cv2.COLOR_BGR2GRAY)
    
    [refPt,num_roi, masked_grays, boundboxes] = big(list_names[0])
    
    #slides = [[0]*num_roi]*(len(list_names)-1) #2D list: image set  x num of ROIs
    #slides = np.zeros((1,num_roi))
    slides_rois = np.zeros((1,num_roi))
    
    for q in range(0,num_roi):
        
        slides_frames = np.zeros(((len(list_names)-1),1))
        #params for ShiTomasi corner detection
        feature_params = dict(maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7,
                               gradientSize = 7,
                               mask = masked_grays[q]) #ROI from trials
        
        
        # goodFeaturesToTrack determines strong corners on an image
        # can be used to initialize any point-based tracker such as the calcOpticalFlowPyrLK
        p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)
        prev_coord = p0
        
        print('number of corners found: '+ str(len(p0)))
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        
        
        for index in range(0,(len(list_names)-1)):
            print('image' + str(index))
            frame = cv2.imread(list_names[index+1])
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                          maxLevel = 10,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))
            
            # calculate optical flow
            # for a sparse feature set using the iterative LK method with pyramids       
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            #p1 = p1[st==1] #shouldn't p1 = p1[st==1]??; i.e. only found points?
            print('p0 = ' + str(len(p0))); print('p1 = ' + str(len(p1))); print('prev_coord = ' + str(len(prev_coord)))
            
            if (len(p1) != len(prev_coord)):
                print('didnt find same corners')
                #can't find corners --> iterate over new lk parameters
                for i in range(20,100,2): #change criteria for max number of iterations first 
                    lk_params['criteria'][1] = i
                    print('criteria = ' + str(i))
                    
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #recalculate p1
                    #p1 = p1[st==1] #shouldn't p1 = p1[st==1]??
                    print ('new p1 length = ' + str(len(p1)))
                    
                    if (len(p1) != len(prev_coord)):
                        print('still bad. continue changing criteria')
                        continue
                    else:
                        print('found good criteria')
                        break
            else:
                print('found same corners')
            
            print('p0 = ' + str(len(p0))); print('p1 = ' + str(len(p1))); print('prev_coord = ' + str(len(prev_coord)))
            #previous block of code should ensure that len(p1) == len(prev_coord)
            
            H, Hmask = cv2.findHomography(p0, p1, cv2.RANSAC,5.0) #H is 3x3 homography matrix
            
            #print(prev_coord[0][0])
            img1_pts.append(prev_coord)
            img2_pts.append(p1)
            H_mat.append(H)
            
            prev_coord = p1
            #print(prev_coord[0][0])
            
            # Select good points #does p0 need to be reshaped to this good_new at the end? shouldn't p1 = p1[st==1]??
            good_new = p1
            good_old = p0
            
            for i in range(len(p1)):
                frame = cv2.putText(frame, str(i), (p1[i][0][0],int(p1[i][0][1]+100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 5) 
                
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) 
            
            
            img = cv2.add(frame,mask)
            img = cv2.add(img,boundboxes[q]) #plot ROIs and OG image; boundbox from trials
            img = cv2.resize(img, (960, 540))  
            
            #slides[index][q] = img #problems updating (only first collumn is updating)
            #slides = np.vstack((slides,img))
            slides_frames = np.vstack((slides_frames,img)) #add images from consective images to slides_frames
        
            # Now update the previous frame and previous points
            old_frame = frame.copy()
            index = index + 1
            #p0 = good_new.reshape(-1,1,2)
        
        slides_rois = np.hstack((slides_rois,slides_frames)) #add slide_frames to images from each roi
        
    #iterate over each consecutive image set, stack ROIs for a respective set together, display, move to next set
    #f, axarr = plt.subplots(2,2)
    for s in range(0,len(slides)):
        plot_image = np.concatenate((slides[s][0], slides[s][1]), axis=1) #slide[image set#][roi #]
        cv2.imshow('all ROIs', plot_image)
        #axarr[1,0].imshow(slides[2][s])
        #axarr[1,1].imshow(slides[3][s])
        
        #cv2.imshow('frame',all_regions)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break    
    cv2.destroyAllWindows()
        
    return [img1_pts, img2_pts, H_mat, slides] #prev_coord and p1 just the latest for verification; img is the last one for plotting
    

######################################################## MAIN ###########################################
if __name__ == "__main__":
    
    [newcameramtx, total_error] = sv_checker_calib.main() #extract the camera distorion matrix
    # Create list of names here from A_Run24_Seq4_00000.tif up to A_Run24_Seq4_00009.tif
    #list_names = ['C:/Users/svutukur/Documents/tbw1_data/A_Run143_Seq' + str(i) + '_00001.tif' for i in range(6,9)]
    #list_names = ['C:/Users/svutukur/Documents/multi_track/cam2_' + '0000' + str(i) + '.tif' for i in range(1,8)] 
    #list_names = ['C:/Users/svutukur/Documents/fancy_wand/cam1_' + '0000' + str(i) + '.tif' for i in range(1,9)]
    list_names = ['C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]
    #list_names = ['/Volumes/SRUTI2019/cv_mdm2019B/tbw1_data/A_Run143_Seq' + str(i) +'_00001.tif' for i in range(6,8)]
    
    print('number of MDM images to track: ' + str(len(list_names)))
      
    [img1_pts, img2_pts, H_mat,slides] = corner_detector(list_names)
    
    ############################displacement calculator#########################################################
#    cv2.imwrite('last_position.jpg',img)
#    
#    disp3 = []
#    first = 0
#    
#    #reshaping pixel coord matrices img1_pts, img2_pts to allow for matrix multiplication 
#    img1_pts = np.array(img1_pts) ; img1_pts = img1_pts[:,:,0,:]  
#    img2_pts = np.array(img2_pts) ; img2_pts = img2_pts[:,:,0,:]
#    
#    shape = np.shape(img1_pts)
#    img1_new = []; img2_new = []
#    
#    for j in range(shape[0]):
#        x = np.ones((shape[1],1))
#        new1 = np.hstack((img1_pts[j],x))
#        new2 = np.hstack((img2_pts[j],x))
#        img1_new.append(new1 )
#        img2_new.append(new2)
#    
#    #img1_pts = np.append(img1_pts, np.ones((2,20,2)), 1)
#    # img1_pts = np.append(img1_pts, np.ones((len(img1_pts),1)), 1)
#    #img2_pts = np.append(img2_pts, np.ones((1,len(img1_pts),1)), 1)
#    # img2_pts = np.append(img2_pts, np.ones((2,20,2)), 2)
#    
#    
#    while first < len(list_names)-1:
#    
#        #get physical coord matrices w1, w2 from homography transformation, camera distortion matrix, scaling factor
#        
#        s = 1 #physical scaling factor
#        w1 = []
#        w2 = []
#        
#        for i in range(shape[1]): #iterates over each corner coordinate
#            #w1.append(np.matmul(inv(newcameramtx), inv(H), img1_new[i])) #w1 physical coord
#            #w2.append(np.matmul(inv(newcameramtx), inv(H), img2_new[i])) #w2 physical coord
#            a = np.matmul(inv(H_mat[first]), img1_new[first][i]) * s
#            w1.append(np.matmul(inv(newcameramtx), a))
#            
#            c = np.matmul(inv(H_mat[first]), img2_new[first][i]) * s
#            w2.append(np.matmul(inv(newcameramtx), c))
#            
#        
#        #get physical displacements from w1, w2
#        disp2 = [] #each element is a displacement between two consecutive images
#        
#        for j in range(len(w1)):
#            disp1 = math.sqrt((w1[j][0]-w2[j][0])**2 + (w1[j][1]-w2[j][1])**2 + (w1[j][2]-w2[j][2])**2) #distance formula 
#            disp2.append(disp1)
#        
#        disp2 = np.array(disp2)
#        disp3.append(disp2)
#        #disp3 = np.hstack((disp3,disp2))
#        first = first + 1;
#        
#        
#    tot_disps = np.sum(disp3, axis = 0) #total displacement across all sets of images
#    # the dimension of the matrices must be the same (same number of targets), otherwise it overwrites