# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:44:37 2019

@author: svutukur
"""

import numpy as np
import cv2
from sv_getROIs import get_ROIs
from sv_get_disps import get_3Ddisps

    ############################displacement calculator#########################################################

    
def corner_detectTrack(camA, camB, tag = True):
    
    # Generate some random colors later used to display movement paths
    #color = np.random.randint(0,255,(100,3))
    color1 = np.ones((100,1))*255; color2 = np.ones((100,1))*246; color3 = np.zeros((100,1))
    color = np.hstack((color1,color2,color3))
    
    index = 0
    
    
    [refPtA, masked_grays1, boundboxes1] = get_ROIs(camA[0])
    [refPtB, masked_grays2, boundboxes2] = get_ROIs(camB[0])
    
     # Take first frame
    old_frame = cv2.imread(camA[0])
    old_gray = cv2.imread(camA[0],0)
     
    slides = [[None]*num_roi for x in range(len(second_cam)-1)] #2D list: image set  x num of ROIs
    disps_roi = {'ROI # ' + str(u): 0 for u in range(num_roi)}
    
    for q in range(0,num_roi):
        
        img1A = []; img2A = []; H_matA = []
        img1B = []; img2B = []; H_matB = []
        
        #params for ShiTomasi corner detection
        feature_params = dict(maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7,
                               gradientSize = 7,
                               mask = masked_grays2[q]) #ROI from trials
        
        
        # goodFeaturesToTrack determines strong corners on an image
        # can be used to initialize any point-based tracker such as the calcOpticalFlowPyrLK
        # first search for the features in the rois from the first camera; 
        # search for those same initialized features in all the images in the second camera
        
        p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)  #search for features from first camera pic
        prev_coord = p0
        
        print('number of corners found: '+ str(len(p0)))
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        
        ##################################################################  for cam A  ###########################################################
        
        while index < (len(camA)-1):
            print('image' + str(index))
            frame = cv2.imread(camA[index+1])
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                          maxLevel = 10,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))
            
            # calculate optical flow
            # for a sparse feature set using the iterative LK method with pyramids       
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            p1 = p1[st==1]
            
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
            #not really needed
            
            #print(prev_coord[0][0])
            img1A.append(prev_coord)
            img2A.append(p1)
            H_matA.append(H)
            
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
            img = cv2.add(img,boundboxes2[q]) #plot ROIs and OG image; boundbox from trials
            
            #slides[index][q] = img #problems updating (only first collumn is updating)
            #slides = np.vstack((slides,img))
            #add images from consective images to slides_frames
            slides[index][q] = img
        
            # Now update the previous frame and previous points
            old_frame = frame.copy()
            index = index + 1
            #p0 = good_new.reshape(-1,1,2)
            
        index = 0;  #reset index for camB
        ##################################################################  for camB  ###########################################################
#        old_frame = cv2.imread(camB[0])
#        old_gray = cv2.imread(camB[0],0)
#        
#        while index < (len(camB)-1):
#            print('image' + str(index))
#            frame = cv2.imread(camB[index+1])
#            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            
#            # Parameters for lucas kanade optical flow
#            lk_params = dict( winSize  = (15,15),
#                          maxLevel = 10,
#                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))
#            
#            # calculate optical flow
#            # for a sparse feature set using the iterative LK method with pyramids       
#            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#            
#            p1 = p1[st==1]
#            #p1 = p1[st==1] #shouldn't p1 = p1[st==1]??; i.e. only found points?
#            print('p0 = ' + str(len(p0))); print('p1 = ' + str(len(p1))); print('prev_coord = ' + str(len(prev_coord)))
#            
#            if (len(p1) != len(prev_coord)):
#                print('didnt find same corners')
#                #can't find corners --> iterate over new lk parameters
#                for i in range(20,100,2): #change criteria for max number of iterations first 
#                    lk_params['criteria'][1] = i
#                    print('criteria = ' + str(i))
#                    
#                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #recalculate p1
#                    #p1 = p1[st==1] #shouldn't p1 = p1[st==1]??
#                    print ('new p1 length = ' + str(len(p1)))
#                    
#                    if (len(p1) != len(prev_coord)):
#                        print('still bad. continue changing criteria')
#                        continue
#                    else:
#                        print('found good criteria')
#                        break
#            else:
#                print('found same corners')
#            
#            print('p0 = ' + str(len(p0))); print('p1 = ' + str(len(p1))); print('prev_coord = ' + str(len(prev_coord)))
#            #previous block of code should ensure that len(p1) == len(prev_coord)
#            
#            H, Hmask = cv2.findHomography(p0, p1, cv2.RANSAC,5.0) #H is 3x3 homography matrix
#            #not really needed
#            
#            #print(prev_coord[0][0])
#            img1B.append(prev_coord)
#            img2B.append(p1)
#            H_matB.append(H)
#            
#            prev_coord = p1
#            #print(prev_coord[0][0])
#            
#            # Select good points #does p0 need to be reshaped to this good_new at the end? shouldn't p1 = p1[st==1]??
#            good_new = p1
#            good_old = p0
#            
#            for i in range(len(p1)):
#                frame = cv2.putText(frame, str(i), (p1[i][0][0],int(p1[i][0][1]+100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 5) 
#                
#            # draw the tracks
#            for i,(new,old) in enumerate(zip(good_new,good_old)):
#                a,b = new.ravel()
#                c,d = old.ravel()
#                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) 
#            
#            
#            img = cv2.add(frame,mask)
#            img = cv2.add(img,boundboxes2[q]) #plot ROIs and OG image; boundbox from trials
#            
#            #slides[index][q] = img #problems updating (only first collumn is updating)
#            #slides = np.vstack((slides,img))
#            #add images from consective images to slides_frames
#            slides[index][q] = img
#        
#            # Now update the previous frame and previous points
#            old_frame = frame.copy()
#            index = index + 1
#            #p0 = good_new.reshape(-1,1,2)
#            
#        index = 0 #reset index for new ROI
#        tot_disps = get_3Ddisps(img1A, img2A, img1B, img2B)
#        disps_roi['ROI # ' + str(q)] = tot_disps
#        
#    #iterate over each consecutive image set, stack ROIs for a respective set together, display, move to next set
#    #f, axarr = plt.subplots(2,2)
#    for s in range(0,len(slides)): #for now, just displays on camB images
#        plot_image = np.concatenate((slides[s][0], slides[s][1]), axis=1) #slide[image set#][roi #]
#        plot_image = cv2.resize(plot_image, (1920, 540))  #do we need to resize
#        cv2.imshow('all ROIs', plot_image)
#        #axarr[1,0].imshow(slides[2][s])
#        #axarr[1,1].imshow(slides[3][s])
#        
#        #cv2.imshow('frame',all_regions)
#        k = cv2.waitKey(0) & 0xff
#        if k == 27:
#            break    
#    cv2.destroyAllWindows()
#    
#    return [disps_roi, slides]

        
######################################################## MAIN ###########################################
if __name__ == "__main__":
    
    #[newcameramtx, total_error] = sv_checker_calib.main() #extract the camera distorion matrix
    # Create list of names here from A_Run24_Seq4_00000.tif up to A_Run24_Seq4_00009.tif
    #list_names = ['C:/Users/svutukur/Documents/tbw1_data/A_Run143_Seq' + str(i) + '_00001.tif' for i in range(6,9)]
    #list_names = ['C:/Users/svutukur/Documents/multi_track/cam2_' + '0000' + str(i) + '.tif' for i in range(1,8)] 
    #list_names = ['C:/Users/svutukur/Documents/fancy_wand/cam1_' + '0000' + str(i) + '.tif' for i in range(1,9)]
    first_cam = ['/A_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]
    second_cam = ['/B_run001_seq007_00' + str(i) + '.tiff' for i in range(10,16)]
    #list_names = ['/Volumes/SRUTI2019/cv_mdm2019B/tbw1_data/A_Run143_Seq' + str(i) +'_00001.tif' for i in range(6,8)]
    
    print('number of MDM images to track: ' + str(len(first_cam)))
     
    num_roi = 3
    
    corner_detectTrack(first_cam, second_cam, tag = True) #tag means that it's the first camera where the features are used for initialization
    
