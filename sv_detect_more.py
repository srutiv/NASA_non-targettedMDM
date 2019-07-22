# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:22:21 2019

@author: svutukur
"""

import numpy as np
import cv2



def corner_detector(res,name):
    
    # Generate some random colors later used to display movement paths
    #color = np.random.randint(0,255,(100,3))
    color1 = np.ones((100,1))*255; color2 = np.ones((100,1))*246; color3 = np.zeros((100,1))
    color = np.hstack((color1,color2,color3))
    
    img = res
    
    #params for ShiTomasi corner detection
    #maskimage = cv2.imread(list_names,0) #1-channel image
    feature_params = dict(maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7,
                           gradientSize = 7,
                           mask = None)    
    
    # goodFeaturesToTrack determines strong corners on an image
    # can be used to initialize any point-based tracker such as the calcOpticalFlowPyrLK
    p0 = cv2.goodFeaturesToTrack(img, **feature_params)
    p1 = p0 #jsut needed for drawing; adapted from optical flow scripts
    
    print('number of corners found: '+ str(len(p0)))
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(img) 
   
    # Select good points #does p0 need to be reshaped to this good_new at the end? shouldn't p1 = p1[st==1]??
    good_new = p1
    good_old = p0
    
    #for i in range(len(p1)):
        #frame = cv2.putText(img, str(i), (p1[i][0][0],int(p1[i][0][1]+100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 5) 
        
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        img = cv2.circle(img,(a,b),5,color[i].tolist(),-1) 
        
    #img = cv2.add(color_img,mask); #img = cv2.resize(img, (960, 540))  
    cv2.imwrite(name,img)
    
    cv2.destroyAllWindows()
        
######################################################## MAIN ###########################################
if __name__ == "__main__":
    
   #image = 'C:/Users/svutukur/Documents/fancy_wand/cam1_00010'
   #image  = 'C:/Users/svutukur/Documents/tbw1_data/A_Run143_Seq6_00001.tif'
   #image  = 'C:/Users/svutukur/Documents/multi_track/cam2_00007.tif'
   image  = 'C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_0020.tiff'

   high_res = cv2.imread(image,0)
   low_res = cv2.pyrDown(high_res) #create Gaussian pyramid (turn down the resolution to find more corners)
    
   corner_detector(high_res,'frameOG.jpg')
   corner_detector(low_res,'frameLow.jpg')