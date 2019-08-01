# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:22:21 2019

@author: svutukur
"""

import numpy as np
import cv2



def corner_detector(gray,write_name):
    p0 = cv2.goodFeaturesToTrack(gray, **feature_params)

    img = gray

    for k in range(0,len(p0)):
        img = cv2.circle(color,(p0[k][0][0],p0[k][0][1]), radius = 10, color = (0,0,255), thickness=-1)

    cv2.imwrite(write_name,img)

    return len(p0)
        
######################################################## MAIN ############## #############################
if __name__ == "__main__":
    
   #list_names = ['C:/Users/svutukur/Documents/tbw1_data/A_Run143_Seq' + str(i) + '_00001.tif' for i in range(6,9)]
   #list_names = ['C:/Users/svutukur/Documents/multi_track/cam2_' + '0000' + str(i) + '.tif' for i in range(1,8)] 
   #list_names = ['C:/Users/svutukur/Documents/fancy_wand/cam1_' + '000' + str(i) + '.tif' for i in range(10,20)]
   list_names = ['C:/Users/svutukur/Documents/tbw3_sample_data/run001/seq007/A_run001_seq007_00' + str(i) + '.tiff' for i in range(11,15)]
   
   A = np.zeros((len(list_names),4))
   
   for i in range(0,len(list_names)):
       
       high_res = cv2.imread(list_names[i],0)
       color = cv2.imread(list_names[i])
       
       feature_params = dict(maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7,
                           gradientSize = 7,
                           mask = None)  
       
       low_res = cv2.pyrDown(high_res) #create Gaussian pyramid (turn down the resolution to find more corners)
    
       #blur image --> apply thresholing --> apply binary thresholding again
       blur = cv2.medianBlur(high_res,5) 
       ret,thresh_init = cv2.threshold(blur,15,255 ,cv2.THRESH_BINARY)
       thresh_img=cv2.adaptiveThreshold(thresh_init,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2) #apply adaptive and binary thresholding again
          
       
       A[i][0] = i
       A[i][1] = corner_detector(high_res,'frameOG.jpg')
       A[i][2] = corner_detector(low_res,'frameLow.jpg')
       A[i][3] = corner_detector(thresh_img,'frameThresh.jpg')
       
