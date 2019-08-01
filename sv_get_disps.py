# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:31:57 2019

@author: svutukur
"""

import numpy as np
import cv2
import math
    

def get_3Ddisps(img1A, img2A, img1B, img2B):
    
    #A/B from separate cameras #1/2 from before pics and after pics
    #reshaping pixel coord matrices img1_pts, img2_pts to allow for matrix multiplication 
    img1A = np.array(img1A) ; img1A = img1A[:,:,0,:] ; img1B = np.array(img1B) ; img1B = img1B[:,:,0,:]   #all before pics
    img2A = np.array(img2A) ; img2A = img2A[:,:,0,:] ; img2B = np.array(img2B) ; img2B = img2B[:,:,0,:] #all after pics
     
     
    shape = np.shape(img1A)
    img1A_new = []; img2A_new = []
    img1B_new = []; img2B_new = []
     
     
    for j in range(shape[0]):
        x = np.ones((shape[1],1))
        new1 = np.hstack((img1A[j],x))
        new2 = np.hstack((img2A[j],x))
        img1A_new.append(new1)
        img2A_new.append(new2)
         
        new3 = np.hstack((img1B[j],x))
        new4 = np.hstack((img2B[j],x))
        img1B_new.append(new3)
        img2B_new.append(new4)
         
    #img1A_new.append(img1B_new) #all before pics (uv1 vector) includes from camera A and B
    before_pts = np.vstack((img1A_new, img1B_new))
    #img2A_new.append(img2B_new) #all after pics (uv1 vector) includes from camera A and B
    after_pts = np.vstack((img2A_new,img2B_new))
    
         
     
    S = np.array([[36355.55,1,1], [1,36369.3, 1], [1,1,1],
                   [29099.4, 1, 1], [1, 29097.97, 1], [1,1,1]])
    
    int_params = np.array([[1, 0.497, 3291.9], [0, 1, 2178.6], [0, 0, 1], 
                           [1, -0.4789, 3315.6], [0, 1, 2189.7],
                           [0, 0, 1]])
    
    ext_params = np.array([[0.00691, 0.9994, 0.0333, 138.4], 
                           [0.9651, -0.0085, 0.26149, 19.52],
                           [0.2619, 0.0303, -0.9646, 70.9],
                           [0.9987, -0.01885, 0.0465, 162.9],
                           [-0.0158, -0.9977, -0.06527, 19.39],
                           [0.04758, 0.06446, -0.9968, 69.4]])
    
    B1 = np.matmul(S,before_pts); A1 = np.matmul(int_params, ext_params)
    B2 = np.matmul(S,after_pts); A2 = np.matmul(int_params, ext_params)
     
    before_realcoords = np.linalg.lstsq(A1, B1, rcond = None) #numpy.linalg.lstsq(a, b, rcond='warn')
    after_realcoords = np.linalg.lstsq(A2, B2, rcond = None)
 
 
    disp3 = []
    first = 0
    
    while first < len(img1A)-1:        
        
        #get physical displacements from w1, w2
        disp2 = [] #each element is a displacement between two consecutive images
        
        for j in range(len(before_realcoords)):
            disp1 = math.sqrt((before_realcoords[j][0]-after_realcoords[j][0])**2 + (before_realcoords[j][1]-after_realcoords[j][1])**2 + (before_realcoords[j][2]-after_realcoords[j][2])**2) #distance formula 
            disp2.append(disp1)
        
        disp2 = np.array(disp2)
        disp3.append(disp2)
        #disp3 = np.hstack((disp3,disp2))
        first = first + 1;
        
        
    tot_disps = np.sum(disp3, axis = 0) #total displacement across all sets of images
    # the dimension of the matrices must be the same (same number of targets), otherwise it overwrites
    return tot_disps
     

    

def get_2Ddisps(img1_pts, img2_pts, H_mat,newcameramtx): 
    
    
    #reshaping pixel coord matrices img1_pts, img2_pts to allow for matrix multiplication 
    img1_pts = np.array(img1_pts) ; img1_pts = img1_pts[:,:,0,:]  
    img2_pts = np.array(img2_pts) ; img2_pts = img2_pts[:,:,0,:]
    
    shape = np.shape(img1_pts)
    img1_new = []; img2_new = []
    
    for j in range(shape[0]):
        x = np.ones((shape[1],1))
        new1 = np.hstack((img1_pts[j],x))
        new2 = np.hstack((img2_pts[j],x))
        img1_new.append(new1 )
        img2_new.append(new2)
    
    #img1_pts = np.append(img1_pts, np.ones((2,20,2)), 1)
    # img1_pts = np.append(img1_pts, np.ones((len(img1_pts),1)), 1)
    #img2_pts = np.append(img2_pts, np.ones((1,len(img1_pts),1)), 1)
    # img2_pts = np.append(img2_pts, np.ones((2,20,2)), 2)
    
    disp3 = []
    first = 0
    
    while first < len(list_names)-1:
    
        #get physical coord matrices w1, w2 from homography transformation, camera distortion matrix, scaling factor
        
        s = 1 #physical scaling factor
        w1 = []
        w2 = []
        
        for i in range(shape[1]): #iterates over each corner coordinate
            #w1.append(np.matmul(inv(newcameramtx), inv(H), img1_new[i])) #w1 physical coord
            #w2.append(np.matmul(inv(newcameramtx), inv(H), img2_new[i])) #w2 physical coord
            a = np.matmul(inv(H_mat[first]), img1_new[first][i]) * s
            w1.append(np.matmul(inv(newcameramtx), a))
            
            c = np.matmul(inv(H_mat[first]), img2_new[first][i]) * s
            w2.append(np.matmul(inv(newcameramtx), c))
            
        
        #get physical displacements from w1, w2
        disp2 = [] #each element is a displacement between two consecutive images
        
        for j in range(len(w1)):
            disp1 = math.sqrt((w1[j][0]-w2[j][0])**2 + (w1[j][1]-w2[j][1])**2 + (w1[j][2]-w2[j][2])**2) #distance formula 
            disp2.append(disp1)
        
        disp2 = np.array(disp2)
        disp3.append(disp2)
        #disp3 = np.hstack((disp3,disp2))
        first = first + 1;
        
        
    tot_disps = np.sum(disp3, axis = 0) #total displacement across all sets of images
    # the dimension of the matrices must be the same (same number of targets), otherwise it overwrites
    return tot_disps

def get_stats(disps_roi):
    for u in range(0,len(disps_roi)):
        data = disps_roi['ROI # ' + str(u)]
        print('ROI: ' + str(u))
        print('mean: ' + str(statistics.mean(data))) 
        print('median: ' + str(statistics.median(data)))
        print('stdev: ' + str(statistics.pstdev(data)))