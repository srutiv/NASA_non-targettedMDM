# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:31:57 2019

@author: svutukur
"""

import numpy as np
import cv2
import math
import statistics
    

def get_3Ddisps(imgA_before, imgA_after, imgB_before, imgB_after):
    
    #A/B from separate cameras #1/2 from before pics and after pics
    #reshaping pixel coord matrices img1_pts, img2_pts to allow for matrix multiplication
    imgA_before = np.array(imgA_before) ; imgA_before = imgA_before[:,:,0,:] ; imgB_before = np.array(imgB_before) ; imgB_before = imgB_before[:,:,0,:]   #all before pics
    imgA_after = np.array(imgA_after) ; imgA_after = imgA_after[:,:,0,:] ; imgB_after = np.array(imgB_after) ; imgB_after = imgB_after[:,:,0,:] #all after pics
   
    #for camera A
    m11a = 0.00691; m12a = .9994; m13a = 0.0333; 
    m21a = 0.9651; m22a = 0.26149; m23a = 0.26149
    m31a = 0.2619; m32a = 0.0303;  m33a = -0.9646
    cxa = 3291.9; cya = 2178.6
    sxa = 36355.55; sya = 36369.3
    txa = 138.4; tya = 19.52; tza = 70.9

    
    #for camera B
    m11b = 0.00691; m12b = .9994; m13b = 0.0333; 
    m21b = 0.9651; m22b = 0.26149; m23b = 0.26149
    m31b = 0.2619; m32b = 0.0303;  m33b = -0.9646
    cxb = 3291.9; cyb = 2178.6
    sxb = 36355.55; syb = 36369.3
    txb = 162.9; tyb = 19.39; tzb = 69.4
    

    ##########################least squares problem to get physical coords ########################
    """
    Apologies because the following chunks of code are not condensed AT ALL. Each block calculates physical coordinates 
    of corners in 1 ROI for camera A or B. For more cameras, this part of the code needs to be condensed"""
    
    
    #before space point for camera A
    before_allrealA = [(0,0,0)]
    shape = np.shape(imgA_before)
    for j in range(0,shape[1]):
        x = imgA_before[0][j][0]
        y = imgA_before[0][j][1]
        
        p1a = (x - cxa)*m31a + sxa*m11a
        p2a = (x - cxa)*m32a + sxa*m12a
        p3a = (x - cxa)*m33a + sxa*m13a
        p4a = (y - cya)*m31a + sya*m21a
        p5a= (y - cya)*m32a + sya*m22a
        p6a = (y - cya)*m33a + sya*m23a
    
        p1b = (x - cxb)*m31b + sxb*m11b
        p2b = (x - cxb)*m32b + sxb*m12b
        p3b = (x - cxb)*m33b + sxb*m13b
        p4b = (y - cyb)*m31b + syb*m21b
        p5b= (y - cyb)*m32b + syb*m22b
        p6b = (y - cyb)*m33b + syb*m23b
        
        
        A = np.array([[p1a, p2a, p3a], 
             [p4a, p5a, p6a],
             [p1b, p2b, p3b],
             [p4b, p5b, p6b]])
        B = np.array([[(p1a*txa + p2a*tya + p3a*tza)],
              [(p4a*txa + p5a*tya + p6a*tza)],
              [(p1b*txb + p2b*tyb + p3b*tzb)],
              [(p4b*txb + p5b*tyb + p6b*tzb)]])*10 #10 is a scale factor; input your own
        
        [coord_beforeA, resid_beforeA, rank_beforeA, s_beforeA] = np.linalg.lstsq(A, B, rcond = None) #numpy.linalg.lstsq(a, b, rcond='warn')
        
        coord_beforeA = coord_beforeA.reshape(-1,1)
        before_allrealA.append(coord_beforeA)
    
    
    
    #before space point for camera B
    before_allrealB = [(0,0,0)]
    shape = np.shape(imgB_before)
    for j in range(0,shape[1]):
        x = imgB_before[0][j][0]
        y = imgB_before[0][j][1]
        
        p1a = (x - cxa)*m31a + sxa*m11a
        p2a = (x - cxa)*m32a + sxa*m12a
        p3a = (x - cxa)*m33a + sxa*m13a
        p4a = (y - cya)*m31a + sya*m21a
        p5a= (y - cya)*m32a + sya*m22a
        p6a = (y - cya)*m33a + sya*m23a
    
        p1b = (x - cxb)*m31b + sxb*m11b
        p2b = (x - cxb)*m32b + sxb*m12b
        p3b = (x - cxb)*m33b + sxb*m13b
        p4b = (y - cyb)*m31b + syb*m21b
        p5b= (y - cyb)*m32b + syb*m22b
        p6b = (y - cyb)*m33b + syb*m23b
        
        
        A = [[p1a, p2a, p3a], 
             [p4a, p5a, p6a],
             [p1b, p2b, p3b],
             [p4b, p5b, p6b]]
        B = [[(p1a*txa + p2a*tya + p3a*tza)],
              [(p4a*txa + p5a*tya + p6a*tza)],
              [(p1b*txb + p2b*tyb + p3b*tzb)],
              [(p4b*txb + p5b*tyb + p6b*tzb)]]
        
        [coord_beforeB, resid_beforeB, rank_beforeB, s_beforeB] = np.linalg.lstsq(A, B, rcond = None) #numpy.linalg.lstsq(a, b, rcond='warn')
        
        coord_beforeB = coord_beforeB.reshape(-1,1)
        before_allrealB.append(coord_beforeB)

    
    #after space point for camera A
    after_allrealA = [(0,0,0)]
    shape = np.shape(imgA_before)
    for j in range(0,shape[1]):
        x = imgA_before[0][j][0] #for all corners in just region 1
        y = imgA_before[0][j][1]
        
        p1a = (x - cxa)*m31a + sxa*m11a
        p2a = (x - cxa)*m32a + sxa*m12a
        p3a = (x - cxa)*m33a + sxa*m13a
        p4a = (y - cya)*m31a + sya*m21a
        p5a= (y - cya)*m32a + sya*m22a
        p6a = (y - cya)*m33a + sya*m23a
    
        p1b = (x - cxb)*m31b + sxb*m11b
        p2b = (x - cxb)*m32b + sxb*m12b
        p3b = (x - cxb)*m33b + sxb*m13b
        p4b = (y - cyb)*m31b + syb*m21b
        p5b= (y - cyb)*m32b + syb*m22b
        p6b = (y - cyb)*m33b + syb*m23b
        
        
        A = np.array([[p1a, p2a, p3a], 
             [p4a, p5a, p6a],
             [p1b, p2b, p3b],
             [p4b, p5b, p6b]])
        B = np.array([[(p1a*txa + p2a*tya + p3a*tza)],
              [(p4a*txa + p5a*tya + p6a*tza)],
              [(p1b*txb + p2b*tyb + p3b*tzb)],
              [(p4b*txb + p5b*tyb + p6b*tzb)]])
        
        [coord_afterA, resid_afterA, rank_afterA, s_afterA] = np.linalg.lstsq(A, B, rcond = None) #numpy.linalg.lstsq(a, b, rcond='warn')
        
        coord_afterA = coord_afterA.reshape(-1,1)
        after_allrealA.append(coord_afterA)
        
    #after space point for camera B
    after_allrealB = [(0,0,0)]
    shape = np.shape(imgB_before)
    for j in range(0,shape[1]):
        x = imgB_before[0][j][0]
        y = imgB_before[0][j][1]
        
        p1a = (x - cxa)*m31a + sxa*m11a
        p2a = (x - cxa)*m32a + sxa*m12a
        p3a = (x - cxa)*m33a + sxa*m13a
        p4a = (y - cya)*m31a + sya*m21a
        p5a= (y - cya)*m32a + sya*m22a
        p6a = (y - cya)*m33a + sya*m23a
    
        p1b = (x - cxb)*m31b + sxb*m11b
        p2b = (x - cxb)*m32b + sxb*m12b
        p3b = (x - cxb)*m33b + sxb*m13b
        p4b = (y - cyb)*m31b + syb*m21b
        p5b= (y - cyb)*m32b + syb*m22b
        p6b = (y - cyb)*m33b + syb*m23b
        
        
        A = [[p1a, p2a, p3a], 
             [p4a, p5a, p6a],
             [p1b, p2b, p3b],
             [p4b, p5b, p6b]]
        B = [[(p1a*txa + p2a*tya + p3a*tza)],
              [(p4a*txa + p5a*tya + p6a*tza)],
              [(p1b*txb + p2b*tyb + p3b*tzb)],
              [(p4b*txb + p5b*tyb + p6b*tzb)]]
        
        [coord_afterB, resid_afterB, rank_afterB, s_afterB] = np.linalg.lstsq(A, B, rcond = None) #numpy.linalg.lstsq(a, b, rcond='warn')
        
        coord_afterB = coord_afterB.reshape(-1,1)
        after_allrealB.append(coord_afterB)

    #from cam A
    disp3 = []
    before_allrealA = before_allrealA[1:]
    before_allrealB = before_allrealB[1:]
    after_allrealA = after_allrealA[1:]
    after_allrealB = after_allrealB[1:]
    first = 0
    
    while first < len(imgA_before)-1:    #len(imgA_before) = number of sets of images    
        
        disp2 = [] #each element is a displacement between two consecutive images
        
        for j in range(len(before_allrealA)):
            #list[corner #][x/y/z]
            disp1 = np.sqrt((before_allrealA[j][0]-after_allrealA[j][0])**2 + 
                              (before_allrealA[j][1]-after_allrealA[j][1])**2 + 
                              (before_allrealA[j][2]-after_allrealA[j][2])**2, dtype = 'float64') #distance formula
            disp2.append(disp1)
    
        disp2 = np.array(disp2)
        disp3.append(disp2)
        #disp3 = np.hstack((disp3,disp2))
        first = first + 1;        
        
    tot_disps = np.sum(disp3, axis = 0) #total displacement across all sets of images
    # the dimension of the matrices must be the same (same number of targets), otherwise it 
    tot_disps = tot_disps
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



"""This was a failed attempt at condensing the matrices for the least squares problem  
    
##    shape = np.shape(imgA_before)
##    imgA_before_new = []; imgA_after_new = []
##    imgB_before_new = []; imgB_after_new = []
##     
##     
##    for j in range(shape[0]):
##        x = np.ones((shape[1],1))
##        new1 = np.hstack((imgA_before[j],x))
##        new2 = np.hstack((imgA_after[j],x))
##        imgA_before_new.append(new1)
##        imgA_after_new.append(new2)
##         
##        new3 = np.hstack((imgB_before[j],x))
##        new4 = np.hstack((imgB_after[j],x))
##        imgB_before_new.append(new3)
##        imgB_after_new.append(new4)
##         
##    #imgA_before_new.append(imgB_before_new) #all before pics (uv1 vector) includes from camera A and B
##    before_pts = np.vstack((imgA_before_new, imgB_before_new))
##    #imgA_after_new.append(imgB_after_new) #all after pics (uv1 vector) includes from camera A and B
##    after_pts = np.vstack((imgA_after_new,imgB_after_new))
##    
##         
##     
##    S = np.array([[36355.55,1,1], [1,36369.3, 1], [1,1,1],
##                   [29099.4, 1, 1], [1, 29097.97, 1], [1,1,1]])
##    
##    int_params = np.array([[1, 0.497, 3291.9], [0, 1, 2178.6], [0, 0, 1], 
##                           [1, -0.4789, 3315.6], [0, 1, 2189.7],
##                           [0, 0, 1]])
##    
##    ext_params = np.array([[0.00691, 0.9994, 0.0333, 138.4], 
##                           [0.9651, -0.0085, 0.26149, 19.52],
##                           [0.2619, 0.0303, -0.9646, 70.9],
##                           [0.9987, -0.01885, 0.0465, 162.9],
##                           [-0.0158, -0.9977, -0.06527, 19.39],
##                           [0.04758, 0.06446, -0.9968, 69.4]])
##    
##    B1 = np.matmul(S,before_pts); A1 = np.matmul(int_params, ext_params)
##    B2 = np.matmul(S,after_pts); A2 = np.matmul(int_params, ext_params)
##     
##    before_realcoords = np.linalg.lstsq(A1, B1, rcond = None) #numpy.linalg.lstsq(a, b, rcond='warn')
##    after_realcoords = np.linalg.lstsq(A2, B2, rcond = None)

"""