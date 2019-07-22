#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:26:26 2019

@author: Sruti
"""

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

my_path = "/Users/svutukur/Documents/GitHub/cv_mdm2019/"# enter the dir name
for fname in os.listdir(my_path):
    if fname.startswith("imagecalibrated.png"):
        os.remove(os.path.join(my_path, fname))

print('passed 1')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print('passed 2')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

a = 9 # of collumn corners
b = 7 # of row corners
objp = np.zeros((b*a,3), np.float32)
scale = 10
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)*scale

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(my_path + 'cam2******.tif')
print(len(images))

counter = 0
for idx,fname in enumerate(images):
    while counter < len(images):
        img = cv2.imread(fname)
        #print('passed 3' + ' ' + str(counter))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print('passed 4' + ' ' + str(counter))
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (a,b),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            #find corners more accurately
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners and pattern
            img = cv2.drawChessboardCorners(img, (a,b), corners2,ret)
            #cv2.imshow('img',img)
            #cv2.imwrite(my_path + str(counter) +'img.png', img)
            #cv2.waitKey(500)
            
            
            counter = counter + 1
            print("image" + str(counter-1))
        

cv2.destroyAllWindows()

# Calibration
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,np.shape(gray_matrix[::-1]), None, None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

###refine the camera matrix
##If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels.
##fix accuracy here?
img = cv2.imread(images[len(images)-1])
print(img)
h = img.shape[0]
w = img.shape[1]
print(h)
print(w)
#Returns the new camera matrix based on the free scaling paramete; play around with alpha parameter
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

## undistortion method 1 (same thing as undistort rectify map method)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x, y, w, h = roi
#x, y, w, h = w, h, w, h
#dst = dst[y:y + h, x:x + w]
cv2.imwrite(my_path + 'imagecalibrated.png', dst)


# Display the original image next to the calibrated image
plt.subplot(221), plt.imshow(img), plt.title('image original')
plt.subplot(222), plt.imshow(dst), plt.title('image calibrated')
plt.show()

#calculate reprojection error
#estimation of how exact is the found parameters
#absolute norm between what we got with our transformation and the corner finding algorithm
#average error = mean of the errors calculate for all the calibration images
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error = mean_error + error

print ("total error: ", mean_error / len(objpoints))