#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:26:26 2019

@author: Sruti
"""

import numpy as np
import cv2
import glob
import math
import pickle
import matplotlib.pyplot as plt
import time

# Calculate distance between two points
def _pdist(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the number of edges we have in the checkerboard (in our case 13 * 13)
n_rows = 13
n_cols = 13
n_cols_and_rows = (n_cols, n_rows)
n_rows_and_cols = (n_rows, n_cols)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((n_rows * n_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:n_rows, 0:n_cols].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# The pictures of the checkerboard we took for the test

mypath = "/home/sruti/Desktop/cv_fun/calibrate_sv/"

# example: mypath="/sruti/Desktop/cv_fun/calibrate_sv/left03.jpg"


print ("Getting images from " + mypath)
images = glob.glob(mypath + '*.JPG')
print ("images is: " + str(images))

criteria_calibrator = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria = criteria_calibrator

for idx, fname in enumerate(images):
    print ("\nImage " + fname)
    if time.sleep(10):
        break
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the edges
    ret, corners = cv2.findChessboardCorners(gray, n_rows_and_cols, None)
    
    # If found, add obj points and image points
    if ret == True:
        print (" found " + str(len(corners)) + " corners.")
    objpoints.append(objp)
    
    # cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) didnt work, 
    #I couldnt make it work copying the calibrator code
    imgpoints.append(corners)

    # Draw and display the edges on the original photo
    cv2.drawChessboardCorners(img, n_rows_and_cols, corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey(500)

# Show how many image points and object points we found
print ("objpoints len: " + str(len(objpoints)))
print ("imgpoints len: " + str(len(imgpoints)))

# Find the camera matrix and save it in the data folder (test photos)

try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    datathings = (ret, mtx, dist, rvecs, tvecs)
    outf = open(mypath + "calibration_return_values_rows_and_cols.pickle", "rb")
    pickle.dump(datathings, outf)
    fieldnames = ["ret", "mtx", "dist", "rvecs", "tvecs"]
    for fieldname, data in zip(fieldnames, datathings):
        print (fieldname + ": ")
        print (data)
    print ("ret, mtx, dist, rvecs, tvecs:")
    print (ret, mtx, dist, rvecs, tvecs)
except:
    print ("Failed getting cv2.calibrateCamera")
    pass
# cv2.destroyAllWindows()


# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread('/home/stagiaire/Bureau/New_Calibrage/GRE/IMG_700101_000255_0000_RGB.JPG')

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistortion
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('Desktop/imagecalibre.png', dst)

# Display the original image next to the calibrated image
plt.subplot(221), plt.imshow(img), plt.title('image originale')
plt.subplot(222), plt.imshow(dst), plt.title('image calibree')
plt.show()

# Calculation of the error
mean_error = 0
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print ("total error: ", mean_error / len(objpoints))