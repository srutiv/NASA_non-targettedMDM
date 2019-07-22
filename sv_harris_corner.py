#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:21:09 2019

@author: Sruti

"""

#following functions the Harris corner detector algorthm
# many of these algorithms from O'Reilly CV python textbook from harris chapter

import sys
import numpy as np
import cv2
from common import anorm2, draw_str
from time import clock
from PIL import Image
import pylab
import matplotlib.pyplot as plt
from scipy.ndimage import filters


def compute_harris_response(im, sigma=3):
    #compute harris corner detector 
    #response function for each pixel in an imported grayscale image
    #returns an image with each pixel containing the value of the Harris response function
    
    #derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    #compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)
    
    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet/Wtr

def get_harris_points(harrisim, min_dis = 10, threshold = 0.1):
    #Return corners from a Harris response image
    #min_dist: min # of pixels separating corners and image boundary
    
    #find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    #get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T
    
    #get candidate values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    #sort candidates
    index = argsort(candidate_values)
    
    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dis:-min_dis,min_dis:-min_dis] = 1
    
    #select the best points taing min_disatnce into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0] - min_dis):(coords[i,0] + min_dis),
                              (coords[i,1] - min_dis):(coords[i,1] + min_dis)] = 0
    return filtered_coords


def plot_harris_points(image,filtered_coords):
    #plotting function to plot detected corner points on the images
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '*')
    axis('off')
    show()
    
"""the following functions are for image correspondence for the Harris
corner detector algorthm (i.e. comparing interest points across
images to find matching corners """

def get_descriptors(image,filtered_coords, wid=5):
    #For each point return, pixel values around the point using a 
    #neighborhood of width 2*wid+1
    #returns descriptors d1 and d2
    
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1, 
                      coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    return desc

def match(desc1, desc2, threshold = 0.05):
    #For each corner point descriptor in the first image, 
    #select its match to second image using normalized cross-correlation.
     
     n = len(desc1[0])
     
     #pair-wise distances
     d = -ones((len(desc1),len(desc2)))
     np.seterr(divide='ignore', invalid='ignore')
     for i in range(len(desc1)):
         for j in range(len(desc2)):
             d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
             d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
             ncc_value = sum(d1*d2)/(n-1)
             #print(ncc_value)
             #break
             if ncc_value > threshold:
                 d[i,j] = ncc_value
     ndx = argsort(-d)
     matchscores = ndx[:,0]

     return matchscores
 
def match_twosided(desc1,desc2,threshold=0.5):
    #Two-sided symmetric version of match() 
    #to filter our the matches that are not best both ways
    
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)
    
    ndx_12 = where(matches_12 >= 0)[0]
    
    #remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12


def appendimages(im1,im2):
    #Return a new image that appends the first two images side-by-side and 
    #connects the matched points with lines using the following code

    #select the image with the fewest rows and fill in gaps with empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2 - rows1, im1.shape[1]))), axis = 0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1 - rows2, im1.shape[1]))), axis = 0)
    
    return concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    #show figure with lines joining the accepted matches
    #input: im1, im2 (images as arrays), locs1, locs2 (feature locations),
    #matchscores (as output from 'match()'),
    #show_below (if OG images should be shown below matches)
    #im3 output
    
    im3 = appendimages(im1,im2)
    #if show_below:
        #im3data = np.vstack((im3data,im3data))
    plt.imshow(im3, cmap='gray')
    print('put im1 and im2 side by side')
    
    
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1], 
                 [locs1[i][0],locs2[m][0]+cols1], 'c' )
    axis('off')
    
    plt.show()
    print('display im1 and im2 side by side and show match lines')


#read image to array
im1 = array(Image.open('A_Run24_Seq4_00000.tif').convert('1'))
im2 = array(Image.open('A_Run24_Seq4_00001.tif').convert('1'))
print('imported images into array data')

wid = 5
harrisim1 = compute_harris_response(im1,5)
filtered_coords1 = get_harris_points(harrisim1,wid+1)
d1 = get_descriptors(im1,filtered_coords1, wid)

harrisim2 = compute_harris_response(im2,5)
filtered_coords2 = get_harris_points(harrisim2,wid+1)
d2 = get_descriptors(im2,filtered_coords2, wid)

print('starting matching')
matches = match(d1,d2)
print('finished matching')

print('start plotting')
plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
print('finished plotting')
