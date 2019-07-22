#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:21:16 2019

@author: Sruti
"""

from PIL import Image
from pylab import *

#read image to array
im = array(Image.open('empire.jpeg').convert('L'))

#create a new figure
figure()
gray()
#show contours with origin set at the upper left corner
contour(im, origin = 'image')
axis('equal')
axis('off')
show()

#read image to array
im1 = array(Image.open('pug.jpg').convert('L'))
print('completed 3')
im2 = array(Image.open('poodle.jpg').convert('L'))
print('completed 2')

#create a new figure
figure()
gray()
#show contours with origin set at the upper left corner
contour(im1, origin = 'image')
axis('equal')
axis('off')
show()
print('completed 3')
figure()
gray()
#show contours with origin set at the upper left corner
contour(im2, origin = 'image')
axis('equal')
axis('off')
show()
print('completed 4')

############################################################################





def main():
    import sys
    print('passed 3')
    try:
        #read image to array
        im1 = array(Image.open('pug.jpg').convert('1'))
        im2 = array(Image.open('poodle.jpg').convert('1'))
        print('passed 4')

        wid = 5
        harrisim = harris.compute_harris_response(im1,5)
        filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
        d1 = harris.get_descriptors(im1,filtered_coords1, wid)
        
        harrisim = harris.compute_harris_response(im2,5)
        filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
        d2 = harris.get_descriptors(im2,filtered_coords2, wid)
        
        print('starting matching')
        matches = harris.match_twosided(d1,d2)
        
        figure()
        gray()
        harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
        show()

if__name__ == '__main__':
    print('passed 2')
    main()