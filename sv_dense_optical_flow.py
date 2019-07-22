#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:40:46 2019

@author: Sruti
"""

import cv2
import numpy as np
from PIL import Image
import os

my_path = "/Users/Sruti/Documents/GitHub/cv_mdm2019/sruti/"# enter the dir name
for fname in os.listdir(my_path):
    for i in range(9):
        if fname.startswith('opticalA' + str(i) + '.jpg'):
            os.remove(os.path.join(my_path, fname))

print('passed removal')

def draw_flow(img, flow, step=16):
    im = Image.open(img)
    w, h = im.size
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    #y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines + 0.5) #what is the 0.5?

#    # create image and draw
#    #vis = im
#    im = np.array(im)
#    cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
#    for (x1,y1),(x2,y2) in lines:
#        cv2.line(im,(x1,y1),(x2,y2),(0,255,0),1)
#        cv2.circle(im,(x1,y1),1,(0,255,0), -1)
#    return im

#    # create image and draw
    vis = np.array(im)
    vis = cv2.cvtColor(np.float32(im), cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# Create list of names here from A_Run24_Seq4_00000.tif up to A_Run24_Seq4_00009.tif
list_names = ['A_Run24_Seq4_0000' + str(i) + '.tif' for i in range(9)]
print(list_names)

# Read in the first frame
index1 = 0
print(index1)
#prev_array = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# Set index2 to read the second frame at the start
index2 = 1
print(index2)

# Until we reach the end of the list...
while index2 < len(list_names):
    frame1 = cv2.imread(list_names[index1],0)
    print(frame1)
    frame2 = cv2.imread(list_names[index2],0)
    print(frame2)
    print('passed 1')
    #next_array = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #hsv = np.zeros_like(frame1)
    #hsv[...,1] = 255

    # Calculate optical flow between the two frames
    flow = cv2.calcOpticalFlowFarneback(frame2, frame1, pyr_scale=0.5, 
                                        levels=5, winsize=13, iterations=10, 
                                        poly_n=5, poly_sigma=1.1, flags=0,flow=None) 
    print(flow)
    print('passed 2')

    # Normalize horizontal and vertical components
#    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
#    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
#    horz = horz.astype('uint8')
#    vert = vert.astype('uint8')
    

    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imwrite('opticalA' + str(index2) + '.jpg',frame1)
    print('passed 3')
    cv2.imwrite('opticalA' + str(index2) + '.jpg',draw_flow(list_names[index2], flow, step=16))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalB' + str(index2) + '.jpg',frame2)
        #cv2.imwrite('opticalhsv.png',rgb)

    # Increment index2 to go to next frame
    index1 = index1 + 1
    index2 = index2 + 1

cv2.destroyAllWindows()