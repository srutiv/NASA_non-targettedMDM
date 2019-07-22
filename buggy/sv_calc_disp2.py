#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:46:52 2019

@author: Sruti
"""

import numpy as np
import cv2
import sys
import time
#nothing, clock, draw_str

MHI_DURATION = 0.5
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    
def draw_motion_comp(vis, x, y, w, h, angle, color):
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0))
    r = min(w/2, h/2)
    cx, cy = x+w/2, y+h/2
    angle = angle*np.pi/180
    cv2.circle(vis, (cx, cy), r, color, 3)
    cv2.line(vis, (cx, cy), (int(cx+np.cos(angle)*r), int(cy+np.sin(angle)*r)), color, 3)


list_names = ['A_Run24_Seq4_0000' + str(i) + '.tif' for i in range(9)]
print(list_names)

# Until we reach the end of the list...
frame1 = cv2.imread(list_names[0],0)
print(frame1)
frame2 = cv2.imread(list_names[1],0)
print(frame2)

print('passed 1')


#    cv2.namedWindow('motempl')
#    visuals = ['input', 'frame_diff', 'motion_hist', 'grad_orient']
#    cv2.createTrackbar('visual', 'motempl', 2, len(visuals)-1, nothing)
#    cv2.createTrackbar('threshold', 'motempl', DEFAULT_THRESHOLD, 255, nothing)

#cam = video.create_capture(video_src, fallback='synth:class=chess:bg=../cpp/lena.jpg:noise=0.01')
#ret, frame = cam.read()
frame = frame1
h, w = frame.shape[:2]
#prev_frame = frame.copy()
prev_frame = frame2
motion_history = np.zeros((h, w), np.float32)
hsv = np.zeros((h, w, 3), np.uint8)
hsv[:,:,1] = 255

print('passed 2')
counter = 0

while True:
    #ret, frame = cam.read()
    frame = frame1
    frame_diff = cv2.absdiff(frame, prev_frame)
    #gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    #thrs = cv2.getTrackbarPos('threshold', 'motempl')
    thrs = 255
    ret, motion_mask = cv2.threshold(frame_diff, thrs, 1, cv2.THRESH_BINARY)
    timestamp = time.clock()
    cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
    mg_mask, mg_orient = cv2.motempl.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5 )
    seg_mask, seg_bounds = cv2.motempl.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)

    #visual_name = visuals[cv2.getTrackbarPos('visual', 'motempl')]
    visual_name = 'input'
    #visual_name = 'frame_diff'
    #visual_name = 'motion_hist'
    #visual_name = 'grad_orient'
    
    print('passed 3' + ' ' + str(counter))
    
    if visual_name == 'input':
        vis = frame.copy()
    elif visual_name == 'frame_diff':
        vis = frame_diff.copy()
    elif visual_name == 'motion_hist':
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif visual_name == 'grad_orient':
        hsv[:,:,0] = mg_orient/2
        hsv[:,:,2] = mg_mask*255
        vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    print('passed 4' + ' ' + str(counter))

    for i, rect in enumerate([(0, 0, w, h)] + list(seg_bounds)):
        x, y, rw, rh = rect
        area = rw*rh
        if area < 64**2:
            continue
        silh_roi   = motion_mask   [y:y+rh,x:x+rw]
        orient_roi = mg_orient     [y:y+rh,x:x+rw]
        mask_roi   = mg_mask       [y:y+rh,x:x+rw]
        mhi_roi    = motion_history[y:y+rh,x:x+rw]
        if cv2.norm(silh_roi, cv2.NORM_L1) < area*0.05:
            continue
        angle = cv2.calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)
        color = ((255, 0, 0), (0, 0, 255))[i == 0]
        draw_motion_comp(vis, rect, angle, color)
    
    print('passed 5' + ' ' + str(counter))

    draw_str(vis, (20, 20), visual_name)
    
    print('passed 6' + ' ' + str(counter))
    
    cv2.imwrite('motemplimg.jpg' + ' ' + str(counter), vis)
    
    counter = counter + 1
    
    prev_frame = frame.copy()


cv2.destroyAllWindows() 	