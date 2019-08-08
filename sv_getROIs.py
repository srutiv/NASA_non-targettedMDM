# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:14:32 2019

@author: svutukur
"""
import numpy as np
import cv2
           

def get_ROIs(im_name):
    
    dPt = []
    drawing = False
    num_roi = 0
    
        
    def click_and_crop(event, x, y, flags, param):
            #global refPt,dPt,drawing
            global dPt, drawing
            # grab references to the global variables
            #global refPt, cropping, num_roi
            
            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being performed
            if event == cv2.EVENT_LBUTTONDOWN:
                
                dPt = [(x,y)]
                #ix, iy = x, y
                drawing = True
                #refPt.append((x, y))
                
            # check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                
                dPt.append((x,y))
                
                # draw a rectangle around the region of interest
                #cv2.rectangle(image, refPt[num_roi*2+1], refPt[num_roi*2+2], (0, 255, 0), 2)
                drawing = False
                cv2.rectangle(image,dPt[0], dPt[1], (0,255,0), 4)
                
                ROI[dPt[0][1]:dPt[1][1],dPt[0][0]:dPt[1][0]] = 255 #the order of these are important; (0,0) at top left
                boundboxes[num_roi] = cv2.rectangle(image, (dPt[0][0],dPt[0][1]), (dPt[1][0],dPt[1][1]), (0,255,0), 6)
                masked_img = cv2.bitwise_and(image,image,mask = ROI)
                masked_grays[num_roi] = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                #each rectangle to be plotted follows the index pattern num_roi*2 + 1
                cv2.imshow("image", image)
            
    color = cv2.imread(im_name)
     
    # perform the actual resizing of the image and show it
    image = color; #image = cv2.resize(image, (960, 540)) #take color image and resize
    #image = color; image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #take color image and resize
    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    
    #create mask
    #replace "image" with "color" --> OG
    ROI = np.zeros(image.shape[:2], np.uint8)
    #x1 = refPt[1][0]; x2 = refPt[2][0]; y1 = refPt[1][1]; y2 = refPt[2][1] 
    
    masked_grays = [0] * 2 #2 = number of roi
    boundboxes = [0]* 2
    
    while num_roi < 2:
    
        cv2.setMouseCallback("image", click_and_crop)
        
        #keep pressing 'c' until the 4 rectangles are drawn
        while True:
            cv2.imshow("image", image)
        
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()
                # if the 'space' key is pressed, a rectangle was captured and you break from the loop
            elif key == ord("c"):
                break
        
        num_roi  = num_roi + 1
    
    cv2.destroyAllWindows()
    

    
#    for r in range(0,num_roi):
#        
#        #are these offset somehow???
#        x1 = np.minimum(refPt[r][0],refPt[r+1][0]); x2 = np.maximum(refPt[r][0],refPt[r+1][0])
#        y1 = np.minimum(refPt[r][1],refPt[r+1][1]); y2 = np.maximum(refPt[r][1],refPt[r+1][1])
#        
#        
#        ROI[y1:y2,x1:x2] = 255 #the order of these are important; (0,0) at top left
#        boundboxes[r] = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
#        masked_img = cv2.bitwise_and(image,image,mask = ROI)
#        masked_grays[r] = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
#        #cv2.imshow('mask', masked_img)
#        #key = cv2.waitKey(2000)
#        #cv2.destroyAllWindows()

    return [masked_grays, boundboxes]

#if __name__ == "__main__":   