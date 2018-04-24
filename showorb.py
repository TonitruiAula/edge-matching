# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math



if __name__ == '__main__':
    imgName = sys.argv[1]
    lr = int(sys.argv[2])
    count = int(sys.argv[3])
    if lr == 0:
    	imgpath = 'images/' + imgName + '/im0.png'
    else:
    	imgpath = 'images/' + imgName + '/im1.png'

    img = cv2.imread(imgpath)

    rst = np.zeros(img.shape,img.dtype)
    orb = cv2.ORB_create(nfeatures = count)
    if len(sys.argv) > 4:
        orb.setFastThreshold(int(sys.argv[4]))
    kp,des = orb.detectAndCompute(img,None) 
    print len(kp)
    for p in kp:
        x = int(round(p.pt[0]))
        y = int(round(p.pt[1]))		
        cv2.circle(img,(x,y),1,(0,255,0))   

    cv2.imshow('orb',img)
    cv2.waitKey(0)

