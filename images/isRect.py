# -*- coding=utf-8 -*-

import sys
import cv2
import numpy as np

def check(picName):
    im0 = cv2.imread(picName + '/im0.png')
    im1 = cv2.imread(picName + '/im1.png')
    rst = np.array([])
    if len(im0.shape) == 3:
        rst = np.zeros([im0.shape[0],im0.shape[1]+im1.shape[1],im0.shape[2]],im0.dtype)
    else:
        rst = np.zeros([im0.shape[0],im0.shape[1]+im1.shape[1]],im0.dtype)
    rst[:,0:im0.shape[1]] = im0
    rst[:,im0.shape[1]:] = im1
    for x in xrange(0,rst.shape[0],16):
        cv2.line(rst,(0,x),(rst.shape[1]-1,x),(0,255,0),1)
    cv2.imshow('rst',rst)
    cv2.waitKey(0)

if __name__ == '__main__':
    picName = sys.argv[1]
    check(picName)