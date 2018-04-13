# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
import operator
from matching import areaCoeff

def draw(arg):
    index = cv2.getTrackbarPos('index',img)
    canvas = rst.copy()
    width = imgL.shape[1]
    y = int(points[index,0])
    xL = int(points[index,1])
    xRe = int(xL - points[index,2]) + width
    xRg = int(xL - points[index,3]) + width
    cv2.circle(canvas,(xL,y),4,(255,0,0))
    cv2.circle(canvas,(xRe,y),4,(0,0,255))
    cv2.circle(canvas,(xRg,y),4,(0,255,0))
    coeffE = areaCoeff(grayL,grayR,2,y,xL,y,xRe-width)
    coeffGT = areaCoeff(grayL,grayR,2,y,xL,y,xRg-width)
    print 'x=%d y=%d perr=%.2f derr=%.2f ratio=%.6f coeffE=%.6f coeffGT=%.6f' % (points[index,1],points[index,0],points[index,4],points[index,7],points[index,8],coeffE,coeffGT)
    cv2.imshow(img,canvas)


if __name__ == '__main__':
    img = sys.argv[1]
    imgpathL = 'images/' + img + '/im0.png'
    imgpathR = 'images/' + img + '/im1.png'

    type = sys.argv[2]

    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


    log = np.loadtxt('images/' + img + '/log.txt')
    loglist = log.tolist()

    if type == '-p':
        loglist.sort(key=operator.itemgetter(4),reverse=True)
    elif type == '-d':
        loglist.sort(key=operator.itemgetter(7),reverse=True)
    elif type == '-r':
        loglist.sort(key=operator.itemgetter(8),reverse=True)
    
    points = np.array(loglist)
    count = log.shape[0]
    rst = np.zeros([imgL.shape[0],imgL.shape[1]+imgR.shape[1],imgL.shape[2]],imgL.dtype)
    width = imgL.shape[1]
    rst[:,0:width] = imgL
    rst[:,width:] = imgR
    cv2.namedWindow(img)
    cv2.createTrackbar('index',img,0,count-1,draw)
    draw(0)
    cv2.waitKey(0)


