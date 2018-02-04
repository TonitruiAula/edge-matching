# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math

def draw(arg):
    index = cv2.getTrackbarPos('index',img)
    canvas = rst.copy()
    width = imgL.shape[1]
    y = int(badlog[index,0])
    xL = int(badlog[index,1])
    xRe = int(xL - badlog[index,2]) + width
    xRg = int(xL - badlog[index,3]) + width
    cv2.circle(canvas,(xL,y),3,(255,0,0))
    cv2.circle(canvas,(xRe,y),3,(0,0,255))
    cv2.circle(canvas,(xRg,y),3,(0,255,0))
    cv2.imshow(img,canvas)


if __name__ == '__main__':
    img = sys.argv[1]
    imgpathL = 'images/' + img + '/im0.png'
    imgpathR = 'images/' + img + '/im1.png'

    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)

    badlog = np.loadtxt('images/' + img + '/badlog.txt')
    count = badlog.shape[0]
    rst = np.zeros([imgL.shape[0],imgL.shape[1]+imgR.shape[1],imgL.shape[2]],imgL.dtype)
    width = imgL.shape[1]
    rst[:,0:width] = imgL
    rst[:,width:] = imgR
    cv2.namedWindow(img)
    cv2.createTrackbar('index',img,0,count-1,draw)
    draw(0)
    cv2.waitKey(0)


