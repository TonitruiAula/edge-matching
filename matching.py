# -*- coding :utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math

def sob(img, t_bin):
    sobelX = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_8UC1, 0, 1)
    sobAbs = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    sobDir = np.zeros(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    for i in xrange(height):
        for j in xrange(width):
            sobDir[i,j] = math.atan2(math.fabs(sobelY[i,j]), math.fabs(sobelX[i,j])) / (math.pi / 2)
    # flag, dst = cv2.threshold(sob, t_bin, 255, cv2.THRESH_OTSU)
    return [sobAbs, sobDir]

def getEdge(imgpathL, imgpathR, t_bin):
    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    edgeLa, edgeLd = sob(grayL, t_bin)
    edgeRa, edgeRd = sob(grayR, t_bin)
    return [edgeLa, edgeLd, edgeRa, edgeRd]



def match1(imgpathL, imgpathR, t_bin, t_dir, t_abs, t_gray):
    edgeLa, edgeLd, edgeRa, edgeRd = getEdge(imgpathL, imgpathR, t_bin)
    cv2.imshow('left', edgeLa)
    cv2.imshow('right', edgeRa)
    cv2.imshow('leftD', edgeLd)
    np.savetxt('ld.txt',edgeLd,'%.3f')
    np.savetxt('la.txt',edgeLa,'%d')    
    cv2.imshow('rightD', edgeRd)


if __name__ == '__main__':
    imgpathL = sys.argv[1]
    imgpathR = sys.argv[2]
    t_bin = float(sys.argv[3])
    match1(imgpathL,imgpathR,t_bin,0,0,0)
    cv2.waitKey(0)
