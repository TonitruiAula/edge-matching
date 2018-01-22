# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math

def sob(img):
    sobelX = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_8UC1, 0, 1)
    sobAbs = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    # sobAbs = sobAbs * (1.0/255)
    sobDir = np.zeros(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    for i in xrange(height):
        for j in xrange(width):
            sobDir[i,j] = math.atan2(math.fabs(sobelY[i,j]), math.fabs(sobelX[i,j])) / (math.pi / 2) * 90
    # flag, dst = cv2.threshold(sob, t_bin, 255, cv2.THRESH_OTSU)
    return [sobAbs, sobDir]

def getEdge1(imgL, imgR):
    assert(imgL.shape == imgR.shape)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    edgeLa, edgeLd = sob(grayL)
    edgeRa, edgeRd = sob(grayR)
    return [edgeLa, edgeLd, edgeRa, edgeRd]

def findCores(grayL, grayR, edge, i, j, kL, seg_len, t_dir, t_abs, h_size=2):
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 匹配梯度方向
    kRs = []
    for k in xrange(0,seg_len):
        if math.fabs(edgeLd[i,j+kL] - edgeRd[i,j+k]) < t_dir:
            kRs.append(k)

    # 匹配梯度大小
        # 窗口归一
    if len(kRs) == 0:
        return -1
    gvecL = []
    ggrayL = []
            
    for _i in xrange(i-h_size,i+h_size+1):
        for _j in xrange(j+kL-h_size,j+kL+h_size+1):
            gvecL.append(edgeLa[_i,_j])
            ggrayL.append(grayL[i,j])

    for k in kRs:
        gvecR = []
        for _i in xrange(i-h_size,i+h_size+1):
            for _j in xrange(j+k-h_size,j+k+h_size+1):
                gvecR.append(edgeRa[_i,_j])
        comp = np.array([gvecL,gvecR])
        coef = np.corrcoef(comp)[0,1]
        if coef < t_abs:
            kRs.remove(k)
            
    if len(kRs) == 0:
        return -1

    # 匹配灰度
    max_coef = -1
    kR = -1
    for k in kRs:
        ggrayR = []
        for _i in xrange(i-h_size,i+h_size+1):
            for _j in xrange(j+k-h_size,j+k+h_size+1):
                ggrayR.append(grayR[_i,_j])
        comp = np.array([ggrayL,ggrayR])
        coef = np.corrcoef(comp)[0,1]
        if coef > max_coef:
            kR = k
    return kR
 


def match1(imgL, imgR, edge, seg_len, t_dir, t_abs, win_size=5):
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    assert(imgL.shape == imgR.shape)
    height = imgL.shape[0]
    width = imgL.shape[1]
    h_size = int(win_size)/2
    disp = np.zeros(imgL.shape)
    for i in xrange(win_size,height-win_size):
        for j in xrange(win_size,width-win_size-seg_len,seg_len):
            maxGa = edgeLa[i,j]
            kL = 0
            
            # 找到梯度最大点
            for k in xrange(1,seg_len):
                # if j+k >= width-win_size:
                #     break
                if edgeLa[i,j+k] > edgeLa[i,j+kL]:
                    maxGa = edgeLa[i,j+k]
                    kL = k
            kR = findCores(grayL, grayR, edge, i, j, kL, seg_len, t_dir, t_abs, h_size)
            ckL = findCores(grayR, grayL, edge, i, j, kR, seg_len, t_dir, t_abs, h_size)
            if kL == ckL:
                disp[i,j+k] = math.fabs(kR-kL)
    
    maxDisp = disp.max()
    disp = disp / float(maxDisp) * 255.0
    disp.astype('uint8')
    return maxDisp, disp

    

if __name__ == '__main__':

    imgpathL = sys.argv[1]
    imgpathR = sys.argv[2]
    seg_len = int(sys.argv[3])
    t_dir = float(sys.argv[4])
    t_abs = float(sys.argv[5])

    # imgpathL = 'imL.png'
    # imgpathR = 'imR.png'
    # seg_len = 10
    # t_dir = 0.5
    # t_abs = 0.5


    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)

    edge = getEdge1(imgL,imgR)

    cv2.imshow('left grad abs',edge[0])
    cv2.imshow('left grad dir',edge[1])
    cv2.imshow('right grad abs',edge[2])
    cv2.imshow('right grad dir',edge[3])

    maxDisp, disp = match1(imgL, imgR, edge, seg_len, t_dir, t_abs)
    print 'max disparity: ' , maxDisp
    cv2.imshow('disp',disp)
    cv2.waitKey(0)
