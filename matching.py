# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math

# 利用Sobel算子计算图片的梯度大小和方向
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
            sobDir[i,j] = math.atan2(math.fabs(sobelY[i,j]), math.fabs(sobelX[i,j])) / (math.pi / 2.0) * 90.0
    # flag, dst = cv2.threshold(sob, t_bin, 255, cv2.THRESH_OTSU)
    return [sobAbs, sobDir]

# 获取左右边缘
def getEdge1(imgL, imgR):
    # assert(imgL.shape == imgR.shape)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    edgeLa, edgeLd = sob(grayL)
    edgeRa, edgeRd = sob(grayR)
    np.savetxt('la.txt',edgeLa,'%d')
    np.savetxt('ld.txt',edgeLd,'%.1f')
    np.savetxt('ra.txt',edgeRa,'%d')
    np.savetxt('rd.txt',edgeRd,'%.1f')
    return [edgeLa, edgeLd, edgeRa, edgeRd]

# 找到左视图对应的匹配点
def findCores(grayL, grayR, edge, i, j, kL, seg_len, t_dir, t_coeff, h_size=2):
    edgeLa, edgeLd, edgeRa, edgeRd = edge

    # 匹配梯度方向
    kRs = []
    for k in xrange(0, seg_len):
        a = edgeLd[i, j + kL]
        b = edgeRd[i, j + k]
        if math.fabs(edgeLd[i, j + kL] - edgeRd[i, j + k]) < t_dir:
            kRs.append(k)

    # 匹配梯度大小
    if len(kRs) == 0:
        return -1
    gvecL = []
    ggrayL = []

    for _i in xrange(i - h_size, i + h_size + 1):
        for _j in xrange(j + kL - h_size, j + kL + h_size + 1):
            gvecL.append(edgeLa[_i, _j])
            ggrayL.append(grayL[_i, _j])

    for k in kRs:
        gvecR = []
        for _i in xrange(i - h_size, i + h_size + 1):
            for _j in xrange(j + k - h_size, j + k + h_size + 1):
                gvecR.append(edgeRa[_i, _j])
        comp = np.array([gvecL, gvecR])
        coef = np.corrcoef(comp)[0, 1]
        if coef < t_coeff:
            kRs.remove(k)

    if len(kRs) == 0:
        return -1

    # 匹配灰度
    max_coef = -1
    kR = -1
    for k in kRs:
        ggrayR = []
        for _i in xrange(i - h_size, i + h_size + 1):
            for _j in xrange(j + k - h_size, j + k + h_size + 1):
                ggrayR.append(grayR[_i, _j])
        comp = np.array([ggrayL, ggrayR])
        coef = np.corrcoef(comp)[0, 1]
        if coef > max_coef:
            kR = k
            max_coef = coef
    return kR

    # 匹配梯度方向
    # xRs = []
    # width = grayL.shape[1]
    # for x in xrange(h_size,width-h_size):
    #     a = edgeLd[i,j+kL]
    #     b = edgeRd[i,x]
    #     if math.fabs(edgeLd[i,j+kL] - edgeRd[i,x]) < t_dir:
    #         xRs.append(x)
    #
    # # 匹配梯度大小
    # if len(xRs) == 0:
    #     return -1
    # gvecL = []
    # ggrayL = []
    #
    # for _i in xrange(i-h_size,i+h_size+1):
    #     for _j in xrange(j+kL-h_size,j+kL+h_size+1):
    #         gvecL.append(edgeLa[_i,_j])
    #         ggrayL.append(grayL[_i,_j])
    #
    # for x in xRs:
    #     gvecR = []
    #     for _i in xrange(i-h_size,i+h_size+1):
    #         for _j in xrange(x-h_size,x+h_size+1):
    #             gvecR.append(edgeRa[_i,_j])
    #     comp = np.array([gvecL,gvecR])
    #     coef = np.corrcoef(comp)[0,1]
    #     if coef < t_abs:
    #         xRs.remove(x)
    #
    # if len(xRs) == 0:
    #     return -1
    #
    # # 匹配灰度
    # max_coef = -1
    # xR = -1
    # for x in xRs:
    #     ggrayR = []
    #     for _i in xrange(i-h_size,i+h_size+1):
    #         for _j in xrange(x-h_size,x+h_size+1):
    #             ggrayR.append(grayR[_i,_j])
    #     comp = np.array([ggrayL,ggrayR])
    #     coef = np.corrcoef(comp)[0,1]
    #     if coef > max_coef:
    #         xR = x
    # return xR
 


def match1(imgL, imgR, edge, seg_len, t_abs, t_dir, t_coeff, win_size=5):
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    assert(imgL.shape == imgR.shape)
    height = imgL.shape[0]
    width = imgL.shape[1]
    h_size = int(win_size)/2
    disp = np.zeros((height,width))
    count = 0
    for i in xrange(win_size,height-win_size):
        for j in xrange(win_size,width-win_size-seg_len,seg_len):

            allK = []
            
            # 找到梯度最大点
            for k in xrange(1,seg_len):
                if edgeLa[i,j+k] > t_abs:
                    allK.append(k)


            while len(allK) > 0:
                maxGa = -1
                kL = -1
                for k in allK:
                    if edgeLa[i,j+k] > edgeLa[i,j+kL]:
                        maxGa = edgeLa[i,j+k]
                        kL = k

                kR = findCores(grayL, grayR, edge, i, j, kL, seg_len, t_dir, t_coeff, h_size)
                if kR == -1:
                    allK.remove(k)
                    continue
                ckL = findCores(grayR, grayL, edge, i, j, kR, seg_len, t_dir, t_coeff, h_size)
                if kL == ckL:
                    disp[i,j+k] = math.fabs(kR-kL)
                    if disp[i,j+k] > 0:
                        count += 1
                        print '(',i,' , ',j+k,' , ',disp[i,j+k],')'

                # xR = findCores(grayL, grayR, edge, i, j, kL, seg_len, t_dir, t_abs, h_size)
                # if xR == -1:
                #     continue
                # cxL = findCores(grayR, grayL, edge, i, j, xR-j, seg_len, t_dir, t_abs, h_size)
                # if kL+j == cxL:
                #     disp[i,j+k] = math.fabs(xR-cxL)
                #     print '(',i,' , ',j+k,' , ',disp[i,j+k],')'

                allK.remove(k)

    ratio = float(count) / float(height*width)
    print 'count : ', count
    print ('ratio : %.6f' %(ratio))


    maxDisp = disp.max()
    # disp = disp / float(maxDisp)# * 255.0
    # disp.astype('uint8')
    return maxDisp, disp

    

if __name__ == '__main__':

    print os.path.abspath(os.curdir)
    print 'starting...'

    imgpathL = sys.argv[1]
    imgpathR = sys.argv[2]
    seg_len = int(sys.argv[3])
    t_abs = float(sys.argv[4])
    t_dir = float(sys.argv[5])
    t_coeff = float(sys.argv[6])

    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)

    print imgL.shape

    edge = getEdge1(imgL,imgR)

    ld = edge[1]/90.0
    ld.astype('uint8')
    rd = edge[3]/90.0
    rd.astype('uint8')
    cv2.imshow('left grad abs',edge[0])
    cv2.imshow('left grad dir',ld)
    # cv2.imshow('right grad abs',edge[2])
    # cv2.imshow('right grad dir',rd)

    maxDisp, disp = match1(imgL, imgR, edge, seg_len, t_abs, t_dir, t_coeff)
    # disp /= float(maxDisp)
    disp /= 255.0
    disp.astype('uint8')
    print 'max disparity: ' , maxDisp
    cv2.imshow('disp',disp)
    cv2.imwrite('disp.png',disp)
    # np.savetxt('disp.txt',disp,'%.2f')

    cv2.waitKey(0)
