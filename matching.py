# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math

def norm(list):
    a = np.array(list)
    length = math.sqrt((a**2).sum())
    if length > 0:
        a = a / length
    return a

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
    # np.savetxt('la.txt',edgeLa,'%d')
    # np.savetxt('ld.txt',edgeLd,'%.1f')
    # np.savetxt('ra.txt',edgeRa,'%d')
    # np.savetxt('rd.txt',edgeRd,'%.1f')
    return [edgeLa, edgeLd, edgeRa, edgeRd]

# 找到对应的匹配点
def getCores(grayL, grayR, index, x, seg, edge, threshold, h_size):
    if x == -1:
        return -1
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    t_abs, t_dir, t_coeff = threshold

    # 梯度方向差低于某一阈值则加入候选匹配点
    allX = []
    for xR in xrange(seg[0], seg[1]):
        if math.fabs(edgeLd[index, x] - edgeRd[index, xR]) < t_dir:
            allX.append(xR)
    
    if len(allX) == 0:
        return -1

    gvecL = []
    ggrayL = []

    # 梯度大小相关计算，若相关系数大于阈值则保留
    for _i in xrange(index - h_size, index + h_size + 1):
        for _j in xrange(x - h_size, x + h_size + 1):
            gvecL.append(edgeLa[_i, _j])
            ggrayL.append(grayL[_i, _j])

    allX2 = []

    for xR in allX:
        gvecR = []
        for _i in xrange(index - h_size, index + h_size + 1):
            for _j in xrange(xR - h_size, xR + h_size + 1):
                gvecR.append(edgeRa[_i, _j])
        gvecL = norm(gvecL)
        gvecR = norm(gvecR)
        comp = np.array([gvecL, gvecR])
        coef = np.corrcoef(comp)[0, 1]
        if coef > t_coeff:
            allX2.append(xR)
        # if coef < t_coeff:
        #     allX.remove(xR)

    if len(allX2) == 0:
        return -1

    # 匹配灰度相关计算，取相关系数最大的作为结果
    max_coef = -1
    xRst = -1
    for xR in allX2:
        ggrayR = []
        for _i in xrange(index - h_size, index + h_size + 1):
            for _j in xrange(xR - h_size, xR + h_size + 1):
                ggrayR.append(grayR[_i, _j])
        comp = np.array([ggrayL, ggrayR])
        coef = np.corrcoef(comp)[0, 1]
        if coef > max_coef:
            xRst = xR
            max_coef = coef
    return xRst


# 按行匹配获取视差
def matchline(grayL, grayR, index, seg, edge, threshold, disp, ndisp, h_size):
    # 如果区间长度不够长则返回
    if seg[1] - seg[0] <= h_size:
        return
    # 边缘数据
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 反向边缘数据
    edgeR = [edgeRa, edgeRd, edgeLa, edgeLd]
    # 获取阈值
    t_abs, t_dir, t_coeff = threshold
    # 获取区间内的最大梯度点
    maxEdge = -1
    x = -1
    for i in xrange(seg[0], seg[1]):
        if edgeLa[index, i] > maxEdge:
            maxEdge = edgeLa[index, i]
            x = i
    if x == -1:
        return
    if maxEdge < t_abs:
        return
    # 找到对应的右图点
    xR = getCores(grayL, grayR, index, x, seg, edge, threshold, h_size)
    # 反向找到对应的左图点
    xL = getCores(grayR, grayL, index, xR, seg, edgeR, threshold, h_size)
    # 如果匹配则计算视差
    if x == xL and math.fabs(xL - xR) < ndisp:
        disp[index,x] = math.fabs(xL - xR)
    # 匹配左右子区间视差
    matchline(grayL, grayR, index, [seg[0],x-h_size], edge, threshold, disp, ndisp, h_size)
    matchline(grayL, grayR, index, [x+h_size,seg[1]], edge, threshold, disp, ndisp, h_size)

    
def match(imgL, imgR, edge, threshold, ndisp, h_size=2):
    # 计算灰度图
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # 获取形状
    assert(imgL.shape == imgR.shape)
    height = imgL.shape[0]
    width = imgL.shape[1]
    # 初始化视差图
    disp = np.zeros((height,width))
    count = 0   #非零视差点个数
    for i in xrange(h_size, height-h_size):
        # 逐行匹配获取视差
        matchline(grayL, grayR, i, [h_size, width-h_size], edge, threshold, disp, ndisp, h_size)
        for j in xrange(width):
            if disp[i,j] > 0:
                count += 1
    ratio = float(count) / float(height*width)
    print 'count : ', count
    print ('ratio : %.6f' %(ratio))


    maxDisp = disp.max()
    # disp = disp / float(maxDisp)# * 255.0
    # disp.astype('uint8')
    return maxDisp, disp





