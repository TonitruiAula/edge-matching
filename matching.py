# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
import lbp

def norm_scale(list):
    a = np.array(list)
    length = math.sqrt((a**2).sum())
    if length > 0:
        a = a / length
    return a

def norm_01(list):
    a = np.array(list)
    Max = a.max()
    Min = a.min()
    if Max-Min > 0:
        a = (a-Min) / (Max-Min)
    return a
def norm_z(list):
    a = np.array(list)
    mu = np.average(a)
    sigma = np.std(a)
    a = (a - mu) / sigma
    return a

# 利用Sobel算子计算图片的梯度大小和方向
def sob(img):
    sobelX = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    sobelX = np.fabs(sobelX)
    sobelY = np.fabs(sobelY)

    sobAbs = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

    sobDir = cv2.phase(sobelX,sobelY,angleInDegrees=True)
    # sobDir = np.zeros(img.shape)
    # height = img.shape[0]
    # width = img.shape[1]
    # for i in xrange(height):
    #     for j in xrange(width):
    #         sobDir[i,j] = math.atan2(math.fabs(sobelY[i,j]), math.fabs(sobelX[i,j])) / (math.pi / 2.0) * 90.0

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
def getCores(grayL, grayR, index, x, seg, edge, lbpData, threshold, h_size):
    if x == -1:
        return -1
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    t_abs, t_dir, t_coeff = threshold
    lbpDataL, lbpDataR = lbpData

    # 如果左图的梯度大小过小则跳过
    if edgeLa[index,x] < t_abs:
        return -1

    # if edgeLd[index,x] == 0.0 or edgeLd[index,x] == 90.0:
    #     return -1

    # LBP特征值相同则加入候选匹配点
    # allX = []
    # for xR in xrange(seg[0], seg[1]):
    #     a = lbpDataL[index,x]
    #     b = lbpDataR[index,xR]
    #     if lbp.diffLBP(lbpDataL[index,x], lbpDataR[index,xR]) < 4:
    #         allX.append(xR)
    #
    # if len(allX) == 0:
    #     return -1

    # 梯度方向差低于某一阈值则加入候选匹配点
    allX2 = []
    for xR in xrange(seg[0], seg[1]):
    # for xR in allX:
        if math.fabs(edgeLd[index, x] - edgeRd[index, xR]) < t_dir:
            allX2.append(xR)
    
    if len(allX2) == 0:
        return -1

    gvecL = []
    ggrayL = []

    # 梯度大小相关计算，若相关系数大于阈值则保留
    for _i in xrange(index - h_size, index + h_size + 1):
        for _j in xrange(x - h_size, x + h_size + 1):
            gvecL.append(edgeLa[_i, _j])
            ggrayL.append(grayL[_i, _j])

    allX3 = []

    for xR in allX2:
        gvecR = []
        for _i in xrange(index - h_size, index + h_size + 1):
            for _j in xrange(xR - h_size, xR + h_size + 1):
                gvecR.append(edgeRa[_i, _j])
        gvecL = norm_scale(gvecL)
        gvecR = norm_scale(gvecR)
        # gvecL = norm_01(gvecL)
        # gvecR = norm_01(gvecR)
        # gvecL = norm_z(gvecL)
        # gvecR = norm_z(gvecR)
        comp = np.array([gvecL, gvecR])
        coef = np.corrcoef(comp)[0, 1]
        if coef > t_coeff:
            allX3.append(xR)
        # if coef < t_coeff:
        #     allX.remove(xR)

    if len(allX3) == 0:
        return -1

    # 匹配灰度相关计算，取相关系数最大的作为结果
    max_coef = -1
    xRst = -1
    for xR in allX3:
        # 如果右图的梯度大小过小则跳过
        if edgeRa[index,xR] < t_abs:
            continue

        # if edgeRd[index,xR] == 0.0 or edgeRd[index,xR] == 90.0:
        #     continue

        ggrayR = []
        for _i in xrange(index - h_size, index + h_size + 1):
            for _j in xrange(xR - h_size, xR + h_size + 1):
                ggrayR.append(grayR[_i, _j])
        comp = np.array([ggrayL, ggrayR])
        coef = np.corrcoef(comp)[0, 1] #+ 1.0 / (1+math.fabs(float(grayL[index,x])-float(grayR[index,xR]))/255.0)
        if coef > max_coef:
            xRst = xR
            max_coef = coef


    # gvecRst = []
    # ggrayRst = []
    # for _i in xrange(index - h_size, index + h_size + 1):
    #     for _j in xrange(xRst - h_size, xRst + h_size + 1):
    #         gvecRst.append(edgeRa[_i, _j])
    #         ggrayRst.append(grayR[_i,_j])
    #
    # offsets = [-1,0,1]
    # max_coef = -1
    # offset = -2
    #
    #
    # for di in offsets:
    #     gvecLd = []
    #     ggrayLd = []
    #     for _i in xrange(index - h_size, index + h_size + 1):
    #         for _j in xrange(x+di - h_size, x+di + h_size + 1):
    #             if x+di >= seg[0] and x+di <= seg[1]:
    #                 gvecLd.append(edgeLa[_i, _j])
    #                 ggrayLd.append(grayL[_i, _j])
    #     comp1 = np.array([ggrayLd, ggrayRst])
    #     comp2 = np.array([gvecLd, gvecRst])
    #     coef1 = np.corrcoef(comp1)[0, 1]
    #     coef2 = np.corrcoef(comp2)[0, 1]
    #     coef = coef1 + coef2
    #     if coef > max_coef:
    #         max_coef = coef
    #         offset = di

    if edgeRa[index,xRst] < t_abs:
        return -1

    return xRst#, offset


# 按行匹配获取视差
def matchline(grayL, grayR, index, seg, edge, lbpData, threshold, disp, ndisp, h_size):
    # 如果区间长度不够长则返回
    seglen = seg[1] - seg[0]
    if seglen <= h_size:
        return
    # 边缘数据
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 反向边缘数据
    edgeR = [edgeRa, edgeRd, edgeLa, edgeLd]
    lbpDataR = [lbpData[1],lbpData[0]]
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
    xR = getCores(grayL, grayR, index, x, seg, edge, lbpData, threshold, h_size)
    # 反向找到对应的左图点
    # newseg = [seg[0]-seglen, seg[1]+seglen]
    # if newseg[0] < h_size:
    #     newseg[0] = h_size
    # if newseg[1] > grayL.shape[1]-h_size:
    #     newseg[1] = grayL.shape[1]-h_size
    xL = getCores(grayR, grayL, index, xR, [h_size, grayL.shape[1]-h_size], edgeR, lbpDataR, threshold, h_size)

    # if xL > 0 and xR > 0 and math.fabs(xL - xR) < ndisp:
    #     disp[index,xL] = math.fabs(xL - xR)

    # 如果匹配则计算视差
    if x == xL and math.fabs(xL - xR) < ndisp:
        disp[index,x] = math.fabs(xL - xR)
    # 匹配左右子区间视差
    matchline(grayL, grayR, index, [seg[0],x], edge, lbpData, threshold, disp, ndisp, h_size)
    matchline(grayL, grayR, index, [x+1,seg[1]], edge, lbpData, threshold, disp, ndisp, h_size)
    # matchline(grayL, grayR, index, [seg[0],x-h_size], edge, lbpData, threshold, disp, ndisp, h_size)
    # matchline(grayL, grayR, index, [x+h_size,seg[1]], edge, lbpData, threshold, disp, ndisp, h_size)

    
def match(imgL, imgR, threshold, ndisp=64, h_size=2):
    # 计算灰度图
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # 获取形状
    assert(imgL.shape == imgR.shape)
    height = imgL.shape[0]
    width = imgL.shape[1]

    edge = getEdge1(imgL,imgR)
    lbpData = [lbp.getLBP(grayL),lbp.getLBP(grayR)]

    # 初始化视差图
    disp = np.zeros((height,width))
    count = 0   #非零视差点个数
    for i in xrange(h_size, height-h_size):
        # 逐行匹配获取视差
        matchline(grayL, grayR, i, [h_size, width-h_size], edge, lbpData, threshold, disp, ndisp, h_size)
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





