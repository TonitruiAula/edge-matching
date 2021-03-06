# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
import lbp
import operator
from preprocess import *
import gms_matcher
import olm

# 计算区域核的相关系数
def areaCoeff(array1,array2,rad,i1,j1,i2,j2):
    if i1-rad < 0 or i1+rad >= array1.shape[0] or i2-rad < 0 or i2+rad >= array2.shape[0] \
        or j1-rad < 0 or j1+rad >= array1.shape[1] or j2-rad < 0 or j2+rad >= array2.shape[1]:
        return -float('inf')
    area1 = array1[i1-rad:i1+rad+1,j1-rad:j1+rad+1]
    area1 = np.reshape(area1,(2*rad+1)**2)
    area2 = array2[i2-rad:i2+rad+1,j2-rad:j2+rad+1]
    area2 = np.reshape(area2,(2*rad+1)**2)
    comp = np.zeros((2,(2*rad+1)**2))
    comp[0,:] = area1
    comp[1,:] = area2
    coef = np.corrcoef(comp)[0, 1]
    return coef

# 三个归一化函数（并没有什么用）
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
# def getCores(grayL, grayR, index, x, seg, edge, lbpData, threshold, h_size):
def getCores(grayL, grayR, index, x, seg, edge, threshold, h_size):
    if x == -1:
        return -1
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    t_abs, t_dir, t_coeff = threshold
    # lbpDataL, lbpDataR = lbpData

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

    # 梯度大小相关计算，若相关系数大于阈值则保留

    allX3 = []

    for xR in allX2:
        coef = areaCoeff(edgeLa,edgeRa,h_size,index,x,index,xR)
        if coef > t_coeff:
            allX3.append(xR)

    if len(allX3) == 0:
        return -1

    # 匹配灰度相关计算，取相关系数最大的作为结果
    # max_coef = -1
    xRst = -1
    grayCoef = []
    for xR in allX3:
        # 如果右图的梯度大小过小则跳过(可能删去正确的点？)
        # if edgeRa[index,xR] < t_abs:
        #     continue

        # if edgeRd[index,xR] == 0.0 or edgeRd[index,xR] == 90.0:
        #     continue
        coef = areaCoeff(grayL,grayR,h_size,index,x,index,xR)
        grayCoef.append([xR,coef])
        # if coef > max_coef:
        #     xRst = xR
        #     max_coef = coef
    if len(grayCoef) == 0:
        return -1
    grayCoef.sort(key=operator.itemgetter(1),reverse = True)
    # if len(grayCoef) > 1 and grayCoef[0][1] - grayCoef[1][1] < 0.1:
    #     return -1
    xRst = grayCoef[0][0]
    # (最大的coeff可能不够大)
    if grayCoef[0][1] < t_coeff:
        return -1
    if edgeRa[index,xRst] < t_abs:
        return -1

    return xRst


# 按行匹配获取视差（递归）
# def matchline(grayL, grayR, index, seg, edge, lbpData, threshold, disp, ndisp, h_size):
def matchline(grayL, grayR, index, seg, edge, threshold, disp, ndisp, h_size):
    # 如果区间长度不够长则返回
    seglen = seg[1] - seg[0]
    if seglen <= h_size:
        return
    # 边缘数据
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 反向边缘数据
    edgeR = [edgeRa, edgeRd, edgeLa, edgeLd]
    # lbpDataR = [lbpData[1],lbpData[0]]
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
    # xR = getCores(grayL,
    #  grayR, index, x, seg, edge, lbpData, threshold, h_size)
    xR = getCores(grayL, grayR, index, x, seg, edge, threshold, h_size)
    
    # 反向找到对应的左图点
    # newseg = [seg[0]-seglen, seg[1]+seglen]
    # if newseg[0] < h_size:
    #     newseg[0] = h_size
    # if newseg[1] > grayL.shape[1]-h_size:
    #     newseg[1] = grayL.shape[1]-h_size
    newseg = [h_size, grayL.shape[1]-h_size]
    # xL = getCores(grayR, grayL, index, xR, newseg, edgeR, lbpDataR, threshold, h_size)
    xL = getCores(grayR, grayL, index, xR, newseg, edgeR, threshold, h_size)

    # if xL > 0 and xR > 0 and math.fabs(xL - xR) < ndisp:
    #     disp[index,xL] = math.fabs(xL - xR)

    # 如果匹配则计算视差
    if x == xL and float(xL - xR) < ndisp and float(xL - xR) > 0:
        disp[index,x] = float(xL - xR)
    # 匹配左右子区间视差
    # matchline(grayL, grayR, index, [seg[0],x], edge, lbpData, threshold, disp, ndisp, h_size)
    # matchline(grayL, grayR, index, [x+1,seg[1]], edge, lbpData, threshold, disp, ndisp, h_size)
    matchline(grayL, grayR, index, [seg[0],x], edge, threshold, disp, ndisp, h_size)
    matchline(grayL, grayR, index, [x+1,seg[1]], edge, threshold, disp, ndisp, h_size)


    # matchline(grayL, grayR, index, [seg[0],x-h_size], edge, lbpData, threshold, disp, ndisp, h_size)
    # matchline(grayL, grayR, index, [x+h_size,seg[1]], edge, lbpData, threshold, disp, ndisp, h_size)


# 按行匹配获取视差（全行区间）
# def matchline2(grayL, grayR, index, seg, edge, lbpData, threshold, disp, ndisp, h_size):
def matchline2(grayL, grayR, index, seg, edge, threshold, disp, ndisp, h_size):
    # 如果区间长度不够长则返回
    seglen = seg[1] - seg[0]
    if seglen <= h_size:
        return
    # 边缘数据
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 反向边缘数据
    edgeR = [edgeRa, edgeRd, edgeLa, edgeLd]
    # lbpDataR = [lbpData[1],lbpData[0]]
    # 获取阈值
    t_abs, t_dir, t_coeff = threshold
    # 获取区间内的最大梯度点
    points = []
    for i in xrange(seg[0], seg[1]):
        if edgeLa[index, i] > t_abs:
            points.append([i,edgeLa[index, i]])
    if len(points) == 0:
        return
    points.sort(key=operator.itemgetter(1))
    coresXL = {}
    while len(points) > 0:
        curpoint = points.pop()
        x = curpoint[0]
        # xR = getCores(grayL, grayR, index, x, seg, edge, lbpData, threshold, h_size)
        xR = getCores(grayL, grayR, index, x, seg, edge, threshold, h_size)

        if xR == -1:
            continue

        if coresXL.has_key(xR) == False:
            # xL = getCores(grayR, grayL, index, xR, seg, edgeR, lbpDataR, threshold, h_size)
            xL = getCores(grayR, grayL, index, xR, seg, edgeR, threshold, h_size)
            coresXL[xR] = xL
        else:
            xL = coresXL[xR]
        
        # 如果匹配则计算视差
        if x == xL and float(xL - xR) < ndisp and float(xL - xR) > 0:
            disp[index,x] = float(xL - xR)

def matchline3(grayL, grayR, index, seg, edge, threshold, disp, ndisp, h_size):
    # 如果区间长度不够长则返回
    seglen = seg[1] - seg[0]
    if seglen <= h_size:
        return
    # 边缘数据
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 反向边缘数据
    edgeR = [edgeRa, edgeRd, edgeLa, edgeLd]
    # lbpDataR = [lbpData[1],lbpData[0]]
    # 获取阈值
    t_abs, t_dir, t_coeff = threshold
    # 获取区间内的最大梯度点
    points = []
    for i in xrange(seg[0], seg[1]):
        if edgeLa[index, i] > t_abs:
            points.append([i,edgeLa[index, i]])
    if len(points) == 0:
        return
    points.sort(key=operator.itemgetter(1))
    coresXL = {}
    width = grayL.shape[1]
    while len(points) > 0:
        curpoint = points.pop()
        x = curpoint[0]
        newseg = [x-ndisp*2-h_size,x+ndisp+h_size*2]
        if newseg[0] <= h_size:
            newseg[0] = h_size+1
        if newseg[1] >= width - h_size:
            newseg[1] = width-(h_size+1)
        
        # xR = getCores(grayL, grayR, index, x, seg, edge, lbpData, threshold, h_size)
        xR = getCores(grayL, grayR, index, x, newseg, edge, threshold, h_size)

        if math.fabs(int(grayL[index,x])-int(grayR[index,xR])) >= 5:
            continue

        if xR == -1:
            continue

        if coresXL.has_key(xR) == False:
            # xL = getCores(grayR, grayL, index, xR, seg, edgeR, lbpDataR, threshold, h_size)
            newsegR = [xR-ndisp*2-h_size,xR+ndisp*2+h_size]
            if newsegR[0] <= h_size:
                newsegR[0] = h_size+1
            if newsegR[1] >= width - h_size:
                newsegR[1] = width-(h_size+1)
            xL = getCores(grayR, grayL, index, xR, newsegR, edgeR, threshold, h_size)
            coresXL[xR] = xL
        else:
            xL = coresXL[xR]
        
        # 如果匹配则计算视差
        if x == xL and float(xL - xR) < ndisp and float(xL - xR) > 0:
            disp[index,x] = float(xL - xR)




 
def match(imgL, imgR, threshold, ndisp=64, h_size=2):
    # 计算灰度图
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # 获取形状
    assert(imgL.shape == imgR.shape)
    height = imgL.shape[0]
    width = imgL.shape[1]

    edge = getEdge1(imgL,imgR)  ###应该改为灰度！！！！！！
    # lbpData = [lbp.getLBP(grayL),lbp.getLBP(grayR)]

    # 初始化视差图
    disp = np.zeros((height,width))
    count = 0   #非零视差点个数
    for i in xrange(h_size, height-h_size):
        # 逐行匹配获取视差
        # matchline2(grayL, grayR, i, [h_size, width-h_size], edge, threshold, disp, ndisp, h_size)
        matchline3(grayL, grayR, i, [h_size, width-h_size], edge, threshold, disp, ndisp, h_size)


    #     for j in xrange(width):
    #         if disp[i,j] > 0:
    #             count += 1
    # ratio = float(count) / float(height*width)
    # print 'count : ', count
    # print ('ratio : %.6f' %(ratio))


    maxDisp = disp.max()
    # disp = disp / float(maxDisp)# * 255.0
    # disp.astype('uint8')
    return maxDisp, disp


def getCores2(grayL, grayR, index, x, seg, edge, h_size):
    if x == -1:
        return -1
    edgeLa, edgeLd, edgeRa, edgeRd = edge

    xRst = -1
    allCoef = []
    for xR in xrange(seg[0], seg[1]):
        coefE = areaCoeff(edgeLa,edgeRa,h_size,index,x,index,xR)
        coefG = areaCoeff(grayL,grayR,h_size,index,x,index,xR)
        allCoef.append([xR,coefE+coefG])


    if len(allCoef) == 0:
        return -1

    if len(allCoef) == 0:
        return -1
    allCoef.sort(key=operator.itemgetter(1),reverse = True)
    # if len(allCoef) > 1 and allCoef[0][1] - allCoef[1][1] < 0.1:
    #     return -1
    xRst = allCoef[0][0]

    return xRst
    


def match2(imgL, imgR, num, ndisp=64, h_size=2):
    orb = cv2.ORB_create(nfeatures = num)
    kp,des = orb.detectAndCompute(imgL,None)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    edge = getEdge1(imgL,imgR)
    # 边缘数据
    edgeLa, edgeLd, edgeRa, edgeRd = edge
    # 反向边缘数据
    edgeR = [edgeRa, edgeRd, edgeLa, edgeLd]
    height = grayL.shape[0]
    width = grayL.shape[1]
    disp = np.zeros((height,width))
    coresXL = {}
    for p in kp:
        x = int(round(p.pt[0]))
        index = int(round(p.pt[1]))
        newseg = [x-ndisp*2-h_size,x+ndisp+h_size*2]
        if newseg[0] <= h_size:
            newseg[0] = h_size+1
        if newseg[1] >= width - h_size:
            newseg[1] = width-(h_size+1)
        
        xR = getCores2(grayL, grayR, index, x, newseg, edge, h_size)

        if math.fabs(int(grayL[index,x])-int(grayR[index,xR])) >= 5:
            continue

        if xR == -1:
            continue

        if coresXL.has_key(xR) == False:
            newsegR = [xR-ndisp*2-h_size,xR+ndisp*2+h_size]
            if newsegR[0] <= h_size:
                newsegR[0] = h_size+1
            if newsegR[1] >= width - h_size:
                newsegR[1] = width-(h_size+1)
            xL = getCores2(grayR, grayL, index, xR, newsegR, edgeR, h_size)
            coresXL[xR] = xL
        else:
            xL = coresXL[xR]
        
        # 如果匹配则计算视差
        if x == xL and float(xL - xR) < ndisp and float(xL - xR) > 0:
            disp[index,x] = float(xL - xR)
    maxDisp = disp.max()
    # disp = disp / float(maxDisp)# * 255.0
    # disp.astype('uint8')
    return maxDisp, disp


def match3(imgL, imgR, num, t_coeff, scale, ndisp=64, h_size=2):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 预处理
    # imgL = cv2.medianBlur(imgL,3)
    # imgR = cv2.medianBlur(imgR,3)
    # imgL = hisEqualc(imgL)
    # imgR = hisEqualc(imgR)

    orb = cv2.ORB_create(nfeatures = int(num*scale))
    orb.setFastThreshold(0)
    kp1,des1 = orb.detectAndCompute(imgL,None)
    kp2,des2 = orb.detectAndCompute(imgR,None)
    if cv2.__version__.startswith('3'):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    if scale != 1:
        matches = sorted(matches, key = lambda x:x.distance) 
    height = imgL.shape[0]
    width = imgL.shape[1]
    disp = np.zeros((height,width))
    num = int(len(matches) / scale)
    print num
    count = 0
    for i in xrange(num):
        x1f = kp1[matches[i].queryIdx].pt[0]
        y1f = kp1[matches[i].queryIdx].pt[1]
        x2f = kp2[matches[i].trainIdx].pt[0]
        y2f = kp2[matches[i].trainIdx].pt[1]
        x1 = int(round(x1f))
        y1 = int(round(y1f))
        x2 = int(round(x2f))
        y2 = int(round(y2f))
        if x1 <= h_size or x1 >= width-h_size:
            continue
        if y1 <= h_size or y1 >= height-h_size:
            continue
        if x2 <= h_size or x2 >= width-h_size:
            continue
        if y2 <= h_size or y2 >= height-h_size:
            continue
        if y1 == y2:# math.fabs(y1f-y2f) < 1.0: 
            coef = areaCoeff(grayL,grayR,h_size,y1,x1,y2,x2)
            if coef >= t_coeff:
                disp[y1,x1] = kp1[matches[i].queryIdx].pt[0] - kp2[matches[i].trainIdx].pt[0]
                count += 1
            # disp[y1,x1] = kp1[matches[i].queryIdx].pt[0] - kp2[matches[i].trainIdx].pt[0]
            # count += 1
    maxDisp = disp.max()
    print count
    return maxDisp, disp


def matchGMS(imgL, imgR, num):
    # imgL = gms_matcher.imresize(imgL, 480)
    # imgR = gms_matcher.imresize(imgR, 480)

    height = imgL.shape[0]
    width = imgL.shape[1]
    disp = np.zeros((height,width))

    orb = cv2.ORB_create(num)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    gms = gms_matcher.GmsMatcher(orb, matcher)

    matches = gms.compute_matches(imgL, imgR)

    num = int(len(matches))
    print num
    count = 0
    for i in xrange(num):
        x1f = gms.keypoints_image1[matches[i].queryIdx].pt[0]
        y1f = gms.keypoints_image1[matches[i].queryIdx].pt[1]
        x2f = gms.keypoints_image2[matches[i].trainIdx].pt[0]
        y2f = gms.keypoints_image2[matches[i].trainIdx].pt[1]
        x1 = int(round(x1f))
        y1 = int(round(y1f))
        x2 = int(round(x2f))
        y2 = int(round(y2f))
        if y1 == y2: 
            disp[y1,x1] = gms.keypoints_image1[matches[i].queryIdx].pt[0] - gms.keypoints_image2[matches[i].trainIdx].pt[0]
            count += 1
    maxDisp = disp.max()
    print count
    return maxDisp, disp


def matchGMS2(imgL, imgR, num, t_coeff, h_size=2):
    # imgL = gms_matcher.imresize(imgL, 480)
    # imgR = gms_matcher.imresize(imgR, 480)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    height = imgL.shape[0]
    width = imgL.shape[1]
    disp = np.zeros((height,width))

    orb = cv2.ORB_create(num)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    gms = gms_matcher.GmsMatcher(orb, matcher)

    matches = gms.compute_matches(imgL, imgR)

    num = int(len(matches))
    print num
    count = 0
    for i in xrange(num):
        x1f = gms.keypoints_image1[matches[i].queryIdx].pt[0]
        y1f = gms.keypoints_image1[matches[i].queryIdx].pt[1]
        x2f = gms.keypoints_image2[matches[i].trainIdx].pt[0]
        y2f = gms.keypoints_image2[matches[i].trainIdx].pt[1]
        x1 = int(round(x1f))
        y1 = int(round(y1f))
        x2 = int(round(x2f))
        y2 = int(round(y2f))
        if y1 == y2: 
            coef = areaCoeff(grayL,grayR,h_size,y1,x1,y2,x2)
            if coef >= t_coeff:
                disp[y1,x1] = gms.keypoints_image1[matches[i].queryIdx].pt[0] - gms.keypoints_image2[matches[i].trainIdx].pt[0]
                count += 1
            
    maxDisp = disp.max()
    print count
    return maxDisp, disp


def calCost(grayL, grayR, sob, i, jL, jR):
    sobLa, sobLd, sobRa, sobRd = sob
    cost1 = (float(grayL[i,jL]) - float(grayR[i,jR])) ** 2
    mL = float(sobLa[i,jL])
    mR = float(sobRa[i,jR])
    thetaL = math.pi/180.0*sobLd[i,jL]
    thetaR = math.pi/180.0*sobRd[i,jR]
    cost2 = (mL)**2+(mR)**2-2*mL*mR*math.cos(thetaL-thetaR)
    cost = math.sqrt(cost1 + cost2)
    return cost

def getEdgeCoord(edge):
    h,w = edge.shape
    ec = []
    for j in xrange(w):
        ecx = []
        for i in xrange(h):
            if edge[i,j]>0:
                ecx.append(i)
        ec.append(ecx)
    # print ec
    # print len(ec)
    return ec

def firstPass(edgeL, edgeR, grayL, grayR, sob, minDisp, maxDisp):
    w = len(edgeL)
    matches = []
    for j in xrange(w):
        for i in edgeL[j]:
            minCost = float('inf')
            jR = -1
            for k in xrange(j-maxDisp,j-minDisp+1):
                if k < 0 or k >= w:
                    break
                if i in edgeR[k]:
                    cost = calCost(grayL,grayR,sob,i,j,k)
                    if cost < minCost:
                        jR = k
                        minCost = cost
            if jR != -1:
                matches.append([i,j,jR])
    print len(matches)
    return matches

def matchDP(imgL, imgR):
    height = imgL.shape[0]
    width = imgL.shape[1]
    disp = np.zeros((height,width))
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    sob = getEdge1(imgL, imgR)
    cannyL = cv2.Canny(grayL,200,400)
    cannyR = cv2.Canny(grayR,200,400)
    edgeL = getEdgeCoord(cannyL)
    edgeR = getEdgeCoord(cannyR)
    minDisp = 4
    maxDisp = 64
    firstMatches = firstPass(edgeL, edgeR, grayL, grayR, sob, minDisp, maxDisp)
    # cannyL, contoursL, hierarchyL = cv2.findContours(cannyL,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cannyR, contoursR, hierarchyR = cv2.findContours(cannyR,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for m in firstMatches:
        i = m[0]
        j = m[1]
        jR = m[2]
        disp[i,j] = j-jR
    maxDisp = disp.max()
    return maxDisp, disp


def keypointsinline(kp,des,height):
    height = int(height)
    kpline = []
    deslinelist = []
    indexes = []
    for i in xrange(height):
        kpline.append([])
        deslinelist.append([])
        indexes.append([])
    num = len(kp)
    for i in xrange(num):
        y = int(round(kp[i].pt[1]))
        kpline[y].append(kp[i])
        deslinelist[y].append(des[i])
        indexes[y].append(i)
    desline = []
    for i in xrange(height):
        desline.append(np.array(deslinelist[i]))
    return kpline,desline,indexes

def findIdx(matches,queryIdx):
    for m in matches:
        if m.queryIdx == queryIdx:
            return m.trainIdx
    return -1

def match4(imgL, imgR, num):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures = num)
    height = imgL.shape[0]
    width = imgL.shape[1]
    orb.setFastThreshold(0)
    kp1,des1 = orb.detectAndCompute(imgL,None)
    kp2,des2 = orb.detectAndCompute(imgR,None)
    kpline1,desline1,indexes1 = keypointsinline(kp1,des1,height)
    kpline2,desline2,indexes2 = keypointsinline(kp2,des2,height)
    if cv2.__version__.startswith('3'):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    finalmatches = []
    t = 64
    for i in xrange(height):
        kpl1 = kpline1[i]
        kpl2 = kpline2[i]
        if len(kpl1) == 0 or len(kpl2) == 0:
            continue
        desl1 = desline1[i]
        desl2 = desline2[i]
        assert(len(kpl1)==len(desl1) and len(kpl2)==len(desl2))
        ltor = bf.knnMatch(desl1,desl2,k=3)
        rtol = bf.match(desl2,desl1)
        for lrm in ltor:
            lIdx = lrm[0].queryIdx
            xL = kpl1[lIdx].pt[0]
            lIdx2 = -1
            for j in xrange(len(lrm)):
                rIdx = lrm[j].trainIdx
                lIdx2 = findIdx(rtol,rIdx)
                if lIdx == lIdx2:# and lrm[j].distance < t:
                    xR = kpl2[rIdx].pt[0]
                    finalmatches.append([i,xL,xR,1])
                    break
            if lIdx2 == -1:
                rIdx = lrm[0].trainIdx
                if lrm[0].distance < t:
                    xR = kpl2[rIdx].pt[0]
                    finalmatches.append([i,xL,xR,2])
    disp = np.zeros((height,width))
    for y,xL,xR,mt in finalmatches:
        xLi = int(round(xL))
        disp[y,xLi] = xL - xR
    maxDisp = disp.max()
    print len(finalmatches)
    return maxDisp, disp

def match5(imgL, imgR, num):
    olmatcher = olm.olMatcher(imgL,imgR,num)
    olmatcher.match()
    gms = gms_matcher.GmsMatcher(olmatcher.orb,olmatcher)
    height = imgL.shape[0]
    width = imgL.shape[1]
    disp = np.zeros((height,width))
    gms.keypoints_image1 = olmatcher.kp1
    gms.keypoints_image2 = olmatcher.kp2
    matches = gms.filtMatches(olmatcher.imgL,olmatcher.imgL,olmatcher.matches)
    num = int(len(matches))
    print num
    for i in xrange(num):
        x1f = olmatcher.kp1[matches[i].queryIdx].pt[0]
        yf = olmatcher.kp1[matches[i].queryIdx].pt[1]
        x2f = olmatcher.kp2[matches[i].trainIdx].pt[0]
        x1 = int(round(x1f))
        y = int(round(yf))
        x2 = int(round(x2f))
        disp[y,x1] = x1f-x2f

    maxDisp = disp.max()
    return maxDisp, disp



