# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
from matching import *

# 分析坏点的各项数据
def analyze(imgName,t_dir,t_coeff,occ=False):
    # 获取图片
    imgList = []
    if imgName == '-all':
        imgListFile = open('imgList.txt','r')
        for img in imgListFile.readlines():
            img = img.strip()
            if img[0] != '#' :
                imgList.append(img)
    else:
        imgList.append(imgName)

    # 坏点分析结果
    if occ == False:
        badstat = open('badstat.txt','w')
    else:
        badstat = open('badstat_no.txt','w')

    for img in imgList:
        # 读取左右视图
        imgpathL = 'images/' + img + '/im0.png'
        imgpathR = 'images/' + img + '/im1.png'

        imgL = cv2.imread(imgpathL)
        imgR = cv2.imread(imgpathR)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        edgeLa, edgeLd, edgeRa, edgeRd = getEdge1(imgL,imgR)

        # 读取坏点信息文件
        if occ == False and os.path.exists('images/' + img + '/badlog.txt'):
            badpoints = np.loadtxt('images/' + img + '/badlog.txt')
        elif os.path.exists('images/' + img + '/badlog_no.txt'):
            badpoints = np.loadtxt('images/' + img + '/badlog_no.txt')
        else:
            continue
        
        count = badpoints.shape[0]  #坏点个数
        if count <= 0:
            continue
        badtypecount = [0,0,0,0,0,0,0,0]    #各种错误的计数
        # 打开分析文件
        if occ == False:
            rst = open('images/' + img + '/analyzing_result.txt','w')
        else:
            rst = open('images/' + img + '/analyzing_result_no.txt','w')
        rst.write('total badpoints = %d, t_dir = %.2f, t_coeff = %.2f\n' % (count,t_dir,t_coeff))
        rst.write('y\txL\txRe\txRg\te\t\tdL\t\tdRe\t\tdRg\t\trGe\t\trGg\t\trAe\t\trAg\t\ttype\n')
        for index in xrange(count):
            y = int(badpoints[index,0])
            xL = int(badpoints[index,1])
            xRe = int(xL - badpoints[index,2])
            xRg = int(xL - badpoints[index,3])
            err = badpoints[index,4]
            dL = edgeLd[y,xL]
            dRe = edgeRd[y,xRe]        
            dRg = edgeRd[y,xRg]
            rGe = areaCoeff(edgeLa,edgeRa,2,y,xL,y,xRe)        
            rGg = areaCoeff(edgeLa,edgeRa,2,y,xL,y,xRg)
            rAe = areaCoeff(grayL,grayR,2,y,xL,y,xRe)        
            rAg = areaCoeff(grayL,grayR,2,y,xL,y,xRg)
            err_type = 0
            if math.fabs(dL-dRg) >= t_dir:
                err_type += 1
            if rGg <= t_coeff:
                err_type += 2
            if rAg < rAe:
                err_type += 4
            badtypecount[err_type] += 1
            rst.write('%d\t%d\t%d\t%d\t%.2f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%d\n' % (y,xL,xRe,xRg,err,dL,dRe,dRg,rGe,rGg,rAe,rAg,err_type))
        rst.close()
        badstat.write('name=%s\t\tstep1=%d/%d=%.2f\tstep2=%d/%d=%.2f\tstep3=%d/%d=%.2f\tnone=%d/%d=%.2f\n' % \
                    (img,\
                    badtypecount[1] + badtypecount[3] + badtypecount[5] + badtypecount[7], count, float(badtypecount[1] + badtypecount[3] + badtypecount[5] + badtypecount[7]) / float(count),\
                    badtypecount[2] + badtypecount[3] + badtypecount[6] + badtypecount[7], count, float(badtypecount[2] + badtypecount[3] + badtypecount[6] + badtypecount[7]) / float(count),\
                    badtypecount[4] + badtypecount[5] + badtypecount[6] + badtypecount[7], count, float(badtypecount[4] + badtypecount[5] + badtypecount[6] + badtypecount[7]) / float(count),\
                    badtypecount[0], count, float(badtypecount[0]) / float(count)))

if __name__ == '__main__':
    imgName = sys.argv[1]
    t_dir = float(sys.argv[2])
    t_coeff = float(sys.argv[3])
    analyze(imgName,t_dir,t_coeff,False)
    analyze(imgName,t_dir,t_coeff,True)
