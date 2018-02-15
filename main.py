# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
from matching import *
from eval import *
from analyzing import *
import pfm

# 计算并保存视差数据
def saveDisp(name, threshold):
    print 'the current image is: ', name
    imgpathL = 'images/' + name + '/im0.png'
    imgpathR = 'images/' + name + '/im1.png'
    calib = 'images/' + name + '/calib.txt'
    f = open(calib,'r')
    ndispL =  f.readlines()[6] 
    ndisp = int(ndispL[6:])      
    f.close()
    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)

    print 'shape :', imgL.shape
    # maxDisp, disp = match1(imgL, imgR, edge, seg_len, t_abs, t_dir, t_coeff)
    maxDisp, disp = match(imgL, imgR, threshold, ndisp)
    
    np.savetxt('images/' + name + '/disp.txt',disp,'%.2f')
    return disp

# 将.pfm格式的视差图转为文本文件
def saveGT(name):
    gtPath = 'images/' + name + '/disp0GT.pfm'
    p = pfm.pfmDisp(gtPath)
    gt = p.getDisp()
    if os.path.exists('images/' + name + '/gt.txt') == False:
        np.savetxt('images/' + name + '/gt.txt',gt,'%.2f')
    return gt

if __name__ == '__main__':

    print os.path.abspath(os.curdir)
    print 'starting...'

    imgName = sys.argv[1]   #图片的名称
    t_abs = float(sys.argv[2])  #t_abs:边缘绝对值的阈值，低于该值则不计算
    t_dir = float(sys.argv[3])  #t_dir:边缘方向差的阈值，左右图像的点的边缘方向差小于该阈值则将其放入候选点
    t_coeff = float(sys.argv[4])    #t_coeff:相关系数的阈值，核的边缘绝对值的相关系数低于该阈值则去掉
        
    threshold = [t_abs, t_dir, t_coeff]

    imgList = []
    # 从./imgList.txt文件中读取图片列表
    if imgName == '-all':
        imgListFile = open('imgList.txt','r')
        for img in imgListFile.readlines():
            img = img.strip()
            if img[0] != '#' :
                imgList.append(img)
    else:
        imgList.append(imgName)

    inf = float('inf')

    # 如果t_abs小于等于0则直接分析结果
    if t_abs > 0:
        for name in imgList:
            gt = saveGT(name)
            # mp是右图匹配点的位置，debug用
            mp = gt.copy()
            for i in xrange(mp.shape[0]):
                for j in xrange(mp.shape[1]):
                    if gt[i,j] < inf:
                        mp[i,j] = j - gt[i,j]
            disp = saveDisp(name, threshold)
            # count, err, maxerr, perfect, ratio = eval(disp,gt)
            # print 'name: '+str(name)+'count: '+str(count)+'average error: '+str(err)+'max error: '+str(maxerr)+'perfect: '+str(perfect)+'perfect ratio: '+str(ratio)
    
    totalRst(imgList)
    analyze(imgName,t_dir,t_coeff)
