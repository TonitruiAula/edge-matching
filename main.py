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
def saveDisp(name, param, fun_type):
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
    if fun_type == '-g':
        maxDisp, disp = match(imgL, imgR, param, ndisp)
    elif fun_type == '-o':
        # maxDisp, disp = match2(imgL, imgR, param, ndisp)
        num, t_coeff, scale = param
        # maxDisp, disp = match3(imgL, imgR, num, t_coeff, scale)
        maxDisp, disp = match4(imgL, imgR, num)
    elif fun_type == '-gms':
        num = param
        maxDisp, disp = matchGMS(imgL, imgR, num)
    elif fun_type == '-gms2':
        num, t_coeff = param
        maxDisp, disp = matchGMS2(imgL, imgR, num, t_coeff)
    elif fun_type == '-dp':
        maxDisp, disp = matchDP(imgL, imgR)
    else:
        print 'function type error!'
        return
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
    fun_type = sys.argv[2]
        

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
    if fun_type == '-g':
        t_abs = float(sys.argv[3])  #t_abs:边缘绝对值的阈值，低于该值则不计算
        t_dir = float(sys.argv[4])  #t_dir:边缘方向差的阈值，左右图像的点的边缘方向差小于该阈值则将其放入候选点
        t_coeff = float(sys.argv[5])    #t_coeff:相关系数的阈值，核的边缘绝对值的相关系数低于该阈值则去掉
        param = [t_abs, t_dir, t_coeff]
        for name in imgList:
            # gt = saveGT(name)
            disp = saveDisp(name, param, fun_type)
        totalRst(imgList)
        analyze(imgName,t_dir,t_coeff,False)
        analyze(imgName,t_dir,t_coeff,True)
    elif fun_type == '-o':
        num = int(sys.argv[3])
        t_coeff = float(sys.argv[4])
        scale = float(sys.argv[5])
        param = [num, t_coeff, scale]
        for name in imgList:
            disp = saveDisp(name, param, fun_type)
        totalRst(imgList)
    elif fun_type == '-gms':
        num = int(sys.argv[3])
        for name in imgList:
            disp = saveDisp(name, num, fun_type)
        totalRst(imgList)
    elif fun_type == '-gms2':
        num = int(sys.argv[3])
        t_coeff = float(sys.argv[4])
        param = [num, t_coeff]
        for name in imgList:
            disp = saveDisp(name, param, fun_type)
        totalRst(imgList)
    elif fun_type == '-dp':
        for name in imgList:
            disp = saveDisp(name, 0, fun_type)
        totalRst(imgList)
    else:
        totalRst(imgList)
