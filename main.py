# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
from matching import *
from eval import *
import pfm

def saveDisp(name):
    print 'the current image is: ', name
    imgpathL = 'images/' + name + '/im0.png'
    imgpathR = 'images/' + name + '/im1.png'
    imgL = cv2.imread(imgpathL)
    imgR = cv2.imread(imgpathR)

    print 'shape :', imgL.shape

    edge = getEdge1(imgL,imgR)

    # ld = edge[1]/90.0
    # ld.astype('uint8')
    # rd = edge[3]/90.0
    # rd.astype('uint8')
    # cv2.imshow('left grad abs',edge[0])
    # cv2.imshow('left grad dir',ld)

    maxDisp, disp = match1(imgL, imgR, edge, seg_len, t_abs, t_dir, t_coeff)
    # disp /= float(maxDisp)
    # disp /= 255.0
    # disp.astype('uint8')
    # print 'max disparity: ' , maxDisp
    # cv2.imshow('disp',disp)
    # cv2.imwrite('disp.png',disp)
    np.savetxt('images/' + name + '/disp.txt',disp,'%.2f')
    return disp

def saveGT(name):
    gtPath = 'images/' + name + '/disp0GT.pfm'
    p = pfm.pfmDisp(gtPath)
    gt = p.getDisp()
    np.savetxt('images/' + name + '/gt.txt',gt,'%.2f')
    return gt

if __name__ == '__main__':

    print os.path.abspath(os.curdir)
    print 'starting...'

    imgName = sys.argv[1]
    seg_len = int(sys.argv[2])
    t_abs = float(sys.argv[3])
    t_dir = float(sys.argv[4])
    t_coeff = float(sys.argv[5])
        
    imgList = []

    if imgName == '-all':
        imgListFile = open('imgList.txt','r')
        for img in imgListFile.readlines():
            img = img.strip()
            imgList.append(img)

    else:
        imgList.append(imgName)
    
    if seg_len > 0:
        for name in imgList:
            disp = saveDisp(name)
            gt = saveGT(name)
            # count, err, maxerr, perfect, ratio = eval(disp,gt)
            # print 'name: '+str(name)+'count: '+str(count)+'average error: '+str(err)+'max error: '+str(maxerr)+'perfect: '+str(perfect)+'perfect ratio: '+str(ratio)
    
    totalRst(imgList)
