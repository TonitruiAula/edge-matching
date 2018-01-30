# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math

def eval(disp, gt, name):
    height = disp.shape[0]
    width = disp.shape[1]
    count = 0
    err = 0.0
    perfect = 0
    maxerr = 0.0
    bad = 0

    badlog = open('images/' + name + '/badlog.txt','w')
    # err_graph = np.zeros((height,width,3),'uint8')
    # err_graph = np.ones((height,width,3),'uint8')
    # err_graph *= 255
    inf = float('inf')
    for i in xrange(height):
        for j in xrange(width):
            d = float(disp[i,j])
            g = float(gt[i,j])
            if d > 0 and g <> inf:
                count += 1
                e = math.fabs(float(d-g))
                # print e
                if e == 0:
                    perfect += 1
                if e > maxerr:
                    maxerr = e
                if e > 10:
                    bad += 1
                    badlog.write('(%d, %d) (disp=%.2f,gt=%.2f,error=%.2f)\n' % (i,j,d,g,e))
                err += e
                # err_graph[i,j,0] -= int(e)
                # err_graph[i,j,1] -= int(e)
    # for i in xrange(height):
    #     for j in xrange(width):
    #             err_graph[i,j,0] -= int(e / maxerr * 255.0)
    #             err_graph[i,j,1] -= int(e / maxerr * 255.0)
    if count > 0:
        err /= count
    # print 'count: ', count
    # print 'average error: ', err
    # print 'max error: ', maxerr
    # print 'perfect: ', perfect
    p_ratio = float(perfect) / float(count)
    b_ratio = float(bad) / float(count)
    # print 'perfect ratio: ', ratio
    badlog.close()
    return [count, err, maxerr, perfect, p_ratio, bad, b_ratio]
    # cv2.imshow('err',err_graph)
    # cv2.waitKey(0)

def evalTxt(name):
    disp = np.loadtxt('images/' + name + '/disp.txt')
    gt = np.loadtxt('images/' + name + '/gt.txt')
    return eval(disp, gt, name)

def totalRst(imgList):
    log = open('rst.txt','w')
    print 'count\taverage error\tmax error\tperfect\tperfect ratio\tbad\tbad ratio\tname'
    log.write('count\taverage error\tmax error\tperfect\tperfect ratio\tbad\tbad ratio\tname\n')
    for img in imgList:
        count, err, maxerr, perfect, p_ratio, bad, b_ratio = evalTxt(img)    
        print '%d\t\t%.2f\t%.2f\t\t%d\t%.5f\t\t%d\t%.5f\t\t%s' % (count, err, maxerr, perfect, p_ratio, bad, b_ratio, img)
        log.write('%d\t\t%.2f\t%.2f\t\t%d\t%.5f\t\t%d\t%.5f\t\t%s\n' % (count, err, maxerr, perfect, p_ratio, bad, b_ratio, img))
    log.close()
