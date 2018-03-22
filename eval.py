# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import operator

def eval(disp, gt, name, occ=False, baderr=2):
    height = disp.shape[0]
    width = disp.shape[1]
    count = 0
    err = 0.0
    perfect = 0
    maxerr = 0.0
    errZ = 0.0
    maxerrZ = 0.0
    pointerr = []
    badpoints = []
    # 获取遮挡信息图
    occpath = 'images/' + name + '/mask0nocc.png'
    caliFile = open('images/' + name + '/calib.txt','r')
    caliLines = caliFile.readlines()
    fl = float(caliLines[0].split('[')[1].split(' ')[0])
    fr = float(caliLines[1].split('[')[1].split(' ')[0])
    f = (fl+fr)/2.0
    doffs = float(caliLines[2].split('=')[1])
    baselines = float(caliLines[3].split('=')[1])
    caliFile.close()
    if os.path.exists(occpath) == False:
        occ = False
    occfile = None
    if occ:
        occfile = cv2.imread(occpath)
        if len(occfile.shape) == 3:
            occfile = cv2.cvtColor(occfile,cv2.COLOR_BGR2GRAY)
        disp = disp * (occfile == 255)
    # else:
    #     badlog = open('images/' + name + '/badlog.txt','w')

    # err_graph = np.zeros((height,width,3),'uint8')
    # err_graph = np.ones((height,width,3),'uint8')
    # err_graph *= 255
    inf = float('inf')
    for i in xrange(height):
        for j in xrange(width):
            d = float(disp[i,j])
            g = float(gt[i,j])
            if d > 0 and g != inf and g != 0:
                count += 1
                e = math.fabs(float(d-g))
                zd = (baselines*f)/(d+doffs)
                zg = (baselines*f)/(g+doffs)
                ze = math.fabs(zd-zg)
                # print e
                if e == 0:
                    perfect += 1
                if e > maxerr:
                    maxerr = e
                if e > baderr:
                    badpoints.append([i,j,d,g,e])
                    # if occ == False:
                    #     badlog.write('(%d, %d) (disp=%.2f,gt=%.2f,error=%.2f)\n' % (i,j,d,g,e))
                err += e
                if ze > maxerrZ:
                    maxerrZ = ze
                errZ += ze
                pointerr.append(e)
                # err_graph[i,j,0] -= int(e)
                # err_graph[i,j,1] -= int(e)
    # for i in xrange(height):
    #     for j in xrange(width):
    #             err_graph[i,j,0] -= int(e / maxerr * 255.0)
    #             err_graph[i,j,1] -= int(e / maxerr * 255.0)
    bad = len(badpoints)
    if count > 0:
        err /= count
        errZ /= count
    # print 'count: ', count
    # print 'average error: ', err
    # print 'max error: ', maxerr
    # print 'perfect: ', perfect
        p_ratio = float(perfect) / float(count)
        b_ratio = float(bad) / float(count)
    else:
        p_ratio = b_ratio = 0
    # print 'perfect ratio: ', ratio
    n, bins, patches = plt.hist(pointerr, bins=int(maxerr), normed=1,edgecolor='None',facecolor='red')  
    badpoints.sort(key=operator.itemgetter(4),reverse=True)
    badarray = np.array(badpoints)
    if occ == False:
        plt.savefig('images/' + name + '/hist.png')
        if bad > 0:
            np.savetxt('images/' + name + '/badlog.txt',badarray,fmt='%d %d %.2f %.2f %.2f')
    else:
        plt.savefig('images/' + name + '/hist_no.png')
        if bad > 0:
            np.savetxt('images/' + name + '/badlog_no.txt',badarray,fmt='%d %d %.2f %.2f %.2f')
    return [count, err, maxerr, perfect, p_ratio, bad, b_ratio, errZ, maxerrZ]
    # cv2.imshow('err',err_graph)
    # cv2.waitKey(0)

def evalTxt(name, occ=False):
    disp = np.loadtxt('images/' + name + '/disp.txt')
    gt = np.loadtxt('images/' + name + '/gt.txt')
    return eval(disp, gt, name, occ)

def totalRst(imgList):
    log = open('rst.txt','w')
    log_no = open('rst_no_occlution.txt','w')
    print 'count\taverage error\tmax error\taverage errorZ\tmax errorZ\tperfect\tperfect ratio\tbad\tbad ratio\tname'
    log.write('count\taverage error\tmax error\taverage errorZ\tmax errorZ\tperfect\tperfect ratio\tbad\tbad ratio\tname\n\n')
    log_no.write('count\taverage error\tmax error\taverage errorZ\tmax errorZ\tperfect\tperfect ratio\tbad\tbad ratio\tname\n')
    for img in imgList:
        count, err, maxerr, perfect, p_ratio, bad, b_ratio, errZ, maxerrZ= evalTxt(img,False)
        print '%d\t\t%.2f\t\t%.2f\t\t%.2f\t\t\t%.2f\t%d\t%.5f\t\t%d\t%.5f \t%s' % (count, err, maxerr, errZ, maxerrZ, perfect, p_ratio, bad, b_ratio, img)
        log.write('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t\t%.5f\t%d\t%.5f\t%s\n' % (count, err, maxerr, errZ, maxerrZ, perfect, p_ratio, bad, b_ratio, img))
        count, err, maxerr, perfect, p_ratio, bad, b_ratio, errZ, maxerrZ = evalTxt(img,True)
        print '%d\t\t%.2f\t\t%.2f\t\t%.2f\t\t\t%.2f\t%d\t%.5f\t\t%d\t%.5f \t%s(NO)' % (count, err, maxerr, errZ, maxerrZ, perfect, p_ratio, bad, b_ratio, img)
        log_no.write('%d\t\t%.2f\t\t%.2f\t\t%.2f\t\t\t%.2f\t%d\t%.5f\t\t%d\t%.5f\t\t%s\n' % (count, err, maxerr, errZ, maxerrZ, perfect, p_ratio, bad, b_ratio, img))

    log.close()
