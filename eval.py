# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import operator


def getLog(disp, gt, name):
    height = disp.shape[0]
    width = disp.shape[1]
    count = 0
    err = 0.0
    maxerr = 0.0
    errZ = 0.0
    maxerrZ = 0.0
    pointerr = []
    points = []
    deptherr = []

    # 获取遮挡信息图
    caliFile = open('images/' + name + '/calib.txt','r')
    caliLines = caliFile.readlines()
    fl = float(caliLines[0].split('[')[1].split(' ')[0])
    fr = float(caliLines[1].split('[')[1].split(' ')[0])
    f = (fl+fr)/2.0
    doffs = float(caliLines[2].split('=')[1])
    baselines = float(caliLines[3].split('=')[1])
    caliFile.close()
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
                zr = ze / zg
                # print e
                if e > maxerr:
                    maxerr = e
                err += e
                if ze > maxerrZ:
                    maxerrZ = ze
                points.append([i,j,d,g,e,zd,zg,ze,zr])
                errZ += ze
                pointerr.append(e)
                deptherr.append(ze)

    if count > 0:
        err /= count
        errZ /= count

    n, bins, patches = plt.hist(pointerr, bins=int(round(maxerr)), normed=1,edgecolor='None',facecolor='red')
    
    points.sort(key=operator.itemgetter(7),reverse=True)
    np.savetxt('images/' + name + '/log.txt',points,fmt='%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.6f')

    return [count, err, maxerr, errZ, maxerrZ]

def getDis(img, scale, t):
    logarray = np.loadtxt('images/' + img + '/log.txt')
    points = logarray.tolist()
    points.sort(key=operator.itemgetter(6))
    cur = 0
    good = 0
    mark = []
    for p in points:
        cur += 1
        if p[8] < scale:
            good += 1
        ratio = float(good)/cur
        mark.append([p[6],p[8],ratio])
    dis = 0.0
    r = 0.0
    ok = True
    for d in mark:
        if d[2] > t:
            dis = d[0]
            r = d[2]
    markArr = np.array(mark)
    np.savetxt('images/' + img + '/mark.txt',markArr,fmt='%.2f\t%.6f\t%.6f')
    if dis == 0.0:
        ok = False
        mark.sort(key=operator.itemgetter(2),reverse=True)
        r = mark[0][2]
        dis = mark[0][0]
    return [dis,ok,r]


def totalRst(imgList, scale=0.05, t=0.8):
    print 'saving result...'
    rst = open('rst.txt','w')
    print 'num \tape \tmpe \tade \tmde \tdis \tname\n'
    rst.write('num \tape \tmpe \tade \tmde \tdis \tname\n\n')
    bad = 0
    totalDis = []
    for img in imgList:
        disp = np.loadtxt('images/' + img + '/disp.txt')
        gt = np.loadtxt('images/' + img + '/gt.txt')
        count, err, maxerr, errZ, maxerrZ = getLog(disp, gt, img)
        dis,ok,r = getDis(img,scale,t)
        totalDis.append(dis)
        if ok == False:
            rst.write('%6d\t%6.2f\t%6.2f\t%8.2f\t%8.2f\t%8.2f\t %s(BAD:%.6f)\n' % (count,err,maxerr,errZ,maxerrZ,dis,img,r))
            print '%6d\t%6.2f\t%6.2f\t%6.2f\t %6.2f\t%8.2f\t %s(BAD:%.6f)' % (count,err,maxerr,errZ,maxerrZ,dis,img,r)
            bad += 1
        else:
            rst.write('%6d\t%6.2f\t%6.2f\t%8.2f\t%8.2f\t%8.2f\t %s\n' % (count,err,maxerr,errZ,maxerrZ,dis,img))
            print '%6d\t%6.2f\t%6.2f\t%6.2f\t %6.2f\t%8.2f\t %s' % (count,err,maxerr,errZ,maxerrZ,dis,img)
    n, bins, patches = plt.hist(totalDis, bins=10,edgecolor='None',facecolor='green')
    plt.title('distance delta=%.4f ratio=%.4f' % (scale,t))
    plt.savefig('dis.png')
    rst.close()
    print 'bad:',bad
    

def isGP(p,isAbs,t):
    if isAbs:
        if p[7] <= t:
            return True
    else:
        if p[8] <= t:
            return True
    return False

def nearEval(dis,mode,t):
    imgList = []
    # 从./imgList.txt文件中读取图片列表
    imgListFile = open('imgList.txt','r')
    for img in imgListFile.readlines():
        img = img.strip()
        if img[0] != '#' :
            imgList.append(img)
    isAbs = (mode == '-a')
    ratios = []
    for img in imgList:
        log = np.loadtxt('images/' + img + '/log.txt')
        loglist = log.tolist()
        loglist.sort(key=operator.itemgetter(6))
        count = 0
        good = 0
        for p in loglist:
            if p[6] > dis:
                break
            count += 1
            if isGP(p,isAbs,t):
                good += 1
        if count == 0:
            ratio = 0
        else:
            ratio = float(good) / count
        ratios.append(ratio)
        print '%s:%d/%d=%.6f' % (img,good,count,ratio)
    n, bins, patches = plt.hist(ratios, bins=10,edgecolor='None',facecolor='blue')
    if isAbs:
        plt.title('nearEval dis=%.1f mode=abs t=%.3f' % (dis,t))
    else:
        plt.title('nearEval dis=%.1f mode=rel t=%.3f' % (dis,t))
    plt.savefig('nearEval.png')


if __name__ == '__main__':
    dis = float(sys.argv[1])
    mode = sys.argv[2]
    t = float(sys.argv[3])
    nearEval(dis,mode,t)

