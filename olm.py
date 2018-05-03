# -*- coding=utf-8 -*-

import cv2
import numpy as np

class olMatcher:
    def __init__(self,imgL, imgR,num,t=0):
        self.imgL = imgL
        self.imgR = imgR
        self.orb = cv2.ORB_create(nfeatures=num)
        self.height = imgL.shape[0]
        self.width = imgL.shape[1]
        self.orb.setFastThreshold(t)
        self.matches = []

    def keypointsinline(self, kp, des, height):
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
        return kpline, desline, indexes

    def findIdx(self, matches, queryIdx):
        for m in matches:
            if m.queryIdx == queryIdx:
                return m.trainIdx
        return -1

    def match(self,t=64):
        self.kp1, self.des1 = self.orb.detectAndCompute(self.imgL, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.imgR, None)
        kpline1, desline1, indexes1 = self.keypointsinline(self.kp1, self.des1, self.height)
        kpline2, desline2, indexes2 = self.keypointsinline(self.kp2, self.des2, self.height)
        if cv2.__version__.startswith('3'):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        for i in xrange(self.height):
            kpl1 = kpline1[i]
            kpl2 = kpline2[i]
            if len(kpl1) == 0 or len(kpl2) == 0:
                continue
            desl1 = desline1[i]
            desl2 = desline2[i]
            assert (len(kpl1) == len(desl1) and len(kpl2) == len(desl2))
            ltor = bf.knnMatch(desl1, desl2, k=3)
            rtol = bf.match(desl2, desl1)
            for lrm in ltor:
                lIdx = lrm[0].queryIdx
                lIdx2 = -1
                for j in xrange(len(lrm)):
                    rIdx = lrm[j].trainIdx
                    lIdx2 = self.findIdx(rtol, rIdx)
                    if lIdx == lIdx2 and lrm[j].distance < t:
                        m = lrm[j]
                        m.queryIdx = indexes1[i][lIdx]
                        m.trainIdx = indexes2[i][rIdx]
                        self.matches.append(m)
                        break
                if lIdx2 == -1:
                    rIdx = lrm[0].trainIdx
                    if lrm[0].distance < t:
                        m = lrm[0]
                        m.queryIdx = indexes1[i][lIdx]
                        m.trainIdx = indexes2[i][rIdx]
                        self.matches.append(m)
