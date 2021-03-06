# -*- coding=utf-8 -*-

import sys
import cv2
import numpy as np
import struct
import math

class pfmDisp:
    def __init__(self, filename):
        self.file = open(filename,'rb')
        self.hasDisp = False
    
    def __del__(self):
        if self.file.closed == False:
            self.file.close()

    # 打印文件头信息
    def printInfo(self):
        print '\nthe information of GT file:'
        print 'type: %s' % self.type
        # print self.type == 'Pf\n'
        print 'width: %d, height: %d' %(self.size[0], self.size[1])
        if self.endian:
            print 'endian: Big-Endian'
        else:
            print 'endian: Little-Endian'
        print 'scale: %f\n' % self.scale

        # print self.file.readline()
        # a = self.file.readline()
        # print a
        # print self.file.readline()
        # f = struct.unpack('<f',self.file.read(4))
        # # f = float(self.file.read(4))
        # print f
        # size = a.split()
        # print size
        # width = int(size[0])
        # height = int(size[1])
        # print [width,height]
        # print int(size)

    # 获取文件头信息
    def getInfo(self):
        self.file.seek(0)
        self.type = self.file.readline()
        s = self.file.readline().split()
        self.size = [int(s[0]), int(s[1])]
        se = float(self.file.readline())
        self.endian = se > 0
        self.scale = math.fabs(se)

    # 将视差.pfm文件转为文本文件
    def getDisp(self):
        self.getInfo()
        width = self.size[0]
        height = self.size[1]
        self.disp = np.array([])
        gray = self.type == 'Pf\n'
        if gray:
            self.disp = np.zeros((height, width))
            # print disp.shape
        else:
            print 'the file in not 1-channel!'
            return
        for i in xrange(height):
            for j in xrange(width):
                data = self.file.read(4)
                # val = self.scale
                val = 1
                if self.endian:
                    val *= struct.unpack('>f',data)[0]
                else:
                    val *= struct.unpack('<f',data)[0]
                # print val
                self.disp[height-1-i,j] = val
        self.hasDisp = True
        return self.disp

    # 将视差图可视化
    def getDispGraph(self):
        if not self.hasDisp:
            self.getDispGraph()
        graph = self.disp.copy()
        width = self.size[0]
        height = self.size[1]
        inf = float('inf')
        maxDisp = 0
        for i in xrange(height):
            for j in xrange(width):
                if graph[i,j] < inf:
                    if graph[i,j] > maxDisp:
                        maxDisp = graph[i,j]
                else:
                    graph[i,j] = 0

        # for i in xrange(height):
        #     for j in xrange(width):
        #         if graph[i,j] is inf:
        #             graph[i,j] = maxDisp
        #             # graph[i,j] = 0

        # graph /= maxDisp
        graph *= self.scale
        graph.astype('uint8')
        return graph, maxDisp
        

