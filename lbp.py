# -*- coding=utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np
import math
from skimage.feature import local_binary_pattern

def getLBP(image):
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image,8,1)
    # print lbp
    # lbpImg = lbp / 255.0
    # cv2.imshow('LBP',lbpImg)
    # cv2.waitKey(0)
    return lbp

# getLBP('images/Jadeplant/im0.png')
def diffLBP(lbp1,lbp2):
    diff = int(lbp1)^int(lbp2)
    return bin(diff).count('1')