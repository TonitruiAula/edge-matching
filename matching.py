# -*- coding :utf-8 -*-

import os
import time
import sys
import cv2
import numpy as np

def sob_bin(img, t_bin):
    sobel_X = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)
    sobel_Y = cv2.Sobel(img, cv2.CV_8UC1, 0, 1)
    sob = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    dst = cv2.threshold(sob, t_bin, 255, cv2.THRESH_OTSU)
    return dst

def getEdge(imgpath_L, imgpath_R, t_bin):
    img_L = cv2.imread(imgpath_L)
    img_R = cv2.imread(imgpath_R)
    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
    edge_L = sob_bin(gray_L, t_bin)
    edge_R = sob_bin(gray_R, t_bin)
    return [edge_L, edge_R]



def match1(imgpath_L, imgpath_R, t_bin, t_dir, t_abs, t_gray):
    edge_L, edge_R = getEdge(imgpath_L, imgpath_R, t_bin)
    cv2.imshow('left', edge_L)
    cv2.imshow('right', edge_R)


if __name__ == '__main__':
    imgpath_L = sys.argv[1]
    imgpath_R = sys.argv[2]
    t_bin = float(sys.argv[3])
    edge_L, edge_R = getEdge(imgpath_L, imgpath_R, t_bin)
    cv2.imshow('left', edge_L)
    cv2.imshow('right', edge_R)
