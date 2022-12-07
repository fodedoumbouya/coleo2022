#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 08:55:10 2022

@author: filippi_j
"""

import cv2 as cv
import numpy as np
import json
# import sys


# Use cage


# going through the dict
def segmentImageSet(imageSet,datadir="."):
    outSet = {}
    for pkey in imageSet.keys():
        outSet[pkey] = segmentBruteForce(pkey,datadir)
        #outSet[pkey] = segmentImage(pkey,datadir)
    return outSet


def segmentImage(imagePath,datadir="."):
    

    # default_file = path
    src = cv.imread(datadir+imagePath, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(
            'Usage: hough_circle.py [image_name -- default ' + imagePath + '] \n')
        return -1


    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 200,
                              param1=30, param2=30,
                              minRadius=1, maxRadius=30)
    mylist = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            model = {
                "coordinates": [int(i[0]),int(i[1]), int(i[2])],
                "label": ""
            }
            mylist.append(model)

    return mylist

def segmentBruteForce(imagePath,datadir="."):
    # default_file = path
    src = cv.imread(datadir+imagePath, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(
            'Usage: hough_circle.py [image_name -- default ' + imagePath + '] \n')
        return -1

    img_span=34
    mylist=[]
    for i in range(img_span, src.shape[0]-img_span, img_span//2):
        for j in range(img_span, src.shape[1]-img_span, img_span//2):
            model = {
                "coordinates": [j,i, img_span],
                "label": ""
            }
            mylist.append(model)

    return mylist
    


