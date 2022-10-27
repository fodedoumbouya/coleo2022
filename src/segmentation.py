#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 08:55:10 2022

@author: filippi_j
"""

import cv2 as cv
import numpy as np
# import json
# import sys


# Use cage


# going through the dict
def segmentDict(dict):
    for pathKey in list(dict.keys()):
        dict[pathKey] = segmentation(path=pathKey)
    return dict


def segmentation(path):
    # default_file = path
    src = cv.imread(cv.samples.findFile(path), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(
            'Usage: hough_circle.py [image_name -- default ' + path + '] \n')
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
                "coord": {
                    "x": int(i[0]),
                    "y": int(i[1]),
                    "s": int(i[2])
                },
                "label": ""
            }
            mylist.append(model)

    # removing the extenstion i
    # lastName = path.split("/")
    # nameWithExt = lastName[len(lastName)-1]
    # index = nameWithExt.split(".")
    # name = index[0]

    # p = "./data/"+name+".json"
    # -creating the json
    # with open(p, "w",) as mon_fichier:
    #     json.dump({path: mylist}, mon_fichier)

    return mylist