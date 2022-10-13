import cv2 as cv
import numpy as np
# import json
# import sys
import json


# Use cage
# import sys et donner le nom du ficher
## segmentation(sys.argv[1:], "coleoi.jpg")

def segmentation(argv, path):
    default_file = path
    # 'coleoi.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(
            'Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
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
    lastName = path.split("/")
    nameWithExt = lastName[len(lastName)-1]
    index = nameWithExt.split(".")
    name = index[0]

    p = "./data/"+name+".json"
    # -creating the json
    with open(p, "w",) as mon_fichier:
        json.dump({path: mylist}, mon_fichier)

    return 0
