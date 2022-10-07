import cv2
import numpy as np

image = cv2.imread('coleoi.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
 
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
valid = 0
ROI_number = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    area = cv2.contourArea(c)

    if len(approx) > 5 and area > 100 and area < 400:
# ---------------------- Enregistrement des sous-images ---------------------- #
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        cv2.imwrite('output/ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1
# ---------------------------------------------------------------------------- #

        valid +=  1
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)

# texte
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1 
color = (255, 0, 0)
thickness = 2
cv2.putText(image, '%d recepteurs detectes'%valid, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.imwrite('detected.png', image)
cv2.waitKey()