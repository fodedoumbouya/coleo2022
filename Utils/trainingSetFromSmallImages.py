import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import PIL
import cv2

import json



plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100 

dirpath="/Users/filippi_j/Documents/workspace/coleo/"
dirpathds=dirpath+"/Utils/dataset/types recepteurs/"
dirclasses=['autres', 'coleonica', 'faux', 'flat', 'placodea', 'pore', 'ringed', 'roundtip', 'sting', 'sunken', 'wrinkled']

dataFiles={}

dataFiles["Utils/dataset/Lower/1.bmp"] = []
dataFiles["Utils/dataset/Upper/1.bmp"] = []
dataFiles["Utils/dataset/Middle/1.bmp"] = []

names = ["Utils/dataset/Lower/1.bmp","Utils/dataset/Upper/1.bmp","Utils/dataset/Middle/1.bmp"]

classifiedList={}
for classname in dirclasses:
    path=dirpathds+classname+"/" 
    for file in os.listdir(path): 
        if(str(file).endswith("Copie.png")):
            print("rm ",path+file)
        
        if(str(file).endswith(".png")):
            px= file.split("y")[0][1:]
            py= file.split("r")[0][len(px)+2:]
            diam= file.split(".png")[0][len(px)+len(py)+3:]
            #print(file, px, py , diam)
            
            img = cv2.imread(path+file, cv2.IMREAD_COLOR)
            classifiedList[file] = {}
            classifiedList[file]["coordinates"] = (int(px),int(py),int(diam))
            classifiedList[file]["label"] = classname
            classifiedList[file]["image"] = img
            classifiedList[file]["width"] = np.shape(img)[0]

dataImg = {}            
for name in names:
    dataImg[name] = cv2.imread(dirpath+name, cv2.IMREAD_COLOR)

sList = list(classifiedList.keys()) 

lS = 10
uCount = 0
scorePlot = 0
for posFig, classKey in enumerate(sList):
    u = classifiedList[classKey]
    x,y,rd = u["coordinates"]
    r = int(u["width"]/2)
    imD = []
    for name in names:
        imD.append(dataImg[name][y-r:y+r,x-r:x+r])
        
    imTest = u["image"]

    scoreSQ = []
    for im in imD:
        scoreSQ.append(np.sum(np.square(im-imTest)))
        
    score = scoreSQ
    score = 1-(score/np.max(score))
    localdict = {}
    localdict["coordinates"] =u["coordinates"]
    localdict["width"] =u["width"]
    localdict["label"] =u["label"]
    
    dataFiles[names[np.argmax(score)]].append(localdict)
 

    if(scorePlot > 0):
        if score[np.argmax(score)]< scorePlot:
            imD[np.argmax(score)][:,:,0] = 200
            for i,im in enumerate(imD):
                plt.subplot(lS, 4,uCount*4 + i+1)
                plt.imshow(im)
            plt.subplot(lS, 4, uCount*4+len(imD)+1)
            plt.imshow(imTest)
            uCount = uCount+1
        plt.show()

with open('trainingSet.json', 'w') as fp:
    json.dump(dataFiles, fp, sort_keys=True, indent=4)

