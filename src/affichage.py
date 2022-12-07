# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import json

'''
json_path = 'trainingSet.json'



#with open(json_path, 'r') as f:
#  data = json.load(f)
'''

def drawImageSet(dataset,labels,fileNamePrefix=None,datadir=".", title="Figure"):
    for i,pkey in enumerate(dataset.keys()):
        fileName = None
        print("plotting ", datadir+pkey)
        if fileNamePrefix is not None:
           fileName = "%s%i.png"%(fileNamePrefix,i)
        show_image(dataset[pkey],datadir+pkey,labels,outfilename=fileName,figureNumber=i, title=title)


def show_image(data, imfFile, labels, outfilename=None , figureNumber = 0, title="Figure"):
 
  
    plt.rcParams['figure.dpi'] = 200
    labels_count = {}
    for key in labels:
        labels_count[key] = 0
    total_circles = 0
    ax = plt.gca()
    ax.cla()
    for circle in range(len(data)):
        x = data[circle]['coordinates'][0]
        y = data[circle]['coordinates'][1]
        size = data[circle]['coordinates'][2]
        label = data[circle]['label']
        if label not in labels:
            label = "non classé"
        labels_count[label] += 1
        total_circles += 1
        circle = plt.Circle((x, y), size, color=labels[label][1], linewidth=0.3,fill=False)
        ax.add_patch(circle)

    patches = [mpatches.Patch(color=labels[label][1],label=label
                              +" - "+str(labels_count[label])
                              +" ("+str(round(labels_count[label]/total_circles*100, 2))+" %)")
               for label in labels_count]
    patches += [mpatches.Patch(color=[0,0,0],label="Total : "+str(total_circles))]
                      
  
    img = mpimg.imread(imfFile)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    lgd = plt.legend(handles=patches, loc='lower left', borderaxespad=0., bbox_to_anchor=(1.1, 0))  
    
    plt.title("%s %d"%(title,figureNumber))
    if outfilename is not None :
        plt.savefig(outfilename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()