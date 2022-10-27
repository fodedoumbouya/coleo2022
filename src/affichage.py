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

def show_image(imageKeySet, outfilename=None ):
    plt.rcParams['figure.dpi'] = 800
    labels_count = {"autres": 0,
                    "coleonica": 0,
                    "faux": 0,
                    "flat": 0,
                    "placodea": 0,
                    "pore": 0,
                    "ringed": 0,
                    "roundtip": 0,
                    "sting": 0,
                    "sunken": 0,
                    "wrinkled": 0,
                    "non classé": 0}
    total_circles = 0
    ax = plt.gca()
    ax.cla()
    for circle in range(len(data[image])):
        x = data[image][circle]['coordinates'][0]
        y = data[image][circle]['coordinates'][1]
        size = data[image][circle]['coordinates'][2]
        label = data[image][circle]['label']
        if label not in labels:
            label = "non classé"
        labels_count[label] += 1
        total_circles += 1
        circle = plt.Circle((x, y), size, color=labels[label], fill=False)
        ax.add_patch(circle)
    patches = [mpatches.Patch(color=labels[label],label=label
                              +" - "+str(labels_count[label])
                              +" ("+str(round(labels_count[label]/total_circles*100, 2))+" %)")
               for label in labels_count]
    patches += [mpatches.Patch(color=[0,0,0],label="Total : "+str(total_circles))]
    image = "../" + image
    img = mpimg.imread(image)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    lgd = plt.legend(handles=patches, loc='lower left', borderaxespad=0., bbox_to_anchor=(1.1, 0))
    if outfilename is not None :
        plt.savefig(outfilename, bbox_extra_artists=(lgd,), bbox_inches='tight')

#name="output/output"
#i=0
#for image in data:
#    show_image(image, name+"_"+str(i))
#    i += 1
