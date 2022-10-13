# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import json

json_path = 'trainingSet.json'
plt.rcParams['figure.dpi'] = 800
labels = {"autres": [238/255,59/255,59/255],
         "coleonica": [124/255,205/255,124/255],
         "faux": [255/255,255/255,0/255],
         "flat": [142/255,229/255,238/255],
         "placodea": [255/255,211/255,155/255],
         "pore": [0/255,201/255,87/255],
         "ringed": [255/255,110/255,180/255],
         "roundtip": [145/255,44/255,238/255],
         "sting": [238/255,121/255,66/255],
         "sunken": [0/255,0/255,255/255],
         "wrinkled": [255/255,131/255,250/255],
         "non classé": [0, 0, 0]}

with open(json_path, 'r') as f:
  data = json.load(f)

#print(data)

def show_image(image, output):
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
    plt.savefig(output, bbox_extra_artists=(lgd,), bbox_inches='tight')

name="output/output"
i=0
for image in data:
    show_image(image, name+"_"+str(i))
    i += 1
