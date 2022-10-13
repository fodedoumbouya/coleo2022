# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import json

json_path = 'trainingSet.json'

with open(json_path, 'r') as f:
  data = json.load(f)

#print(data)

def show_image(image, output):
    ax = plt.gca()
    ax.cla()
    for circle in range(len(data[image])):
        x = data[image][circle]['coordinates'][0]
        y = data[image][circle]['coordinates'][1]
        size = data[image][circle]['coordinates'][2]
        circle = plt.Circle((x, y), size, color='r', fill=False)
        ax.add_patch(circle)
    image = "../" + image
    img = mpimg.imread(image)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.savefig(output, dpi=300)

name="output/output"
i=0
for image in data:
    show_image(image, name+"_"+str(i))
    i += 1
