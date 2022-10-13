# -- coding: utf-8 --
"""
Created on Thu Oct 13 14:12:11 2022

@author: notta
"""

import json
import numpy as np
import tensorflow as tf
import cv2

class_names=[]
coords=[]

file=open('Utils/trainingSet.json')
train_dict=json.load(file)


nrecep=0
class_names=[]

#get nrecep and class and coords lists
for key in list(train_dict.keys()):
    nrecep+=len(train_dict[key])
    for sub_key in list(train_dict[key]):
        class_names.append(sub_key['label'])
        coords.append(sub_key['coordinates'])

#copter le nombre total de recepteurs : nrecep

train_images = np.zeros((nrecep,28,28),dtype=np.uint8)
train_labels=np.zeros(nrecep)
imcount = 0

classes=np.unique(class_names)


for key in list(train_dict.keys()):
    dataImg = cv2.imread(key, cv2.IMREAD_COLOR)
    for receptor in  train_dict[key]:
        if imcount<nrecep:
            x,y,rd = receptor["coordinates"]
            r=int(28/2)
            x=int(x)
            y=int(y)
            rd=int(rd)
            train_images[imcount,:,:]=dataImg[y-r:y+r,x-r:x+r][:,:,2]
            train_labels[imcount] = np.where(classes==receptor["label"])[0][0]
            imcount = imcount+1

test_images=train_images[:1000]
train_images=train_images[1000:]

test_labels=train_labels[:1000]
train_labels=train_labels[1000:]


model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(11),

        ])





model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

'''
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
'''




model.save('AI/trained_model.h5')