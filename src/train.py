# -- coding: utf-8 --
"""
Created on Thu Oct 13 14:12:11 2022

@author: notta
"""

import json
import numpy as np
import tensorflow as tf
import cv2

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class_names=[]

file=open('../Utils/trainingSet.json')
train_dict=json.load(file)

from tensorflow.keras import datasets, layers, models
nrecep=0
class_names=[]

imsize = 34
#get nrecep and class and coords lists
for key in list(train_dict.keys()):
    nrecep+=len(train_dict[key])
    for sub_key in list(train_dict[key]):
        class_names.append(sub_key['label'])

#copter le nombre total de recepteurs : nrecep


read_images = np.zeros((nrecep,imsize,imsize),dtype=np.uint8)
read_labels=np.zeros(nrecep,dtype=np.uint8)
imcount = 0

classes=list(np.unique(class_names))



for key in list(train_dict.keys()):
    dataImg = cv2.imread("../"+key, cv2.COLOR_BGR2GRAY)
   # dataImg2 =  cv2.cvtColor(dataImg, cv2.COLOR_BGR2GRAY)
   # print(key,np.shape(dataImg),np.shape(dataImg2))
    for receptor in  train_dict[key]:

        x,y,rd = receptor["coordinates"]
        r=int(imsize/2)
        x=int(x)
        y=int(y)
        rd=int(rd)
        read_images[imcount,:,:]=dataImg[y-r:y+r,x-r:x+r]
        read_labels[imcount] = classes.index(receptor["label"])

        imcount = imcount+1

limS = int(imcount/4)
print(imcount, nrecep,limS)

total_images, total_labels = unison_shuffled_copies(read_images, read_labels)

test_images=total_images[:limS]
train_images=total_images[limS:]



test_labels=total_labels[:limS]
for i in range(len(classes)):
    print(i, classes[i],list(total_labels).count(i))
train_labels=total_labels[limS:]

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(imsize, imsize)),
        tf.keras.layers.Dense(1024, activation='relu'),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(classes)),
        ])

model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)


loss, acc = model.evaluate(test_images, test_labels, verbose=2)


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_images)
print("predictions shape:", predictions.shape, test_labels.shape)

predictCounted = np.array(test_labels)
for i in range(len(test_labels)):
    predictCounted[i] = np.argmax(predictions[i])

for i in range(len(classes)):
    print(i, classes[i], list(test_labels).count(i),list(predictCounted).count(i))

    
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))




config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"])
print(config["layers"][0]["config"])


'''
trainingSetPath = "../Utils/trainingSet.json"

activationSetPath = "../Utils/trainingSet.json"


#model = learnLabelledSet(json.load(open(trainingSetPath))
#model.save('AI/trained_model.h5')


imagePathSet = json.load(open(trainingSetPath))

segmentedSet = segmentImageSet(imagePathSet)

classifiedSet = classifyImageSet(model,segmentedSet)

for i,image in enumerate(classifiedSet):
    generateImage(image, name+"_"+str(i))
'''
# Sortie de la surface de l'antenne
# Sortie tableau   "espece;NbWrinked..;DiamMoyenWrinkled;NbFlat...."





