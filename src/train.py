# -- coding: utf-8 --
"""
Created on Thu Oct 13 14:12:11 2022

@author: notta
"""

import json
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import datasets, layers, models

from sklearn.utils import class_weight


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def learnLabelledSet(train_dict,datadir=".",model_v=1,nEpochs=20,flag_balanced=False):
    class_names=[]
 

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

    num_classes = len(classes)

    for key in list(train_dict.keys()):
        dataImg = cv2.imread(datadir+key, cv2.COLOR_BGR2GRAY)
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
            
    # limS  = proportion test / train
    
    limS = int(imcount*0.1)
    print(imcount, nrecep,limS)
    
    # mélange du dataset
    total_images, total_labels = unison_shuffled_copies(read_images, read_labels)

    test_images=total_images[:limS]
    test_labels=total_labels[:limS]
    
    train_images=total_images[limS:]
    train_labels=total_labels[limS:]
    

    class_w  = {}
    if (flag_balanced):
        class_w = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train_labels),y=train_labels)
        class_w_d = dict(enumerate(class_w.flatten(), 0))
 #   train_images=total_images[:]
 #   train_labels=total_labels[:]


    for i in range(len(classes)):
        print(i, classes[i],list(total_labels).count(i))
   
    if (model_v==1):
            model = tf.keras.Sequential([
              tf.keras.layers.Flatten(input_shape=(imsize, imsize)),
              tf.keras.layers.Dense(256, activation='relu'),
              tf.keras.layers.Dense(len(classes)),
            ])
    if (model_v==2):
             inputs = tf.keras.Input(shape = (imsize,imsize,1))
             # x = data_augmentation(inputs)
             x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
             x= tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
             x = tf.keras.layers.MaxPooling2D()(x)
             x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
             x = tf.keras.layers.MaxPooling2D()(x)
             x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
             x =  tf.keras.layers.MaxPooling2D()(x)
             x = tf.keras.layers.Flatten()(x)
             x = tf.keras.layers.Dense(128, activation='relu')(x)
             outputs = tf.keras.layers.Dense(num_classes,activation="softmax")(x) 
             model = tf.keras.Model(inputs=inputs,outputs = outputs)
                 

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=nEpochs,class_weight=class_w_d)


    loss, acc = model.evaluate(test_images, test_labels, verbose=2)



    print("Generate predictions for 3 samples")
    predictions = model.predict(test_images)
    print("predictions shape:", predictions.shape, test_labels.shape)

    predictCounted = np.array(test_labels)
    for i in range(len(test_labels)):
        predictCounted[i] = np.argmax(predictions[i])

    # perfromance par classe
    for i in range(len(classes)):
        print(i, classes[i], "obs:",list(test_labels).count(i),"   pred:",list(predictCounted).count(i))


    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model



def classifyImageSet(segmentedSet, model , labels, datadir="."):
    outSet = {}
    for pkey in segmentedSet.keys():
        outSet[pkey] = classifyImage(segmentedSet[pkey],model,labels, datadir+pkey)
    return outSet

def classifyImage(recepList, model , labels, imagePath):
    
    class_names=labels
   
    nrecep=0
    class_names={}
    for key in list(labels.keys()):
        class_names[labels[key][0]] = key
    print("model shape", model.layers[0].get_output_at(0).get_shape())
    imsize = int((np.sqrt(model.layers[0].get_output_at(0).get_shape()[-1])))
    imsize = 34

    #get nrecep and class and coords lists
    nrecep =len(recepList)

    #copter le nombre total de recepteurs : nrecep

    read_images = np.zeros((nrecep,imsize,imsize),dtype=np.uint8)
    read_labels=np.zeros(nrecep,dtype=np.uint8)
    imcount = 0


    dataImg = cv2.imread(imagePath, cv2.COLOR_BGR2GRAY)
    
 
    print(np.shape(dataImg))
    for receptor in recepList:
        x,y,rd = receptor["coordinates"]
        r=int(imsize/2)
        x=int(x)
        y=int(y)
        rd=int(rd)
        read_images[imcount,:,:]= dataImg[y-r:y+r,x-r:x+r]
        imcount = imcount+1

  
    predictions = model.predict(read_images)
    
    print("predictions shape:", predictions.shape, read_labels.shape)
    
    threshold=0.9
    
    for i,receptor in enumerate(recepList):
        receptor["label"] = class_names[np.argmax(predictions[i])]
        read_labels[i] = np.argmax(predictions[i])
        if(read_labels[i]>threshold):
            receptor["label"] = class_names[read_labels[i]]
        else:
            receptor["label"] =""

    return recepList
    
    


def loadImageLabeller(modelPath):
    return tf.keras.models.load_model(modelPath)
    





