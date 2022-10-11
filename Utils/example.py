#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:53:36 2022

@author: rbaggio
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import cv2

def learnClasses1(train_ds,val_ds,nEpochs=32,cw = {}):
          
   earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 20, verbose = 1,restore_best_weights = True)
   
  
   inputs = tf.keras.Input(shape = (img_height,img_width,1))
   x = data_augmentation(inputs)
   x = tf.keras.layers.Rescaling(1.0 / 255)(x)
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
  
   optimizer = 'adam'
  
   # model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
   model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
   model.summary()
   history = model.fit(train_ds, epochs=nEpochs, validation_data=val_ds, verbose=1,class_weight=cw,callbacks=[earlystop])

   hd = history.history
   lv = hd['loss']
   lv1 = hd['val_loss']
   acc = hd['accuracy']
   acc1 = hd['val_accuracy']
   epochs = range(1,len(lv)+1)
   plt.figure(1)
   plt.clf()
   plt.subplot(2,1,2)
   plt.plot(epochs,acc,'o-')
   plt.plot(epochs,acc1,'o-')
   plt.subplot(2,1,1)
   plt.plot(epochs,lv,'o-')
   plt.plot(epochs,lv1,'o-')
   return model

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ]
)

training_data_dir = "dataset/types_recepteurs"

training_data_dir = pathlib.Path(training_data_dir)

data_dir = "dataset/Upper"

img_for_output = "dataset/Upper/1.bmp"

#Create dataset

image_count = len(list(training_data_dir.glob('*\*.png')))
print(image_count)

batch_size = 32
img_height = 60
img_width = 60

train_ds = tf.keras.utils.image_dataset_from_directory(
  training_data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  color_mode="grayscale",
  label_mode="int",
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  training_data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  color_mode="grayscale",
  label_mode="int",
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
plt.show()
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

### Training the Model
num_classes = len(class_names)
model=learnClasses1(train_ds,val_ds,nEpochs=32,cw = {})

#Segmentation
color = {"10": "255,131,250", "1": "124,205,124", "0": "238,59,59", "2": "255,255,0", "3": "142,229,238", "4": "255,211,155", "5": "0,201,87", "6": "255,110,180", "7": "145,44,238", "8": "238,121,66", "9": "135,206,255"}

def get_coord_and_label(img_path,trainedModel,MARGIN):    
    # Read image.
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))    
    # Apply Hough transform on the blurred image.
    raw_detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 34, param2 = 30, minRadius = 1, maxRadius = 24)
    #print(raw_detected_circles)    
    # Draw circles that are detected.
    coord=[]
    label_list=[]
    if raw_detected_circles is not None:    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(raw_detected_circles))
        ROI_number = 0
#         #create an image
        
#         # ---------------------- Enregistrement des sous-images ---------------------- #
#         x,y,w,h = cv2.boundingRect(c)
#         ROI = image[y:y+h, x:x+w]
#         cv2.imwrite('output/ROI_{}.png'.format(ROI_number), ROI)
#         ROI_number += 1
# # ---------------------------------------------------------------------------- #

#         valid +=  1
#         ((x, y), r) = cv2.minEnclosingCircle(c)
#         cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        
        
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            rd=int(1.1*r)
            im = gray[b-rd:b+rd,a-rd:a+rd]
            im=cv2.resize(im, (60, 60)) 
            tt=tf.keras.preprocessing.image.img_to_array(im)
            tt=tf.expand_dims(tt,0)
            rr=trainedModel.predict(tt)
            ind=rr[0].argmax()
            label_list.append(ind)
            cor=str(a)+'_'+str(b)+'_'+str(r)
            coord.append(cor)
            ################################
        cv2.waitKey(0)
    return coord,label_list
MARGIN:int = 20
coord,labels=get_coord_and_label(img_for_output,model,MARGIN)


plt.figure(figsize=(10, 10))
img = cv2.imread(img_for_output, cv2.IMREAD_COLOR)  
color = {"10": "255,131,250", "1": "124,205,124", "0": "238,59,59", "2": "255,255,0", "3": "142,229,238", "4": "255,211,155", "5": "0,201,87", "6": "255,110,180", "7": "145,44,238", "8": "238,121,66", "9": "135,206,255"}
#cv2.imshow("output", img)
# for i in range(len(labels)):        
#     x=coord[i].split('_')        
#     c=color[str(labels[i])].split(',')    
#     cv2.circle(img, ( int(x[0]), int(x[1] )), int(x[2]) , ( int(c[0]), int(c[1]), int(c[2]) ) , 2 )  

# imS = cv2.resize(img, (1600, 900)) 
# cv2.imshow("output", imS)  
# cv2.waitKey(0)

#plt.imshow("output", cv2.cvtColor(imS, cv2.COLOR_BGR2RGB))  