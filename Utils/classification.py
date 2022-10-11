import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os
import cv2
from cv2 import imshow
import numpy as np 

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

def get_coord(img_path,MARGIN):    
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
    if raw_detected_circles is not None:    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(raw_detected_circles))
        ROI_number = 0
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]    
            cor=str(a)+'_'+str(b)+'_'+str(r)
            coord.append(cor)
        cv2.waitKey(0)
    return coord

import pathlib
#dataset_url = ".keras\\datasets\\types recepteurs"
training_data_dir = "dataset\\types recepteurs"
training_data_dir = pathlib.Path(training_data_dir)

data_dir = "data\\1 S seabrai M\\analyse 2560x1920\\Upper"

img_for_output = "data\\1 S seabrai M\\analyse 2560x1920\\Upper\\1.bmp"


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

import matplotlib.pyplot as plt

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

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)
model=learnClasses1(train_ds,val_ds,nEpochs=32,cw = {})

liste = []
for subdir, dirs, files in os.walk(data_dir):    
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):            
            #print (filepath)
            liste.append(filepath)
#print(liste)

print("Veuillez patienter svp, le traitement peut durer quelques minutes.")

#new
def getLabels(trainedModel, nameList, color_mode='grayscale'):
    ll=[]
    for nn in nameList:
        nnn=nn
        ii=tf.keras.utils.load_img(nnn,color_mode=color_mode, target_size=(img_width, img_height))
        ll.append(ii)
    label_list=[]
    for im in ll:
        tt=tf.keras.preprocessing.image.img_to_array(im)
        tt=tf.expand_dims(tt,0)
        rr=trainedModel.predict(tt)
        ind=rr[0].argmax()
        label_list.append(ind)
    return label_list

list=getLabels(model, liste)

MARGIN:int = 20
    
coord=get_coord(img_for_output,MARGIN)

img = cv2.imread(img_for_output, cv2.IMREAD_COLOR)  
color = {"10": "255,131,250", "1": "124,205,124", "0": "238,59,59", "2": "255,255,0", "3": "142,229,238", "4": "255,211,155", "5": "0,201,87", "6": "255,110,180", "7": "145,44,238", "8": "238,121,66", "9": "135,206,255"}
#cv2.imshow("output", img)
for i in range(len(list)):        
    x=coord[i].split('_')        
    c=color[str(list[i])].split(',')    

    cv2.circle(img, ( int(x[0]), int(x[1] )), int(x[2]) , ( int(c[0]), int(c[1]), int(c[2]) ) , 2 )  

imS = cv2.resize(img, (1600, 900)) 
cv2.imshow("output", imS)  

cv2.waitKey(0)





