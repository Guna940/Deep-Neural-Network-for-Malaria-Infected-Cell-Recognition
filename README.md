# Ex04-Deep-Neural-Network-for-Malaria-Infected-Cell-Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
 In order to conduct a series of experiments, publicly available malaria dataset was used. Data collection and data preprocessing techniques are discussed in the following subsequent subsections. Out of the series of experiments, we choose our best model in terms of both performances and effectiveness thereby, which is discussed in proposed model architecture subsections. Experimental details and experimental settings are discussed in training details subsections. Training of the models is discussed under three training procedure which are general training procedure, distillation training procedure and autoencoder training procedure, details are provided in the designated subsections.Malaria dataset contains 27,558 cell images classified into two groups called parasitized and uninfected cells, where each cell contains an equal number of instances. Data was taken from 150 P.

## Neural Network Model

![Screenshot (43)](https://user-images.githubusercontent.com/89703145/193622898-08f040af-04c7-4b54-9621-8981e71ca787.png)

## DESIGN STEPS

### STEP 1:

Define the directory for the dataset.Extract the dataset files if needed. Define the image Generator engine with the necessary parameters.

### STEP 2:

Pass the directory to the image generator. Define the model with the appropriate neurons.

### STEP 3:

Pass the training and validation data to the model. Plot  the necessary graphs.

## PROGRAM

```python3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GP
config.log_device_placement = True # to log device placement (on which device the o
sess = tf.compat.v1.Session(config=config)
set_session(sess)
%matplotlib inline
```
```python3
my_data_dir = '/home/ailab/hdd/dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)
```

```python3
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
 img = imread(test_path+'/uninfected'+'/'+image_filename)
 d1,d2,colors = img.shape
 dim1.append(d1)
 dim2.append(d2)
 ```
 
 ```python3
sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a m
                               height_shift_range=0.10, # Shift the pic height by a
                               rescale=1/255, # Rescale the image by normalzing it
                               shear_range=0.1, # Shear means cutting away part of
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with th
                               )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
```

```python3
model = models.Sequential([
    layers.Input((130,130,3)),
    layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32,kernel_size=3,activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32,kernel_size=3,activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(32,activation="relu"),
    layers.Dense(1,activation="sigmoid")])
model.compile(loss="binary_crossentropy", metrics='accuracy',optimizer="adam")
```

```python3
batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=4,validation_data=test_image_gen)
```

```python3
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
```

```python3
import random
list_dir=["uninfected","parasitized"]
dire=(random.choice(list_dir))
para_img= imread(train_path+
 '/'+dire+'/'+
 os.listdir(train_path+'/'+dire)[random.randint(0,10000)])
img = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))
img = img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5)
plt.title("Model Prediction: "+("Parasitized" if pred else "Uninfected")+"\nActual
plt.axis("off")
plt.imshow(img)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![WhatsApp Image 2022-10-03 at 9 54 46 AM](https://user-images.githubusercontent.com/89703145/193609144-ea325860-8445-4c1d-a64a-531f07fef041.jpeg)

### Classification Report

![Screenshot (40)](https://user-images.githubusercontent.com/89703145/193609209-a0760f06-b272-4ab9-abb9-f348bdc9941a.png)

### Confusion Matrix

![Screenshot (41)](https://user-images.githubusercontent.com/89703145/193609263-00f33336-2029-4423-b1a0-9bf4746517f4.png)

### New Sample Data Prediction

![Screenshot (42)](https://user-images.githubusercontent.com/89703145/193609317-081fe9dc-fa98-4410-a376-f33f82d90ff4.png)

## RESULT

Hence we have succesfully created Convolutional Neural Network for Malaria infected cell recognition and analyzed the performance of the new sample data also.
