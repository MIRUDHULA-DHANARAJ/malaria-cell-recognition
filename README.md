# EX 4 Deep Neural Network for Malaria Infected Cell Recognition

## AIM
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.
## Problem Statement and Dataset
To develop a deep neural network model to classify microscopic cell images as either malaria-infected or uninfected. This is essential for aiding in the rapid and accurate diagnosis of malaria. The dataset includes a structured directory with images of both infected and uninfected cells. To enhance model accuracy, various techniques like hyperparameter tuning, model architecture adjustments, or transfer learning can be applied. Using an ImageDataGenerator, image data is loaded from the dataset directory, with configurations for data augmentation. The model's effectiveness is evaluated through metrics, a confusion matrix, and a plot of training vs. validation loss over epochs.
## Neural Network Model)
![image](https://github.com/user-attachments/assets/6dfe1dbf-7cad-477a-80ce-1ac23d5a3701)

## DESIGN STEPS

* STEP 1: Import Libraries: Load necessary libraries for data processing, visualization, and building the neural network.

* Step 2: Configure GPU: Set up TensorFlow to allow dynamic GPU memory allocation for efficient computation.

* Step 3: Define Dataset Paths: Set directory paths for training and testing data to organize the data flow.

* Step 4: Load Sample Images: Visualize example images from both infected and uninfected classes to better understand the dataset.

* Step 5: Explore Data Dimensions: Check image dimensions and visualize distribution patterns to assess consistency.

* Step 6: Set Image Shape: Define a consistent image shape to standardize inputs for the model.

* Step 7: Data Augmentation: Use ImageDataGenerator to apply transformations like rotation, shift, zoom, and flipping to prevent overfitting.

* Step 8: Build Model Architecture: Create a CNN with convolutional and pooling layers to extract important features.

* Step 9: Add Dense Layers: Flatten the model and add dense layers for deeper learning, finishing with a sigmoid output layer.

* Step 10: Compile Model: Set the loss function, optimizer, and evaluation metrics to prepare the model for training.

* Step 11: Train the Model: Train the model over a specified number of epochs using the training and validation datasets.

* Step 12: Evaluate and Visualize: Assess model performance, generate metrics, and plot training and validation loss to monitor progress.

## PROGRAM
### Name:  Mirudhula D

### Register Number: 212221230060
```py
# to share the GPU
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline

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

my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[200]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[200])

plt.imshow(para_img)

# Checking the image dimensions
dim1 = []
dim2 = []

for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)


model = models.Sequential()
# Write your code here

model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

batch_size = 16
help(image_gen.flow_from_directory)
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

results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )

losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names

model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)


test_image_gen.classes

predictions = pred_probabilities > 0.5

print("Mirudhula\n212221230060\n")
print(classification_report(test_image_gen.classes,predictions))

print("Mirudhula\n212221230060\n")
print(confusion_matrix(test_image_gen.classes,predictions))

import random as rand
list_dir=["Un Infected","parasitized"]
dir_=(rnd.choice(list_dir))
p_img=imread(train_path+'/'+dir_+'/'+os.listdir(train_path+'/'+dir_)[rnd.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(p_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/0daa370a-a018-4f60-be51-c1b9ad2c9fb4)


### Classification Report

![image](https://github.com/user-attachments/assets/a91d417c-7ad6-4c8e-931a-9665a614b099)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/f998d0a6-bf4a-4661-9457-14848e108b96)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a6d4cad3-2e71-4f60-9440-b88eb63ac9b6)


## RESULT
The model's performance is evaluated through training and testing, and it shows potential for assisting healthcare professionals in diagnosing malaria more efficiently and accurately.
