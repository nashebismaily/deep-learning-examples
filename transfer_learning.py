#!/usr/bin/env python
# coding: utf-8


#Import libraries

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from IPython.display import display # Library to help view images
from PIL import Image # Library to help view images
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Library for data augmentation
import os, shutil # Library for navigating files
np.random.seed(42)


# Specify the base directory where images are located.  You need to save your data here.
base_dir = '/storage/msds686/cats_and_dogs/data/'


# Specify the traning, validation, and test dirrectories.  
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# Specify the the classess in the training, validataion, and test dirrectories
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


# We need to normalize the pixels in the images.  The data will 'flow' through this generator.
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#set Epoch
epoch = 50

# Since the file images are in a dirrectory we need to move them from the dirrectory into the model.  
# Keras as a function that makes this easy. Documentaion is here: https://keras.io/preprocessing/image/

train_generator = train_datagen.flow_from_directory(
    train_dir, # The directory where the train data is located
    target_size=(150, 150), # Reshape the image to 150 by 150 pixels. This is important because it makes sure all images are the same size.
    batch_size=20, # We will take images in batches of 20.
    class_mode='binary') # The classification is binary.

validataion_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


# Now we build the model.

backend.clear_session()
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam',
               loss = 'binary_crossentropy',
               metrics = ['accuracy'])

history = model.fit_generator( # The image data must come from the image generator that takes the images from the correct dirrectory. https://keras.io/models/sequential/
    train_generator, # Images are taken from the train_generator
    steps_per_epoch=100, # The number of steps from the train_generator before one epoch if finished.  
                       # 100 steps * 20 batch size in train generator = 2000 training images per epoch. This way each traning image will be sampled once per epoch.
    epochs=epoch, # Train data for 50 epochs
    validation_data=validataion_generator, # Take data from the validataion generator
    validation_steps=50, # 50 steps * 20 batch size in validation generator = 1000 validation images per epoch
    verbose = 2,
    callbacks = [EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights=True)])
    


test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50) # Test images are in a dirrectory so they must flow from dirrectory. 
                                                                           # 50 steps * 20 batch size in test generator = 1000 test images per epoch
print('base_model_test_acc:', test_acc)


print('The above model came out with about 70% accuracy. Which is pretty good considering we only used 2000 training images!')
 

# Now lets improve using data augmentation.
# Data augmentation allows us to randomally transform images before sending them to the model for training.  
# The random transformation changes the images into 'new' images and allows for an increase in traning data without have additional images. 
# https://keras.io/preprocessing/image/ 

train_datagen2 = ImageDataGenerator(
    rescale=1./255,# The image augmentaion function in Keras
    rotation_range=40, # Rotate the images randomly by 40 degrees
    width_shift_range=0.2, # Shift the image horizontally by 20%
    height_shift_range=0.2, # Shift the image veritcally by 20%
    zoom_range=0.2, # Zoom in on image by 20%
    horizontal_flip=True, # Flip image horizontally 
    fill_mode='nearest') # How to fill missing pixels after a augmentaion opperation


test_datagen2 = ImageDataGenerator(rescale=1./255) #Never apply data augmentation to test data. You only want to normalize and resize test data. 

train_generator2 = train_datagen2.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validataion_generator2 = train_datagen2.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_generator2 = test_datagen2.flow_from_directory( # Resize test data
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


backend.clear_session()
model_aug = models.Sequential()

model_aug.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)))
model_aug.add(layers.MaxPool2D((2,2)))
model_aug.add(BatchNormalization())
model_aug.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model_aug.add(layers.MaxPool2D((2,2)))
model_aug.add(BatchNormalization())
model_aug.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model_aug.add(layers.MaxPool2D((2,2)))
model_aug.add(BatchNormalization())
model_aug.add(layers.Flatten())
model_aug.add(layers.Dense(64, activation='relu'))
model_aug.add(layers.Dropout(0.5))

model_aug.add(layers.Dense(1, activation='sigmoid'))

model_aug.compile(optimizer = 'adam',
               loss = 'binary_crossentropy',
               metrics = ['accuracy'])

history = model_aug.fit_generator(
    train_generator2,
    steps_per_epoch=400,
    epochs=epoch,
    validation_data=validataion_generator2,
    validation_steps=50,
    verbose = 2,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights= True)])

test_loss, test_acc = model_aug.evaluate_generator(test_generator2, steps = 50)

print('Data_Augmentation_test_acc:', test_acc)

print('An inprovment, but not a suprise. Having more data helps our accuracy.') 


# But why go through the hassle of building our own CNN when there are many networks that have used powerful GPUs to classify images? 
# We can use the weights of their models and apply them to something as simple as classiying a dog and cat.  
# We will use weights of the VGG16 CNN that was trained using ImageNet data.  https://keras.io/applications/

from tensorflow.keras.applications import VGG16 # Import the VGG16 library. 

backend.clear_session()
conv_base = VGG16 (weights = 'imagenet', #Useing the VGG66 CNN that was trained on ImageNet data.  
                  include_top = False, # We are using our own classification (dog or cat) and not the ImageNet multiclassification. So include top = false.
                  input_shape = (150, 150, 3))



print('VGG base summary:', conv_base.summary()) # View the VGG16 model architecture.


conv_base.trainable = False # Freeze the VGG16 weights.

print('VGG base summary after frozen:', conv_base.summary())

modelvgg16 = models.Sequential()
modelvgg16.add(conv_base) # Add the VG166 weights
modelvgg16.add(layers.Flatten())
modelvgg16.add(layers.Dense(512, activation = 'relu'))
modelvgg16.add(layers.Dense(1, activation = 'sigmoid'))

# We will still use the same data augmentation from above

modelvgg16.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelvgg16.fit_generator(
    train_generator2,
    steps_per_epoch=200,
    epochs=epoch,
    validation_data=validataion_generator2,
    validation_steps=50,
    verbose = 2,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)])


test_loss, test_acc = modelvgg16.evaluate_generator(test_generator2, steps = 50)

print('VGG_frozen_test_acc:', test_acc)


print('Our model keeps getting better.') 

#We might want to add a few 2D convolutions and max pooling layers before the dense layer.
# Or we can train the last three 2D convolution layers and maxpooling layer of the VGG16 model.  


# Now we can freeze all the VGG weights except the last few, and train those before adding it to our dense layer.
backend.clear_session()

vgg16_base_2 = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))

# Here we freeze all the layers except the last 4.
for layer in vgg16_base_2.layers[:-4]:
  layer.trainable = False
for layer in vgg16_base_2.layers:
  print(layer, layer.trainable)


print('VGG model 2 summary:', vgg16_base_2.summary())

modelvgg16_train = models.Sequential()
modelvgg16_train.add(vgg16_base_2)
modelvgg16_train.add(layers.Flatten())
modelvgg16_train.add(layers.Dense(512, activation = 'relu'))
modelvgg16_train.add(layers.Dense(1, activation = 'sigmoid'))


# We will still use the same data augmentation from above

modelvgg16_train.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelvgg16_train.fit_generator(
    train_generator2,
    steps_per_epoch=200,
    epochs=epoch,
    validation_data=validataion_generator2,
    verbose = 2,
    callbacks=[EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True)])


test_loss, test_acc = modelvgg16_train.evaluate_generator(test_generator2, steps = 50)


print('VGG_train_test_acc:', test_acc)
print('The best model yet')

# Your Turn
# Build and optimize another model. 
# Use weights from a different pretrained network (ie, ResNet, Inception, etc. not VGG. https://keras.io/applications/) from the Keras library. Optimize the model by adding additional layers, regularization, change activaction, adjust data augmentation etc.


from keras.applications.inception_v3 import InceptionV3

backend.clear_session()

inception_base =  InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

print('inception base summary:', inception_base.summary()) 

inception_base.trainable = False

print('inception base summary after frozen:', inception_base.summary())


modelinception = models.Sequential()
modelinception.add(inception_base)
modelinception.add(layers.GlobalAveragePooling2D())
modelinception.add(layers.Dense(256, activation='relu'))
modelinception.add(layers.Dropout(.25))
modelinception.add(layers.BatchNormalization())
modelinception.add(layers.Dense(1, activation='sigmoid'))

modelinception.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelinception.fit_generator(
    train_generator2,
    steps_per_epoch=32,    
    epochs=epoch,
    validation_data=validataion_generator2,
    validation_steps=50,
    verbose = 2,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)])

test_loss, test_acc = modelinception.evaluate_generator(test_generator2, steps = 50)

print('inception_test_acc:', test_acc)
