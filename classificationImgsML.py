# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:51:03 2020

@author: NETIFUL

Database used : https://datarepository.wolframcloud.com/resources/FER-2013

Source: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Citation: I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner, W. Cukierski, Y. Tang, D. Thaler,
          D.-H. Lee, Y. Zhou, C. Ramaiah, F. Feng, R. Li, X. Wang, D. Athanasakis, J. Shawe-Taylor, M. Milakov, J. Park,
          R. Ionescu, M. Popescu, C. Grozea, J. Bergstra, J. Xie, L. Romaszko, B. Xu, Z. Chuang, and Y. Bengio.
Challenges in representation learning: A report on three machine learning contests.
Neural Networks, 64:59--63, 2015. Special Issue on "Deep Learning of Representations"

PUBLISHER INFORMATION
Contributed By: Wolfram Research
Publisher of Record: Wolfram Research
"""


import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import keyboard
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'datasets/training'
val_dir = 'datasets/test'

# Get number of images for training and test data
nbImgTrain = 0
nbImgTest = 0
for direc in os.listdir(train_dir):
    nbImgTrain += len(os.listdir("datasets/training/" + direc))
for direc in os.listdir(val_dir):
    nbImgTest += len(os.listdir("datasets/test/" + direc))

num_train = nbImgTrain
num_val = nbImgTest
batch_size = 30
num_epoch = 15

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

reponse = None
print("Do you want to load a model ? y/n")
while True:
    if keyboard.is_pressed('n'):
        reponse = 'n'
        break
    elif keyboard.is_pressed('y'):
        reponse = 'y'
        break
if reponse == 'y': # Load model
    listModelDirectories = os.listdir("models")
    listModels = []
    for direc in listModelDirectories:
        for model in os.listdir(os.path.join("models", direc)):
            if os.path.isfile(os.path.join("models", direc, model)) and ".h5" in model:
                listModels.append(os.path.join("models", direc, model))
            elif os.path.isdir(os.path.join("models", direc, model)):
                for file in os.path.join("models", direc, model):
                    if os.path.isfile(os.path.join("models", direc, model, file)):
                        listModels.append(os.path.join("models", direc, model, file))
    if len(listModels) == 0:
        sys.exit("No existing model found")
    print("Choose a model (between 1 and " + str(len(listModels)) + ") : ")
    for model in listModels:
        print(str(listModels.index(model) + 1) + " : " + model)
    reponse = ""
    while True:
        reponse = input()
        try:
            reponse = int(reponse)
        except ValueError:
            print("Illegal value !")
            continue
        if reponse < 1:
            print("Number too small !")
        elif reponse > len(listModels):
            print("Number too high !")
        else:
            break
    model.load_weights(listModels[reponse - 1])
elif reponse == 'n': # Create new model
    print("Creating new model...")

# Train the model
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=200,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=100)
plot_model_history(model_info)
modelName = input("Enter a name to save your model (no need to add .h5 at the end. Default name is model.h5): ")
if modelName == "":
    modelName = "model"
# Save model
try:
    os.mkdir(os.path.join("models", modelName))
except FileExistsError:
    pass
model.save_weights(os.path.join("models", modelName, modelName + '.h5'))