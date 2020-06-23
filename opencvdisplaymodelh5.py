# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:25:06 2020

@author: NETIFUL
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keyboard
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import dlib
from imutils import face_utils

heu = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create model
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

# Get all existing models
listModelDirectories = os.listdir("models")
listModels = []
for direc in listModelDirectories:
    for model in os.listdir(os.path.join("models", direc)):
        if os.path.isfile(os.path.join("models", direc, model)):
            listModels.append(os.path.join("models", direc, model))
        elif os.path.isdir(os.path.join("models", direc, model)):
            for file in os.path.join("models", direc, model):
                if os.path.isfile(os.path.join("models", direc, model, file)):
                    listModels.append(os.path.join("models", direc, model, file))
if len(listModels) == 0:
    sys.exit("No existing model found")
# Let user choose model
print("Choose a model (between 1 and " + str(len(listModels)) + ") : ")
for model in listModels:
    print(str(listModels.index(model) + 1) + " : " + model)
reponse = ""
while True:
    reponse = input()
    try:
        reponse = int(reponse)
    except:
        pass
    if reponse < 1:
        print("Number too small !")
    elif reponse > len(listModels):
        print("Number too big !")
    else:
        break

model.load_weights(listModels[reponse - 1])

listeOutputs = [layer.output for layer in model.layers]
# Prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "ANGRY", 1:"DISGUSTED", 2: "SCARED", 3: "HAPPY", 4: "NEUTRAL", 5: "SAD", 6: "SURPRISED"}

# Start the webcam feed
# Check for usb camera. If none, check for integrated camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No camera detected.")
    print("Do you want to try using the default camera ? y/n")
    while True:
        if keyboard.is_pressed('y') or keyboard.is_pressed('n'):
            break
        continue
    if keyboard.is_pressed('y'):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("No camera found")
        else:
            print("Camera found")
    elif keyboard.is_pressed('n'):
        sys.exit("User exit")

# Initialize all haarcascade files and the facial landmarks detector
profilCasc = cv2.CascadeClassifier("haarcascades/haarcascades/haarcascade_profileface.xml")
facecasc = cv2.CascadeClassifier('haarcascades/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
time_start = 0
time_end = 0
nbDistractions = 0
time_start_film = time.time()

# Start filming
while True:
        posFace = 0
        heightFace = 0
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        profiles = profilCasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        landmarks = detector(gray, 1)

        for (x, y, w, h) in faces: # Emotion detection
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            heightFace = h + 10
            posFace = y
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            label = emotion_dict[maxindex]
            cv2.putText(frame, label , (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if maxindex == 3 :
                heu += 1
        for (x,y,w,h) in profiles: # Detect profile faces and if user is distracted
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (128, 128, 0), 2)
            if time_start == None:
                time_start = time.time()
            else:
                time_end = time.time()
                diff_time = time_end - time_start
                if diff_time > 3:
                    cv2.putText(frame, "DISTRACTED", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        if len(profiles) == 0: # Check if user is distracted (repetitive head rotation)
            if time_start:
                nbDistractions += 1
                time_distracted = time_start - time_start_film
                freqDistraction = time_distracted // nbDistractions
                if freqDistraction <= 10:
                    cv2.putText(frame, "DISTRACTED", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            time_start = None
        for rect in landmarks: # Detect tiredness with mouth opening (yawning)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            (x1, y1) = shape[62]
            (x2, y2) = shape[66]
            diffY = y2 - y1
            if diffY >= (heightFace // 8):
                cv2.putText(frame, "TIRED", (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,0,128), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()