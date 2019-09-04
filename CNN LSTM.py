# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:23:43 2019

@author: Muhhamed Tarek
"""









import time
import os
import cv2
#import pandas as pd
import numpy as np
#from PIL import Image
#import tensorflow as tf
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten ,TimeDistributed ,LSTM 
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
#from keras.losses import categorical_crossentropy
#from keras.optimizers import Adam
#from keras.regularizers import l2
#from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
#from keras.models import load_model









lst = []
X=np.array([])
Y = []
start_time = time.time()

for subdir, dirs, files in os.walk("DATASET"):
  print(subdir, dirs)
  for file in files:
        fra=[]
        cap = cv2.VideoCapture(subdir+"\\"+file)
        cap.set(cv2.CAP_PROP_POS_FRAMES,(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)-44)
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT),"---",cap.get(cv2.CAP_PROP_POS_FRAMES))
        for o in range(88):
            ret , frame = cap.read()
            
            gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            #for (x, y, w, h) in faces:
             
             #roi_gray = gray[y:y + h, x:x + w]
             #cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (56, 56)), -1), 0)
             #cv2.imshow("F",cv2.resize(roi_gray, (120, 120)))
             #cv2.waitKey(0)
            if o%2==0: 
                fra.append(cv2.resize(gray, (56, 56)))
            
        
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        file = int(file[7:8]) - 1 
        arr = fra, file
        
        lst.append(arr)
        Y.append(file)
        X = np.append(X,np.array(fra) )
        print(np.array(lst).shape)
      # If the file is not valid, skip it






print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))









print ("building model…")

model = Sequential()
# define CNN model
model.add(TimeDistributed(Conv2D(32, (3, 3), activation = 'relu') ,input_shape= (44,56,56,1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))

# define LSTM model
model.add(LSTM(256,activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
optimizer='adam', metrics=['accuracy'])

print (model.summary())
batch_size=44
nb_epoch=100
print (len(X))
print (len(Y))


X = X.reshape(-1,44,56,56,1)
X_train, X_test, y_train, y_test =train_test_split( X, Y, test_size=0.33, random_state=42)


X_train = np.array(X_train)
y_train =   np.array(y_train)
X_test =  np.array(X_test)

y_test = np.array(y_test)

print ("printing final shapes…")
print ("X_train: ", X_train.shape)
print ("y_train: ", y_train.shape)
print ("X_test: ", X_test.shape)
print ("y_test: ", y_test.shape)
print

print("Train…")

model.fit(X_train, y_train, batch_size=batch_size, epochs = nb_epoch,
validation_data=(X_test, y_test) )

print("Evaluate…")
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
show_accuracy=True)
print("Test score:", score)
print("Test accuracy:", acc)

