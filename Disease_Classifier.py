# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 08:43:25 2020

@author: Ishan Nilotpal
"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

IMAGE_SIZE = [224,224]

train_path = 'C:/Users/hp/Documents/Machine Learning Projects/cotton Disease Classifier/Data/train'
test_path = 'C:/Users/hp/Documents/Machine Learning Projects/cotton Disease Classifier/Data/test'

resnet = InceptionV3(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
for layer in resnet.layers:
    layer.trainable=False
    
folders = glob('C:/Users/hp/Documents/Machine Learning Projects/cotton Disease Classifier/Data/train/*')

x = Flatten()(resnet.output)

prediction = Dense(len(folders),activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=prediction)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen =  ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,target_size=(224,224),batch_size=32,class_mode='categorical')
 
test_set = test_datagen.flow_from_directory(test_path,target_size=(224,224),batch_size=32,class_mode='categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

model.save('InceptionV3.h5')