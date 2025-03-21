import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import statsmodels.api as sm

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#train
mainDIR =os.listdir('/home/jin/Downloads/chest_xray/')
print(mainDIR)
train_folder= '/home/jin/Downloads/chest_xray/train/'
val_folder = '/home/jin/Downloads/chest_xray/val/'
test_folder = '/home/jin/Downloads/chest_xray/test/'
# train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_poic_address = train_n + norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneumonia picture title:', sic_pic)

#Load the images
norm_load = Image.open(norm_poic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1,2,2)
img_plt = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()
#let's build the CNN model

num_of_test_samples = 600
batch_size = 32

# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

# Test / Validation
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('/home/jin/Downloads/chest_xray/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_generator = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_set = test_generator.flow_from_directory('/home/jin/Downloads/chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# 간단한 신경망 모델 생성
# model_fin = Sequential([
#     Dense(32, input_shape=(10,), activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
# 모델 구조 요약
# model_fin.summary()

## HW 5-3. Build model
model_in = Input(shape=(64,64,3))
x = Flatten()(model_in)
x = Dense(128, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
model_out = Dense(1, activation = 'sigmoid')(x) #봐야할 사진이 2개라서 sigmoid. 3개 이상 softmax
model = Model(inputs=model_in, outputs=model_out)
model.summary()

#Model Compilation
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'],)
ann_model = model.fit(training_set, steps_per_epoch = 163, epochs = 10, validation_data = test_set, validation_steps = 624)

import pickle
with open('ann_pneumonia_history', 'wb') as pf:
    pickle.dump(ann_model, pf)

model.save('ann_pneumonia.keras')