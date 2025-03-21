import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- defineevaluation metrics


mainDIR = os.listdir('/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray')
print(mainDIR)
train_folder= '/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/train/'
val_folder = '/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/val/'
test_folder = '/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/test/'
# train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)


# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)
#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()
# let's build the CNN model

#HW 3 start
model_in = Input(shape=(64,64,3))
x = Flatten()(model_in)
x = Dense(128, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
model_out = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs=model_in, outputs=model_out)
model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#HW 3 end

num_of_test_samples = 600
batch_size = 32

#training 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/train',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

#Test / vaildation

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_set = train_datagen.flow_from_directory('/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/train',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

ann_model = model.fit(training_set,
                      steps_per_epoch = 163,
                      epochs = 1,
                      validation_data = test_set,
                      validation_steps = 624) 

import pickle
with open('ann_pneumonia_history', 'wb') as pf:
    pickle.dump(ann_model, pf)
    
model.save('ann_penumonia.keras')
