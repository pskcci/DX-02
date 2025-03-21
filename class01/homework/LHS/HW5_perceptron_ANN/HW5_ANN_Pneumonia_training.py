'''ANN with Mnist 실습 2'''

import numpy as np # for linear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images

# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,\
Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix
# <- define evaluation metrics

mainDIR = os.listdir('/home/rg/Downloads/chest_xray')
MainDIR = '/home/rg/Downloads/chest_xray/' #위치 설정용 변수
print(mainDIR)
print('\nmainDIR :\n', MainDIR)
train_folder = MainDIR + 'train/'
val_folder = MainDIR + 'val/'
test_folder = MainDIR + 'test/'
# train path
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'
print('train_n :\n', train_n, '\ntrain_p :\n', train_p)

#Normal pic
#삭제######length_train_p = len(os.listdir(train_p))
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title:\n',norm_pic)
norm_pic_address = train_n + norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address).convert("L")   #.convert("L") -> 흑백으로
pneu_load = Image.open(sic_address).convert("L")
#Let's plt these images
f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(pneu_load)
a2.set_title('Pneumonia')
#plt.savefig('chest_xray.jpg', 'rgb')
plt.show()
# let's build the CNN model

## HW 5-3. Build model
model_in = Input(shape = (64, 64, 3))
x = Flatten()(model_in)
x = Dense(128, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
model_out = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = model_in, outputs = model_out)
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])

'''
#cnn = Sequential()
#Convolution
model_in = Input(shape = (64, 64, 3))
model = Flatten()(model_in)
# Fully Connected Layers
model = Dense(activation = 'relu', units = 128) (model)
model = Dense(activation = 'sigmoid', units = 1)(model)
# Compile the Neural network
model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
'''


num_of_test_samples = 600
batch_size = 32
# Training
# 이미지 데이터를 증강, 증강된 데이터를 CNN 모델에 전달하여 훈련을 진행
# Test(테스트) / Validation(검증)
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Image normalization.
training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

# training_set = test_set
test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')



'''
validation_generator = test_datagen.flow_from_directory('/home/rg/Downloads/chest_xray/val/',
target_size=(64, 64),
batch_size=32,
class_mode='binary')
test_set = test_datagen.flow_from_directory('/home/rg/Downloads/chest_xray/test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
model.summary()
'''

# HW 5-4. fit(training)
ann_model = model.fit(training_set,
                      steps_per_epoch = 163,
                      epochs = 10,
                      validation_data = test_set,
                      validation_steps = 624)
test_accu = model.evaluate(test_set,steps=624)

import pickle
with open('ann_pneumonia_history', 'wb') as pf:
    pickle.dump(ann_model, pf)

model.save('ann_pneumonia.keras')

'''
print('The testing accuracy is :',test_accu[1]*100, '%')
Y_pred = model.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)
'''
