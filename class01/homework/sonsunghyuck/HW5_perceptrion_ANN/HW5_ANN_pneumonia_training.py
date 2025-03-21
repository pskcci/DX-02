#MEDICAL_IMAGE_CLASSIFICATION1


import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics


# 학습할 이미지들의 파일경로를 가져오기

# mainDIR = os.listdir('/home/sunghyuck/git-training/project-y/workspace/HW_5_perceptron_ANN/HW5_ANN_pneumonia_training.py')

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

# train path
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

sic_pic_address = train_p + norm_pic

# 학습할 이미지를 불러오고 PLOT

# Load the images
norm_load = Image.open(norm_pic_address).convert("L")
sic_load = Image.open(sic_address).convert("L")

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.savefig('chest_xray.jpg')
plt.show()

# let's build the CNN model


## HW 5-3 Build model
model_in = Input(shape=(64,64,3))
x = Flatten()(model_in)
x = Dense(128, activation ='relu')(x)
x = Dense(32, activation ='relu')(x)
model_out  = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs=model_in, outputs=model_out)
model.summary() # 
model.compile(optimizer='adam',loss ='binary_crossentropy',metrics =['accuracy'])

number_of_test_samples = 600
batch_size = 32

# Training
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory(train_folder, target_size=(64,64),batch_size = batch_size, class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip=True)

test_set = test_datagen.flow_from_directory(test_folder, target_size=(64,64),batch_size = batch_size, class_mode = 'binary')

ann_model = model.fit(training_set, steps_per_epoch = 163, epochs = 10 , validation_data = test_set, validation_steps =624)

import pickle
with open('ann_pneumonia_history','wb') as pf:
    pickle.dump(ann_model,pf)
    
model.save('ann_pneumonia.keras')

plt.plot(ann_model.history['accuracy'])
plt.plot(ann_model.history['val_accuracy'])
plt.title('모델 정확도')
plt.ylabel('정확도')
plt.xlabel('에포크')
plt.legend(['훈련셋', '검증셋'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show()

# 손실 그래프 출력
plt.plot(ann_model.history['val_loss'])
plt.plot(ann_model.history['loss'])
plt.title('모델 손실')
plt.ylabel('손실')
plt.xlabel('에포크')
plt.legend(['훈련셋', '검증셋'], loc='upper left')
plt.savefig('train_loss.png')
plt.show()

# 25개의 추가 이미지를 랜덤하게 선택하여 한 페이지에 출력
num_images = 25 # 사용하여 출력할 이미지 수를 설정
plt.figure(figsize=(20, 20)) # 전체 그림의 크기를 설정

for i in range(num_images): # 전체 그림의 크기를 설정
    # 랜덤하게 Normal 또는 Pneumonia 이미지 선택
    if np.random.rand() < 0.5: # Normal 또는 Pneumonia 이미지를 랜덤하게 선택
        image_path = os.path.join(train_n, np.random.choice(os.listdir(train_n))) # 사용하여 랜덤한 이미지 파일 경로를 생성
        label = 'Normal'
    else:
        image_path = os.path.join(train_p, np.random.choice(os.listdir(train_p)))
        label = 'Pneumonia'

    # 이미지 불러오기 및 출력
    img = Image.open(image_path).convert("L")
    plt.subplot(5, 5, i + 1)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')

plt.savefig('25_random_images.png')
plt.show()

