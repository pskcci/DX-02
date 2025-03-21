# # test_1

# import numpy as np  # 선형대수 연산을 위한 라이브러리
# import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리
# import os
# from PIL import Image  # 이미지를 읽기 위한 라이브러리
# # Keras 라이브러리 <- CNN을 사용하기 위한 라이브러리
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models, Model, Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# # from sklearn.metrics import classification_report, confusion_matrix # 평가 지표를 정의하기 위한 라이브러리

# # 학습할 이미지들의 파일 경로를 가져오기

# mainDIR = os.listdir('./chest_xray')  # 'chest_xray' 폴더 내의 파일 목록을 가져옵니다.
# print(mainDIR)
# train_folder = './chest_xray/train/'
# val_folder = './chest_xray/val/'
# test_folder = './chest_xray/test/'

# # 학습 데이터 폴더 경로 설정
# os.listdir(train_folder)
# train_n = train_folder + 'NORMAL/'  # 정상 폐 이미지 폴더 경로
# train_p = train_folder + 'PNEUMONIA/'  # 폐렴 이미지 폴더 경로

# # 정상 이미지 샘플
# print(len(os.listdir(train_n)))  # 정상 이미지 파일 개수 출력
# rand_norm = np.random.randint(0, len(os.listdir(train_n)))  # 정상 이미지 중 무작위로 선택
# norm_pic = os.listdir(train_n)[rand_norm]  # 선택된 정상 이미지 파일명
# print('정상 사진 제목: ', norm_pic)
# norm_pic_address = train_n + norm_pic  # 정상 이미지 경로

# # 폐렴 이미지 샘플
# rand_p = np.random.randint(0, len(os.listdir(train_p)))  # 폐렴 이미지 중 무작위로 선택
# sic_pic = os.listdir(train_p)[rand_p]  # 선택된 폐렴 이미지 파일명
# sic_address = train_p + sic_pic  # 폐렴 이미지 경로
# print('폐렴 사진 제목:', sic_pic)

# # 이미지를 불러오고 출력

# # 이미지를 로드합니다.
# norm_load = Image.open(norm_pic_address).convert("L")  # 정상 이미지 로드 (그레이스케일)
# sic_load = Image.open(sic_address).convert("L")  # 폐렴 이미지 로드 (그레이스케일)

# # 이미지를 출력합니다.
# # f = plt.figure(figsize=(10, 6))
# # a1 = f.add_subplot(1, 2, 1)
# # img_plot = plt.imshow(norm_load)
# # a1.set_title('Normal')

# # a2 = f.add_subplot(1, 2, 2)
# img_plot = plt.imshow(sic_load)
# a2.set_title('Pneumonia')
# plt.show()

# # CNN 모델 구축

# # 모델의 정의 부분에서 오류가 있었으므로, 이를 수정합니다.
# model = tf.keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),  # Conv2D 층 추가
#     layers.MaxPooling2D((2, 2)),  # MaxPooling2D 층 추가
#     layers.Flatten(),  # 평탄화
#     layers.Dense(128, activation='relu'),  # Dense 층
#     layers.Dense(64, activation='relu'),  # Dense 층
#     layers.Dense(1, activation='sigmoid')  # 최종 출력을 2가지 클래스 (정상, 폐렴)로 분류
# ])

# # 모델 컴파일
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # 모델 요약 출력
# model.summary()

# # 모델 학습 (train_generator는 데이터 생성기입니다)
# # train_generator는 ImageDataGenerator로 정의되어야 합니다.
# # 예시로 ImageDataGenerator와 flow_from_directory를 사용한 코드 예시를 추가합니다.

# train_datagen = ImageDataGenerator(rescale=1./255)  # 데이터 증강
# train_generator = train_datagen.flow_from_directory(
#     train_folder,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',  # 이진 분류 (정상, 폐렴)
#     color_mode='grayscale'  # 그레이스케일로 설정
# )

# # 모델 학습
# model.fit(train_generator, epochs=10, batch_size=32)

# # 모델 저장
# model.save('mainDIR.h5')  # 모델 저장


import numpy as np  # forlinear algebra
import matplotlib.pyplot as plt  # for plotting things
import os
from PIL import Image  # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics


# 학습할 이미지들의 파일경로를 가져오기

# mainDIR = os.listdir('/home/sunghyuck/git-training/project-y/workspace/HW_5_perceptron_ANN/HW5_ANN_pneumonia_training.py')

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder = './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

# train path
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'

# Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n + norm_pic

# Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneumonia picture title:', sic_pic)

sic_pic_address = train_p + norm_pic

# 학습할 이미지를 불러오고 PLOT

# Load the images
norm_load = Image.open(norm_pic_address).convert("L")
sic_load = Image.open(sic_address).convert("L")

# Let's plt these images
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.savefig('chest_xray.jpg')
plt.show()

# let's build the CNN model


## HW 5-3 Build model
model_in = Input(shape=(64, 64, 3))
x = Flatten()(model_in)
x = Dense(128, activation='relu')(x)
x = Dense(32, activation='relu')(x)
model_out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=model_in, outputs=model_out)
model.summary()  #
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

number_of_test_samples = 600
batch_size = 32

# Training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory(train_folder, target_size=(64, 64), batch_size=batch_size,
                                                 class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_set = test_datagen.flow_from_directory(test_folder, target_size=(64, 64), batch_size=batch_size,
                                                class_mode='binary')

ann_model = model.fit(training_set, steps_per_epoch=163, epochs=5, validation_data=test_set, validation_steps=624)

import pickle

with open('ann_pneumonia_history', 'wb') as pf:
    pickle.dump(ann_model, pf)

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
num_images = 25
plt.figure(figsize=(20, 20))

for i in range(num_images):
    # 랜덤하게 Normal 또는 Pneumonia 이미지 선택
    if np.random.rand() < 0.5:
        image_path = os.path.join(train_n, np.random.choice(os.listdir(train_n)))
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