'''ANN with Mnist 실습'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import cv2


# Model load: MNIST / Fashion MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# or
mnist = tf.keras.datasets.mnist

#밑에꺼 두개 바꾸면 데이터셋 바뀜
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
#(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

# normalized iamges
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

#순서대로 라벨링
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(data_format=None))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(
    optimizer='adam', #'아담'이라는 알고리즘을 사용하겠다
    loss='sparse_categorical_crossentropy', #일부 히든 날리는거
    metrics=['accuracy'],   #정확도
    )
model.fit(f_image_train, f_label_train, epochs=10, batch_size=32)   #모델 학습시킴
model.summary() #모델 확인함
model.save('fashion_mnist.h5')  #모델 저장함

model = tf.keras.models.load_model('./fashion_mnist.h5')    #확장자가 keras를 사용

num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))
