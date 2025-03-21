# Ex_TT3

import tensorflow as tf
from tensorflow import keras



model = Sequential()

model.add(tf.keras.layer.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.conpile(
    optimizer ='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
)
model.fit(image_train, label_train, epochs = 10, batch_size = 10)
model.summary()
model.save('fashion_mnist.h5')


