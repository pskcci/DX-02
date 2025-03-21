'''ANN with Mnist 실습'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Model load: MNIST / Fashion MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# or
mnist = tf.keras.datasets.mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
#(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

# Normalize images and reshape to (28, 28, 1)
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
f_image_train = np.expand_dims(f_image_train, -1)  # Reshape to (28, 28, 1)
f_image_test = np.expand_dims(f_image_test, -1)    # Reshape to (28, 28, 1)

# 순서대로 라벨링
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display a sample of images
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i].reshape(28, 28), cmap='gray')
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

# CNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# ANN (Flatten -> Dense layers)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))  # Make sure input_shape is correct
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
model.fit(f_image_train, f_label_train, epochs=10, batch_size=32)

# Save the model
model.save('fashion_mnist.h5')

# Load the model
model = tf.keras.models.load_model('./fashion_mnist.h5')

# Make predictions
num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Prediction: ", np.argmax(predict, axis=1))
