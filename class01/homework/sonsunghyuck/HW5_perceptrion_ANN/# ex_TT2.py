# ex_TT2

from tensorflow.keras.models import Sequential
import tensorflow as tf

# 모델 생성
model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 모델 훈련
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)

# 모델 요약 출력
model.summary()

# 모델 저장
model.save('fashion_mnist.h5')
