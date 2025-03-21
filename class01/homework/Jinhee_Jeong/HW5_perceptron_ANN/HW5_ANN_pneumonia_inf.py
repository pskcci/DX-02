import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import image_dataset_from_directory

img_height = 64
img_width = 64
batch_size = 32

def prepare(ds, shuffle = False, augment = False) :
    ds = ds.map(lambda x, y : (tf.image.resize(x, [img_height, img_width]), y))
    if shuffle :
        ds = ds.shuffle(1000)
    return ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        
model = tf.keras.models.load_model('ann_pneumonia.keras')
model.summary()

test_folder = '/home/jin/Downloads/chest_xray/test'
test_n = test_folder + '/NORMAL'
test_p = test_folder + '/PNEUMONIA/'

test_ds = image_dataset_from_directory(test_folder,
                                       validation_split = 0.2,
                                       subset = "validation",
                                       seed=123,
                                       image_size=(img_height, img_width),
                                       batch_size = batch_size)
test_ds = prepare(test_ds)
test_img, test_label = next(iter(test_ds))
test_img = np.array(test_img)
test_label = np.array(test_label)


plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_img[i].numpy().astype("uint8"))
    plt.title(f'Label: {test_label[i].numpy()}')
    plt.axis('off')
plt.show()