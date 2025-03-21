import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('ann_pneumonia_history', 'rb') as pf:
    ann_model = pickle.load(pf)

#Accuracy
plt.plot(ann_model.history['accuracy'])
plt.plot(ann_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')

#Loss
plt.plot(ann_model.history['val_loss'])
plt.plot(ann_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')

plt.show()