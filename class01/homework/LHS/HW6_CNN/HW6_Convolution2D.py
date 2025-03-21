'''Convolution 예제'''

import cv2
import numpy as np
import tensorflow as tf

#폴더 위치 + 이미지 파일
p = '/home/rg/workspace/HW6_CNN/'
P = p + 'color.png' #Orig.png


#img = cv2.imread(P, cv2.IMREAD_GRAYSCALE)
#img = cv2.imread(P, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(P, cv2.IMREAD_COLOR_RGB)

#kernal = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
#kernal = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
#kernal = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
#kernal = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#kernal = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
kernal = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
#kernal = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256

print(kernal)
output = cv2.filter2D(img, -1, kernal)
cv2.imshow('edge', output)
cv2.waitKey(0)