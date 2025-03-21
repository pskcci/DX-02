import cv2
import numpy as np

img = cv2.imread('lena.jpeg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],
                   [1, -8, 1], 
                   [1, 1, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('dege1', output)
cv2.waitKey(0)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1], 
                   [0, -1, 0]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('dege2', output)
cv2.waitKey(0)

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])/9
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('dege3', output)
cv2.waitKey(0)