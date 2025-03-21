import cv2
import numpy as np

img = cv2.imread('/home/jin/workspace/HW6_Convolution/subin1.png', cv2.IMREAD_GRAYSCALE)

kernel1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
kernel2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
kernel3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel4 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel5 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])*1/9
kernel6 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])*1/16
kernel7 = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])*1/256
print(kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7)


output1 = cv2.filter2D(img, -1, kernel1)
output2 = cv2.filter2D(img, -1, kernel2)
output3 = cv2.filter2D(img, -1, kernel3)
output4 = cv2.filter2D(img, -1, kernel4)
output5 = cv2.filter2D(img, -1, kernel5)
output6 = cv2.filter2D(img, -1, kernel6)
output7 = cv2.filter2D(img, -1, kernel7)

cv2.imshow('1', output1)
cv2.imshow('2', output2)
cv2.imshow('3', output3)
cv2.imshow('4', output4)
cv2.imshow('5', output5)
cv2.imshow('6', output6)
cv2.imshow('7', output7)
cv2.waitKey(0) #!