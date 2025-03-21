import cv2
import numpy as np

img = cv2.imread('/home/yun/workspace/HW6/cat.png', cv2.IMREAD_COLOR_BGR)
kernel = np.array ([[1, 1 ,1], [1, -8, 1], [1, 1, 1]])
print(kernel)
output1 = cv2.filter2D(img, -1, kernel)

Identity = np.array ([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
print(Identity)
output2 = cv2.filter2D(img, -1, Identity)

edge_detection = np.array ([[-1, -1 ,-1], [-1, 8, -1], [-1, -1, -1]])
print(edge_detection)
output3 = cv2.filter2D(img, -1, edge_detection)

Sharpen = np.array ([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
print(Sharpen)
output4 = cv2.filter2D(img, -1, Sharpen)

Box_blur = np.array ([[1, 1 ,1], [1, 1, 1], [1, 1, 1]])*1/9
print(Box_blur)
output5 = cv2.filter2D(img, -1, Box_blur)

Gaussian_blur1 = np.array ([[1, 2 ,1], [2, 4, 2], [1, 2, 1]])*1/16    # 3x3
print(Gaussian_blur1)
output6 = cv2.filter2D(img, -1, Gaussian_blur1)

Gaussian_blur2 = np.array ([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])*1/256  # 5x5
print(Gaussian_blur2)
output7 = cv2.filter2D(img, -1, Gaussian_blur2)

cv2.imshow('kernel', output1)
cv2.imshow('Identity', output2)
cv2.imshow('edge_detection', output3)
cv2.imshow('Sharpen', output4)
cv2.imshow('Box_blur', output5)
cv2.imshow('Gaussian_blur1', output6)
cv2.imshow('Gaussian_blur2', output7)

cv2.waitKey(0)