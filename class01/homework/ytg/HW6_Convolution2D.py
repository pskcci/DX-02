
import cv2
import numpy as np

#img =  cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
img =  cv2.imread('pikachu.jpg', cv2.IMREAD_GRAYSCALE) # 흑백 사진으로 가져옴
kernel = np.array([[1,1,1], [1,-8,1], [1,1,1]])
kernel2 = np.array([[0,0,0], [0,1,0], [0,0,0]]) # identity
kernel3 = np.array([[1,1,1], [1,1,1], [1,1,1]]) * 0.1111 # box blur
kernel4 = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]]) # Ridge
kernel5 = np.array([[1,2,1], [2,4,2], [1,2,1]]) # gaussian_blur

output = cv2.filter2D(img, -1, kernel)
output2 = cv2.filter2D(img, -1, kernel2)
output3 = cv2.filter2D(img, -1, kernel3)
output4 = cv2.filter2D(img, -1, kernel4)
output5 = cv2.filter2D(img, -1, kernel5)



cv2.imshow('edge',output)
cv2.imshow('identity',output2)
cv2.imshow('box blur',output3)
cv2.imshow('Ridge',output4)
cv2.imshow('gaussian_blur',output5)


cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()

