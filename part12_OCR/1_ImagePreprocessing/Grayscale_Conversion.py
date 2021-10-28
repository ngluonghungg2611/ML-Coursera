import cv2

img = cv2.imread('test111.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('test', img)
cv2.waitKey(0)