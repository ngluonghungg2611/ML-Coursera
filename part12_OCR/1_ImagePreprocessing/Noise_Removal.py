import cv2
img = cv2.imread('Untitled.png')
img_medianblur = cv2.medianBlur(img, 5)
# cv2.imshow('test', img)
cv2.imshow('blur', img_medianblur)
cv2.waitKey(0)