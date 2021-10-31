import cv2
img = cv2.imread('skew_image.PNG')
cv2.imshow('skew image', img)
cv2.waitKey(0)

#   Detec the box
  
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# PHuong phap nghich dao nhi phan -> Van ban mau trang va nen mau den
ret, threshold_image = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True) # reverse = True de giam dan
max_contours = contours[0]
cv2.drawContours(threshold_image, contours, -1, (255,0,0), 3)
cv2.imshow('skew image', threshold_image)
cv2.waitKey(0)
angle=cv2.minAreaRect(max_contours)[-1] # Ham nay tra ve bo gia tri (x, y) va (w,h) va (goc). Va chung ta can lay ra goc
if angle < -45:
    angle = 90 + angle
print(angle)
h, w, _ = img.shape
center = (h/2, w/2)
M = cv2.getRotationMatrix2D(center, angle, 1)
dst = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
cv2.imshow('Deskew image', dst)
cv2.waitKey(0)

