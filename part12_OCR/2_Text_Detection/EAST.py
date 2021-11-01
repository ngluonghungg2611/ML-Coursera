import cv2
import numpy as np
from numpy.core.defchararray import center
img = cv2.imread('skew_image.PNG')
cv2.imshow('original image', img)
cv2.waitKey(0)
model = cv2.dnn.readNet('frozen_east_text_detection.pb')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse=True)
max_contour = contours[0]
angle = cv2.minAreaRect(max_contour)[-1]
if angle < -45:
    angle = angle + 90

#   Prepare Image
h, w, _ = img.shape
new_h = (h//32)*32
new_w = (w//32)*32
center = (new_h/2, new_w/2)
M = cv2.getRotationMatrix2D(center, angle, 1)
dst = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#   Get the ratio change in width and height
h_ratio = h/new_h
w_ratio = w/new_w

blob = cv2.dnn.blobFromImage(dst, 1, (new_w, new_h), (123.68, 116.78, 103.94), True, False)

# Pass the image to network and exact score and geometry map
model.setInput(blob)
model.getUnconnectedOutLayersNames()
(geometry, scores) = model.forward(model.getUnconnectedOutLayersNames())

# Post-processing

rectangles = []
confidence_score = []
for i in range(geometry.shape[2]):
    for j in range(0,geometry.shape[3]):

        if scores[0][0][i][j] < 0.1:
            continue
        
        bottom_x = int(4*j + geometry[0][1][i][j])
        bottom_y = int(4*i + geometry[0][2][i][j])

        top_x = int(4*j - geometry[0][3][i][j])
        top_y = int(4*i - geometry[0][0][i][j])
        rectangles.append((top_x, top_y, bottom_x, bottom_y))
        confidence_score.append(float(scores[0][0][i][j]))

# Non-maximum suppression
from imutils.object_detection import non_max_suppression 
find_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.5)
img_copy = dst.copy()

for (x1,y1, x2,y2) in find_boxes:
    x1 = int(x1 * w_ratio)
    y1 = int(y1 * h_ratio)
    x2 = int(x2 * w_ratio)
    y2 = int(y2 * h_ratio)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255,0,0), 2)
    
cv2.imshow('image', img_copy)
cv2.waitKey(0)


