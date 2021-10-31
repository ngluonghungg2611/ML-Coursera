import cv2
import numpy as np

#   Divide by max
def devide_by_max(img):
    return img/255

#   Mix-max scaling
def min_max_scaking(img):
     img_min = np.min(img)
     img_max = np.max(img)
     return (img - img_min) / (img_max - img_min)

# Standardization
def standardization(img):
     mu = img.mean()
     std = img.std()
     return (img - mu) / std

img = cv2.imread('test111.png')
print(devide_by_max(img),'\n')
print(min_max_scaking(img),'\n')
print(standardization(img),'\n')