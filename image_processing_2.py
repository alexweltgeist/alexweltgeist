# -*- coding: utf-8 -*-
"""
HAndling Images with PIL / imutil and OpenCV
Created on Thu Apr 29 19:42:53 2021

@author: alex
"""

!pip install imutils

import cv2
from PIL import Image

# Redaing and Displaying the file
image = cv2.imread(r'love.jpg')
cv2.imshow("Image", image)
cv2.waitKey(0)

#Displying iage using PIL
pil_image= Image.open(r'love.jpg')
pil_image.show("PIL Image")

# Resizing the image to fit to the screen
image = cv2.imread(r'love.jpg')
screen_res = 1080, 720
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Display', window_width, window_height)
cv2.imshow('Display', image)
cv2.waitKey(0)

# Resizing the image to fit the Normal window
# If cv2.WINDOW_NORMAL is set, the user can resize the window 
image = cv2.imread(r'love.jpg')
cv2.namedWindow('Normal Window', cv2.WINDOW_NORMAL)
cv2.imshow('Normal Window', image)
cv2.waitKey(0)

# Gray image
image = cv2.imread(r'love.jpg')
gray_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)

# Gray image using PIL
pil_image= Image.open(r'love.jpg')
gray_pil=pil_image.convert('L')
gray_pil.show()

#Rotating the image
image = cv2.imread(r'love.jpg')
cv2.namedWindow("Rotated Image", cv2.WINDOW_NORMAL)
rotated_img= cv2.rotate(image,cv2.ROTATE_180 )
cv2.imshow("Rotated Image", rotated_img)
cv2.waitKey(0)

import imutils
import numpy as np
#Rotating the image
image = cv2.imread(r'love.jpg')

# loop over the rotation angles
for angle in np.arange(0, 360, 60):
    cv2.namedWindow("Rotated", cv2.WINDOW_NORMAL)
    rotated = imutils.rotate(image, angle)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    
    
# Rotate image using PIL
pil_image= Image.open(r'love.jpg')
rotate_img_pil=pil_image.rotate(110)
rotate_img_pil.show()


image= cv2.imread(r'taj.jpg')
cv2.namedWindow("Noised Image", cv2.WINDOW_NORMAL)
cv2.imshow("Noised Image", image)
cv2.waitKey(0)


image= cv2.imread(r'taj.jpg')
cv2.namedWindow("Denoised Image", cv2.WINDOW_NORMAL)
denoised_image = cv2.fastNlMeansDenoisingColored(image,None, h=5)
cv2.imshow("Denoised Image", denoised_image)
cv2.waitKey(0)


image= cv2.imread(r'taj.jpg')
cv2.namedWindow("Denoised Image", cv2.WINDOW_NORMAL)
denoised_image = cv2.GaussianBlur(image, (5,5), 0 )
cv2.imshow("Denoised Image", denoised_image)
cv2.waitKey(0)

image= cv2.imread(r'taj.jpg')
cv2.namedWindow("Edge", cv2.WINDOW_NORMAL)
denoised_image = cv2.Canny(image, threshold1 =100,threshold2=200 )
cv2.imshow("Edge", denoised_image)
cv2.waitKey(0)


#resizing
image= cv2.imread(r'taj.jpg')
#cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
#r, the ratio of the new width to the old width
r = 100.0 / image.shape[1]
print(r)
dim = (100, int(image.shape[0] * r))
print(dim)
# perform the actual resizing of the image and show it
#new dimensions of the image by using 100 pixels for the width, 
#and r x the old image height. Doing this allows us to maintain the aspect ratio of the image.
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resized)
cv2.waitKey(0)

image= cv2.imread(r'taj.jpg')
scale_percent =200 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resized)
cv2.waitKey(0)

image= cv2.imread(r'taj.jpg')
cropped_img= image[15:170, 20:200]
cv2.imshow("Cropped", cropped_img)
cv2.waitKey(0)

from PIL import Image 
  
# Opens a image in RGB mode 
pil_image = Image.open(r'taj.jpg') 
  
# Size of the image in pixels (size of orginal image) 
# (This is not mandatory) 
width, height = pil_image.size 
  
# Setting the points for cropped image 
left = 3
top = height /25
right = 200
bottom = 3 * height / 4
  
# Cropped image of above dimension 
# (It will not change orginal image) 
cropped_image = pil_image.crop((left, top, right, bottom))
  
# Shows the image in image viewer 
cropped_image.show() 
