"""
Image Segmentation based on the colour green
Provides percentage of the colour green
Highlights largest contour of green cover
"""
import numpy as np
import cv2

#Read image
img = cv2.imread("2009M.jpg")

# boundary conditions for green color H,S,V (Can be tweaked if required)
lowerBound = np.array([36, 70, 40])
upperBound = np.array([102, 255, 255])

# image processing for easy segmentation
img_resize = cv2.resize(img, (340, 220))
img_blur = cv2.GaussianBlur(img_resize, (3,3), cv2.BORDER_DEFAULT)

# Colour segmentation is easily achievable in HSV domain
imgHSV = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

# create a filter or mask to filer out a specific color here we filter green color
mask = cv2.inRange(imgHSV, lowerBound, upperBound)

# Finding and drawing the largest region of green cover in image
conts, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


if len(conts) != 0:
    # find the biggest countour (c) by the area
    c = max(conts, key=cv2.contourArea)

    # draw in green the largest contour
    cv2.drawContours(img_resize, c, -1, (0,255,0), 2)

# show the image with largest contour
cv2.imshow("Largest Green Cover", img_resize)

# Contours of all green cover
# cv2.drawContours(img_resize, conts, -1, (0,255,0), 2)
# cv2.imshow("Green Regions", img_resize)

# Region of green cover & percentage of green in image
cv2.imshow('Green', mask)
print((mask.mean() * 100 / 255))

cv2.waitKey(0)
cv2.destroyAllWindows()
