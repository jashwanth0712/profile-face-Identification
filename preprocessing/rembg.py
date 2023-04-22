import cv2
import numpy as np

img = cv2.imread('Ear2.jpg')
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image to remove noise
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply Otsu's thresholding to obtain a binary image
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Apply morphology to remove small objects from the foreground
kernel = np.ones((5,5), np.uint8)
morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Apply the mask to the original image
result = cv2.bitwise_and(img, img, mask=morphed)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()