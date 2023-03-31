import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read the image
img = cv2.imread('boxes.jpg')
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 500)
# Apply binary threshold
ret, thresh = cv2.threshold(edges, 150, 355, cv2.THRESH_BINARY)
# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Loop through all the contours
num_books = 0
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.10 * cv2.arcLength(cnt, True), True)
    if len(approx) ==2:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.1 < aspect_ratio < 2.0:
            num_books += 1
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)
# Print the number of books
print("Number of books: ", num_books)
# Display the original image with rectangles
plt.imshow(img)
plt.show()
