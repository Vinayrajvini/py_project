from skimage import io
from skimage import color
from skimage import feature
from skimage import measure
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
# Load the image
image = io.imread('b5.jpg')
# Convert the image to grayscale & Apply edge detection to the grayscale image
gray_image = color.rgb2gray(image)
edges = feature.canny(gray_image)
# Apply binary dilation to the edges
dilated_edges = morphology.dilation(edges, np.ones((15, 15)))
# Label the connected regions in the dilated edges
labels = measure.label(dilated_edges)
# Count the number of unique labels in the labels array
num_books = len(np.unique(labels))
# Display the image with the labeled regions
plt.imshow(labels, cmap='jet')
plt.show()
# Print the number of books in img
print("Number of books in the image:", num_books)
