import skimage
from skimage import feature, color, io

# Load the face image
image = io.imread('preprocessing/4.jpg')

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Resize the image to a standard size
resized_image = skimage.transform.resize(gray_image, (256, 256))
# Compute the HOG features
hog_features, hog_image = feature.hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), block_norm='L2-Hys',
                                       visualize=True, feature_vector=True)

# Plot the HOG features
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.imshow(resized_image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = skimage.exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax2.axis('off')

plt.show()
