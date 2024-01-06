import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# Load and preprocess MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.0

# Take the first image from the dataset
original_image = x_train[0].squeeze()

# Save the original image
plt.imsave('./Project_svd+cnn/original_image.png', original_image, cmap='gray')

# Define a function for data augmentation
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

# Apply augmentation to the first image
augmented_image = augment(x_train[0]).numpy().squeeze()

# Save the augmented image
plt.imsave('./Project_svd+cnn/augmented_image.png', augmented_image, cmap='gray')

# Flatten the augmented image for SVD
augmented_image_flat = augmented_image.reshape(-1)

# Apply SVD
n_components = 150  # Adjust as needed
svd = TruncatedSVD(n_components=n_components)
augmented_image_svd = svd.fit_transform(augmented_image_flat.reshape(1, -1))

# Reconstruct an approximation of the original image using the inverse transform
reconstructed_image = svd.inverse_transform(augmented_image_svd)
reconstructed_image_reshaped = reconstructed_image.reshape(28, 28)

# Save the reconstructed image
plt.imsave('./Project_svd+cnn/reconstructed_image.png', reconstructed_image_reshaped, cmap='gray')
