import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), _ = mnist.load_data()

# Convert the images to float32 and reshape for the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# Normalize the pixel values
train_images = train_images / 255.0

# Convert labels to int64
train_labels = train_labels.astype('int64')

# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

# Filter for only the digit '4'
train_dataset = train_dataset.filter(lambda image, label: label == 5)

# Augment data function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

# Take one sample to show before and after augmentation
for image, label in train_dataset.take(1):
    # Plot before augmentation
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.title('Before Augmentation')

    # Augment the image
    augmented_image, _ = augment(image, label)

    # Plot after augmentation
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image[:,:,0], cmap='gray')
    plt.title('After Augmentation')
    plt.show()
