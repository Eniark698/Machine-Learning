import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import Initializer

# Example data for SVD
train_images = np.random.random((100, 64))  # Example matrix with 100 samples and 64 features

# Decompose the matrix using SVD
U, S, VT = np.linalg.svd(train_images, full_matrices=False)

class SVDInitializer(Initializer):
    def __init__(self, svd_components, strategy='VT', additional_initializer=None, scale=1.0):
        self.svd_components = svd_components
        self.strategy = strategy
        self.additional_initializer = additional_initializer or tf.keras.initializers.GlorotUniform()
        self.scale = scale

    def __call__(self, shape, dtype=None):
        if self.strategy == 'U':
            component_matrix = self.svd_components[0]  # Use U matrix
        elif self.strategy == 'S':
            component_matrix = np.diag(self.svd_components[1])  # Use S matrix
        elif self.strategy == 'VT':
            component_matrix = self.svd_components[2]  # Use VT matrix
        else:
            raise ValueError("Invalid strategy. Use 'U', 'S', or 'VT'.")

        # Ensure the matrix has the correct shape
        initialized_matrix = component_matrix[:shape[0], :shape[1]]

        # Apply scaling
        initialized_matrix *= self.scale

        # If the component matrix is smaller, fill the rest with additional_initializer
        if initialized_matrix.shape != shape:
            additional_matrix = self.additional_initializer(shape, dtype=dtype)
            initialized_matrix = np.pad(initialized_matrix, ((0, shape[0] - initialized_matrix.shape[0]),
                                                            (0, shape[1] - initialized_matrix.shape[1])),
                                        'constant', constant_values=0)
            initialized_matrix += additional_matrix[:shape[0], :shape[1]]

        return tf.convert_to_tensor(initialized_matrix, dtype=dtype)

    def get_config(self):
        return {
            'svd_components': self.svd_components,
            'strategy': self.strategy,
            'additional_initializer': tf.keras.initializers.serialize(self.additional_initializer),
            'scale': self.scale
        }

# Number of components to use
n_components = 10
svd_components = (U[:, :n_components], S[:n_components], VT[:n_components, :])

# Using the SVD Initializer
svd_initializer = SVDInitializer(svd_components, strategy='VT', scale=0.1)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,)),  # Assuming input shape is 64
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer=svd_initializer),
    tf.keras.layers.Dense(1)  # Example output layer
])

model.compile(optimizer='adam', loss='mse')
model.summary()
