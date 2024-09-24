import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (pixel values range from 0 to 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Print shape of the dataset
print(f"Training data shape: {x_train.shape}")

import numpy as np

def pixel_to_spike_trains(image, threshold=0.5):
    spike_train = np.where(image > threshold, 1, 0)  # Spike if pixel > threshold
    return spike_train

# Convert first training image to spike train
first_image = x_train[0]
spike_train = pixel_to_spike_trains(first_image)

# Print spike train
print(spike_train)
