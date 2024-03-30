import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import cv2

# Load your image
image = cv2.imread('7h.jpg')

# Resize the image to 28x28 pixels
image_resized = cv2.resize(image, (28, 28))

# Convert the image to grayscale
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Normalize the pixel values
image_normalized = image_gray / 255.0

# Reshape the image to match the input shape expected by the model
image_reshaped = np.expand_dims(image_normalized, axis=0)

# Define your model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Make predictions
predictions = model.predict(image_reshaped)

# Display predictions
print(predictions)

recognized_digit = np.argmax(predictions)

# Display predictions
print("Predicted digit:", recognized_digit)
