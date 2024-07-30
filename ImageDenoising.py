# Import necessary libraries
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
# 28 x 28 x 1 
# (Grayscale 'Clean' Images of Numbers With Their Labels 'Which We're not interested in')
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize The Intensity Levels: [0 - 1]
# uint8 [0-255] -> float32 [0 - 1]
x_train = x_train.astype('float32') / 255. # (60000, 28, 28)
x_test = x_test.astype('float32') / 255. # (10000, 28, 28)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) # (60000, 28, 28, 1)
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) # (10000, 28, 28, 1)

# Adding some noise to the images
noise_factor = 0.5  
#Add Some Random Value With a Mean Around 0 'loc = 0.0' and Standard Deviation of 1 'scale = 1.0'
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

# Clipping values Bigger than one to 1 and Smaller Than zero to 0
# Reduces the gray levels
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Displaying images with noise
plt.figure(figsize=(20, 2))
for i in range(1, 10):
    ax = plt.subplot(1, 10, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")
plt.show()

# Define the autoencoder model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
model.summary()

# Train the model
# Fitting Noisy Images to Clean Images
# 
model.fit(x_train_noisy, x_train, epochs=10, batch_size=256, shuffle=True, 
          validation_data=(x_test_noisy, x_test))

# Evaluate the model
model.evaluate(x_test_noisy, x_test)

# Save the model
model.save('denoising_autoencoder.model')

# Denoise test images and display
no_noise_img = model.predict(x_test_noisy)

plt.figure(figsize=(40, 4))
for i in range(10):
    # Display original noisy image
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")
    
    # Display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 + i + 1)
    plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

plt.show()
