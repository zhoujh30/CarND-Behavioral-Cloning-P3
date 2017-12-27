import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import TensorBoard

# Read data
path = 'data/driving_log.csv'
samples = []
with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
print('Successfully loading {0} records'.format(len(samples)))

# Split data
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
print('Training sets: {0}'.format(len(train_samples)))
print('Validation sets: {0}'.format(len(validation_samples)))

# Update path to an image
def update_img_path(path):
    return path.split('/')[-3] + '/' + path.split('/')[-2] + '/' + path.split('/')[-1]

# Crop the image 70px from the top and 25px from the bottom
def img_crop(image):
    height = image.shape[0]
    width = image.shape[1]
    image = image[70:height-25, 0:width]
    return image

# Data augmentation: add flipped images and inverted angles 
def add_flip(images, angles):
    aug_images, aug_angles = [], []
    for image, angle in zip(images, angles):
        aug_images.append(image)
        aug_angles.append(angle)
        aug_images.append(np.fliplr(image))
        aug_angles.append(angle * -1.0)
    return aug_images, aug_angles

def generator(samples, batch_size = 32):
    num_samples = len(samples)

    # Loop forever so the generator never terminates (set TRUE to plot images):
    show_plot = False
    
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Extract the steering angle and create adjusted steering angles for the side camera images
                angle_center = float(batch_sample[3])
                correction = 0.2                        
                angle_left = angle_center + correction
                angle_right = angle_center - correction

                # Read in images from center, left and right cameras
                path_center, path_left, path_right = batch_sample[0], batch_sample[1], batch_sample[2]
                
                # Update path to an images
                path_center = update_img_path(path_center)
                path_left   = update_img_path(path_left)
                path_right  = update_img_path(path_right)
                
                # Use OpenCV to load an images
                img_center = cv2.cvtColor(cv2.imread(path_center), cv2.COLOR_BGR2RGB)
                img_left   = cv2.cvtColor(cv2.imread(path_left), cv2.COLOR_BGR2RGB)
                img_right  = cv2.cvtColor(cv2.imread(path_right), cv2.COLOR_BGR2RGB)
                
                # Add Images and Angles to dataset
                images.extend([img_center, img_left, img_right])
                angles.extend([angle_center, angle_left, angle_right])

            # Data augmentation: add flipped images and inverted angles 
            aug_images, aug_angles = add_flip(images, angles)

            # Convert images and steering angles to NumPy arrays
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)


# Apply Nvidia model
def nvidia_cnn():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24,5,5, border_mode='valid', activation = "relu", subsample = (2,2)))
    model.add(Convolution2D(36,5,5, border_mode='valid', activation = "relu", subsample = (2,2)))
    model.add(Convolution2D(48,5,5, border_mode='valid', activation = "relu", subsample = (2,2)))
    model.add(Convolution2D(64,3,3, border_mode='valid', activation = "relu", subsample = (1,1)))
    model.add(Convolution2D(64,3,3, border_mode='valid', activation = "relu", subsample = (1,1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# Train the model
model = nvidia_cnn()
model.summary()
model.compile(loss='mse', optimizer='adam')

# Visualize model and its performance using tensorboard
tensorboard = TensorBoard(log_dir = "model_logs", write_graph = True, write_images = True)

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*5,
    validation_data   = validation_generator, nb_val_samples    = len(validation_samples)*5, 
    nb_epoch = 4, verbose  = 1, callbacks = [tensorboard])

# Print the keys 
print(history_object.history.keys())

# Save the train model
model.save('model.h5')
exit()