import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import os
import cv2
from PIL import Image
from matplotlib import image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import random


CATEGORIES = ["fork", "spoon"]


path_fork = "../fork/"

path_spoon = "../spoon/"


batch_size = 128
epochs = 15
IMG_SIZE = 100


def get_fork_spoon_Data():
    all_images = []

    script_dir = os.path.dirname(os.path.abspath(path_fork))
    for img_path in os.listdir(path_fork):
        try:
            if img_path.endswith(".jpg"):
                img_array = cv2.imread(os.path.join(
                    path_fork, img_path), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                all_images.append([new_array, 0])

        except IOError:
            pass

    script_dir = os.path.dirname(os.path.abspath(path_spoon))
    for img_path in os.listdir(path_spoon):
        try:
            if img_path.endswith(".jpg"):
                img_array = cv2.imread(os.path.join(
                    path_spoon, img_path), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                all_images.append([new_array, 1])

        except IOError:
            pass

    return all_images


dataset = get_fork_spoon_Data()


random.shuffle(dataset)

X = []  # features
y = []  # labels

for features, label in dataset:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


X = X/255.0

# Building the model
model = Sequential()
# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 2 neurons, for 2 classes
model.add(Dense(2))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Training the model, with 50 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, y, batch_size=32, epochs=50, validation_split=0.1)

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')
