import os
import pickle

import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from helpers import resize_to_fit

data = []
labels = []
base_images_folder = "images/base_letters"

images = paths.list_images(base_images_folder)

for file in images:
    label = file.split(os.path.sep)[-2]
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Standardize image to 20x20
    image = resize_to_fit(image, 20, 20)

    # Add one dimension to keras can read the image
    image = np.expand_dims(image, axis=2)

    # Add lists of data and labels
    labels.append(label)
    data.append(image)

data = np.array(data, dtype="float") / 255
labels = np.array(labels)

# Separate in data of trainning (75%) and data of test(25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(
    data, labels, test_size=0.25, random_state=0)

# Convert with one-hot enconding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the labelbinarizer in a file with pickle
with open('labels_model.dat', 'wb') as file_pickle:
    pickle.dump(lb, file_pickle)

# Create and train the AI
model = Sequential()

# Create layers of the neural network
model.add(Conv2D(20, (5, 5), padding="same",
          input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Create second layer
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Create third layer
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Create exit layer
model.add(Dense(26, activation="softmax"))

# Compile all layers
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train AI
model.fit(X_train, Y_train, validation_data=(
    X_test, Y_test), batch_size=26, epochs=10, verbose=1)

# Save model in a file
model.save("model_trained.hdf5")
