import os
import pickle

import cv2
import numpy as np
from imutils import paths
from tensorflow.keras.models import load_model

from captcha_treatment import treat_images
from helpers import resize_to_fit


def break_captcha():
  # Import trained model and import translator
    with open('labels_model.dat', 'rb') as translator_file:
        lb = pickle.load(translator_file)

    model = load_model('model_trained.hdf5')

    # Use the model to resolve captchas
    treat_images('resolve', destiny_folder="resolve")

    ############################################################
    files = list(paths.list_images('resolve'))

    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert image to binary
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

        # Find contours of each letter
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        letters_area = []

        # Filter contours that are letters
        for contour in contours:
            (x, y, width, height) = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if area > 115:
                letters_area.append((x, y, width, height))

        # Order letters by axis x
        letters_area = sorted(letters_area, key=lambda x: x[0])

        # Draw countours and separate the letters in individual files
        final_image = cv2.merge([image] * 3)
        prediction = []

        for rectangle in letters_area:
            x, y, width, height = rectangle
            letter_image = image[y-2:y+height+2, x-2:x+width+2]

            # Resize letter to 20x20
            letter_image = resize_to_fit(letter_image, 20, 20)

            # Treatment to Keras recognize image
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            #  Pass the letter to IA identify
            predicted_letter = model.predict(letter_image)
            predicted_letter = lb.inverse_transform(predicted_letter)[0]

            prediction.append(predicted_letter)

        prediction_text = ''.join(prediction)

        print(prediction_text)
        return prediction_text
    ############################################################
