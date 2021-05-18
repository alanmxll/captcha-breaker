import glob
import os

import cv2

files = glob.glob('images/treated_images/*')
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

    if len(letters_area) != 5:
        continue

    # Draw countours and separate the letters in individual files
    final_image = cv2.merge([image] * 3)

    index = 0
    for rectangle in letters_area:
        x, y, width, height = rectangle
        letter_image = image[y-2:y+height+2, x-2:x+width+2]
        index += 1
        filename = os.path.basename(file).replace(
            ".png", f"letra{index}.png")
        cv2.imwrite(f'images/letters/{filename}', letter_image)
        cv2.rectangle(final_image, (x-2, y-2),
                      (x+width+2, y+height+2), (0, 255, 0), 1)

    filename = os.path.basename(file)
    cv2.imwrite(f"images/identified/{filename}", final_image)
