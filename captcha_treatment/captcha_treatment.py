import glob
import os

import cv2
from PIL import Image


def treat_images(origin_folder, destiny_folder='images/treated_images'):
    files = glob.glob(f"{origin_folder}/*")
    for file in files:
        image = cv2.imread(file)
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, treated_image = cv2.threshold(
            grey_image, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        filename = os.path.basename(file)
        cv2.imwrite(
            f"{destiny_folder}/{filename}", treated_image)

    files = glob.glob(f"{destiny_folder}/*")
    for file in files:
        image = Image.open(file)
        image = image.convert("P")
        copy_image = Image.new("P", image.size, 255)

        for x in range(image.size[1]):
            for y in range(image.size[0]):
                pixel_color = image.getpixel((y, x))
                if(pixel_color < 115):
                    copy_image.putpixel((y, x), 0)

        filename = os.path.basename(file)
        copy_image.save(f'{destiny_folder}/{filename}')


def separate_letters():
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
