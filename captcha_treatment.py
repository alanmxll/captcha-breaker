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
