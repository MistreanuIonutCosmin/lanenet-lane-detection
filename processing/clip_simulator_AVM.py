import numpy as np
import cv2
import sys
import os
from shutil import copy

np.set_printoptions(threshold=sys.maxsize)

# script used for copying/moving/renaming images
IMG_WIDTH = 416
IMG_HEIGHT = 288

CROP_MIN = 143
CROP_MAX = 1136

if __name__ == '__main__':
    img_path = "/media/remus/datasets/AVMSnapshots/AVM/val_images/"
    out_path = "/media/remus/datasets/AVMSnapshots/AVM/val_images/"


    images_list = os.listdir(img_path)
    images_list.sort()

    for image_name in images_list:
        print(image_name)

        image = cv2.imread(img_path + image_name)
        image = image[:, CROP_MIN:CROP_MAX, :]
        # image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(out_path + image_name, image)




