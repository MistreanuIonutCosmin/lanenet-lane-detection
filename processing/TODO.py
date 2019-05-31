import numpy as np
import cv2
import os
from shutil import copy

# script used for copying/moving/renaming images
IMG_WIDTH = 512
IMG_HEIGHT = 256

if __name__ == '__main__':
    # val_file = open("/media/remus/datasets/AVMSnapshots/AVM/val.txt", "r")
    #
    gt_images = "/media/remus/datasets/AVMSnapshots/AVM/images/0001_AVMFrontCamera.png"
    # seg_images = "/media/remus/datasets/AVMSnapshots/AVM/segmentation/"
    # out_dir = "/media/remus/datasets/AVMSnapshots/AVM/seg_overlay/"

    image = np.ones((512, 288, 1)) * 255
    cv2.imwrite("/media/remus/datasets/AVMSnapshots/AVM/ignore_labels_white.png", cv2.resize(image, (512, 288)))

    # horizontal_flip = image.copy()
    #
    # horizontal_flip = cv2.flip(horizontal_flip, 0)
    # vertical_flip = image.copy()
    #
    # vertical_flip = cv2.flip(vertical_flip, 1)
    #
    # cv2.imshow("horizontal", horizontal_flip)
    # cv2.imshow("vertical", vertical_flip)
    # cv2.waitKey(0)
    #
    # images = os.listdir(gt_images)
    # segmentations = os.listdir(seg_images)
    # images.sort()
    # segmentations.sort()

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #
    # for img_name, seg_name in zip(images, segmentations):
    #     image = cv2.imread(gt_images + img_name)
    #     seg = cv2.imread(seg_images + seg_name)
    #
    #     argmax_layers = cv2.addWeighted(seg, 0.4, image, 0.6, 0)
    #     cv2.imwrite(os.path.join(out_dir, img_name), argmax_layers)
    #     img = cv2.imread(gt_images + name)
    #     img = cv2.resize(img,
    #                dsize=(512, 256),
    #                dst=img,
    #                interpolation=cv2.INTER_LINEAR)
    #     cv2.imwrite(gt_images + name, img)
    #
    # print(1)
