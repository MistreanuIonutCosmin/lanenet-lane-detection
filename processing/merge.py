import cv2
import numpy as np
import sys
import os

dir = '/media/remus/datasets/AVMSnapshots/'

if __name__ == "__main__":
    path1 = "/media/remus/datasets/AVMSnapshots/AVM/mobilenet_pTuS_dup_deconv/image"
    path2 = "/media/remus/datasets/AVMSnapshots/AVM/mobilenet_tuS_pre_val_newGT/image"
    # path3 = "/media/remus/datasets/AVMSnapshots/AVM/output_val/tusimple/4lanes_5lanes/"
    output = "/media/remus/datasets/AVMSnapshots/AVM/newGT_asym"

    if not os.path.exists(output):
        os.makedirs(output)
    image1 = os.listdir(path1)
    image2 = os.listdir(path2)
    # common = list(set(image1).intersection(image2))
    for filename in image2:

        img1 = cv2.imread(path1 + '/' + filename)
        img2 = cv2.imread(path2 + '/' + filename)
        # img3 = cv2.imread(path3 + '/' + filename)
        # print(filename, np.shape(img2), np.shape(img3))
        # img3 = cv2.resize(img3, (1280, 720), interpolation=cv2.INTER_NEAREST)
        img_conc = np.concatenate((img2, img1), axis=1)

        cv2.imwrite(output + '/' + filename, img_conc)