import os
import json
import cv2
import numpy as np
import sys
import get_segmentation_mask_from_json as gs
from sklearn.linear_model import LinearRegression

H_SAMPLES = np.array([160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
             370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
             580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])

lr = LinearRegression()


def get_angle(xs, y_samples):
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs) > 1:
        lr.fit(ys[:, None], xs)
        k = lr.coef_[0]
        theta = np.arctan(k)
    else:
        theta = 0
    return theta


def get_line_representation(mask, line_value):

    line = []
    for h in H_SAMPLES:
        indices = np.where(mask[h] == line_value)
        if len(indices[0]) > 0:
            line.append(indices[0][len(indices[0]) / 2])
        else:
            line.append(-2)

    return line


def get_13_masks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    values = np.unique(gray)

    masks = np.zeros((13, np.shape(gray)[0], np.shape(gray)[1]))
    masks[0] = np.where(gray == 0, 1, 0)

    left_lines = []
    right_lines = []

    for value in values:
        if value == 0:
            continue

        line = get_line_representation(gray, value)

        angle = get_angle(np.array(line), H_SAMPLES)

        if angle < 0:
            left_lines.append([angle, value])
        else:
            right_lines.append([angle, value])

    left_lines.sort(key=lambda l: -l[0])
    right_lines.sort(key=lambda l: l[0])

    idx = 4
    for line in left_lines:
        value = line[1]
        masks[idx] = np.where(gray == value, 1, 0)
        idx -= 1

    idx = 6
    for line in right_lines:
        value = line[1]
        masks[idx] = np.where(gray == value, 1, 0)
        idx += 1

    return masks


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception(
            'Incorrect number or args! Script expects: [1-image_path]'
            '\n image_path - path of output or instance segmentation ground truth')

    image_path = sys.argv[1]

    # takes images from argv[1] and fit a polynomial equation through each line
    # fit_polynomial_to_lanenet_output(image_path)
    img = cv2.imread(image_path + "0016_AVMFrontCamera.png")
    
    masks = get_13_masks(img)

    for i in range(13):
        cv2.imshow("mask" + str(i), masks[i])
        cv2.waitKey(0)







