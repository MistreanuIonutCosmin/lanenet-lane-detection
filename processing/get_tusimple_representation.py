import os
import json
import time

import cv2
import numpy as np
import sys
import get_segmentation_mask_from_json as gs

from sklearn.linear_model import LinearRegression

HEIGHTS = [250, 300, 350, 400]
H_SAMPLES = np.array([160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
     370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
     580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])
lr = LinearRegression()
pixel_thresh = 20
pt_thresh = 0.85


def get_angle(xs, y_samples):
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs) > 1:
        lr.fit(ys[:, None], xs)
        k = lr.coef_[0]
        theta = np.arctan(k)
    else:
        theta = 0
    return theta


def get_indexes(x, xs):
    return [i for (y, i) in zip(xs, range(len(xs))) if x == y]


def sample_lines(mask):
    vote = []
    for height in HEIGHTS:
        # get indices where mask is not black
        labels = np.where(mask[height] != 0)

        # get unique values as np.where returns two arrays, one for each axis (width, chanel)
        uniq = np.unique(labels[0])

        # ranges contains start, end of an consecutive interval of
        ranges = sum((list(t) for t in zip(uniq, uniq[1:]) if t[0] + 1 != t[1]), [])

        indices = [get_indexes(x, uniq)[0] for x in ranges]
        vote.append([indices, uniq, height])

    # pick the height with most lines crossed
    return max(vote, key=lambda m: len(m[0]))


def recolor_mask(mask, most_voted):
    indices = [0] + most_voted[0] + [len(most_voted[1]) - 1]
    # print(most_voted)
    h = most_voted[2]
    new_colors = []

    # traverse boundaries of each interval,
    for i in range(0, len(indices), 2):
        begin = indices[i]
        end = indices[i + 1]

        pick = begin + (end - begin) / 2
        w = most_voted[1][pick]

        color = mask[h, w]
        n_color = (i / 2 + 1) * 31
        mask[mask == color] = n_color
        new_colors.append(n_color)

    return new_colors


def get_line_representation(mask, line_value):
    line = []
    for h in H_SAMPLES:
        indices = np.where(mask[h] == line_value)
        if len(indices[0]) > 0:
            line.append(int(indices[0][int(len(indices[0]) / 2)]))
        else:
            line.append(-2)

    return line


def get_json_representation(mask, name):
    height, width = mask.shape
    # most_voted = sample_lines(mask)
    # print(most_voted)
    # lines_values = recolor_mask(mask, most_voted)
    lines_values = np.unique(mask)

    print(lines_values)
    lines = []

    for value in lines_values:
        if value == 0:
            continue
        lines.append(get_line_representation(mask, value))

    predictions = []

    angles = [get_angle(np.array(x_gts), np.array(H_SAMPLES)) for x_gts in lines]
    threshs = [pixel_thresh / np.cos(angle) for angle in angles]

    if name in predictions:
        pred = cv2.imread(sys.argv[4] + name)

    for line, t in zip(lines, threshs):
        for i in range(len(H_SAMPLES)):
            if line[i] == -2:
                continue

            cv2.circle(mask, (line[i], H_SAMPLES[i]), radius=3, color=255, thickness=1)

            if name in predictions:
                x_minus = int(line[i] - t)
                x_plus = int(line[i] + t)

                if 0 <= x_minus:
                    cv2.circle(pred, (x_minus, H_SAMPLES[i]), radius=1, color=(0, 150, 0), thickness=2)
                if x_plus < width:
                    cv2.circle(pred, (x_plus, H_SAMPLES[i]), radius=1, color=(0, 150, 0), thickness=2)

    if name in predictions:
        cv2.imwrite(sys.argv[4] + name, pred)

    return lines


def draw_line_from_ecuation(color, degree, image, p, begin, end, step):
    z = 0
    (height, width) = np.shape(image)

    x1 = begin
    y1 = gs.compute_y(x1, degree, p)

    stop_height = H_SAMPLES[5]

    for _x in range(begin + 1, end, step):
        thickness = int(20 * (y1) / height)
        # print(thickness)
        x2 = _x
        y2 = gs.compute_y(_x, degree, p)

        if y2 > z != 0:
            break
        z = y2

        if stop_height < y2 < height:
            cv2.line(image, (x1, y1), (x2, y2), int(color), thickness)

        x1 = x2
        y1 = y2


def fit_polynomial_to_lanenet_output(lanenet_output_path, tusimple_path):
    images = os.listdir(lanenet_output_path)
    for name in images:
        image = cv2.imread(lanenet_output_path + name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        t_start = time.time()

        lines_values = np.unique(image)
        n_image = np.zeros(np.shape(image))

        for value in lines_values:
            if value == 0:
                continue
            line = np.array(get_line_representation(image, value))

            xs, ys = line[line >= 0], H_SAMPLES[line >= 0]

            p = np.polyfit(xs, ys, 2)

            if 0 < get_angle(xs, ys):
                draw_line_from_ecuation(value, 2, n_image, p, 1280, 0, -1)
            else:
                draw_line_from_ecuation(value, 2, n_image, p, 0, 1280, 1)
        print(name, " - time: ", time.time() - t_start)

        cv2.imwrite(tusimple_path + name, n_image)


if __name__ == '__main__':

    if len(sys.argv) < 4:
        raise Exception(
            'Incorrect number or args! Script expects: [1-image_path, 2-tusimple_path, 3-json_path]'
            '\n image_path - path of output or instance segmentation ground truth'
            '\n tusimple_path - path of where to output masks with selected points (tusimple representation)'
            '\n json_path - json file where to write tusimple representation')

    ignore_mask = cv2.imread("/media/remus/datasets/AVMSnapshots/AVM/ignore_labels.png")
    ignore_mask = cv2.cvtColor(ignore_mask, cv2.COLOR_RGB2GRAY)
    ignore_mask = cv2.resize(ignore_mask, (993, 720), interpolation=cv2.INTER_NEAREST)

    image_path = sys.argv[1]
    tusimple_path = sys.argv[2]
    poly = sys.argv[4]

    # takes images from argv[1] and fit a polynomial equation through each line
    if poly == "true":
        fit_polynomial_to_lanenet_output(image_path, tusimple_path)
        image_path = tusimple_path

    output_file = open(sys.argv[3], 'w')

    image_list = os.listdir(image_path)
    image_list.sort()

    for image_name in image_list:
        print(image_name)

        image_dict = {}
        image = cv2.imread(image_path + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image[ignore_mask == 0] = 0

        image = cv2.resize(image, (993, 720), interpolation=cv2.INTER_NEAREST)
        image_lines = get_json_representation(image, image_name)

        if not os.path.exists(tusimple_path):
            os.makedirs(tusimple_path)

        cv2.imwrite(tusimple_path + image_name, image)

        image_dict["h_samples"] = list(map(int, H_SAMPLES))
        image_dict["lines"] = image_lines
        image_dict["raw_file"] = image_name

        image_representation = json.dumps(image_dict)
        output_file.write(image_representation + "\n")

    output_file.close()
