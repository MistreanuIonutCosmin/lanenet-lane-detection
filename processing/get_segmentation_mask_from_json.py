import numpy as np
import cv2
import json
import sys
import os

colors = [20, 70, 120, 170, 220]
colors_size = len(colors)

RESIZE_WIDTH = 512
RESIZE_HEIGHT = 256


# path = "/media/remus/simulator/build/AVMSnapshots/"


def draw_line_from_ecuation(color, degree, image, p, vanish_y, begin, end, step):
    z = 0
    (height, width) = np.shape(image)

    x1 = begin
    y1 = compute_y(x1, degree, p)

    stop_height = vanish_y + 20

    if degree < 2:
        stop_height += 10

    step = float(step)
    interval = np.arange(begin + 1, end, step)
    for _x in interval:
        thickness = int(20 * (y1) / height)
        # print(thickness)
        x2 = _x
        y2 = compute_y(_x, degree, p)

        if y2 > z != 0:
            break
        z = y2

        if stop_height < y2 < height:
            cv2.line(image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness)

        x1 = x2
        y1 = y2


def compute_y(_x, degree, p):
    _y = 0
    for n in range(degree, -1, -1):
        _y += _x ** n * p[degree - n]
    return int(_y)


def draw_lines(image_seg, vanish_point, lines):
    color_no = 0
    color = colors[color_no]
    height, width = np.shape(image_seg)

    lines.sort(key=lambda line: line["id"])

    for line in lines:
        p = line["weights"]
        degree = len(p) - 1

        if line["id"] < 5 or line["id"] == 11:
            draw_line_from_ecuation(color, degree, image_seg, p, vanish_point[1], begin=1, end=vanish_point[0], step=1)
        else:
            draw_line_from_ecuation(color, degree, image_seg, p, vanish_point[1], begin=width - 1, end=vanish_point[0],
                                    step=-1)

        color_no = 0 if color_no + 1 > colors_size - 1 else color_no + 1
        color = colors[color_no]


def create_images_segmentation(image_list, output_path, image_size, ignore_labels):
    width, height = image_size

    instance_path = "/workspace/storage" + sys.argv[3] + "/instance/"
    binary_path = "/workspace/storage" + sys.argv[3] + "/binary/"
    image_path = "/workspace/storage" + sys.argv[3] + "/images/"

    train_file = open(output_path + "/train.txt", 'w')
    val_file = open(output_path + "/val.txt", 'w')

    for image_dict in image_list:
        image_name = image_dict["image_name"]

        # Write segmentation over the image from path or in a new blan image
        # output_image = cv2.imread(path + image_name)
        output_image = np.zeros((height, width))
        draw_lines(output_image, image_dict["vanish_point"], image_dict["lines"])

        if not os.path.exists(output_path + "/instance"):
            os.makedirs(output_path + "/instance")
        if not os.path.exists(output_path + "/binary"):
            os.makedirs(output_path + "/binary")

        output_image = cv2.resize(output_image,
                                  dsize=(RESIZE_WIDTH, RESIZE_HEIGHT),
                                  dst=output_image,
                                  interpolation=cv2.INTER_NEAREST)

        output_image[ignore_labels == 0] = 0

        cv2.imwrite(output_path + "/instance/" + image_name, output_image)

        print(image_name, np.unique(output_image))

        output_image[np.where(output_image > 0)] = 255
        cv2.imwrite(output_path + "/binary/" + image_name, output_image)

        if np.sum(output_image) == 0:
            continue

        val = np.random.random()
        # if val > 0.9:
        #     val_file.write(
        #         image_path + image_name + " " + binary_path + image_name + " " + instance_path + image_name + "\n")
        # else:
        #     train_file.write(
        #         image_path + image_name + " " + binary_path + image_name + " " + instance_path + image_name + "\n")


if __name__ == '__main__':

    if len(sys.argv) < 4:
        raise Exception(
            'Incorrect number or args! Script expects: [1-output_path, 2-json_path, 3-docker_path]'
            '\n docker_path - path after /workspace/storage, starts with \"/\" ends without \"/\".')

    output_path = sys.argv[1]
    input_file = open(sys.argv[2], 'r')

    json_object = input_file.readlines()
    image_list = json.loads(json_object[0])

    ignore_labels = cv2.imread('/media/remus/datasets/AVMSnapshots/AVM/ignore_labels.png')
    ignore_labels = cv2.cvtColor(ignore_labels, cv2.COLOR_BGR2GRAY)

    create_images_segmentation(image_list, output_path, (1280, 720), ignore_labels)
