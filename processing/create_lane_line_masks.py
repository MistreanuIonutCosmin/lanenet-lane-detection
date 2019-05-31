import os
import cv2
import json
import argparse
import numpy as np
import pycocotools.mask as mask_tools
from PIL import Image
from get_tusimple_representation import get_angle
from get_segmentation_mask_from_json import compute_y

H_SAMPLES = np.array([160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220,
                      225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285,
                      290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350,
                      355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415,
                      420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480,
                      485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545,
                      550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610,
                      615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
                      680, 685, 690, 695, 700, 705, 710])

AVM_road_categories = {
    # 33: 'lane',
    8: 'yellowline',
    # 9: 'blueline',
    10: 'whiteline',
    # 11: 'parkingline',
}

new_train_list = []
count = 0

BASE_DIR = "/media/remus/datasets/DeepLGE/line_dataset/line_dataset_01/"
MASK_ROOT = "/media/remus/datasets/DeepLGE/line_dataset/line_dataset_01/test_mask"


def merge_masks(mask):  # used
    mask = np.sum(mask, axis=2)
    return mask


def count_outliers(x, threshold):
    outliers = 0
    for i in range(len(x) - 1):
        if np.abs(x[i] - x[i + 1]) > threshold:
            outliers += 1
    return outliers


def draw_polynomial_boundaries(mask):
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = np.array(np.amax(np.array(mask), axis=2))

    lines_values = np.unique(mask)
    line_value = lines_values[1]

    lines = []
    threshold = 20  # diagonal of a square with l = 10  ~sqrt(l^2 + l^2)

    left_line = []
    right_line = []
    size = float(len(H_SAMPLES))

    for i in range(len(H_SAMPLES)):
        h = H_SAMPLES[i]
        indices = np.where(mask[h] == line_value)

        if len(indices[0]) > 0:
            l = indices[0][0]
            r = indices[0][len(indices[0]) - 1]

            t = threshold + float(float(i) / size) * 4 * threshold
            if r - l < t:
                left_line.append(-2)
                right_line.append(-2)
            else:
                left_line.append(l)
                right_line.append(r)

        else:
            left_line.append(-2)
            right_line.append(-2)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    index = np.where(mask != (0, 0, 0))
    mask[index[0], index[1], :] = (0, 125, 125)
    left_line = np.int64(left_line)
    right_line = np.int64(right_line)
    left_model = None
    right_model = None

    xl, yl = left_line[left_line >= 0], H_SAMPLES[left_line >= 0]
    if len(xl) > 30:
        left_outliers = count_outliers(xl, threshold)
        f = 0.9 - float(left_outliers) / float(len(xl))
        left_model = ransac_polyfit(xl, yl, f=f)

    xr, yr = right_line[right_line >= 0], H_SAMPLES[right_line >= 0]
    if len(xr) > 30:
        right_outliers = count_outliers(xr, threshold)
        f = 0.9 - float(right_outliers) / float(len(xr))
        right_model = ransac_polyfit(xr, yr, f=f)

    draw_left_right(xl, yl, left_model, xr, yr, right_model, mask)
    #
    size = float(len(H_SAMPLES))
    for i in range(int(size)):
        l = left_line[i]
        r = right_line[i]

        if l != -2:
            cv2.circle(mask, (l, H_SAMPLES[i]), radius=2, color=(0, 0, 100), thickness=2)

        if r != -2:
            cv2.circle(mask, (r, H_SAMPLES[i]), radius=2, color=(0, 100, 0), thickness=2)

    return mask


def draw_left_right(xl, yl, left_model, xr, yr, right_model, mask):
    if left_model is not None:
        min_x = np.min(xl)
        max_x = np.max(xl)

        xl = np.arange(min_x, max_x, 5)
        yl = np.polyval(left_model, xl).astype(np.int32)

        pts = np.array([[x, y] for x, y in zip(xl, yl)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], False, (0, 0, 255), thickness=10)

    if right_model is not None:
        min_x = np.min(xr)
        max_x = np.max(xr)

        xr = np.arange(min_x, max_x, 5)
        yr = np.polyval(right_model, xr).astype(np.int32)

        pts = np.array([[x, y] for x, y in zip(xr, yr)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], False, (0, 255, 0), thickness=10)


def is_stopline(mask):
    lines_values = np.unique(mask)
    line_value = lines_values[1]

    lower, upper = -0.1, 0.1
    indices = np.where(mask == line_value)

    size = len(indices[0])
    choice = np.random.choice([True, False], size=size, p=[0.05, 0.95])
    # sample_points = np.random.randint(size, size=size / 10)

    x = indices[1][choice]
    y = indices[0][choice]

    print(get_angle(y, x))

    if lower < get_angle(y, x) < upper:
        return True

    return False


def ransac_polyfit(x, y, order=2, n=16, k=500, t=10, d=30, f=0.8):
    # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus

    # n - minimum number of data points required to fit the model
    # k - maximum number of iterations allowed in the algorithm
    # t - threshold value to determine when a data point fits a model
    # d - number of close data points required to assert that a model fits well to data
    # f - fraction of close data points required

    besterr = np.inf
    bestfit = None

    for kk in xrange(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit


def filter_json_lane(idx, labels_json, image_file_name):
    """Function used to keep only ego-lane instances in json.
            Args:
                idx: index of json file
                labels_json: json file containing lables for a certain image
                image_file_name: full path to image corresponding to certain json
            Return:
                image - dictionary to be written to unified json
        """
    image = labels_json["images"][0]
    img = cv2.imread(image_file_name)

    mask_path = image_file_name[len(BASE_DIR) + len("image/"):]
    img_fname_no_ext = os.path.splitext(mask_path)[0]
    mask_dir = os.path.dirname(mask_path)

    image["id"] = idx
    height = image["height"]
    width = image["width"]
    image["file_name"] = image_file_name

    # colors for debbuging
    # # yellow
    my_lane_color = (0, 255, 0)
    # # red
    other_lane_color = (0, 0, 255)
    # white
    white_lines_color = (255, 255, 255)
    # # # blue
    blue_lines_color = (255, 0, 0)
    # # # yellow
    yellow_lines_color = (0, 125, 125)
    # # #
    parking_lines_color = (0, 255, 255)

    alpha = 0.5
    annotations = []

    total_mask = np.zeros((height, width), dtype=np.uint8)
    mask_layers = []
    for i, annotation in enumerate(labels_json["annotations"]):
        # print(int(annotation["category_id"]))
        # print(AVM_robust_categories.keys())
        if int(annotation["category_id"]) in AVM_road_categories.keys():
            # print(image_file_name)

            annotation["id"] = i
            annotation["category_name"] = AVM_road_categories[annotation["category_id"]]
            annotation["image_id"] = idx
            annotations.append(annotation)

            seg = annotation["segmentation"]
            # if len(seg) > 1:
            #     continue

            try:
                seg_RLE = mask_tools.frPyObjects(seg, height, width)
            except:
                print("fucked up image !!!!!!!", image_file_name)
                continue

            seg_mask = mask_tools.decode(seg_RLE)
            color_mask = np.zeros(np.shape(img))

            if len(seg_mask[0][0]) > 1:
                seg_mask = merge_masks(seg_mask)

            seg_mask = seg_mask.astype(np.uint8)
            # if (np.amax(np.unique(seg_mask))) > 1:
            #     print("sad i=", i)

            stopline = False

            reshaped_seg_mask = seg_mask.reshape((seg_mask.shape[0], seg_mask.shape[1]))
            # print(np.unique(reshaped_seg_mask, return_counts=True))


            if annotation["category_name"] == 'yellowline':
                seg_mask[seg_mask >= 1] = 125
                color_mask = draw_polynomial_boundaries(seg_mask)
                # color_mask[color_mask != (0, 0, 0)] = (0, 255, 255)



            elif annotation["category_name"] == 'whiteline':
                seg_mask[seg_mask >= 1] = 255
                stopline = is_stopline(seg_mask)
                color_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)


            mask_layers.append(color_mask)

    # total_mask[total_mask > 2] = 2
    argmax_layers = []
    if mask_layers:
        argmax_layers = np.amax(np.array(mask_layers), axis=0)

    out_mask_dir = os.path.join(MASK_ROOT, mask_dir)
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)

    if argmax_layers is not []:
        global count
        count += 1
        print(count)

        argmax_layers = cv2.addWeighted(img, 0.3, argmax_layers, 0.7, 0)
        cv2.imwrite(os.path.join(MASK_ROOT, img_fname_no_ext) + ".png", argmax_layers)

    image["annotations"] = annotations
    return image


def main():
    # jsons_file = args.conditions + '_' + args.set + '.txt'
    jsons_file = 'test_annotation.txt'
    data_path = '/media/remus/datasets/DeepLGE/line_dataset/line_dataset_01/'
    with open(os.path.join(data_path, jsons_file)) as f:
        jsons = f.read().splitlines()
        jsons = [jason for jason in jsons]
        images = []

        # file_name = '02/usual/180418/bump1_003_bmp/front/Output_0_1524022595052933'
        for idx, file_name in enumerate(jsons):
            file_name = file_name.strip(".json")
            json_file_name = data_path + 'label/' + file_name + '.json'
            image_file_name = data_path + 'image/' + file_name + '.bmp'
            # label_path = '/home/remusm/projects/deeplab/arnia_deliver_usual'
            # labels_json = json.load(open(os.path.join(data_path, 'label', jason)))
            try:
                labels_json = json.load(open(os.path.join(json_file_name)))
            except:
                print("file doesnt exist", image_file_name)
                print(idx)
                continue

            image = filter_json_lane(idx, labels_json, image_file_name)
            images.append(image)
    # output_file = args.filter_by + '_' + args.conditions + '_' + args.set + '.json'
    # output_file = 'lane_train.json'
    #
    # f = open(os.path.join(data_path, 'label', output_file), 'w+')
    # json.dump({"images": images}, f)
    # f.close()
    print("done")

    with open("Output_test.txt", "w") as text_file:
        text_file.write('\n'.join(new_train_list))


if __name__ == '__main__':
    main()
