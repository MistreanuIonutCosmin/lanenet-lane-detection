import os
import numpy as np
import cv2
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression


IMG_WIDTH = 993
THRESHOLD = 39
THRESHOLD_UNION = 20

color_map = [[20, 20, 20], [70, 70, 70], [120, 120, 120], [170, 170, 170], [220, 220, 220]]
# color_map = [np.array([255, 0, 0]), np.array([0, 255, 0]),np.array([0, 0, 255]), np.array([125, 125, 0]),
#              np.array([0, 125, 125])]

H_SAMPLES = np.array([200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236,
                      239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275,
                      278, 281, 284, 287, 290, 293, 296, 299, 302, 305, 308, 311, 314,
                      317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353,
                      356, 359, 362, 365, 368, 371, 374, 377, 380, 383, 386, 389, 392,
                      395, 398, 401, 404, 407, 410, 413, 416, 419, 422, 425, 428, 431,
                      434, 437, 440, 443, 446, 449, 452, 455, 458, 461, 464, 467, 470,
                      473, 476, 479, 482, 485, 488, 491, 494, 497, 500, 503, 506, 509,
                      512, 515, 518, 521, 524, 527, 530, 533, 536, 539, 542, 545, 548,
                      551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581, 584, 587,
                      590, 593, 596, 599, 602, 605, 608, 611, 614, 617, 620, 623, 626,
                      629, 632, 635, 638, 641, 644, 647, 650, 653, 656, 659, 662, 665,
                      668, 671, 674, 677, 680, 683, 686, 689, 692, 695, 698, 701, 704,
                      707, 710])

def get_angle(xs, y_samples):
    lr = LinearRegression()

    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    # xs, ys = xs[y_samples < 230], y_samples[y_samples < 230]
    if len(xs) > 1:
        lr.fit(ys[:, None], xs)
        k = lr.coef_[0]
        theta = np.arctan(k)
    else:
        theta = 0
    return theta

def get_indexes(x, xs):
    return [i for (y, i) in zip(xs, range(len(xs))) if x == y]


def clusters_union(points, prev_points, clusters):
    for p in points:
        _p = False
        m = IMG_WIDTH
        for prev in prev_points:

            if distance.euclidean(p, prev) < THRESHOLD_UNION:
                if m > distance.euclidean(p, prev):
                    m = distance.euclidean(p, prev)
                    point = prev
                _p = True

        if _p:
            for c in clusters:
                if point in c:
                    c.add(p)
            index = prev_points.index(point)
            prev_points.pop(index)
        else:

            clusters.append({p})


def clean_points(points, loop):
    img_mid_w = IMG_WIDTH / 2

    pop = True
    while loop > 0 and pop:
        pop = False
        for i in range(len(points) - 1):
            p1 = points[i][0]
            p2 = points[i + 1][0]

            if np.abs(p1 - p2) < THRESHOLD:
                pop = True
                if np.abs(img_mid_w - p1) > np.abs(img_mid_w - p2):
                    points.pop(i)
                    break
                else:
                    points.pop(i + 1)
                    break
        loop = loop - 1


def get_h_points(gray, image_mask):
    prev_points = None

    clusters = []

    for h in H_SAMPLES:
        inter = np.where(gray[h] != 0)
        points = inter[0]

        if len(points) < 1:
            continue

        ranges = sum((list([t[0]]) for t in zip(points, points[1:]) if t[0] + 1 != t[1]), [])
        indices = [get_indexes(x, points)[0] for x in ranges]

        indices = [0] + indices + [len(points)]
        indices = [int(indices[i - 1] + (indices[i] - indices[i - 1]) / 2) for i in range(1, len(indices))]

        points = [(points[i], h) for i in indices]

        clean_points(points, 50)

        if prev_points:
            clusters_union(points, prev_points, clusters)
        else:
            for p in points:
                clusters.append({p})
        prev_points = points

    k = 0
    for c in clusters:
        if len(c) > 20:

            color = color_map[k]
            k = k + 1 if k < 4 else 0

            for p in c:
                cv2.circle(image_mask, p, radius=1, color=color, thickness=2)
    # cv2.imshow("test", image_mask)
    # cv2.waitKey(0)

    return clusters


def get_equation_from_cc(image_mask, mask):
    boundaries = [
        ([50, 50, 50], [255, 255, 255]),
        # ([0, 0, 0], [0, 0, 0]),
    ]

    # black = np.ones(np.shape(image_mask))

    # loop over the boundaries and get white pixels as a mask
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        m = cv2.inRange(image_mask, lower, upper)
        image_mask = cv2.bitwise_and(image_mask, image_mask, mask=m)

    # gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    image_mask[image_mask != 0] = 255

    gray = np.where(mask == [255, 255, 255])
    image_mask[gray] = 0

    gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    clusters = get_h_points(gray, image_mask)
    gt_mask = np.uint8(np.zeros(np.shape(image_mask)))

    interval_angle_remove = (-1.28, 1.28)
    k = 0
    lines = []

    for c in clusters:
        if len(c) > 20:
            coord = list(c)
            coord.sort(key=lambda c: c[1])
            xs = np.array([x[0] for x in coord])
            y = np.array([x[1] for x in coord])

            angle = get_angle(xs, y)
            index = np.where(y == 284)
            # if len(index[0]) < 1:
            #     index = np.where(y == 251)

            if interval_angle_remove[0] < angle < interval_angle_remove[1]:
                lines.append((xs[index], coord))

    lines.sort(key=lambda line: -len(line[1]))
    lines = lines[:4]
    lines.sort(key=lambda line: line[0])

    for line in lines:
        color = color_map[k]
        cv2.polylines(img=gt_mask, pts=np.int32([line[1]]), isClosed=False, color=color, thickness=8)
        k = k + 1

    return gt_mask


if __name__ == '__main__':
    # val_file = open("/media/remus/datasets/AVMSnapshots/AVM/val.txt", "r")
    #

    # IMG_WIDTH = 416
    # IMG_HEIGHT = 288
    CROP_MIN = 143
    CROP_MAX = 1136

    test_seg = "/media/remus/datasets/AVMSnapshots/AVM/segmentation/0014_RoadCamera.png"
    test_image = "/media/remus/datasets/AVMSnapshots/AVM/original_images/0014_AVMFrontCamera.png"

    not_ok_path = "/media/remus/datasets/AVMSnapshots/AVM/images_not_ok.txt"
    images_path = "/media/remus/datasets/AVMSnapshots/AVM/original_images/"
    seg_path = "/media/remus/datasets/AVMSnapshots/AVM/segmentation/"
    mask_path = "/media/remus/datasets/AVMSnapshots/AVM/road_mask.png"
    instance_path = "/media/remus/datasets/AVMSnapshots/AVM/instance/"
    binary_path = "/media/remus/datasets/AVMSnapshots/AVM/binary/"
    mask = cv2.imread(mask_path)

    # gt = get_equation_from_cc(cv2.imread(test_seg), mask)
    # image = cv2.imread(test_image)
    # gt = cv2.addWeighted(gt, 0.7, image, 0.7, 0)
    # cv2.imshow("test", gt)
    # cv2.waitKey(0)

    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    if not os.path.exists(binary_path):
        os.makedirs(binary_path)

    segmentations = os.listdir(seg_path)
    segmentations.sort()

    images = os.listdir(images_path)
    images.sort()

    if not_ok_path is not None:
        not_ok_file = open(not_ok_path, 'r')
        not_ok = not_ok_file.readlines()
        not_ok = map(lambda x: x.rstrip(), not_ok)

    for name, img in zip(segmentations, images):
        print(name)

        img_name = name.strip('0').split('_')[0]

        if not_ok_path is not None and img_name in not_ok:
            continue

        # image = cv2.imread(images_path + img)
        gt = np.uint8(get_equation_from_cc(cv2.imread(seg_path + name), mask))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt = gt[:, CROP_MIN:CROP_MAX]

        gt = cv2.resize(gt, (416, 288), interpolation=cv2.INTER_NEAREST)
        # gt = cv2.addWeighted(gt, 0.5, image, 0.5, 0)
        cv2.imwrite(instance_path + img, gt)

        gt[gt != 0] = 255
        cv2.imwrite(binary_path + img, gt)
