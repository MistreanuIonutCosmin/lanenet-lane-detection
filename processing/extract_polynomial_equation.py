import os
import cv2
import numpy as np
import json
import sys
from sklearn.linear_model import LinearRegression

lr = LinearRegression()


def get_angle(xs, y_samples):
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    # xs, ys = xs[y_samples < 230], y_samples[y_samples < 230]
    if len(xs) > 1:
        lr.fit(ys[:, None], xs)
        k = lr.coef_[0]
        theta = np.arctan(k)
    else:
        theta = 0
    return theta


# improve by
def get_equation_from_cc(image_mask, image_name, degree):
    # orig_mask = np.copy(image_mask)
    boundaries = [
        ([100, 100, 100], [255, 255, 255]),
    ]

    # loop over the boundaries and get white pixels as a mask
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        m = cv2.inRange(image_mask, lower, upper)
        image_mask = cv2.bitwise_and(image_mask, image_mask, mask=m)

    gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    gray[gray != 0] = 255
    gray[:210, :] = 0

    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)

    gray = cv2.dilate(gray, np.ones((3, 3)), iterations=1)
    gray = cv2.erode(gray, np.ones((3, 1)), iterations=1)
    # gray = cv2.dilate(gray, np.ones((3, 3)), iterations=1)
    # gray = cv2.erode(gray, np.ones((3, 1)), iterations=1)

    labelnum, labelimg, conturs, _ = cv2.connectedComponentsWithStats(gray)

    minimum_cc_sum = 250
    height, width = np.shape(gray)

    vanish_x, vanish_y = get_vanish_point(conturs, degree, labelimg, labelnum, width)

    left_lines = []
    right_lines = []
    image_line_equations = {"lines": [], "vanish_point": [vanish_x, vanish_y], "image_name": image_name}

    interval_angle_remove = (-1.3, 1.3)
    interval_angle_degree = (-0.53, 0.53)

    for label in range(1, labelnum):
        _, _, _, _, size = conturs[label]

        if size < minimum_cc_sum:
            gray[np.where(labelimg == label)] = 0
        else:
            y, x = np.where(labelimg == label)

            points = zip(y, x)
            if len(points) > 20000:
                points = np.array(list(dict(points).items()))

            # sample some points from the CC on which is fitted the polynom
            choice = np.random.choice([True, False], size=size, p=[0.05, 0.95])

            # TODO points[choice][]
            poly_x = np.array([])
            poly_y = np.array([])

            for i in xrange(len(points)):
                if choice[i]:
                    poly_x = np.append(poly_x, points[i][1])
                    poly_y = np.append(poly_y, points[i][0])

            if degree == 2:
                for p in range(0, 2):
                    poly_y = np.append(poly_y, vanish_y)
                    poly_x = np.append(poly_x, vanish_x)

            p = np.polyfit(poly_x, poly_y, degree)

            angle = get_angle(poly_x, poly_y)

            # if interval_angle_degree[0] < angle < interval_angle_degree[1]:
            #     ind = len(poly_y)
            #     ind = [ind - 2, ind - 1]
            #
            #     poly_x = np.delete(poly_x, ind)
            #     poly_y = np.delete(poly_y, ind)
            #     p = np.polyfit(poly_x, poly_y, degree - 1)

            # print(angle)
            if interval_angle_remove[0] < angle < interval_angle_remove[1]:
                if angle < 0:
                    left_lines.append((angle, p))

                else:
                    right_lines.append((angle, p))

    left_lines.sort(key=lambda line: -line[0])
    right_lines.sort(key=lambda line: line[0])

    # pick 2 lines on right and 2 on the left

    # left_lines = left_lines[:2]
    # right_lines = right_lines[:2]

    line_id = 4
    for left, polynom_weights in left_lines:
        image_line_equations["lines"].append({"id": line_id, "weights": list(polynom_weights)})
        line_id -= 1

    line_id = 6
    for right, polynom_weights in right_lines:
        image_line_equations["lines"].append({"id": line_id, "weights": list(polynom_weights)})
        line_id += 1

    return image_line_equations


# Identify the closest two lanes to the car
# Computing their equation
# Find the x intercept
def get_vanish_point(conturs, degree, labelimg, labelnum, width):
    components_sizes = []
    for label in range(1, labelnum):
        _, _, _, _, size = conturs[label]
        components_sizes.append((size, label))
    components_sizes.sort()
    polynom_deg = degree
    components_poly = np.array([np.ones(degree + 1), np.ones(degree + 1)])
    comp_no = 0
    for component in components_sizes[-2:]:

        y, x = np.where(labelimg == component[1])

        points = zip(y, x)

        step = len(points) / 50
        poly_points = np.array([])
        ind = step
        poly_x = np.array([])
        poly_y = np.array([])

        while ind < len(points):
            np.append(poly_points, points[ind])

            poly_x = np.append(poly_x, points[ind][1])
            poly_y = np.append(poly_y, points[ind][0])
            ind += step

        p = np.polyfit(poly_x, poly_y, polynom_deg)
        components_poly[comp_no] = p
        comp_no += 1

    vanish_point_eq = components_poly[0] - components_poly[1]
    vanish_x_ = np.roots(vanish_point_eq)

    if len(vanish_x_) > 1:
        vanish_x = vanish_x_[0] if np.abs(vanish_x_[0] - width / 2) < np.abs(vanish_x_[1] - width / 2) else vanish_x_[1]
    else:
        vanish_x = vanish_x_

    vanish_y = np.polyval(components_poly[0], vanish_x)

    return int(vanish_x), int(vanish_y - 15)


if __name__ == '__main__':

    if len(sys.argv) < 4:
        raise Exception(
            'Incorrect number or args! Script expects: [1-images_path, 2-segmentations_path, 3-json_path, '
            '4-not_ok_path(optional)]')

    image_path = sys.argv[1]
    segmentation_path = sys.argv[2]
    json_output_path = sys.argv[3]
    not_ok_path = None

    if len(sys.argv) == 5:
        not_ok_path = sys.argv[4]

    avm_images = os.listdir(image_path)
    avm_seg = os.listdir(segmentation_path)

    avm_images.sort()
    avm_seg.sort()

    # extract equations for the whole dataset
    images_equations_list = []

    # read a list with the images which are bad from dataset and these will be skipped in computing lines
    if not_ok_path is not None:
        not_ok_file = open(not_ok_path, 'r')
        not_ok = not_ok_file.readlines()
        not_ok = map(lambda x: x.rstrip(), not_ok)
    #
    for i in xrange(len(avm_seg)):
        img_name = avm_images[i]
        img_name = img_name.strip('0').split('_')[0]

        if not_ok_path is not None and img_name in not_ok:
            continue

        print ("Image number: ", i)

        avm_mask = cv2.imread(segmentation_path + avm_seg[i])
        images_equations_list.append(get_equation_from_cc(avm_mask, avm_images[i], degree=2))

    # output_file = open(json_output_path, 'w')
    # output_file.write(json.dumps(images_equations_list))
    # output_file.close()

    # mask = cv2.imread(segmentation_path + "2491_RoadCamera.png")
    # image_equations = get_equation_from_cc(mask, "name", 2)
    # print(image_equations)
    # equations = json.dumps(image_equations)
    # print(equations)
