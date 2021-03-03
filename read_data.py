import os

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

from consts import DEST_IMAGES_PATH, CASE_PATTERN, PNG_IN_CASE_PATTERN, DEST_KIDNEY_MASKS_PATH

images_dir = "../" + DEST_IMAGES_PATH
kidney_masks_dir = "../" + DEST_KIDNEY_MASKS_PATH


def get_images_and_masks(expected_img_width, expected_img_height, start_case_index, end_case_index,
                         should_resize=False):
    x_train = np.zeros((0, expected_img_width, expected_img_height, 3), dtype=np.uint8)
    y_train = np.zeros((0, expected_img_width, expected_img_height, 1), dtype=np.bool)
    for case_index in range(start_case_index, end_case_index + 1):
        number_of_images_in_case = len(os.listdir(images_dir + CASE_PATTERN.format(case_index)))
        case_x_train = np.zeros((number_of_images_in_case, expected_img_width, expected_img_height, 3), dtype=np.uint8)
        case_y_train = np.zeros((number_of_images_in_case, expected_img_width, expected_img_height, 1), dtype=np.bool)

        for png_index in range(0, number_of_images_in_case):
            case_x_train[png_index] = resize_and_get_image(expected_img_height, expected_img_width, case_index,
                                                           png_index, should_resize)
            case_y_train[png_index] = resize_and_get_mask(expected_img_height, expected_img_width, case_index,
                                                          png_index, should_resize)

        x_train = np.append(x_train, case_x_train, axis=0)
        y_train = np.append(y_train, case_y_train, axis=0)

    return x_train, y_train


def resize_and_get_mask(expected_img_height, expected_img_width, case_index, file_index, should_resize=False):
    img = imread(kidney_masks_dir + PNG_IN_CASE_PATTERN.format(case_index, file_index))
    if should_resize:
        img = resize(img, (expected_img_height, expected_img_width), mode='constant', preserve_range=True)
    grey_img = rgb2gray(img)
    grey_img = np.expand_dims(grey_img, axis=-1)
    return grey_img


def resize_and_get_image(expected_img_height, expected_img_width, case_index, file_index, should_resize=False):
    img = imread(images_dir + PNG_IN_CASE_PATTERN.format(case_index, file_index))
    if should_resize:
        img = resize(img, (expected_img_height, expected_img_width), mode='constant', preserve_range=True)
    return img
