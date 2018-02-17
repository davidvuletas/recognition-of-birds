import cv2
import numpy as np


def erode(img):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def dilate(img):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def find_contours(img):
    _, image_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    picture_contours = sorted(contours, key=cv2.contourArea)
    return picture_contours


def average(num_of_colors_in_blocks, colors_for_remove):
    sum_of_colors = 0
    i = 0
    for c in num_of_colors_in_blocks:
        sum_of_colors += c[1]
    average_val = sum_of_colors / len(num_of_colors_in_blocks)
    for c in num_of_colors_in_blocks:
        if num_of_colors_in_blocks[i][1] >= average_val:
            colors_for_remove.append(c)
        i = i + 1
    return colors_for_remove
