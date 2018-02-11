import cv2
import numpy as np


def erode(img):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img
