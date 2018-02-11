from copy import deepcopy
from time import time

import cv2
import numpy as np
import pyximport

import utils
from blocks import get_blocks, getAllColorsFromBlocks

pyximport.install()
from cython_code.ch_cl import new_colors
import os


def resize_picture(img):
    return cv2.resize(img, (300, 200))


if __name__ == '__main__':
    mainFolder = 'pictures/TRAIN'
    for type in os.listdir(mainFolder):
        for pic in os.listdir(mainFolder + '//' + type):
            img = cv2.imread(mainFolder + '//' + type + '//' + pic)
            colors = []
            imgOrig = img.copy()
            img = resize_picture(img)
            imgOrig = resize_picture(imgOrig)
            # cv2.imshow('g', img)
            # cv2.waitKey(0)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            print('starting')
            start = time()
            img = new_colors(img)
            print('ending', time() - start)
            (h, w, c) = img.shape
            top = img[0:int(h / 8), int(w / 8):7 * int(w / 8)]

            left = img[0:h, 0:int(w / 8)]
            down = img[7 * int(h / 8):, int(w / 8):7 * int(w / 8)]
            right = img[0:h, 7 * int(w / 8):]

            all_blocks = [get_blocks(top), get_blocks(left), get_blocks(down), get_blocks(right)]

            num_of_colors_in_blocks = getAllColorsFromBlocks(all_blocks)
            colors_for_remove = deepcopy(num_of_colors_in_blocks)
            colors_for_remove.clear()
            i = 0
            suma = 0
            for c in num_of_colors_in_blocks:
                suma += c[1]
            average = suma / len(num_of_colors_in_blocks)
            for c in num_of_colors_in_blocks:
                if num_of_colors_in_blocks[i][1] >= average:
                    colors_for_remove.append(c)
                i = i + 1

            for color in colors_for_remove:
                r, g, b = color[0].replace(']', '').replace('[', '').replace(' ', '').split(',')
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        if (img[x][y][0] == int(r) and
                                img[x][y][1] == int(g) and
                                img[x][y][2] == int(b)):
                            img[x][y][0] = 255
                            img[x][y][1] = 255
                            img[x][y][2] = 255

            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_erode = utils.erode(img_gray)
            img_erode = 255 - img_erode
            _, image_bin = cv2.threshold(img_erode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, contours, _ = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            img_final = imgOrig.copy()
            picture_contours = []
            picture_contours = sorted(contours, key=cv2.contourArea)
            mask = np.zeros(img_final.shape, np.uint8)
            mask.fill(255)
            cv2.drawContours(mask, [picture_contours[-1]], 0, (0, 0, 0), -1)
            removed = cv2.bitwise_or(img_final, mask)
            if not os.path.exists('processed_image/' + type):
                os.makedirs('processed_image/' + type)
            cv2.imwrite('processed_image/' + type + '/' + pic, removed)
