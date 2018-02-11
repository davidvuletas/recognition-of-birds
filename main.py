from copy import deepcopy
from time import time

import cv2
import numpy as np
import pyximport
from PIL import Image

import utils
from blocks import get_blocks, getAllColorsFromBlocks

pyximport.install()
from cython_code.ch_cl import newcolors


def resize_picture(img):
    img = Image.fromarray(img)
    basewidth = 250
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return np.array(img)


if __name__ == '__main__':
    startTime = time()
    folderPath = 'D:\Workspace\Pycharm\SOFT\pictures\\001.Black_Footed_Albatross'
    # for pic in os.listdir(folderPath):
    colors = []
    img = cv2.imread('D:\Workspace\Pycharm\SOFT\pictures\\010.Red_Winged_Blackbird\Red_Winged_Blackbird_0025_5342.jpg')
    #img = cv2.imread(folderPath + '\\' + 'Black_Footed_Albatross_0008_796083.jpg')
    imgOrig = img.copy()
    img = resize_picture(img)
    imgOrig = resize_picture(imgOrig)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('g', img)
    cv2.waitKey(0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    print('starting')
    start = time()
    img = newcolors(img)
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

    img = Image.fromarray(img)
    img = np.array(img)
    converted = []

    # colors_for_remove = colors_for_remove[0:math.floor(len(colors_for_remove)/6)]
    for color in colors_for_remove:
        a, b, c = color[0].replace(']', '').replace('[', '').replace(' ', '').split(',')
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if (img[x][y][0] == int(a) and
                        img[x][y][1] == int(b) and
                        img[x][y][2] == int(c)):
                    img[x][y][0] = 255
                    img[x][y][1] = 255
                    img[x][y][2] = 255

    cv2.imwrite('D:\Workspace\Pycharm\SOFT\generated\\003.Sooty_Albatross\dasd.jpg',
                 cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cv2.imshow('change', img)
    cv2.waitKey(0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_erode = utils.erode(img_gray)
    img_erode = 255 - img_erode
    _, image_bin = cv2.threshold(img_erode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_final = imgOrig.copy()
    picture_contours = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    picture_contours = sorted(contours, key=cv2.contourArea)
    #cv2.drawContours(img_final, picture_contours[-1], -1, (255, 0, 0), 2)
    #picture_contours = picture_contours[:-1]
    mask = np.zeros(img_final.shape, np.uint8)

    mask = np.ones(img.shape, np.uint8)
    cv2.drawContours(mask, [picture_contours[-1]], 0, (255, 255, 255), -1)
    removed = cv2.bitwise_and(img, mask)
    # c = picture_contours[-1]
    # rect = cv2.boundingRect(c)
    # x, y, w, h = rect
    # box = cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cropped = img_final[y: y + h, x: x + w]
    cv2.imshow('final_img', removed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
