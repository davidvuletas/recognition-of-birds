import math
import operator
from copy import deepcopy
from time import time

import cv2
import numpy as np
import pyximport
from PIL import Image

pyximport.install()
from cython_code.ch_cl import newcolors


def resizePicture(img):
    img = Image.fromarray(img)
    basewidth = 200
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return np.array(img)


# def changeColors(img):
#     reader = csv.DictReader(bz2.open('D:\Workspace\Pycharm\SOFT\colors\lab_matrix.csv.bz2', mode='rt'))
#     lab_matrix = np.array([list(map(float, row.values())) for row in reader])
#     colors = []
#     img = resizePicture(img)
#     (w, h, c) = img.shape
#
#     print('waiting.....')
#     start = time()
#     for x in range(w):
#         for y in range(h):
#             color = LabColor(lab_l=img[x][y][0] / (255 / 100),
#                              lab_a=img[x][y][1] - 128,
#                              lab_b=img[x][y][2] - 128)
#             color = np.array([color.lab_l, color.lab_a, color.lab_b])
#             delta = delta_e_cie1976(color, lab_matrix)
#
#             # find the closest match to `color` in `lab_matrix`
#             nearest_color = lab_matrix[np.argmin(delta)]
#             if (nearest_color.tolist() not in colors):
#                 colors.append(nearest_color.tolist())
#             # img = img.astype(np.int)
#             img = img.astype(np.float)
#             img[x][y] = nearest_color
#
#     print('finished')
#     print('time for work', time() - start)
#     return img, colors


def get_blocks(side):
    b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = []
    i = 0
    for row in side:
        splitted_row = np.array_split(row, 8)
        if i == 0:
            b1 = splitted_row[0]
            b2 = splitted_row[1]
            b3 = splitted_row[2]
            b4 = splitted_row[3]
            b5 = splitted_row[4]
            b6 = splitted_row[5]
            b7 = splitted_row[6]
            b8 = splitted_row[7]
            i += 1
            continue

        b1 = np.concatenate([b1, splitted_row[0]])
        b2 = np.concatenate([b2, splitted_row[1]])
        b3 = np.concatenate([b3, splitted_row[2]])
        b4 = np.concatenate([b4, splitted_row[3]])
        b5 = np.concatenate([b5, splitted_row[4]])
        b6 = np.concatenate([b6, splitted_row[5]])
        b7 = np.concatenate([b7, splitted_row[6]])
        b8 = np.concatenate([b8, splitted_row[7]])

    b1 = np.unique(b1, axis=0)
    b2 = np.unique(b2, axis=0)
    b3 = np.unique(b3, axis=0)
    b4 = np.unique(b4, axis=0)
    b5 = np.unique(b5, axis=0)
    b6 = np.unique(b6, axis=0)
    b7 = np.unique(b7, axis=0)
    b8 = np.unique(b8, axis=0)

    return [b1, b2, b3, b4, b5, b6, b7, b8]


def getAllColorsFromBlocks(all_blocks):
    num_of_colors_in_blocks = {}
    for segment in all_blocks:
        for block in segment:
            for c in block:
                if str(c.tolist()) in num_of_colors_in_blocks.keys():
                    num_of_colors_in_blocks[str(c.tolist())] += 1
                else:
                    num_of_colors_in_blocks[str(c.tolist())] = 1
    num_of_colors_in_blocks = sorted(num_of_colors_in_blocks.items(), key=operator.itemgetter(1), reverse=True)
    return num_of_colors_in_blocks


if __name__ == '__main__':
    folderPath = 'D:\Workspace\Pycharm\SOFT\pictures\\'
    pic = '004.Groove_Billed_Ani\Groove_Billed_Ani_0015_1653.jpg'
    colors = []
    img = cv2.imread(str(folderPath + pic))
    img = resizePicture(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print('starting')
    start = time()
    img = newcolors(img)
    print('ending', time() - start)
    (h, w, c) = img.shape
    top = img[0:int(h / 8), int(w/8):7 * int(w / 8)]

    left = img[0:h, 0:int(w / 8)]
    down = img[7 * int(h / 8):, int(w/8):7 * int(w / 8)]
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
        i = i+1

    img = Image.fromarray(img)
    img = np.array(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    converted = []

    #colors_for_remove = colors_for_remove[0:math.floor(len(colors_for_remove)/3)]
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
            # converted = np.where(img == [int(a), int(b), int(c)])
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #img = cv2.Canny(img,100,200)
    _, tresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(tresh, connectivity=4)
    # sizes = stats[:, -1]
    #
    # max_label = 1
    # max_size = sizes[1]
    # for i in range(2, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         max_size = sizes[i]
    #
    # img2 = np.zeros(output.shape)
    # img2[output == max_label] = 255
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('ed', tresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
