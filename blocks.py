import operator

import numpy as np


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


def divide_margins(img, h, w):
    top = img[0:int(h / 8), int(w / 8):7 * int(w / 8)]
    left = img[0:h, 0:int(w / 8)]
    down = img[7 * int(h / 8):, int(w / 8):7 * int(w / 8)]
    right = img[0:h, 7 * int(w / 8):]
    return top, left, right, down


def get_all_colors_from_blocks(all_blocks):
    num_of_colors_in_blocks = {}
    for margin in all_blocks:
        for block in margin:
            for color in block:
                if str(color.tolist()) in num_of_colors_in_blocks.keys():
                    num_of_colors_in_blocks[str(color.tolist())] += 1
                else:
                    num_of_colors_in_blocks[str(color.tolist())] = 1
    num_of_colors_in_blocks = sorted(num_of_colors_in_blocks.items(), key=operator.itemgetter(1), reverse=True)
    return num_of_colors_in_blocks
