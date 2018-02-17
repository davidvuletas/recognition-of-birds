import os
from copy import deepcopy
from time import time

import pyximport

from blocks import *
from color_operations import remove_colors
from cython_code.ch_cl import new_colors
from utils import *
from neural_network import predict

pyximport.install()


def resize_picture(image):
    return cv2.resize(image, (300, 200))


if __name__ == '__main__':
    mainFolder = 'demo'
    command = input('\tIf you want to see how picture is processed by steps press \'1\',\n'
                    ' or if you want to see what is accuracy of trained network, press \'2\' ')
    if command == '1':
        for type_of_bird in os.listdir(mainFolder):
            for pic in os.listdir(mainFolder + '//' + type_of_bird):
                img = cv2.imread(mainFolder + '//' + type_of_bird + '//' + pic)

                img = resize_picture(img)
                img_orig = img.copy()
                cv2.imshow('origin', img)
                cv2.waitKey(0)
                img = cv2.GaussianBlur(img, (5, 5), 0)

                print('starting')
                start = time()
                img = new_colors(img)
                print('ending', time() - start)

                cv2.imshow('changed_colors', img)
                cv2.waitKey(0)

                (h, w, c) = img.shape
                top, left, right, down = divide_margins(img, h, w)
                all_blocks = [get_blocks(top), get_blocks(left), get_blocks(down), get_blocks(right)]

                num_of_colors_in_blocks = get_all_colors_from_blocks(all_blocks)
                colors_for_remove = deepcopy(num_of_colors_in_blocks)
                colors_for_remove.clear()
                colors_for_remove = average(num_of_colors_in_blocks, colors_for_remove)
                img = remove_colors(colors_for_remove, img)
                cv2.imshow('removed colors from img', img)
                cv2.waitKey(0)

                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_erode = erode(img_gray)

                img_erode = dilate(img_erode)

                cv2.imshow('img after erode', img_erode)
                cv2.waitKey(0)

                img_erode = 255 - img_erode

                cv2.imshow('img after invert', img_erode)
                cv2.waitKey(0)
                picture_contours = find_contours(img_erode)
                mask = np.zeros(img_orig.shape, np.uint8)
                mask.fill(255)
                cv2.drawContours(mask, [picture_contours[-1]], 0, (0, 0, 0), -1)
                cv2.imshow('largest contour', mask)
                cv2.waitKey(0)

                removed = cv2.bitwise_or(img_orig, mask)

                cv2.imshow('processed image', removed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if not os.path.exists('processed_images/' + type_of_bird):
                    os.makedirs('processed_images/' + type_of_bird)
                cv2.imwrite('processed_images/' + type_of_bird + '/' + pic, removed)
    if command == '2':
        predict('trained_network.h5')
