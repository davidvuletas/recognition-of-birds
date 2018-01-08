import bz2
import csv
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

from colormath.color_objects import LabColor
import operator
if __name__ == '__main__':
    reader = csv.DictReader(bz2.open('D:\Workspace\Pycharm\SOFT\colors\lab_matrix.csv.bz2',mode='rt'))
    lab_matrix = np.array([list(map(float, row.values())) for row in reader])
    lab_matrix = sorted(lab_matrix,key = lambda x: (x[0], x[1],x[2]))
    #lab_matrix = np.load('D:\Workspace\Pycharm\SOFT\lab-matrix.pk')
    # color_list = pickle.load(open('D:\Workspace\Pycharm\SOFT\colors\lab-colors.pk','rb'))  # file pointer is now at end of first object
    # f = open('D:\Workspace\Pycharm\SOFT\colors\lab-matrix.pk','rb')
    # u = pickle._Unpickler(f)
    # u.encoding = 'latin1'
    # color_matrix = u.load()
    # lab_matrix = {}
    # for cn,cm in zip(color_list,color_matrix):
    #     if cn in lab_matrix.keys():
    #         list = cm.tolist()
    #         lab_matrix[cn].append((list[0],list[1],list[2]))
    #     else:
    #         list = cm.tolist()
    #         lab_matrix[cn] = [(list[0],list[1],list[2])]
    #
    img = np.zeros((1024, 1024, 3), np.float64)
    i = 4
    for c in lab_matrix:
         i = i + 10
         cv2.rectangle(img,(10,i),(80+i,50+i),(c[0],c[1],c[2]),1)
    cv2.imshow('vla',img)
    cv2.waitKey(0)
    img = cv2.imread('D:\Workspace\Pycharm\SOFT\pictures\p7.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('img',img)
    cv2.waitKey(0)