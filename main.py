from __future__ import print_function
#import potrebnih biblioteka
import cv2
import collections
from colour import Color
import os as os
from matplotlib import  colors as mcolors

import numpy as np
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt


if __name__ == '__main__':
    colors = dict(mcolors.BASE_COLORS,**mcolors.CSS4_COLORS)
    imgplot= plt.imread('D:\Workspace\Pycharm\SOFT\ptica.jpg')
    plt.hist(imgplot)