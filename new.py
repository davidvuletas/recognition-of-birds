from operator import itemgetter
import colorsys
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    im = cv2.imread('D:\Workspace\Pycharm\SOFT\p3.jpg')
    im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    im[im >= 128] = 255
    im[im < 128] = 0
    cv2.imshow('out.jpg', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()