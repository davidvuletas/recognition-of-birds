import bz2
import csv
from time import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
from colormath.color_objects import LabColor
from colormath.color_diff_matrix import delta_e_cie1976,delta_e_cie1994,delta_e_cmc, delta_e_cie2000
import os
folderPath = 'D:\Workspace\Pycharm\SOFT\pictures\\'
imgPath = 'p2.jpg'

def detectColour(img):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    (w,h,c) = img.shape
    img = Image.fromarray(img)
    basewidth = 80
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    reader = csv.DictReader(bz2.open('D:\Workspace\Pycharm\SOFT\colors\lab_matrix.csv.bz2',mode='rt'))
    lab_matrix = np.array([list(map(float, row.values())) for row in reader])
    #f = open('D:\Workspace\Pycharm\SOFT\lab-matrix.pk', 'rb')
    #u = pickle._Unpickler(f)
    #u.encoding = 'latin1'
    #lab_matrix = u.load()
    img = np.array(img)
    (w,h,ch) = img.shape
    print(w,h,c)

    print('waiting.....')
    start = time()
    for x in range(w):
        for y in range(h):
            print(x,y)
            color = LabColor(lab_l=img[x][y][0]/(255/100),
                             lab_a=img[x][y][1]-128,
                             lab_b=img[x][y][2]-128)
            color = np.array([color.lab_l,color.lab_a,color.lab_b])
            delta = delta_e_cie1994(color, lab_matrix)


            # find the closest match to `color` in `lab_matrix`
            nearest_color = lab_matrix[np.argmin(delta)]

            img[x][y][0] = nearest_color[0]
            img[x][y][1] = nearest_color[1]
            img[x][y][2] = nearest_color[2]
            #print('%s is closest to %s' % (color, nearest_color))
    print('finished')
    print('time for work',time()-start)
    return img
    #print('%s is closest to %s' % (color, nearest_color))

if __name__ == '__main__':
    folderPath = 'D:\Workspace\Pycharm\SOFT\pictures\\'
    pic = 'p2.jpg'
    # for pic in os.listdir(folderPath):
    #     if '_seg' not in pic:
    #
    img = cv2.imread(str(folderPath+pic))
    img = detectColour(img)
    cv2.imwrite(str(folderPath+pic[:-4]+'_seq.jpg'),img)

    #img = cv2.imread(folderPath+'pil.jpg')
    #img = cv2.resize(img,(200,300))
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()