from __future__ import print_function
#import potrebnih biblioteka
import struct
import matplotlib.colors as mcolors
import binascii
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from PIL import Image
from operator import itemgetter
import os
NUM_CLUSTERS = 5


if __name__ == '__main__':
    picture = 'p15.jpg'
    folderPath = 'D:\Workspace\Pycharm\SOFT\pictures\\'
    #for pic in os.listdir(folderPath):
    image = cv2.imread(str(folderPath + picture))
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=8)
    labels = clt.fit_predict(image)
    print(clt.cluster_centers_)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(folderPath+picture[0:-3]+'_seg'+'.jpg'), np.hstack([quant]))

    im = Image.open(str(folderPath+picture[0:-3]+'_seg'+'.jpg'))
    im = im.resize((300, 300),Image.ANTIALIAS)
    s = im.size
    print(s[0])
    pixels = im.getcolors(s[0]*s[1])
    pixels = sorted(pixels, key=itemgetter(0))

    colors_for_remove = pixels[-6:]

    m = im.load()
    for x in range(s[0]):
        for y in range(s[1]):
            current_color = im.getpixel((x, y))
            for color in colors_for_remove:
                if current_color == color[1]:
                    im.putpixel((x, y), (255,0,0))

    #cv2.imwrite(str(folderPath+'p12.jpg'[0:-3]+'_seg'+'.jpg'), np.array(im))
    cv2.imshow('picture',np.array(im))
    cv2.waitKey(0)
    cv2.destroyAllWindows()