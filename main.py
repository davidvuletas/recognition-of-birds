from __future__ import print_function
#import potrebnih biblioteka
import struct
import matplotlib.colors as mcolors
import binascii
import cv2
from colour import Color
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from PIL import Image
NUM_CLUSTERS = 5


if __name__ == '__main__':
    im = Image.open("D:\Workspace\Pycharm\SOFT\ptica.png")
    color = max(im.getcolors(im.size[0] * im.size[1]))

    '''
    print('reading image')
    im = cv2.imread('D:\Workspace\Pycharm\SOFT\ptica.jpg')
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    #print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences

    index_max = scipy.argmax(counts)  # find most frequent
    peak = codes[index_max]
    colour = ''.join(chr(int(c)) for c in peak)
    #print('most frequent is %s (#%s)' % (peak, colour.encode('hex')))
   # colors = dict(mcolors.BASE_COLORS,**mcolors.CSS4_COLORS)

    '''
    n = Image.open('D:\Workspace\Pycharm\SOFT\ptica.jpg')
    n = n.resize((300, 300),Image.ANTIALIAS)

    m = n.load()

    # get x,y size
    s = n.size

    # Process every pixel
    for x in range(s[0]):
        for y in range(s[1]):
            current_color = n.getpixel((x, y))
            ####################################################################
            # Do your logic here and create a new (R,G,B) tuple called new_color
            ####################################################################
            if current_color == color:
                n.putpixel((x, y), (255,0,0))

    n.save('sans_red.jpg', "JPEG")
    img = cv2.imread('D:\Workspace\Pycharm\SOFT\sans_red.jpg')
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
