
import cython
import numpy as np
cimport numpy as np
from rgb_find_nearest import ColorNames

@cython.boundscheck(False)
def newcolors(unsigned char [:,:,:] image):
    #it1 = np.nditer(img, flags=['multi_index'])
    cdef int x, y, w, h
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    narr = np.asarray(image)
    for x in range(0,h):
        for y in range(0,w):
            co = ColorNames.findNearestColorName(image[x,y],ColorNames.Color)
            hex = ColorNames.Color[co]
            hex = hex[1:]
            rgb = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
            narr[x][y]=[rgb[0],rgb[1],rgb[2]]
    return narr

