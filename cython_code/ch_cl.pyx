
import cython
import numpy as np
cimport numpy as np
from rgb_find_nearest import ColorNames

@cython.boundscheck(False)
def new_colors(unsigned char [:,:,:] image):
    cdef int x, y, w, h
    h = image.shape[0]
    w = image.shape[1]
    narr = np.asarray(image)
    for x in range(0,h):
        for y in range(0,w):
            co = ColorNames.findNearestColor(image[x,y])
            hex = ColorNames.Color[co]
            hex = hex[1:]
            rgb = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
            narr[x][y]=[rgb[0],rgb[1],rgb[2]]
    return narr


@cython.boundscheck(False)
def remove_colors(int r,int b,int g,unsigned char[:,:,:] img):
    cdef int x,y,w, h
    h = img.shape[0]
    w = img.shape[1]
    narr = np.asarray(img)

    for x in range(0,h):
        for y in range(0,w):
            if (narr[x][y][0] == r and
                    narr[x][y][1] == g and
                    narr[x][y][2] == b):
                narr[x][y] = [255,255,255]

    return narr
