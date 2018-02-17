
import cython
import numpy as np
cimport numpy as np
from color_operations import find_nearest_color
from colorspace import Color

@cython.boundscheck(False)
def new_colors(unsigned char [:,:,:] image):
    cdef int x, y, w, h
    h = image.shape[0]
    w = image.shape[1]
    narr = np.asarray(image)
    for x in range(0,h):
        for y in range(0,w):
            co = find_nearest_color(image[x,y])
            hex = Color[co]
            hex = hex[1:]
            rgb = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
            narr[x][y]=[rgb[0],rgb[1],rgb[2]]
    return narr
