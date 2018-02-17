from colorspace import *


def rgb_from_str(s):
    # s starts with a #.
    r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    return r, g, b


def find_nearest_color(points):
    (R, G, B) = points
    mindiff = None
    for key, value in Color.items():
        r, g, b = rgb_from_str(value)
        diff = (R - r) * (R - r) + (G - g) * (G - g) + (B - b) * (B - b)
        if mindiff is None or diff < mindiff:
            mindiff = diff
            mincolorname = key
    return mincolorname


def remove_colors(colors_for_remove, img):
    for color in colors_for_remove:
        r, g, b = color[0].replace(']', '').replace('[', '').replace(' ', '').split(',')
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if (img[x][y][0] == int(r) and
                        img[x][y][1] == int(g) and
                        img[x][y][2] == int(b)):
                    img[x][y][0] = 255
                    img[x][y][1] = 255
                    img[x][y][2] = 255
    return img
