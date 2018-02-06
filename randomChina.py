import numpy as np
import pickle
import cv2


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

if __name__ == '__main__':

    # color_list = pickle.load(
    #     open('D:\Workspace\Pycharm\SOFT\colors\lab-colors.pk', 'rb'))  # file pointer is now at end of first object
    # f = open('D:\Workspace\Pycharm\SOFT\colors\lab-matrix.pk', 'rb')
    # objects = []
    # u = pickle._Unpickler(f)
    # u.encoding = 'latin1'
    # color_matrix = u.load()
    # lab_matrix = {}
    # for cn, cm in zip(color_list, color_matrix):
    #     lab_matrix[cn] = [(cm.tolist()[0],cm.tolist()[1],cm.tolist()[2])]
    #     lab_matrix[cn].append((1,2,3))
    # print(lab_matrix)

    # Read image
    im = cv2.imread("D:\Workspace\Pycharm\SOFT\pictures\p1_seq.jpg", cv2.IMREAD_GRAYSCALE)
    #edges = cv2.Canny(im, 150,150)
    blurred = cv2.GaussianBlur(im, (3, 3), 0)
    edges = cv2.Canny(blurred,200,200)
    #x, y, w, h = cv2.boundingRect(c)
    # draw the book contour (in green)
    #cv2.rectangle(edges, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('img,',edges)
    cv2.waitKey(0)