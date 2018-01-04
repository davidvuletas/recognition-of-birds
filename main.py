from __future__ import print_function
#import potrebnih biblioteka
import struct
import matplotlib.colors as mcolors
import binascii
import cv2
import numpy as np
<<<<<<< HEAD
import scipy
import scipy.misc
import scipy.cluster
import matplotlib.cm as cm
=======
from sklearn.cluster import MiniBatchKMeans
>>>>>>> 1458056c2c7e13b7cba4d6c554dea0462d698d35
from PIL import Image
from operator import itemgetter

NUM_CLUSTERS = 5


if __name__ == '__main__':
<<<<<<< HEAD
    im = cv2.imread("D:\Workspace\Pycharm\SOFT\ptica.jpg")
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    cv2.imshow('img',im)

=======
    im = Image.open("D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\ptica.png")
    color = max(im.getcolors(im.size[0] * im.size[1]))


>>>>>>> 1458056c2c7e13b7cba4d6c554dea0462d698d35
    print('reading image')
    im = cv2.imread('D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\ptica.jpg')
    '''
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
    print('most frequent is %s' % (codes))
   # colors = dict(mcolors.BASE_COLORS,**mcolors.CSS4_COLORS)
   '''
    image = cv2.imread("D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\ptica.jpg")
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
    clt = MiniBatchKMeans(n_clusters=18)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    cv2.imwrite('D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\sans_red.jpg', np.hstack([quant]))
    cv2.imshow('fd',np.hstack([quant]))
    im =Image.open('D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\sans_red.jpg')

    pixels = im.getcolors(w*h)
    most_frequent_pixel = pixels[0]
    pixels = sorted(pixels, key=itemgetter(0))

    colors_for_remove = pixels[-6:-1]
    print(colors_for_remove)
    for count, colour in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, colour)

    print(most_frequent_pixel[0])

<<<<<<< HEAD


    n = Image.open('D:\Workspace\Pycharm\SOFT\ptica.jpg')
=======
    '''
    n = Image.open('D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\ptica.png')
>>>>>>> 1458056c2c7e13b7cba4d6c554dea0462d698d35
    n = n.resize((300, 300),Image.ANTIALIAS)

    m = n.load()
    b, g, r = n.split()
    n = Image.merge("RGB", (r, g, b))
    # get x,y size
    '''
    # Process every pixel
    im =  im.resize((300, 300),Image.ANTIALIAS)
    s = im.size

    for x in range(s[0]):
        for y in range(s[1]):
<<<<<<< HEAD
            current_color = n.getpixel((x, y))
            ####################################################################
            # Do your logic here and create a new (R,G,B) tuple called new_color
            ####################################################################
            if int(current_color[0]) == int(peak[0]) and int(current_color[1]) == int(peak[1]) and int(current_color[2]) == int(peak[2]):
                n.putpixel((x, y), (0,0,0))
    ''''''
    n.save('sans_red.jpg', "JPEG")
    img = cv2.imread('D:\Workspace\Pycharm\SOFT\sans_red.jpg')
=======
            current_color = im.getpixel((x, y))
            if current_color == colors_for_remove[0][1] or current_color == colors_for_remove[1][1] or \
                current_color == colors_for_remove[2][1] or current_color == colors_for_remove[3][1]:
                im.putpixel((x, y), (0,0,0))

    im.save('sans_red.jpg', "JPEG")
    img = cv2.imread('D:\SIIT\IV godina\VII semestar\Soft kompjuting\projekat\\recognition-of-birds\sans_red.jpg')
    cv2.imshow('img',img)

>>>>>>> 1458056c2c7e13b7cba4d6c554dea0462d698d35
    cv2.waitKey(0)
    cv2.destroyAllWindows()