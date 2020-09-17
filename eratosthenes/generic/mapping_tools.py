import numpy as np

from scipy import ndimage


def castOrientation(I, Az):  # generic
    """
    Emphasises intentisies within a certain direction
    input:   I              array (n x m)     band with intensity values
             Az             array (n x m)     band of azimuth values
    output:  Ican           array (m x m)     Shadow band
    """
#    kernel = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) # Sobel
    kernel = np.array([[17], [61], [17]])*np.array([-1, 0, 1])/95  # Kroon
    Idx = ndimage.convolve(I, np.flip(kernel, axis=1))  # steerable filters
    Idy = ndimage.convolve(I, np.flip(np.transpose(kernel), axis=0))
    Ican = (np.multiply(np.cos(np.radians(Az)), Idy)
            - np.multiply(np.sin(np.radians(Az)), Idx))
    return Ican


def bboxBoolean(img):  # generic
    """
    get image coordinates of maximum bounding box extent of True values in a
    boolean matrix
    input:   img            array (n x m)     band with boolean values
    output:  rmin           integer           minimum row
             rmax           integer           maximum row
             cmin           integer           minimum collumn
             cmax           integer           maximum collumn
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def RefTrans(Transform, dI, dJ):  # generic
    """
    translate reference transform
    input:   Transform      array (1 x 6)     georeference transform of
                                              an image
             dI             integer           translation in rows
             dJ             integer           translation in collumns
    output:  newransform    array (1 x 6)     georeference transform of
                                              transformed image
    """
    newTransform = (Transform[0]+dI*Transform[1]+dJ*Transform[2],
                    Transform[1], Transform[2],
                    Transform[3]+dI*Transform[4]+dJ*Transform[5],
                    Transform[4], Transform[5])
    return newTransform


def RefScale(Transform, scaling):  # generic
    """
    scale reference transform
    input:   Transform      array (1 x 6)     georeference transform of
                                              an image
             scaling        integer           scaling in rows and collumns
    output:  newransform    array (1 x 6)     georeference transform of
                                              transformed image
    """
    # not using center of pixel
    newTransform = (Transform[0], Transform[1]*scaling, Transform[2]*scaling,
                    Transform[3], Transform[4]*scaling, Transform[5]*scaling)
    return newTransform


def rotMat(theta):  # generic
    """
    build rotation matrix, theta is in degrees
    input:   theta        integer             angle
    output:  R            array (2 x 2)     2D rotation matrix
    """
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return R


def pix2map(geoTransform, i, j):  # generic
    """
    Transform image coordinates to map coordinates
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             i              array (n x 1)     row coordinates in image space
             j              array (n x 1)     column coordinates in image space
    output:  x              array (n x 1)     map coordinates
             y              array (n x 1)     map coordinates
    """
    x = (geoTransform[0]
         + geoTransform[1] * j
         + geoTransform[2] * i
         )
    y = (geoTransform[3]
         + geoTransform[4] * j
         + geoTransform[5] * i
         )

    # offset the center of the pixel
    x += geoTransform[1] / 2.0
    y += geoTransform[5] / 2.0
    return x, y


def map2pix(geoTransform, x, y):  # generic
    """
    Transform map coordinates to image coordinates
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             x              array (n x 1)     map coordinates
             y              array (n x 1)     map coordinates
    output:  i              array (n x 1)     row coordinates in image space
             j              array (n x 1)     column coordinates in image space
    """
    # offset the center of the pixel
    x -= geoTransform[1] / 2.0
    y -= geoTransform[5] / 2.0
    x -= geoTransform[0]
    y -= geoTransform[3]

    if geoTransform[2] == 0:
        j = x / geoTransform[1]
    else:
        j = (x / geoTransform[1]
             + y / geoTransform[2])

    if geoTransform[4] == 0:
        i = y / geoTransform[5]
    else:
        i = (x / geoTransform[4]
             + y / geoTransform[5])

    return i, j
