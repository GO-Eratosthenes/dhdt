import numpy as np


def get_image_subset(img, bbox):  # generic
    """
    get subset of an image specified by the bounding box extents
    input:   img            array (n x m)     band with boolean values
             bbox           array (1 x 4)     array giving:
                                              [
                                                  minimum row,
                                                  maximum row,
                                                  minimum collumn,
                                                  maximum collumn
                                              ]
    output:  sub_img        array (k x l)     band with values
    """
    sub_img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return sub_img
