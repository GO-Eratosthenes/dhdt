import numpy as np


def get_image_subset(img, rmin, rmax, cmin, cmax):  # generic
    """
    get subset of an image specified by the bounding box extents
    input:   img            array (n x m)     band with boolean values
             rmin           integer           minimum row
             rmax           integer           maximum row
             cmin           integer           minimum collumn
             cmax           integer           maximum collumn
    output:  sub_img        array (k x l)     band with boolean values         
    """
    
    sub_img = img[rmin:rmax,cmin:cmax]
    return sub_img