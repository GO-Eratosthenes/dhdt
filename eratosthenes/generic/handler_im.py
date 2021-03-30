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
    sub_img = img[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    return sub_img

def bilinear_interpolation(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);
    
    if im.ndim==3: # make resitent to multi-dimensional arrays
        Ia = im[ y0, x0, :]
        Ib = im[ y1, x0, :]
        Ic = im[ y0, x1, :]
        Id = im[ y1, x1, :]
    else:
        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    if im.ndim==3:
        wa = np.repeat(wa[:, :, np.newaxis], im.shape[2], axis=2)
        wb = np.repeat(wb[:, :, np.newaxis], im.shape[2], axis=2)
        wc = np.repeat(wc[:, :, np.newaxis], im.shape[2], axis=2)
        wd = np.repeat(wd[:, :, np.newaxis], im.shape[2], axis=2)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id