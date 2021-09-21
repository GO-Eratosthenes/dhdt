import numpy as np
from scipy import interpolate
from scipy import ndimage

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

def rotated_sobel(az, size=3):
    d = size //2
    az = np.radians(az)
    
    steps = np.arange(-d, +d+1)
    I = np.repeat(steps[np.newaxis,:], size, axis=0)
    J = np.rot90(I)
    
    np.seterr(divide='ignore', invalid='ignore')
    if az==0:
        H_x = I / (np.multiply(I,I) + np.multiply(J,J)) 
    else:
        H_x = (np.cos(az)*I) + (np.sin(az)*J) / \
              (np.multiply(I,I) + np.multiply(J,J))
    H_x[d,d] = 0
    return H_x
    
def get_grad_filters(ftype='sobel', size=3, order=1):
    """
    construct gradient filters

    :param dat_path:    STRING
        the type of gradient filter to get
          :options : sobel - 
                     kroon - 
                     scharr - Scharr, 2000 in "Optimal operators in digital 
                     image processing"
                     robinson- 
                     kayyali -
                     prewitt -
                     simoncelli - from Farid and Simoncelli, 1997 in "Optimally 
                     rotation-equivariant directional derivative kernels"
    :param size:       INTEGER
        dimenson/length of the template 
    :param order:      INTEGER
        1st or 2nd order derivative (not all have this)
    
    :return if order==1:
            H_x        ARRAY (_,_)     
            H_y        ARRAY (_,_) 
    :return if order==2:
            H_xx       ARRAY (_,_)     
            H_xy       ARRAY (_,_)             
    """
    
    ftype = ftype.lower() # make string lowercase

    if ftype=='kroon':
        if size==5:
            if order==2:
                H_xx = np.array([[+.0033, +.0045, -0.0156, +.0045, +.0033],
                                [ +.0435, +.0557, -0.2032, +.0557, +.0435],
                                [ +.0990, +.1009, -0.4707, +.1009, +.0990],
                                [ +.0435, +.0557, -0.2032, +.0557, +.0435],
                                [ +.0033, +.0045, -0.0156, +.0045, +.0033]])
                H_xy = np.array([[-.0034, -.0211, 0., +.0211, +.0034],
                                [ -.0211, -.1514, 0., +.1514, +.0211],
                                [ 0, 0, 0, 0, 0],
                                [ +.0211, +.1514, 0., -.1514, -.0211],
                                [ +.0034, +.0211, 0., -.0211, -.0034]])              
            else:
                H_x = np.array([[-.0007, -.0053, 0, +.0053, +.0007],
                               [ -.0108, -.0863, 0, +.0863, +.0108],
                               [ -.0270, -.2150, 0, +.2150, +.0270],
                               [ -.0108, -.0863, 0, +.0863, +.0108],
                               [ -.0007, -.0053, 0, +.0053, +.0007]])
        else:
            if order==2:
                H_xx = 8 / np.array([[105, -46, 105],
                                    [  50, -23,  50],
                                    [ 105, -46, 105]] )
                H_xy = 11 / np.array([[-114, np.inf, +114],
                                     [np.inf, np.inf, np.inf],
                                     [+114, np.inf, -114]] )
            else:
                H_x = np.outer(np.array([17, 61, 17]), \
                       np.array([-1, 0, +1])) / 95
    elif ftype=='scharr':
        if size==5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]), \
                       np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            H_x = np.outer(np.array([3, 10, 3]), \
                           np.array([-1, 0, +1])) / 16
    elif ftype=='sobel':
        if size==5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]), \
                       np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            H_x = np.outer(np.array([1, 2, 1]), \
                           np.array([-1, 0, +1])) / 4   
    elif ftype=='robinson':
        H_x = np.array([[-17., 0., 17.],
               [-61., 0., 61.],
               [-17., 0., 17.]]) / 95
    elif ftype=='kayyali':
        H_x = np.outer(np.array([1, 0, 1]), \
                       np.array([-1, 0, +1])) / 2
    elif ftype=='prewitt':
        H_x = np.outer(np.array([1, 1, 1]), \
                       np.array([-1, 0, +1])) / 3
    elif ftype=='simoncelli':
        k = np.array([0.125, 0.0, 0.250, 0.500, 1.000])
        d = np.array([0.125, 0.0, 0.250, 0.500, 1.000])
        H_x = np.outer(d,k)
    
    # elif ftype=='savitzky':
            
    if order==1:
        out1, out2 = H_x, np.flip(H_x.T, axis=0)
    else:
        out1, out2 = H_xx, H_xy
    return (out1, out2)

def harst_conv(I, ftype='bezier'):
    """
    construct gradient filters, following Harst, 2014: 
        "Simple filter design for first and second order derivatives by a 
        double filtering approach""

    :param size:       ARRAY  (_,_)
        array with intensities
    :param dat_path:   STRING
        the type of gradient filter to get
          :options : beizer - 
                     bspline - 
                     catmull -
                     cubic- 
    :return I_y:       ARRAY  (_,_)
        gradient in vertical direction
    :return I_x:       ARRAY  (_,_)
        gradient in horizontal direction        
    """

    # define kernels
    un = np.array([0.125, 0.250, 0.500, 1.000])
    up= np.array([0.75, 1.0, 1.0, 0.0]) 
    if ftype=='bezier':
        M = np.array([[-1, -3, +3, -1],
                     [ +0, +3, -6, +3],
                     [ +0, +0, +3, -3],
                     [ +0, +0, +0, +1]])
    elif ftype=='bspline':
        M = np.array([[-1, +3, -3, +1],
                     [ +3, -6, +3, +0],
                     [ -3, +0, +3, +0],
                     [ +1, +4, +1, +0]]) /6
    elif ftype=='catmull':
        M = np.array([[-1, +3, -3, +1],
                     [ +2, -5, +4, -1],
                     [ -1, +0, +1, +0],
                     [ +0, +2, +0, +0]]) /2
    elif ftype=='cubic':
        M = np.array([[-1, +3, -3, +1],
                     [ +3, -6, +3, +0],
                     [ -2, -3, +6, -1],
                     [ +0, +6, +0, +0]]) /6
    # elif ftype=='trigonometric':
    #     M = np.array([[+1, +1, +0, +1],
    #                  [ +1, np.sqrt(3/4), .5, .5],
    #                  [ +1, .5, np.sqrt(3/4), -.5],
    #                  [ +1, +0, +1, -1]])
    #     M = np.linalg.inv(M)
    #     un = np.array([1.0, +np.sqrt(.5), +np.sqrt(.5), +0.0])
    #     up = np.array([0.0, -np.sqrt(.5), +np.sqrt(.5), -2.0]) 
        
    d = np.matmul(un, M)
    k = np.matmul(up, M)
    #m = np.outer(np.convolve(d, k), np.convolve(k, k))         
    I_y = ndimage.convolve1d(ndimage.convolve1d(I, np.convolve(d, k), axis=0), \
                             np.convolve(k, k), axis=1)
    I_x = ndimage.convolve1d(ndimage.convolve1d(I, np.convolve(k, k), axis=0), \
                             np.convolve(d, k), axis=1)
    return (I_y, I_x)   