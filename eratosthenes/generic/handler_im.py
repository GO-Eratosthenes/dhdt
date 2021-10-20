# generic libraries
import numpy as np

from scipy import ndimage
from scipy.interpolate import griddata

def select_boi_from_stack(I, boi):
    """ give array with selection given by a pointing array 

    Parameters
    ----------
    I : np.array, size=(m,n,b), ndim={2,3}
        data array.
    boi : np.array, size=(k,1), dtype=integer
        array giving the bands of interest.

    Returns
    -------
    I_new : np.array, size=(m,n,k), ndim={2,3}
        selection of bands, in the order given by boi.
    """
    assert type(I)==np.ndarray, ("please provide an array")
    assert type(boi)==np.ndarray, ("please provide an array")
    
    if boi.shape[0]>0:
        if I.ndim>2:
            ndim = I.shape[2]        
            assert(ndim>=np.max(boi)) # boi pointer should not exceed bands
            I_new = I[:,:,boi]
        else:
            I_new = I
    else:
        I_new = I
    return I_new

def get_image_subset(I, bbox):
    """ get subset of an image specified by the bounding box extents

    Parameters
    ----------
    I : np.array, size=(m,n), ndim={2,3,4}
        data array
    bbox : np.array, size=(1,4)
        [minimum row, maximum row, minimum collumn maximum collumn].

    Returns
    -------
    sub_I : np.array, size=(k,l), ndim={2,3,4}
        sub set of the data array.

    """
    assert type(I)==np.ndarray, ("please provide an array")
    assert(bbox[0]<=bbox[1])
    assert(bbox[2]<=bbox[3])
    
    if I.ndim==2:
        sub_I = I[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    elif I.ndim==3:
        sub_I = I[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
    elif I.ndim==4:
        sub_I = I[bbox[0]:bbox[1],bbox[2]:bbox[3],:,:]
    return sub_I

def bilinear_interpolation(I, x, y):
    """ do bilinear interpolation of an image at different locations

    Parameters
    ----------
    I : np.array, size=(m,n), ndim={2,3}
        data array.
    x : np.array, size=(k,l), ndim=2
        horizontal locations, within local image frame.
    y : np.array, size=(k,l), ndim=2
        vertical locations, within local image frame.

    Returns
    -------
    I_new : np.array, size=(k,l), ndim=2, dtype={float,complex}
        interpolated values.

    See Also
    --------
    .mapping_tools.bilinear_interp_excluding_nodat
    """
    assert type(I)==np.ndarray, ("please provide an array")
    x,y = np.asarray(x), np.asarray(y)

    x0,y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1,y1 = x0+1, y0+1

    x0,x1 = np.clip(x0, 0, I.shape[1]-1), np.clip(x1, 0, I.shape[1]-1)
    y0,y1 = np.clip(y0, 0, I.shape[0]-1), np.clip(y1, 0, I.shape[0]-1)
    
    if I.ndim==3: # make resitent to multi-dimensional arrays
        Ia,Ib,Ic,Id = I[y0,x0,:], I[y1,x0,:], I[y0,x1,:], I[y1,x1,:]
    else:
       Ia,Ib,Ic,Id = I[y0,x0], I[y1,x0], I[y0,x1], I[y1,x1]

    wa,wb,wc,wd = (x1-x)*(y1-y), (x1-x)*(y-y0), (x-x0)*(y1-y), (x-x0)*(y-y0)

    if I.ndim==3:
        wa = np.repeat(wa[:, :, np.newaxis], I.shape[2], axis=2)
        wb = np.repeat(wb[:, :, np.newaxis], I.shape[2], axis=2)
        wc = np.repeat(wc[:, :, np.newaxis], I.shape[2], axis=2)
        wd = np.repeat(wd[:, :, np.newaxis], I.shape[2], axis=2)
        
    I_new = wa*Ia + wb*Ib + wc*Ic + wd*Id 
    return I_new

def rescale_image(I, sc): 
    """ generate a zoomed-in version of an image, through isotropic scaling

    Parameters
    ----------
    I : np.array, size=(m,n)
        image
    sc : float
        scaling factor, >1 is zoom-in, <1 is zoom-out.

    Returns
    -------
    I_new : np.array, size=(m,n)
        rescaled, but not resized image

    """
    assert type(I)==np.ndarray, ("please provide an array")
    T = np.array([[sc, 0], [0, sc]]) # scaling matrix
    
    # make local coordinate system
    (mI,nI) = I.shape    
    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, mI), np.linspace(-1, 1, nI))
    
    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T
    
    grd_2 = np.matmul(T, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (mI, nI))
    grd_j2 = np.reshape(grd_2[1,:], (mI, nI))
    
    # apply scaling
    I_new = griddata(stk_1, I.flatten().T, (grd_i2,grd_j2), 
                     method='cubic') # method='nearest')
    return I_new

def rotated_sobel(az, size=3, indexing='ij'):
    """ construct gradient filters

    Parameters
    ----------    
    az : float, unit=degrees
        direction, in degrees with clockwise direction
    size : integer
        radius of the template 
    indexing : {‘xy’, ‘ij’}  
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates

    Returns
    -------    
    H_x : np.array
        gradient filter     
        
         * "xy" : followning the azimuth argument
         * "ij" : has the horizontal direction as origin

    See Also
    --------
    .mapping_tools.cast_orientation : 
        convolves oriented gradient over full image

    Notes
    ----- 
    The angle(s) are declared in the following coordinate frame: 
        
        .. code-block:: text
        
                 ^ North & y
                 |  
            - <--|--> +
                 |
                 +----> East & x

    Two different coordinate system are used here: 
        
        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x 
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |



    References
    ----------
    .. [1] Sobel, "An isotropic 3×3 gradient operator" Machine vision for 
       three-dimensional scenes, Academic press, pp.376–379, 1990.
    """
    d = size //2
    az = np.radians(az)
    
    steps = np.arange(-d, +d+1)
    I = np.repeat(steps[np.newaxis,:], size, axis=0)
    J = np.rot90(I)
    
    np.seterr(divide='ignore', invalid='ignore')
    if az==0:
        H_x = I / (np.multiply(I,I) + np.multiply(J,J)) 
    else:
        if indexing=='ij':
            H_x = (+np.cos(az)*I) + (+np.sin(az)*J) / \
                  (np.multiply(I,I) + np.multiply(J,J))
        else:
            H_x = (-np.sin(az)*I) + (-np.cos(az)*J) / \
                  (np.multiply(I,I) + np.multiply(J,J))
    H_x[d,d] = 0
    return H_x
    
def get_grad_filters(ftype='sobel', tsize=3, order=1):
    """ construct gradient filters

    Parameters
    ----------    
    ftype : {'sobel' (default), 'kroon', 'scharr', 'robinson', 'kayyali', \
             'prewitt', 'simoncelli'}
        Specifies which the type of gradient filter to get:
            
          * 'sobel' : see [1]
          * 'kroon' : see [2]
          * 'scharr' : see [3]
          * 'robinson' : see [4],[5]
          * 'kayyali' :
          * 'prewitt' : see [6]
          * 'simoncelli' : see [7]
    tsize : {3,5}, dtype=integer
        dimenson/length of the template 
    order : {1,2}, dtype=integer
        first or second order derivative (not all have this)

    Returns
    -------    
    * order=1:
            H_x : np.array, size=(tsize,tsize)
                horizontal first order derivative     
            H_y : np.array, size=(tsize,tsize)
                vertical first order derivative  
    * order=2:
            H_xx : np.array, size=(tsize,tsize)
                horizontal second order derivative  
            H_xy : np.array, size=(tsize,tsize)
                cross-directional second order derivative  

    References
    ---------- 
    .. [1] Sobel, "An isotropic 3×3 gradient operator" Machine vision for 
       three-dimensional scenes, Academic press, pp.376–379, 1990.
    .. [2] Kroon, "Numerical optimization of kernel-based image derivatives", 
       2009
    .. [3] Scharr, "Optimal operators in digital image processing" Dissertation 
       University of Heidelberg, 2000.
    .. [4] Kirsch, "Computer determination of the constituent structure of 
       biological images" Computers and biomedical research. vol.4(3) 
       pp.315–32, 1970.
    .. [5] Roberts, "Machine perception of 3-D solids, optical and 
       electro-optical information processing" MIT press, 1965.
    .. [6] Prewitt, "Object enhancement and extraction" Picture processing and 
       psychopictorics, 1970. 
    .. [7] Farid & Simoncelli "Optimally rotation-equivariant directional 
       derivative kernels" Proceedings of the international conference on 
       computer analysis of images and patterns, pp207–214, 1997
    """
    
    ftype = ftype.lower() # make string lowercase

    if ftype=='kroon':
        if tsize==5:
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
        if tsize==5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]), \
                       np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            H_x = np.outer(np.array([3, 10, 3]), \
                           np.array([-1, 0, +1])) / 16
    elif ftype=='sobel':
        if tsize==5:
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
    return out1, out2

def harst_conv(I, ftype='bezier'):
    """ construct derivatives through double filtering [1]

    Parameters
    ----------
    I : np.array, size=(m,n)
        array with intensities
    ftype : {'bezier', 'bspline','catmull','cubic'}
        the type of gradient filter to use. The default is 'bezier'.

    Returns
    -------
    I_y : np.array, size=(m,n)
        gradient in vertical direction
    I_x : np.array, size=(m,n)
        gradient in horizontal direction     

    References
    ---------- 
    .. [1] Harst, "Simple filter design for first and second order derivatives 
       by a double filtering approach" Pattern recognition letters, vol.42
       pp.65–71, 2014.        
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
    return I_y, I_x