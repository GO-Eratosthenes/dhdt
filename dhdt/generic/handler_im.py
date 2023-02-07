# generic libraries
import numpy as np

from scipy import ndimage, linalg
from scipy.interpolate import griddata

def _check_same_im(I1,I2):

    assert ((I1.ndim>=2) and (I2.ndim>=2)), \
        'please provide grids'
    assert len(set({I1.shape[0], I2.shape[0]}))==1,\
         'please provide arrays of the same size'
    assert len(set({I1.shape[1], I2.shape[1]}))==1,\
         'please provide arrays of the same size'
    return I1, I2

def conv_2Dfilter(I, kernel):

    sub_shape = tuple(map(lambda i,j: i-j+1, I.shape, kernel.shape))
    view_shape = tuple(np.subtract(I.shape, sub_shape) + 1) + sub_shape
    strides = I.strides + I.strides

    sub_matrices = np.lib.stride_tricks.as_strided(I, view_shape, strides)
    V = np.einsum('ij,ijkl->kl', kernel, sub_matrices)
    return V

# def conv_2Dfft(): todo: https://gist.github.com/thearn/5424195

def toeplitz_block(kernel, image_size):
    # from https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
    # admin
    k_h, k_w = kernel.shape
    i_h, i_w = image_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(linalg.toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)),
                                        r=(*kernel[r], *np.zeros(i_w-k_w))) )

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)
    return W_conv

def nan_resistant_conv2(I, kernel, size='same', cval=np.nan):
    """ estimates a convolution on a corrupted array, making use of the energy
    perserving principle, see [1].

    Parameters
    ----------
    I : np.array, size=(m,n)
        array with intensity value
    kernel : np.array, size=(k,l)
        kernel to be used for the convolution
    size : {'same','smaller'}
        padding is done with NaN's
    cval : integer
        constant padding value

    Returns
    -------
    U : np.array

    References
    ----------
    .. [1] Altena et al. "Correlation dispersion as a measure to better estimate
       uncertainty of remotely sensed glacier displacements", The Crysophere,
       vol.16(6) pp.2285–2300, 2022.
    """
    m,n = I.shape
    k,l = kernel.shape

    # set the stage
    C = np.isnan(I.flatten()) # classification
    A = toeplitz_block(kernel, I.shape).T # design matrix

    for i in range(A.shape[1]):
        if np.sum(A[:,i]) != np.sum(A[C,i]):
            idx = np.where(A[:,i]!=0)[0]
            val, sat = np.squeeze(A[idx,i]), C[idx]

            if np.all(~sat): # all NaN's
                A[idx,:] = 0
            else:
                # redistribute the energy to comply with the closed system
                # constrain and based on relative weights
                surplus = np.sum(val[sat]) # energy to distribute
                contrib = surplus*(val[~sat]/np.sum(val[~sat]))
                A[idx[~sat],i] += contrib

    # estimate
    I = I.flatten()
    I[C] = 0
    U = np.dot(A.T,I)

    # reorganize
    pad_size = (k//2, l//2)
    U = U.reshape(m-2*pad_size[0],n-2*pad_size[1])
    if size in ['same']:
        U = np.pad(U, ((pad_size[0],pad_size[0]),(pad_size[1],pad_size[1])),
                   mode='constant', constant_values=cval)
    return U

def nan_resistant_diff2(I, kernel, size='same', cval=np.nan):
    """ estimates a differential convolution on a corrupted array, making use
    of the energy perserving principle, see [1].

    Parameters
    ----------
    I : np.array, size=(m,n)
        array with intensity value
    kernel : np.array, size=(k,l)
        kernel to be used for the convolution
    size : {'same','smaller'}
        padding is done with NaN's
    cval : integer
        constant padding value

    Returns
    -------
    U : np.array

    References
    ----------
    .. [1] Altena et al. "Correlation dispersion as a measure to better estimate
       uncertainty of remotely sensed glacier displacements", The Crysophere,
       vol.16(6) pp.2285–2300, 2022.
    """
    m,n = I.shape
    k,l = kernel.shape

    # set the stage
    C = np.isnan(I.flatten()) # classification
    A = toeplitz_block(kernel, I.shape).T # design matrix

    for i in range(A.shape[1]):
        if np.sum(A[:,i]) != np.sum(A[C,i]):
            idx = np.where(A[:,i]!=0)[0]
            val = np.squeeze(A[idx,i])
            sat = C[idx]

            # redistribute the energy to comply with the closed system constrain
            surplus = np.sum(val[sat]) # energy to distribute
            posse = np.sign(val)

            if (np.sign(surplus)==-1) and np.any(posse==+1):
                verdict = +1
            elif (np.sign(surplus)==+1) and np.any(posse==-1):
                verdict = -1
            else:
                verdict = 0

            if verdict==0:
                A[:,i] = 0
            else:
                idx_sub = idx[np.logical_and(posse==verdict, ~sat)]
                contrib = np.divide(surplus, idx_sub.size)
                A[idx_sub,i] += contrib

    # estimate
    I = I.flatten()
    I[C] = 0
    U = np.dot(A.T,I)

    # reorganize
    pad_size = (k//2, l//2)
    U = U.reshape(m-2*pad_size[0],n-2*pad_size[1])
    if size in ['same']:
        U = np.pad(U, ((pad_size[0],pad_size[0]),(pad_size[1],pad_size[1])),
                   mode='constant', constant_values=cval)

    return U

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

def bilinear_interpolation(I, di, dj):
    """ do bilinear interpolation of an image at different locations

    Parameters
    ----------
    I : np.array, size=(m,n), ndim={2,3}
        data array.
    di :
        * np.array, size=(k,l), ndim=2
            vertical locations, within local image frame.
        * float
            uniform horizontal displacement
    dj :
        * np.array, size=(k,l), ndim=2
            horizontal locations, within local image frame.
        * float
            uniform horizontal displacement
    Returns
    -------
    I_new : np.array, size=(k,l), ndim=2, dtype={float,complex}
        interpolated values.

    See Also
    --------
    .mapping_tools.bilinear_interp_excluding_nodat

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       j               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | i         map         |
          based      v           based       |

    """
    assert type(I) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")

    if isinstance(dj, (float, int)): # integer values: constant displacement
        mI, nI = I.shape[0], I.shape[1]
        (dj, di) = np.meshgrid(np.linspace(0, nI-1, nI)+dj,
                               np.linspace(0, mI-1, mI)+di)
    else:
        di,dj = np.asarray(di), np.asarray(dj)

    i0,j0 = np.floor(di).astype(int), np.floor(dj).astype(int)
    i1,j1 = i0+1, j0+1

    j0,j1 = np.clip(j0, 0, I.shape[1]-1), np.clip(j1, 0, I.shape[1]-1)
    i0,i1 = np.clip(i0, 0, I.shape[0]-1), np.clip(i1, 0, I.shape[0]-1)

    if I.ndim==3: # make resitent to multi-dimensional arrays
        Ia,Ib,Ic,Id = I[i0,j0,:], I[i1,j0,:], I[i0,j1,:], I[i1,j1,:]
    else:
       Ia,Ib,Ic,Id = I[i0,j0], I[i1,j0], I[i0,j1], I[i1,j1]

    wa,wb,wc,wd = (j1-dj)*(i1-di), (j1-dj)*(di-i0), (dj-j0)*(i1-di), \
                  (dj-j0)*(di-i0)

    if I.ndim==3:
        wa = np.repeat(np.atleast_3d(wa), I.shape[2], axis=2)
        wb = np.repeat(np.atleast_3d(wb), I.shape[2], axis=2)
        wc = np.repeat(np.atleast_3d(wc), I.shape[2], axis=2)
        wd = np.repeat(np.atleast_3d(wd), I.shape[2], axis=2)

    I_new = wa*Ia + wb*Ib + wc*Ic + wd*Id
    return I_new

def simple_nearest_neighbor(I, i, j):
    I_ij = np.zeros((i.size))
    i,j = np.round(i).astype(int), np.round(j).astype(int)
    IN = np.logical_and.reduce((i>=0, i<I.shape[0],
                                j>=0, j<I.shape[1]))
    I_ij[IN] = I[i[IN],j[IN]]
    return I_ij

def rescale_image(I, sc, method='cubic'):
    """ generate a zoomed-in version of an image, through isotropic scaling

    Parameters
    ----------
    I : numpy.array, size=(m,n), ndim={2,3}
        image
    sc : float
        scaling factor, >1 is zoom-in, <1 is zoom-out.
    method : {‘linear’, ‘nearest’, ‘cubic’ (default)}
        * 'nearest' - nearest neighbor interpolation
        * 'linear' - linear interpolation
        * 'cubic' - cubic spline interpolation

    Returns
    -------
    I_new : numpy.array, size={(m,n),(m,n,b)}, ndim={2,3}
        rescaled, but not resized image

    """
    assert type(I)==np.ndarray, ("please provide an array")
    T = np.array([[sc, 0], [0, sc]]) # scaling matrix

    # make local coordinate system
    m,n = I.shape[:2]
    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n))

    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = np.matmul(T, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (m, n))
    grd_j2 = np.reshape(grd_2[1,:], (m, n))

    # apply scaling
    if I.ndim<3:
         I_new = griddata(stk_1, I.flatten().T, (grd_i2,grd_j2),
                          method=method)
    else:
         b = I.shape[2]
         I_new = np.zeros((m,n,b))
         for i in range(b):
             I_new[...,i] = griddata(stk_1, I[...,i].flatten().T,
                                     (grd_i2,grd_j2),
                                     method=method)
    return I_new

def rotated_sobel(az, size=3, indexing='ij'):
    """ construct gradient filter along a specific angle

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

def diff_compass(I,direction='n'):
    """ calculates the neighboring directional difference

    Parameters
    ----------
    I : np.array, size=(m-2,n-2), dtype=float
        intensity array
    direction : {'n','ne','e','se','s','sw','w','nw'}
        compass direction

    Returns
    -------
    dI : np.array, size=(m-2,n-2), dtype=float
        directional neighboring difference
    """
    if direction in ('n', 'north', 'up'):
        if I.ndim==2:
            dI = I[:-2,1:-1] - I[1:-1,1:-1]
        else:
            dI = I[:-2,1:-1,:] - I[1:-1,1:-1,:]
    elif direction in ('e', 'east', 'right'):
        if I.ndim == 2:
            dI = I[1:-1,+2:] - I[1:-1,1:-1]
        else:
            dI = I[1:-1,+2:,:] - I[1:-1,1:-1,:]
    elif direction in ('s', 'south', 'down'):
        if I.ndim == 2:
            dI = I[+2:,1:-1] - I[1:-1,1:-1]
        else:
            dI = I[+2:,1:-1,:] - I[1:-1,1:-1,:]
    elif direction in ('w', 'west', 'left'):
        if I.ndim == 2:
            dI = I[1:-1,:-2] - I[1:-1,1:-1]
        else:
            dI = I[1:-1,:-2,:] - I[1:-1,1:-1,:]
    elif direction in ('ne', 'norhteast', 'ur', 'ru','upperright'):
        if I.ndim == 2:
            dI = I[:-2,+2:] - I[1:-1,1:-1]
        else:
            dI = I[:-2,+2:,:] - I[1:-1,1:-1,:]
    elif direction in ('se', 'southeast', 'lr', 'rl', 'lowerright'):
        if I.ndim == 2:
            dI = I[+2:,+2:] - I[1:-1,1:-1]
        else:
            dI = I[+2:,+2:,:] - I[1:-1,1:-1,:]
    elif direction in ('sw', 'southwest', 'll', 'lowerleft'):
        if I.ndim == 2:
            dI = I[+2:,:-2] - I[1:-1,1:-1]
        else:
            dI = I[+2:,:-2,:] - I[1:-1,1:-1,:]
    elif direction in ('nw', 'northwest','ul','lu','upperleft'):
        if I.ndim == 2:
            dI = I[:-2,:-2] - I[1:-1,1:-1]
        else:
            dI = I[:-2,:-2,:] - I[1:-1,1:-1,:]
    return dI

def get_grad_filters(ftype='sobel', tsize=3, order=1, indexing='xy'):
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
    indexing : {'xy','ij'}
        Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output, see Notes
        for more details.

    Returns
    -------
    * order=1:
            H_x : np.array, size=(tsize,tsize)
                horizontal first order derivative
            H_y : np.array, size=(tsize,tsize)
                vertical first order derivative
            if 'ij':
                order is flipped
    * order=2:
            H_xx : np.array, size=(tsize,tsize)
                horizontal second order derivative
            H_xy : np.array, size=(tsize,tsize)
                cross-directional second order derivative
            if 'ij':
                'H_xx' becomes 'H_jj'

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       j               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | i         map         |
          based      v           based       |

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
                H_x = np.outer(np.array([17, 61, 17]),
                       np.array([-1, 0, +1])) / 95
    elif ftype=='scharr':
        if tsize==5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]),
                       np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            if order==2:
                H_xx = np.outer(np.array([3, 10, 3]),
                                np.array([+1, -2, +1])) / 16
                H_xy = np.outer(np.array([+1, 0, -1]),
                                np.array([-1, 0, +1])) / 4
            else:
                H_x = np.outer(np.array([3, 10, 3]),
                               np.array([-1, 0, +1])) / 16
    elif ftype=='sobel':
        if tsize==5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]),
                       np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            if order==2:
                H_xx = np.outer(np.array([1, 2, 1]),
                                np.array([+1, -2, +1])) / 4
                H_xy = np.outer(np.array([+1, 0, -1]),
                                np.array([-1, 0, +1])) / 4
            else:
                H_x = np.outer(np.array([1, 2, 1]),
                               np.array([-1, 0, +1])) / 4
    elif ftype=='robinson':
        H_x = np.array([[-17., 0., 17.],
               [-61., 0., 61.],
               [-17., 0., 17.]]) / 95
    elif ftype=='kayyali':
        H_x = np.outer(np.array([1, 0, 1]),
                       np.array([-1, 0, +1])) / 2
    elif ftype=='prewitt':
        H_x = np.outer(np.array([1, 1, 1]),
                       np.array([-1, 0, +1])) / 3
    elif ftype=='simoncelli':
        k = np.array([0.125, 0.0, 0.250, 0.500, 1.000])
        d = np.array([0.125, 0.0, 0.250, 0.500, 1.000])
        H_x = np.outer(d,k)

    # elif ftype=='savitzky': #todo

    if order==1:
        if indexing=='xy':
            out1, out2 = H_x, np.flip(H_x.T, axis=0)
        else: # 'ij'
            out1, out2 = H_x.T, H_x
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

    d = np.matmul(un, M)
    k = np.matmul(up, M)

    I_y = ndimage.convolve1d(ndimage.convolve1d(I, np.convolve(d, k), axis=0),
                             np.convolve(k, k), axis=1)
    I_x = ndimage.convolve1d(ndimage.convolve1d(I, np.convolve(k, k), axis=0),
                             np.convolve(d, k), axis=1)
    return I_y, I_x
