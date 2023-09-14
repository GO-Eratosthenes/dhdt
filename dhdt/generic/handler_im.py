# generic libraries
import numpy as np

from scipy import ndimage, linalg
from scipy.interpolate import griddata


def _check_same_im(I1, I2):
    assert ((I1.ndim >= 2) and (I2.ndim >= 2)), \
        'please provide grids'
    assert len(set({I1.shape[0], I2.shape[0]})) == 1, \
        'please provide arrays of the same size'
    assert len(set({I1.shape[1], I2.shape[1]})) == 1, \
        'please provide arrays of the same size'
    return I1, I2


def conv_2Dfilter(Z, kernel):
    sub_shape = tuple(map(lambda i, j: i - j + 1, Z.shape, kernel.shape))
    view_shape = tuple(np.subtract(Z.shape, sub_shape) + 1) + sub_shape
    strides = Z.strides + Z.strides

    sub_matrices = np.lib.stride_tricks.as_strided(Z, view_shape, strides)
    V = np.einsum('ij,ijkl->kl', kernel, sub_matrices)
    return V


def toeplitz_block(kernel, image_size):
    # from https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n  # noqa: E501
    # admin
    k_h, k_w = kernel.shape
    i_h, i_w = image_size
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(
            linalg.toeplitz(c=(kernel[r, 0], *np.zeros(i_w - k_w)),
                            r=(*kernel[r], *np.zeros(i_w - k_w))))

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i + j, :] = B

    W_conv.shape = (h_blocks * h_block, w_blocks * w_block)
    return W_conv


def nan_resistant_conv2(Z, kernel, size='same', cval=np.nan):
    """ estimates a convolution on a corrupted array, making use of the energy
    perserving principle, see [Al22]_.

    Parameters
    ----------
    Z : np.array, size=(m,n)
        array with intensity value
    kernel : np.array, size=(j,k)
        kernel to be used for the convolution
    size : {'same','smaller'}
        padding is done with NaN's
    cval : integer
        constant padding value

    Returns
    -------
    U : numpy.ndarray

    References
    ----------
    .. [Al22] Altena et al. "Correlation dispersion as a measure to better
              estimate uncertainty of remotely sensed glacier displacements",
              The Crysophere, vol.16(6) pp.2285–2300, 2022.
    """
    m, n = Z.shape
    j, k = kernel.shape

    # set the stage
    C = np.isnan(Z.flatten())  # classification
    A = toeplitz_block(kernel, Z.shape).T  # design matrix

    for i in range(A.shape[1]):
        if np.sum(A[:, i]) != np.sum(A[C, i]):
            idx = np.where(A[:, i] != 0)[0]
            val, sat = np.squeeze(A[idx, i]), C[idx]

            if np.all(~sat):  # all NaN's
                A[idx, :] = 0
            else:
                # redistribute the energy to comply with the closed system
                # constrain and based on relative weights
                surplus = np.sum(val[sat])  # energy to distribute
                contrib = surplus * (val[~sat] / np.sum(val[~sat]))
                A[idx[~sat], i] += contrib

    # estimate
    Z = Z.flatten()
    Z[C] = 0
    U = np.dot(A.T, Z)

    # reorganize
    pad_size = (j // 2, k // 2)
    U = U.reshape(m - 2 * pad_size[0], n - 2 * pad_size[1])
    if size in ['same']:
        U = np.pad(U, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1])),
                   mode='constant',
                   constant_values=cval)
    return U


def nan_resistant_diff2(Z, kernel, size='same', cval=np.nan):
    """ estimates a differential convolution on a corrupted array, making use
    of the energy perserving principle, see [Al22]_.

    Parameters
    ----------
    Z : np.array, size=(m,n)
        array with intensity value
    kernel : np.array, size=(j,k)
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
    .. [Al22] Altena et al. "Correlation dispersion as a measure to better
              estimate uncertainty of remotely sensed glacier displacements",
              The Crysophere, vol.16(6) pp.2285–2300, 2022.
    """
    m, n = Z.shape
    j, k = kernel.shape

    # set the stage
    C = np.isnan(Z.flatten())  # classification
    A = toeplitz_block(kernel, Z.shape).T  # design matrix

    for i in range(A.shape[1]):
        if np.sum(A[:, i]) != np.sum(A[C, i]):
            idx = np.where(A[:, i] != 0)[0]
            val = np.squeeze(A[idx, i])
            sat = C[idx]

            # redistribute energy to comply with the closed system constrain
            surplus = np.sum(val[sat])  # energy to distribute
            posse = np.sign(val)

            if (np.sign(surplus) == -1) and np.any(posse == +1):
                verdict = +1
            elif (np.sign(surplus) == +1) and np.any(posse == -1):
                verdict = -1
            else:
                verdict = 0

            if verdict == 0:
                A[:, i] = 0
            else:
                idx_sub = idx[np.logical_and(posse == verdict, ~sat)]
                contrib = np.divide(surplus, idx_sub.size)
                A[idx_sub, i] += contrib

    # estimate
    Z = Z.flatten()
    Z[C] = 0
    U = np.dot(A.T, Z)

    # reorganize
    pad_size = (j // 2, k // 2)
    U = U.reshape(m - 2 * pad_size[0], n - 2 * pad_size[1])
    if size in ['same']:
        U = np.pad(U, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1])),
                   mode='constant',
                   constant_values=cval)

    return U


def select_boi_from_stack(Z, boi):
    """ give array with selection given by a pointing array

    Parameters
    ----------
    Z : np.array, size=(m,n,b), ndim={2,3}
        data array.
    boi : np.array, size=(k,1), dtype=integer
        array giving the bands of interest.

    Returns
    -------
    Z_new : np.array, size=(m,n,k), ndim={2,3}
        selection of bands, in the order given by boi.
    """
    assert type(Z) == np.ndarray, ("please provide an array")
    assert type(boi) == np.ndarray, ("please provide an array")

    if boi.shape[0] > 0:
        if Z.ndim > 2:
            ndim = Z.shape[2]
            assert (ndim >= np.max(boi))  # boi pointer should not exceed bands
            Z_new = Z[:, :, boi]
        else:
            Z_new = Z
    else:
        Z_new = Z
    return Z_new


def get_image_subset(Z, bbox):
    """ get subset of an image specified by the bounding box extents

    Parameters
    ----------
    Z : np.array, size=(m,n), ndim={2,3,4}
        data array
    bbox : np.array, size=(1,4)
        [minimum row, maximum row, minimum collumn maximum collumn].

    Returns
    -------
    sub_Z : np.array, size=(k,l), ndim={2,3,4}
        sub set of the data array.

    """
    assert type(Z) == np.ndarray, ("please provide an array")
    assert (bbox[0] <= bbox[1])
    assert (bbox[2] <= bbox[3])

    if Z.ndim == 2:
        sub_Z = Z[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    elif Z.ndim == 3:
        sub_Z = Z[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    elif Z.ndim == 4:
        sub_Z = Z[bbox[0]:bbox[1], bbox[2]:bbox[3], :, :]
    return sub_Z


def bilinear_interpolation(Z, di, dj):
    """ do bilinear interpolation of an image at different locations

    Parameters
    ----------
    Z : np.array, size=(m,n), ndim={2,3}
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
    Z_new : np.array, size=(k,l), ndim=2, dtype={float,complex}
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
    assert type(Z) in (np.ma.core.MaskedArray, np.ndarray), \
        "please provide an array"

    if isinstance(dj, (float, int)):  # integer values: constant displacement
        mI, nI = Z.shape[0], Z.shape[1]
        (dj, di) = np.meshgrid(
            np.linspace(0, nI - 1, nI) + dj,
            np.linspace(0, mI - 1, mI) + di)
    else:
        di, dj = np.asarray(di), np.asarray(dj)

    i0, j0 = np.floor(di).astype(int), np.floor(dj).astype(int)
    i1, j1 = i0 + 1, j0 + 1

    j0, j1 = np.clip(j0, 0, Z.shape[1] - 1), np.clip(j1, 0, Z.shape[1] - 1)
    i0, i1 = np.clip(i0, 0, Z.shape[0] - 1), np.clip(i1, 0, Z.shape[0] - 1)

    if Z.ndim == 3:  # make resitent to multi-dimensional arrays
        Ia, Ib, Ic, Id = Z[i0, j0, :], Z[i1, j0, :], Z[i0, j1, :], Z[i1, j1, :]
    else:
        Ia, Ib, Ic, Id = Z[i0, j0], Z[i1, j0], Z[i0, j1], Z[i1, j1]

    wa = (j1 - dj) * (i1 - di)
    wb = (j1 - dj) * (di - i0)
    wc = (dj - j0) * (i1 - di)
    wd = (dj - j0) * (di - i0)

    if Z.ndim == 3:
        wa = np.repeat(np.atleast_3d(wa), Z.shape[2], axis=2)
        wb = np.repeat(np.atleast_3d(wb), Z.shape[2], axis=2)
        wc = np.repeat(np.atleast_3d(wc), Z.shape[2], axis=2)
        wd = np.repeat(np.atleast_3d(wd), Z.shape[2], axis=2)

    Z_new = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return Z_new


def simple_nearest_neighbor(Z, i, j):
    Z_ij = np.zeros((i.size))
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    IN = np.logical_and.reduce((i >= 0, i < Z.shape[0], j >= 0, j
                                < Z.shape[1]))
    Z_ij[IN] = Z[i[IN], j[IN]]
    return Z_ij


def rescale_image(Z, sc, method='cubic'):
    """ generate a zoomed-in version of an image, through isotropic scaling

    Parameters
    ----------
    Z : numpy.array, size=(m,n), ndim={2,3}
        image
    sc : float
        scaling factor, >1 is zoom-in, <1 is zoom-out.
    method : {‘linear’, ‘nearest’, ‘cubic’ (default)}
        * 'nearest' - nearest neighbor interpolation
        * 'linear' - linear interpolation
        * 'cubic' - cubic spline interpolation

    Returns
    -------
    Z_new : numpy.array, size={(m,n),(m,n,b)}, ndim={2,3}
        rescaled, but not resized image

    """
    assert type(Z) == np.ndarray, ("please provide an array")
    T = np.array([[sc, 0], [0, sc]])  # scaling matrix

    # make local coordinate system
    m, n = Z.shape[:2]
    (grd_i1, grd_j1) = np.meshgrid(np.linspace(-1, 1, m),
                                   np.linspace(-1, 1, n))

    stk_1 = np.column_stack([grd_i1.flatten(), grd_j1.flatten()])

    grd_2 = np.matmul(T, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0, :], (m, n))
    grd_j2 = np.reshape(grd_2[1, :], (m, n))

    # apply scaling
    if Z.ndim < 3:
        Z_new = griddata(stk_1, Z.flatten().T, (grd_i2, grd_j2), method=method)
    else:
        b = Z.shape[2]
        Z_new = np.zeros((m, n, b))
        for i in range(b):
            Z_new[..., i] = griddata(stk_1,
                                     Z[..., i].flatten().T, (grd_i2, grd_j2),
                                     method=method)
    return Z_new


def rotated_sobel(az, size=3, indexing='ij'):
    """ construct gradient filter along a specific angle, see [S090]_

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
    .. [S090] Sobel, "An isotropic 3×3 gradient operator" Machine vision for
              three-dimensional scenes, Academic press, pp.376–379, 1990.
    """
    d = size // 2
    az = np.radians(az)

    steps = np.arange(-d, +d + 1)
    Z = np.repeat(steps[np.newaxis, :], size, axis=0)
    J = np.rot90(Z)

    np.seterr(divide='ignore', invalid='ignore')
    if az == 0:
        H_x = Z / (np.multiply(Z, Z) + np.multiply(J, J))
    else:
        if indexing == 'ij':
            H_x = (+np.cos(az) * Z) + (+np.sin(az) * J) / \
                  (np.multiply(Z, Z) + np.multiply(J, J))
        else:
            H_x = (-np.sin(az) * Z) + (-np.cos(az) * J) / \
                  (np.multiply(Z, Z) + np.multiply(J, J))
    H_x[d, d] = 0
    return H_x


def diff_compass(Z, direction='n'):
    """ calculates the neighboring directional difference

    Parameters
    ----------
    Z : np.array, size=(m-2,n-2), dtype=float
        intensity array
    direction : {'n','ne','e','se','s','sw','w','nw'}
        compass direction

    Returns
    -------
    dZ : np.array, size=(m-2,n-2), dtype=float
        directional neighboring difference
    """
    if direction in ('n', 'north', 'up'):
        if Z.ndim == 2:
            dZ = Z[:-2, 1:-1] - Z[1:-1, 1:-1]
        else:
            dZ = Z[:-2, 1:-1, :] - Z[1:-1, 1:-1, :]
    elif direction in ('e', 'east', 'right'):
        if Z.ndim == 2:
            dZ = Z[1:-1, +2:] - Z[1:-1, 1:-1]
        else:
            dZ = Z[1:-1, +2:, :] - Z[1:-1, 1:-1, :]
    elif direction in ('s', 'south', 'down'):
        if Z.ndim == 2:
            dZ = Z[+2:, 1:-1] - Z[1:-1, 1:-1]
        else:
            dZ = Z[+2:, 1:-1, :] - Z[1:-1, 1:-1, :]
    elif direction in ('w', 'west', 'left'):
        if Z.ndim == 2:
            dZ = Z[1:-1, :-2] - Z[1:-1, 1:-1]
        else:
            dZ = Z[1:-1, :-2, :] - Z[1:-1, 1:-1, :]
    elif direction in ('ne', 'norhteast', 'ur', 'ru', 'upperright'):
        if Z.ndim == 2:
            dZ = Z[:-2, +2:] - Z[1:-1, 1:-1]
        else:
            dZ = Z[:-2, +2:, :] - Z[1:-1, 1:-1, :]
    elif direction in ('se', 'southeast', 'lr', 'rl', 'lowerright'):
        if Z.ndim == 2:
            dZ = Z[+2:, +2:] - Z[1:-1, 1:-1]
        else:
            dZ = Z[+2:, +2:, :] - Z[1:-1, 1:-1, :]
    elif direction in ('sw', 'southwest', 'll', 'lowerleft'):
        if Z.ndim == 2:
            dZ = Z[+2:, :-2] - Z[1:-1, 1:-1]
        else:
            dZ = Z[+2:, :-2, :] - Z[1:-1, 1:-1, :]
    elif direction in ('nw', 'northwest', 'ul', 'lu', 'upperleft'):
        if Z.ndim == 2:
            dZ = Z[:-2, :-2] - Z[1:-1, 1:-1]
        else:
            dZ = Z[:-2, :-2, :] - Z[1:-1, 1:-1, :]
    return dZ


def get_grad_filters(ftype='sobel', tsize=3, order=1, indexing='xy'):
    """ construct gradient filters

    Parameters
    ----------
    ftype : {'sobel' (default), 'kroon', 'scharr', 'robinson', 'kayyali', \
             'prewitt', 'simoncelli'}
        Specifies which the type of gradient filter to get:

          * 'sobel' : see [S090]_
          * 'kroon' : see [Kr09]_
          * 'scharr' : see [Sc00]_
          * 'robinson' : see [Ki70]_,[Ro65]_
          * 'kayyali' :
          * 'prewitt' : see [Pr70]_
          * 'simoncelli' : see [FS97]_
    tsize : {3,5}, dtype=integer
        dimenson/length of the template
    order : {1,2}, dtype=integer
        first or second order derivative (not all have this)
    indexing : {'xy','ij'}
        Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output, see
        Notes for more details.

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
    .. [S090] Sobel, "An isotropic 3×3 gradient operator" Machine vision for
              three-dimensional scenes, Academic press, pp.376–379, 1990.
    .. [Kr09] Kroon, "Numerical optimization of kernel-based image
              derivatives", 2009
    .. [Sc00] Scharr, "Optimal operators in digital image processing"
              PhD dissertation University of Heidelberg, 2000.
    .. [Ki70] Kirsch, "Computer determination of the constituent structure of
              biological images" Computers and biomedical research. vol.4(3)
              pp.315–32, 1970.
    .. [Ro65] Roberts, "Machine perception of 3-D solids, optical and
              electro-optical information processing" MIT press, 1965.
    .. [Pr70] Prewitt, "Object enhancement and extraction" Picture processing
              and psychopictorics, 1970.
    .. [FS97] Farid & Simoncelli "Optimally rotation-equivariant directional
              derivative kernels" Proceedings of the international conference
              on computer analysis of images and patterns, pp207–214, 1997
    """

    ftype = ftype.lower()  # make string lowercase

    if ftype == 'kroon':
        if tsize == 5:
            if order == 2:
                H_xx = np.array([[+.0033, +.0045, -0.0156, +.0045, +.0033],
                                 [+.0435, +.0557, -0.2032, +.0557, +.0435],
                                 [+.0990, +.1009, -0.4707, +.1009, +.0990],
                                 [+.0435, +.0557, -0.2032, +.0557, +.0435],
                                 [+.0033, +.0045, -0.0156, +.0045, +.0033]])
                H_xy = np.array([[-.0034, -.0211, 0., +.0211, +.0034],
                                 [-.0211, -.1514, 0., +.1514, +.0211],
                                 [0, 0, 0, 0, 0],
                                 [+.0211, +.1514, 0., -.1514, -.0211],
                                 [+.0034, +.0211, 0., -.0211, -.0034]])
            else:
                H_x = np.array([[-.0007, -.0053, 0, +.0053, +.0007],
                                [-.0108, -.0863, 0, +.0863, +.0108],
                                [-.0270, -.2150, 0, +.2150, +.0270],
                                [-.0108, -.0863, 0, +.0863, +.0108],
                                [-.0007, -.0053, 0, +.0053, +.0007]])
        else:
            if order == 2:
                H_xx = 8 / np.array([[105, -46, 105], [50, -23, 50],
                                     [105, -46, 105]])
                H_xy = 11 / np.array([[-114, np.inf, +114],
                                      [np.inf, np.inf, np.inf],
                                      [+114, np.inf, -114]])
            else:
                H_x = np.outer(np.array([17, 61, 17]), np.array([-1, 0, +1
                                                                 ])) / 95
    elif ftype == 'scharr':
        if tsize == 5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]),
                           np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            if order == 2:
                H_xx = np.outer(np.array([3, 10, 3]), np.array([+1, -2, +1
                                                                ])) / 16
                H_xy = np.outer(np.array([+1, 0, -1]), np.array([-1, 0, +1
                                                                 ])) / 4
            else:
                H_x = np.outer(np.array([3, 10, 3]), np.array([-1, 0, +1
                                                               ])) / 16
    elif ftype == 'sobel':
        if tsize == 5:
            H_x = np.outer(np.array([0.0233, 0.2415, 0.4704, 0.2415, 0.0233]),
                           np.array([-0.0836, -0.3327, 0, +0.3327, +0.0836]))
        else:
            if order == 2:
                H_xx = np.outer(np.array([1, 2, 1]), np.array([+1, -2, +1
                                                               ])) / 4
                H_xy = np.outer(np.array([+1, 0, -1]), np.array([-1, 0, +1
                                                                 ])) / 4
            else:
                H_x = np.outer(np.array([1, 2, 1]), np.array([-1, 0, +1])) / 4
    elif ftype == 'robinson':
        H_x = np.array([[-17., 0., 17.], [-61., 0., 61.], [-17., 0., 17.]
                        ]) / 95
    elif ftype == 'kayyali':
        H_x = np.outer(np.array([1, 0, 1]), np.array([-1, 0, +1])) / 2
    elif ftype == 'prewitt':
        H_x = np.outer(np.array([1, 1, 1]), np.array([-1, 0, +1])) / 3
    elif ftype == 'simoncelli':
        k = np.array([0.125, 0.0, 0.250, 0.500, 1.000])
        d = np.array([0.125, 0.0, 0.250, 0.500, 1.000])
        H_x = np.outer(d, k)

    # elif ftype=='savitzky': #todo

    if order == 1:
        if indexing == 'xy':
            out1, out2 = H_x, np.flip(H_x.T, axis=0)
        else:  # 'ij'
            out1, out2 = H_x.T, H_x
    else:
        out1, out2 = H_xx, H_xy

    return out1, out2


def harst_conv(Z, ftype='bezier'):
    """ construct derivatives through double filtering, see [Ha14]_

    Parameters
    ----------
    Z : np.array, size=(m,n)
        array with intensities
    ftype : {'bezier', 'bspline','catmull','cubic'}
        the type of gradient filter to use. The default is 'bezier'.

    Returns
    -------
    Z_y : np.array, size=(m,n)
        gradient in vertical direction
    Z_x : np.array, size=(m,n)
        gradient in horizontal direction

    References
    ----------
    .. [Ha14] Harst, "Simple filter design for first and second order
              derivatives by a double filtering approach" Pattern recognition
              letters, vol.42 pp.65–71, 2014.
    """
    # define kernels
    un = np.array([0.125, 0.250, 0.500, 1.000])
    up = np.array([0.75, 1.0, 1.0, 0.0])
    if ftype == 'bezier':
        M = np.array([[-1, -3, +3, -1], [+0, +3, -6, +3], [+0, +0, +3, -3],
                      [+0, +0, +0, +1]])
    elif ftype == 'bspline':
        M = np.array([[-1, +3, -3, +1], [+3, -6, +3, +0], [-3, +0, +3, +0],
                      [+1, +4, +1, +0]]) / 6
    elif ftype == 'catmull':
        M = np.array([[-1, +3, -3, +1], [+2, -5, +4, -1], [-1, +0, +1, +0],
                      [+0, +2, +0, +0]]) / 2
    elif ftype == 'cubic':
        M = np.array([[-1, +3, -3, +1], [+3, -6, +3, +0], [-2, -3, +6, -1],
                      [+0, +6, +0, +0]]) / 6

    d = np.matmul(un, M)
    k = np.matmul(up, M)

    Z_y = ndimage.convolve1d(ndimage.convolve1d(Z, np.convolve(d, k), axis=0),
                             np.convolve(k, k),
                             axis=1)
    Z_x = ndimage.convolve1d(ndimage.convolve1d(Z, np.convolve(k, k), axis=0),
                             np.convolve(d, k),
                             axis=1)
    return Z_y, Z_x
