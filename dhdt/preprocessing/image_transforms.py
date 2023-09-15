import numpy as np

from scipy import ndimage

from dhdt.generic.data_tools import gompertz_curve
from dhdt.generic.handler_im import get_grad_filters


def mat_to_gray(Z, notZ=None, vmin=None, vmax=None):
    """ transform matix array  to float, omitting nodata values

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), dtype={integer,float}
        matrix of integers with data
    notZ : numpy.ndarray, size=(m,n), dtype=bool
        matrix assigning which is a no data. The default is None.

    Returns
    -------
    Znew : numpy.ndarray, size=(m,n), dtype=float, range=0...1
        array with normalized intensity values

    See Also
    --------
    gamma_adjustment, log_adjustment, s_curve, inverse_tangent_transformation

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> np.max(img)
    255
    >>> gray = mat_to_gray(img[:,:,0], img[:,:,0]==0)
    >>> np.max(gray)
    1.0
    """
    if notZ is None:
        yesZ = np.ones(Z.shape, dtype=bool)
        notZ = ~yesZ
    else:
        yesZ = ~notZ
    Znew = np.zeros_like(Z, dtype=np.float64)  # /2**16

    if np.ptp(Z) == 0:
        return Znew

    if vmin is not None:
        Z[yesZ] = np.maximum(Z[yesZ], vmin)
    if vmax is not None:
        Z[yesZ] = np.minimum(Z[yesZ], vmax)

    Znew[yesZ] = np.interp(Z[yesZ], (Z[yesZ].min(), Z[yesZ].max()), (0, +1))
    Znew[notZ] = 0
    return Znew


def gamma_adjustment(Z, gamma=1.0):
    """ transform intensity in non-linear way

    enhances high intensities when gamma>1

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), dtype=integer
        array with intensity values
    gamma : float
        power law parameter

    Returns
    -------
    Z_new : numpy.ndarray, size=(m,n)
        array with transform
    """
    if Z.dtype.type == np.uint8:
        radio_range = 2**8 - 1
        look_up_table = np.array([((i / radio_range)**gamma) * radio_range
                                  for i in np.arange(0, radio_range + 1)
                                  ]).astype("uint8")
        Z_new = np.take(look_up_table, Z, out=np.zeros_like(Z))
    elif Z.dtype.type == np.uint16:
        radio_range = 2**16 - 1
        look_up_table = np.array([((i / radio_range)**gamma) * radio_range
                                  for i in np.arange(0, radio_range + 1)
                                  ]).astype("uint16")
        Z_new = np.take(look_up_table, Z, out=np.zeros_like(Z))
    else:  # float
        if np.ptp(Z) <= 1.0:
            Z_new = Z**gamma
        else:
            radio_range = 2**16 - 1
            Z = np.round(Z * radio_range).astype(np.uint16)
            look_up_table = np.array([((i / radio_range)**gamma) * radio_range
                                      for i in np.arange(0, radio_range + 1)
                                      ]).astype("uint16")
            Z_new = np.take(look_up_table, Z, out=np.zeros_like(Z))
            Z_new = (Z_new / radio_range).astype(np.float64)
    return Z_new


def gompertz_adjustment(Z, a=None, b=None):
    """ transform intensity in non-linear way, so high reflected surfaces like
    snow have more emphasized.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), dtype=integer
        array with intensity values
    a,b : float
        parameters for the Gompertz function

    Returns
    -------
    Z_new : numpy.ndarray, size=(m,n)
        grid with transform
    """
    if Z.dtype.type == np.uint8:
        if (a is None) or (b is None):
            a, b = 10, .025
        Z_new = gompertz_curve(Z.astype(float), a, b).astype(np.uint8)
    elif Z.dtype.type == np.uint16:
        if (a is None) or (b is None):
            a, b = 10, .000125
        Z_new = gompertz_curve(Z.astype(float), a, b).astype(np.uint16)
    return Z_new


def log_adjustment(Z):
    """ transform intensity in non-linear way

    enhances low intensities

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n)
        grid with intensity values

    Returns
    -------
    Z_new : numpy.ndarray, size=(m,n)
        grid with transform
    """
    if Z.dtype.type == np.uint8:
        radio_range = 2**8 - 1
    else:
        radio_range = 2**16 - 1

    c = radio_range / (np.log(1 + np.amax(Z)))
    Z_new = c * np.log(1 + Z)
    if Z.dtype.type == np.uint8:
        Z_new = Z_new.astype("uint8")
    else:
        Z_new = Z_new.astype("uint16")
    return Z_new


def hyperbolic_adjustment(Z, intercept):
    """ transform intensity through an hyperbolic sine function

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), dtype=float
        grid with intensity values

    Returns
    -------
    Z_new : numpy.ndarray, size=(m,n), dtype=float, range=-1...+1
        grid with transform
    """

    bottom = np.min(Z)
    ptp = np.ptp(np.array([bottom, intercept]))

    Z_new = np.arcsinh((Z - intercept) * (10 / ptp)) / 3
    Z_new = np.minimum(Z_new, 1)
    return Z_new


def inverse_tangent_transformation(x):
    """ enhances high and low intensities

    Parameters
    ----------
    x : numpy.ndarray, size=(m,n)
        grid with intensity values

    Returns
    -------
    fx : numpy.ndarray, size=(_,_)
        grid with transform

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-1,1,.001)
    >>> y = inverse_tangent_transformation(x)
    >>> plt.figure(), plt.plot(x,y), plt.show()
    shows the mapping function
    """
    fx = np.arctan(x)
    return fx


def get_histogram(img):
    """ calculate the normalized histogram of an image

    Parameters
    ----------
    img : numpy.ndarray, size=(m,n)
        gray scale image array

    Returns
    -------
    hist : numpy.ndarray, size={(2**8,2), (2**16,2)}
        histogram array
    """
    assert img.ndim == 2, ('please provide a 2D array')
    m, n = img.shape[:2]
    bit_depth = 8 if img.dtype.type == np.uint8 else 16
    hist = [0.] * 2**bit_depth

    for i in range(m):
        for j in range(n):
            hist[img[i, j]] += 1
    hist = np.array(hist) / (m * n)
    return hist


def normalize_histogram(img):
    """ transform image to have a uniform distribution

    Parameters
    ----------
    img : numpy.ndarray, size=(m,n), dtype={uint8,unit16}
        image array

    Returns
    -------
    new_img : numpy.ndarray, size=(m,n), dtype={uint8,unit16}
        transfrormed image

    See Also
    --------
    histogram_equalization
    """
    if np.issubdtype(img.dtype, np.integer):
        hist = get_histogram(img)
        cdf = np.cumsum(hist)

        # determine the normalization values for each unit of the cdf
        if img.dtype.type == np.uint8:
            scaling = np.uint8(cdf * 2**8)
        else:
            scaling = np.uint16(cdf * 2**16)

        # normalize the normalization values
        new_img = scaling[img]
    else:  # when array is a float
        cdf = np.argsort(img.flatten(order='C'))
        (m, n) = img.shape
        new_img = np.zeros_like(img).flatten()
        new_img[cdf] = np.linspace(1, m * n, m * n) / (m * n)
        new_img = np.reshape(new_img, (m, n), 'C')
    return new_img


def histogram_equalization(img, img_ref):
    """ transform the intensity values of an array, so they have a similar
    histogram as the reference.

    Parameters
    ----------
    img : numpy.array, size=(m,n), ndim=2
        asd
    img_ref : numpy.array, size=(m,n), ndim=2

    Returns
    -------
    img : numpy.array, size=(m,n), ndim=2
        transformed image

    See Also
    --------
    normalize_histogram
    general_midway_equalization : equalize multiple imagery
    """
    assert img.ndim == 2, ('please provide a 2D array')
    assert img_ref.ndim == 2, ('please provide a 2D array')

    mn = img.shape[:2]
    img, img_ref = img.flatten(), img_ref.flatten()

    if type(img) in (np.ma.core.MaskedArray, ):
        np.putmask(img, np.ma.getmaskarray(img), np.nan)
    if type(img_ref) in (np.ma.core.MaskedArray, ):
        np.putmask(img_ref, np.ma.getmaskarray(img_ref), np.nan)

    # make resistant to NaN's
    IN, IN_ref = ~np.isnan(img), ~np.isnan(img_ref)
    # do not do processing if there is a lack of data
    if (np.sum(IN) < 4) or (np.sum(IN_ref) < 4):
        return img.reshape(mn)

    new_img = np.zeros_like(img)
    val_img, val_idx, cnt_img = np.unique(img[IN],
                                          return_counts=True,
                                          return_inverse=True)
    val_ref, cnt_ref = np.unique(img_ref[IN_ref], return_counts=True)

    qnt_img = np.cumsum(cnt_img).astype(np.float64)
    qnt_ref = np.cumsum(cnt_ref).astype(np.float64)
    qnt_img /= qnt_img[-1]  # normalize
    qnt_ref /= qnt_ref[-1]

    # implement the mapping
    intp_img = np.interp(qnt_img, qnt_ref, val_ref)

    new_img[IN] = intp_img[val_idx]
    new_img[~IN] = np.nan
    return new_img.reshape(mn)


def cum_hist(img):
    """ compute the cumulative histogram of an image

    Parameters
    ----------
    img : numpy.array, size=(m,n), ndim=2
        grid with intensity values

    Returns
    -------
    cdf : numpy.array, size=(2**n,), dtype=integer
        emperical cumulative distribution function, i.e. a cumulative
        histogram.
    """
    bit_depth = 8 if img.dtype.type == np.uint8 else 16

    if type(img) in (np.ma.core.MaskedArray, ):
        w = np.invert(np.ma.getmaskarray(img)).astype(float)
        pdf = np.histogram(np.ma.getdata(img),
                           weights=w,
                           bins=range(0, 2**bit_depth))[0]
    else:
        pdf = np.histogram(img, bins=range(0, 2**bit_depth))[0]
    cdf = np.cumsum(pdf).astype(float)
    cdf = cdf / np.sum(pdf)
    return cdf


def psuedo_inv(cumhist, val):
    """
    Parameters
    ----------
    cumhist : numpy.array, size=(2**n,), dtype={integer, float}
        emperical cumulative distribution function, i.e. a cumulative
        histogram.
    val : {integer,float}

    Returns
    -------
    res : {integer,float}
        the pseudo-inverse of cumhist

    Notes
    -----
    The following acronyms are used:

    CDF - cumulative distribution function
    """
    res = np.min(np.where(cumhist >= val))
    return res


def general_midway_equalization(Z):
    """ equalization of multiple imagery, based on [De04]_.

    Parameters
    ----------
    Z : numpy.array, dim=3, type={uint8,uint16}
        stack of imagery

    Returns
    -------
    Z_new : numpy.array, dim=3, type={uint8,uint16}
        harmonized stack of imagery

    References
    ----------
    .. [De04] Delon, "Midway image equalization", Journal of mathematical
              imaging and Vision", vol.21(2) pp.119-134, 2004

    Notes
    -----
    The following acronyms are used:

    CDF - cumulative distribution function
    """

    # compute the corresponding cumulative histograms (CDFs) of all N images:

    CDFs = []

    assert Z.ndim == 3, ('please provide an array with multiple dimensions')
    bit_depth = 8 if Z.dtype.type == np.uint8 else 16
    im_depth = Z.shape[2]
    for k in range(im_depth):
        if type(Z) in (np.ma.core.MaskedArray, ):
            CDFs.append(cum_hist(Z[..., k].compressed()))
        else:
            CDFs.append(cum_hist(Z[..., k]))

    idx_min_CDF, idx_max_CDF = (2**bit_depth) - 1, 0
    for k in range(im_depth):
        idx_min_CDF = np.minimum(np.argmin(CDFs[k] == 0), idx_min_CDF)
        idx_max_CDF = np.maximum(np.argmax(CDFs[k] == 1), idx_max_CDF)

    Z_new = np.zeros_like(Z, dtype=np.float64)
    for x in range(im_depth):
        res = np.zeros((Z_new.shape[0:2]), dtype=np.float64)
        for p in range(im_depth):
            # compute pseudo inverse
            inv_CDF = np.empty_like(CDFs[p])
            for i in range(idx_min_CDF, idx_max_CDF):
                inv_CDF[i] = psuedo_inv(CDFs[p], CDFs[x][i])
            inv_CDF[idx_max_CDF:] = np.max(inv_CDF)
            # compute midway equalization (using floating-point):
            res += np.take(inv_CDF, Z[..., x])
            # minus is there to resolve the indexing difference between
            # Python and pixel intensities
        res *= 1 / im_depth
        Z_new[..., x] = res

    return Z_new


def high_pass_im(Im, radius=10):
    """
    Parameters
    ----------
    Im : numpy.array, ndim={2,3}, size=(m,n,_)
        data array with intensities
    radius : float, unit=pixels
        hard threshold of the sphere in Fourier space

    Returns
    -------
    Inew : numpy.array, ndim={2,3}, size=(m,n,_)
        data array with high-frequency signal
    """
    m, n = Im.shape[:2]

    # Fourier coordinate system
    (I, J) = np.mgrid[:m, :n]
    Hi = np.fft.fftshift(1 -
                         np.exp(-np.divide(((I - m / 2)**2 +
                                            (J - n / 2)**2), (2 * radius)**2)))

    if Im.ndim == 2:
        If = np.fft.fft2(Im)
        Inew = np.real(np.fft.ifft2(If * Hi))
        return Inew
    else:
        Inew = np.zeros_like(Im)
        for i in range(Im.shape[2]):
            If = np.fft.fft2(Im[..., i])
            Inew[..., i] = np.real(np.fft.ifft2(If * Hi))
        return Inew


def multi_spectral_grad(Ims):
    """ get the dominant multi-spectral gradient of the stack, stripped down
    version of the fusion technique, described in [SW02]_.

    Parameters
    ----------
    Ims : numpy.array, dim=3, dtype=float
        stack of imagery

    Returns
    -------
    grad_I : numpy.array, dim=2, dtype=complex
        fused gradient of the image stack

    References
    ----------
    .. [SW02] Socolinsky & Wolff, "Multispectral image visualization through
              first-order fusion", IEEE transactions on image fusion, vol.11(8)
              pp.923-931, 2002.
    """
    fx, fy = get_grad_filters(ftype='sobel', tsize=3, order=1)

    Idx = ndimage.convolve(Ims, np.atleast_3d(fx))  # steerable filters
    Idy = ndimage.convolve(Ims, np.atleast_3d(fy))

    _, s, v = np.linalg.svd(np.stack((Idx, Idy), axis=-1), full_matrices=False)
    grad_I = np.squeeze(s[..., 0] * v[..., 0] + 1j * s[..., 0] * v[..., 1])
    return grad_I
