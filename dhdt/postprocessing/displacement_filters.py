import numpy as np
from scipy import fftpack, ndimage

from ..generic.filtering_statistical import mad_filtering
from ..generic.handler_im import diff_compass


def mad_func(buffer):
    IN = mad_filtering(buffer, thres=1.4826)
    return IN[IN.size // 2]


def local_mad_filter(Z, tsize=5):
    I_new = ndimage.generic_filter(Z, mad_func, size=(tsize, tsize))
    return I_new


def infill_func(buffer):
    central_pix = buffer[buffer.size // 2]

    if ~np.isnan(central_pix):
        return central_pix
    if np.all(np.isnan(buffer)):
        return np.nan

    # construct distance matrix
    d = int(np.sqrt(buffer.size))
    r = d // 2
    IJ = np.mgrid[-r:+r + 1, -r:+r + 1]  # local coordinate frame
    D = np.linalg.norm(IJ, axis=0)

    D = D[~np.isnan(buffer)]  # remove no-data values in array
    # inverse distance weighting
    W = np.divide(1, D, out=np.zeros_like(D), where=np.isnan(D != 0))
    W /= np.sum(W)  # normalize weights
    interp_pix = np.sum(np.multiply(W, buffer))
    return interp_pix


def local_infilling_filter(Z, tsize=5):
    """ apply local inverse distance weighting, on no-data values in an array

    Parameters
    ----------
    Z : numpy.array, size=(m,n), dtype=float
        intensity array with no-data values
    tsize : integer, {x ∈ ℕ | x ≥ 1}, unit=pixel
        size of the neighborhood

    Returns
    -------
    Z_new : numpy.array, size=(m,n), dtype=float
        intensity array with interpolated values at the data gap locations

    References
    ----------
    .. [Jo02] Joughin "Ice-sheet velocity mapping: a combined interferometric
              and speckle-tracking approach", Annuals of glaciology vol.34
              pp.195-201, 2002.
    .. [Jo17] Joughin et al. "Greenland ice mapping project 2 (GIMP-2)
              algorithm theoretical basis document", Making earth system data
              records for use in research environment (MEaSUREs) documentation,
              2017.
    """

    Z_new = ndimage.generic_filter(Z, infill_func, size=(tsize, tsize))
    return Z_new


def nan_resistant_func(buffer, kernel):
    OUT = np.isnan(buffer)
    # ndimage.generic_filter seems to work from last dimension backwards...
    filter = np.flip(kernel, (0, 1)).copy().ravel()
    if np.all(OUT):
        return np.nan
    elif np.all(~OUT):
        return np.nansum(buffer[~OUT] * filter[~OUT])

    surplus = np.sum(filter[OUT])  # energy to distribute
    grouping = np.sign(filter).astype(int)
    np.putmask(grouping, OUT, 0)  # do not include no-data location in re-organ

    if np.sign(surplus) == -1 and np.any(grouping == -1):
        idx = grouping == -1
        filter[idx] += surplus / np.sum(idx)
        central_pix = np.nansum(buffer[~OUT] * filter[~OUT])
    elif np.sign(surplus) == +1 and np.any(grouping == +1):
        idx = grouping == +1
        filter[idx] += surplus / np.sum(idx)
        central_pix = np.sum(buffer[~OUT] * filter[~OUT])
    elif surplus == 0:
        central_pix = np.sum(buffer[~OUT] * filter[~OUT])
    else:
        central_pix = np.nan

    return central_pix


def nan_resistant_filter(Z, kernel):
    """ operate zonal filter on image with missing values. Energy within the
    kernel that fals on missing instances is distributed over other elements.
    Hence, regions with no data are shrinking, instead of growing. For a more
    detailed describtion see [Al22]_

    Parameters
    ----------
    Z : {numpy.ndarray, numpy.masked.array}, size=(m,n)
        grid with values
    kernel : numpy.ndarray, size=(k,l)
        convolution kernel

    Returns
    Z_new : numpy.ndarray, size=(m,n)
        convolved estimate

    References
    ----------
    .. [Al22] Altena et al. "Correlation dispersion as a measure to better
              estimate uncertainty of remotely sensed glacier displacements",
              The Crysophere, vol.16(6) pp.2285–2300, 2022.
    """
    if type(Z) in (np.ma.core.MaskedArray, ):
        OUT = Z.mask
        Z = Z.data
        np.putmask(Z, OUT, np.nan)

    Z_new = ndimage.generic_filter(Z,
                                   nan_resistant_func,
                                   mode='reflect',
                                   size=kernel.shape,
                                   extra_keywords={'kernel': kernel})
    return Z_new


def var_func(buffer):
    central_pix = buffer[buffer.size // 2]
    if ~np.isnan(central_pix):
        return central_pix

    # construct distance matrix
    d = int(np.sqrt(buffer.size))
    r = d // 2
    IJ = np.mgrid[-r:+r + 1, -r:+r + 1]  # local coordinate frame
    A = np.array([IJ[0].flatten(), IJ[1].flatten(), np.ones(r**2)])
    x = np.linalg.lstsq(A, buffer)[0]
    y_hat = A @ x
    e_hat = y_hat - buffer
    var_pix = np.std(e_hat)
    return var_pix


def local_nonlin_var_filter(Z, tsize=5):
    Z_var = ndimage.generic_filter(Z, var_func, size=(tsize, tsize))
    return Z_var


def local_variance(V, tsize=5):
    """ local non-linear variance calculation

    Parameters
    ----------
    V : numpy.array, size=(m,n), dtype=float
        array with one velocity component, all algorithms are indepent of their
        axis.

    Parameters
    ----------
    sig_V : numpy.array, size=(m,n), dtype=float
        statistical local variance, based on the procedure described in
        [JoXX]_, [JoYY]

    References
    ----------
    .. [Jo02] Joughin "Ice-sheet velocity mapping: a combined interferometric
              and speckle-tracking approach", Annuals of glaciology vol.34
              pp.195-201, 2002.
    .. [Jo17] Joughin et al. "Greenland ice mapping project 2 (GIMP-2)
              algorithm theoretical basis document", Making earth system data
              records for use in research environment (MEaSUREs) documentation,
              2017.
    """
    V_class = local_mad_filter(V, tsize=tsize)
    V[V_class] = np.nan

    V_0 = local_infilling_filter(V, tsize=tsize)

    # running mean adjustment
    mean_kernel = np.ones((tsize, tsize), dtype=float) / (tsize**2)
    V = ndimage.convolve(V, mean_kernel)

    # plane fitting and variance of residual
    sig_V = local_nonlin_var_filter(V, tsize=tsize)
    return sig_V


# todo: relax labelling
