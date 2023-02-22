import numpy as np
from scipy import ndimage, fftpack

from ..generic.filtering_statistical import mad_filtering
from ..generic.handler_im import diff_compass

def mad_func(buffer):
    IN = mad_filtering(buffer, thres=1.4826)
    return IN[IN.size//2]

def local_mad_filter(I, tsize=5):
    I_new = ndimage.generic_filter(I, mad_func, size=(tsize, tsize))
    return I_new

def infill_func(buffer):
    central_pix = buffer[buffer.size//2]

    if ~np.isnan(central_pix): return central_pix
    if np.all(np.isnan(buffer)): return np.nan

    # construct distance matrix
    d = int(np.sqrt(buffer.size))
    r = d//2
    IJ = np.mgrid[-r:+r+1, -r:+r+1] # local coordinate frame
    D = np.linalg.norm(IJ, axis=0)

    D = D[~np.isnan(buffer)] # remove no-data values in array
    # inverse distance weighting
    W = np.divide(1, D,
                  out=np.zeros_like(D),
                  where=np.isnan(D!=0))
    W /= np.sum(W) # normalize weights
    interp_pix = np.sum(np.multiply( W, buffer))
    return interp_pix

def local_infilling_filter(I, tsize=5):
    """ apply local inverse distance weighting, on no-data values in an array

    Parameters
    ----------
    I : numpy.array, size=(m,n), dtype=float
        intensity array with no-data values
    tsize : integer, {x ∈ ℕ | x ≥ 1}, unit=pixel
        size of the neighborhood

    Returns
    -------
    I_new : numpy.array, size=(m,n), dtype=float
        intensity array with interpolated values at the data gap locations

    References
    ----------
    .. [JoXX] Joughin "Ice-sheet velocity mapping: a combined interferometric
              and speckle-tracking approach", Annuals of glaciology vol.34
              pp.195-201.
    .. [JoYY] Joughin et al. "Greenland ice mapping project 2 (GIMP-2) algorithm
              theoretical basis document", Making earth system data records for
              use in research environment (MEaSUREs) documentation.
    """

    I_new = ndimage.generic_filter(I, infill_func, size=(tsize, tsize))
    return I_new

def nan_resistant(buffer, kernel):
    OUT = np.isnan(buffer)
    filter = kernel.copy().ravel()
    if np.all(OUT):
        return np.nan
    elif np.all(~OUT):
        return np.nansum(buffer[~OUT] * filter[~OUT])

    surplus = np.sum(filter[OUT]) # energy to distribute
    grouping = np.sign(filter).astype(int)
    np.putmask(grouping, OUT, 0) # do not include no-data location in re-organ

    if np.sign(surplus)==-1 and np.any(grouping==-1):
        idx = grouping==+1
        filter[idx] += surplus/np.sum(idx)
        central_pix = np.nansum(buffer[~OUT]*filter[~OUT])
    elif np.sign(surplus)==+1 and np.any(grouping==+1):
        idx = grouping == -1
        filter[idx] += surplus /np.sum(idx)
        central_pix = np.nansum(buffer[~OUT] * filter[~OUT])
    elif surplus == 0:
        central_pix = np.nansum(buffer[~OUT] * filter[~OUT])
    else:
        central_pix = np.nan

    return central_pix

def nan_resistant_filter(I, filter):
    """ apply convolution over array with no-data values.

    If no data gaps are present, the kernel is adjusted and more weights are
    given to the other neighbors. If to much data is missing, this results in
    no estimation. From [Al22]_.

    Parameters
    ----------
    I : numpy.array, size=(m,n), dtype=float
        intensity array with no-data values
    filter : numpy.array, size=(k,l), dtype=float
        filter to be applied to the image

    Returns
    -------
    I_new : numpy.array, size=(m,n), dtype=float
        intensity array convolved through the filter

    References
    ----------
    .. [Al22] Altena et al. "Correlation dispersion as a measure to better
              estimate uncertainty of remotely sensed glacier displacements",
              The Crysophere, vol.16(6) pp.2285–2300, 2022.
    """
    #todo if all entities have the same sign
    I_new = ndimage.generic_filter(I, nan_resistant,
                                   size=(filter.shape[0], filter.shape[1]),
                                   extra_arguments=(filter,))
    return I_new

def var_func(buffer):
    central_pix = buffer[buffer.size//2]
    if ~np.isnan(central_pix):
        return central_pix

    # construct distance matrix
    d = int(np.sqrt(buffer.size))
    r = d//2
    IJ = np.mgrid[-r:+r+1, -r:+r+1] # local coordinate frame
    A = np.array([IJ[0].flatten(),
                  IJ[1].flatten(),
                  np.ones(r**2)])
    x = np.linalg.lstsq(A, buffer)[0]
    y_hat = A @ x
    e_hat = y_hat - buffer
    var_pix = np.std(e_hat)
    return var_pix

def local_nonlin_var_filter(I, tsize=5):
    I_var = ndimage.generic_filter(I, var_func, size=(tsize, tsize))
    return I_var

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
        statistical local variance, based on the procedure described in [1],[2]

    References
    ----------
    .. [JoXX] Joughin "Ice-sheet velocity mapping: a combined interferometric
              and speckle-tracking approach", Annuals of glaciology vol.34
              pp.195-201.
    .. [JoYY] Joughin et al. "Greenland ice mapping project 2 (GIMP-2) algorithm
              theoretical basis document", Making earth system data records for
              use in research environment (MEaSUREs) documentation.
    """
    V_class = local_mad_filter(V, tsize=tsize)
    V[V_class] = np.nan

    V_0 = local_infilling_filter(V, tsize=tsize)

    # running mean adjustment
    mean_kernel = np.ones((tsize, tsize), dtype=float)/(tsize**2)
    V = ndimage.convolve(V, mean_kernel)

    # plane fitting and variance of residual
    sig_V = local_nonlin_var_filter(V, tsize=tsize)
    return sig_V

#todo: relax labelling

