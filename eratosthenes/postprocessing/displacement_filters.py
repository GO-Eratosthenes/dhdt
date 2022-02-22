import numpy as np
from scipy import ndimage

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
    if ~np.isnan(central_pix):
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
    else:
        return central_pix

def local_infilling_filter(I, tsize=5):
    I_new = ndimage.generic_filter(I, infill_func, size=(tsize, tsize))
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
    V : np.array, size=(m,n), dtype=float
        array with one velocity component, all algorithms are indepent of their
        axis.

    Parameters
    ----------
    sig_V : np.array, size=(m,n), dtype=float
        statistical local variance, based on the procedure described in [1],[2]

    References
    ----------
    .. [1] Joughin "Ice-sheet velocity mapping: a combined interferometric and
       speckle-tracking approach", Annuals of glaciology vol.34 pp.195-201.
    .. [2] Joughin et al. "Greenland ice mapping project 2 (GIMP-2) algorithm
       theoretical basis document", Making earth system data records for use in
       research environment (MEaSUREs) documentation, .
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

