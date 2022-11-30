import numpy as np

from scipy.ndimage import convolve #, filters

from .handler_im import get_grad_filters, conv_2Dfilter
from .filtering_statistical import make_2D_Laplacian

def terrain_curvature(Z):
#    C = filters.laplace(Z)
    l = make_2D_Laplacian(alpha=.5)
    C = convolve(Z,-l)
    return C

def ridge_orientation(Z): #todo: check if correct
    dx2,_ = get_grad_filters(ftype='sobel', tsize=3, order=2)
    dZ_dx2 = convolve(Z,-dx2)
    dy2 = np.rot90(dx2)
    dZ_dy2 = convolve(Z,-dy2)

    Az = np.degrees(np.arctan2(dZ_dx2,dZ_dy2))
    return Az

def terrain_slope(Z, spac=10.):
    """ use simple local estimation, to calculate slope along mapping axis

    Parameters
    ----------
    Z : numpy.array, size=(m,n), float, unit=meters
        grid with elevation values
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    Z_dx, Z_dy : np.array, size=(m,n), float
        slope angles along the two mapping axis

    See Also
    --------
    get_aspect_slope

    References
    ----------
    .. [1] Horn, "Hill shading and the reflectance map", Proceedings of the IEEE
       vol.69(1) pp.14--47, 1981.
    """
    fx, fy = get_grad_filters(ftype='kroon', tsize=3, order=1, indexing='xy')
    fx /= spac
    fy /= spac
    Z_dx, Z_dy = conv_2Dfilter(Z, fx), conv_2Dfilter(Z, fy)
    Z_dx, Z_dy = np.pad(Z_dx, 1, mode='linear_ramp'), \
                 np.pad(Z_dy, 1, mode='linear_ramp')
    return Z_dx, Z_dy

def terrain_aspect_slope(Z, spac=10.):
    """ use simple local estimation, to calculate terrain direction and steepest
    slope

    Parameters
    ----------
    Z : numpy.array, size=(m,n), float, unit=meters
        array with elevation values
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    Slp,Asp : np.array, size=(m,n), float
        mean slope and aspect angle in the template

    See Also
    --------
    get_template_aspect_slope

    References
    ----------
    .. [1] Horn, "Hill shading and the reflectance map", Proceedings of the IEEE
       vol.69(1) pp.14--47, 1981.
    """
    # calculate gradients
    Z_dx, Z_dy = terrain_slope(Z, spac=spac)

    # calculate terrain parameters
    Asp = np.arctan2(Z_dy, Z_dx)
    Slp = np.hypot(Z_dy, Z_dx)
    return Asp, Slp