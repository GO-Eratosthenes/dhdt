import numpy as np
from scipy.ndimage import convolve

from .filtering_statistical import make_2D_Laplacian
from .handler_im import conv_2Dfilter, get_grad_filters


def terrain_curvature(Z):
    lapl = make_2D_Laplacian(alpha=.5)
    C = convolve(Z, -lapl)
    return C


def ridge_shape(Z):  # todo: check if correct
    # can be peak, hole, valley or saddle
    dx2, _ = get_grad_filters(ftype='kroon', tsize=3, order=2)
    dZ_dx2 = convolve(Z, -dx2)
    dy2 = np.rot90(dx2)
    dZ_dy2 = convolve(Z, -dy2)

    Az = np.degrees(np.arctan2(dZ_dx2, dZ_dy2))
    return Az


def terrain_slope(Z, spac=10.):
    """ use simple local estimation, to calculate slope along mapping axis.
    Following [Ho81]_.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), float, unit=meters
        grid with elevation values
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    Z_dx, Z_dy : numpy.ndarray, size=(m,n), float
        slope angles along the two mapping axis

    See Also
    --------
    get_aspect_slope

    References
    ----------
    .. [Ho81] Horn, "Hill shading and the reflectance map", Proceedings of the
              IEEE vol.69(1) pp.14--47, 1981.
    """
    fx, fy = get_grad_filters(ftype='kroon', tsize=3, order=1, indexing='xy')
    fx /= spac
    fy /= spac
    Z_dx = conv_2Dfilter(Z, fx)
    Z_dy = conv_2Dfilter(Z, fy)
    Z_dx = np.pad(Z_dx, 1, mode='linear_ramp')
    Z_dy = np.pad(Z_dy, 1, mode='linear_ramp')
    return Z_dx, Z_dy


def terrain_aspect_slope(Z, spac=10.):
    """ use simple local estimation, to calculate terrain direction and
    steepest slope. Following [Ho81]_.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), float, unit=meters
        array with elevation values
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    Slp,Asp : numpy.ndarray, size=(m,n), float
        mean slope and aspect angle in the template

    See Also
    --------
    get_template_aspect_slope

    References
    ----------
    .. [Ho81] Horn, "Hill shading and the reflectance map", Proceedings of the
              IEEE vol.69(1) pp.14--47, 1981.
    """
    # calculate gradients
    Z_dx, Z_dy = terrain_slope(Z, spac=spac)

    # calculate terrain parameters
    Asp = np.arctan2(Z_dy, Z_dx)
    Slp = np.hypot(Z_dy, Z_dx)
    return Asp, Slp


def get_slope_corrected_area(Z, spac=10.):
    """ take the slope of the surface into account when the area is calculated

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), unit=meter
        array with elevation
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    A : numpy.ndarray, size=(m,n), unit=meter
        array with area values

    References
    ----------
    .. [RF81] Rasmussen & Ffolliott, "Surface area corrections applied to a
              resource map", Water resources bulletin, vol.17(6), pp.1079–1082,
              1981.
    """
    Z_dx, Z_dy = terrain_slope(Z, spac=10.)
    Slp = np.hypot(Z_dy, Z_dx)
    A = np.cos(Slp) * spac**2
    return A
