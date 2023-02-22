import numpy as np
from scipy import ndimage

from skimage.filters.rank import entropy

from ..generic.handler_im import get_grad_filters
from ..generic.filtering_statistical import make_2D_Gaussian
from ..preprocessing.image_transforms import mat_to_gray
from ..postprocessing.solar_tools import make_shading

# pre-processing tools

# todo: A design of contour generation for topographic maps
# with adaptive DEM smoothing. Kettunen et al. 2017.

def selective_blur_func(C, t_size):
    # decompose arrays, these seem to alternate....
    C_col = C.reshape(t_size[0],t_size[1],2)
    if np.any(np.sign(C[::2])==-1):
        M, I = -1*C_col[:,:,0].flatten(), C_col[:,:,1].flatten()
    else: # redundant calculation, just ignore and exit
        return 0

    m_central = M[(t_size[0]*t_size[1])//2]

    W = make_2D_Gaussian(t_size, fwhm=m_central).flatten()
    W /= np.sum(W)
    new_intensity = np.sum(W*I)
    return new_intensity

def adaptive_elevation_smoothing(Z, t_size=7, g_max=7):
    """ smooth elevation data, depending on the local entropy, based on [Pa20]_.

    Parameters
    ----------
    Z : np.array, size=(m,n)
        array with elevation values

    Returns
    -------
    Z_new : np.array, size=(m,n)
        array with adjusted elevation values

    References
    ----------
    .. [Pa20] Raposo, "Variable DEM generalization using local entropy for
              terrain representation through scale" International journal of
              cartography, vol.6(1) pp.99-120, 2020.
    """
    #radii=5*90 [m]
    # entropy
    E = entropy(mat_to_gray(Z), np.ones((50, 50), dtype=bool))

    # invert and normalize
    max_entr = np.max(E)
    W = (max_entr - E) / max_entr

    G = (W*(g_max-1) +1)
    Combo = np.dstack((Z, -1.*(G)))

    # adaptive moving window of gaussian smoothing
    Z_new = ndimage.generic_filter(Combo, selective_blur_func,
                                   footprint=np.ones((t_size, t_size, 2)),
                                   mode='mirror', cval=np.nan,
                                   extra_keywords = {'t_size': (t_size, t_size)}
                                  )
    #todo build scale space box

    Z_new = Z_new[...,0]
    return Z_new

def curvature_enhanced_shading(Z, az=-45, zn=+45, spac=10, c=0.2):
    """ combine single illumination shading with slope shading, see [Ke08]_.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n)
        array with elevation values
    az : float, unit=degrees
        azimuth or argument angle
    zn : float, unit=degrees
        zenith angle of illumination source
    spac : float
        planimetric spacing in relation to the elevation values

    Returns
    -------
    Shd : numpy.ndarray, size=(m,n), dtype=float
        shading with additional emphasis on steep terrain

    References
    ----------
    .. [Ke08] Kennelly, "Terrain maps displaying hill-shading with curvature"
              Geomorphology, vol.102 pp.567-577, 2008.
    """
    Sh_imh = make_shading(Z, az, zn, spac=spac)

    h_xx,h_xy = get_grad_filters(ftype='kroon', tsize=3, order=2)
    H = ndimage.convolve(Z, h_xx) + ndimage.convolve(Z, h_xy) + \
        ndimage.convolve(Z, h_xy.T)
    H = c*mat_to_gray(H)

    Shd = Sh_imh + H
    return Shd

# advanced shading functions
def enhanced_vertical_shading(Z, az=-45, zn=+45, spac=10, c=.5):
    """ combine single illumination shading with slope shading, see [Ke08]_.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n)
        array with elevation values
    az : float, unit=degrees
        azimuth or argument angle
    zn : float, unit=degrees
        zenith angle of illumination source
    spac : float
        planimetric spacing in relation to the elevation values
    c : float, range=0...1
        relative weighting, c=0: only shading, c=1: only zenith illumination

    Returns
    -------
    Shd : numpy.ndarray, size=(m,n), dtype=float
        shading with additional emphasis on steep terrain

    References
    ----------
    .. [Ke08] Kennelly, "Terrain maps displaying hill-shading with curvature"
              Geomorphology, vol.102 pp.567-577, 2008.
    """
    Sh_imh = make_shading(Z, az, zn, spac=spac)
    Sh_zen = make_shading(Z,+45, 0., spac=spac)
    Shd = ((1-c)*Sh_imh + c*Sh_zen)/2
    return Shd

def luminance_perspective(g, z, g_flat, k=1.3):
    """enhance shadow transform, through classification knowledge, see [Br74]_
    and [JBXX]_.

    Parameters
    ----------
    g : np.array, size=(m,n)
        shading imagery
    z : np.array, size=(m,n)
        elevation data
    g_flat : value for flat regions

    Returns
    -------
    g_prime : np.array, size=(m,n)
        perspective shading

    References
    ----------
    .. [Br74] Brassel. "A model for automatic hill-shading" The American
              cartographer, vol.1(1) pp.15-27, 1974.
    .. [JPXX] Jenny & Patterson. "Aerial perspective for shaded relief"
              Cartography and geographic information science, pp.1-8.
    """
    z_star = mat_to_gray(z)

    g_prime = (g - g_flat)*np.exp(z_star*np.log(k)) + g_flat
    return g_prime

def contrast_perspective(v, z, v_flat, z_thres, v_min=0, k=2): #todo
    """enhance shadow transform, through classification knowledge, see [JPXX]_.

    Parameters
    ----------
    g : np.array, size=(m,n)
        shading imagery
    z : np.array, size=(m,n)
        elevation data
    g_flat : value for flat regions


    Returns
    -------
    g_prime : np.array, size=(m,n)
        perspective shading

    References
    -----
    .. [JPXX] Jenny & Patterson. "Aerial perspective for shaded relief"
              Cartography and geographic information science, pp.1-8.
    """
    z_star = mat_to_gray(z)

    # (2) in [1]
    v_prime = v_min + ((v_flat-v_min)/v_flat) *v

    # (4) in [1]
    w = 1 - (z_star/z_thres)**k

    v_pp = w*v_prime + (1-w)*v
    return v_pp
