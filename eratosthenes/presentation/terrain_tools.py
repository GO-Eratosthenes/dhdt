import numpy as np

from ..preprocessing.image_transforms import mat_to_gray

def luminance_perspective(g, z, g_flat, k=1):
    """enhance shadow transform, through classification knowledge

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

    Notes
    -----
    [1] Brassel. "A model for automatic hill-shading" The American cartographer,
    vol.1(1) pp.15-27, 1974.
    [2] Jenny & Patterson. "Aerial perspective for shaded relief" Cartography
    and geographic information science, pp.1-8.
    """
    z_star = mat_to_gray(z)

    g_prime = (g - g_flat)*np.exp(z_star*np.log(k)) + g_flat
    return g_prime

def contrast_perspective(v, z, v_flat, z_thres, v_min=0, k=2):
    """enhance shadow transform, through classification knowledge

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

    Notes
    -----
    [1] Jenny & Patterson. "Aerial perspective for shaded relief" Cartography
    and geographic information science, pp.1-8.
    """
    z_star = mat_to_gray(z)

    # (2) in [1]
    v_prime = v_min + ((v_flat-v_min)/v_flat) *v

    # (4) in [1]
    w = 1 - (z_star/z_thres)**k

    v_pp = w*v_prime + (1-w)*v
    return v_pp


