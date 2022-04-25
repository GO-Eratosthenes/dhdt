import numpy as np

# image processing libraries
from scipy import ndimage

from .coupling_tools import \
    create_template_at_center, create_template_off_center
from .matching_tools import pad_radius

# precision estimation
def fast_noise_estimation(I, t_size, grd_i, grd_j, Gaussian=True):
    """

    Parameters
    ----------
    I : np.array, size=(m,n), dtype=float
        image with intensities
    t_size : {integer, tuple}
        width and height of the template
    grd_i : np.array, size=(k,l), dtype=integer
        vertical location of the grid points to estimate the variance
    grd_j : np.array, size=(k,l), dtype=integer
        horizontal location of the grid points to estimate the variance
    Gausian : dtype=bool, default=True
        there are two methods presented in [1],  if one assumes a Gaussian
        distribution, then one can use a simpler formulation

    Returns
    -------
    S : np.array, size=(k,l), dtype=float
        image with pixel based noise estimates, based upon [1]

    References
    ----------
    .. [1] Immerkær "Fast noise variance estimation" Computer vision and image
       understanding, vol.64(2) pp.300-302, 1996.
    .. [2] Debella-Gilo and Kääb "Locally adaptive template sizes for matching
       repeat images of Earth surface mass movements" ISPRS journal of
       photogrammetry and remote sensing, vol.69 pp.10-28, 2012.
    """
    # admin
    if not type(t_size) is tuple: t_size = (t_size, t_size)
    t_rad = (t_size[0] //2, t_size[1] //2)

    I = pad_radius(I, t_rad)
    grd_i += t_rad[0]
    grd_j += t_rad[1]

    # single pixel esitmation
    N = np.array([[1, -2, 1],[-2, 4, -2],[1, -2, 1]])

    if Gaussian is True:
        S = ndimage.convolve(I,N)**2
        preamble = 1/(36*(t_size[0]-2)*(t_size[1]-2))
    else:
        S = ndimage.convolve(I,N)
        preamble = np.sqrt(np.pi/2)/(6*(t_size[0]-2)*(t_size[1]-2))

    (m,n) = grd_i.shape
    grd_i,grd_j = grd_i.flatten(), grd_j.flatten()
    L = np.zeros_like(grd_i)
    if np.any(np.mod(t_size,2)): # central template
        for idx, i_coord in enumerate(grd_i):
            S_sub = create_template_at_center(S, i_coord, grd_j[idx], t_rad)
            L[idx] = np.sum(S_sub)
    else: # off-center template
        for idx, i_coord in enumerate(grd_i):
            S_sub = create_template_off_center(S, i_coord, grd_j[idx], t_rad)
            L[idx] = np.sum(S_sub)

    L *= preamble
    return L

# foerstner & Haralick Shapiro, color

# precision descriptors
def helmert_point_error(sig_xx, sig_yy):
    """

    Parameters
    ----------
    sig_xx : np.array, size=(m,n), dtype=float
        estimated standard deviation of the displavement estimates
    sig_yy : np.array, size=(m,n), dtype=float
        estimated standard deviation of the displavement estimates

    Returns
    -------
    sig_H : np.array, size=(m,n), dtype=float
        Helmert point error

    References
    .. [1] Foerstner and Wrobel, "Photogrammetric computer vision. Statistics,
       geometry, orientation and reconstruction", Series on geometry and
       computing vol.11. pp.366, 2016.
    """

    sig_H = np.hypot(sig_xx, sig_yy)
    return sig_H

def geom_mean(sig_xx, sig_yy):
    """

    Parameters
    ----------
    sig_xx : np.array, size=(m,n), dtype=float
        estimated standard deviation of the displavement estimates
    sig_yy : np.array, size=(m,n), dtype=float
        estimated standard deviation of the displavement estimates

    Returns
    -------
    sig_H : np.array, size=(m,n), dtype=float
        geometric mean of the standard error

    References
    .. [1] Foerstner and Wrobel, "Photogrammetric computer vision. Statistics,
       geometry, orientation and reconstruction", Series on geometry and
       computing vol.11. pp.367, 2016.
    """
    L = np.sqrt(np.multiply(sig_xx , sig_yy), 4)
    return L
