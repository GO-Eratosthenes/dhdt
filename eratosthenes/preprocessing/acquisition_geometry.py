import numpy as np

from ..generic.filtering_statistical import make_2D_Gaussian
from .shadow_geometry import estimate_surface_normals

def get_template_aspect_slope(Z,i_samp,j_samp,tsize,spac=10.):
    """

    Parameters
    ----------
    Z : np.array, size=(m,n), float, unit=meters
        array with elevation values
    i_samp : np.array, size=(k,l), integer, unit=pixels
        array with collumn coordinates of the template centers
    j_samp : np.array, size=(k,l), integer, unit=pixels
        array with row coordinates of the template centers
    tsize : integer, unit=pixels
        size of the template
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    Slope, Aspect : np.array, size=(k,l), float
        mean slope and aspect angle in the template

    Notes
    -----
    It is important to know what type of coordinate systems exist, hence:

        .. code-block:: text

          coordinate |           coordinate  ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       j               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
                     | i                     |
                     v                       |
    """
    if (tsize % 2)==0:
        offset = 0 # even template dimension
    else:
        offset = 1 # uneven template dimension

    # create template
    kernel = make_2D_Gaussian((tsize, tsize), fwhm=tsize)
    kernel /= np.sum(kernel)

    # get aspect and slope from elevation
    Normal = estimate_surface_normals(Z, spac)

    Slope = np.zeros_like(i_samp, dtype=np.float64)
    Aspect = np.zeros_like(i_samp, dtype=np.float64)
    for idx_ij, val_i in enumerate(i_samp.flatten()):
        idx_i, idx_j = np.unravel_index(idx_ij, i_samp.shape, order='C')
        i_min = i_samp[idx_i, idx_j] - (tsize // 2) + 0
        j_min = j_samp[idx_i, idx_j] - (tsize // 2) + 0
        i_max = i_samp[idx_i, idx_j] + (tsize // 2) + offset
        j_max = j_samp[idx_i, idx_j] + (tsize // 2) + offset

        n_sub = Normal[i_min:i_max, j_min:j_max]

        # estimate mean surface normal
        slope_bar = np.arccos(np.sum(n_sub[:, :, -1] * kernel))  # slope
        aspect_bar = np.arctan2(np.sum(n_sub[:, :, 0] * kernel),
                                np.sum(n_sub[:, :, 1] * kernel))
        Slope[idx_i, idx_j] = np.degrees(slope_bar)
        Aspect[idx_i, idx_j] = np.degrees(aspect_bar)
    return Slope, Aspect

def get_template_acquisition_angles(Az,Zn,Det,i_samp,j_samp,tsize):
    """

    Parameters
    ----------
    Az : np.array, size=(m,n), float
        array with sensor azimuth angles
    Zn : np.array, size=(m,n), float
        array with sensor zenith angles
    Det : np.array, size=(m,n), bool
        array with sensor detector ids
    i_samp : np.array, size=(k,l), integer
        array with collumn coordinates of the template centers
    j_samp : np.array, size=(k,l), integer
        array with row coordinates of the template centers
    tsize : integer, unit=pixels
        size of the template

    Returns
    -------
    Azimuth, Zenith : np.array, size=(k,l), float
        mean observation angles, that is, azimuth and zenith

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x
    """
    if (tsize % 2)==0:
        offset = 0 # even template dimension
    else:
        offset = 1 # uneven template dimension
    # create template
    kernel = make_2D_Gaussian((tsize,tsize), fwhm=tsize)
    kernel /= np.sum(kernel)

    Azimuth = np.zeros_like(i_samp, dtype=np.float64)
    Zenith = np.zeros_like(i_samp, dtype=np.float64)
    for idx_ij,val_i in enumerate(i_samp.flatten()):
        idx_i, idx_j = np.unravel_index(idx_ij, i_samp.shape, order='C')
        i_min = i_samp[idx_i, idx_j] - (tsize // 2) + 0
        j_min = j_samp[idx_i, idx_j] - (tsize // 2) + 0
        i_max = i_samp[idx_i, idx_j] + (tsize // 2) + offset
        j_max = j_samp[idx_i, idx_j] + (tsize // 2) + offset

        d_sub = Det[i_min:i_max, j_min:j_max]
        z_sub = Zn[i_min:i_max, j_min:j_max]
        a_sub = Az[i_min:i_max, j_min:j_max]

        zenith_bar = np.sum(z_sub*kernel)
        # get majority of detector id, since aspect angles are different
        det_mode = np.bincount(d_sub.flatten()).argmax()
        IN = d_sub==det_mode
        azimuth_bar = np.sum(a_sub[IN]/np.sum(IN))

        Azimuth[idx_i,idx_j] = azimuth_bar
        Zenith[idx_i,idx_j] = zenith_bar
    return Azimuth, Zenith
