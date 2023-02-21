import numpy as np

from ..generic.filtering_statistical import make_2D_Gaussian
from ..generic.handler_im import \
    bilinear_interpolation
from ..generic.mapping_tools import create_offset_grid, vel2pix
from ..processing.matching_tools import pad_radius
from .shadow_geometry import estimate_surface_normals

def slope_along_perp(Z, Az, spac=10):
    """ get the slope of the terrain along a specific direction

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), units=meters
        elevation model
    Az : {numpy.ndarray, float}, units=degrees
        argument or azimuth angle of interest
    spac : float, units=meters
        spacing used for the elevation model

    Returns
    -------
    Slp_para : numpy.ndarray, size=(m,n), units=degrees
        slope in the along direction of the azimuth angle
    Slp_perp : numpy.ndarray, size=(m,n), units=degrees
        slope in the perpendicular direction of the azimuth angle
    """
    dy, dx = np.gradient(Z, spac)

    Slp_para = dx*np.cos(np.rad2deg(Az)) + dy*np.sin(np.rad2deg(Az))
    Slp_perp = dx*np.sin(np.rad2deg(Az)) - dy*np.cos(np.rad2deg(Az))

    # transfer to slopes in degrees
    Slp_para = np.degrees(np.arctan(Slp_para))
    Slp_perp = np.degrees(np.arctan(Slp_perp))
    return Slp_perp, Slp_para

def get_ortho_offset(Z, dx, dy, obs_az, obs_zn, geoTransform):
    """ get terrain displacement due to miss-registration

    Parameters
    ----------
    Z : numpy.array, size=(m,n), unit=meter
        elevation model
    dx, dy : float, unit=meter
        mis-registration
    obs_az : numpy.array, size=(m,n), unit=degrees
        azimuth angle of observation
    obs_zn : numpy.ndarray, size=(m,n), unit=degrees
        zenith angle of observation
    geoTransform : tuple, size={(6,), (8,)}
        affine transformation coefficients of array Z

    Returns
    -------
    dI, dJ : numpy.ndarray, size=(m,n), unit=pixels
        terrain displacements

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

    The azimuth angle is declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x

    The angles related to the satellite are as follows:

        .. code-block:: text

               #*#                   #*# satellite
          ^    /                ^    /|
          |   /                 |   / | nadir
          |-- zenith angle      |  /  v
          | /                   | /|
          |/                    |/ | elevation angle
          +----- surface        +------

    """
    I_grd, J_grd = create_offset_grid(Z, dx, dy, geoTransform)
    Z_dij = bilinear_interpolation(Z, I_grd, J_grd)

    # estimate elevation change due to miss-registration
    dZ = Z-Z_dij

    # estimate orthorectification compensation
    ortho_ρ = np.tan(np.deg2rad(obs_zn))*dZ

    dI = -np.cos(np.deg2rad(obs_az[0,0]))*ortho_ρ
    dJ = +np.sin(np.deg2rad(obs_az))*ortho_ρ
    return dI, dJ

def compensate_ortho_offset(I, Z, dx, dy, obs_az, obs_zn, geoTransform):
    """

    Parameters
    ----------
    I : numpy.array, size=(m,n), ndim=2, dtype={float,integer}
        grid with intensities to be compensated
    Z : numpy.array, size=(m,n), ndim=2, dtype={float,integer}, unit=meters
        grid with elevations
    dx, dy : float, unit=meter
        mis-registration offset
    obs_az : numpy.array, size=(m,n), unit=degrees
        azimuth angle of observation
    obs_zn : numpy.ndarray, size=(m,n), unit=degrees
        zenith angle of observation
    geoTransform1 : tuple
        affine transformation coefficients of array Z

    Returns
    -------
    I_cor : numpy.array, size=(m,n), ndim=2, dtype=float
        corrected intensities

    See Also
    --------
    get_ortho_offset
    """
    # warp
    dI,dJ = get_ortho_offset(Z, dx, dy, obs_az, obs_zn, geoTransform)

    mI, nI = Z.shape[0], Z.shape[1]
    I_grd, J_grd = np.meshgrid(np.linspace(0, mI-1, mI),
                               np.linspace(0, nI-1, nI), indexing='ij')

    I_warp = bilinear_interpolation(I, I_grd+dI, J_grd+dJ)
    del I # sometime the files are very big, so memory is emptied
    # remove registration mismatch
    dI, dJ = vel2pix(geoTransform, dx, dy)
    I_cor = bilinear_interpolation(I_warp, dI, dJ)
    return I_cor

def get_template_aspect_slope(Z,i_samp,j_samp,t_size,spac=10.):
    """

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), float, unit=meters
        array with elevation values
    i_samp : numpy.ndarray, size=(k,l), integer, unit=pixels
        array with collumn coordinates of the template centers
    j_samp : numpy.ndarray, size=(k,l), integer, unit=pixels
        array with row coordinates of the template centers
    t_size : integer, {x ∈ ℕ | x ≥ 1}, unit=pixels
        size of the template
    spac : float, unit=meters
        spacing of the elevation grid

    Returns
    -------
    Slope, Aspect : numpy.ndarray, size=(k,l), float
        mean slope and aspect angle in the template

    See Also
    --------
    get_aspect_slope

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
    # take care of border cases
    t_rad = t_size // 2
    Z = pad_radius(Z, t_rad)
    i_samp += t_rad
    j_samp += t_rad

    if (t_size % 2)==0:
        offset = 0 # even template dimension
    else:
        offset = 1 # uneven template dimension

    # create template
    kernel = make_2D_Gaussian((t_size, t_size), fwhm=t_size)
    kernel /= np.sum(kernel)

    # get aspect and slope from elevation
    Normal = estimate_surface_normals(Z, spac)

    Slope = np.zeros_like(i_samp, dtype=np.float64)
    Aspect = np.zeros_like(i_samp, dtype=np.float64)
    for idx_ij, val_i in enumerate(i_samp.flatten()):
        idx_i, idx_j = np.unravel_index(idx_ij, i_samp.shape, order='C')
        i_min = i_samp[idx_i, idx_j] - t_rad + 0
        j_min = j_samp[idx_i, idx_j] - t_rad + 0
        i_max = i_samp[idx_i, idx_j] + t_rad + offset
        j_max = j_samp[idx_i, idx_j] + t_rad + offset

        n_sub = Normal[i_min:i_max, j_min:j_max]
        # estimate mean surface normal
        try:
            slope_bar = np.arccos(np.sum(n_sub[...,-1] * kernel))  # slope
            aspect_bar = np.arctan2(np.sum(n_sub[..., 0] * kernel),
                                    np.sum(n_sub[..., 1] * kernel))
        except:
            breakpoint
        Slope[idx_i, idx_j] = np.degrees(slope_bar)
        Aspect[idx_i, idx_j] = np.degrees(aspect_bar)
    return Slope, Aspect

#todo:

def get_template_acquisition_angles(Az,Zn,Det,i_samp,j_samp,t_size):
    """

    Parameters
    ----------
    Az : numpy.ndarray, size=(m,n), float
        array with sensor azimuth angles
    Zn : numpy.ndarray, size=(m,n), float
        array with sensor zenith angles
    Det : numpy.ndarray, size=(m,n), bool
        array with sensor detector ids
    i_samp : numpy.ndarray, size=(k,l), integer
        array with collumn coordinates of the template centers
    j_samp : numpy.ndarray, size=(k,l), integer
        array with row coordinates of the template centers
    t_size : integer, {x ∈ ℕ | x ≥ 1}, unit=pixels
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
    # take care of border cases
    t_rad = t_size // 2
    Az,Zn = pad_radius(Az,t_rad), pad_radius(Zn,t_rad)
    Det = pad_radius(Det,t_rad)
    i_samp += t_rad
    j_samp += t_rad

    if (t_size % 2)==0:
        offset = 0 # even template dimension
    else:
        offset = 1 # uneven template dimension
    # create template
    kernel = make_2D_Gaussian((t_size,t_size), fwhm=t_size)
    kernel /= np.sum(kernel)

    Azimuth = np.zeros_like(i_samp, dtype=np.float64)
    Zenith = np.zeros_like(i_samp, dtype=np.float64)
    for idx_ij,val_i in enumerate(i_samp.flatten()):
        idx_i, idx_j = np.unravel_index(idx_ij, i_samp.shape, order='C')
        i_min = i_samp[idx_i, idx_j] - (t_size // 2) + 0
        j_min = j_samp[idx_i, idx_j] - (t_size // 2) + 0
        i_max = i_samp[idx_i, idx_j] + (t_size // 2) + offset
        j_max = j_samp[idx_i, idx_j] + (t_size // 2) + offset

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