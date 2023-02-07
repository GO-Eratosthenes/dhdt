import numpy as np

from scipy import ndimage

# local functions
from ..generic.attitude_tools import rot_mat
from ..processing.matching_tools_frequency_filters import \
    make_fourier_grid


def project_along_flow(dX_raw,dY_raw,dX_prio,dY_prio,e_perp):
    """

    Parameters
    ----------
    dX_raw : numpy.ndarray, size=(m,n), dtype=float
        raw horizontal displacement with mixed signal
    dY_raw : numpy.ndarray, size=(m,n), dtype=float
        raw vertical displacement with mixed signal
    dX_prio : numpy.ndarray, size=(m,n), dtype=float
        reference of horizontal displacement (a-priori knowledge)
    dY_prio : numpy.ndarray, size=(m,n), dtype=float
        reference of vertical displacement (a-priori knowledge)
    e_perp : numpy.ndarray, size=(2,1), float
        vector in the perpendicular direction to the flightline (bearing).

    Returns
    -------
    dX_proj : numpy.ndarray, size=(m,n), dtype=float
        projected horizontal displacement in the same direction as reference.
    dY_proj : numpy.ndarray, size=(m,n), dtype=float
        projected vertical displacement in the same direction as reference.
    
    Notes
    ----- 
    The projection function is as follows:

    .. math:: P = ({d_{x}}e^{\perp}_{x} - {d_{y}}e^{\perp}_{y}) / ({\hat{d}_{x}}e^{\perp}_{x} - {\hat{d}_{y}}e^{\perp}_{y})

    See also Equation 10 and Figure 2 in [1].

    Furthermore, two different coordinate system are used here: 
        
        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x 
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |


    References
    ----------
    .. [1] Altena & Kääb. "Elevation change and improved velocity retrieval 
       using orthorectified optical satellite data from different orbits" 
       Remote Sensing vol.9(3) pp.300 2017.        
    """
    # e_{\para} = bearing satellite...
    assert(dX_raw.size == dY_raw.size) # all should be of the same size
    assert(dX_prio.size == dY_prio.size)
    assert(dX_raw.size == dX_prio.size)
    
    d_proj = ((dX_raw*e_perp[0])-(dY_raw*e_perp[1])) /\
        ((dX_prio*e_perp[0])-(dY_prio*e_perp[1]))

    dX_proj = d_proj * dX_raw
    dY_proj = d_proj * dY_raw 
    return dX_proj,dY_proj 

def rot_covar(V, R):
    V_r = R @ V @ np.transpose(R)
    return V_r

def rotate_variance(Theta, qii, qjj, rho):
    """ rotate the elements of the covariance into a direction of interest

    Parameters
    ----------
    Theta : numpy.ndarray, size=(m,n), unit=degrees
        direction of interest
    qii : numpy.ndarray, size=(m,n), unit={meters, pixels}
        variance in vertical direction
    qjj : numpy.ndarray, size=(m,n), unit={meters, pixels}
        variance in horizontal direction
    rho : numpy.ndarray, size=(m,n), unit=degrees
        orientation of the ellipse

    Returns
    -------
    qii_r : numpy.ndarray, size=(m,n), unit={meters, pixels}
        variance in given direction
    qii_r : numpy.ndarray, size=(m,n), unit={meters, pixels}
        variance in right angle direction

    See Also
    --------
    rotate_disp_field : generic function, using a constant rotation
    """
    qii_r, qjj_r = np.zeros_like(qii), np.zeros_like(qjj)
    for iy, ix in np.ndindex(Theta.shape):
        if ~np.isnan(Theta[iy,ix]) and ~np.isnan(rho[iy,ix]):
            # construct co-variance matrix
            qij = rho[iy,ix]*\
                  np.sqrt(np.abs(qii[iy,ix]))*np.sqrt(np.abs(qjj[iy,ix]))

            V = np.array([[qjj[iy,ix], qij],
                          [qij, qii[iy,ix]]])
            R = rot_mat(Theta[iy,ix])
            V_r = rot_covar(V, R)
            qii_r[iy, ix], qjj_r[iy, ix] = V_r[1][1], V_r[0][0]
        else:
            qii_r[iy,ix], qjj_r[iy,ix] = np.nan, np.nan
    return qii_r, qjj_r

def rotate_disp_field(UV, theta):
    """ rotate a complex number through an angle

    Parameters
    ----------
    UV : numpy.array, size=(m,n), dtype=complex
        grid with displacements in complex form
    theta : {numpy.array, float}, angle, unit=degrees
        rotation angle

    UV_r :  numpy.array, size=(m,n), dtype=complex
        grid with rotated displacements in complex form
    """
    assert(UV.dtype is np.dtype('complex'))
    UV_r = np.multiply(UV, np.exp(1j*np.deg2rad(theta)),
                       out=UV, where=np.invert(np.isnan(UV)))
    return UV_r

def helmholtz_hodge(dX, dY):
    """ apply Helmholtz-Hodge decomposition

    Parameters
    ----------
    dX : numpy.array, size=(m,n)
        horizontal displacement
    dY : numpy.array, size=(m,n)
        vertical displacement

    Returns
    -------
    dX_divfre, dY_divfre : numpy.array, size=(m,n)
        divergence-free vector field
    dX_irrot, dY_irrot : numpy.array, size=(m,n)
        irrotational vector field
    """
    assert type(dX) == np.ndarray, ("please provide an array")
    assert type(dY) == np.ndarray, ("please provide an array")

    # a Fourier transform can not handle nan's very well, hence mask with zeros
    Msk = np.logical_or(np.isnan(dX), np.isnan(dY))
    if np.any(Msk): dX[Msk], dY[Msk] = 0, 0

    dX_F, dY_F = np.fft.fftn(dX), np.fft.fftn(dX)
    F1, F2 = make_fourier_grid(dX, indexing='ij', system='unit')
    K = np.hypot(F1, F2)
    K[0,0] = 1

    div_dXY_F = np.divide(dX_F*F2 + dY_F*F1, K,
                          out=np.zeros_like(dX_F), where=K!=0)

    dX_irrot = np.real(np.fft.ifftn(div_dXY_F*F2))
    dY_irrot = np.real(np.fft.ifftn(div_dXY_F*F1))
    dX_divfre, dY_divfre = dX - dX_irrot, dY - dY_irrot

    if np.any(Msk): dX_irrot[Msk], dY_irrot[Msk] = np.nan, np.nan
    return dX_divfre, dY_divfre, dX_irrot, dY_irrot

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

def relaxation_labelling(I, tsize=3, magnitude=False):
    I_new = ndimage.generic_filter(I, nan_resistant, size=(tsize, tsize),
                                   extra_arguments=(magnitude,))
    return I_new


# geometric_configuration
# temporal_configuration  