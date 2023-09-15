import numpy as np

from ..generic.handler_im import get_grad_filters, conv_2Dfilter
from ..preprocessing.image_transforms import mat_to_gray

from .matching_tools_frequency_filters import make_fourier_grid


def estimate_albedo(Z, unit='angles', normalize=True):
    """ estimate illuminant direction, albedo, and shape from shading with
    constant albedo

    Parameters
    ----------
    Z : numpy.array
        array with intensities
    unit : {'angles','vector'}

    Returns
    -------
    albedo : unit=[], range=0...1
        surface relfectivity

    References
    ----------
    .. [ZC91] Zheng and Chellapa, "Estimation of illuminant direction, albedo,
              and shape from shading," IEEE transactions on pattern analysis
              and machine intelligence, vol.13(7), pp.680-702, 1991.
    """
    if normalize:
        Z = mat_to_gray(Z)
    mu_1, mu_2 = np.mean(Z.flatten()), np.mean(Z.flatten()**2)

    fx, fy = get_grad_filters(ftype='kroon', tsize=3, order=1, indexing='xy')
    I_dx, I_dy = conv_2Dfilter(Z, fx), conv_2Dfilter(Z, fy)

    I_xy = np.hypot(I_dx, I_dy)
    I_x_norm = np.divide(I_dx, I_xy, where=I_xy != 0)
    I_y_norm = np.divide(I_dy, I_xy, where=I_xy != 0)

    mu_x, mu_y = np.mean(I_x_norm.flatten()), np.mean(I_y_norm.flatten())

    # estimate parameters: global albedo, as well as, local slant and tilt
    gamma = np.sqrt((6 * (np.pi**2) * mu_2) - (48 * (mu_1**2)))
    albedo = np.divide(gamma, np.pi, where=gamma != 0)

    tilt = np.arctan2(mu_y, mu_x)
    slant = np.arccos(np.divide(4 * mu_1, gamma, where=mu_1 != 0))

    if unit in (
            'angles',
            'degrees',
    ):
        tilt, slant = np.rad2deg(tilt), np.rad2deg(slant)
        return albedo, tilt, slant
    else:
        sun = np.array([
            np.cos(tilt) * np.sin(slant),
            np.sin(tilt) * np.sin(slant),
            np.cos(slant)
        ])
        return albedo, sun


def linear_fourier_surface_estimation(img, tilt, slant):
    """

    Parameters
    ----------
    img : numpy.array, size=(m,n)
        grid with intensity values
    tilt : float, unit=degrees, range=0...90
        the angle between the illumination vector and Z-axis
    slant : float, unit=degrees, range=-180...+180
        the argument of the illumination projected onto the plane
    iter : integer
        maximum amount of iterations

    Returns
    -------
    Z: numpy.array, size=(m,n)
        grid with elevation values

    See Also
    --------
    dhdt.processing.photoclinometric_functions.linear_reflectance_surface_estimation

    References
    ----------
    .. [Pe90] Pentland, "Linear shape from shading." International journal of
              computer Vision vol.4(2) pp.153-162, 1990.
    """
    tilt, slant = np.deg2rad(tilt), np.deg2rad(slant)
    w_x, w_y = make_fourier_grid(img,
                                 indexing='ij',
                                 system='radians',
                                 shift=False,
                                 axis='corner')

    s_1 = np.cos(tilt) * np.sin(slant)
    s_2 = np.sin(tilt) * np.sin(slant)  # eq.5 in [1]

    I_F = np.fft.fft2(img)
    Z_F = np.divide(I_F, -1j * w_x * s_1 - 1j * w_y * s_2)  # eq.10 in [1]
    Z = np.abs(np.fft.ifft2(Z_F))
    return Z


def global_surface_estimation(img, sun, albedo, kappa=1E3, iter=2.5E3):
    """

    Parameters
    ----------
    img : numpy.array, size=(m,n)
        grid with intensity values
    sun : numpy.array, size=(3,1), dtype=float, range=0...1
        unit vector in the direction of the sun.
    albedo : unit=[], range=0...1
        surface reflectivity
    kappa : float, default=1E3
        smoothness parameter
    iter : integer
        maximum amount of iterations

    Returns
    -------
    Z: numpy.array, size=(m,n)
        grid with elevation values

    """
    p, q = np.zeros_like(img), np.zeros_like(img)  # first order derivatives
    w = .25 * np.ones((3, 3))
    w.flat[::2] = 0

    w_x, w_y = make_fourier_grid(img,
                                 indexing='ij',
                                 system='radians',
                                 shift=False,
                                 axis='corner')
    w_mag = np.hypot(w_x, w_y)

    for k in range(iter):
        # second order derivatives
        p2, q2 = conv_2Dfilter(p, w), conv_2Dfilter(q, w)
        pq = 1 + p**2 + q**2
        sqroot_pq = np.sqrt(pq)
        # reflectance map
        R = np.divide(albedo * (-sun[0] * p - sun[1] * q + sun[2]), sqroot_pq)

        dRdp = np.divide(-albedo * sun[0], sqroot_pq)
        dRdq = np.divide(-albedo * sun[1], sqroot_pq)

        dRb = np.multiply(
            -sun[0] * albedo * p - sun[1] * albedo * q + sun[2] * albedo,
            -1 * p * np.power(pq, -3 / 2))
        dRdp += dRb
        dRdq += dRb

        # update
        p = p2 + np.divide(1, 4 * kappa) * (img - R) * dRdp
        q = q2 + np.divide(1, 4 * kappa) * (img - R) * dRdq

        # make integratable, via Fourier domain
        p_F, q_F = np.fft.fft2(p), np.fft.fft2(q)

        C = -1j * (w_x * p_F + w_y * q_F) / w_mag
        p, q = np.fft.ifft2(1j * w_x * C), np.fft.ifft2(1j * w_y * C)

    Z = np.abs(np.fft.ifft2(C))
    return Z


def linear_reflectance_surface_estimation(img, tilt, slant, iter=2.5E2):
    """

    Parameters
    ----------
    img : numpy.array, size=(m,n)
        grid with intensity values
    tilt : float, unit=degrees, range=0...90
        the angle between the illumination vector and Z-axis
    slant : float, unit=degrees, range=-180...+180
        the argument of the illumination projected onto the plane
    iter : integer
        maximum amount of iterations

    Returns
    -------
    Z: numpy.array, size=(m,n)
        grid with elevation values

    See Also
    --------
    dhdt.processing.photoclinometric_functions.linear_fourier_surface_estimation

    References
    ----------
    .. [TS94] Tsai & Shah, "Shape from shading using linear approximation"
              Image and vision computing, vol.12(8), pp.487--498, 1994.
    """
    p, q = np.zeros_like(img), np.zeros_like(img)  # first order derivatives
    fx, fy = get_grad_filters(ftype='kroon', tsize=3, order=1, indexing='xy')

    tilt, slant = np.deg2rad(tilt), np.deg2rad(slant)
    E_x, E_y = np.cos(tilt) * np.tan(slant), np.sin(tilt) * np.tan(slant)
    sqroot_E = np.sqrt(1 + E_x**2 + E_y**2)

    Z = np.zeros_like(img)
    for k in range(iter):
        pq = 1 + p**2 + q**2
        sqroot_pq = np.sqrt(pq)
        # reflectance map
        R = np.divide(
            p * np.cos(tilt) * np.sin(slant) +
            q * np.sin(tilt) * np.sin(slant) + np.cos(slant), sqroot_sq)
        R = np.max(R, 0)
        f = img - R
        df_dZ = np.divide(np.multiply(p + q, E_x * p + E_y * q + 1),
                          np.multiply(np.sqrt(pq**3), sqroot_E)) - np.divide(
                              E_x + E_y, (sqroot_pq * sqroot_E))

        # update elevation, and derivatives
        Z -= -np.divide(f, df_dZ, where=df_dZ != 0)
        p, q = conv_2Dfilter(Z, fx), conv_2Dfilter(Z, fy)
    return Z
