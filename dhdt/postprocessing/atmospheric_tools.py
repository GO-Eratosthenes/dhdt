import numpy as np

from dhdt.auxilary.handler_era5 import get_era5_monthly_surface_wind
from ..generic.mapping_tools import get_mean_map_lat_lon

def calculate_coriolis(ϕ):
    """

    Parameters
    ----------
    ϕ : float, unit=degrees, range=-90...+90
        latitude
    """
    return 2 * 7.2921e-5 * np.sin(np.deg2rad(ϕ))

def water_vapour_scale_height(T_0, alpha=0.0065, R_v=461, L=2.5E6):
    """

    Parameters
    ----------
    T_0 : float, unit=Celsius
        surface temperature
    alpha : float, unit=K km-1, default=0.0065
        lapse rate in the Troposphere
    R_v : float, unit=J kg-1 K-1, default=461
        universal gas constant
    L : float, unit=J K-1, default=2.5E6
        latent heat of water vapour

    Returns
    -------
    H_w : float, unit=m, range=1000...5000
        vertical height scale of water in the atmosphere

    See Also
    --------
    oro_precip
    """
    H_w = np.divide(-R_v * T_0**2, L*alpha)
    return H_w

def oro_precip(Z, geoTransform, spatialRef, u_wind, v_wind,
               conv_time=1000, fall_time=1000, nm=1E-2, T_0=0,
               lapse_rate=-5.8E-3, lapse_rate_m=-6.5E-3,ref_ρ=7.4E-3):
    """ estimate the orographic precipitation pattern from an elevation model,
    based upon [1]

    Parameters
    ----------
    Z: numpy.array, size=(m,n), unit=m
        array with elevation values
    geoTransform : tuple, size={(1,6), (1,8)}
        georeference transform of the array 'Z'
    spatialRef : osgeo.osr.SpatialReference() object
        coordinate reference system
    u_wind,v_wind : float, unit=m s-1
        wind vector
    conv_time : unit=s, range=200...2000
        conversion time
    fall_time : unit=s, range=200...2000
        fall out time
    nm : float, unit=s-1, range=0...0.1
        moist stability frequency
    T_0 : float, unit=Celsius
        surface temperature
    lapse_rate : float, unit=K km-1, default=-5.8E-3
        environmental lapse rate
    lapse_rate_m : float, unit=K km-1, default=-6.5E-3
        moist adiabatic lapse rate
    ref_ρ : float, unit=kg m**3, default=7.4E-3
        reference saturation water vapor density, i.e. at the surface

    Returns
    -------
    P : numpy.array, size=(m,n), unit=mm hr-1
        array with estimate of the precipitation rate

    See Also
    --------
    calculate_coriolis, water_vapour_scale_height

    Notes
    -----
    The following acronyms are used:
        - CRS : coordinate reference system

    References
    ----------
    .. [1] Smith and Barstad, "A linear theory of orographic precipitation",
       Journal of atmospheric science, vol.61, pp.1377-1391, 2004.
    .. [2] Schuler et al. "Distribution of snow accumulation on the Svartisen
       ice cap, Norway, assessed by a model of orographic precipitation",
       Hydrologic processes, vol.22 pp.3998-4008, 2008.
    """

    # sensitivity uplift factor, unit=kg m**3
    C_w = np.divide(ref_ρ*lapse_rate_m, lapse_rate)

    # vertical height scale of water in the atmosphere, unit=m
    H_w = water_vapour_scale_height(T_0)

    # estimate coriolis force
    ϕ = get_mean_map_lat_lon(geoTransform, spatialRef)[0]
    f_cor = calculate_coriolis(ϕ)

    # pad raster
    m,n = Z.shape[0:2]
    m_pow2, n_pow2 = 2**np.ceil(np.log2(m)), 2**np.ceil(np.log2(n))
    dm, dn = (m_pow2-m).astype(int), (n_pow2-n).astype(int)
    mn_pad = ((dm//2, dm-dm//2), (dn//2, dn-dn//2))
    Z = np.pad(Z, mn_pad, 'constant')
    Z_f = np.fft.fftn(Z)

    k_x = np.fft.fftfreq(m_pow2.astype(int), np.abs(geoTransform[1]))
    k_y = np.fft.fftfreq(n_pow2.astype(int), np.abs(geoTransform[5]))

    K_X,K_Y = np.meshgrid(k_y,k_x)
    K_X *= 2*np.pi
    K_Y *= 2*np.pi
    sigma = K_X * u_wind + K_Y * v_wind # vertical wave number, unit=m

    mf_num = nm**2 - sigma**2
    mf_den = sigma**2 - f_cor**2

    # numerical stability, dividing by zero is not recommended
    np.putmask(mf_num, mf_num<0, 0.)
    ɛ = np.finfo(float).eps
    np.putmask(mf_den, (mf_den < +ɛ) & (mf_den>=0), +ɛ)
    np.putmask(mf_den, (mf_den > -ɛ) & (mf_den<0), -ɛ)

    dir = np.ones_like(Z)
    np.putmask(dir, sigma<0, -1)

    # vertical wavenumber, eq.3 in [2]
    m = dir * np.sqrt(np.abs(np.divide(mf_num,mf_den) *
                              np.hypot(K_X, K_Y)))

    # --- transfer function
    P_hat = ((C_w * 1j * sigma * Z_f) /
             ((1 - (H_w * m * 1j)) *
             (1 + (sigma * conv_time * 1j)) *
             (1 + (sigma * fall_time *1j))))

    P = np.fft.ifftn(P_hat)
    P = np.real(P[mn_pad[0][0]:-mn_pad[0][1], mn_pad[1][0]:-mn_pad[1][1]])
    P *= 3600   # convert to mm hr-1
    np.putmask(P, P<0, 0)
    return P

def annual_precip(Z, geoTransform, spatialRef, year=2018):
    ϕ,λ = get_mean_map_lat_lon(geoTransform, spatialRef)
    U,V,Rh,T = get_era5_monthly_surface_wind(ϕ, λ, year)

    return