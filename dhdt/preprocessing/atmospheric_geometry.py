import os
import warnings

import pandas
import numpy as np

from ..generic.debugging import loggg
from ..generic.unit_conversion import datetime2doy, \
    celsius2kelvin, kelvin2celsius
from dhdt.auxilary.handler_era5 import get_era5_atmos_profile


def get_CO2_level(lat, date, m=3, nb=3):
    """
    Rough estimation of CO₂, based upon measurements of global 'Station data'__

    The annual trend is based upon the 'Global trend'__ in CO₂ while the
    seasonal trend is based upon relations as shown in Figure 1 in [Ba16]_.
    Where a latitudal component is present in the ampltitude, as well as, an
    asymmetric triangular wave.

    Parameters
    ----------
    lat : float, unit=degrees, range=-90...+90
        latitude of the location of interest
    date : np.datetime64, range=1950...2030
        time of interest
    m : float, default=3
        assymetry of the wave
        - m<=2: crest is leftwards leaning
        - m==2: symmetric wave
        - m>=2: the ramp is leftwards leaning
    nb : integer, default=3
        number of base functions to use, higher numbers will give a more pointy
        wave function

    Returns
    -------
    total_co2 : float, unit=ppm
        estimate of spatial temporal carbon dioxide (CO₂) value

    Notes
    -----
    Data where this fitting is based on, can be found here:

    .. _'Global trend': https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5
    .. _'Station data': https://gml.noaa.gov/dv/iadv/

    The following acronyms are used:

        - CO2 : carbon dioxide
        - ppm : parts per million

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Plot the annual trend at 60ºN

    >>> t = np.arange("1950-01-01","2022-01-01",dtype="M8[Y]")
    >>> co2 = get_CO2_level(60., t)
    >>> plt.plot(t, co2)

    Plot seasonal signal

    >>> t = np.arange("2018-01-01","2022-01-01",dtype="M8[D]")
    >>> co2 = get_CO2_level(60., t)
    >>> plt.plot(t, co2)

    References
    ----------
    .. [Ba16] Barnes et al. "Isentropic transport and the seasonal cycle
              amplitude of CO₂" Journal of geophysical research: atmosphere,
              vol.121 pp.8106-8124, 2016.
    """

    # global trend:
    # using a polynomial upwards trend, unfortunately
    decadal_co2 = np.poly1d(np.array([0.012786034930234986,
                                      -49.31617858270089,
                                      47857.733173381115]))
    year,doy = datetime2doy(date)
    frac_yr = doy/365.25
    co2_base = decadal_co2(year+frac_yr)

    # annual trend:
    # using an asymmetric triangular wave, with latitudinal component
    if np.sign(lat)==-1: m *= -1
    b = np.zeros(nb) # estimate base
    for n in np.linspace(1,nb,nb):
        b[int(n-1)] = -np.divide(2*(-1**n)*(m**2) ,(n**2)*(m-1)*(np.pi**2)) * \
           np.sin(np.divide(n*(m-1)*np.pi,m))
    annual_co2 = 0
    austral_offset = np.pi if lat<0  else 0
    for idx,para in enumerate(b):
        fb = - para * np.sin(np.divide(((idx + 1) * (frac_yr*np.pi)), 0.5))
        annual_co2 += fb

    annual_amp = 18 * np.tanh(np.abs(lat) / 30)
    annual_co2 *= -annual_amp

    total_co2 = co2_base + annual_co2
    return total_co2

def get_height_tropopause(ϕ):
    """ The tropopause can be related to its latitude [Al73]_ and a least
    squares fit is given in [No99]_.

    Parameters
    ----------
    ϕ : {float, numpy.ndarray}, unit=degrees, range=-90...+90
        latitude of interest

    Returns
    -------
    h_t : {float, numpy.ndarray}, unit=meters
        estimate of tropopause, that is the layer between the troposphere and
        the stratosphere

    See Also
    --------
    get_mean_lat

    Notes
    -----
    The Strato- and Troposphere have different refractive behaviour.
    A simplified view of this can be like:

        .. code-block:: text

                              z_s
            Mesosphere       /
          ┌-----------------/-┐  ↑ 80 km    |   *       T_s  |*
          |                /  |  |          |   *            |*
          | Stratosphere  /   |  |          |  *             |*
          |              /    |  |          |  *             |*
          ├--Tropopause-/-----┤  |  ↑ 11 km |_*_________T_t  |*________
          |            |      |  |  |       | **             | *
          |  Tropo     |      |  |  |       |    **          |  **
          |  sphere    | z_h  |  |  |       |       **  T_h  |     ****
          └------------#------┘  ↓  ↓       └-----------T_0  └---------
           Earth surface                    Temperature      Pressure

    References
    ----------
    .. [Al73] Allen, "Astrophysical quantities", pp.121-124, 1973.
    .. [No99] Noerdlinger, "Atmospheric refraction effects in Earth remote
              sensing" ISPRS journal of photogrammetry and remote sensing,
              vol.54, pp.360-373, 1999.
    """
    ϕ = np.deg2rad(ϕ)
    h_t = 17786.1 - 9338.96*np.abs(ϕ) + 1271.91*(ϕ**2)  # eq. A9 in [No99]_
    return h_t

def get_T_sealevel(lat, method='noerdlinger'):
    """ The temperature at sea level can be related to its latitude [Al73]_ and
    a least squares fit is given in [No99]_ and [Ya16]_.

    Parameters
    ----------
    lat : {float, numpy.ndarray}, unit=degrees, range=-90...+90
        latitude of interest

    Returns
    -------
    T_sl : {float, numpy.array}, unit=Kelvin
        estimate of temperature at sea level
    method : {'noerdlinger', 'yan'}
        - 'noerdlinger', from [2] has no dependency between hemispheres
        - 'yan', from [3] is hemisphere dependent

    References
    ----------
    .. [Al73] Allen, "Astrophysical quantities", pp.121-124, 1973.
    .. [No99] Noerdlinger, "Atmospheric refraction effects in Earth remote
              sensing" ISPRS journal of photogrammetry and remote sensing,
              vol.54, pp.360-373, 1999.
    .. [Ya16] Yan et al, "Correction of atmospheric refraction geolocation error
              of high resolution optical satellite pushbroom images"
              Photogrammetric engineering & remote sensing, vol.82(6)
              pp.427-435, 2016.
    """
    if method in ['noerdlinger']:
        T_sl = 245.856 + 53.4894*np.cos(np.deg2rad(lat))    # eq. A10 in [2]
    else:
        if isinstance(lat, float):
            if lat>=0: # northern hemisphere
                T_sl = 52.07 * np.cos(np.deg2rad(lat)) + (273.15 - 26.4)
            else:
                T_sl = 78.08 * np.cos(np.deg2rad(lat)) + (273.15 - 48.2)
        else:
            T_sl = np.zeros_like(lat)
            N = lat>=0
            np.putmask(T_sl, N,
                       52.07 * np.cos(np.deg2rad(lat[N])) + (273.15 - 26.4))
            np.putmask(T_sl, ~N,
                       78.08 * np.cos(np.deg2rad(lat[~N])) + (273.15 - 48.2))
                                                            # eq. 19 in [3]
    return T_sl

def get_T_in_troposphere(h, T_0):
    """
    Using parameters of the first layer of the eight layer atmospheric model of
    ISO 2533:1970 to estimate the temperature at different altitudes

    Parameters
    ----------
    h : {float, numpy.array}, unit=m
        altitude above sea-level
    T_0 : float, unit=Kelvin
        temperature at the surface (reference)

    Returns
    -------
    T_h : {float, numpy.array}, unit=Kelvin
        temperature in the troposphere

    References
    ----------
    .. [Ya16] Yan et al, "Correction of atmospheric refraction geolocation error
              of high resolution optical satellite pushbroom images"
              Photogrammetric engineering & remote sensing, vol.82(6)
              pp.427-435, 2016.
    """

    t_0 = kelvin2celsius(T_0)
    t_h = t_0 + np.divide((-56.5 - t_0)*h, 11019)           # eq. 20 in [1]
    T_h = celsius2kelvin(t_h)
    return T_h

def get_water_vapor_press(T):
    """

    Parameters
    ----------
    t : {float, numpy.array}, unit=Kelvin
        temperature
    Returns
    -------
    P_w : {float, numpy.array}, unit=Pascal
        water vapor pressure

    See Also
    --------
    get_sat_vapor_press

    References
    ----------
    .. [Ya16] Yan et al, "Correction of atmospheric refraction geolocation error
              of high resolution optical satellite pushbroom images"
              Photogrammetric engineering & remote sensing, vol.82(6)
              pp.427-435, 2016.
    """
    t = kelvin2celsius(T)
    P_w = (7.38E-3*t + 0.8072)**8 - \
        1.9E-5*(1.8*t + 48) + \
        1.316E-3
    P_w *= 3386.39          # eq. 23 in [1]
    return P_w

def get_sat_vapor_press(T, method='IAPWS'):
    """ uses either [1] or the international standard of IAPWS [2].

    Parameters
    ----------
    T : {float, numpy.array}, unit=Kelvin
        temperature

    Returns
    -------
    svp : {float, numpy.array}
        saturation vapor pressure

    See Also
    --------
    get_water_vapor_press_ice, get_water_vapor_enhancement

    References
    ----------
    .. [Gi82] Giacomo, "Equation for the determination of the density of moist
              air" Metrologica, vol.18, pp.33-40, 1982.
    .. [WP02] Wagner and Pruß, "The IAPWS formulation 1995 for the thermodynamic
              properties of ordinary water substance for general and scientific
              use.” Journal of physical and chemical reference data, vol.31(2),
              pp.387-535, 2002.
    """
    if method in ('simple', 'old'):
        A, B, C, D = 1.2378847E-5, -1.9121316E-2, 33.93711047, -6.3431645E3
        svp = np.exp(A*T**2 + B*T + C + D/T)        # eq.22 in [1]
    else:
        k_1,k_2 = 1.16705214528E3, -7.24213167032E5
        k_3,k_4,k_5 = -1.70738469401E1, 1.20208247025E4, -3.23255503223E6
        k_6,k_7,k_8 = 1.49151086135E1, -4.82326573616E3, 4.05113405421E5
        k_9,k_10 = -2.38555575678E-1, 6.50175348448E2

        omega = T + np.divide(k_9 ,(T - k_10))
        a = omega**2 + np.multiply(k_1, omega) + k_2
        b = np.multiply(k_3, omega**2) + np.multiply(k_4, omega) + k_5
        c = np.multiply(k_6, omega**2) + np.multiply(k_7, omega) + k_8
        x = -b + np.sqrt(b**2 - (4 * a * c))
        svp = 1E6*np.power(np.divide(2*c, x), 4)
    return svp

def get_sat_vapor_press_ice(T):
    """ following the international standard of IAPWS [2].

    Parameters
    ----------
    T : {float, numpy.array}, unit=Kelvin
        temperature

    Returns
    -------
    svp : {float, numpy.array}
        saturation vapor pressure

    References
    ----------
    .. [WP02] Wagner and Pruß, "The IAPWS formulation 1995 for the thermodynamic
              properties of ordinary water substance for general and scientific
              use.” Journal of physical and chemical reference data, vol.31(2),
              pp.387-535, 2002.
    """
    a_1, a_2 = -13.928169, 34.7078238
    t_frac = np.divide(T, 273.16)
    y = np.multiply(a_1, 1 - np.power(t_frac, -1.5)) + \
        np.multiply(a_2, 1 - np.power(t_frac, -1.25))
    svp = 611.657 * np.exp(y)
    return svp

def get_water_vapor_press_ice(T):
    """

    Parameters
    ----------
    T : {float, numpy.array}, unit=Kelvin
        temperature

    Returns
    -------
    P_w : {float, numpy.array}, unit=Pascal
        water vapor pressure

    See Also
    --------
    get_sat_vapor_press, get_water_vapor_press

    References
    ----------
    .. [MM93] Marti & Mauersberger, "A survey and new measurements of ice vapor
              pressure at temperatures between 170 and 250K" Geophysical
              research letters, vol.20(5) pp.363-366, 1993.
    """
    A, B = -2663.5, 12.537
    P_w = 10**(np.divide(A,T)+B)                            # eq. 1 from [1]
    return P_w

def get_water_vapor_enhancement(t,p):
    """ estimate the enhancement factor for the humidity calculation

    Parameters
    ----------
    t : {float, numpy.array}, unit=Celsius
        temperature
    p : {float, numpy.array}, unit=Pascal
        atmospheric pressure

    Returns
    -------
    f : {float, numpy.array}
        enhancement factor

    See Also
    --------
    get_water_vapor_press_ice, get_water_vapor_enhancement

    References
    ----------
    .. [Gi82] Giacomo, "Equation for the determination of the density of moist
              air" Metrologica, vol.18, pp.33-40, 1982.
    """
    alpha, beta, gamma = 1.00062, 3.14E-8, 5.6E-7
    f = alpha + beta*p + gamma*t**2                 # eq.23 in [1]
    return f

def get_density_fraction(ϕ, h_0, alpha=0.0065, R=8314.36, M_t=28.825):
    """

    Parameters
    ----------
    ϕ : unit=degrees
        latitude of point of interest
    h_0 : float, unit=meters
        initial altitude above geoid
    alpha : float, unit=K km-1, default=0.0065
        lapse rate in the Troposphere
    R : float, unit=J kmol-1 K-1, default=8314.36
        universal gas constant
    M_t : float, unit=kg kmol-1, default=28.825
        mean moleculear mass in the Troposphere

    Returns
    -------
    ρ_frac : float
        fraction of density in relation to density as sea-level

    Notes
    -----
    The following parameters are used

    T_sl : unit=Kelvin, temperature at sea level
    T_t : unit=Kelvin, temperature at the Tropopause
    h_t : unit=meters, altitude of the Tropopause
    g : unit=m s-2, acceleration due to gravity

    References
    ----------
    .. [No99] Noerdlinger, "Atmospheric refraction effects in Earth remote
              sensing" ISPRS journal of photogrammetry and remote sensing,
              vol.54, pp.360-373, 1999.
    """
    T_sl = get_T_sealevel(ϕ)
    T_frac_0 = 1 - np.divide(h_0*alpha, T_sl)           # eq. A4 in [No99]

    # get estimate of gravity
    g = 9.784 * (1 - (2.6E-3 * np.cos(np.deg2rad(2 * ϕ))) - (2.8E-7 * h_0))

    Gamma = np.divide(M_t*g, R*alpha) - 1               # eq. A6 in [No99]
    ρ_frac_t = T_frac_0**Gamma                        # eq. A5 in [No99]

    # estimate parameters of the tropopause
    h_t = get_height_tropopause(ϕ)
    T_t = h_t*alpha + T_sl

    ρ_frac_s = T_frac_0**Gamma * \
        np.exp(h_0-h_t) * np.divide(M_t*g, R*T_t)       # eq. A7 in [1]
    return ρ_frac_t, ρ_frac_s

def get_simple_density_fraction(ϕ):
    """ The density fraction is related to latitude through a least squares fit
    as is given in [No99]_.

    Parameters
    ----------
    ϕ : {float, numpy.array}, unit=degrees, range=-90...+90
        latitude of interest

    Returns
    -------
    densFrac : {float, numpy.array}
        estimate of the density fraction

    See Also
    --------
    get_mean_lat

    References
    ----------
    .. [No99] Noerdlinger, "Atmospheric refraction effects in Earth remote
              sensing" ISPRS journal of photogrammetry and remote sensing,
              vol.54, pp.360-373, 1999.
    """
    ρ_frac = 1.14412 - 0.185488*np.cos(np.deg2rad(ϕ))   # eq. A11 in [No99]_
    return ρ_frac

def refractive_index_broadband_vapour(σ):
    """

    Parameters
    ----------
    σ : {float, numpy.array}, unit=µm-1
        vacuum wavenumber, that is the reciprocal of the vacuum wavelength

    Returns
    -------
    n_ws : {float, numpy.array}
        refractive index for standard moist air at sea level

    References
    ----------
    .. [Ow67] Owens, "Optical refractive index of air: Dependence on pressure,
              temperature and composition", Applied optics, vol.6(1) pp.51-59,
              1967.
    .. [Ci96] Ciddor, "Refractive index of air: new equations for the visible
              and near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    """
    cf = 1.022
    w_0, w_1, w_2, w_3 = 295.235, 2.6422, -0.032380, 0.004028
                                                            # eq.13 in [Ow67]
                                                            # eq.3 in [Ci96]
    n_ws = cf*(w_0 + w_1*np.power(σ,2)+ w_2*np.power(σ,4)+ w_3*np.power(σ,6))
    n_ws /= 1E8
#    n_ws += 1
    return n_ws

def water_vapor_frac(p,T,fH):
    """

    Parameters
    ----------
    p : float, unit=Pascal
        total pressure
    T : float, unit=Kelvin
        temperature
    fH : float, range=0...1
        fractional humidity

    Returns
    -------
    x_w : float
        molar fraction of water vapor in moist air

    Notes
    -----
    The following parameters are used in this function:

    T : unit=Kelvin, temperature
    svp : unit=Pascal, saturation vapor pressure
    t : unit=Celcius, temperature

    References
    ----------
    .. [Ci96] Ciddor, "Refractive index of air: new equations for the visible
              and near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    .. [MM93] Marti & Mauersberger, "A survey and new measurements of ice vapor
              pressure at temperatures between 170 and 250K" Geophysical
              research letters, vol.20(5) pp.363-366, 1993.
    .. [Gi82] Giacomo, "Equation for the determination of the density of moist
              air" Metrologica, vol.18, pp.33-40, 1982.
    """
    if isinstance(T, float):
        T,p = np.asarray([T]), np.asarray([p])
    svp = np.zeros_like(T)

    if T.size!=p.size:
        p = p*np.ones_like(T)

    Freeze = T <= 273.15

    np.putmask(svp, Freeze, get_sat_vapor_press_ice(T[Freeze]))
    np.putmask(svp, ~Freeze, get_sat_vapor_press(T[~Freeze]))

    t = kelvin2celsius(T) # transform to Celsius scale
    # enhancement factor of water vapor
    f = np.ones_like(t)
    if np.any(~Freeze):
        np.putmask(f, ~Freeze,
                   get_water_vapor_enhancement(t[~Freeze],p[~Freeze]))

    x_w = np.divide(f*fH*svp, p)            # eq.19 in [3]
    return x_w

def compressability_moist_air(p, T, x_w):
    """

    Parameters
    ----------
    p : float, unit=Pascal
        total pressure
    T : float, unit=Kelvin
        temperature
    x_w : float
        molar fraction of water vapor in moist air

    Returns
    -------
    Z : float
        compressability factor

    See Also
    --------
    water_vapor_frac

    References
    ----------
    .. [Ci96] Ciddor, "Refractive index of air: new equations for the visible
              and near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    .. [DaXX] Davis, "Equation for the determination of the density of moist air
              (1981/91)" Metrologica, vol.29 pp.67-70.
    """
    if isinstance(T, float): T = np.array([T])

    # values from Table1 in [2], taking the 1991 version
    a_0, a_1, a_2 = 1.58123E-6, -2.9331E-8, 1.1043E-10
    b_0, b_1 = 5.707E-6, -2.051E-8
    c_0, c_1 = 1.9898E-4, -2.376E-6
    d, e = 1.83E-11, -0.765E-8

    # there seem to be two temperature scales used in the equation...
    T_k, t = T.copy(), kelvin2celsius(T.copy())

    p_tk = p/T_k
    Z = 1 - p_tk*(a_0 + a_1*t + a_2*(t**2)         # eq.12 in [1]
                   + (b_0 + b_1*t)*x_w             # eq.5 in [2]
                   + (c_0 + c_1*t)*(x_w**2)) \
                   + p_tk**2 * (d + e*(x_w**2))
    return Z

def get_density_air(T,P,fH, moist=True, CO2=450, M_w=0.018015, R=8.314510,
                    saturation=False):
    """

    Parameters
    ----------
    T : float, unit=Kelvin
        temperature at location
    P : float, unit=Pascal
        the total pressure in Pascals
    fH : float, range=0...1
        fractional humidity
    moist : bool
        estimate moist (True) or dry air (False)
    CO2 : float, unit=ppm of CO₂
        parts per million of CO₂ in the atmosphere
    M_w : float, unit=kg mol-1
        molar mass of water vapor
    R : float, unit=J mol-1 K-1
        universal gas constant

    Returns
    -------
    ρ : float, unit=kg m3
        density of moist air

    Notes
    -----
    The following parameters are used in this function:

        - M_a : unit=kg mol-1, molar mass of dry air
        - x_w : molecular fraction of water vapor
        - Z : compressibility of moist air
        - CO2 : unit=ppm, carbon dioxide
        - ppm : parts per million volume

    References
    ----------
    .. [Ci96] Ciddor, "Refractive index of air: new equations for the visible
              and near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    .. [Da92] Davis, "Equation for the determination of the density of moist air
              (1981/91)" Metrologica, vol.29 pp.67-70, 1992.
    """
    # molar mass of dry air containing XXX ppm of CO2
    M_a = ((CO2-400)*12.011E-6 + 28.9635)*1E-3 # eq2 in [Da92]_

    x_w = water_vapor_frac(P,T,fH)
    if np.any(fH==1):
        np.putmask(x_w, fH==1, 1.)

    Z = compressability_moist_air(P, T, x_w)
    if moist is True: # calculate moisture component
        ρ = np.divide(x_w*P*M_a, Z*R*T)*\
              (1 - x_w*(1-np.divide(M_w, M_a)) )
    else:
        ρ = np.divide((1-x_w)*P*M_a, Z*R*T)*\
              (1 - x_w*(1-np.divide(M_w, M_a)) ) # eq. 4 in [Ci96]_
    return ρ

def ciddor_eq5(ρ_a, ρ_w, n_axs, n_ws, ρ_axs, ρ_ws):
    """ estimating the refractive index through summation

    Parameters
    ----------
    ρ_a : {numpy.array, float}, unit=kg m-3
        density of dry air
    ρ_w : (numpy.array, float}, unit=kg m-3
        density of moist air
    n_axs : numpy.array, size=(k,)
        refractive index for dry air at sea level at a specific wavelength
    n_ws : numpy.array, size=(k,)
        refractive index for standard moist air at sea level for a specific
        wavelength
    ρ_axs : float, unit=kg m-3
        density of standard dry air at sea level
    ρ_ws : float, unit=kg m-3
        density of standard moist air at sea level

    Returns
    -------
    n_prop : float
        refractive index

    References
    ----------
    .. [Ow67] Owens, "Optical refractive index of air: Dependence on pressure,
              temperature and composition", Applied optics, vol.6(1) pp.51-59,
              1967.
    """
    # eq. 4 in [1]
    n_prop = 1 + \
             np.multiply(np.divide(ρ_a, ρ_axs), n_axs) + \
             np.multiply(np.divide(ρ_w, ρ_ws), n_ws)

    return n_prop

def ciddor_eq6(ρ_a, ρ_w, n_axs, n_ws, ρ_axs, ρ_ws):
    """ estimating the refractive index through the Lorenz-Lorentz relation

    Parameters
    ----------
    ρ_a : float, unit=kg m-3
        density of dry air
    ρ_w : float, unit=kg m-3
    n_axs : float
        refractive index for dry air at sea level
    n_ws : float
        refractive index for standard moist air at sea level
    ρ_axs : float, unit=kg m-3
        density of standard dry air at sea level
    ρ_ws : float, unit=kg m-3
        density of standard moist air at sea level

    Returns
    -------
    n_LL : float
        refractive index

    References
    ----------
    .. [Ow67] Owens, "Optical refractive index of air: Dependence on pressure,
              temperature and composition", Applied optics, vol.6(1) pp.51-59,
              1967.
    .. [Ci96] Ciddor, "Refractive index of air: new equations for the visible
              and near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    """
    L_a = np.divide( n_axs**2 -1, n_axs**2 +2)                  # eq.6 in [2]
    L_w = np.divide( n_ws**2 - 1, n_ws**2 + 2)                  # eq.6 in [2]

    L = np.multiply.outer( np.divide(ρ_a, ρ_axs), L_a) + \
        np.multiply.outer( np.divide(ρ_w, ρ_ws), L_w)           # eq.7 in [2]

    n_LL = np.sqrt(np.divide( (1 + 2*L), (1 - L)))              # eq. 8 in [2]
                                                                # eq. 4 in [1]
    return n_LL

def refractive_index_visible(df, T, P, p_w=None, CO2=None):
    """ calculate refractive index, based on measurement around 633nm, see [1].

    It applies to ambient atmospheric conditions over the range of wavelengths
    350 nm to 650 nm.

    Parameters
    ----------
    df : {dataframe, float}, unit=µm
        central wavelengths of the spectral bands
    T : {float, numpy.array}, unit=Celcius
        temperature at location
    P : {float, numpy.array}, unit=Pascal
        atmospheric pressure at location
    CO2 : {float, numpy.array}, unit=ppm
        concentration of carbon dioxide (CO₂) in the atmosphere

    Returns
    -------
    n_0 : numpy.array
        index of refraction

    See Also
    --------
    refractive_index_broadband

    References
    ----------
    .. [BJ94] Birch and Jones, "Correction to the updated Edlén equation for the
              refractive index of air", Metrologica, vol.31(4) pp.315-316, 1994.
    .. [Ed66] Edlén, "The refractive index of air", Metrologica, vol.2(2)
              pp.71-80, 1966.
    .. [BJ94] Birch and Jones, "An updated Edlén equation for the refractive
              index of air", Metrologica, vol.30 pp.155-162, 1993.
    """
    if isinstance(T, float): T = np.array([T])
    if isinstance(P, float): P = np.array([P])
    assert T.size==P.size, \
        ('please provide Temperature and Pressure array of the same size')

    if type(df) in (pandas.core.frame.DataFrame, ):
        sigma = 1/df['center_wavelength'].to_numpy() # (vacuum) wavenumber
    else:
        sigma = np.array([1/df])

    # (n-1)_s : refractivity of standard air
    n_s = 8342.54 + \
         np.divide(15998, 38.9-(sigma**2)) + \
         np.divide(2406147, 130-(sigma**2))    # eq.2 in [1]
    n_s *= 1E-8

    if CO2 is not None:
        n_s = refractive_index_CO2(n_s, CO2)

    # (n-1)_tp : refractivity of standard air at a certain temp & pressure
    D_s = 96095.43 # density factor of standard air,
    # which should be in the range of 5...30C [2], which gives 720.775 as this
    # is torr.
    if n_s.ndim==2: # multiple wavelengths
        n_tp_1 = np.divide(np.outer(P,n_s), D_s)
    else:
        n_tp_1 = np.divide(P*n_s, D_s)[:, None]
    n_tp_2 = np.divide(1 + 1E-8*(0.601-(0.00972*T))*P,
                                 1 + (0.0036610*T) )

    # Lorenz-Lorenz equation
    n_tp = np.squeeze(n_tp_1 * n_tp_2[:, None])  # eq.1 in [1]

    n_0 = n_tp
    if p_w is not None:
        dn_tpf = -p_w * \
                (3.7345 - 0.0401*sigma**2) * 10E-10     # eq.3 in [1]
        n_0 -= dn_tpf                                   # eq.6 in [3] in torr

    n_0 += 1.
    return n_0

def refractive_index_CO2(n_s, CO2):
    """ correct standard refractivity for diffferent carbon dioxide
    concentrations, see [2] for more information.

    Parameters
    ----------
    n_s : {float, numpy.array}
        refractivity of standard air
    CO2 : {float, numpy.array}, unit=ppm
        concentration of carbon dioxide (CO₂) in the atmosphere

    Returns
    -------
    n_x : {float, numpy.array}
        refractivity of standard air at a given CO₂ concentration

    Notes
    -----
    The following acronyms are used:

        - CO2 : carbon dioxide
        - ppm : parts per million

    References
    ----------
    .. [BD93] Birch & Downs, "An updated Edlen equation for the refractive index
              of air" Metrologica, vol.30(155) pp.155-162, 1993.
    .. [Ed66] Edlén, "The refractive index of air", Metrologica, vol.2(2)
              pp.71-80, 1966.
    """
    CO2 /= 1E6 # bring to unit
    n_x = np.outer(1+0.540*(CO2-0.0003), n_s)  # eq.7 in [1]
    return np.squeeze(n_x)

def refractive_index_broadband(df, T_0, P_0, fH_0, CO2=450.,
                               T_a=15., P_a=101325.,
                               T_w=20., P_w=1333.,
                               LorentzLorenz=False):
    """ Estimate the refractive index in for wavelengths in the range of
    300...1700 nm, see also [1].

    Parameters
    ----------
    df : {dataframe, float}, unit=µm
        central wavelengths of the spectral bands
    CO2 : float, unit=ppm of CO₂, default=450.
        parts per million of CO₂
    T_0 : float, unit=Celsius
        temperature at location
    P_0 : float, unit=Pascal
        the total pressure in Pascal
    fH_0 : float, range=0...1
        fractional humidity
    T_a : float, unit=Celcius
        temperature of standard dry air, see Notes
    P_a : float, unit=Pascal
        pressure of standard dry air, see Notes
    T_w : float, unit=Celsius
        temperature of standard moist air, see [1]
    P_w : float, unit=Pascal
        pressure of standard moist air, see [1]
    LorentzLorenz : bool
        Two ways are possible to calculate the proportional refraction index,
        when
            - True : the Lorentz-Lorenz relationship is used.
            - False : component wise addition is used

    Returns
    -------
    n : {float, numpy.array}
        index of refraction

    Notes
    -----
    Standard air [2] is defined to be dry air at a temperature of 15 degrees
    Celsius and a total pressure of 1013.25 mb, having the following composition
    by molar percentage:
        - 78.09% nitrogen (N₂)
        - 20.95% oxygen (O₂)
        -  0.93% argon (Ar)
        -  0.03% carbon dioxide (C0₂)

    See Also
    --------
    ciddor_eq5 : calculation used for the Lorentz-Lorenz relationship
    ciddor_eq6 : calculation used for component wise addition
    refractive_index_visible

    References
    ----------
    .. [Ci96] Ciddor, "Refractive index of air: new equations for the visible
              and near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    .. [Ow67] Owens, "Optical refractive index of air: Dependence on pressure,
              temperature and composition", Applied optics, vol.6(1) pp.51-59,
              1967.
    """
    if type(df) in (pandas.core.frame.DataFrame, ):
        sigma = 1/df['center_wavelength'].to_numpy() # (vacuum) wavenumber
    else:
        sigma = np.array([1/df])

    if np.any(np.logical_or(sigma>1/0.3,sigma<1/1.7)):
        warnings.warn("wavelength(s) seem to be out of the domain,"+
                      " i.e.: 0.3...1.7 µm")

    # convert to correct unit
    T_0 = celsius2kelvin(T_0)
    T_a, T_w = celsius2kelvin(T_a), celsius2kelvin(T_w)

    # refractivities of dry air,
    # amended for changes in temperature and CO2 content
    k_0, k_1, k_2, k_3 = 238.0185, 5792105., 57.362, 167917.

    n_as = (np.divide(k_1, k_0 - sigma**2) +
            np.divide(k_3, k_2 - sigma**2) )*1E-8            # eq.1 in [1]

    n_axs = (n_as * (1 + (.534E-6 * (CO2 - 450))))           # eq.2 in [1]
    n_ws = refractive_index_broadband_vapour(sigma)          # eq.3 in [1]

    rho_a = get_density_air(T_0,P_0,fH_0,moist=False, CO2=CO2)
    rho_w = get_density_air(T_0,P_0,fH_0,moist=True, CO2=CO2)    # eq.4 in [1]

    rho_axs = get_density_air(T_a,P_a,0.,moist=False, CO2=CO2)
    rho_ws = get_density_air(T_w,P_w,1.,moist=True, CO2=CO2)

    if not LorentzLorenz:
        n = ciddor_eq5(rho_a, rho_w, n_axs, n_ws, rho_axs, rho_ws)
                                                                # eq.5 in [1]
    else:
        n = ciddor_eq6(rho_a, rho_w, n_axs, n_ws, rho_axs, rho_ws)
                                                                # eq.6 in [1]
    n = np.squeeze(n)
    return n

def refractive_index_simple(p,RH,t):
    """ simple shop-floor formula fro refraction, see appendix B in [1].

    Parameters
    ----------
    p : {numpy.array, float}, unit=kPa
        pressure
    RH : {numpy.array, float}, unit=percent
        relative humidity
    t : {numpy.array, float}, unit=Celsius
        temperature

    Returns
    -------
    n : {numpy.array, float}
        refractive index

    References
    ----------
    .. [1] https://emtoolbox.nist.gov/wavelength/Documentation.asp#AppendixB
    """
    n = 1+7.86E-4*p/(273+t)-1.5E-11 * RH *(t**2+160)
    return n

def refraction_angle_analytical(zn_0, n_0):
    """

    Parameters
    ----------
    zn_0 : {float, numpy.array}, unit=degrees
        observation angle, when atmosphere is not taken into account
    n_0 : {float, numpy.array}
        refractive index at the surface

    Returns
    -------
    zn : {float, numpy.array}, unit=degrees
        corrected observation angle, taking refraction into account

    References
    ----------
    .. [No99] Noerdlinger, "Atmospheric refraction effects in Earth remote
              sensing" ISPRS journal of photogrammetry and remote sensing,
              vol.54, pp.360-373, 1999.
    """
    zn = np.rad2deg(np.arcsin(np.sin(np.deg2rad(zn_0))/n_0))
                                                                # eq. 11 in [1]
    return zn

@loggg
def get_refraction_angle(dh, x_bar, y_bar, spatialRef, central_wavelength,
                         h, simple_refraction=True):
    """ estimate the refraction angle

    Parameters
    ----------
    dh : pandas.DataFrame
        photohypsometric data collection, having at least the following columns:
            - timestamp : date stamp
            - caster_X, caster_Y : map coordinate of start of shadow
            - caster_Z : elevation of the start of the shadow trace
            - casted_X, casted_Y : map coordinate of end of shadow
            - azimuth : sun orientation, unit=degrees
            - zenith : overhead angle of the sun, unit=degrees
    x_bar, y_bar : numpy.array, unit=meter
        map location of interest
    spatialRef : osgeo.osr.SpatialReference
        projection system describtion
    central_wavelength : {pandas.DataFrame, float}, unit=µm
        central wavelengths of the spectral bands
    h : numpy.array, unit=meter
        altitudes of interest
    simple_refraction : boolean, deault=True
        - True : estimate refraction in the visible range
        - False : more precise in a broader spectral range

    Returns
    -------
    dh : pandas.DataFrame
        photohypsometric data collection, with an addtional column, namely:
            - zenith_refrac : overhead angle of the sun which does take the
                              atmosphere into account, unit=degrees
    """
    assert type(dh) in (pandas.core.frame.DataFrame, ), \
        ('please provide a pandas dataframe')
    assert np.all([header in dh.columns for header in
                  ('timestamp','zenith', 'caster_X', 'caster_Y', 'caster_Z')])

    lat, lon, z, Temp, Pres, fracHum, t_era = get_era5_atmos_profile(
        dh['timestamp'].unique(), x_bar, y_bar, spatialRef, z=h)

    # calculate refraction for individual dates
    for timestamp in dh['timestamp'].unique():
        IN = dh['timestamp'] == timestamp
        sub_dh = dh.loc[IN]
        era_idx = np.where(t_era == timestamp)[0][0]

        # get temperature and pressure at location
        dh_T = np.interp(sub_dh['caster_Z'], z, np.squeeze(Temp[...,era_idx]))
        dh_P = np.interp(sub_dh['caster_Z'], z, np.squeeze(Pres[...,era_idx]))
        if simple_refraction:
            n_0 = refractive_index_visible(central_wavelength, dh_T, dh_P)
        else:
            dh_fH= np.interp(sub_dh['caster_Z'], z, np.squeeze(fracHum))
            dh_T = kelvin2celsius(dh_T)
            CO2 = get_CO2_level(lat, timestamp, m=3, nb=3)[0]
            n_0 = refractive_index_broadband(central_wavelength,
                                             dh_T, dh_P, dh_fH, CO2=CO2)

        zn_new = refraction_angle_analytical(sub_dh['zenith'], n_0)

        if not 'zenith_refrac' in dh.columns: dh['zenith_refrac'] = None
        dh.loc[IN,lambda df: ['zenith_refrac']] = zn_new
    return dh

def update_list_with_corr_zenith_pd(dh, file_dir, file_name='conn.txt'):
    """

    Parameters
    ----------
    dh : pandas.DataFrame
        array with photohypsometric information
    file_dir : string
        directory where the file should be situated
    file_name : string, default='conn.txt'
        name of the file to export the data columns to

    Notes
    -----
    The nomenclature is as follows:

        .. code-block:: text

          * sun
           \
            \ <-- sun trace
             \
             |\ <-- caster
             | \
             |  \                    ^ Z
             |   \                   |
         ----┴----+  <-- casted      └-> {X,Y}

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          |-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └------ {X,Y}

        The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & Y
                 |
            - <--|--> +
                 |
                 +----> East & X

    The text file has at least the following columns:

        .. code-block:: text

            * conn.txt
            ├ caster_X      <- map location of the start of the shadow trace
            ├ caster_Y
            ├ casted_X
            ├ casted_Y      <- map location of the end of the shadow trace
            ├ azimuth       <- argument of the illumination direction
            ├ zenith        <- off-normal angle of the sun without atmosphere
            └ zenith_refrac <- off-normal angle of the sun with atmosphere

    See Also
    --------
    get_refraction_angle
    """
    # organize how the collumns need to be organized in the text file
    col_order = ('caster_X', 'caster_Y', 'casted_X', 'casted_Y',
                 'azimuth', 'zenith', 'zenith_refrac')
    col_frmt = ('{:+8.2f}', '{:+8.2f}', '{:+8.2f}', '{:+8.2f}',
                '{:+8.2f}', '{:+8.2f}',
                '{:+3.4f}', '{:+3.4f}', '{:+3.4f}')
    assert np.all([header in dh.columns for header in
                   col_order+('timestamp',)])
    col_order = list(col_order)
    col_order.insert(4, 'casted_X_refine')
    col_order.insert(5, 'casted_Y_refine')
    col_order = tuple(col_order)

    timestamp = dh['timestamp'].unique()
    if timestamp.size !=1:
        warnings.warn('multiple dates present in DataFrame:no output generated')
        return
    else:
        timestamp = np.datetime_as_string(timestamp[0], unit='D')

    dh_sel = dh[dh.columns.intersection(list(col_order))]

    # write to file
    f = open(os.path.join(file_dir, file_name), 'w')

    # if timestamp is give, add this to the
    print('# time: '+timestamp, file=f)

    # add header
    col_idx = dh_sel.columns.get_indexer(list(col_order))
    IN = col_idx!=-1
    print('# '+' '.join(tuple(np.array(col_order)[IN])), file=f)
    col_idx = col_idx[IN]

    # write rows of dataframe into text file
    for k in range(dh_sel.shape[0]):
        line = ''
        for idx,val in enumerate(col_idx):
            line += col_frmt[idx].format(dh_sel.iloc[k,val])
            line += ' '
        print(line[:-1], file=f) # remove last spacer
    f.close()
    return

def refraction_spherical_symmetric(zn_0, lat=None, h_0=None):
    """
    Parameters
    ----------
    zn_0, float
        observation angle, when atmosphere is not taken into account
    lat : float, unit=degrees, range=-90...+90
        latitude of point of interest
    h_0 : float, unit=meters
        initial altitude above geoid

    Returns
    -------
    zn_hat : float
        analytical refraction angle

    See Also
    --------
    get_mean_lat

    Notes
    -----
    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ↑                     ↑    /
          |                     |   /
          |-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          +----                 +------

    References
    ----------
    .. [No99] Noerdlinger, "Atmospheric refraction effects in Earth remote
              sensing" ISPRS journal of photogrammetry and remote sensing,
              vol.54, pp.360-373, 1999.
    .. [BJ94] Birch and Jones, "Correction to the updated Edlen equation for the
              refractive index of air", Metrologica, vol.31(4) pp.315-316, 1994.
    """
    if lat is None:
        n_0 = 1.0002904
    elif h_0 is None:
        rho_frac = get_simple_density_fraction(lat)
        n_0 = 1.0 + 0.0002905*rho_frac                      # eq. A8 in [1]
    else:
        rho_frac = get_density_fraction(lat, h_0)
        n_0 = 1.0 + 0.0002905*rho_frac

    zn_hat = refraction_angle_analytical(zn_0, n_0)
    return zn_hat

#def refraction_angle_layered(df, zn_0, lat, h_0, h_interval=10000): #todo
#    """
#
#    Parameters
#    ----------
#    df : dataframe, unit=µm
#        central wavelengths of the spectral bands
#    lat
#    h_0
#
#    Returns
#    -------
#
#    Notes
#    -----
#    The following parameters are used:
#
#    i : unit=degrees
#        incidence angle
#    R : unit=meters
#        mean radius of the Earth
#    H : unit=meters
#        altitude of the platform above the Earth surface
#
#    References
#    ----------
#    .. [Ya16] Yan et al, "Correction of atmospheric refraction geolocation
#              error of high resolution optical satellite pushbroom images"
#              Photogrammetric engineering & remote sensing, vol.82(6)
#              pp.427-435, 2016.
#    """
#
#    theta_0 =
#
#    i = np.arcsin(np.divide(R+H, R+h)* np.sin(zn_0))        # eq. 3 in [1]
#    r = np.arcsin(np.divide(np.sin(i),n))                   # eq. 5 in [1]
#    theta_k = np.arcsin(np.divide((R+h)*np.sin(i), R)) - r  # eq. 6 in [1]
