import numpy as np

from ..generic.unit_conversion import datetime2doy, \
    celsius2kelvin, kelvin2celsius

def reflective_index_visible_vapour(df, Rh, T=15):
    sigma = 1/df['center_wavelength'].to_numpy() # wavenumber
    # Saturation Properties for Steam - Temperature Table in Pascal
    TP = np.array([[0.01, 0.00061E6],
                   [5.,   0.00087E6],
                   [10.,  0.00123E6],
                   [15.,  0.00171E6],
                   [20.,  0.00234E6],
                   [25.,  0.00317E6],
                   [30.,  0.00425E6],
                   [35.,  0.00563E6],
                   [40.,  0.00739E6],
                   [45.,  0.00960E6],
                   [50.,  0.01235E6],
                   [55.,  0.01576E6],
                   [60.,  0.01995E6],
                   [65.,  0.02504E6],
                   [70.,  0.03120E6],
                   [75.,  0.03860E6],
                   [80.,  0.04741E6],
                   [85.,  0.05787E6],
                   [90.,  0.07018E6],
                   [95.,  0.08461E6],
                   [100., 0.10142E6],
                   [110., 0.14338E6],
                   [120,  0.19867E6]])

    f = Rh*np.interp(T, TP[:,0], TP[:,1])
    dn_0 = -f*(3.7345 -(0.0401*sigma))*1E-10
    return dn_0

def get_CO2_level(lat, date, m=3, nb=3):
    """
    Rough estimation of CO2, based upon measurements of global 'Station data'_

    The annual trend is based upon the 'Global trend'_ in CO2 while the seasonal
    trend is based upon relations as shown in Figure 1 in [1]. Where a latitudal
    component is present in the ampltitude, as well as, an asymmetric triangular
    wave.

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
    total_co2 : float
        estimate of spatial temporal CO2 value

    Notes
    -----
    .. _'Global trend': https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5
    .. _'Station data': https://gml.noaa.gov/dv/iadv/

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Plot the annual trend at 60ºN

    >>> t = np.arange("1950-01-01","2022-01-01",dtype="M8[Y]")
    >>> co2 = get_CO2_level(60, t)
    >>> plt.plot(t, co2)

    Plot seasonal signal
    >>> t = np.arange("2018-01-01","2022-01-01",dtype="M8[D]")
    >>> co2 = get_CO2_level(60, t)
    >>> plt.plot(t, co2)

    References
    ----------
    .. [1] Barnes et al. "Isentropic transport and the seasonal cycle amplitude
       of CO2" Journal of geophysical research: atmosphere, vol.121
       pp.8106-8124, 2016.
    """

    # global trend:
    # using a polynomial upwards trend, unofortunately
    decadal_co2 = np.poly1d(np.array([0.012786034930234986,
                                      -49.31617858270089,
                                      47857.733173381115]))
    year,doy = datetime2doy(date)
    frac_yr = doy/365.25
    co2_base = decadal_co2(year+frac_yr)

    # annual trend:
    # using an asymmetric triangular wave, with latitudal element
    b = np.zeros(nb) # estimate base
    for n in np.linspace(1,nb,nb):
        b[int(n-1)] = -np.divide(2*(-1**n)*(m**2) ,(n**2)*(m-1)*(np.pi**2)) * \
           np.sin(np.divide(n*(m-1)*np.pi,m))
    annual_co2 = 0
    for idx,para in enumerate(b):
        fb = - para * np.sin(np.divide(((idx + 1) * (frac_yr * np.pi)), 0.5))
        annual_co2 += fb

    annual_amp = 18 * np.tanh(lat / 30)
    annual_co2 *= -annual_amp

    total_co2 = co2_base + annual_co2
    return total_co2

def get_height_tropopause(lat):
    """ The tropopause can be related to its latitude [1] and a least squares
    fit is given in [2].

    Parameters
    ----------
    lat : {float, numpy.array}, unit=degrees, range=-90...+90
        latitude of interest

    Returns
    -------
    h_t : {float, numpy.array}, unit=meters
        estimate of tropopause, that is the layer between the troposphere and
        the stratosphere

    Notes
    -----
    The Strato- and Troposphere have different refractive behaviour.
    A simplified view of this can be like:

        .. code-block:: text
                              z_s
            Mesosphere       /
          +-----------------/-+  ^ 80 km    |   *       T_s  |*
          |                /  |  |          |   *            |*
          | Stratosphere  /   |  |          |  *             |*
          |              /    |  |          |  *             |*
          +--Tropopause-/-----+  |  ^ 11 km |_*_________T_t  |*________
          |            |      |  |  |       | **             | *
          |  Tropo     |      |  |  |       |    **          |  **
          |  sphere    | z_h  |  |  |       |       **  T_h  |     ****
          +------------#------+  v  v       +-----------T_0  +---------
           Earth surface                    Temperature      Pressure

    References
    ----------
    .. [1] Allen, "Astrophysical quantities", pp.121-124, 1973.
    .. [2] Noerdlinger, "Atmospheric refraction effects in Earth remote sensing"
       ISPRS journal of photogrammetry and remote sensing, vol.54, pp.360-373,
       1999.
    """
    lat = np.deg2rad(lat)
    h_t = 17786.1 - 9338.96*np.abs(lat) + 1271.91*(lat**2)  # eq. A9 in [2]
    return h_t

def get_T_sealevel(lat, method='noerdlinger'):
    """ The temperature at sea level can be related to its latitude [1] and a
    least squares fit is given in [2] and [3].

    Parameters
    ----------
    lat : {float, numpy.array}, unit=degrees, range=-90...+90
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
    .. [1] Allen, "Astrophysical quantities", pp.121-124, 1973.
    .. [2] Noerdlinger, "Atmospheric refraction effects in Earth remote sensing"
       ISPRS journal of photogrammetry and remote sensing, vol.54, pp.360-373,
       1999.
    .. [3] Yan et al, "Correction of atmospheric refraction geolocation error of
       high resolution optical satellite pushbroom images" Photogrammetric
       engineering & remote sensing, vol.82(6) pp.427-435, 2016.
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

def get_T_in_troposphere(h, t_0):
    """
    Using parameters of the first layer of the eight layer atmospheric model of
    ISO 2533:1970 to estimate the temperature at different altitudes

    Parameters
    ----------
    h : {float, numpy.array}, unit=m
        altitude above sea-level
    t_0 : float, unit=K
        temperature at the surface (reference)

    Returns
    -------
    t_h : {float, numpy.array}, unit=K
        temperature in the troposphere

    References
    ----------
    .. [1] Yan et al, "Correction of atmospheric refraction geolocation error of
       high resolution optical satellite pushbroom images" Photogrammetric
       engineering & remote sensing, vol.82(6) pp.427-435, 2016.
    """

    t_0 = kelvin2celsius(t_0)
    t_h = t_0 + np.divide((-56.5 - t_0)*h, 11019)
        # eq. 20 in [1]
    t_h = celsius2kelvin(t_h)
    return t_h

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

    References
    ----------
    .. [1] Yan et al, "Correction of atmospheric refraction geolocation error of
       high resolution optical satellite pushbroom images" Photogrammetric
       engineering & remote sensing, vol.82(6) pp.427-435, 2016.
    """
    t = kelvin2celsius(T)
    P_w = (7.38E-3*t + 0.8072)**8 - \
        1.9E-5*(1.8*t + 48) + \
        1.316E-3
    P_w *= 3386.39          # eq. 23 in [1]
    return P_w

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

    References
    ----------
    .. [1] Marti & Mauersberger, "A survey and new measurements of ice vapor
       pressure at temperatures between 170 and 250K" Geophysical research
       letters, vol.20(5) pp.363-366, 1993.
    """
    A, B = -2663.5, 12.537
    P_w = 10**(np.divide(A,T)+B)                            # eq. 1 from [1]
    return P_w

def get_density_fraction(lat, h_0, alpha=0.0065, R=8314.36, M_t=28.825):
    """

    Parameters
    ----------
    lat : unit=degrees
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
    rho_frac : float
        fraction of density in relation to density as sea-level

    Nomenclature
    ------------
    T_sl : unit=Kelvin, temperature at sea level
    T_t : unit=Kelvin, temperature at the Tropopause
    h_t : unit=meters, altitude of the Tropopause
    g : unit=m s-2, acceleration due to gravity

    References
    ----------
    .. [1] Noerdlinger, "Atmospheric refraction effects in Earth remote sensing"
       ISPRS journal of photogrammetry and remote sensing, vol.54, pp.360-373,
       1999.
    """
    T_sl = get_T_sealevel(lat)
    T_frac_0 = 1 - np.divide(h_0*alpha, T_sl)           # eq. A4 in [1]

    # get estimate of gravity
    g = 9.784 * (1 - (2.6E-3 * np.cos(np.deg2rad(2 * lat))) - (2.8E-7 * h_0))

    Gamma = np.divide(M_t*g, R*alpha) - 1               # eq. A6 in [1]
    rho_frac_t = T_frac_0**Gamma                        # eq. A5 in [1]

    # estimate parameters of the tropopause
    h_t, T_t = get_height_tropopause(lat), h_t*alpha + T_sl

    rho_frac_s = T_frac_0**Gamma * \
        np.exp(h_0-h_t) * np.divide(M_t*g, R*T_t)       # eq. A7 in [1]
    return rho_frac_t, rho_frac_s

def get_simple_density_fraction(lat):
    """ The density fraction is related to latitude through a least squares fit
    as is given in [1].

    Parameters
    ----------
    lat : {float, numpy.array}, unit=degrees, range=-90...+90
        latitude of interest

    Returns
    -------
    densFrac : {float, numpy.array}
        estimate of the density fraction

    References
    ----------
    .. [1] Noerdlinger, "Atmospheric refraction effects in Earth remote sensing"
       ISPRS journal of photogrammetry and remote sensing, vol.54, pp.360-373,
       1999.
    """
    rho_frac = 1.14412 - 0.185488*np.cos(np.deg2rad(lat))   # eq. A11 in [1]
    return rho_frac

def refractive_index_broadband_vapour(sigma):
    """

    Parameters
    ----------
    sigma : {float, numpy.array}, unit=µm-1
        vacuum wavenumber, that is the reciprocal of the vacuum wavelength

    Returns
    -------
    n_ws : {float, numpy.array}
        refractive index for standard moist air at sea level

    References
    ----------
    .. [1] Owens, "Optical refractive index of air: Dependence on pressure,
       temperature and composition", Applied optics, vol.6(1) pp.51-59, 1967.
    """
    cf = 1.022
    w_0, w_1, w_2, w_3 = 295.235, 2.6422, -0.032380, 0.004028
                                                            # eq. 13 in [1]
    n_ws = cf*(w_0 + w_2*(sigma**2)+ w_2*(sigma**4)+ w_3*(sigma**6))
    n_ws /= 1E8
    n_ws += 1
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

    Nomenclature
    ------------
    svp : unit=Pascal, saturation vapor pressure
    t : unit=Celcius, temperature

    References
    ----------
    .. [1] Ciddor, "Refractive index of air: new equations for the visible and
           near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    .. [2] Marti & Mauersberger, "A survey and new measurements of ice vapor
           pressure at temperatures between 170 and 250K" Geophysical research
           letters, vol.20(5) pp.363-366, 1993.
    """
    svp = np.zeros_like(T)
    if isinstance(T, float):
        T,p = np.asarray([T]), np.asarray([p])

    if T.size!=p.size:
        p = p*np.ones_like(T)

    A, B, C, D = 1.2378847E-5, -1.9121316E-2, 33.93711047, -6.3431645E3
    Freeze = T <= 273.15

    np.putmask(svp, Freeze, get_water_vapor_press_ice(T[Freeze]))
    np.putmask(svp, ~Freeze,
               np.exp(A*(T[~Freeze]**2) + B*T[~Freeze] + C + D/T[~Freeze]))

    t = kelvin2celsius(T) # transform to Celsius scale
    # enhancement factor of water vapor
    f = np.ones_like(t)
    if np.any(~Freeze):
        alpha, beta, gamma = 1.00062, 3.14E-8, 5.6E-7
        np.putmask(f, ~Freeze,
                   alpha + beta*p[~Freeze] + gamma*(t[~Freeze]**2))

    x_w = np.divide(f*fH*svp, p)
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

    See Also
    --------
    water_vapor_frac

    References
    ----------
    .. [1] Ciddor, "Refractive index of air: new equations for the visible and
           near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    """
    a_0, a_1, a_2 = 1.58123E-6, -2.9331E-8, 1.1043E-10
    b_0, b_1 = 5.707E-6, -2.051E-8
    c_0, c_1 = 1.9898E-4, -2.376E-6
    d, e = 1.83E-11, -0.765E-8

    Z = 1 - (p/T)*(a_0 + a_1*T + a_2*(T**2)
                   + (b_0 + b_1*T)*x_w
                   + (c_0 + c_1*T)*(x_w**2)) + ((p/T)**2) * (d + e*(x_w**2))
                                                            # eq. 12 in [1]
    return Z

def get_density_air(T,P,fH, moist=True, CO2=450, M_w=0.018015, R=8.314510):
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
    CO2 : float, unit=ppm of CO2
        parts per million of CO2 in the atmosphere
    M_w : float, unit=kg mol-1
        molar mass of water vapor
    R : float, unit=J mol-1 K-1
        universal gas constant

    Returns
    -------
    rho : float, unit=kg m3
        density of moist air

    Nomenclature
    ------------
    M_a : unit=kg mol-1, molar mass of dry air
    x_w : molecular fraction of water vapor
    Z : compressibility of moist air

    References
    ----------
    .. [1] Ciddor, "Refractive index of air: new equations for the visible and
       near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    """
    M_a = ((CO2-400)*12.011E-6 + 28.9635)*1E-3

    if moist is True:
        x_w = water_vapor_frac(P,T,fH)
    else:
        x_w = 0.

    Z = compressability_moist_air(P, T, x_w)
    rho = np.divide(P*M_a, Z*R*T)*(1 - x_w*(1-np.divide(M_w, M_a)) )
                                                            # eq. 4 in [1]
    return rho

def ciddor_eq5(rho_a, rho_w, n_axs, n_ws, rho_axs, rho_ws):
    """ estimating the refractive index through summation

    Parameters
    ----------
    rho_a : {numpy.array, float}, unit=kg m-3
        density of dry air
    rho_w : (numpy.array, float}, unit=kg m-3
    n_axs : numpy.array, size=(k,)
        refractive index for dry air at sea level at a specific wavelength
    n_ws : numpy.array, size=(k,)
        refractive index for standard moist air at sea level for a specific
        wavelength
    rho_axs : float, unit=kg m-3
        density of standard dry air at sea level
    rho_ws : float, unit=kg m-3
        density of standard moist air at sea level

    Returns
    -------
    n_prop : float
        refractive index

    References
    ----------
    .. [1] Owens, "Optical refractive index of air: Dependence on pressure,
       temperature and composition", Applied optics, vol.6(1) pp.51-59, 1967.
    """

    n_prop = 1 + \
             np.multiply.outer(np.divide(rho_a, rho_axs),(n_axs-1)) + \
             np.multiply.outer(np.divide(rho_w, rho_ws), (n_ws-1))        # eq. 4 in [1]

    return n_prop

def ciddor_eq6(rho_a, rho_w, n_axs, n_ws, rho_axs, rho_ws):
    """ estimating the refractive index through the Lorenz-Lorentz relation

    Parameters
    ----------
    rho_a : float, unit=kg m-3
        density of dry air
    rho_w : float, unit=kg m-3
    n_axs : float
        refractive index for dry air at sea level
    n_ws : float
        refractive index for standard moist air at sea level
    rho_axs : float, unit=kg m-3
        density of standard dry air at sea level
    rho_ws : float, unit=kg m-3
        density of standard moist air at sea level

    Returns
    -------
    n_LL : float
        refractive index

    References
    ----------
    .. [1] Owens, "Optical refractive index of air: Dependence on pressure,
       temperature and composition", Applied optics, vol.6(1) pp.51-59, 1967.
    """
    L_a = np.divide( (n_axs**2) -1, (n_axs**2) +2)
    L_w = np.divide( (n_ws**2) - 1, (n_ws**2) + 2)

    L = np.multiply.outer( np.divide(rho_a, rho_axs), L_a) + \
        np.multiply.outer( np.divide(rho_w, rho_ws), L_w)

    n_LL = np.sqrt(np.divide( (1 + 2*L), (1 - L)))              # eq. 1 in [1]
    return n_LL

def refractive_index_visible(df, T):
    """

    Parameters
    ----------
    df : dataframe, unit=µm
        central wavelengths of the spectral bands
    T : float, unit=Celcius
        temperature at location

    Returns
    -------
    n_0 : {float, numpy.array}
        index of refraction

    References
    ----------
    .. [1] Birch and Jones, "Correction to the updated Edlen equation for the
       refractive index of air", Metrologica, vol.31(4) pp.315-316, 1994.
    """
    sigma = 1/df['center_wavelength'].to_numpy() # (vacuum) wavenumber

    n_0 = 1 + \
        np.divide(1, 1+(0.003661*T)) * \
          (1.05442E-8 * (1 + ((4053*(0.601 - (0.00972*T)))/4E6)) *
           (8342.54 +
            (np.divide(15998, 38.9-(sigma**2)) +
             (np.divide(2406147, 130-(sigma**2))))
            ))
    return n_0

def refractive_index_broadband(df, T_0, P_0, fH_0, h_0, CO2=450.,
                               T_a=15., P_a=101325.,
                               T_w=20., P_w=1333.,
                               LorentzLorenz=False):
    """ Estimate the refractive index at the surface

    Parameters
    ----------
    df : dataframe, unit=µm
        central wavelengths of the spectral bands
    CO2 : float, unit=ppm of CO2, default=450.
        parts per million of CO2
    T_0 : float, unit=Celsius
        temperature at location
    P_0 : float, unit=Pascal
        the total pressure in Pascal
    fH_0 : float, range=0...1
        fractional humidity
    h_0 : float, unit=meters
        altitude above sea level at location
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
    - 78.09% nitrogen (N2)
    - 20.95% oxygen (O2)
    -  0.93% argon (Ar)
    -  0.03% carbon dioxide (C02)

    References
    ----------
    .. [1] Ciddor, "Refractive index of air: new equations for the visible and
       near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    .. [2] Owens, "Optical refractive index of air: Dependence on pressure,
       temperature and composition", Applied optics, vol.6(1) pp.51-59, 1967.
    """
    # convert to correct unit
    T_0 = celsius2kelvin(T_0)
    T_a, T_w = celsius2kelvin(T_a), celsius2kelvin(T_w)
    sigma = 1/df['center_wavelength'].to_numpy() # wave number

    # refractivities of dry air,
    # amended for changes in temperature and CO2 content
    k_0, k_1, k_2, k_3 = 238.0185, 5792105., 57.362, 167917.

    n_as = 1 + \
           (np.divide(k_1, k_2 - (sigma ** 2)) +
           np.divide(k_1, k_2 - (sigma ** 2)) )*1E-8            # eq.1 in [1]

    n_axs = 1 + ((n_as - 1) * (1 + .534E-6 * (CO2 - 450)))      # eq.2 in [1]
    n_ws = refractive_index_broadband_vapour(sigma)             # eq.3 in [1]

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
    .. [1] Noerdlinger, "Atmospheric refraction effects in Earth remote sensing"
       ISPRS journal of photogrammetry and remote sensing, vol.54, pp.360-373,
       1999.
    """
    zn = np.rad2deg(np.arcsin(np.sin(np.deg2rad(zn_0))/n_0))
                                                                # eq. 11 in [1]
    return zn

def refraction_spherical_symmetric(zn_0, lat=None, h_0=None):
    """

    Returns
    -------

    Notes
    -----
    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          |-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          +----                 +------

    References
    ----------
    .. [1] Noerdlinger, "Atmospheric refraction effects in Earth remote sensing"
       ISPRS journal of photogrammetry and remote sensing, vol.54, pp.360-373,
       1999.
    .. [2] Birch and Jones, "Correction to the updated Edlen equation for the
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
#    Nomenclature
#    ------------
#    i : unit=degrees
#        incidence angle
#    R : unit=meters
#        mean radius of the Earth
#    H : unit=meters
#        altitude of the platform above the Earth surface
#
#    References
#    ----------
#    .. [1] Yan et al, "Correction of atmospheric refraction geolocation error of
#       high resolution optical satellite pushbroom images" Photogrammetric
#       engineering & remote sensing, vol.82(6) pp.427-435, 2016.
#    """
#
#    theta_0 =
#
#    i = np.arcsin(np.divide(R+H, R+h)* np.sin(zn_0))        # eq. 3 in [1]
#    r = np.arcsin(np.divide(np.sin(i),n))                   # eq. 5 in [1]
#    theta_k = np.arcsin(np.divide((R+h)*np.sin(i), R)) - r  # eq. 6 in [1]
