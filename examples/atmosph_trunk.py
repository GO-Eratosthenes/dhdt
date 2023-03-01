from scipy import stats

def explanatory_constants_at(r_i,r,
                             alpha, delta, g, R, RH, M_d, M_w, T_0, A, P_0):
    c_1 = alpha
    c_2 = np.divide(g*M_d, R)
    c_3 = np.divide(c_2, c_1)
    c_4 = delta
    pw_i = np.interp(r_i, r, RH) * ((T_0/247.1)**delta)
    c_5 = np.divide(pw_i * (1-(M_d/M_w)) * c_3, delta - c_3)
    c_6 = np.divide( np.outer((P_0 + c_5), A), T_0)
    c_7 = np.divide( np.outer(c_5, A) +
                     np.squeeze(np.tile(11.2684E-6 * pw_i, (len(A),1))).T,
                     T_0)
    c_8 = np.divide( alpha * (c_3 - 1) * c_6, T_0)
    c_9 = np.divide( alpha * (delta - 1) * c_7, T_0)
    return c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9

def get_explanatory_constants(alpha, delta, g, R, PW_0, M_d, M_w, T_0,
                              A, P_0):
    """ construct constants, as in Step 3 at pp.142 given by [1]

    Parameters
    ----------
    alpha
    delta

    Returns
    -------
    c_1 ... c_9 : floats

    References
    ----------
    .. [1] Seidelmann, "Explanatory supplement to the Astronomical almanac",
       University Science Books, pp.141-143, 1992

    """
    T_stack = np.squeeze(np.tile(T_0, (len(A),1))).T
    c_1 = alpha
    c_2 = np.divide(g*M_d, R)
    c_3 = np.divide(c_2, c_1)
    c_4 = delta
    c_5 = np.divide(PW_0 * (1-(M_d/M_w)) * c_3, delta - c_3)
    c_6 = np.divide( np.outer((P_0 + c_5), A), T_stack)
    c_7 = np.divide( np.outer(c_5, A) +
                     np.squeeze(np.tile(11.2684E-6 * PW_0, (len(A),1))).T,
                     T_stack)
    c_8 = np.divide( alpha * (c_3 - 1) * c_6, T_stack)
    c_9 = np.divide( alpha * (delta - 1) * c_7, T_stack)
    return c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9


def refraction_two_layer(lat, h_0, zn_0, z, T, P, Rh, df,
               R=8314.36, M_d=28.966, M_w=18.016, delta=18.36,
               r_e=6378120., h_t=11E3, h_s=80E3, iter_frac=16):
    """ estimate refraction through a two layered atmospheric model

    Parameters
    ----------
    lat : unit=degrees
        latitude of point of interest
    h_0 : float, unit=meters
        initial altitude above geoid
    zn_0 : unit=degrees
        initial zenith angle
    z : unit=meters
        altitude of the initial atmospheric profile
    T : unit=K
        initial temperature profile, i.e.: T(z)
    P :
        initial pressure profile, i.e.: P(z)
    Rh :
        relative humidity profile, i.e.: Rh(z)
    df : dataframe, unit=µm
        central wavelengths of the bands
    R : float, unit=J kmol-1 K-1
        universal gas constant
    M_d : float, unit=kg kmol-1
        moleculear mass of dry air
    M_w : float, unit=kg kmol-1
        moleculear mass of water
    delta :
        XXXX
    r_e : float, unit=km, default=6378120.
        radius of the Earth
    h_t : float, unit=m, default=11E3
        altitude of the troposphere
    h_s : float, unit=m, default=80E3
        altitude of the stratosphere
    iter_frac : integer, default=16
        fraction to subdivide the later at each iteration, based upon Simpson's
        rule, see also [1]

    Returns
    -------

    Notes
    -----
    A two-layered model is used, where the Strato- and Troposphere have
    different refractive behaviour. A simplified view of this can be like:

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
    .. [1] Hohenkerk et al. "Celestial reference systems" in Seidelmann,
       "Explanatory supplement to the Astronomical almanac", ch.3 pp.141-143,
       1992.
    .. [2] Rada Giacaman, "High-precision measurement of height differences from
       shadows in non-stereo imagery: new methodology and accuracy assessment",
       Remote sensing, vol.14 pp.1702, 2022.
    """
    # admin
    cwl = df['center_wavelength'].to_numpy()*1E-3
    r_e *= 1E3 # convert from kilometers to meters

    # estimate lapse-rate of the troposphere through interpolation
    # alpha : float, unit=K km-1, typically 0.0065
    Tropo = z<h_t
    alpha = stats.theilslopes(T[Tropo], z[Tropo], 0.90, method='separate')[0]
    Tropo_idx = np.max(np.where(Tropo)[0])

    # step 2
    r_h, r_t, r_s, r = h_0+r_e, h_t+r_e, h_s+r_e, z+r_e
    T_h, T_t = np.interp(h_0, z, T), np.interp(h_t, z, T)
    P_h = np.interp(h_0, z, P)

    # step 3
    PW = Rh * ((T/247.1)**delta)
    g = 9.784 * (1 - (2.6E-3*np.cos(np.deg2rad(2*lat))) - (2.8E-7 * h_0))
    A = ( 287.604 + np.divide(1.6288, cwl**2) + np.divide(0.0136, cwl**4) ) * \
        np.divide(2.7315E-8, 1013.25)

    c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9 = get_explanatory_constants(alpha,
        delta, g, R, PW, M_d, M_w, T, A, P_h)

    # step 4
    n, dndr, zn = np.zeros_like(c_6), np.zeros_like(c_6), np.zeros_like(c_6)
    n_h = 1 + c_6[0,:] - c_7[0,:]
    dndr_0 = -c_8 + c_9

    T_stack = np.squeeze(np.tile(T, (len(A),1))).T
    Tropo_st = np.squeeze(np.tile(Tropo, (len(A),1))).T

    # estimate for the Troposphere
    np.putmask(n, Tropo_st, 1 + (c_6[Tropo_st] *
                                 (np.divide(T_stack[Tropo_st], T_h)**(c_3-2))
                                 - c_7[Tropo_st]*
                                 (np.divide(T_stack[Tropo_st], T_h)**(c_4-2)))
               * np.divide(T_stack[Tropo_st], T_h)
               )
    np.putmask(dndr, Tropo_st, (-c_8[Tropo_st] *
                                ((T_stack[Tropo_st] / T_h)**(c_3-2))) +\
                               (c_9[Tropo_st]*
                                ((T_stack[Tropo_st] / T_h)**(c_4-2)))
               )
    # estimate for the Stratosphere
    n_t = n[Tropo_idx,:]
    np.putmask(n, ~Tropo_st, 1 + ((np.outer(np.ones(np.sum(~Tropo)),n_t)-1)*
                                  np.outer(np.exp(-c_2 * ((r[~Tropo]-r_t)/T_t))),
                                  np.ones(len(n_t))
                                  )
               )
    np.putmask(dndr, ~Tropo_st, (-(c_2/T_t) *
                                 (np.outer(np.ones(np.sum(~Tropo)),n_t) - 1) *
                                 np.outer(np.exp(-c_2*((r[~Tropo]-r_t)/T_t)),
                                          np.ones(len(n_t))))
               )

    # Snellius' law
    n_s = n[-1,:]
    zn_t = np.rad2deg(np.arcsin(n_h*r_h*np.sin(np.deg2rad(zn_0))/(n_t*r_t)))
    zn_s = np.rad2deg(np.arcsin(n_t*r_t*np.sin(np.deg2rad(zn_t))/(n_s*r_s)))


    e_t,e_s = Inf, Inf
    e = e_t + e_s
    r_i,n_i = r_0,n_0
    dndr = c_9-c_8

    # de = 1 # difference in iteration steps
    # while de>np.deg2rad(1E-3):

    dh_t, dh_s = np.divide(h_0-h_t, iter_frac), np.divide(h_t-h_s, iter_frac)
    # optimize Tropospheric ray through iteration
    for zn_i in np.linspace(zn_0, zn_t, iter_frac):
        for i in range(4):
            dr = ((n_i*r - n_0*r_0*np.sin(np.deg2rad(zn_0))/
                   np.sin(np.deg2rad(zn_i)))
                  /(n_i+r_i*dndr))              # eq. 3.281-5
            r_i -= dr
            T_i = np.interp(r_i, z+r_e, T)

            _,_,c_3,c_4,_,c_6,c_7,c_8,c_9 = explanatory_constants_at(r_i, z+r_e,
                 alpha, delta, g, R, Rh, M_d, M_w, T_h, A, P_h)
            n_i = 1 + \
                  (c_6 *((T_i/T_h)**(c_3-2) ) -
                   c_7 *((T_i/T_h)**(c_4-2) )
                   )*(T_i/T_h)
            dndr = - c_8*((T_i/T_0)**(c_3-2)) \
                   + c_9*((T_i/T_0)**(c_4-2))       # eq. 3.281-6
        f = (r_i*dndr)/(n_i+r_i*dndr)               # eq. 3.281-8
        e_t += f*dh

    # optimize Stratospheric ray through iteration
    n_t, r_t = np.copy(n_i), np.copy(r_i)
    c_2 = np.divide(g*M_d, R)
    for zn_i in np.linspace(zn_t, zn_s, iter_frac):
        for i in range(4):
            dr = ((n_i*r_i - n_0*r_0*np.sin(np.deg2rad(zn_0))/
                   np.sin(np.deg2rad(zn_i)))
                  /(n_i+r_i*dndr))                  # eq. 3.281-5
            r_i -= dr
            n_i = 1 + (n_t-1)*np.exp(-c_2*((r_i-r_t)/T_t))
        dndr = -(c_2/T_t) * \
               (n_t-1)*np.exp(-c_2*((r_i-r_t)/T_t)) # eq. 3.281-7
        f = (r_i*dndr)/(n_i+r_i*dndr)               # eq. 3.281-8
        e_s += f*dh

    de = np.abs((e_t + e_s) - e)
    e = e_t + e_s
    iter_frac *= 2

    return e

