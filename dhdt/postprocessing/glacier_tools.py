from .group_statistics import get_midpoint_altitude

def volume2icemass(V, Z=None, RGI=None, ELA=None, distinction='Kaeaeb'):
    """

    Parameters
    ----------
    V : numpy.array, dtype={float,integer}, size=(m,n), unit=m
        array with elevation change values
    Z : numpy.array, dtype={float,integer}, size=(m,n), unit=m
        array with elevation
    RGI : numpy.array, dtype={boolean,integer}, size=(m,n)
        array with labelled glaciers
    distinction:
        asdas
            - 'Kaeaeb' : make a distinction between accumulation and ablation
            - 'Huss' : use a generic conversion factor of 850

    Returns
    -------
    Mass : numpy.array, size=(m,n), unit=kg

    Notes
    -----
    The following acronyms are used:

    - RGI : Randolph glacier inventory
    - ELA : equilibrium line altitude

    References
    ----------
    .. [1] Kääb et al. "Contrasting patterns of early twenty-first-century
       glacier mass change in the Himalayas", Nature, vol.488, pp.495–498, 2012.
    .. [2] Huss, "Density assumptions for converting geodetic glacier volume
       change to mass change", The cryosphere, vol.7 pp.877–887, 2013.
    """

    if distinction in ('Kaeaeb', 'Kääb',):
        a_abl, a_acc = 300, 900
        if ELA is None:
            ELA = get_midpoint_altitude(RGI, Z)
    else:
        a = 850

    M = None
    return M