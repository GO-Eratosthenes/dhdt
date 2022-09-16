import numpy as np

def get_midpoint_altitude(RGI, Z, roi=None):
    """ the mid-point altitude is a for the ELA [1]

    Parameters
    ----------
    RGI : numpy.array, unit=meter
        array with labelled glaciers
    Z : numpy.array, unit=meter
        array with elevation
    roi : integer
        the RGI id of the glacier of interest

    Returns
    -------
    labels : numpy.array
        the RGI ids of the glacier of interest
    altitude : numpy.array
        mid-point altitudes of the glaciers given by "labels"

    Notes
    -----
    The following acronyms are used:

    - RGI : Randolph glacier inventory
    - ELA : equilibirum line altitude

    References
    ----------
    .. [1] Raper & Braithwaite, "Glacier volume response time and its links to
       climate and topography based on a conceptual model of glacier hypsometry"
       The cryosphere, vol.3 pp.183-194, 2009.
    """

    f = lambda x: np.quantile(x, 0.5)
    if roi is None:
        labels, altitude = get_stats_from_labelled_array(RGI, Z, f)
        labels = labels.astype(int)
    else:
        labels, altitude = np.array([roi]), f(Z[RGI==roi])
    return labels, altitude

def get_normalized_hypsometry(RGI, Z, dZ, bins=20):
    """

    Parameters
    ----------
    RGI : numpy.array, unit=meter
        array with labelled glaciers
    Z : numpy.array, unit=meter
        array with elevation
    dZ : numpy.array, unit=meter
        array with elevation change
    bins : integer, default=20
        amount of bins to use for the hypsometric estimation

    Returns
    -------
    hypsometry : np.array, size=(bins,), unit=meter
        normalized hypsometric elevation change
    count : numpy.array, size=(bins,)
        sample size at the different bins

    References
    ----------
    .. [1] McNabb et al. "Sensitivity of glacier volume estimation to DEM
       void interpolation", The cryosphere, vol.13 pp.895-910, 2019.
    .. [2] Mannerfelt et al. "Halving of Swiss glacier volume since 1931
       observed from terrestrial image photogrammetry", The crysophere
       discussions
    .. [3] Huss et al. "Future high-mountain hydrology: a new parameterization
       of glacier retreat", Hydrology and earth system sciences, vol.14
       pp.815-829, 2010
    """
    func = lambda x: np.quantile(x, 0.5)

    hypsometry = None
    for rgiid in np.unique(RGI):
        z_r, dz_r = Z[RGI==rgiid], dZ[RGI==rgiid]
        z_n = np.floor(np.divide(z_r -np.min(z_r), np.ptp(z_r)) * (bins-1))
        labels, indices, counts = np.unique(z_n, return_counts=True, return_inverse=True)
        arr_split = np.split(dz_r[indices.argsort()], counts.cumsum()[:-1])

        result = np.array(list(map(func, arr_split)))
        container, info = np.zeros(bins), np.zeros(bins)
        container[labels.astype(int)] = result
        info[labels.astype(int)] = 1
        if hypsometry is None:
            hypsometry, count = container.copy(), info.copy()
        else:
            hypsometry += container
            count += info
    # take the mean
    hypsometry /= count
    return hypsometry, count

def get_general_hypsometry(Z, dZ, interval=100., quant=.5):
    """ robust estimation of the general hypsometry

    Parameters
    ----------
    Z : numpy.array, unit=meter
        array with elevation
    dZ : numpy.array, unit=meter
        array with elevation change
    interval : {float, integer}, default=100, unit=meter
        bin interval for the hypsometry estimation
    quant : float, default=.5, range=0...1
        quantile to be estimated, when .5 is taken this corresponds to the median

    Returns
    -------
    label : numpy.array, unit=meter
        value of the interval center
    hypsometry : numpy.array, unit=meter
        group statistics within the elevation interval
    """
    assert len(set({Z.size, dZ.size})) == 1, \
        ('please provide arrays of the same size')
    assert isinstance(quant, flaot)
    assert quant<=1

    L = np.round(Z/interval)
    f =  lambda x: np.quantile(x, quant)
    label, hypsometry, counts = get_stats_from_labelled_array(L, dZ, f)
    label *= interval
    return label, hypsometry, counts

def get_stats_from_labelled_array(L,I,func): #todo is tuple: multiple functions?
    """ get properties of a labelled array

    Parameters
    ----------
    L : numpy.array, size=(m,n), {bool,integer,float}
        array with unique labels
    I : numpy.array, size=(m,n)
        data array of interest
    func : function
        group operation, that extracts information about the labeled group

    Returns
    -------
    labels : numpy.array, size=(k,1), {bool,integer,float}
        array with the given labels
    result : numpy.array, size=(k,1)
        array with the group property, as given by func

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters import threshold_otsu

    >>> image = data.coins()
    >>> thresh = threshold_otsu(image)
    >>> L = image > thresh
    >>> func = lambda x: np.quantile(x, 0.7)
    >>> label,quant = get_stats_from_labelled_array(L,image,func)
    False: 74
    True: 172
    """
    L, I = L.flatten(), I.flatten()
    labels, indices, counts = np.unique(L,
                                        return_counts=True,
                                        return_inverse=True)
    arr_split = np.split(I[indices.argsort()], counts.cumsum()[:-1])

    result = np.array(list(map(func, arr_split)))
    return labels, result, counts

def hypsometeric_void_interpolation(z,dz,Z,deg=3):
    """

    Parameters
    ----------
    z :  numpy.array, size=(k,), unit=meters
        binned elevation
    dz : numpy.array, size=(k,), unit=meters
        general elevation change of the elevation bin given by 'z'
    Z : numpy.array, size=(m,n), unit=meters
        array with elevation
    deg : integer, default=3
        order of the polynomial

    Returns
    -------
    dZ : numpy.array, size=(m,n), unit=meters
        interpolated elevation change

    References
    ----------
    .. [1] McNabb et al. "Sensitivity of glacier volume estimation to DEM
       void interpolation", The cryosphere, vol.13 pp.895-910, 2019.
    """
    f = np.polyfit1d(np.polyfit(z,dz, deg=deg))
    dZ = f(Z)
    return dZ
