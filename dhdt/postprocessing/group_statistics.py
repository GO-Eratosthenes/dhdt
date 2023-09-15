import numpy as np
import pandas as pd
from scipy import ndimage

from dhdt.generic.terrain_tools import terrain_slope


def get_midpoint_altitude(RGI, Z, roi=None):
    """ the mid-point altitude is a for the ELA [RB09]_.

    Parameters
    ----------
    RGI : numpy.ndarray, unit=meter
        array with labelled glaciers
    Z : numpy.ndarray, unit=meter
        array with elevation
    roi : integer, {x ∈ ℕ | 1 ≥ x ≥ 19}
        the RGI id of the glacier of interest

    Returns
    -------
    labels : numpy.ndarray
        the RGI ids of the glacier of interest
    altitude : numpy.ndarray
        mid-point altitudes of the glaciers given by "labels"

    Notes
    -----
    The following acronyms are used:

    - RGI : Randolph glacier inventory
    - ELA : equilibirum line altitude

    References
    ----------
    .. [RB09] Raper & Braithwaite, "Glacier volume response time and its links
              to climate and topography based on a conceptual model of glacier
              hypsometry" The cryosphere, vol.3 pp.183-194, 2009.
    """

    def f(x):
        return np.quantile(x, 0.5)

    if roi is None:
        labels, altitude, _ = get_stats_from_labelled_array(RGI, Z, f)
        labels = labels.astype(int)
    else:
        labels, altitude = np.array([roi]), f(Z[RGI == roi])
    return labels, altitude


def get_hypsometric_index(RGI, Z):
    """ get the hypsometric index of glaciers based on [Ji09]_.

    Parameters
    ----------
    RGI : numpy.ndarray, size=(m,n), unit=meter
        array with labelled glaciers
    Z : numpy.ndarray, size=(m,n), unit=meter
        array with elevation

    Returns
    -------
    labels : numpy.ndarray, size=(k,)
        the RGI ids of the glacier of interest
    HI : numpy.ndarray, size=(k,)
        mid-point altitudes of the glaciers given by "labels"

    Notes
    -----
    This index can be grouped into five categories:
        - very top-heavy (HI < –1.5)
        - top-heavy (–1.5 < HI< -1.2)
        - through equidimensional (–1.2 < HI< 1.2) to
        - bottom-heavy (1.2 < HI < 1.5)
        - very bottom-heavy (H > 1.5)

    References
    ----------
    .. [Ji09] Jiskoot, et al. "Changes in Clemenceau Icefield and Chaba Group
              glaciers, Canada, related to hypsometry, tributary detachment,
              length–slope and area–aspect relations." Annals of glaciology,
              vol.50(53) pp.133-143, 2009.
    """

    f = [
        lambda x: np.nanmax(x), lambda x: np.nanmedian(x),
        lambda x: np.nanmin(x)
    ]

    labels, stats, _ = get_stats_from_labelled_array(RGI, Z, f)
    denom = stats[:, 1] - stats[:, 2]
    HI = np.divide(stats[:, 0] - stats[:, 1], denom, where=denom != 0)
    return labels, HI


def get_kurowsky_altitude(RGI, Z):
    """ get the 'Kurowski’s Schneegrenze' of glaciers based on [Ku91]_.

    Parameters
    ----------
    RGI : numpy.ndarray, size=(m,n), unit=meter
        array with labelled glaciers
    Z : numpy.ndarray, size=(m,n), unit=meter
        array with elevation

    Returns
    -------
    labels : numpy.ndarray, size=(k,)
        the RGI ids of the glacier of interest
    HI : numpy.ndarray, size=(k,)
        mid-way altitudes of the glaciers given by "labels"

    References
    ----------
    .. [Ku91] Kurowski, "Die höhe der schneegrenze mit besonderer
              berücksichtigung der Finsteraarhorn-gruppe" Pencks geographische
              abhandlungen vol.5 pp.119–160, 1891.
    """
    f = [lambda x: np.nanmax(x), lambda x: np.nanmin(x)]

    labels, stats, _ = get_stats_from_labelled_array(RGI, Z, f)
    K = np.divide(stats[:, 0] + stats[:, 1], 2)
    return labels, K


def get_normalized_hypsometry(RGI, Z, dZ, bins=20):
    """

    Parameters
    ----------
    RGI : numpy.ndarray, unit=meter
        array with labelled glaciers
    Z : numpy.ndarray, unit=meter
        array with elevation
    dZ : numpy.ndarray, unit=meter
        array with elevation change
    bins : integer, {x ∈ ℕ | x ≥ 1}, default=20
        amount of bins to use for the hypsometric estimation

    Returns
    -------
    hypsometry : np.array, size=(bins,), unit=meter
        normalized hypsometric elevation change
    count : numpy.array, size=(bins,)
        sample size at the different bins

    References
    ----------
    .. [Mc19] McNabb et al. "Sensitivity of glacier volume estimation to DEM
              void interpolation", The cryosphere, vol.13 pp.895-910, 2019.
    .. [Ma22] Mannerfelt et al. "Halving of Swiss glacier volume since 1931
              observed from terrestrial image photogrammetry", The crysophere
              vol.16(8) pp.3249-3268, 2022.
    .. [Hu10] Huss et al. "Future high-mountain hydrology: a new
              parameterization of glacier retreat", Hydrology and earth system
              sciences, vol.14 pp.815-829, 2010
    """

    def func(x):
        return np.quantile(x, 0.5)

    hypsometry = None
    for rgiid in np.unique(RGI):
        z_r, dz_r = Z[RGI == rgiid], dZ[RGI == rgiid]
        z_n = np.floor(np.divide(z_r - np.min(z_r), np.ptp(z_r)) * (bins - 1))
        labels, indices, counts = np.unique(z_n,
                                            return_counts=True,
                                            return_inverse=True)
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


def get_general_hypsometry(Z, dZ, interval=100.):
    """ robust estimation of the general hypsometry

    Parameters
    ----------
    Z : {numpy.array, pandas.Series}, unit=meter
        array with elevation
    dZ : {numpy.array, pandas.Series}, unit=meter
        array with elevation change
    interval : {float, integer}, default=100, unit=meter
        bin interval for the hypsometry estimation

    Returns
    -------
    label : numpy.array, unit=meter
        value of the interval center
    hypsometry : numpy.array, unit=meter
        group statistics within the elevation interval
    counts : numpy.ndarray
        amount of datapoints used for estimating
    """
    if type(Z) is pd.core.series.Series:
        Z = Z.to_numpy()
    if type(dZ) is pd.core.series.Series:
        dZ = dZ.to_numpy()

    assert len(set({Z.size, dZ.size})) == 1, \
        'please provide arrays of the same size'

    L = (Z // interval).astype(int)

    def f(x):
        return np.nanmedian(x) if x.size != 0 else np.nan

    label, hypsometry, counts = get_stats_from_labelled_array(L, dZ, f)
    label = np.multiply(label.astype(float), interval)
    return label, hypsometry, counts


def get_stats_from_labelled_array(L, Z, func):
    """ get properties of a labelled array

    Parameters
    ----------
    L : numpy.array, size=(m,n), {bool,integer,float}
        array with unique labels
    Z : numpy.array, size=(m,n)
        data array of interest
    func : {function, list of functions}
        group operation, that extracts information about the labeled group

    Returns
    -------
    labels : numpy.array, size=(k,1), {bool,integer,float}
        array with the given labels
    result : numpy.array, size={(k,1), (k,l)}
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
    if type(L) in (np.ma.core.MaskedArray, ):
        OUT = np.logical_or(L.mask, Z.mask)
    else:
        OUT = np.logical_or(np.isnan(L), np.isnan(Z))
    L, Z = L.data[~OUT], Z.data[~OUT]

    labels, indices, counts = np.unique(L,
                                        return_counts=True,
                                        return_inverse=True)
    arr_split = np.split(Z[indices.argsort()], counts.cumsum()[:-1])

    if isinstance(func, list):
        result = np.empty((len(arr_split), len(func)))
        for idx, f in enumerate(func):
            result[..., idx] = np.array(list(map(f, arr_split)))
    else:
        result = np.array(list(map(func, arr_split)))
    return labels, result, counts


def get_stats_from_labelled_arrays(L1, L2, Z, func):
    """ get properties of an array via a double labelling

    Parameters
    ----------
    L1,L2 : numpy.array, size=(m,n), {bool,integer,float}
        arrays with unique labels
    Z : numpy.array, size=(m,n)
        data array of interest
    func : {function, list of functions}
        group operation, that extracts information about the labeled group

    Returns
    -------
    L : numpy.ndarray, size=(k,l), float
        array with group statistics
    labels_1, labels_2 : numpy.ndarray, size={(k,),(l,)}
        array with the given labels
    C : numpy.ndarray, size={(k,1), (k,l)}
        array with the sample size, used to estimate the property given in L

    """

    def _get_mask_dat(A):
        Msk = A.mask if type(A) in (np.ma.core.MaskedArray, ) else np.isnan(A)
        Dat = A.data if type(A) in (np.ma.core.MaskedArray, ) else A
        return Dat, Msk

    L1, M1 = _get_mask_dat(L1)
    L2, M2 = _get_mask_dat(L2)
    Z, M = _get_mask_dat(Z)
    OUT = np.logical_or(M1, M2, M)
    L1, L2, Z = L1[~OUT], L2[~OUT], Z[~OUT]
    del OUT, M1, M2, M

    labels_1, indices, counts = np.unique(L1,
                                          return_counts=True,
                                          return_inverse=True)
    labels_2 = np.unique(L2)

    arr_Z_split = np.split(Z[indices.argsort()], counts.cumsum()[:-1])
    arr_L_split = np.split(L2[indices.argsort()], counts.cumsum()[:-1])

    L = np.zeros((labels_1.size, labels_2.size))
    C = np.zeros_like(L)
    for idx, Z_L1 in enumerate(arr_Z_split):
        L2_split = arr_L_split[idx]
        labels, indices, counts = np.unique(L2_split,
                                            return_counts=True,
                                            return_inverse=True)
        Z_L2 = np.split(Z_L1[indices.argsort()], counts.cumsum()[:-1])
        result = np.array(list(map(func, Z_L2)))
        L[idx, np.searchsorted(labels_2, labels)] = result
        C[idx, np.searchsorted(labels_2, labels)] = counts
    return L, labels_1, labels_2, C
