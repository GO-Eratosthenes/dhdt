import numpy as np

from scipy import ndimage

from dhdt.postprocessing.group_statistics import get_midpoint_altitude, \
    get_stats_from_labelled_array

def _create_ela_grid(R, r_id, ela):
    lookup = np.empty((np.max(r_id) + 1), dtype=int)
    lookup[r_id] = np.arange(len(r_id))
    indices = lookup[R]
    ELA = ela[indices]
    np.putmask(ELA, indices==0, np.nan)
    return ELA

def volume2icemass(V, Z=None, RGI=None, ela=None, distinction='Kaeaeb'):
    """ Estimate the amount of mass change, given the volume change.

    Parameters
    ----------
    V : numpy.ndarray, dtype={float,integer}, size=(m,n), unit=m
        array with elevation change values
    Z : numpy.ndarray, dtype={float,integer}, size=(m,n), unit=m
        array with elevation
    RGI : numpy.ndarray, dtype={boolean,integer}, size=(m,n)
        array with labelled glaciers
    ela : float
        equilibrium line altitude
    distinction:
        either following [Kä12]_ or [Hu13]_
            - 'Kaeaeb' : make a distinction between accumulation and ablation
            - 'Huss' : use a generic conversion factor of 850 for glacier ice

    Returns
    -------
    M : numpy.ndarray, size=(m,n), unit=kg
        array with differences in meters water equivalent

    Notes
    -----
    The following acronyms are used:

    - RGI : Randolph glacier inventory
    - ELA : equilibrium line altitude

    References
    ----------
    .. [Kä12] Kääb et al. "Contrasting patterns of early twenty-first-century
              glacier mass change in the Himalayas", Nature, vol.488,
              pp.495–498, 2012.
    .. [Hu13] Huss, "Density assumptions for converting geodetic glacier volume
              change to mass change", The cryosphere, vol.7 pp.877–887, 2013.
    """

    if distinction in ('Kaeaeb', 'Kääb',):
        ρ_abl, ρ_acc = 300, 900
        if ela is None:
            rid, ela = get_midpoint_altitude(RGI, Z)
        # create ELA array
        ELA = _create_ela_grid(RGI, rid, ela)
        ρ = np.choose((Z>ELA).astype(int), [ρ_acc, ρ_abl])
    else: # do uniform density estimate
        ρ = 850

    M = np.multiply(ρ, V)
    return M

def mass_change2specific_glaciers(dM, RGI):

    f = lambda x: np.quantile(x, 0.5)
    labels, mb, count = get_stats_from_labelled_array(RGI, dM, f)

    IN = ~np.isnan(mb[...,0])
    labels, mb, count = labels[IN], mb[IN], count[IN]
    return labels, mb, count

def mass_changes2specific_glacier_hypsometries(dM, Z, RGI,
                                               step=None,start=None,stop=None):
    # ndimage.labeled_comprehension
    labels, mb, header = []
    return labels, mb, header