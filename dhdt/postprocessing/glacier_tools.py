import numpy as np
import morphsnakes as ms

from scipy import ndimage

from dhdt.generic.terrain_tools import terrain_curvature
from dhdt.preprocessing.shadow_filters import L0_smoothing
from dhdt.postprocessing.group_statistics import get_midpoint_altitude, \
    get_stats_from_labelled_array, get_stats_from_labelled_arrays

def update_glacier_mask(Z, R, iter=50, curv_max=2.):
    """
    refine glacier outline, via snakes

    Parameters
    ----------
    Z : {numpy.ndarray, numpy.masked.array}, size=(m,n), unit=meter
        grid with elevation values
    R : {numpy.ndarray, numpy.masked.array}, size=(m,n)
        grid with glacier labels
    iter : integer
        iteration steps, that is the maximum shrinking or growing size
    curv_max : float
        cap on curvature

    Returns
    -------
    R_new : {numpy.ndarray, numpy.masked.array}, size=(m,n)
        grid with refined glacier labels

    """
    if type(Z) in (np.ma.core.MaskedArray,):
        Z = Z.data
    if type(R) in (np.ma.core.MaskedArray,):
        maskedArray = True
        R = R.data
    else:
        maskedArray = False

    C = np.minimum(np.abs(terrain_curvature(Z)), curv_max)
    # remove block effect, but enhance edges
    C_sm = L0_smoothing(ndimage.gaussian_filter(C, sigma=3.), beta_max=1E-1)
    # refine glacier outline
    R_new = ms.morphological_chan_vese(C_sm, iter, smoothing=0, lambda1=5,
                                       init_level_set=R)

    # for multi-polygons remove the smaller ones
    for r_id in np.unique(R_new):
        if r_id == 0: continue
        L, num_poly = ndimage.label(R_new==r_id)
        if num_poly == 1: continue
        idx_poly = np.arange(1, num_poly + 1)
        siz_poly = ndimage.sum_labels(np.ones_like(L), L,
                                      index=idx_poly)
        max_poly = idx_poly[np.argmax(siz_poly)]
        OUT = np.logical_and(L != 0, L != max_poly)
        np.putmask(R_new, OUT, 0)

    if maskedArray:
        R_new = np.ma.array(R_new, mask=R_new==0)
    return R_new

def _create_ela_grid(R, r_id, ela):
    lookup = np.empty((np.max(r_id) + 1), dtype=int)
    lookup[r_id] = np.arange(len(r_id))
    if type(R) in (np.ma.core.MaskedArray,):
        np.putmask(R.data, R.mask, 0)
        indices = np.take(lookup, R.data)
    else:
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


def mass_changes2specific_glacier_hypsometries(dM, Z, RGI, interval=100,
                                               Z_start=None, Z_stop=None):
    """

    Parameters
    ----------
    dM : numpy.ndarray, size=(m,n), unit={meter, water meter equivalent}
        grid with elevation changes
    Z : numpy.ndarray, size=(m,n), unit={meter}
        elevation grid
    RGI : numpy.ndarray, size=(m,n), dtype=int
        grid with labels of the glacier numbers
    interval : float, unit=meters
        interval between elevation bins
    Z_start : float, unit=meters
        elevation beginning of the binning
    Z_stop : float, unit=meters
        elevation ending of the binning

    Returns
    -------
    Mb : numpy.ndarray, size=(k,l), dtype=float
        hypsometric elevation change
    rgi : numpy.ndarray, size=(k,), dtype=integer
        glacier numbering, in descending ordering in relation to its size
    header : numpy.ndarray, size=(l,), dtype=float, unit=meter
        elvation binnning of "Mb"
    """

    if Z_start is not None:
        Z = np.maximum(Z, Z_start)
    if Z_stop is not None:
        Z = np.minimum(Z, Z_stop)

    Z_bin = (Z // interval).astype(int)
    f = lambda x: np.quantile(x, 0.5) if x.size>0 else np.nan
    Mb, rgi, z_bin, Count = get_stats_from_labelled_arrays(RGI, Z_bin, dM, f)

    # sort array, so biggest glacier is first
    new_idx = np.flip(np.argsort(np.sum(Count, axis=1)))
    rgi = rgi[new_idx]
    Mb, Count = Mb[new_idx,:], Count[new_idx,:]


    header = z_bin.astype(float) * interval
    # np.putmask(Mb, Count==0, np.nan)
    # plt.plot(np.tile(header, (Mb.shape[0],1)).T, Mb.T);
    return Mb, rgi, header

def hypsometric_void_interpolation(z, dz, Z, deg=3):
    """

    Parameters
    ----------
    z :  numpy.array, size=(k,), unit=meters
        binned elevation
    dz : numpy.array, size=(k,), unit=meters
        general elevation change of the elevation bin given by 'z'
    Z : numpy.array, size=(m,n), unit=meters
        array with elevation
    deg : integer, {x ∈ ℕ | x ≥ 0}, default=3
        order of the polynomial

    Returns
    -------
    dZ : numpy.array, size=(m,n), unit=meters
        interpolated elevation change

    References
    ----------
    .. [Mc19] McNabb et al. "Sensitivity of glacier volume estimation to DEM
              void interpolation", The cryosphere, vol.13 pp.895-910, 2019.
    """
    if type(Z) in (np.ma.core.MaskedArray,):
        Z = Z.data
    f = np.poly1d(np.polyfit(z,dz, deg=deg))
    dZ = f(Z)
    return dZ