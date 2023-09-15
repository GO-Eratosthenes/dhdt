import re
import warnings

import numpy as np

from .unit_conversion import deg2arg


def zenit_angle_check(zn):
    zn = np.maximum(zn, 0.)
    zn = np.minimum(zn, 90.)
    return zn


def lat_lon_angle_check(ϕ, λ):
    """

    Parameters
    ----------
    ϕ : float, unit=degrees, range=-90...+90
        latitude
    λ : float, unit=degrees
        longitude

    Returns
    -------
    ϕ : float
        latitude
    λ : float
        longitude

    """
    ϕ = np.maximum(ϕ, -90.)
    ϕ = np.minimum(ϕ, +90.)

    λ = deg2arg(λ)
    return ϕ, λ


def is_crs_an_srs(crs):
    """ is the coordinate reference system given in degrees, or a spatial
    reference system (with a metric scale).

    Parameters
    ----------
    crs : string
        osr.SpatialReference in well known text

    Returns
    -------
    verdict : bool
        is the reference system a mapping system

    See Also
    --------
    dhdt.testing.mapping_tools.create_local_crs
    """
    if not isinstance(crs, str):
        crs = crs.ExportToWkt()
    return crs.find('"metre"') != -1


def correct_geoTransform(geoTransform):
    assert isinstance(geoTransform, tuple), 'geoTransform should be a tuple'
    assert len(geoTransform) in (
        6,
        8,
    ), 'geoTransform not of the correct size'
    if np.hypot(geoTransform[1], geoTransform[2]) != \
            np.hypot(geoTransform[4], geoTransform[5]):
        warnings.warn("pixels seem to be rectangular")
    if len(geoTransform) == 6:
        assert all(isinstance(n, float) for n in geoTransform)
    else:
        assert all(isinstance(n, float) for n in geoTransform[:-2])
        assert all(isinstance(n, int) for n in geoTransform[-2:])
        assert all(n >= 0 for n in geoTransform[-2:])
    return geoTransform


def correct_floating_parameter(a):
    # it is possible that a numpy array is given
    if type(a) in (np.ma.core.MaskedArray, np.ndarray):
        assert a.size == 1, 'please provide one parameter'
        a = a[0]
    if type(a) in (list, tuple):
        assert len(a) == 1, 'please provide one parameter'
        a = a[0]

    assert isinstance(a, (int, float)), 'please provide an integer'
    if isinstance(a, int):
        a = float(a)
    return a


def are_two_arrays_equal(A, B):
    assert type(A) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert type(B) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert A.ndim == B.ndim, ('please provide arrays of same dimension')
    assert A.shape == B.shape, ('please provide arrays of equal shape')
    return


def are_three_arrays_equal(A, B, C):
    assert type(A) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert type(B) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert type(C) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert len(set({A.shape[0], B.shape[0], C.shape[0]})) == 1, \
        ('please provide arrays of the same size')
    if np.logical_and.reduce((A.ndim > 1, B.ndim > 1, C.ndim > 1)):
        assert len(set({A.shape[1], B.shape[1], C.shape[1]})) == 1, \
            ('please provide arrays of the same size')
    assert len(set({A.ndim, B.ndim, C.ndim})) == 1, \
        ('please provide arrays of the same dimension')
    return


def check_mgrs_code(tile_code):
    """
    Validate a MGRS tile code and make it uppercase

    Parameters
    ----------
    tile_code : str
        MGRS tile code

    Returns
    -------
    tile_code : str
        validated and normalized tile code

    Notes
    -----
    The following acronyms are used:

        - MGRS : US Military Grid Reference System
    """

    if not isinstance(tile_code, str):
        raise TypeError("please provide a string")

    tile_code = tile_code.upper()

    if not bool(re.match("[0-9][0-9][A-Z][A-Z][A-Z]", tile_code)):
        raise ValueError("please provide a correct MGRS tile code")

    return tile_code
