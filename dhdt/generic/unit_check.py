import numpy as np
import warnings

from .unit_conversion import deg2arg

def zenit_angle_check(zn):
    zn = np.maximum(zn, 0.)
    zn = np.minimum(zn, 90.)
    return zn

def lat_lon_angle_check(lat,lon):
    lat = np.maximum(lat, -90.)
    lat = np.minimum(lat, +90.)

    lon = deg2arg(lon)
    return lat, lon

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
    """
    if not isinstance(crs, str): crs = crs.ExportToWkt()
    return crs.find('"degree"')==-1

def correct_geoTransform(geoTransform):
    assert isinstance(geoTransform, tuple), 'geoTransform should be a tuple'
    assert len(geoTransform) in (6,8,), 'geoTransform not of the correct size'
    if np.hypot(geoTransform[1], geoTransform[2]) != \
            np.hypot(geoTransform[4], geoTransform[5]):
        warnings.warn("pixels seem to be rectangular")
    if len(geoTransform)==6:
        assert all(isinstance(n, float) for n in geoTransform)
    else:
        assert all(isinstance(n, float) for n in geoTransform[:-2])
        assert all(isinstance(n, int) for n in geoTransform[-2:])
        assert all(n>=0 for n in geoTransform[-2:])
    return geoTransform

def are_two_arrays_equal(A, B):
    assert type(A) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert type(B) in (np.ma.core.MaskedArray, np.ndarray), \
        ('please provide an array')
    assert A.ndim==B.ndim, ('please provide arrays of same dimension')
    assert A.shape==B.shape, ('please provide arrays of equal shape')
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
    assert len(set({A.shape[1], B.shape[1], C.shape[1]})) == 1, \
         ('please provide arrays of the same size')
    assert len(set({A.ndim, B.ndim, C.ndim})) == 1, \
         ('please provide arrays of the same size')
    return