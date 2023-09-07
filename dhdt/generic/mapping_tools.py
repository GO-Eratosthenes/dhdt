import os
import warnings

import numpy as np

# geospatial libaries
from osgeo import ogr, osr

# raster/image libraries
from scipy import ndimage

from dhdt.generic.attitude_tools import rot_mat
from dhdt.generic.handler_im import get_grad_filters
from dhdt.generic.unit_check import correct_geoTransform, \
    are_two_arrays_equal, correct_floating_parameter, is_crs_an_srs

def cart2pol(x, y):
    """ transform Cartesian coordinate(s) to polar coordinate(s)

    Parameters
    ----------
    x,y : {float, numpy.ndarray}
        Cartesian coordinates

    Returns
    -------
    ρ : {float, numpy.ndarray}
        radius of the vector
    φ : {float, numpy.ndarray}, unit=radians
        argument of the vector

    See Also
    --------
    pol2xyz : equivalent function, but for three dimensions
    cart2pol : inverse function
    """
    if type(x) in (np.ma.core.MaskedArray, np.ndarray):
        are_two_arrays_equal(x, y)
    else: # if only a float is given
        x,y = correct_floating_parameter(x), correct_floating_parameter(y)

    ρ = np.hypot(x, y)
    φ = np.arctan2(y, x)
    return ρ, φ

def pol2cart(ρ, φ):
    """ transform polar coordinate(s) to Cartesian coordinate(s)

    Parameters
    ----------
    ρ : {float, numpy.ndarray}
        radius of the vector
    φ : {float, numpy.ndarray}, unit=radians
        argument of the vector

    Returns
    -------
    x,y : {float, numpy.ndarray}
        cartesian coordinates

    See Also
    --------
    pol2xyz : equivalent function, but for three dimensions
    cart2pol : inverse function
    """
    if type(ρ) in (np.ma.core.MaskedArray, np.ndarray):
        are_two_arrays_equal(ρ, φ)
    else: # if only a float is given
        ρ,φ = correct_floating_parameter(ρ), correct_floating_parameter(φ)

    x = ρ * np.cos(φ)
    y = ρ * np.sin(φ)
    return x, y

def pol2xyz(Az, Zn):
    """ transform from angles to unit vector in 3D

    Parameters
    ----------
    Az : numpy.ndarray, size=(m,n), unit=degrees
        array with azimuth angles
    Zn : numpy.ndarray, size=(m,n), unit=degrees
        array with zenith angles

    Returns
    -------
    X,Y : numpy.ndarray, size=(m,n), unit=degrees
        array with planar components
    Z : numpy.ndarray, size=(m,n), unit=degrees
        array with height component

    See Also
    --------
    pol2cart : equivalent function, but for two dimensions

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x

    The angles related to the suns' heading are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └--┴---

    """
    Az, Zn = are_two_arrays_equal(Az, Zn)
    X = np.sin(np.deg2rad(Az))*np.sin(np.deg2rad(Zn))
    Y = np.cos(np.deg2rad(Az))*np.sin(np.deg2rad(Zn))
    Z = np.cos(np.deg2rad(Zn))
    return X, Y, Z

def pix2map(geoTransform, i, j):
    """ transform local image coordinates to map coordinates

    Parameters
    ----------
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    i : np.array, ndim={1,2,3}, dtype=float
        row coordinate(s) in local image space
    j : np.array, ndim={1,2,3}, dtype=float
        column coordinate(s) in local image space

    Returns
    -------
    x : np.array, size=(m), ndim={1,2,3}, dtype=float
        horizontal map coordinate.
    y : np.array, size=(m), ndim={1,2,3}, dtype=float
        vertical map coordinate.

    See Also
    --------
    map2pix

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    geoTransform = correct_geoTransform(geoTransform)
    if type(i) in (np.ma.core.MaskedArray, np.ndarray):
        are_two_arrays_equal(i, j)
    else: # if only a float is given
        i,j = correct_floating_parameter(i), correct_floating_parameter(j)

    x = geoTransform[0] + \
        np.multiply(geoTransform[1], j) + np.multiply(geoTransform[2], i)

    y = geoTransform[3] + \
        np.multiply(geoTransform[4], j) + np.multiply(geoTransform[5], i)

    # # offset the center of the pixel
    # x += geoTransform[1] / 2.0
    # y += geoTransform[5] / 2.0
    return x, y

def map2pix(geoTransform, x, y):
    """ transform map coordinates to local image coordinates

    Parameters
    ----------
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    x : np.array, size=(m), ndim={1,2,3}, dtype=float
        horizontal map coordinate.
    y : np.array, size=(m), ndim={1,2,3}, dtype=float
        vertical map coordinate.

    Returns
    -------
    i : np.array, ndim={1,2,3}, dtype=float
        row coordinate(s) in local image space
    j : np.array, ndim={1,2,3}, dtype=float
        column coordinate(s) in local image space

    See Also
    --------
    pix2map

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    geoTransform = correct_geoTransform(geoTransform)
    if type(x) in (np.ma.core.MaskedArray, np.ndarray):
        are_two_arrays_equal(x, y)
    else: # if only a float is given
        x,y = correct_floating_parameter(x), correct_floating_parameter(y)

    A = np.array(geoTransform[:-2]).reshape(2, 3)[:, 1:]
    A_inv = np.linalg.inv(A)
    # # offset the center of the pixel
    # x -= geoTransform[1] / 2.0
    # y -= geoTransform[5] / 2.0
    # ^- this messes-up python with its pointers....
    x_loc = x - geoTransform[0]
    y_loc = y - geoTransform[3]

    j = np.multiply(x_loc, A_inv[0,0]) + np.multiply(y_loc, A_inv[0,1])
    i = np.multiply(x_loc, A_inv[1,0]) + np.multiply(y_loc, A_inv[1,1])
    return i,j

def vel2pix(geoTransform, dx, dy):
    """ transform map displacements to local image displacements

    Parameters
    ----------
    geoTransform : tuple, size=(6,1)
        georeference transform of an image.
    dx : np.array, size=(m), ndim={1,2,3}, dtype=float
        horizontal map displacement.
    dy : np.array, size=(m), ndim={1,2,3}, dtype=float
        vertical map displacement.

    Returns
    -------
    di : np.array, ndim={1,2,3}, dtype=float
        row coordinate(s) in local image space
    dj : np.array, ndim={1,2,3}, dtype=float
        column coordinate(s) in local image space

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    geoTransform = correct_geoTransform(geoTransform)
    if type(dx) in (np.ma.core.MaskedArray, np.ndarray):
        are_two_arrays_equal(dx, dy)
    else:
        dx,dy = correct_floating_parameter(dx), correct_floating_parameter(dy)

    if geoTransform[2] == 0:
        dj = dx / geoTransform[1]
    else:
        dj = (dx / geoTransform[1] + dy / geoTransform[2])

    if geoTransform[4] == 0:
        di = dy / geoTransform[5]
    else:
        di = (dx / geoTransform[4] + dy / geoTransform[5])

    return di, dj

def ecef2llh(xyz):
    """ transform 3D cartesian Earth Centered Earth fixed coordinates, to
    spherical angles and height above the ellipsoid

    Parameters
    ----------
    xyz : np.array, size=(m,3), unit=meter
        np.array with 3D coordinates, in WGS84. In the following form:
        [[x, y, z], [x, y, z], ... ]

    Returns
    -------
    llh : np.array, size=(m,2), unit=(deg,deg,meter)
        np.array with angles and height. In the following form:
        [[lat, lon, height], [lat, lon, height], ... ]
    """

    ecefSpatialRef = osr.SpatialReference()
    ecefSpatialRef.ImportFromEPSG(4978)

    llhSpatialRef = osr.SpatialReference()
    llhSpatialRef.ImportFromEPSG(4979)

    coordTrans = osr.CoordinateTransformation(ecefSpatialRef, llhSpatialRef)
    llh = coordTrans.TransformPoints(list(xyz))
    llh = np.stack(llh, axis=0)
    return llh

def get_utm_zone_simple(λ):
    return ((np.floor((λ + 180) / 6) % 60) + 1).astype(int)

def ll2map(ll, spatialRef):
    """ transforms angles to map coordinates (that is 2D) in a projection frame

    Parameters
    ----------
    llh : np.array, size=(m,2), unit=(deg,deg)
        np.array with spherical coordinates. In the following form:
        [[lat, lon], [lat, lon], ... ]
    spatialRef : osgeo.osr.SpatialReference
        target projection system

    Returns
    -------
    xyz : np.array, size=(m,2), unit=meter
        np.array with 2D coordinates. In the following form:
        [[x, y], [x, y], ... ]

    Examples
    --------
    Get the Universal Transverse Mercator (UTM) cooridnates from spherical
    coordinates:
    >>> import numpy as np
    >>> from osgeo import osr
    >>> proj = osr.SpatialReference()
    >>> proj.SetWellKnownGeogCS('WGS84')
    >>> lat, lon = 52.09006426183974, 5.173794246145571# Utrecht University
    >>> NH = True if lat>0 else False
    >>> proj.SetUTM(32, True)

    >>> xy = ll2map(np.array([[lat, lon]]), proj)
    >>> xy
    array([[ 237904.03625329, 5777964.65056734,       0.        ]])
    """
    if isinstance(spatialRef, str):
        spatialStr = spatialRef
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromWkt(spatialStr)
    llSpatialRef = osr.SpatialReference()
    llSpatialRef.ImportFromEPSG(4326)

    coordTrans = osr.CoordinateTransformation(llSpatialRef, spatialRef)
    xy = coordTrans.TransformPoints(list(ll))
    xy = np.stack(xy, axis=0)
    return xy

def map2ll(xy, spatialRef):
    """ transforms map coordinates (that is 2D) in a projection frame to angles

    Parameters
    ----------
    xy : np.array, size=(m,2), unit=meter
        np.array with 2D coordinates. In the following form:
        [[x, y], [x, y], ... ]
    spatialRef : osgeo.osr.SpatialReference
        target projection system

    Returns
    -------
    ll : np.array, size=(m,2), unit=(deg,deg)
        np.array with spherical coordinates. In the following form:
        [[lat, lon], [lat, lon], ... ]
    """
    if isinstance(spatialRef, str):
        spatialStr = spatialRef
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromWkt(spatialStr)
    llSpatialRef = osr.SpatialReference()
    llSpatialRef.ImportFromEPSG(4326)

    coordTrans = osr.CoordinateTransformation(spatialRef, llSpatialRef)
    ll = coordTrans.TransformPoints(list(xy))
    ll = np.stack(ll, axis=0)
    return ll[:,:-1]

def ecef2map(xyz, spatialRef):
    """ transform 3D cartesian Earth Centered Earth fixed coordinates, to
    map coordinates (that is 2D) in a projection frame

    Parameters
    ----------
    xyz : np.array, size=(m,3), float
        np.array with 3D coordinates, in WGS84. In the following form:
        [[x, y, z], [x, y, z], ... ]
    spatialRef : osgeo.osr.SpatialReference
        target projection

    Returns
    -------
    xyz : np.array, size=(m,2), float
        np.array with planar coordinates, within a given projection frame
    """
    if isinstance(spatialRef, str):
        spatialStr = spatialRef
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromWkt(spatialStr)

    llh = ecef2llh(xyz) # get spherical coordinates and height
    xy = ll2map(llh[:, :-1], spatialRef)
    return xy

def haversine(Δlat, Δlon, lat, radius=None):
    """ calculate the distance along a great-circle

    Parameters
    ----------
    Δlat, Δlon : float, unit=degrees
        step size or angle difference
    lat : float, unit=degrees
        (mean) latitude where angles are situated
    radius : float, unit=meters
        estimate of the radius of the earth
    """
    Δlat, Δlon, lat = np.deg2rad(Δlat), np.deg2rad(Δlon), np.deg2rad(lat)

    if radius is None:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        a,b = wgs84.GetSemiMajor(), wgs84.GetSemiMinor()
        radius = np.sqrt(np.divide((a**2*np.cos(lat))**2 +
                                   (b**2*np.sin(lat))**2,
                                   (a*np.cos(lat))**2 + (b*np.sin(lat))**2)
                         )

    a = np.sin(Δlat / 2) ** 2 + \
        np.cos(lat) * np.cos(lat+Δlat) * np.sin(Δlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # great circle distance
    d = radius * c
    return d

def rot_covar(V, R):
    V_r = R @ V @ np.transpose(R)
    return V_r

def covar2err_ellipse(sigma_1, sigma_2, ρ):
    """ parameter transform from co-variance matrix to standard error-ellipse
    For more info see [Al22]_

    Parameters
    ----------
    sigma_1, sigma_2 : float
        variance along each axis
    ρ : float
        dependency between axis

    Returns
    -------
    lambda_2, lambda_2 : float
        variance along major and minor axis
    θ : float, unit=degrees
        orientation of the standard ellipse

    References
    ----------
    .. [Al22] Altena et al. "Correlation dispersion as a measure to better
              estimate uncertainty of remotely sensed glacier displacements",
              The Crysophere, vol.16(6) pp.2285–2300, 2022.
    """
    # intermediate variables
    s1s2_min = sigma_1**2 - sigma_2**2
    s1s2_plu = sigma_1**2 + sigma_2**2
    s1s2r = sigma_1 * sigma_2 * ρ
    lamb = np.sqrt(np.divide(s1s2_min**2,4) + s1s2r)
    lambda_1 = s1s2_plu/2 + lamb                            # eq.5 in [Al22]
    lambda_2 = s1s2_plu/2 - lamb
    θ = np.rad2deg(np.arctan2(2*s1s2r, s1s2_min) / 2)   # eq.6 in [Al22]
    return lambda_1, lambda_2, θ

def rotate_variance(θ, qii, qjj, ρ):
    qii_r, qjj_r = np.zeros_like(qii), np.zeros_like(qjj)
    for iy, ix in np.ndindex(θ.shape):
        if ~np.isnan(θ[iy,ix]) and ~np.isnan(ρ[iy,ix]):
            # construct co-variance matrix
            try:
                qij = ρ[iy,ix]*np.sqrt(qii[iy,ix])*np.sqrt(qjj[iy,ix])
            except:
                breakpoint
            V = np.array([[qjj[iy,ix], qij],
                          [qij, qii[iy,ix]]])
            R = rot_mat(θ[iy, ix])
            V_r = rot_covar(V, R)
            qii_r[iy, ix], qjj_r[iy, ix] = V_r[1][1], V_r[0][0]
        else:
            qii_r[iy,ix], qjj_r[iy,ix] = np.nan, np.nan
    return qii_r, qjj_r

def cast_orientation(I, Az, indexing='ij'):
    """ emphasises intensities within a certain direction, following [Fr91]_.

    Parameters
    ----------
    I : np.array, size=(m,n)
        array with intensity values
    Az : {float,np.array}, size{(m,n),1}, unit=degrees
        band of azimuth values or single azimuth angle.
    indexing : {‘xy’, ‘ij’}
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates

    Returns
    -------
    Ican : np.array, size=(m,n)
        array with intensity variation in the preferred direction.

    See Also
    --------
    .handler_im.get_grad_filters : collection of general gradient filters
    .handler_im.rotated_sobel : generate rotated gradient filter

    Notes
    -----
        The angle(s) are declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x

        Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    References
    ----------
    .. [Fr91] Freeman et al. "The design and use of steerable filters." IEEE
              transactions on pattern analysis and machine intelligence
              vol.13(9) pp.891-906, 1991.
    """
    assert type(I)==np.ndarray, ("please provide an array")

    fx,fy = get_grad_filters(ftype='sobel', tsize=3, order=1)

    Idx = ndimage.convolve(I, fx)  # steerable filters
    Idy = ndimage.convolve(I, fy)

    if indexing=='ij':
        Ican = (np.multiply(np.cos(np.radians(Az)), Idy)
                - np.multiply(np.sin(np.radians(Az)), Idx))
    else:
        Ican = (np.multiply(np.cos(np.radians(Az)), Idy)
                    + np.multiply(np.sin(np.radians(Az)), Idx))
    return Ican

def estimate_geoTransform(I,J,X,Y, samp=10):
    m,n = I.shape[:2]
    y = np.vstack([X[::samp,::samp].reshape(-1,1),
                   Y[::samp,::samp].reshape(-1,1)])
    # design matrix
    A_sub = np.hstack([I[::samp,::samp].reshape(-1,1),
                       J[::samp,::samp].reshape(-1,1)])
    A_sub = np.pad(A_sub, ((0,0),(1,0)), constant_values=1.)
    A = np.kron(np.eye(2), A_sub)

    x_hat = np.linalg.lstsq(A, y, rcond=None)[0]
    geoTransform = tuple(np.squeeze(x_hat))+(m,n)
    return geoTransform

def ref_trans(geoTransform, dI, dJ):
    """ translate reference transform

    Parameters
    ----------
    geoTransform : tuple, size=(8,)
        georeference transform of an image.
    dI : dtype={float,integer}
        translation in rows.
    dJ : dtype={float,integer}
        translation in collumns.

    Returns
    -------
    newTransform : tuple, size=(8,)
        georeference transform of an image with shifted origin.

    See Also
    --------
    ref_scale

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |
    """
    geoTransform = correct_geoTransform(geoTransform)
    dI, dJ = correct_floating_parameter(dI), correct_floating_parameter(dJ)

    newTransform = (geoTransform[0]+ dJ*geoTransform[1] + dI*geoTransform[2],
                    geoTransform[1], geoTransform[2],
                    geoTransform[3]+ dJ*geoTransform[4] + dI*geoTransform[5],
                    geoTransform[4], geoTransform[5])
    if len(geoTransform) == 8: # also include info of image extent
        newTransform = newTransform + geoTransform[-2:]
    return newTransform

def ref_scale(geoTransform, scaling):
    """
    scale image reference transform

    Parameters
    ----------
    Transform : tuple, size=(1,6)
        georeference transform of an image.
    scaling : float
        isotropic scaling in both rows and collumns.

    Returns
    -------
    newTransform : tuple, size=(1,6)
        georeference transform of an image with different pixel spacing.

    See Also
    --------
    ref_trans, ref_update, ref_rotate
    """
    geoTransform = correct_geoTransform(geoTransform)
    if len(geoTransform)==8: # sometimes the image dimensions are also included
        geoTransform = geoTransform[:-2]
        # these will not be included in the newTransform, since the pixel
        # size has changed, so this will likely be due to the generation of
        # a grid based on a group of pixels or some sort of kernel

    # not using center of pixel
    A = np.asarray(geoTransform).reshape((2,3)).T
    A[0,:] += np.diag(A[1:,:]*scaling/2)
    A[1:,:] = A[1:,:]*float(scaling)
    newTransform = tuple(map(tuple, np.transpose(A).reshape((1,6))))

    return newTransform[0]

def ref_rotate(geoTransform, θ):
    """
    rotate image reference transform

    Parameters
    ----------
    Transform : tuple, size=(1,6)
        georeference transform of an image.
    scaling : float
        isotropic scaling in both rows and collumns.

    Returns
    -------
    newTransform : tuple, size=(1,6)
        georeference transform of an image with different pixel spacing.

    See Also
    --------
    dhdt.generic.mapping_tools.ref_trans
    dhdt.generic.mapping_tools.ref_update
    dhdt.generic.mapping_tools.ref_scale
    dhdt.generic.attitude_tools.rot_mat
    """
    geoTransform = correct_geoTransform(geoTransform)
    m,n = geoTransform[-2:]
    A = np.asarray(geoTransform[:-2]).reshape((2, 3)).T
    A[1:,:] = A[1:,:]@rot_mat(θ)

    newTransform = tuple(float(x) for x in
                         np.squeeze(np.transpose(A).ravel())) + tuple((m,n))
    return newTransform

def ref_update(geoTransform, rows, cols):
    geoTransform = correct_geoTransform(geoTransform)
    rows, cols = int(rows), int(cols)
    if len(geoTransform)==6:
        geoTransform = geoTransform + (rows, cols,)
    elif len(geoTransform)==8: # replace last elements
        gt_list = list(geoTransform)
        gt_list[-2:] = [rows, cols]
        geoTransform = tuple(gt_list)
    return geoTransform

def aff_trans_template_coord(A, t_radius, fourier=False):
    if not fourier:
        x = np.arange(-t_radius, t_radius+1)
        y = np.arange(-t_radius, t_radius+1)
    else:
        x = np.linspace(-t_radius+.5, t_radius-.5, 2*t_radius)
        y = np.linspace(-t_radius+.5, t_radius-.5, 2*t_radius)

    X = np.repeat(x[np.newaxis, :], y.shape[0], axis=0)
    Y = np.repeat(y[:, np.newaxis], x.shape[0], axis=1)

    X_aff = X * A[0, 0] + Y * A[0, 1]
    Y_aff = X * A[1, 0] + Y * A[1, 1]
    return X_aff, Y_aff

def rot_trans_template_coord(θ, t_radius):
    x = np.arange(-t_radius, t_radius + 1)
    y = np.arange(-t_radius, t_radius + 1)

    X = np.repeat(x[np.newaxis, :], y.shape[0], axis=0)
    Y = np.repeat(y[:, np.newaxis], x.shape[0], axis=1)

    θ = np.deg2rad(θ)
    X_r = +np.cos(θ) * X + np.sin(θ) * Y
    Y_r = -np.sin(θ) * X + np.cos(θ) * Y
    return X_r, Y_r

def pix_centers(geoTransform, rows=None, cols=None, make_grid=True):
    """ provide the pixel coordinate from the axis, or the whole grid

    Parameters
    ----------
    geoTransform : tuple, size={(6,), (8,)}
        georeference transform of an image.
    rows : integer, {x ∈ ℕ | x ≥ 0}, default=None
        amount of rows in an image.
    cols : integer, {x ∈ ℕ | x ≥ 0}, default=None
        amount of collumns in an image.
    make_grid : bool, optional
        Should a grid be made. The default is True.

    Returns
    -------
    X : np.array, dtype=float
         * "make_grid" : size=(m,n)
         otherwise : size=(1,n)
    Y : np.array, dtype=float
         * "make_grid" : size=(m,n)
         otherwise : size=(m,1)

    See Also
    --------
    map2pix, pix2map

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       j               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | i         map         |
          based      v           based       |

    """
    geoTransform = correct_geoTransform(geoTransform)
    if rows is None:
        assert len(geoTransform)==8, ('please provide the dimensions of the ' +
                                      'imagery, or have this included in the ' +
                                      'geoTransform.')
        rows,cols = int(geoTransform[-2]), int(geoTransform[-1])
    i,j = np.linspace(0, rows-1, rows), np.linspace(0, cols-1, cols)

    if make_grid:
        J,I = np.meshgrid(j, i)
        X,Y = pix2map(geoTransform, I, J)
        return X, Y
    else:
        x,y_dummy = pix2map(geoTransform, np.repeat(i[0], len(j)), j)
        x_dummy,y = pix2map(geoTransform, i, np.repeat(j[0], len(i)))
        return x, y

def bilinear_interp_excluding_nodat(I, i_I, j_I, noData=-9999):
    """ do simple bi-linear interpolation, taking care of nodata

    Parameters
    ----------
    I : np.array, size=(m,n), ndim={2,3}
        data array.
    i_I : np.array, size=(k,l), ndim=2
        horizontal locations, within local image frame.
    j_I : np.array, size=(k,l), ndim=2
        vertical locations, within local image frame.
    noData : TYPE, optional
        DESCRIPTION. The default is -9999.

    Returns
    -------
    I_new : np.array, size=(k,l), ndim=2, dtype={float,complex}
        interpolated values.

    See Also
    --------
    .handler_im.bilinear_interpolation

    Notes
    -----
        Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    assert type(I)==np.ndarray, ("please provide an array")
    are_two_arrays_equal(i_I, j_I)
    m, n = I.shape[0:2]

    i_prio, i_post = np.floor(i_I).astype(int), np.ceil(i_I).astype(int)
    j_prio,j_post = np.floor(j_I).astype(int), np.ceil(j_I).astype(int)

    wi,wj = np.remainder(i_I,1), np.remainder(j_I,1)

    # make resistant to locations outside the bounds of the array
    I_1, I_2 = np.ones_like(i_prio)*noData, np.ones_like(i_prio)*noData
    I_3, I_4 = np.ones_like(i_prio)*noData, np.ones_like(i_prio)*noData

    IN = np.logical_and.reduce(((i_prio >= 0), (i_post <= (m-1)),
                                (j_prio >= 0), (j_post <= (n-1))))

    I_1[IN],I_2[IN] = I[i_prio[IN],j_prio[IN]], I[i_prio[IN],j_post[IN]]
    I_3[IN],I_4[IN] = I[i_post[IN],j_post[IN]], I[i_post[IN],j_prio[IN]]

    no_dat = np.any(np.vstack((I_1,I_2,I_3,I_4))==noData, axis=0)

    DEM_new = wj*(wi*I_1 + (1-wi)*I_4) + (1-wj)*(wi*I_2 + (1-wi)*I_3)
    DEM_new[no_dat] = noData
    return DEM_new

def bbox_boolean(img):
    """ get image coordinates of maximum bounding box extent of True values
    in a boolean matrix

    Parameters
    ----------
    img : np.array, size=(m,n), dtype=bool
        data array.

    Returns
    -------
    r_min : integer, {x ∈ ℕ | x ≥ 0}
        minimum row with a true boolean.
    r_max : integer, {x ∈ ℕ | x ≥ 0}
        maximum row with a true boolean.
    c_min : integer, {x ∈ ℕ | x ≥ 0}
        minimum collumn with a true boolean.
    c_max : integer, {x ∈ ℕ | x ≥ 0}
        maximum collumn with a true boolean.
    """
    assert type(img)==np.ndarray, ("please provide an array")

    rows,cols = np.any(img, axis=1), np.any(img, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max, c_min, c_max

def get_bbox(geoTransform, rows=None, cols=None):
    """ given array meta data, calculate the bounding box

    Parameters
    ----------
    geoTransform : tuple, size=(1,6)
        georeference transform of an image.
    rows : integer, {x ∈ ℕ | x ≥ 0}
        amount of rows in an image.
    cols : integer, {x ∈ ℕ | x ≥ 0}
        amount of collumns in an image.

    Returns
    -------
    bbox : np.ndarray, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y

    See Also
    --------
    map2pix, pix2map, pix_centers, get_map_extent

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    geoTransform = correct_geoTransform(geoTransform)
    if rows==None:
        assert len(geoTransform)>=8, ('please provide raster information')
        rows, cols = geoTransform[6], geoTransform[7]

    X = geoTransform[0] + \
        np.array([0, cols])*geoTransform[1] + np.array([0, rows])*geoTransform[2]

    Y = geoTransform[3] + \
        np.array([0, cols])*geoTransform[4] + np.array([0, rows])*geoTransform[5]

    bbox = np.hstack((np.sort(X), np.sort(Y)))
#    # get extent not pixel centers
#    spacing_x = np.sqrt(geoTransform[1]**2 + geoTransform[2]**2)/2
#    spacing_y = np.sqrt(geoTransform[4]**2 + geoTransform[5]**2)/2
#    bbox += np.array([-spacing_x, +spacing_x, -spacing_y, +spacing_y])
    return bbox

def get_shape_extent(bbox, geoTransform):
    """ given geographic meta data of array, calculate its shape

    Parameters
    ----------
    bbox : np.array, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y
    geoTransform : tuple, size={(1,6), (1,8)}
        georeference transform of an image.

    Returns
    -------
    rows : integer, {x ∈ ℕ | x ≥ 0}
        amount of rows in an image.
    cols : integer, {x ∈ ℕ | x ≥ 0}
        amount of collumns in an image.

    See Also
    --------
    map2pix, pix2map, pix_centers, get_map_extent, get_bbox

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    OBS: rasterio uses a different formulation for the bounding box, stated as
    (min_X, min_Y, max_X, max_Y). Transformation can be done through:
    bbox_swap = bbox.reshape((2,2)).T.ravel()
    """
    geoTransform = correct_geoTransform(geoTransform)
    rows = np.divide(bbox[3]-bbox[2],
                      np.hypot(geoTransform[4], geoTransform[5])
                     ).astype(int)
    cols = np.divide(bbox[1]-bbox[0],
                      np.hypot(geoTransform[1], geoTransform[2])
                     ).astype(int)
    return (rows, cols)

def get_max_pixel_spacing(geoTransform):
    """ calculate the maximum spacing between pixels

    Parameters
    ----------
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.

    Returns
    -------
    spac : float, unit=meter
        maximum spacing between pixels
    """
    geoTransform = correct_geoTransform(geoTransform)
    spac = np.maximum(np.hypot(geoTransform[1], geoTransform[2]),
                      np.hypot(geoTransform[4], geoTransform[5]))
    return spac

def get_pixel_spacing(geoTransform, spatialRef):
    """ provide the pixel spacing of an image in meters

    """
    geoTransform = correct_geoTransform(geoTransform)
    if is_crs_an_srs(spatialRef):
        d_1 = np.hypot(geoTransform[1], geoTransform[2])
        d_2 = np.hypot(geoTransform[4], geoTransform[5])
        return d_1, d_2
    else:
        d_1 = haversine(geoTransform[2], geoTransform[1], geoTransform[3])
        d_2 = haversine(geoTransform[5], geoTransform[4], geoTransform[3])
        return d_1, d_2

def get_map_extent(bbox):
    """ generate coordinate list in counterclockwise direction from boundingbox

    Parameters
    ----------
    bbox : np.array, size=(1,4)
        bounding box, in the following order: min max X, min max Y

    Returns
    -------
    xB,yB : numpy.ndarray, size=(5,1)
        coordinate list of horizontal and vertical outer points

    See Also
    --------
    get_bbox, get_bbox_polygon

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |
    """
    xB = np.array([[ bbox[0], bbox[0], bbox[1], bbox[1], bbox[0] ]]).T
    yB = np.array([[ bbox[3], bbox[2], bbox[2], bbox[3], bbox[3] ]]).T
    return xB, yB

def get_mean_map_location(geoTransform, spatialRef, rows=None, cols=None):
    """ estimate the central location of a geo-referenced image

    Parameters
    ----------
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    spatialRef : osgeo.osr.SpatialReference
        target projection system
    rows, cols : integer, {x ∈ ℕ | x ≥ 0}, default=None
        amount of rows/collumns in an image.

    Returns
    -------
    x_bar, y_bar : float
        mean location of the image in the reference system given by spatialRef
    """
    geoTransform = correct_geoTransform(geoTransform)
    bbox = get_bbox(geoTransform, rows=rows, cols=cols)
    x_bbox,y_bbox = get_map_extent(bbox)
    x_bar, y_bar = np.mean(x_bbox), np.mean(y_bbox)
    return x_bar, y_bar

def get_mean_map_lat_lon(geoTransform, spatialRef, rows=None, cols=None):
    """ estimate the central latitude and longitude of a geo-referenced image

    Parameters
    ----------
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    spatialRef : osgeo.osr.SpatialReference
        target projection system
    rows, cols : integer, {x ∈ ℕ | x ≥ 0}, default=None
        amount of rows/collumns in an image.

    Returns
    -------
    lat_bar, lon_bar : float, unit=degrees, , range=-90...+90, -180...+180
        latitude and longitude of the scene center
    """
    geoTransform = correct_geoTransform(geoTransform)
    bbox = get_bbox(geoTransform, rows=rows, cols=cols)

    if is_crs_an_srs(spatialRef):
        x_bbox, y_bbox = get_map_extent(bbox)
        ll = map2ll(np.concatenate((x_bbox, y_bbox), axis=1), spatialRef)
        lat_bar,lon_bar = np.mean(ll[:,0]), np.mean(ll[:,1])
        return lat_bar,lon_bar
    else:
        return np.mean(bbox[-2:]), np.mean(bbox[:2])

def get_bbox_polygon(geoTransform, rows, cols):
    """ given array meta data, create bounding polygon

    Parameters
    ----------
    geoTransform : tuple, size=(6,1)
        georeference transform of an image.
    rows : integer, {x ∈ ℕ | x ≥ 0}
        amount of rows in an image.
    cols : integer, {x ∈ ℕ | x ≥ 0}
        amount of collumns in an image.

    Returns
    -------
    poly : OGR polygon
        bounding polygon of image.

    See Also
    --------
    get_bbox, get_map_extent
    """
    # build tile polygon
    geoTransform = correct_geoTransform(geoTransform)
    bbox = get_bbox(geoTransform, rows, cols)
    xB,yB = get_map_extent(bbox)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in range(5):
        ring.AddPoint(float(xB[i]),float(yB[i]))
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    # poly_tile.ExportToWkt()
    return poly

def find_overlapping_DEM_tiles(dem_path,dem_file, poly_tile):
    '''
    loop through a shapefile of tiles, to find overlap with a given geometry
    '''
    # Get the DEM tile layer
    demShp = ogr.Open(os.path.join(dem_path, dem_file))
    demLayer = demShp.GetLayer()

    url_list = ()
    # loop through the tiles and see if there is an intersection
    for i in range(0, demLayer.GetFeatureCount()):
        # Get the input Feature
        demFeature = demLayer.GetFeature(i)
        geom = demFeature.GetGeometryRef()

        intersection = poly_tile.Intersection(geom)
        if(intersection is not None and intersection.Area()>0):
            url_list += (demFeature.GetField('fileurl'),)

    return url_list

def make_same_size(Old, geoTransform_old, geoTransform_new,
                   rows_new=None, cols_new=None):
    """ clip array to the same size as another array

    Parameters
    ----------
    Old : np.array, size=(m,n), dtype={float,complex}
        data array to be clipped.
    geoTransform_old : tuple, size={(6,), (8,)}
        georeference transform of the old image.
    geoTransform_new : tuple, size={(6,), (8,)}
        georeference transform of the new image.
    rows_new : integer, {x ∈ ℕ | x ≥ 0}
        amount of rows of the new image.
    cols_new : integer, {x ∈ ℕ | x ≥ 0}
        amount of collumns of the new image.

    Returns
    -------
    New : np.array, size=(k,l), dtype={float,complex}
        clipped data array.
    """
    geoTransform_old = correct_geoTransform(geoTransform_old)
    geoTransform_new = correct_geoTransform(geoTransform_new)

    if len(geoTransform_new)==8:
        rows_new, cols_new = geoTransform_new[-2], geoTransform_new[-1]

    # look at upper left coordinate
    dj = np.round((geoTransform_new[0]-geoTransform_old[0])/geoTransform_new[1])
    di = np.round((geoTransform_new[3]-geoTransform_old[3])/geoTransform_new[1])

    if np.sign(dj)==-1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(np.expand_dims(Old[:,0], axis = 1),
                                        abs(dj), axis=1), Old), axis = 1)
    elif np.sign(dj)==1: # reduce array
        Old = Old[:,abs(dj).astype(int):]

    if np.sign(di)==-1: # reduce array
        Old = Old[abs(di).astype(int):,:]
    elif np.sign(di)==1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(np.expand_dims(Old[0,:], axis = 1).T,
                                        abs(di), axis=0), Old), axis = 0)

    # as they are now alligned, look at the lower right corner
    di,dj = rows_new - Old.shape[0], cols_new - Old.shape[1]

    if np.sign(dj)==-1: # reduce array
        Old = Old[:,:dj]
    elif np.sign(dj)==1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(Old, np.expand_dims(Old[:,-1], axis=1),
                                        abs(dj), axis=1)), axis = 1)

    if np.sign(di)==-1: # reduce array
        Old = Old[:di,:]
    elif np.sign(di)==1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(Old, np.expand_dims(Old[-1,:], axis=1).T,
                                        abs(di), axis=0)), axis = 0)

    New = Old
    return New

def create_offset_grid(I, dx, dy, geoTransform):
    geoTransform = correct_geoTransform(geoTransform)
    dx, dy = correct_floating_parameter(dx), correct_floating_parameter(dy)
    di, dj = vel2pix(geoTransform, dx, dy)
    if len(geoTransform)==8: # sometimes the image dimensions are also included
        mI, nI = geoTransform[-2], geoTransform[-1]
    else:
        mI, nI = I.shape[0:2]
    dI_grd, dJ_grd = np.meshgrid(np.linspace(0, mI-1, mI)+di,
                                 np.linspace(0, nI-1, nI)+dj, indexing='ij')
    return dI_grd, dJ_grd
