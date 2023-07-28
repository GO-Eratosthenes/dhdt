import numpy as np

from scipy import ndimage

from dhdt.generic.unit_conversion import deg2arg
from dhdt.generic.mapping_tools import get_max_pixel_spacing


# general location functions
def az_to_sun_vector(az, indexing='ij'):
    """ transform azimuth angle to 2D-unit vector

    Parameters
    ----------
    az : float, unit=degrees
        azimuth of sun.
    indexing : {‘xy’, ‘ij’}
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates

    Returns
    -------
    sun : numpy.array, size=(2,1), range=0...1
        unit vector in the direction of the sun.

    See Also
    --------
    sun_angles_to_vector

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--┼--> +
                 |
                 ┼----> East & x

    The angles related to the suns' heading are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └--┴---

    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------┼-------->      --------┼-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    if indexing == 'ij':
        sun = np.array([[-np.cos(np.radians(az))], [+np.sin(np.radians(az))]])
    else:  # 'xy' that is a map coordinate system
        sun = np.array([[+np.sin(np.radians(az))], [+np.cos(np.radians(az))]])
    return sun


def sun_angles_to_vector(az, zn, indexing='ij'):
    """ transform azimuth and zenith angle to 3D-unit vector

    Parameters
    ----------
    az : {float, numpy.ndarray}, unit=degrees
        azimuth angle of sun.
    zn : {float, numpy.ndarray}, unit=degrees
        zenith angle of sun.
    indexing : {‘xy’, ‘ij’}
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates

    Returns
    -------
    sun : numpy.array, size=(3,1), dtype=float, range=0...1
        unit vector in the direction of the sun.

    See Also
    --------
    dhdt.postprocessing.solar_tools.az_to_sun_vector
    dhdt.postprocessing.solar_tools.vector_to_sun_angles

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--┼--> +
                 |
                 ┼----> East & x

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └---- surface -----   └--┴---

    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------┼-------->      --------┼-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    az, zn = np.deg2rad(az), np.deg2rad(zn)
    if indexing == 'ij':  # local image system
        sun = np.dstack(
            (-np.cos(az) * np.sin(zn), +np.sin(az) * np.sin(zn), +np.cos(zn)))
    else:  # 'xy' that is map coordinates
        sun = np.dstack(
            (+np.sin(az) * np.sin(zn), +np.cos(az) * np.sin(zn), +np.cos(zn)))

    if type(az) in (np.ndarray, ):
        return sun
    else:  # if single entry is given
        return np.squeeze(sun)


def vector_to_sun_angles(sun, indexing='ij'):
    """ transform 3D-unit vector to azimuth and zenith angle

    Parameters
    ----------
    sun : {float,numpy.array}, size=(m,), dtype=float, range=0...1
        the three different unit vectors in the direction of the sun.

    Returns
    -------
    az : {float, numpy.ndarray}, unit=degrees
        azimuth angle of sun.
    zn : {float, numpy.ndarray}, unit=degrees
        zenith angle of sun.

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

    if indexing == 'ij':  # local image system
        az = np.rad2deg(np.arctan2(sun[..., 1], -sun[..., 0]))
    else:  # 'xy' that is map coordinates
        az = np.rad2deg(np.arctan2(sun[..., 0], sun[..., 1]))
    zn = np.rad2deg(np.arccos(sun[..., 2]))
    return az, zn


# elevation model based functions
def make_shadowing(Z,
                   az,
                   zn,
                   spac=10,
                   weights=None,
                   radiation=False,
                   border='copy'):
    """ create synthetic shadow image from given sun angles

    Parameters
    ----------
    Z : numpy.array, size=(m,n), dtype={integer,float}
        grid with elevation data
    az : float, unit=degrees
        azimuth angle
    zn : {float, numpy.array}, unit=degrees
        zenith angle
    spac : float, optional
        resolution of the square grid. The default is 10.
    weights : numpy.array
        if the zenith angle is a vector, it can be used to get a weighted sum
    border : string, {'copy','zeros'}
        specify what to do at the border, since it is a zonal operation

    Returns
    -------
    Sw : numpy.array, size=(m,n), dtype=bool
        estimated shadow grid

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--┼--> +
                 |
                 ┼----> East & x

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └--┴---
    """
    az = deg2arg(az)  # give range -180...+180
    if isinstance(spac, tuple): spac = get_max_pixel_spacing(spac)

    Zr = ndimage.rotate(Z, az, axes=(1, 0), cval=-1, order=3)
    # mask based
    Mr = ndimage.rotate(np.zeros(Z.shape, dtype=bool), az, axes=(1, 0), \
                        cval=False, order=0, prefilter=False)

    if not isinstance(zn, np.ndarray):
        zn, weights = np.array([zn]), np.array([1])
    if weights is None:
        weights = np.ones_like(zn)

    for idx, zenit in enumerate(zn):
        dZ = np.tan(np.radians(90 - zenit)) * spac
        for i in range(1, Zr.shape[0]):
            Mr[i, :] = (Zr[i, :]) < (Zr[i - 1, :] - dZ)
            Zr[i, :] = np.maximum(Zr[i, :], Zr[i - 1, :] - dZ)
        if radiation:
            Mr = np.invert(Mr)

        if idx > 0:
            Mrs += weights[idx] * Mr.astype(int)
        else:
            Mrs = weights[idx] * Mr.astype(int).copy()

    Ms = ndimage.rotate(Mrs,
                        -az,
                        axes=(1, 0),
                        cval=0,
                        order=0,
                        mode='constant',
                        prefilter=False)
    i_min = int(np.floor((Ms.shape[0] - Z.shape[0]) / 2))
    i_max = int(np.floor((Ms.shape[0] + Z.shape[0]) / 2))
    j_min = int(np.floor((Ms.shape[1] - Z.shape[1]) / 2))
    j_max = int(np.floor((Ms.shape[1] + Z.shape[1]) / 2))

    Sw = Ms[i_min:i_max, j_min:j_max]
    # remove edge cases by copying
    if border in ('copy'):
        Sw[:, -1], Sw[-1, :] = Sw[:, -2], Sw[-2, :]
        Sw[:, +0], Sw[+0, :] = Sw[:, +1], Sw[+1, :]
    else:
        Sw[:, 0], Sw[:, -1], Sw[0, :], Sw[-1, :] = 0, 0, 0, 0
    return Sw


def make_shading(Z, az, zn, spac=10):
    """ create synthetic shading image from given sun angles

    A simple Lambertian reflection model is used here.

    Parameters
    ----------
    Z : numpy.array, size=(m,n), dtype={integer,float}, unit=meter
        grid with elevation data
    az : float, unit=degrees
        azimuth angle
    zn : float, unit=degrees
        zenith angle
    spac : {float, tuple}, default=10, unit=meter
        resolution of the square grid. Optionally the geoTransform is given.

    Returns
    -------
    Sh : numpy.array, size=(m,n), dtype=float, range=0...1
        estimated shading grid

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--┼--> +
                 |
                 ┼----> East & x

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └--┴---
    """
    if isinstance(spac, tuple):
        spac = get_max_pixel_spacing(spac)
    sun = sun_angles_to_vector(az, zn, indexing='xy')

    # estimate surface normals

    # the first array stands for the gradient in rows and
    # the second one in columns direction
    dy, dx = np.gradient(Z * spac)

    normal = np.dstack((dx, dy, np.ones_like(Z)))
    n = np.linalg.norm(normal, axis=2)
    normal[..., 0] /= n
    normal[..., 1] /= n
    normal[..., 2] /= n

    Sh = normal[...,0]*sun[...,0] + \
         normal[...,1]*sun[...,1] + \
         normal[...,2]*sun[...,2]
    return Sh


def make_doppler_range(Z, az, zn, Lambertian=True, spac=10):
    """

    Parameters
    ----------
    Z : numpy.array, unit=meters
        array with elevation values
    az : float, unit=degrees, range=-180...+180
        flight orientation of the satellite
    zn : {float,array},  unit=degrees, range=0...+90
        illumination angle from the satellite

    Returns
    -------

    Notes
    -----


    """

    # rotate
    Z_r = ndimage.rotate(Z, az, axes=(1, 0), cval=-1, order=3)
    # mask based
    M_r = ndimage.rotate(np.ones_like(Z, dtype=bool), az, axes=(1, 0), \
                        cval=False, order=0, prefilter=False)

    K_r = np.fliplr(
        np.meshgrid(np.linspace(0, M_r.shape[0] - 1, M_r.shape[0]),
                    np.linspace(0, M_r.shape[1] - 1, M_r.shape[1]))[0])
    np.putmask(K_r, ~M_r, 0)

    D_r = np.multiply(np.cos(np.deg2rad(zn)), Z_r) + \
          np.multiply(np.sin(np.deg2rad(zn)), K_r*spac)

    if Lambertian:  # do a weighted histogram
        Sd = make_shading(Z, az, zn, spac=10)
        Sd_r = ndimage.rotate(Sd, az, axes=(1, 0), cval=-1, order=3)
        np.putmask(Sd_r, ~M_r, 0)

    # loop through the rows and create histogram
    S_r = np.zeros_like(Z_r, dtype=float)
    for i in range(Z_r.shape[0]):
        if Lambertian:
            his = np.histogram(D_r[i, :],
                               bins=np.arange(0, K_r.shape[1] + 1),
                               weights=Sd_r[i, :])[0]
        else:
            his = np.histogram(D_r[i, :],
                               bins=np.arange(0, K_r.shape[1] + 1),
                               weights=M_r[i, :].astype(float))[0]
        S_r[i, :] = his
    return


def make_shading_minnaert(Z, az, zn, k=1, spac=10):
    """ create synthetic shading image from given sun angles

    A simple Minnaert reflection model is used here.

    Parameters
    ----------
    Z : numpy.array, size=(m,n), dtype={integer,float}, unit=meter
        grid with elevation data
    az : float, unit=degrees
        azimuth angle
    zn : float, unit=degrees
        zenith angle
    spac : float, default=10, unit=meter
        resolution of the square grid.

    Returns
    -------
    Sh : numpy.array, size=(m,n), dtype=float, range=0...1
        estimated shading grid

    Notes
    -----
    The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--┼--> +
                 |
                 ┼----> East & x

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └--┴---
    """
    if isinstance(spac, tuple): spac = get_max_pixel_spacing(spac)
    sun = sun_angles_to_vector(az, zn, indexing='xy')

    # estimate surface normals
    dy, dx = np.gradient(Z * spac)

    normal = np.dstack((dx, dy, np.ones_like(Z)))
    n = np.linalg.norm(normal, axis=2)
    normal[..., 0] /= n
    normal[..., 1] /= n
    normal[..., 2] /= n

    L = normal[...,0]*sun[...,0] + \
        normal[...,1]*sun[...,1] + \
        normal[...,2]*sun[...,2]
    # assume overhead
    Sh = L**(k + 1) * (1 - normal[..., 2])**(1 - k)
    return Sh


# topocalc has horizon calculations
# based upon Dozier & Frew 1990
# implemented by Maxime: https://github.com/maximlamare/REDRESS
