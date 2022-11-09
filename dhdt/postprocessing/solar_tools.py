import numpy as np

from pysolar.solar import get_azimuth, get_altitude
from pysolar.radiation import get_radiation_direct
from datetime import datetime, timedelta
from pytz import timezone

from scipy import ndimage

from ..generic.unit_conversion import doy2dmy, deg2arg
from ..generic.data_tools import estimate_sinus
from ..generic.mapping_tools import get_mean_map_lat_lon

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
    if indexing=='ij':
        sun = np.array([[ -np.cos(np.radians(az)) ],
                        [ +np.sin(np.radians(az)) ]])
    else: # 'xy' that is a map coordinate system
        sun = np.array([[ +np.sin(np.radians(az)) ],
                        [ +np.cos(np.radians(az)) ]])
    return sun

def sun_angles_to_vector(az, zn, indexing='ij'):
    """ transform azimuth and zenith angle to 3D-unit vector

    Parameters
    ----------
    az : float, unit=degrees
        azimuth angle of sun.
    zn : float, unit=degrees
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
    az_to_sun_vector

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
    if indexing=='ij': # local image system
        sun = np.dstack((-np.cos(np.radians(az)), \
                         +np.sin(np.radians(az)), \
                         +np.tan(np.radians(zn)))
                        )
    else: # 'xy' that is map coordinates
        sun = np.dstack((+np.sin(np.radians(az)), \
                         +np.cos(np.radians(az)), \
                         +np.tan(np.radians(zn)))
                        )

    n = np.linalg.norm(sun, axis=2)
    sun[:, :, 0] /= n
    sun[:, :, 1] /= n
    sun[:, :, 2] /= n
    return sun

# elevation model based functions
def make_shadowing(Z, az, zn, spac=10, weights=None, radiation=False,
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
    az = deg2arg(az) # give range -180...+180
    Zr = ndimage.rotate(Z, az, axes=(1, 0), cval=-1, order=3)
    # mask based
    Mr = ndimage.rotate(np.zeros(Z.shape, dtype=bool), az, axes=(1, 0), \
                        cval=False, order=0, prefilter=False)

    if not isinstance(zn, np.ndarray):
        zn, weights = np.array([zn]), np.array([1])
    if weights is None:
        weights = np.ones_like(zn)

    for idx, zenit in enumerate(zn):
        dZ = np.tan(np.radians(90-zenit))*spac
        for i in range(1,Zr.shape[0]):
            Mr[i,:] = (Zr[i,:])<(Zr[i-1,:]-dZ)
            Zr[i,:] = np.maximum(Zr[i,:], Zr[i-1,:]-dZ)
        if radiation:
            Mr = np.invert(Mr)

        if idx>0:
            Mrs += weights[idx]*Mr.astype(int)
        else:
            Mrs = weights[idx]*Mr.astype(int).copy()

    Ms = ndimage.rotate(Mrs, -az, axes=(1, 0),
                        cval=0, order=0,
                        mode='constant', prefilter=False)
    i_min = int(np.floor((Ms.shape[0] - Z.shape[0]) / 2))
    i_max = int(np.floor((Ms.shape[0] + Z.shape[0]) / 2))
    j_min = int(np.floor((Ms.shape[1] - Z.shape[1]) / 2))
    j_max = int(np.floor((Ms.shape[1] + Z.shape[1]) / 2))

    Sw = Ms[i_min:i_max, j_min:j_max]
    # remove edge cases by copying
    if border in ('copy'):
        Sw[:,-1], Sw[-1,:] = Sw[:,-2], Sw[-2,:]
        Sw[:,+0], Sw[+0,:] = Sw[:,+1], Sw[+1,:]
    else:
        Sw[:,0], Sw[:,-1], Sw[0,:], Sw[-1,:] = 0, 0, 0, 0
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
    sun = sun_angles_to_vector(az, zn, indexing='xy')

    # estimate surface normals

    # the first array stands for the gradient in rows and
    # the second one in columns direction
    dy, dx = np.gradient(Z*spac)

    normal = np.dstack((dx, dy, np.ones_like(Z)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    Sh = normal[:,:,0]*sun[:,:,0] + \
        normal[:,:,1]*sun[:,:,1] + \
        normal[:,:,2]*sun[:,:,2]
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

    K_r = np.fliplr(np.meshgrid(np.linspace(0,M_r.shape[0]-1,M_r.shape[0]),
                                np.linspace(0,M_r.shape[1]-1,M_r.shape[1]))[0])
    np.putmask(K_r, ~M_r, 0)

    D_r = np.multiply(np.cos(np.deg2rad(zn)), Z_r) + \
          np.multiply(np.sin(np.deg2rad(zn)), K_r*spac)

    if Lambertian: # do a weighted histogram
        Sd = make_shading(Z, az, zn, spac=10)
        Sd_r = ndimage.rotate(Sd, az, axes=(1, 0), cval=-1, order=3)
        np.putmask(Sd_r, ~M_r, 0)

    # loop through the rows and create histogram
    S_r = np.zeros_like(Z_r, dtype=float)
    for i in range(Z_r.shape[0]):
        if Lambertian:
            his,_ = np.histogram(D_r[i,:],
                                 bins=np.arange(0, K_r.shape[1]+1),
                                 weights=Sd_r[i,:])
        else:
            his,_ = np.histogram(D_r[i,:],
                                 bins=np.arange(0, K_r.shape[1]+1),
                                 weights=M_r[i,:].astype(float))
        S_r[i,:] = his
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
    sun = sun_angles_to_vector(az, zn, indexing='xy')

    # estimate surface normals
    dy, dx = np.gradient(Z*spac)

    normal = np.dstack((dx, dy, np.ones_like(Z)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    L = normal[:,:,0]*sun[:,:,0] + \
        normal[:,:,1]*sun[:,:,1] + \
        normal[:,:,2]*sun[:,:,2]
    # assume overhead
    Sh = L**(k+1) * (1-normal[:,:,2])**(1-k)
    return Sh

def annual_solar_graph(latitude=51.707524, longitude=6.244362, deg_sep=.5,
                       year = 2018, sec_resol=20):
    """ calculate the solar graph of a location

    Parameters
    ----------
    latitude : float, unit=degrees, range=-90...+90
        latitude of the location of interest
    longitude : float, unit=degrees, range=-180...+180
        longitude of the location of interest
    deg_sep : float, unit=degrees
        resolution of the solargraph grid
    year : integer
        year of interest
    sec_resol : float, unit=seconds, default=20
        resolution of sun location calculation

    Returns
    -------
    Sky : np.array, size=(k,l), dtype=integer
        array with solargraph
    az : np.array, size=(l,_), dtype=float, unit=degrees
        azimuth values, that is the axis ticks of the solar graph
    zenit : np.array, size=(k,_), dtype=float, unit=degrees
        zenit values, that is the axis ticks of the solar graph

    See Also
    --------
    annual_solar_population : calculate the occurance
    """
    az = np.arange(0, 360, deg_sep)
    zn = np.flip(np.arange(-.5, +90, deg_sep))

    Sol = np.zeros((zn.shape[0], az.shape[0]))

    month = np.array([12, 6])     # 21/12 typical winter solstice - lower bound
    day = np.array([21, 21])      # 21/06 typical summer solstice - upper bound

    # loop through all times to get sun paths
    for i in range(0,2):
        for hour in range(0, 24):
            for minu in range(0, 60):
                for sec in range(0, 60, sec_resol):
                    sun_zen = get_altitude(latitude, longitude,
                                          datetime(year, month[i], day[i],
                                                   hour, minu, sec,
                                                   tzinfo=timezone('UTC')))
                    sun_azi = get_azimuth(latitude, longitude,
                                          datetime(year, month[i], day[i],
                                                   hour, minu, sec,
                                                   tzinfo=timezone('UTC')))

                az_id = (np.abs(az - sun_azi)).argmin()
                zn_id = (np.abs(zn - sun_zen)).argmin()
                if i==0:
                    Sol[zn_id,az_id] = -1
                else:
                    Sol[zn_id,az_id] = +1
    # remove the line below the horizon
    Sol = Sol[:-1,:]

    # mathematical morphology to do infilling, and extent the boundaries a bit
    Sol_plu, Sol_min = Sol==+1, Sol==-1
    Sol_plu = ndimage.binary_dilation(Sol_plu, np.ones((5,5))).cumsum(axis=0)==1
    Sol_min = np.flipud(ndimage.binary_dilation(Sol_min, np.ones((5,5))))
    Sol_min = np.flipud(Sol_min.cumsum(axis=0)==1)

    # populated the solargraph between the upper and lower bound
    Sky = np.zeros(Sol.shape)
    for i in range(0,Sol.shape[1]):
        mat_idx = np.where(Sol_plu[:,i]==+1)
        if len(mat_idx[0]) > 0:
            start_idx = mat_idx[0][0]
            mat_idx = np.where(Sol_min[:,i]==1)
            if len(mat_idx[0]) > 0:
                end_idx = mat_idx[0][-1]
            else:
                end_idx = Sol.shape[1]
            Sky[start_idx:end_idx,i] = 1

    return Sky, az, zn

def annual_solar_population(latitude=51.707524, longitude=6.244362, deg_sep=.5,
                            year = 2018, poly=True, radiation=True):
    """ calculate the solar graph distribution of a location

    Parameters
    ----------
    latitude : float, unit=degrees, range=-90...+90
        latitude of the location of interest
    longitude : float, unit=degrees, range=-180...+180
        longitude of the location of interest
    deg_sep : {float,list}, unit=degrees
        resolution of the solargraph grid, if a list is given
        the first element corresponds to the azimuth resolution,
        the second element corresponds to the zenith resolution
    year : integer
        year of interest
    poly : boolean, default=True
        estimate a polynomial through the sun's daily orbit
    radiation : boolean, default=True
        estimate the radiation

    Returns
    -------
    Sol : np.array, size=(k,l), dtype=integer
        array with solargraph occurance
    az : np.array, size=(l,_), dtype=float, unit=degrees
        azimuth values, that is the axis ticks of the solar graph
    zn : np.array, size=(k,_), dtype=float, unit=degrees
        zenit values, that is the axis ticks of the solar graph

    See Also
    --------
    annual_solar_graph : calculate the spread
    """
    if not isinstance(deg_sep, list):
        deg_sep = [deg_sep, deg_sep]
    if len(deg_sep)==1:
        deg_sep = [deg_sep[0], deg_sep[0]]

    az = np.arange(0, 360, deg_sep[0])
    zn = np.arange(-.5, +90, deg_sep[1])

    Sol = np.zeros((zn.shape[0], az.shape[0]))

    # loop through all days of the year
    for doy in np.arange(0, 365):
        d_1,m_1,y_1 = doy2dmy(np.array([doy]), np.array([year]))
        d_2,m_2,y_2 = doy2dmy(np.array([doy+1]), np.array([year]))

        time_steps = np.arange(datetime(y_1,m_1,d_1),
                               datetime(y_2,m_2,d_2),
                               timedelta(hours=1)).astype(datetime)

        sun_zen = np.zeros_like(time_steps, dtype=float)
        sun_azi = np.zeros_like(time_steps, dtype=float)
        for i,t in enumerate(time_steps):
            t = t.replace(tzinfo=timezone('UTC'))
            sun_zen[i] = get_altitude(latitude, longitude, t)
            sun_azi[i] = get_azimuth(latitude, longitude, t)

        # estimate curve, does not seem to be a sinus...
        if poly is True:
            p = np.poly1d(np.polyfit(sun_azi, sun_zen, 6))
            zn_hat = p(az)
        else:
            # estimate sinus
            az_hat,amp_hat,bias = estimate_sinus(sun_azi, sun_zen)
            zn_hat = (amp_hat*np.cos(np.deg2rad(az-az_hat))) + bias

        zn_id = np.abs(np.tile(zn_hat[np.newaxis,:], (zn.size, 1)) -
                       np.tile(zn[:,np.newaxis], (1, zn_hat.size)) ).argmin(axis=0)

        if radiation == True:
            delta_sec = (deg_sep[0]/360)*(24*60*60) # amount of seconds
            for counter, idx in enumerate(zn_id):
                tot_rad = delta_sec*get_radiation_direct(t, zn[idx])
                Sol[zn_id[idx],counter] += tot_rad
        else:
            Sol[zn_id,np.arange(0,az.size)] += 1

    # remove the line below the horizon
    Sol, zn = np.flipud(Sol[1:,:]), np.flip(zn[1:])
    return Sol, az, zn

def annual_illumination(Z, geoTransform, spatialRef, deg_sep=[5,1], year=2018):
    """ estimate the annual illumination of for a section of terrain

    Parameters
    ----------
    Z : numpy.array, size=(m,n), dtype={integer,float}, unit=meter
        grid with elevation data
    geoTransform : tuple, size={(1,6), (1,8)}
        georeference transform of the array 'Z'
    spatialRef : osgeo.osr.SpatialReference() object
        coordinate reference system
    deg_sep : {float,list}, unit=degrees
        resolution of the solargraph grid
    year : integer
        year of interest

    Returns
    -------
    SW : numpy.array, size=(m,n), dtype=integer
        annual illumination on the terrain

    See Also
    --------
    annual_solar_population, make_shadowing

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from dhdt.generic.mapping_io import read_geo_image
    >>> from dhdt.postprocessing.solar_tools import annual_illumination

    >>> Z,spatialRef,geoTransform,_ = read_geo_image(fpath)
    >>> SW = annual_illumination(Z,spatialRef,geoTransform)

    >>> plt.figure()
    >>> plt.imshow(SW, cmap=plt.cm.magma)
    >>> plt.show()
    """
    if not isinstance(deg_sep, list):
        deg_sep = [deg_sep, deg_sep]
    if len(deg_sep)==1:
        deg_sep = [deg_sep[0], deg_sep[0]]

    lat,lon = get_mean_map_lat_lon(geoTransform, spatialRef)

    Sol,az,el = annual_solar_population(latitude=lat,longitude=lon,
                                        deg_sep=deg_sep, year=year, poly=True,
                                        radiation=True)
    SW = np.zeros_like(Z)
    for idx, azimuth in enumerate(az):
        occurance = Sol[:, idx]
        if np.sum(occurance)>0:
            IN = occurance!=0
            zenits, occurance = 90-el[IN], occurance[IN]

            SW += make_shadowing(Z, azimuth, zenits, weights=occurance,
                                 radiation=True)
    return SW

# topocalc has horizon calculations
# based upon Dozier & Frew 1990
# implemented by Maxime: https://github.com/maximlamare/REDRESS


