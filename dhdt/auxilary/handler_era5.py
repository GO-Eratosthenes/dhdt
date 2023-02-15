import os

import numpy as np

import pygrib
import cdsapi

# https://cds.climate.copernicus.eu/api-how-to

# local libraries
from dhdt.generic.mapping_tools import map2ll
from dhdt.generic.unit_conversion import \
    datetime2calender, datenum2datetime, hpa2pascal

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels

def get_space_time_id(date, ϕ, λ, full=True):
    """

    Parameters
    ----------
    date : np.datetime64
        times of interest
    ϕ : float, unit=degrees, range=-90...+90
        central latitude of interest
    λ : float, unit=degrees, range=-180...+180
        central longitude of interest
    full : boolean
        give the whole date, otherwise only the year will be given

    Returns
    -------
    fid : string
        format, XX{N,S}XXX-XX{E,W}XXX-YYYY-MM-DD
    """

    # construct part of string about the location
    fname = str(np.floor(np.abs(ϕ)).astype(int)).zfill(2)
    fname += 'S' if ϕ<0 else 'N'
    fname += f'{1 - ϕ%1:.3f}'[2:]

    fname += '-' + str(np.floor(np.abs(λ)).astype(int)).zfill(3)
    fname += 'W' if λ<0 else 'E'
    fname += f'{1 - λ % 1:.3f}'[2:]

    # construct part of string about the date
    year, month, day = datetime2calender(date)
    fname += '-' + str(year)
    if not full:
        return fname
    fname += '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
    return fname

def get_amount_of_timestamps(name_str):
    """ Reanalysis data from ERA5 can have different timestamps. These are given
    in the name, separtated by an indent. This function counts these indents,
    so the file does not need to be read.

    Parameters
    ----------
    name_str : string
        name of the grid file

    Returns
    -------
    nmbr : integer, {x ∈ ℕ | x ≥ 0}
        amount of datapoints

    Examples
    --------
    >>> import pygrib
    >>> fname = 'download.grib'
    >>> grbs = pygrib.open(fname)
    >>> get_amount_of_timestamps(grbs.name)
    2
    """
    nmbr = len(name_str.split('-')[2].split(' '))
    return nmbr

def get_era5_pressure_levels():
    pres_levels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50,
                            70, 100, 125, 150, 175, 200, 225,
                            250, 300, 350, 400, 450, 500, 550,
                            600, 650, 700, 750, 775, 800, 825,
                            850, 875, 900, 925, 950, 975, 1000])
    return pres_levels

# "General Regularly distributed Information in Binary form"
def get_pressure_from_grib_file(fname='download.grib'):
    """ extract data from .grib files

    Parameters
    ----------
    fname : string
        filename (and location) of binary ERA5 data

    Returns
    -------
    ϕ, λ: {float, numpy.ndarray}, size={1,k}, unit=degrees
        location of datapoints
    G, Rh, T : numpy.ndarray, size={m,k,l}
        data arrays at a specific place and time
    t : numpy.ndarray, size={l}
        timestamp of the data arrays
    """

    pres_levels = get_era5_pressure_levels()
    G, Rh, T, t = None, None, None, None

    grbs = pygrib.open(fname)
    for i in np.linspace(1,grbs.messages,grbs.messages, dtype=int):
        grb = grbs.message(i)
        idx = np.where(pres_levels==grb['level'])[0][0]
        toi = grb['dataDate'] # time of interest

        # initial admin
        if G is None:
            m,n = len(pres_levels), grb['numberOfDataPoints']
            G = np.zeros((m,n,1),
                         dtype=np.float64)
            Rh, T = np.zeros_like(G), np.zeros_like(G)
            ϕ, λ = grb['latitudes'], grb['longitudes']
            t = np.array([toi])

        t_IN = t==toi # is date already visited before
        if np.any(t_IN):
            idx_t = np.where(t_IN)[0][0]
        else: #update with new data entry for a specific time stamp
            t = np.append(t, toi)
            idx_t = t.size-1
            G = np.dstack((G, np.zeros((m, n, 1))))
            Rh = np.dstack((G, np.zeros((m, n, 1))))
            T = np.dstack((G, np.zeros((m, n, 1))))

        if grb['parameterName']=='Geopotential':
            G[idx,:,idx_t] = grb['values']
        elif grb['parameterName']=='Relative humidity':
            Rh[idx,:,idx_t] = grb['values']
        elif grb['parameterName']=='Temperature':
            T[idx,:,idx_t] = grb['values']
    grbs.close()

#    if t.size==1:
#        G, Rh, T = np.squeeze(G), np.squeeze(Rh), np.squeeze(T)
    return ϕ, λ, G, Rh, T, t

def get_wind_from_grib_file(fname='download.grib', pres_level=1000):
    pres_levels = get_era5_pressure_levels()
    U, V, Rh, T = None, None, None, None

    grbs = pygrib.open(fname)
    for i in np.linspace(1,grbs.messages,grbs.messages, dtype=int):
        grb = grbs.message(i)
        idx = np.where(pres_level==grb['level'])[0][0]

        # initial admin
        if T is None:
            T = np.zeros((len(pres_levels), grb['numberOfDataPoints']))
            Rh, U, V = np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)
            ϕ, λ = grb['latitudes'], grb['longitudes']

        if grb['parameterName']=='Geopotential':
            U[idx,:] = grb['values']
        elif grb['parameterName']=='Geopotential':
            V[idx,:] = grb['values']
        elif grb['parameterName']=='Relative humidity':
            Rh[idx,:] = grb['values']
        elif grb['parameterName']=='Temperature':
            T[idx,:] = grb['values']
    grbs.close()
    return ϕ, λ, U, V, Rh, T

def get_era5_atmos_profile(date, x, y, spatialRef,
                           z=10**np.linspace(0,5,100)):
    """

    Parameters
    ----------
    date : np.datetime64
        times of interest
    x, y : np.array, unit=meter
        map location of interest
    spatialRef : osgeo.osr.SpatialReference
        projection system describtion
    z : np.array, unit=meter
        altitudes of interest

    Returns
    -------
    ϕ, λ : np.array(), unit=degrees
        latitude and longitude of the ERA5 nodes
    z : np.array, unit=meter
        altitudes of interest
    Temp : np.array, unit=Kelvin
        temperature profile at altitudes given by "z"
    Pres : unit=Pascal
        pressure profile at altitudes given by "z"
    fracHum : unit=percentage,  range=0...1
        fractional humidity profile at altitudes given by "z"

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> from osgeo import osr
    >>> from dhdt.generic.mapping_tools import ll2map

    >>> proj = osr.SpatialReference()
    >>> proj.SetWellKnownGeogCS('WGS84')
    >>> lat, lon = 59.937633, 10.7207376 # Oslo University
    >>> NH = True if lat > 0 else False
    >>> proj.SetUTM(33, NH)

    >>> xy = ll2map(np.array([[lat, lon]]), proj)
    >>> past = np.datetime64('2018-05-16')
    >>> h = 10**np.linspace(0,5,100)
    >>> lat, lon, z, Temp, Pres, fracHum = get_era5_atmos_profile(
            past, xy[0][0], xy[0][1], proj,z=h)

    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    >>> fig.suptitle('Atmospheric profile')
    >>> ax1.plot(Temp,z,'r'), ax1.set_title('Temperature [K]'), ax1.set_yscale('log')
    >>> ax2.plot(Pres,z,'g'), ax2.set_title('Pressure [Pa]'), ax2.set_yscale('log')
    >>> ax3.plot(fracHum,z,'b'), ax3.set_title('Humidity [%]'), ax3.set_yscale('log')
    >>> plt.show()
    """
    if isinstance(x, float): x = np.array([x])
    if isinstance(y, float): y = np.array([y])
    pres_levels = get_era5_pressure_levels()

    hour = 11
    #year,month,day = datetime2calender(date)
    if isinstance(date, np.int64):
        date = np.array([date])

    # convert from mapping coordinates to lat, lon
    ll = map2ll(np.stack((x, y)).T, spatialRef)

    ϕ_min, λ_min = np.min(ll[:,0]), np.min(ll[:,1])
    ϕ_max, λ_max = np.max(ll[:,0]), np.max(ll[:,1])

    fname = get_space_time_id(date, np.mean(ll[:,0]), np.mean(ll[:,1]))
    fname += '.grib' # using grib-data format

    # retrieve data
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'geopotential', 'relative_humidity', 'temperature',
            ],
            'pressure_level': np.ndarray.tolist(pres_levels.astype(str)),
            'date': np.ndarray.tolist(np.datetime_as_string(date, unit='D')),
            'time': [
                str(hour-1).zfill(2)+':00',
                str(hour+1).zfill(2)+':00',
            ],
            'area': [
                ϕ_max, λ_min, ϕ_min, λ_max,
            ],
        },
        fname)

    ϕ, λ, G, Rh, T, t = get_pressure_from_grib_file(fname=fname)
    os.remove(fname)

    # extract atmospheric profile, see also
    # https://confluence.ecmwf.int/pages/viewpage.action?pageId=111155328
    g_wmo = 9.80665 # gravity constant defined by WMO
    z_G = G/g_wmo

    posts = 1 if z_G.ndim==1 else z_G.shape[1]
    stamps = t.size
    Temp = np.zeros((len(z), posts, stamps), dtype=np.float64)
    Pres, RelHum = np.zeros_like(Temp), np.zeros_like(Temp)
    for i in range(posts):
        for j in range(stamps):
            Pres[:,i,j] = np.interp(z, np.flip(z_G[:,i,j]),
                                np.flip(pres_levels.astype(float)))
            Temp[:,i,j] = np.interp(z, np.flip(z_G[:,i,j]), np.flip(T[:,i,j]))
            RelHum[:,i,j] = np.interp(z, np.flip(z_G[:,i,j]), np.flip(Rh[:,i,j]))

    Pres = hpa2pascal(Pres)
    fracHum = RelHum/100
    t = datenum2datetime(t)
    return ϕ, λ, z, Temp, Pres, fracHum, t

def get_era5_monthly_surface_wind(ϕ,λ,year):

    fname = get_space_time_id(np.array([year-1970]).astype('datetime64[Y]')[0],
                              ϕ, λ, full=False)
    fname += '.grib' # using grib-data format

    # era5 reanalysis grid is in .25[deg]
    deg_xtr = .5
    ϕ_min, ϕ_max = ϕ-deg_xtr, ϕ+deg_xtr
    λ_min, λ_max = λ-deg_xtr, λ+deg_xtr

    # retrieve data
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'format': 'grib',
            'variable': [
                'u_component_of_wind', 'v_component_of_wind',
                'relative_humidity', 'temperature',
            ],
            'pressure_level': '1000',
            'year': str(year),
            'month': ['01', '02', '03', '04', '05', '06',
                      '07', '08', '09', '10', '11', '12', ],
            'area': [
                ϕ_max, λ_min, ϕ_min, λ_max,
            ],
        },
        fname)
    ϕ, λ, U, V, Rh, T = get_wind_from_grib_file(fname=fname)
    os.remove(fname)

    return U,V, Rh, T