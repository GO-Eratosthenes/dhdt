import os

import numpy as np

import pygrib
import cdsapi

# https://cds.climate.copernicus.eu/api-how-to

# local libraries
from ..generic.mapping_tools import map2ll
from ..generic.unit_conversion import datetime2calender, hpa2pascal

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels

def get_space_time_id(date, lat, lon):
    """

    Parameters
    ----------
    date : np.datetime64
        times of interest
    lat : float, unit=degrees, range=-90...+90
        central latitude of interest
    lon : float, unit=degrees, range=-180...+180
        central longitude of interest

    Returns
    -------
    fid : string
        format, XX{N,S}XXX-XX{E,W}XXX-YYYY-MM-DD
    """

    # construct part of string about the location
    fname = str(np.floor(np.abs(lat)).astype(int)).zfill(2)
    fname += 'S' if lat<0 else 'N'
    fname += f'{1 - lat%1:.3f}'[2:]

    fname += '-' + str(np.floor(np.abs(lon)).astype(int)).zfill(3)
    fname += 'W' if lon<0 else 'E'
    fname += f'{1 - lon % 1:.3f}'[2:]

    # construct part of string about the date
    year, month, day = datetime2calender(date)
    fname += '-' + str(year) + '-' + str(month).zfill(2) + \
             '-' + str(day).zfill(2)

    return fname

# "General Regularly distributed Information in Binary form"
def get_pressure_from_grib_file(fname='download.grib'):
    pres_levels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50,
                            70, 100, 125, 150, 175, 200, 225,
                            250, 300, 350, 400, 450, 500, 550,
                            600, 650, 700, 750, 775, 800, 825,
                            850, 875, 900, 925, 950, 975, 1000])

    G, Rh, T = None, None, None

    grbs = pygrib.open(fname)
    for i in np.linspace(1,grbs.messages,grbs.messages, dtype=int):
        grb = grbs.message(i)
        idx = np.where(pres_levels==grb['level'])[0][0]

        # initial admin
        if G is None:
            G = np.zeros((37, grb['numberOfDataPoints']))
            Rh, T = np.zeros_like(G), np.zeros_like(G)
            lat, lon = grb['latitudes'], grb['longitudes']

        if grb['parameterName']=='Geopotential':
            G[idx,:] = grb['values']
        elif grb['parameterName']=='Relative humidity':
            Rh[idx,:] = grb['values']
        elif grb['parameterName']=='Temperature':
            T[idx,:] = grb['values']
    grbs.close()
    return lat, lon, G, Rh, T

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
    lat, lon : np.array(), unit=degrees
        latitude and longitude of the ERA5 nodes
    z : np.array, unit=meter
        altitudes of interest
    Temp : np.array, unit=Kelvin
        temperature profile at altitudes given by "z"
    Pres : unit=Pascal
        pressure profile at altitudes given by "z"
    fracHum : unit=percentage,  range=0...1
        fractional humidity profile at altitudes given by "z"
    """
    pres_levels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50,
                            70, 100, 125, 150, 175, 200, 225,
                            250, 300, 350, 400, 450, 500, 550,
                            600, 650, 700, 750, 775, 800, 825,
                            850, 875, 900, 925, 950, 975, 1000])

    hour = 11
    year,month,day = datetime2calender(date)

    # convert from mapping coordinates to lat, lon
    ll = map2ll(np.stack((x, y)).T, spatialRef)

    lat_min, lon_min = np.min(ll[:,0]), np.min(ll[:,1])
    lat_max, lon_max = np.max(ll[:,0]), np.max(ll[:,1])

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
            'year': str(year),
            'month': str(month).zfill(2),
            'day': str(day).zfill(2),
            'time': [
                str(hour-1).zfill(2)+':00',
                str(hour+1).zfill(2)+':00',
            ],
            'area': [
                lat_max, lon_min, lat_min,
                lon_max,
            ],
        },
        fname)

    lat, lon, G, Rh, T = get_pressure_from_grib_file(fname=fname)
    os.remove(fname)

    # extract atmospheric profile, see also
    # https://confluence.ecmwf.int/pages/viewpage.action?pageId=111155328
    g_wmo = 9.80665 # gravity constant defined by WMO
    z_G = G/g_wmo

    posts = z_G.shape[1]
    Temp = np.zeros((len(z), posts))
    Pres, RelHum = np.zeros_like(Temp), np.zeros_like(Temp)
    for i in range(posts):
        Pres[:,i] = np.interp(z, np.flip(z_G[:, i]),
                            np.flip(pres_levels.astype(float)))
        Temp[:, i] = np.interp(z, np.flip(z_G[:, i]), np.flip(T[:, i]))
        RelHum[:, i] = np.interp(z, np.flip(z_G[:, i]), np.flip(Rh[:, i]))

    Pres = hpa2pascal(Pres)
    fracHum = RelHum/100
    return lat, lon, z, Temp, Pres, fracHum
